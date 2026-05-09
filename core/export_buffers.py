"""Write per-part geometry buffers for BoneMerge exports."""

from __future__ import annotations

import math
import os
import struct
import time
from dataclasses import dataclass

from .texcoord_attrs import (
    color_component_to_raw_byte,
    texcoord_color_attr_names,
    texcoord_component_attr_names,
)
from .export_package import ExportPartPlan, write_r32_index_buffer
from .numpy_compat import optional_numpy
from .uv_transform import DEFAULT_UV_FLIP_V, blender_uv_to_game
from .vertex_format import encode_game_packed_normal, pack_vertex_format


@dataclass(frozen=True)
class _LoopVertex:
    mesh_obj: object
    mesh: object
    vertex_index: int
    loop_index: int
    polygon: object


@dataclass
class _EvaluatedMeshRecord:
    mesh: object | None
    matrix_world_applied: bool
    owned_mesh: object | None


class _EvaluatedMeshStore:
    def __init__(self) -> None:
        self.records: dict[int, _EvaluatedMeshRecord] = {}
        self.evaluated_count = 0
        self.fallback_count = 0

    def record(self, mesh_obj) -> _EvaluatedMeshRecord:
        key = id(mesh_obj)
        cached = self.records.get(key)
        if cached is not None:
            return cached
        mesh, matrix_world_applied, owned_mesh = _evaluated_export_mesh(mesh_obj)
        cached = _EvaluatedMeshRecord(
            mesh=mesh,
            matrix_world_applied=matrix_world_applied,
            owned_mesh=owned_mesh,
        )
        if owned_mesh is not None:
            self.evaluated_count += 1
        else:
            self.fallback_count += 1
        self.records[key] = cached
        return cached

    def release(self) -> None:
        for record in self.records.values():
            if record.owned_mesh is not None:
                _release_owned_mesh(record.owned_mesh)
                record.owned_mesh = None


@dataclass
class _MeshExportCache:
    mesh_obj: object
    mesh: object
    group_index_to_global: dict[int, int]
    mirror_flip: bool
    uv_flip_v: bool
    matrix_world_applied: bool
    vertex_position_values: list[tuple[float, float, float]] | None
    loop_normal_values: list[tuple[float, float, float]] | None
    game_position_values: list[tuple[float, float, float]] | None
    game_normal_values: list[tuple[float, float, float]] | None
    game_packed_normal_values: list[int] | None
    game_position_by_vertex: dict[int, tuple[float, float, float]]
    game_normal_by_loop: dict[int, tuple[float, float, float]]
    top4_by_vertex: dict[int, tuple[tuple[float, float, float, float], tuple[int, int, int, int]]]
    top4_packed_by_vertex: dict[int, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]
    blend_weights_by_vertex: object | None
    blend_indices_by_vertex: object | None
    uv_layers: dict[str, object | None]
    uv_values_by_layer: dict[str, list[tuple[float, float]] | None]
    game_uv_values_by_layer: dict[str, list[tuple[float, float]] | None]
    uv_by_layer_loop: dict[tuple[str, int], tuple[float, float]]
    attribute_refs: dict[str, object | None]
    color_attribute_refs: dict[str, object | None]
    texcoord_snorm4_sources: dict[tuple[str, int], tuple]


@dataclass
class _ExportPartCache:
    part: ExportPartPlan
    mirror_flip_default: bool
    uv_flip_v_default: bool
    mesh_store: _EvaluatedMeshStore
    palette_to_local: dict[int, int]
    mesh_caches: dict[int, _MeshExportCache]

    @classmethod
    def from_part(
        cls,
        part: ExportPartPlan,
        *,
        mesh_store: _EvaluatedMeshStore,
        mirror_flip_default: bool = True,
        uv_flip_v_default: bool = DEFAULT_UV_FLIP_V,
    ) -> "_ExportPartCache":
        return cls(
            part=part,
            mirror_flip_default=bool(mirror_flip_default),
            uv_flip_v_default=bool(uv_flip_v_default),
            mesh_store=mesh_store,
            palette_to_local={int(global_bone): local_index for local_index, global_bone in enumerate(part.palette_values)},
            mesh_caches={},
        )

    def mesh_cache(self, loop_vertex: _LoopVertex) -> _MeshExportCache:
        return self.object_mesh_cache(loop_vertex.mesh_obj)

    def object_mesh_cache(self, mesh_obj) -> _MeshExportCache:
        key = id(mesh_obj)
        cache = self.mesh_caches.get(key)
        if cache is None:
            mesh_record = self.mesh_store.record(mesh_obj)
            cache = _MeshExportCache(
                mesh_obj=mesh_obj,
                mesh=mesh_record.mesh,
                group_index_to_global=_group_index_to_global(mesh_obj),
                mirror_flip=_object_mirror_flip(mesh_obj, self.mirror_flip_default),
                uv_flip_v=self.uv_flip_v_default,
                matrix_world_applied=mesh_record.matrix_world_applied,
                vertex_position_values=None,
                loop_normal_values=None,
                game_position_values=None,
                game_normal_values=None,
                game_packed_normal_values=None,
                game_position_by_vertex={},
                game_normal_by_loop={},
                top4_by_vertex={},
                top4_packed_by_vertex={},
                blend_weights_by_vertex=None,
                blend_indices_by_vertex=None,
                uv_layers={},
                uv_values_by_layer={},
                game_uv_values_by_layer={},
                uv_by_layer_loop={},
                attribute_refs={},
                color_attribute_refs={},
                texcoord_snorm4_sources={},
            )
            self.mesh_caches[key] = cache
        return cache


@dataclass
class _PreparedVertexSlot:
    slot_name: str
    slot_layout: dict
    role_name: str
    file_name: str
    file_path: str
    stride: int
    fields: list[dict]
    field_offsets: list[tuple[dict, int]]
    output: bytearray


def write_part_geometry_buffers(
    buffer_dir: str,
    parts: tuple[ExportPartPlan, ...],
    vertex_layout_table: dict,
    *,
    mirror_flip_default: bool = True,
    uv_flip_v_default: bool = DEFAULT_UV_FLIP_V,
) -> list[dict]:
    """Write IB/VB buffers for every export part and return manifest records."""

    os.makedirs(buffer_dir, exist_ok=True)
    records: list[dict] = []
    mesh_store = _EvaluatedMeshStore()
    try:
        for part in parts:
            records.append(
                _write_part_geometry_buffers(
                    buffer_dir,
                    part,
                    vertex_layout_table,
                    mesh_store=mesh_store,
                    mirror_flip_default=mirror_flip_default,
                    uv_flip_v_default=uv_flip_v_default,
                )
            )
    finally:
        mesh_store.release()
    return records


def _write_part_geometry_buffers(
    buffer_dir: str,
    part: ExportPartPlan,
    vertex_layout_table: dict,
    *,
    mesh_store: _EvaluatedMeshStore,
    mirror_flip_default: bool = True,
    uv_flip_v_default: bool = DEFAULT_UV_FLIP_V,
) -> dict:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    stage_start = time.perf_counter()
    layout = _resolve_part_layout(part, vertex_layout_table)
    normalized_layout = _normalize_vertex_layout(layout)
    timings["layout"] = time.perf_counter() - stage_start

    export_cache = _ExportPartCache.from_part(
        part,
        mesh_store=mesh_store,
        mirror_flip_default=mirror_flip_default,
        uv_flip_v_default=uv_flip_v_default,
    )

    stage_start = time.perf_counter()
    loop_vertices, indices = _collect_part_loop_vertices(part, export_cache)
    timings["collect_loops"] = time.perf_counter() - stage_start
    if not loop_vertices:
        raise ValueError(f"{part.region.key}/{part.part_name}: no triangles to export")

    stage_start = time.perf_counter()
    ib_file_name = f"{part.file_stem}-Index.buf"
    ib_file_path = os.path.join(buffer_dir, ib_file_name)
    write_r32_index_buffer(ib_file_path, indices)
    timings["write_ib"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    vb_records: dict[str, dict] = {}
    prepared_slots: list[_PreparedVertexSlot] = []
    for slot_name, slot_layout in sorted(normalized_layout.items(), key=lambda item: item[0]):
        if slot_name == "vb3":
            # Runtime aliases vb3 to vb0 for this game layout, so exporting a
            # duplicate position buffer only costs time and disk bandwidth.
            continue
        role_name = _slot_file_role(slot_name)
        file_name = f"{part.file_stem}-{role_name}.buf"
        file_path = os.path.join(buffer_dir, file_name)
        prepared_slots.append(_prepare_vertex_slot(slot_name, slot_layout, role_name, file_name, file_path, len(loop_vertices)))
        vb_records[slot_name] = {
            "role": role_name,
            "file_name": file_name,
            "file_path": file_path,
            "stride": int(slot_layout["stride"]),
            "vertex_count": len(loop_vertices),
            "fields": list(slot_layout["fields"]),
        }
    timings["prepare_vb"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    slot_timings = _write_prepared_vertex_slots(prepared_slots, loop_vertices, export_cache)
    timings["write_vb"] = time.perf_counter() - stage_start
    timings["total"] = time.perf_counter() - total_start

    return {
        "region_collection": part.region.collection_name,
        "part_name": part.part_name,
        "object_names": [usage.name for usage in part.object_usages],
        "ib_hash": part.region.ib_hash,
        "match_first_index": int(part.region.match_first_index),
        "match_index_count": int(part.region.match_index_count),
        "part_index": int(part.part_index),
        "index_buffer": {
            "file_name": ib_file_name,
            "file_path": ib_file_path,
            "format": "DXGI_FORMAT_R32_UINT",
            "index_count": len(indices),
        },
        "vertex_buffers": vb_records,
        "stats": {
            "mesh_object_count": len(part.mesh_objects),
            "loop_vertex_count": len(loop_vertices),
            "index_count": len(indices),
            "vb_slot_count": len(prepared_slots),
        },
        "timings": {name: round(seconds, 3) for name, seconds in timings.items()},
        "slot_timings": {name: round(seconds, 3) for name, seconds in slot_timings.items()},
    }


def _resolve_part_layout(part: ExportPartPlan, vertex_layout_table: dict) -> dict:
    if not part.mesh_objects:
        raise ValueError(f"{part.region.key}/{part.part_name}: no mesh objects to export")
    table = dict(vertex_layout_table or {})
    layout = table.get(part.region.key)
    if not isinstance(layout, dict):
        raise ValueError(
            f"{part.region.key}/{part.part_name}: no vertex layout in capture_manifest.vertex_layout_table; "
            "run Analyze Main/Build Pool before export"
        )
    return layout


def _slot_file_role(slot_name: str) -> str:
    normalized = str(slot_name or "").lower()
    if normalized == "vb0":
        return "Position"
    if normalized == "vb1":
        return "Texcoord"
    if normalized == "vb2":
        return "Blend"
    if normalized.startswith("vb") and normalized[2:].isdigit():
        return f"VB{int(normalized[2:])}"
    return normalized or "Vertex"


def _normalize_vertex_layout(layout: dict) -> dict[str, dict]:
    raw_buffers = dict(layout.get("buffers", layout.get("vertex_buffers", {})) or {})
    normalized: dict[str, dict] = {}
    for raw_slot_name, raw_slot in raw_buffers.items():
        slot_name = str(raw_slot.get("slot", raw_slot_name) or raw_slot_name).lower()
        if not slot_name.startswith("vb"):
            slot_index = int(raw_slot.get("slot_index", raw_slot.get("input_slot", -1)) or -1)
            if slot_index >= 0:
                slot_name = f"vb{slot_index}"
        stride = int(raw_slot.get("stride", 0) or 0)
        if stride <= 0:
            raise ValueError(f"{slot_name}: invalid vertex stride")
        raw_fields = list(raw_slot.get("elements", raw_slot.get("fields", [])) or [])
        fields = []
        for raw_field in raw_fields:
            semantic_name = str(raw_field.get("semantic_name", "") or "").upper()
            semantic_index = int(raw_field.get("semantic_index", 0) or 0)
            semantic = str(raw_field.get("semantic", "") or "").upper()
            if semantic and not semantic_name:
                semantic_name, semantic_index = _split_semantic(semantic)
            fields.append(
                {
                    "semantic_name": semantic_name,
                    "semantic_index": semantic_index,
                    "semantic": f"{semantic_name}{semantic_index}",
                    "format": str(raw_field.get("format", "") or "").upper(),
                    "aligned_byte_offset": int(raw_field.get("aligned_byte_offset", raw_field.get("offset", 0)) or 0),
                }
            )
        normalized[slot_name] = {"stride": stride, "fields": fields}
    if not normalized:
        raise ValueError("Vertex layout does not contain any vertex buffers")
    return normalized


def _collect_part_loop_vertices(part: ExportPartPlan, export_cache: _ExportPartCache) -> tuple[list[_LoopVertex], list[int]]:
    loop_vertices: list[_LoopVertex] = []
    indices: list[int] = []
    next_index = 0
    for mesh_obj in part.mesh_objects:
        mesh = export_cache.object_mesh_cache(mesh_obj).mesh
        if mesh is None:
            continue
        for polygon in getattr(mesh, "polygons", []) or []:
            loop_indices = [int(value) for value in getattr(polygon, "loop_indices", []) or []]
            if len(loop_indices) < 3:
                continue
            polygon_loop_vertices: list[int] = []
            for loop_index in loop_indices:
                loop = getattr(mesh, "loops", [])[loop_index]
                loop_vertices.append(
                    _LoopVertex(
                        mesh_obj=mesh_obj,
                        mesh=mesh,
                        vertex_index=int(getattr(loop, "vertex_index", 0)),
                        loop_index=int(loop_index),
                        polygon=polygon,
                    )
                )
                polygon_loop_vertices.append(next_index)
                next_index += 1
            for index in range(1, len(polygon_loop_vertices) - 1):
                triangle = (
                    polygon_loop_vertices[0],
                    polygon_loop_vertices[index],
                    polygon_loop_vertices[index + 1],
                )
                indices.extend(_reverse_triangle_winding(triangle))
    return loop_vertices, indices


def _prepare_vertex_slot(
    slot_name: str,
    slot_layout: dict,
    role_name: str,
    file_name: str,
    file_path: str,
    vertex_count: int,
) -> _PreparedVertexSlot:
    stride = int(slot_layout["stride"])
    fields = list(slot_layout["fields"])
    field_offsets: list[tuple[dict, int]] = []
    for field in fields:
        offset = int(field["aligned_byte_offset"])
        if offset < 0:
            raise ValueError(f"{slot_name}: field {field['semantic']} has negative offset")
        field_offsets.append((field, offset))
    return _PreparedVertexSlot(
        slot_name=slot_name,
        slot_layout=slot_layout,
        role_name=role_name,
        file_name=file_name,
        file_path=file_path,
        stride=stride,
        fields=fields,
        field_offsets=field_offsets,
        output=bytearray(vertex_count * stride),
    )


def _write_prepared_vertex_slots(
    prepared_slots: list[_PreparedVertexSlot],
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> dict[str, float]:
    slot_timings: dict[str, float] = {}
    for slot in prepared_slots:
        stage_start = time.perf_counter()
        if not _write_specialized_vertex_slot(slot, loop_vertices, export_cache):
            for record_index, loop_vertex in enumerate(loop_vertices):
                _write_vertex_slot_record(slot, record_index, loop_vertex, export_cache)
        directory = os.path.dirname(slot.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(slot.file_path, "wb") as file_handle:
            file_handle.write(slot.output)
        slot_timings[f"{slot.slot_name}:{slot.role_name}"] = time.perf_counter() - stage_start
    return slot_timings


def _write_vertex_slot_record(
    slot: _PreparedVertexSlot,
    record_index: int,
    loop_vertex: _LoopVertex,
    export_cache: _ExportPartCache,
) -> None:
    record_base = record_index * slot.stride
    for field, offset in slot.field_offsets:
        field_bytes = _field_bytes(slot.slot_name, slot.slot_layout, field, loop_vertex, export_cache)
        end_offset = offset + len(field_bytes)
        if end_offset > slot.stride:
            raise ValueError(f"{slot.slot_name}: field {field['semantic']} exceeds stride {slot.stride}")
        slot.output[record_base + offset:record_base + end_offset] = field_bytes


def _write_fast_prepared_vertex_slot(
    slot: _PreparedVertexSlot,
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    field_plans = _fast_field_plans(slot)
    if field_plans is None:
        return False

    output = slot.output
    stride = int(slot.stride)
    pack_into = struct.pack_into
    for record_index, loop_vertex in enumerate(loop_vertices):
        record_base = record_index * stride
        for plan in field_plans:
            kind = plan[0]
            offset = record_base + int(plan[1])
            if kind == "position3":
                pack_into("<3f", output, offset, *_game_position(loop_vertex, export_cache))
            elif kind == "normal_packed":
                pack_into("<I", output, offset, int(encode_game_packed_normal(_game_normal(loop_vertex, export_cache))))
            elif kind == "normal3":
                pack_into("<3f", output, offset, *_game_normal(loop_vertex, export_cache))
            elif kind == "uv":
                uv = _uv(loop_vertex, str(plan[2]), export_cache)
                pack_into("<2f", output, offset, float(uv[0]), float(uv[1]))
            elif kind == "texcoord_snorm4":
                _write_texcoord_snorm4_into(output, offset, slot.slot_name, int(plan[2]), loop_vertex, export_cache)
            elif kind == "blend_weights":
                weights, _indices = _local_top4_weights(loop_vertex, export_cache)
                if len(plan) > 2 and _normalize_format(str(plan[2])) == "R32G32B32A32_FLOAT":
                    pack_into("<4f", output, offset, *weights)
                else:
                    pack_into("<4H", output, offset, *(_unorm16(value) for value in weights))
            elif kind == "blend_indices":
                _weights, indices = _local_top4_weights(loop_vertex, export_cache)
                if len(plan) > 2 and _normalize_format(str(plan[2])) == "R32G32B32A32_UINT":
                    pack_into("<4I", output, offset, *indices)
                else:
                    pack_into("<4B", output, offset, *(_uint8(value) for value in indices))
            else:
                return False
    return True


def _write_specialized_vertex_slot(
    slot: _PreparedVertexSlot,
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    field_plans = _fast_field_plans(slot)
    if field_plans is None:
        return False
    role_name = str(slot.role_name or "").lower()
    if role_name == "blend":
        return _write_blend_slot(slot, field_plans, loop_vertices, export_cache)
    if role_name == "position":
        return _write_position_slot(slot, field_plans, loop_vertices, export_cache)
    if role_name == "texcoord":
        return _write_texcoord_slot(slot, field_plans, loop_vertices, export_cache)
    return _write_fast_prepared_vertex_slot(slot, loop_vertices, export_cache)


def _write_position_slot(
    slot: _PreparedVertexSlot,
    field_plans: list[tuple],
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    if any(plan[0] not in {"position3", "normal_packed", "normal3"} for plan in field_plans):
        return False
    position_offset = _single_plan_offset(field_plans, "position3")
    normal_packed_offset = _single_plan_offset(field_plans, "normal_packed")
    normal3_offset = _single_plan_offset(field_plans, "normal3")
    if position_offset is None and normal_packed_offset is None and normal3_offset is None:
        return False
    if (
        int(slot.stride) == 16
        and position_offset == 0
        and normal_packed_offset == 12
        and normal3_offset is None
    ):
        return _write_position_packed_normal_slot16(slot, loop_vertices, export_cache)
    if _write_numpy_position_slot(slot, field_plans, loop_vertices, export_cache):
        return True
    output = slot.output
    stride = int(slot.stride)
    pack_into = struct.pack_into
    current_key = None
    game_positions: list[tuple[float, float, float]] = []
    packed_normals: list[int] = []
    for record_index, loop_vertex in enumerate(loop_vertices):
        record_base = record_index * stride
        mesh_key = id(loop_vertex.mesh_obj)
        if mesh_key != current_key:
            mesh_cache = export_cache.mesh_cache(loop_vertex)
            game_positions = _game_position_values(loop_vertex.mesh, mesh_cache)
            packed_normals = _game_packed_normal_values(loop_vertex.mesh, mesh_cache)
            current_key = mesh_key
        if position_offset is not None:
            position = game_positions[loop_vertex.vertex_index] if loop_vertex.vertex_index < len(game_positions) else _game_position(loop_vertex, export_cache)
            pack_into("<3f", output, record_base + position_offset, *position)
        if normal_packed_offset is not None:
            packed_normal = packed_normals[loop_vertex.loop_index] if loop_vertex.loop_index < len(packed_normals) else int(encode_game_packed_normal(_game_normal(loop_vertex, export_cache)))
            pack_into("<I", output, record_base + normal_packed_offset, int(packed_normal))
        if normal3_offset is not None:
            pack_into("<3f", output, record_base + normal3_offset, *_game_normal(loop_vertex, export_cache))
    return True


def _write_position_packed_normal_slot16(
    slot: _PreparedVertexSlot,
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    if _write_numpy_position_packed_normal_slot16(slot, loop_vertices, export_cache):
        return True
    output = slot.output
    pack_into = struct.pack_into
    current_key = None
    game_positions: list[tuple[float, float, float]] = []
    packed_normals: list[int] = []
    for record_index, loop_vertex in enumerate(loop_vertices):
        mesh_key = id(loop_vertex.mesh_obj)
        if mesh_key != current_key:
            mesh_cache = export_cache.mesh_cache(loop_vertex)
            game_positions = _game_position_values(loop_vertex.mesh, mesh_cache)
            packed_normals = _game_packed_normal_values(loop_vertex.mesh, mesh_cache)
            current_key = mesh_key
        position = game_positions[loop_vertex.vertex_index] if loop_vertex.vertex_index < len(game_positions) else _game_position(loop_vertex, export_cache)
        packed_normal = packed_normals[loop_vertex.loop_index] if loop_vertex.loop_index < len(packed_normals) else int(encode_game_packed_normal(_game_normal(loop_vertex, export_cache)))
        pack_into("<3fI", output, record_index * 16, float(position[0]), float(position[1]), float(position[2]), int(packed_normal))
    return True


def _write_numpy_position_packed_normal_slot16(
    slot: _PreparedVertexSlot,
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    np = optional_numpy()
    if np is None or not loop_vertices or int(slot.stride) != 16:
        return False
    try:
        records = np.empty(
            len(loop_vertices),
            dtype=np.dtype([("position", "<f4", (3,)), ("normal", "<u4")]),
        )
        for start, end in _loop_vertex_mesh_ranges(loop_vertices):
            first = loop_vertices[start]
            mesh_cache = export_cache.mesh_cache(first)
            positions = np.asarray(_game_position_values(first.mesh, mesh_cache), dtype="<f4")
            normals = np.asarray(_game_packed_normal_values(first.mesh, mesh_cache), dtype="<u4")
            vertex_indices = np.fromiter(
                (loop_vertices[index].vertex_index for index in range(start, end)),
                dtype=np.intp,
                count=end - start,
            )
            loop_indices = np.fromiter(
                (loop_vertices[index].loop_index for index in range(start, end)),
                dtype=np.intp,
                count=end - start,
            )
            if vertex_indices.size and int(vertex_indices.max()) >= len(positions):
                return False
            if loop_indices.size and int(loop_indices.max()) >= len(normals):
                return False
            records["position"][start:end] = positions[vertex_indices]
            records["normal"][start:end] = normals[loop_indices]
        slot.output[:] = records.tobytes()
        return True
    except Exception:
        return False


def _write_numpy_position_slot(
    slot: _PreparedVertexSlot,
    field_plans: list[tuple],
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    np = optional_numpy()
    if np is None or not loop_vertices:
        return False
    if any(plan[0] not in {"position3", "normal_packed", "normal3"} for plan in field_plans):
        return False
    position_offset = _single_plan_offset(field_plans, "position3")
    normal_packed_offset = _single_plan_offset(field_plans, "normal_packed")
    normal3_offset = _single_plan_offset(field_plans, "normal3")
    if position_offset is None and normal_packed_offset is None and normal3_offset is None:
        return False
    stride = int(slot.stride)
    try:
        records = np.zeros((len(loop_vertices), stride), dtype=np.uint8)
        for start, end in _loop_vertex_mesh_ranges(loop_vertices):
            first = loop_vertices[start]
            mesh_cache = export_cache.mesh_cache(first)
            vertex_indices = np.fromiter(
                (loop_vertices[index].vertex_index for index in range(start, end)),
                dtype=np.intp,
                count=end - start,
            )
            loop_indices = np.fromiter(
                (loop_vertices[index].loop_index for index in range(start, end)),
                dtype=np.intp,
                count=end - start,
            )
            if position_offset is not None:
                positions = np.asarray(_game_position_values(first.mesh, mesh_cache), dtype=np.float32)
                if vertex_indices.size and int(vertex_indices.max()) >= len(positions):
                    return False
                _numpy_assign_bytes(records[start:end], int(position_offset), positions[vertex_indices])
            if normal_packed_offset is not None:
                packed_normals = np.asarray(_game_packed_normal_values(first.mesh, mesh_cache), dtype=np.uint32)
                if loop_indices.size and int(loop_indices.max()) >= len(packed_normals):
                    return False
                _numpy_assign_bytes(records[start:end], int(normal_packed_offset), packed_normals[loop_indices])
            if normal3_offset is not None:
                normals = np.asarray(_game_normal_values(first.mesh, mesh_cache), dtype=np.float32)
                if loop_indices.size and int(loop_indices.max()) >= len(normals):
                    return False
                _numpy_assign_bytes(records[start:end], int(normal3_offset), normals[loop_indices])
        slot.output[:] = records.tobytes()
        return True
    except Exception:
        return False


def _write_texcoord_slot(
    slot: _PreparedVertexSlot,
    field_plans: list[tuple],
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    uv_plans = [plan for plan in field_plans if plan[0] == "uv"]
    texcoord_snorm4_plans = [plan for plan in field_plans if plan[0] == "texcoord_snorm4"]
    if len(uv_plans) + len(texcoord_snorm4_plans) != len(field_plans):
        return False
    if _write_numpy_texcoord_slot(slot, field_plans, loop_vertices, export_cache):
        return True
    output = slot.output
    stride = int(slot.stride)
    pack_into = struct.pack_into
    single_uv = len(uv_plans) == 1
    single_snorm = len(texcoord_snorm4_plans) == 1
    uv_offset = int(uv_plans[0][1]) if single_uv else -1
    uv_layer_name = str(uv_plans[0][2]) if single_uv else ""
    snorm_offset = int(texcoord_snorm4_plans[0][1]) if single_snorm else -1
    snorm_semantic_index = int(texcoord_snorm4_plans[0][2]) if single_snorm else -1
    current_key = None
    mesh_cache = None
    primary_uv_values: list[tuple[float, float]] | None = None
    primary_snorm_source: tuple | None = None
    for record_index, loop_vertex in enumerate(loop_vertices):
        record_base = record_index * stride
        mesh_key = id(loop_vertex.mesh_obj)
        if mesh_key != current_key:
            mesh_cache = export_cache.mesh_cache(loop_vertex)
            primary_uv_values = _required_game_uv_values(loop_vertex, uv_layer_name, mesh_cache) if single_uv else None
            primary_snorm_source = (
                _texcoord_snorm4_source(loop_vertex.mesh, slot.slot_name, snorm_semantic_index, mesh_cache)
                if single_snorm
                else None
            )
            current_key = mesh_key
        if single_uv and primary_uv_values is not None:
            uv = primary_uv_values[loop_vertex.loop_index] if loop_vertex.loop_index < len(primary_uv_values) else _uv(loop_vertex, uv_layer_name, export_cache)
            if single_snorm and primary_snorm_source is not None and primary_snorm_source[0] == "zero":
                if stride == 12 and uv_offset == 0 and snorm_offset == 8:
                    pack_into("<2fI", output, record_base, float(uv[0]), float(uv[1]), 0)
                else:
                    pack_into("<2f", output, record_base + uv_offset, float(uv[0]), float(uv[1]))
                continue
            pack_into("<2f", output, record_base + uv_offset, float(uv[0]), float(uv[1]))
        else:
            for _kind, field_offset, layer_name in uv_plans:
                uv = _uv(loop_vertex, str(layer_name), export_cache)
                pack_into("<2f", output, record_base + int(field_offset), float(uv[0]), float(uv[1]))
        for _kind, field_offset, semantic_index in texcoord_snorm4_plans:
            _write_texcoord_snorm4_into(output, record_base + int(field_offset), slot.slot_name, int(semantic_index), loop_vertex, export_cache)
    return True


def _write_numpy_texcoord_slot(
    slot: _PreparedVertexSlot,
    field_plans: list[tuple],
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    np = optional_numpy()
    if np is None or not loop_vertices:
        return False
    uv_plans = [plan for plan in field_plans if plan[0] == "uv"]
    texcoord_snorm4_plans = [plan for plan in field_plans if plan[0] == "texcoord_snorm4"]
    if len(uv_plans) + len(texcoord_snorm4_plans) != len(field_plans):
        return False
    stride = int(slot.stride)
    try:
        records = np.zeros((len(loop_vertices), stride), dtype=np.uint8)
        for start, end in _loop_vertex_mesh_ranges(loop_vertices):
            first = loop_vertices[start]
            mesh_cache = export_cache.mesh_cache(first)
            loop_indices = np.fromiter(
                (loop_vertices[index].loop_index for index in range(start, end)),
                dtype=np.intp,
                count=end - start,
            )
            for _kind, field_offset, layer_name in uv_plans:
                uv_values = _game_uv_values(first.mesh, str(layer_name), mesh_cache)
                if uv_values is None:
                    return False
                uv_values = np.asarray(uv_values, dtype=np.float32)
                if loop_indices.size and int(loop_indices.max()) >= len(uv_values):
                    return False
                _numpy_assign_bytes(records[start:end], int(field_offset), uv_values[loop_indices])
            for _kind, field_offset, semantic_index in texcoord_snorm4_plans:
                source = _texcoord_snorm4_source(first.mesh, slot.slot_name, int(semantic_index), mesh_cache)
                snorm_values = _numpy_texcoord_snorm4_values(source, loop_vertices, start, end, loop_indices)
                if snorm_values is None:
                    return False
                _numpy_assign_bytes(records[start:end], int(field_offset), snorm_values)
        slot.output[:] = records.tobytes()
        return True
    except Exception:
        return False


def _write_blend_slot(
    slot: _PreparedVertexSlot,
    field_plans: list[tuple],
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    if any(plan[0] not in {"blend_weights", "blend_indices"} for plan in field_plans):
        return False
    weights_plan = _single_plan_detail(field_plans, "blend_weights")
    indices_plan = _single_plan_detail(field_plans, "blend_indices")
    weights_offset = int(weights_plan[1]) if weights_plan is not None else None
    indices_offset = int(indices_plan[1]) if indices_plan is not None else None
    if weights_offset is None and indices_offset is None:
        return False
    output = slot.output
    stride = int(slot.stride)
    pack_into = struct.pack_into
    weights_format = _normalize_format(str(weights_plan[2])) if weights_plan is not None and len(weights_plan) > 2 else ""
    indices_format = _normalize_format(str(indices_plan[2])) if indices_plan is not None and len(indices_plan) > 2 else ""
    if _write_numpy_blend_slot(slot, field_plans, loop_vertices, export_cache):
        return True
    for record_index, loop_vertex in enumerate(loop_vertices):
        record_base = record_index * stride
        weights, indices = _local_top4_weights(loop_vertex, export_cache)
        if weights_offset is not None:
            if weights_format == "R32G32B32A32_FLOAT":
                pack_into("<4f", output, record_base + weights_offset, *weights)
            else:
                pack_into("<4H", output, record_base + weights_offset, *(_unorm16(value) for value in weights))
        if indices_offset is not None:
            if indices_format == "R32G32B32A32_UINT":
                pack_into("<4I", output, record_base + indices_offset, *indices)
            else:
                pack_into("<4B", output, record_base + indices_offset, *(_uint8(value) for value in indices))
    return True


def _write_numpy_blend_slot(
    slot: _PreparedVertexSlot,
    field_plans: list[tuple],
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> bool:
    np = optional_numpy()
    if np is None or not loop_vertices:
        return False
    weights_plan = _single_plan_detail(field_plans, "blend_weights")
    indices_plan = _single_plan_detail(field_plans, "blend_indices")
    weights_offset = int(weights_plan[1]) if weights_plan is not None else None
    indices_offset = int(indices_plan[1]) if indices_plan is not None else None
    if weights_offset is None and indices_offset is None:
        return False
    weights_format = _normalize_format(str(weights_plan[2])) if weights_plan is not None and len(weights_plan) > 2 else ""
    indices_format = _normalize_format(str(indices_plan[2])) if indices_plan is not None and len(indices_plan) > 2 else ""
    if weights_format not in {"", "R16G16B16A16_UNORM", "R32G32B32A32_FLOAT"}:
        return False
    if indices_format not in {"", "R8G8B8A8_UINT", "R32G32B32A32_UINT"}:
        return False
    try:
        records = np.zeros((len(loop_vertices), int(slot.stride)), dtype=np.uint8)
        for start, end in _loop_vertex_mesh_ranges(loop_vertices):
            first = loop_vertices[start]
            mesh_cache = export_cache.mesh_cache(first)
            vertex_indices = np.fromiter(
                (loop_vertices[index].vertex_index for index in range(start, end)),
                dtype=np.intp,
                count=end - start,
            )
            weights_by_vertex, indices_by_vertex = _local_top4_vertex_arrays(first.mesh, mesh_cache, export_cache)
            weights_by_vertex = np.asarray(weights_by_vertex)
            indices_by_vertex = np.asarray(indices_by_vertex)
            if weights_offset is not None:
                if vertex_indices.size and int(vertex_indices.max()) >= len(weights_by_vertex):
                    return False
                weights_array = weights_by_vertex[vertex_indices]
                if weights_format == "R16G16B16A16_UNORM":
                    weights_array = np.rint(np.clip(np.asarray(weights_array, dtype=np.float32), 0.0, 1.0) * 65535.0).astype(np.uint16)
                else:
                    weights_array = np.asarray(weights_array, dtype=np.float32)
                _numpy_assign_bytes(records[start:end], weights_offset, weights_array)
            if indices_offset is not None:
                if vertex_indices.size and int(vertex_indices.max()) >= len(indices_by_vertex):
                    return False
                indices_array = indices_by_vertex[vertex_indices]
                if indices_format == "R32G32B32A32_UINT":
                    indices_array = np.asarray(indices_array, dtype=np.uint32)
                else:
                    indices_array = np.asarray(indices_array, dtype=np.uint8)
                _numpy_assign_bytes(records[start:end], indices_offset, indices_array)
        slot.output[:] = records.tobytes()
        return True
    except Exception:
        return False


def _loop_vertex_mesh_ranges(loop_vertices: list[_LoopVertex]):
    if not loop_vertices:
        return
    start = 0
    current_key = id(loop_vertices[0].mesh_obj)
    for index in range(1, len(loop_vertices)):
        key = id(loop_vertices[index].mesh_obj)
        if key == current_key:
            continue
        yield start, index
        start = index
        current_key = key
    yield start, len(loop_vertices)


def _numpy_assign_bytes(target, offset: int, values) -> None:
    np = optional_numpy()
    if np is None:
        raise ValueError("numpy is required for the fast path")
    array = np.ascontiguousarray(values)
    if array.size == 0:
        return
    byte_view = array.view(np.uint8).reshape(int(array.shape[0]), -1)
    target[:, int(offset):int(offset) + byte_view.shape[1]] = byte_view


def _numpy_texcoord_snorm4_values(
    source: tuple,
    loop_vertices: list[_LoopVertex],
    start: int,
    end: int,
    loop_indices,
) -> object | None:
    np = optional_numpy()
    if np is None:
        return None
    kind = source[0]
    count = end - start
    if kind == "zero":
        return np.zeros((count, 4), dtype=np.uint8)
    if kind == "point":
        component_data = source[1]
        values = np.zeros((count, 4), dtype=np.uint8)
        for component, data in enumerate(component_data):
            values[:, component] = np.fromiter(
                (
                    _raw_byte_from_attribute_data(data, int(loop_vertices[index].vertex_index))
                    for index in range(start, end)
                ),
                dtype=np.uint8,
                count=count,
            )
        return values
    if kind == "color":
        attribute, use_loop_index = source[1], bool(source[2])
        data = getattr(attribute, "data", [])
        values = np.zeros((count, 4), dtype=np.uint8)
        for local_index, loop_index in enumerate(loop_indices):
            data_index = int(loop_index if use_loop_index else loop_vertices[start + int(local_index)].vertex_index)
            item = data[data_index] if data_index < len(data) else None
            color_values = _color_values_from_item(item) if item is not None else None
            if color_values is None:
                continue
            for component, value in enumerate(color_values[:4]):
                values[local_index, component] = color_component_to_raw_byte(value)
        return values
    return None


def _single_plan_offset(field_plans: list[tuple], kind: str) -> int | None:
    offsets = [int(plan[1]) for plan in field_plans if plan[0] == kind]
    if len(offsets) > 1:
        return None
    return offsets[0] if offsets else None


def _single_plan_detail(field_plans: list[tuple], kind: str) -> tuple | None:
    plans = [plan for plan in field_plans if plan[0] == kind]
    if len(plans) > 1:
        return None
    return plans[0] if plans else None


def _fast_field_plans(slot: _PreparedVertexSlot) -> list[tuple] | None:
    plans: list[tuple] = []
    for field, offset in slot.field_offsets:
        semantic_name = str(field["semantic_name"]).upper()
        semantic_index = int(field["semantic_index"])
        fmt = _normalize_format(str(field["format"]))
        if semantic_name == "POSITION" and semantic_index == 0 and fmt == "R32G32B32_FLOAT":
            plans.append(("position3", offset))
        elif semantic_name == "NORMAL" and semantic_index == 0 and fmt == "R32_FLOAT":
            plans.append(("normal_packed", offset))
        elif semantic_name == "NORMAL" and semantic_index == 0 and fmt == "R32G32B32_FLOAT":
            plans.append(("normal3", offset))
        elif semantic_name == "TEXCOORD" and semantic_index in {0, 1} and fmt == "R32G32_FLOAT":
            plans.append(("uv", offset, f"UV{semantic_index}"))
        elif semantic_name == "TEXCOORD" and fmt == "R8G8B8A8_SNORM":
            plans.append(("texcoord_snorm4", offset, semantic_index))
        elif semantic_name == "BLENDWEIGHTS" and semantic_index == 0 and fmt in {"R16G16B16A16_UNORM", "R32G32B32A32_FLOAT"}:
            plans.append(("blend_weights", offset, fmt))
        elif semantic_name == "BLENDINDICES" and semantic_index == 0 and fmt in {"R8G8B8A8_UINT", "R32G32B32A32_UINT"}:
            plans.append(("blend_indices", offset, fmt))
        else:
            return None
    return plans


def _unorm16(value: float) -> int:
    return max(0, min(65535, round(max(0.0, min(1.0, float(value))) * 65535)))


def _uint8(value: int) -> int:
    return max(0, min(255, int(round(float(value)))))


def _write_texcoord_snorm4_into(
    output: bytearray,
    offset: int,
    slot_name: str,
    semantic_index: int,
    loop_vertex: _LoopVertex,
    export_cache: _ExportPartCache,
) -> None:
    mesh_cache = export_cache.mesh_cache(loop_vertex)
    source = _texcoord_snorm4_source(loop_vertex.mesh, slot_name, semantic_index, mesh_cache)
    kind = source[0]
    if kind == "zero":
        struct.pack_into("<I", output, offset, 0)
        return
    if kind == "point":
        attribute_data = source[1]
        vertex_index = int(loop_vertex.vertex_index)
        for component, data in enumerate(attribute_data):
            output[offset + component] = _raw_byte_from_attribute_data(data, vertex_index)
        return
    if kind == "color":
        attribute, use_loop_index = source[1], bool(source[2])
        data = getattr(attribute, "data", [])
        data_index = int(loop_vertex.loop_index if use_loop_index else loop_vertex.vertex_index)
        values = _color_values_from_item(data[data_index]) if data_index < len(data) else None
        if values is None:
            struct.pack_into("<I", output, offset, 0)
            return
        for component, value in enumerate(values[:4]):
            output[offset + component] = color_component_to_raw_byte(value)
        return
    struct.pack_into("<I", output, offset, 0)


def _texcoord_snorm4_source(mesh, slot_name: str, semantic_index: int, mesh_cache: _MeshExportCache) -> tuple:
    key = (str(slot_name), int(semantic_index))
    if key in mesh_cache.texcoord_snorm4_sources:
        return mesh_cache.texcoord_snorm4_sources[key]

    component_data = []
    for component in range(4):
        attribute = _point_attribute(mesh, texcoord_component_attr_names(slot_name, semantic_index, component), mesh_cache)
        if attribute is None:
            component_data = []
            break
        component_data.append(getattr(attribute, "data", []))
    if len(component_data) == 4:
        source = ("point", tuple(component_data))
    else:
        color_attribute = _color_attribute(mesh, texcoord_color_attr_names(slot_name, semantic_index), mesh_cache)
        if color_attribute is None:
            source = ("zero",)
        else:
            data = getattr(color_attribute, "data", [])
            domain = str(getattr(color_attribute, "domain", "") or "").upper()
            use_loop_index = domain == "CORNER" or len(data) == len(getattr(mesh, "loops", []) or [])
            source = ("color", color_attribute, use_loop_index)

    mesh_cache.texcoord_snorm4_sources[key] = source
    return source


def _point_attribute(mesh, names: tuple[str, ...], mesh_cache: _MeshExportCache):
    for name in names:
        attribute = mesh_cache.attribute_refs.get(name)
        if name not in mesh_cache.attribute_refs:
            attributes = getattr(mesh, "attributes", None)
            attribute = None
            if attributes is not None:
                getter = getattr(attributes, "get", None)
                if callable(getter):
                    attribute = getter(name)
                elif isinstance(attributes, dict):
                    attribute = attributes.get(name)
            mesh_cache.attribute_refs[name] = attribute
        if attribute is not None:
            return attribute
    return None


def _raw_byte_from_attribute_data(data, index: int) -> int:
    if index >= len(data):
        return 0
    item = data[index]
    if hasattr(item, "value"):
        return int(getattr(item, "value")) & 0xFF
    if hasattr(item, "vector"):
        vector = getattr(item, "vector")
        if vector:
            return int(vector[0]) & 0xFF
    return 0


def _field_bytes(
    slot_name: str,
    slot_layout: dict,
    field: dict,
    loop_vertex: _LoopVertex,
    export_cache: _ExportPartCache,
) -> bytes:
    semantic_name = str(field["semantic_name"]).upper()
    semantic_index = int(field["semantic_index"])
    fmt = str(field["format"]).upper()
    if semantic_name == "POSITION" and semantic_index == 0:
        return pack_vertex_format(fmt, _game_position(loop_vertex, export_cache))
    if semantic_name == "NORMAL" and semantic_index == 0:
        normal = _game_normal(loop_vertex, export_cache)
        if _normalize_format(fmt) == "R32_FLOAT":
            return int(encode_game_packed_normal(normal)).to_bytes(4, "little", signed=False)
        return pack_vertex_format(fmt, normal)
    if semantic_name == "TEXCOORD" and semantic_index in {0, 1} and _normalize_format(fmt) == "R32G32_FLOAT":
        return pack_vertex_format(fmt, _uv(loop_vertex, f"UV{semantic_index}", export_cache))
    if semantic_name == "TEXCOORD":
        return _texcoord_field_bytes(slot_name, slot_layout, field, loop_vertex, export_cache)
    if semantic_name == "BLENDWEIGHTS" and semantic_index == 0:
        weights, _indices = _local_top4_weights(loop_vertex, export_cache)
        return pack_vertex_format(fmt, weights)
    if semantic_name == "BLENDINDICES" and semantic_index == 0:
        _weights, indices = _local_top4_weights(loop_vertex, export_cache)
        return pack_vertex_format(fmt, indices)
    raise ValueError(f"{slot_name}: unsupported export semantic {semantic_name}{semantic_index}")


def _game_position(loop_vertex: _LoopVertex, export_cache: _ExportPartCache) -> tuple[float, float, float]:
    mesh_cache = export_cache.mesh_cache(loop_vertex)
    cached = mesh_cache.game_position_by_vertex.get(loop_vertex.vertex_index)
    if cached is not None:
        return cached
    game_positions = _game_position_values(loop_vertex.mesh, mesh_cache)
    if loop_vertex.vertex_index < len(game_positions):
        co = game_positions[loop_vertex.vertex_index]
        mesh_cache.game_position_by_vertex[loop_vertex.vertex_index] = co
        return co
    position_values = _vertex_position_values(loop_vertex.mesh, mesh_cache)
    if loop_vertex.vertex_index < len(position_values):
        co = position_values[loop_vertex.vertex_index]
    else:
        vertex = getattr(loop_vertex.mesh, "vertices")[loop_vertex.vertex_index]
        co = _vector3(getattr(vertex, "co", (0.0, 0.0, 0.0)))
    if not mesh_cache.matrix_world_applied:
        co = _transform_point(loop_vertex.mesh_obj, co)
    if mesh_cache.mirror_flip:
        co = (-co[0], co[1], co[2])
    mesh_cache.game_position_by_vertex[loop_vertex.vertex_index] = co
    return co


def _game_normal(loop_vertex: _LoopVertex, export_cache: _ExportPartCache) -> tuple[float, float, float]:
    mesh_cache = export_cache.mesh_cache(loop_vertex)
    cached = mesh_cache.game_normal_by_loop.get(loop_vertex.loop_index)
    if cached is not None:
        return cached
    normal_values = _game_normal_values(loop_vertex.mesh, mesh_cache)
    if loop_vertex.loop_index < len(normal_values):
        normal = normal_values[loop_vertex.loop_index]
    else:
        loop = getattr(loop_vertex.mesh, "loops", [])[loop_vertex.loop_index]
        normal = _vector3(getattr(loop, "normal", None) or getattr(loop_vertex.polygon, "normal", None) or (0.0, 0.0, 1.0))
        if not mesh_cache.matrix_world_applied:
            normal = _transform_normal(loop_vertex.mesh_obj, normal)
        if mesh_cache.mirror_flip:
            normal = (-normal[0], normal[1], normal[2])
        normal = _normalize3(normal)
    mesh_cache.game_normal_by_loop[loop_vertex.loop_index] = normal
    return normal


def _uv(loop_vertex: _LoopVertex, layer_name: str, export_cache: _ExportPartCache) -> tuple[float, float]:
    mesh_cache = export_cache.mesh_cache(loop_vertex)
    cache_key = (layer_name, loop_vertex.loop_index)
    cached = mesh_cache.uv_by_layer_loop.get(cache_key)
    if cached is not None:
        return cached
    uv_values = _game_uv_values(loop_vertex.mesh, layer_name, mesh_cache)
    if uv_values is None:
        raise ValueError(f"{getattr(loop_vertex.mesh_obj, 'name', '<mesh>')}: missing required UV layer {layer_name}")
    result = uv_values[loop_vertex.loop_index] if loop_vertex.loop_index < len(uv_values) else (0.0, 0.0)
    mesh_cache.uv_by_layer_loop[cache_key] = result
    return result


def _resolve_uv_layer(uv_layers, layer_name: str):
    if uv_layers is None:
        return None
    normalized = str(layer_name or "").upper()
    if normalized == "UV0":
        return _active_uv_layer(uv_layers) or _named_uv_layer(uv_layers, layer_name) or _indexed_uv_layer(uv_layers, 0)
    if normalized == "UV1":
        return (
            _named_uv_layer(uv_layers, layer_name)
            or _indexed_uv_layer(uv_layers, 1)
            or _active_uv_layer(uv_layers)
            or _indexed_uv_layer(uv_layers, 0)
        )
    return _named_uv_layer(uv_layers, layer_name)


def _named_uv_layer(uv_layers, layer_name: str):
    getter = getattr(uv_layers, "get", None)
    if callable(getter):
        layer = getter(layer_name)
        if layer is not None:
            return layer
    if isinstance(uv_layers, dict):
        return uv_layers.get(layer_name)
    return None


def _active_uv_layer(uv_layers):
    for attribute_name in ("active_render", "active"):
        layer = getattr(uv_layers, attribute_name, None)
        if layer is not None:
            return layer
    active_index = getattr(uv_layers, "active_index", None)
    if active_index is not None:
        return _indexed_uv_layer(uv_layers, int(active_index))
    return None


def _indexed_uv_layer(uv_layers, index: int):
    if index < 0:
        return None
    if isinstance(uv_layers, dict):
        values = list(uv_layers.values())
        return values[index] if index < len(values) else None
    try:
        return uv_layers[index]
    except Exception:
        pass
    try:
        values = list(uv_layers)
    except Exception:
        return None
    return values[index] if index < len(values) else None


def _game_position_values(mesh, mesh_cache: _MeshExportCache) -> list[tuple[float, float, float]]:
    if mesh_cache.game_position_values is not None:
        return mesh_cache.game_position_values
    np = optional_numpy()
    values = _vertex_position_values(mesh, mesh_cache)
    if np is not None and hasattr(values, "shape"):
        positions = np.asarray(values, dtype=np.float32)
        if not mesh_cache.matrix_world_applied:
            positions = _transform_points_numpy(mesh_cache.mesh_obj, positions)
        if mesh_cache.mirror_flip:
            positions = positions.copy()
            positions[:, 0] *= -1.0
        mesh_cache.game_position_values = positions
        return positions
    result = []
    for co in values:
        position = co
        if not mesh_cache.matrix_world_applied:
            position = _transform_point(mesh_cache.mesh_obj, position)
        if mesh_cache.mirror_flip:
            position = (-position[0], position[1], position[2])
        result.append(position)
    mesh_cache.game_position_values = result
    return result


def _game_normal_values(mesh, mesh_cache: _MeshExportCache) -> list[tuple[float, float, float]]:
    if mesh_cache.game_normal_values is not None:
        return mesh_cache.game_normal_values
    np = optional_numpy()
    values = _loop_normal_values(mesh, mesh_cache)
    if np is not None and hasattr(values, "shape"):
        normals = np.asarray(values, dtype=np.float32)
        if not mesh_cache.matrix_world_applied:
            normals = _transform_normals_numpy(mesh_cache.mesh_obj, normals)
        if mesh_cache.mirror_flip:
            normals = normals.copy()
            normals[:, 0] *= -1.0
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.where(lengths <= 1e-12, 1.0, lengths)
        normals = normals / lengths
        mesh_cache.game_normal_values = normals
        return normals
    result = []
    for normal in values:
        game_normal = normal
        if not mesh_cache.matrix_world_applied:
            game_normal = _transform_normal(mesh_cache.mesh_obj, game_normal)
        if mesh_cache.mirror_flip:
            game_normal = (-game_normal[0], game_normal[1], game_normal[2])
        result.append(_normalize3(game_normal))
    mesh_cache.game_normal_values = result
    return result


def _game_packed_normal_values(mesh, mesh_cache: _MeshExportCache) -> list[int]:
    if mesh_cache.game_packed_normal_values is not None:
        return mesh_cache.game_packed_normal_values
    np = optional_numpy()
    values = _game_normal_values(mesh, mesh_cache)
    if np is not None and hasattr(values, "shape"):
        normals = np.asarray(values, dtype=np.float32)
        if normals.size == 0:
            packed = np.zeros((0,), dtype=np.uint32)
        else:
            x_values = normals[:, 0]
            y_values = normals[:, 1]
            z_values = normals[:, 2]
            inv_l1 = 1.0 / np.maximum(np.abs(x_values) + np.abs(y_values) + np.abs(z_values), 1e-12)
            oct_x = x_values * inv_l1
            oct_y = y_values * inv_l1
            fold_mask = z_values < 0.0
            if np.any(fold_mask):
                sign_x = np.where(oct_x >= 0.0, 1.0, -1.0)
                sign_y = np.where(oct_y >= 0.0, 1.0, -1.0)
                folded_x = (1.0 - np.abs(oct_y)) * sign_x
                folded_y = (1.0 - np.abs(oct_x)) * sign_y
                oct_x = np.where(fold_mask, folded_x, oct_x)
                oct_y = np.where(fold_mask, folded_y, oct_y)
            quant_x = np.rint(np.clip(oct_x, -1.0, 1.0) * 511.0).astype(np.int32)
            quant_y = np.rint(np.clip(oct_y, -1.0, 1.0) * 511.0).astype(np.int32)
            packed = (0x40000000 | (quant_x & 0x3FF) | ((quant_y & 0x3FF) << 10)).astype(np.uint32)
        mesh_cache.game_packed_normal_values = packed
        return packed
    result = []
    for normal in values:
        result.append(int(encode_game_packed_normal(normal)))
    mesh_cache.game_packed_normal_values = result
    return result


def _required_game_uv_values(loop_vertex: _LoopVertex, layer_name: str, mesh_cache: _MeshExportCache) -> list[tuple[float, float]]:
    uv_values = _game_uv_values(loop_vertex.mesh, layer_name, mesh_cache)
    if uv_values is None:
        raise ValueError(f"{getattr(loop_vertex.mesh_obj, 'name', '<mesh>')}: missing required UV layer {layer_name}")
    return uv_values


def _game_uv_values(mesh, layer_name: str, mesh_cache: _MeshExportCache) -> list[tuple[float, float]] | None:
    if layer_name in mesh_cache.game_uv_values_by_layer:
        return mesh_cache.game_uv_values_by_layer[layer_name]
    layer = mesh_cache.uv_layers.get(layer_name)
    if layer_name not in mesh_cache.uv_layers:
        uv_layers = getattr(mesh, "uv_layers", None)
        layer = _resolve_uv_layer(uv_layers, layer_name)
        mesh_cache.uv_layers[layer_name] = layer
    if layer is None:
        mesh_cache.game_uv_values_by_layer[layer_name] = None
        return None
    raw_uv_values = _uv_layer_values(layer, layer_name, mesh_cache)
    np = optional_numpy()
    if np is not None and hasattr(raw_uv_values, "shape"):
        uv_values = np.asarray(raw_uv_values, dtype=np.float32)
        if mesh_cache.uv_flip_v:
            uv_values = uv_values.copy()
            uv_values[:, 1] = 1.0 - uv_values[:, 1]
        mesh_cache.game_uv_values_by_layer[layer_name] = uv_values
        return uv_values
    uv_values = [
        blender_uv_to_game((float(uv[0]), float(uv[1])), flip_v=mesh_cache.uv_flip_v)
        for uv in raw_uv_values
    ]
    mesh_cache.game_uv_values_by_layer[layer_name] = uv_values
    return uv_values


def _vertex_position_values(mesh, mesh_cache: _MeshExportCache) -> list[tuple[float, float, float]]:
    if mesh_cache.vertex_position_values is None:
        mesh_cache.vertex_position_values = _float_tuple_values(getattr(mesh, "vertices", []) or [], "co", 3)
    return mesh_cache.vertex_position_values


def _loop_normal_values(mesh, mesh_cache: _MeshExportCache) -> list[tuple[float, float, float]]:
    if mesh_cache.loop_normal_values is None:
        mesh_cache.loop_normal_values = _float_tuple_values(getattr(mesh, "loops", []) or [], "normal", 3)
    return mesh_cache.loop_normal_values


def _uv_layer_values(layer, layer_name: str, mesh_cache: _MeshExportCache) -> list[tuple[float, float]]:
    if layer_name not in mesh_cache.uv_values_by_layer:
        mesh_cache.uv_values_by_layer[layer_name] = _float_tuple_values(getattr(layer, "data", []) or [], "uv", 2)
    cached = mesh_cache.uv_values_by_layer[layer_name]
    return cached if cached is not None else []


def _float_tuple_values(collection, attribute_name: str, component_count: int) -> list[tuple]:
    count = len(collection)
    if count <= 0:
        return []
    np = optional_numpy()
    if np is not None:
        try:
            flat_values = np.empty(count * component_count, dtype=np.float32)
            foreach_get = getattr(collection, "foreach_get", None)
            if callable(foreach_get):
                foreach_get(attribute_name, flat_values)
                return flat_values.reshape((count, component_count))
        except Exception:
            pass
    flat_values = [0.0] * (count * component_count)
    foreach_get = getattr(collection, "foreach_get", None)
    if callable(foreach_get):
        try:
            foreach_get(attribute_name, flat_values)
            return [
                tuple(float(flat_values[index + component]) for component in range(component_count))
                for index in range(0, len(flat_values), component_count)
            ]
        except Exception:
            pass

    values = []
    fallback = (0.0,) * component_count
    for item in collection:
        raw_value = getattr(item, attribute_name, fallback)
        try:
            values.append(tuple(float(raw_value[component]) for component in range(component_count)))
        except Exception:
            values.append(tuple(float(fallback[component]) for component in range(component_count)))
    return values


def _optional_uv(loop_vertex: _LoopVertex, layer_name: str, export_cache: _ExportPartCache) -> tuple[float, float] | None:
    try:
        return _uv(loop_vertex, layer_name, export_cache)
    except ValueError:
        return None


def _texcoord_field_bytes(
    slot_name: str,
    slot_layout: dict,
    field: dict,
    loop_vertex: _LoopVertex,
    export_cache: _ExportPartCache,
) -> bytes:
    fmt = _normalize_format(str(field["format"]))
    semantic_index = int(field["semantic_index"])
    component_count = _format_component_count(fmt)
    values = [
        _point_attribute_value(
            loop_vertex.mesh,
            texcoord_component_attr_names(slot_name, semantic_index, component),
            loop_vertex.vertex_index,
            export_cache.mesh_cache(loop_vertex),
        )
        for component in range(component_count)
    ]
    if fmt == "R8G8B8A8_SNORM":
        if not any(value is None for value in values):
            return bytes((int(value) & 0xFF) for value in values)
        color_bytes = _texcoord_color_bytes(slot_name, semantic_index, loop_vertex, export_cache)
        if color_bytes is not None:
            return color_bytes
        return _default_texcoord_field_bytes(slot_layout, field, loop_vertex, export_cache)
    elif not any(value is None for value in values):
        return pack_vertex_format(fmt, [float(value) for value in values])

    return _default_texcoord_field_bytes(slot_layout, field, loop_vertex, export_cache)


def _default_texcoord_field_bytes(
    slot_layout: dict,
    field: dict,
    loop_vertex: _LoopVertex,
    export_cache: _ExportPartCache,
) -> bytes:
    fmt = _normalize_format(str(field["format"]))
    if fmt == "R8G8B8A8_SNORM":
        return b"\x00\x00\x00\x00"
    if fmt == "R32G32_FLOAT":
        uv_layer = _aliased_texcoord_uv_layer(slot_layout, field)
        uv = _optional_uv(loop_vertex, uv_layer, export_cache) if uv_layer else None
        if uv is None:
            uv = _optional_uv(loop_vertex, "UV1", export_cache)
        if uv is None:
            uv = _optional_uv(loop_vertex, "UV0", export_cache)
        return pack_vertex_format(fmt, uv if uv is not None else (0.0, 0.0))
    if fmt == "R32G32B32_FLOAT":
        return pack_vertex_format(fmt, _game_position(loop_vertex, export_cache))
    return pack_vertex_format(fmt, [0.0] * _format_component_count(fmt))


def _aliased_texcoord_uv_layer(slot_layout: dict, field: dict) -> str:
    field_offset = int(field.get("aligned_byte_offset", 0) or 0)
    field_format = _normalize_format(str(field.get("format", "") or ""))
    for other in list(slot_layout.get("fields", []) or []):
        if other is field:
            continue
        if str(other.get("semantic_name", "") or "").upper() != "TEXCOORD":
            continue
        other_index = int(other.get("semantic_index", -1) or -1)
        if other_index not in {0, 1}:
            continue
        if int(other.get("aligned_byte_offset", -1) or -1) != field_offset:
            continue
        if _normalize_format(str(other.get("format", "") or "")) != field_format:
            continue
        return f"UV{other_index}"
    return ""


def _local_top4_weights_for_vertex(
    mesh,
    vertex_index: int,
    mesh_cache: _MeshExportCache,
    export_cache: _ExportPartCache,
) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
    cached = mesh_cache.top4_by_vertex.get(vertex_index)
    if cached is not None:
        return cached
    vertex = getattr(mesh, "vertices")[vertex_index]
    weighted: list[tuple[int, float]] = []
    for group_element in getattr(vertex, "groups", []) or []:
        global_group = mesh_cache.group_index_to_global.get(int(group_element.group))
        if global_group is None:
            continue
        local_index = export_cache.palette_to_local.get(int(global_group))
        if local_index is None:
            continue
        weight = float(group_element.weight)
        if weight <= 0.0:
            continue
        weighted.append((local_index, weight))
    weighted.sort(key=lambda item: (-item[1], item[0]))
    top = weighted[:4]
    total = sum(weight for _index, weight in top)
    if total <= 1e-12:
        result = (0.0, 0.0, 0.0, 0.0), (0, 0, 0, 0)
        mesh_cache.top4_by_vertex[vertex_index] = result
        return result
    weights = [weight / total for _index, weight in top]
    indices = [index for index, _weight in top]
    while len(weights) < 4:
        weights.append(0.0)
        indices.append(0)
    result = tuple(weights[:4]), tuple(indices[:4])
    mesh_cache.top4_by_vertex[vertex_index] = result
    return result


def _local_top4_weights(loop_vertex: _LoopVertex, export_cache: _ExportPartCache) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
    mesh_cache = export_cache.mesh_cache(loop_vertex)
    return _local_top4_weights_for_vertex(loop_vertex.mesh, loop_vertex.vertex_index, mesh_cache, export_cache)


def _local_top4_packed(loop_vertex: _LoopVertex, export_cache: _ExportPartCache) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    mesh_cache = export_cache.mesh_cache(loop_vertex)
    cached = mesh_cache.top4_packed_by_vertex.get(loop_vertex.vertex_index)
    if cached is not None:
        return cached
    weights, indices = _local_top4_weights(loop_vertex, export_cache)
    result = (
        tuple(_unorm16(value) for value in weights),
        tuple(_uint8(value) for value in indices),
    )
    mesh_cache.top4_packed_by_vertex[loop_vertex.vertex_index] = result
    return result


def _local_top4_vertex_arrays(mesh, mesh_cache: _MeshExportCache, export_cache: _ExportPartCache):
    np = optional_numpy()
    vertex_count = len(getattr(mesh, "vertices", []) or [])
    if np is not None:
        cached_weights = mesh_cache.blend_weights_by_vertex
        cached_indices = mesh_cache.blend_indices_by_vertex
        if cached_weights is not None and cached_indices is not None:
            return cached_weights, cached_indices
        weights_array = np.zeros((vertex_count, 4), dtype=np.float32)
        indices_array = np.zeros((vertex_count, 4), dtype=np.uint32)
        for vertex_index in range(vertex_count):
            weights, indices = _local_top4_weights_for_vertex(mesh, vertex_index, mesh_cache, export_cache)
            weights_array[vertex_index] = weights
            indices_array[vertex_index] = indices
        mesh_cache.blend_weights_by_vertex = weights_array
        mesh_cache.blend_indices_by_vertex = indices_array
        return weights_array, indices_array
    cached_weights = mesh_cache.blend_weights_by_vertex
    cached_indices = mesh_cache.blend_indices_by_vertex
    if cached_weights is not None and cached_indices is not None:
        return cached_weights, cached_indices
    weights_list: list[tuple[float, float, float, float]] = []
    indices_list: list[tuple[int, int, int, int]] = []
    for vertex_index in range(vertex_count):
        weights, indices = _local_top4_weights_for_vertex(mesh, vertex_index, mesh_cache, export_cache)
        weights_list.append(weights)
        indices_list.append(indices)
    mesh_cache.blend_weights_by_vertex = weights_list
    mesh_cache.blend_indices_by_vertex = indices_list
    return weights_list, indices_list


def _group_index_to_global(mesh_obj) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for vertex_group in getattr(mesh_obj, "vertex_groups", []) or []:
        try:
            global_group = int(str(vertex_group.name))
        except ValueError:
            continue
        mapping[int(vertex_group.index)] = global_group
    return mapping


def _texcoord_color_bytes(
    slot_name: str,
    semantic_index: int,
    loop_vertex: _LoopVertex,
    export_cache: _ExportPartCache,
) -> bytes | None:
    mesh_cache = export_cache.mesh_cache(loop_vertex)
    attribute = _color_attribute(loop_vertex.mesh, texcoord_color_attr_names(slot_name, semantic_index), mesh_cache)
    if attribute is None:
        return None
    data = getattr(attribute, "data", [])
    domain = str(getattr(attribute, "domain", "") or "").upper()
    if domain == "CORNER" or len(data) == len(getattr(loop_vertex.mesh, "loops", []) or []):
        data_index = loop_vertex.loop_index
    else:
        data_index = loop_vertex.vertex_index
    if data_index >= len(data):
        return None
    values = _color_values_from_item(data[data_index])
    if values is None:
        return None
    return bytes(color_component_to_raw_byte(value) for value in values[:4])


def _color_attribute(mesh, names: tuple[str, ...], mesh_cache: _MeshExportCache):
    for name in names:
        attribute = mesh_cache.color_attribute_refs.get(name)
        if name not in mesh_cache.color_attribute_refs:
            attribute = None
            color_attributes = getattr(mesh, "color_attributes", None)
            if color_attributes is not None:
                getter = getattr(color_attributes, "get", None)
                if callable(getter):
                    attribute = getter(name)
            if attribute is None:
                attributes = getattr(mesh, "attributes", None)
                getter = getattr(attributes, "get", None) if attributes is not None else None
                if callable(getter):
                    attribute = getter(name)
            mesh_cache.color_attribute_refs[name] = attribute
        if attribute is not None:
            return attribute
    return None


def _color_values_from_item(item) -> tuple[float, float, float, float] | None:
    value = getattr(item, "color", None)
    if value is None:
        value = getattr(item, "vector", None)
    if value is None:
        return None
    if len(value) < 4:
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _point_attribute_value(mesh, names: tuple[str, ...], vertex_index: int, mesh_cache: _MeshExportCache):
    for name in names:
        attribute = mesh_cache.attribute_refs.get(name)
        if name not in mesh_cache.attribute_refs:
            attributes = getattr(mesh, "attributes", None)
            attribute = None
            if attributes is not None:
                getter = getattr(attributes, "get", None)
                if callable(getter):
                    attribute = getter(name)
                elif isinstance(attributes, dict):
                    attribute = attributes.get(name)
            mesh_cache.attribute_refs[name] = attribute
        if attribute is None:
            continue
        data = getattr(attribute, "data", [])
        if vertex_index >= len(data):
            continue
        item = data[vertex_index]
        if hasattr(item, "value"):
            return item.value
        if hasattr(item, "vector"):
            return item.vector
    return None


def _object_get(obj, key: str, default=None):
    getter = getattr(obj, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return obj[key]
    except Exception:
        return default


def _object_mirror_flip(obj, default: bool) -> bool:
    value = _object_get(obj, "bmc_mirror_flip", None)
    if value is None:
        value = _object_get(obj, "modimp_mirror_flip", None)
    if value is None:
        return bool(default)
    return bool(value)


def _evaluated_export_mesh(mesh_obj) -> tuple[object | None, bool, object | None]:
    original_mesh = getattr(mesh_obj, "data", None)
    try:
        import bpy  # type: ignore
    except Exception:
        return original_mesh, False, None

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_obj = mesh_obj.evaluated_get(depsgraph)
    except Exception:
        return original_mesh, False, None

    try:
        mesh_copy = bpy.data.meshes.new_from_object(
            evaluated_obj,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
    except TypeError:
        try:
            mesh_copy = bpy.data.meshes.new_from_object(evaluated_obj, depsgraph=depsgraph)
        except Exception:
            return original_mesh, False, None
    except Exception:
        return original_mesh, False, None
    if mesh_copy is None:
        return original_mesh, False, None

    try:
        mesh_copy.transform(evaluated_obj.matrix_world)
        mesh_copy.update()
        _triangulate_mesh_copy(mesh_copy)
        mesh_copy.update()
        calc_loop_triangles = getattr(mesh_copy, "calc_loop_triangles", None)
        if callable(calc_loop_triangles):
            calc_loop_triangles()
    except Exception:
        _release_owned_mesh(mesh_copy)
        return original_mesh, False, None
    return mesh_copy, True, mesh_copy


def _triangulate_mesh_copy(mesh) -> None:
    polygons = getattr(mesh, "polygons", []) or []
    if not any(int(getattr(polygon, "loop_total", len(getattr(polygon, "loop_indices", []) or [])) or 0) != 3 for polygon in polygons):
        return
    try:
        import bmesh  # type: ignore
    except Exception:
        return
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
    finally:
        bm.free()


def _release_owned_mesh(mesh) -> None:
    try:
        import bpy  # type: ignore
    except Exception:
        return
    try:
        bpy.data.meshes.remove(mesh)
    except Exception:
        pass


def _split_semantic(semantic: str) -> tuple[str, int]:
    raw = str(semantic or "").upper()
    suffix = ""
    while raw and raw[-1].isdigit():
        suffix = raw[-1] + suffix
        raw = raw[:-1]
    return raw, int(suffix or 0)


def _normalize_format(fmt: str) -> str:
    normalized = str(fmt or "").upper()
    if normalized.startswith("DXGI_FORMAT_"):
        normalized = normalized[len("DXGI_FORMAT_"):]
    return normalized


def _format_component_count(fmt: str) -> int:
    normalized = _normalize_format(fmt)
    if normalized.startswith("R8G8B8A8") or normalized.startswith("R16G16B16A16") or normalized.startswith("R32G32B32A32"):
        return 4
    if normalized.startswith("R32G32B32"):
        return 3
    if normalized.startswith("R32G32"):
        return 2
    return 1


def _vector3(value) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _transform_point(mesh_obj, value: tuple[float, float, float]) -> tuple[float, float, float]:
    matrix = getattr(mesh_obj, "matrix_world", None)
    if matrix is None:
        return value
    try:
        transformed = matrix @ value
        return _vector3(transformed)
    except Exception:
        vector = _mathutils_vector(value)
        if vector is None:
            return value
        try:
            return _vector3(matrix @ vector)
        except Exception:
            return value


def _transform_points_numpy(mesh_obj, values):
    np = optional_numpy()
    if np is None:
        return values
    matrix = _matrix_world_numpy(mesh_obj)
    if matrix is None:
        return values
    try:
        points = np.asarray(values, dtype=np.float32)
        if points.size == 0:
            return points
        hom = np.ones((len(points), 4), dtype=np.float32)
        hom[:, :3] = points[:, :3]
        transformed = hom @ matrix.T
        return transformed[:, :3]
    except Exception:
        return values


def _transform_normal(mesh_obj, value: tuple[float, float, float]) -> tuple[float, float, float]:
    matrix = getattr(mesh_obj, "matrix_world", None)
    if matrix is None:
        return value
    try:
        normal_matrix = matrix.to_3x3()
        transformed = normal_matrix @ value
        return _vector3(transformed)
    except Exception:
        vector = _mathutils_vector(value)
        if vector is None:
            return value
        try:
            return _vector3(matrix.to_3x3() @ vector)
        except Exception:
            return value


def _transform_normals_numpy(mesh_obj, values):
    np = optional_numpy()
    if np is None:
        return values
    matrix = _matrix_world_numpy(mesh_obj)
    if matrix is None:
        return values
    try:
        normals = np.asarray(values, dtype=np.float32)
        if normals.size == 0:
            return normals
        matrix3 = matrix[:3, :3]
        transformed = normals @ matrix3.T
        lengths = np.linalg.norm(transformed, axis=1, keepdims=True)
        lengths = np.where(lengths <= 1e-12, 1.0, lengths)
        return transformed / lengths
    except Exception:
        return values


def _normalize3(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (value[0] / length, value[1] / length, value[2] / length)


def _reverse_triangle_winding(triangle: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(triangle[0]), int(triangle[2]), int(triangle[1])


def _mathutils_vector(value: tuple[float, float, float]):
    try:
        from mathutils import Vector  # type: ignore
    except Exception:
        return None
    return Vector(value)


def _matrix_world_numpy(mesh_obj):
    np = optional_numpy()
    if np is None:
        return None
    matrix = getattr(mesh_obj, "matrix_world", None)
    if matrix is None:
        return None
    try:
        return np.asarray(matrix, dtype=np.float32)
    except Exception:
        try:
            return np.asarray([list(row) for row in matrix], dtype=np.float32)
        except Exception:
            return None
