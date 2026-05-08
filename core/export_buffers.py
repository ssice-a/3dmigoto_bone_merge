"""Write per-part geometry buffers for BoneMerge exports."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

from .texcoord_attrs import (
    color_component_to_raw_byte,
    texcoord_color_attr_names,
    texcoord_component_attr_names,
)
from .export_package import ExportPartPlan, write_r32_index_buffer
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
    game_position_by_vertex: dict[int, tuple[float, float, float]]
    game_normal_by_loop: dict[int, tuple[float, float, float]]
    top4_by_vertex: dict[int, tuple[tuple[float, float, float, float], tuple[int, int, int, int]]]
    uv_layers: dict[str, object | None]
    uv_by_layer_loop: dict[tuple[str, int], tuple[float, float]]
    attribute_refs: dict[str, object | None]
    color_attribute_refs: dict[str, object | None]


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
                game_position_by_vertex={},
                game_normal_by_loop={},
                top4_by_vertex={},
                uv_layers={},
                uv_by_layer_loop={},
                attribute_refs={},
                color_attribute_refs={},
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
    _write_prepared_vertex_slots(prepared_slots, loop_vertices, export_cache)
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
) -> None:
    for record_index, loop_vertex in enumerate(loop_vertices):
        for slot in prepared_slots:
            _write_vertex_slot_record(slot, record_index, loop_vertex, export_cache)
    for slot in prepared_slots:
        directory = os.path.dirname(slot.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(slot.file_path, "wb") as file_handle:
            file_handle.write(slot.output)


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
    layer = mesh_cache.uv_layers.get(layer_name)
    if layer_name not in mesh_cache.uv_layers:
        uv_layers = getattr(loop_vertex.mesh, "uv_layers", None)
        layer = _resolve_uv_layer(uv_layers, layer_name)
        mesh_cache.uv_layers[layer_name] = layer
    if layer is None:
        raise ValueError(f"{getattr(loop_vertex.mesh_obj, 'name', '<mesh>')}: missing required UV layer {layer_name}")
    uv = getattr(layer.data[loop_vertex.loop_index], "uv", (0.0, 0.0))
    result = blender_uv_to_game((float(uv[0]), float(uv[1])), flip_v=mesh_cache.uv_flip_v)
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


def _local_top4_weights(loop_vertex: _LoopVertex, export_cache: _ExportPartCache) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
    mesh_cache = export_cache.mesh_cache(loop_vertex)
    cached = mesh_cache.top4_by_vertex.get(loop_vertex.vertex_index)
    if cached is not None:
        return cached
    vertex = getattr(loop_vertex.mesh, "vertices")[loop_vertex.vertex_index]
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
        mesh_cache.top4_by_vertex[loop_vertex.vertex_index] = result
        return result
    weights = [weight / total for _index, weight in top]
    indices = [index for index, _weight in top]
    while len(weights) < 4:
        weights.append(0.0)
        indices.append(0)
    result = tuple(weights[:4]), tuple(indices[:4])
    mesh_cache.top4_by_vertex[loop_vertex.vertex_index] = result
    return result


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
