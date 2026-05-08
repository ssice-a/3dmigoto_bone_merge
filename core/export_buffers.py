"""Write per-part geometry buffers for BoneMerge exports."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .texcoord_attrs import (
    color_component_to_raw_byte,
    texcoord_color_attr_names,
    texcoord_component_attr_names,
)
from .export_package import ExportPartPlan, write_r32_index_buffer
from .vertex_format import encode_game_packed_normal, pack_vertex_format


@dataclass(frozen=True)
class _LoopVertex:
    mesh_obj: object
    mesh: object
    vertex_index: int
    loop_index: int
    polygon: object


@dataclass
class _MeshExportCache:
    mesh_obj: object
    mesh: object
    group_index_to_global: dict[int, int]
    mirror_flip: bool
    top4_by_vertex: dict[int, tuple[tuple[float, float, float, float], tuple[int, int, int, int]]]
    uv_layers: dict[str, object | None]
    attribute_refs: dict[str, object | None]
    color_attribute_refs: dict[str, object | None]


@dataclass
class _ExportPartCache:
    part: ExportPartPlan
    mirror_flip_default: bool
    palette_to_local: dict[int, int]
    mesh_caches: dict[int, _MeshExportCache]

    @classmethod
    def from_part(cls, part: ExportPartPlan, *, mirror_flip_default: bool = True) -> "_ExportPartCache":
        return cls(
            part=part,
            mirror_flip_default=bool(mirror_flip_default),
            palette_to_local={int(global_bone): local_index for local_index, global_bone in enumerate(part.palette_values)},
            mesh_caches={},
        )

    def mesh_cache(self, loop_vertex: _LoopVertex) -> _MeshExportCache:
        return self.object_mesh_cache(loop_vertex.mesh_obj)

    def object_mesh_cache(self, mesh_obj) -> _MeshExportCache:
        key = id(mesh_obj)
        cache = self.mesh_caches.get(key)
        if cache is None:
            mesh = getattr(mesh_obj, "data", None)
            cache = _MeshExportCache(
                mesh_obj=mesh_obj,
                mesh=mesh,
                group_index_to_global=_group_index_to_global(mesh_obj),
                mirror_flip=_object_mirror_flip(mesh_obj, self.mirror_flip_default),
                top4_by_vertex={},
                uv_layers={},
                attribute_refs={},
                color_attribute_refs={},
            )
            self.mesh_caches[key] = cache
        return cache


def write_part_geometry_buffers(
    buffer_dir: str,
    parts: tuple[ExportPartPlan, ...],
    vertex_layout_table: dict,
    *,
    mirror_flip_default: bool = True,
) -> list[dict]:
    """Write IB/VB buffers for every export part and return manifest records."""

    os.makedirs(buffer_dir, exist_ok=True)
    records: list[dict] = []
    for part in parts:
        records.append(
            _write_part_geometry_buffers(
                buffer_dir,
                part,
                vertex_layout_table,
                mirror_flip_default=mirror_flip_default,
            )
        )
    return records


def _write_part_geometry_buffers(
    buffer_dir: str,
    part: ExportPartPlan,
    vertex_layout_table: dict,
    *,
    mirror_flip_default: bool = True,
) -> dict:
    layout = _resolve_part_layout(part, vertex_layout_table)
    normalized_layout = _normalize_vertex_layout(layout)
    loop_vertices, indices = _collect_part_loop_vertices(part)
    if not loop_vertices:
        raise ValueError(f"{part.region.key}/{part.part_name}: no triangles to export")
    export_cache = _ExportPartCache.from_part(
        part,
        mirror_flip_default=mirror_flip_default,
    )

    ib_file_name = f"{part.file_stem}-Index.buf"
    ib_file_path = os.path.join(buffer_dir, ib_file_name)
    write_r32_index_buffer(ib_file_path, indices)

    vb_records: dict[str, dict] = {}
    for slot_name, slot_layout in sorted(normalized_layout.items(), key=lambda item: item[0]):
        if slot_name == "vb3":
            # Runtime aliases vb3 to vb0 for this game layout, so exporting a
            # duplicate position buffer only costs time and disk bandwidth.
            continue
        role_name = _slot_file_role(slot_name)
        file_name = f"{part.file_stem}-{role_name}.buf"
        file_path = os.path.join(buffer_dir, file_name)
        _write_vertex_slot_buffer(file_path, slot_name, slot_layout, loop_vertices, export_cache)
        vb_records[slot_name] = {
            "role": role_name,
            "file_name": file_name,
            "file_path": file_path,
            "stride": int(slot_layout["stride"]),
            "vertex_count": len(loop_vertices),
            "fields": list(slot_layout["fields"]),
        }

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


def _collect_part_loop_vertices(part: ExportPartPlan) -> tuple[list[_LoopVertex], list[int]]:
    loop_vertices: list[_LoopVertex] = []
    indices: list[int] = []
    next_index = 0
    for mesh_obj in part.mesh_objects:
        mesh = getattr(mesh_obj, "data", None)
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


def _write_vertex_slot_buffer(
    file_path: str,
    slot_name: str,
    slot_layout: dict,
    loop_vertices: list[_LoopVertex],
    export_cache: _ExportPartCache,
) -> None:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    stride = int(slot_layout["stride"])
    output = bytearray(len(loop_vertices) * stride)
    fields = list(slot_layout["fields"])
    for record_index, loop_vertex in enumerate(loop_vertices):
        record_base = record_index * stride
        for field in fields:
            field_bytes = _field_bytes(slot_name, slot_layout, field, loop_vertex, export_cache)
            offset = int(field["aligned_byte_offset"])
            end_offset = offset + len(field_bytes)
            if offset < 0 or end_offset > stride:
                raise ValueError(f"{slot_name}: field {field['semantic']} exceeds stride {stride}")
            output[record_base + offset:record_base + end_offset] = field_bytes
    with open(file_path, "wb") as file_handle:
        file_handle.write(output)


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
    vertex = getattr(loop_vertex.mesh, "vertices")[loop_vertex.vertex_index]
    co = _vector3(getattr(vertex, "co", (0.0, 0.0, 0.0)))
    co = _transform_point(loop_vertex.mesh_obj, co)
    if export_cache.mesh_cache(loop_vertex).mirror_flip:
        co = (-co[0], co[1], co[2])
    return co


def _game_normal(loop_vertex: _LoopVertex, export_cache: _ExportPartCache) -> tuple[float, float, float]:
    loop = getattr(loop_vertex.mesh, "loops", [])[loop_vertex.loop_index]
    normal = _vector3(getattr(loop, "normal", None) or getattr(loop_vertex.polygon, "normal", None) or (0.0, 0.0, 1.0))
    normal = _transform_normal(loop_vertex.mesh_obj, normal)
    if export_cache.mesh_cache(loop_vertex).mirror_flip:
        normal = (-normal[0], normal[1], normal[2])
    return _normalize3(normal)


def _uv(loop_vertex: _LoopVertex, layer_name: str, export_cache: _ExportPartCache) -> tuple[float, float]:
    mesh_cache = export_cache.mesh_cache(loop_vertex)
    layer = mesh_cache.uv_layers.get(layer_name)
    if layer_name not in mesh_cache.uv_layers:
        uv_layers = getattr(loop_vertex.mesh, "uv_layers", None)
        layer = None
        if uv_layers is not None:
            getter = getattr(uv_layers, "get", None)
            if callable(getter):
                layer = getter(layer_name)
            elif isinstance(uv_layers, dict):
                layer = uv_layers.get(layer_name)
        mesh_cache.uv_layers[layer_name] = layer
    if layer is None:
        raise ValueError(f"{getattr(loop_vertex.mesh_obj, 'name', '<mesh>')}: missing required UV layer {layer_name}")
    uv = getattr(layer.data[loop_vertex.loop_index], "uv", (0.0, 0.0))
    return (float(uv[0]), float(uv[1]))


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
