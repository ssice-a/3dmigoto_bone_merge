"""Import redesigned candidate IB slices from FrameAnalysis buffers."""

from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Iterable

from .main_analyze import BufferHeader, HeaderElement, _parse_buffer_header


_VERTEX_DATA_RE = re.compile(
    r"^vb(?P<slot>\d+)\[(?P<vertex>\d+)\]\+(?P<offset>\d+)\s+[^:]+:\s*(?P<values>.+)$"
)


@dataclass(frozen=True)
class LoadedCandidateGeometry:
    display_name: str
    ib_hash: str
    match_first_index: int
    match_index_count: int
    positions: list[tuple[float, float, float]]
    triangles: list[tuple[int, int, int]]
    original_vertex_ids: list[int]
    uv0: list[tuple[float, float]]
    uv1: list[tuple[float, float]]
    normals: list[tuple[float, float, float]]
    normal_packed: list[int]
    texcoord4_raw: list[tuple[int, int, int, int]]
    blend_indices: list[tuple[int, int, int, int]]
    blend_weights: list[tuple[float, float, float, float]]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _SlotSlice:
    slot_name: str
    slot_index: int
    buf_path: str
    header: BufferHeader
    elements: dict[tuple[str, int], HeaderElement]
    base_offset: int


def load_candidate_geometry(candidate: dict, frameanalysis_dir: str = "") -> LoadedCandidateGeometry:
    """Load one candidate from .buf slices into a compact CPU geometry payload."""

    import_paths = dict(candidate.get("import_paths", {}) or {})
    ib_txt_path = _resolve_path(str(import_paths.get("ib", "") or ""), frameanalysis_dir)
    ib_buf_path = _resolve_path(str(import_paths.get("ib_buf", "") or ""), frameanalysis_dir)
    if not ib_txt_path or not os.path.exists(ib_txt_path):
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no IB txt path")
    if not ib_buf_path or not os.path.exists(ib_buf_path):
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no IB buf path")

    ib_header = _parse_buffer_header(ib_txt_path)
    indices = _read_index_buffer(ib_buf_path, ib_header)
    if len(indices) < 3:
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no triangles")

    raw_triangles = [
        (int(indices[index]), int(indices[index + 1]), int(indices[index + 2]))
        for index in range(0, len(indices) - 2, 3)
    ]
    original_vertex_ids = sorted({vertex_index for triangle in raw_triangles for vertex_index in triangle})
    remap = {old_vertex_index: new_vertex_index for new_vertex_index, old_vertex_index in enumerate(original_vertex_ids)}
    triangles = [
        (remap[triangle[0]], remap[triangle[1]], remap[triangle[2]])
        for triangle in raw_triangles
    ]

    vb_payload = dict(import_paths.get("vb", {}) or {})
    warnings: list[str] = []
    vb0 = _load_slot_slice(vb_payload, "vb0", 0, frameanalysis_dir, warnings)
    if vb0 is None:
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no vb0 slice")
    positions = _read_required_float3(vb0, "POSITION", 0, original_vertex_ids)

    normal_element = _find_element(vb0, "NORMAL", 0)
    if normal_element is not None and _is_format(normal_element.fmt, "R32_FLOAT"):
        normal_packed = _read_uint32_records(vb0, normal_element, original_vertex_ids)
        normals = [decode_game_packed_normal(value) for value in normal_packed]
    elif normal_element is not None and _is_format(normal_element.fmt, "R32G32B32_FLOAT"):
        normals = [
            _normalize_vector((float(record[0]), float(record[1]), float(record[2])))
            for record in _read_format_records(vb0, normal_element, original_vertex_ids)
        ]
        normal_packed = [0 for _ in original_vertex_ids]
    else:
        warnings.append(f"{vb0.slot_name}: NORMAL0 is missing; imported normals use +Z fallback")
        normal_packed = [0 for _ in original_vertex_ids]
        normals = [(0.0, 0.0, 1.0) for _ in original_vertex_ids]

    vb1 = _load_slot_slice(vb_payload, "vb1", 1, frameanalysis_dir, warnings)
    if vb1 is not None:
        uv0 = _read_float2_or_default(vb1, "TEXCOORD", 0, original_vertex_ids, (0.0, 0.0))
        uv1_element = _find_element(vb1, "TEXCOORD", 1)
        if uv1_element is not None:
            uv1 = _read_float2_or_default(vb1, "TEXCOORD", 1, original_vertex_ids, (0.0, 0.0))
        else:
            uv1 = list(uv0)
        texcoord4_element = _find_element(vb1, "TEXCOORD", 4)
        if texcoord4_element is not None and _is_format(texcoord4_element.fmt, "R8G8B8A8_SNORM"):
            texcoord4_raw = _read_int8x4_records(vb1, texcoord4_element, original_vertex_ids)
        else:
            texcoord4_raw = [(0, 0, 0, 0) for _ in original_vertex_ids]
    else:
        warnings.append("vb1 is missing; UV0/UV1/TEXCOORD4 use fallback values")
        uv0 = [(0.0, 0.0) for _ in original_vertex_ids]
        uv1 = list(uv0)
        texcoord4_raw = [(0, 0, 0, 0) for _ in original_vertex_ids]

    vb2 = _load_slot_slice(vb_payload, "vb2", 2, frameanalysis_dir, warnings)
    if vb2 is not None:
        blend_weights = _read_blend_weights(vb2, original_vertex_ids)
        blend_indices = _read_blend_indices(vb2, original_vertex_ids)
        if not _blend_indices_are_valid(blend_indices):
            warnings.append(f"{vb2.slot_name}: BLENDINDICES0 contains values outside 0..255; weights skipped")
            blend_weights = [(0.0, 0.0, 0.0, 0.0) for _ in original_vertex_ids]
            blend_indices = [(0, 0, 0, 0) for _ in original_vertex_ids]
    else:
        warnings.append("vb2 is missing; blend weights and indices use fallback values")
        blend_weights = [(0.0, 0.0, 0.0, 0.0) for _ in original_vertex_ids]
        blend_indices = [(0, 0, 0, 0) for _ in original_vertex_ids]

    return LoadedCandidateGeometry(
        display_name=str(candidate.get("display_name", "") or _candidate_display_name(candidate)),
        ib_hash=str(candidate.get("ib_hash", "") or "").lower(),
        match_first_index=int(candidate.get("match_first_index", 0) or 0),
        match_index_count=int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0),
        positions=positions,
        triangles=triangles,
        original_vertex_ids=original_vertex_ids,
        uv0=uv0,
        uv1=uv1,
        normals=normals,
        normal_packed=normal_packed,
        texcoord4_raw=texcoord4_raw,
        blend_indices=blend_indices,
        blend_weights=blend_weights,
        warnings=warnings,
    )


def import_selected_candidates(context, manifest: dict, selected_display_names: Iterable[str], target_collection):
    """Create Blender mesh objects for selected candidate display names."""

    import bpy  # type: ignore

    selected_names = {str(value) for value in selected_display_names if str(value)}
    if not selected_names:
        return []
    frameanalysis_dir = str(manifest.get("frameanalysis_dir", "") or "")
    imported_objects = []
    for candidate in manifest.get("candidate_ibs", []) or []:
        display_name = str(candidate.get("display_name", "") or _candidate_display_name(candidate))
        if display_name not in selected_names:
            continue
        geometry = load_candidate_geometry(candidate, frameanalysis_dir)
        imported_object = create_blender_object_from_geometry(
            bpy,
            geometry,
            target_collection,
            draw_indices=list(candidate.get("draw_indices", []) or []),
            shadow_draw_indices=list(candidate.get("shadow_draw_indices", []) or []),
        )
        imported_objects.append(imported_object)

    if imported_objects:
        for selected_object in context.selected_objects:
            selected_object.select_set(False)
        for imported_object in imported_objects:
            imported_object.select_set(True)
        context.view_layer.objects.active = imported_objects[0]
    return imported_objects


def create_blender_object_from_geometry(
    bpy_module,
    geometry: LoadedCandidateGeometry,
    target_collection,
    *,
    draw_indices: list[int],
    shadow_draw_indices: list[int],
):
    mesh = bpy_module.data.meshes.new(geometry.display_name)
    imported_object = bpy_module.data.objects.new(geometry.display_name, mesh)
    target_collection.objects.link(imported_object)

    # Flip winding for Blender front faces while keeping source vertex data intact.
    blender_triangles = [(a, c, b) for a, b, c in geometry.triangles]
    mesh.from_pydata(geometry.positions, [], blender_triangles)
    mesh.validate(verbose=False, clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    _apply_uv_layer(mesh, "UV0", geometry.uv0)
    _apply_uv_layer(mesh, "UV1", geometry.uv1)
    _apply_custom_normals(mesh, geometry.normals)
    _store_int_attribute(mesh, "bmc_orig_vertex_id", geometry.original_vertex_ids)
    _store_uint32_split_attributes(mesh, "bmc_normal_packed", geometry.normal_packed)
    for channel_index in range(4):
        _store_int_attribute(
            mesh,
            f"bmc_texcoord4_raw_{channel_index}",
            [record[channel_index] for record in geometry.texcoord4_raw],
        )
        _store_int_attribute(
            mesh,
            f"bmc_blend_index_{channel_index}",
            [record[channel_index] for record in geometry.blend_indices],
        )
        _store_float_attribute(
            mesh,
            f"bmc_blend_weight_{channel_index}",
            [record[channel_index] for record in geometry.blend_weights],
        )

    _assign_vertex_groups(imported_object, geometry.blend_indices, geometry.blend_weights)
    imported_object.merge_ib_hash = geometry.ib_hash
    imported_object.merge_match_index_count = int(geometry.match_index_count)
    imported_object.merge_ib_autodetected = False
    imported_object["bmc_source_ib_hash"] = geometry.ib_hash
    imported_object["bmc_match_first_index"] = int(geometry.match_first_index)
    imported_object["bmc_match_index_count"] = int(geometry.match_index_count)
    imported_object["bmc_draw_indices"] = ",".join(str(value) for value in draw_indices)
    imported_object["bmc_shadow_draw_indices"] = ",".join(str(value) for value in shadow_draw_indices)
    if geometry.warnings:
        imported_object["bmc_import_warnings"] = "\n".join(geometry.warnings)
    return imported_object


def decode_game_packed_normal(value: int) -> tuple[float, float, float]:
    """Best-effort decode for the packed NORMAL0 path used by this game."""

    packed = int(value) & 0xFFFFFFFF
    if not (packed & 0x40000000):
        return (0.0, 0.0, 1.0)
    x_value = _signed_10(packed & 0x3FF) / 511.0
    y_value = _signed_10((packed >> 10) & 0x3FF) / 511.0
    z_value = 1.0 - abs(x_value) - abs(y_value)
    normal_x = x_value
    normal_y = y_value
    if z_value < 0.0:
        normal_x = (1.0 - abs(y_value)) * (1.0 if x_value >= 0.0 else -1.0)
        normal_y = (1.0 - abs(x_value)) * (1.0 if y_value >= 0.0 else -1.0)
    return _normalize_vector((normal_x, normal_y, z_value))


def _candidate_display_name(candidate: dict) -> str:
    return (
        f"{str(candidate.get('ib_hash', '') or '').lower()}-"
        f"{int(candidate.get('match_index_count', candidate.get('source_index_count', 0)) or 0)}-"
        f"{int(candidate.get('match_first_index', 0) or 0)}"
    )


def _resolve_path(path: str, frameanalysis_dir: str = "") -> str:
    if not path:
        return ""
    expanded_path = os.path.abspath(path) if os.path.isabs(path) else os.path.abspath(os.path.join(frameanalysis_dir, path))
    return expanded_path


def _read_index_buffer(path: str, header: BufferHeader) -> list[int]:
    index_count = int(header.index_count)
    if index_count <= 0:
        return []
    fmt = str(header.fmt or "").upper()
    if "R16_UINT" in fmt:
        stride = 2
        unpack_format = "<H"
    elif "R32_UINT" in fmt:
        stride = 4
        unpack_format = "<I"
    else:
        raise ValueError(f"Unsupported IB format: {header.fmt}")
    first_index = int(header.first_index)
    with open(path, "rb") as file_handle:
        file_handle.seek(int(header.byte_offset) + first_index * stride)
        data = file_handle.read(index_count * stride)
    if len(data) < index_count * stride:
        raise ValueError(f"IB buffer is shorter than expected: {path}")
    return [
        int(struct.unpack_from(unpack_format, data, index * stride)[0])
        for index in range(index_count)
    ]


def _load_slot_slice(
    vb_payload: dict,
    slot_name: str,
    slot_index: int,
    frameanalysis_dir: str,
    warnings: list[str],
) -> _SlotSlice | None:
    slot_payload = dict(vb_payload.get(slot_name, {}) or {})
    if not slot_payload:
        return None
    txt_paths = [
        _resolve_path(str(path), frameanalysis_dir)
        for path in list(slot_payload.get("txt", []) or [])
        if str(path)
    ]
    layout_paths = [
        _resolve_path(str(path), frameanalysis_dir)
        for path in list(slot_payload.get("layout_txt", []) or [])
        if str(path)
    ]
    header_paths = _existing_paths(txt_paths + layout_paths)
    if not header_paths:
        warnings.append(f"{slot_name}: no txt layout path exists")
        return None
    header = _first_valid_header(header_paths)
    if header.stride <= 0 or header.vertex_count <= 0:
        warnings.append(f"{slot_name}: invalid stride/count in layout")
        return None
    elements = _merge_slot_elements(header_paths, slot_index)
    buf_path = _resolve_path(str(slot_payload.get("buf", "") or ""), frameanalysis_dir)
    if not buf_path or not os.path.exists(buf_path):
        warnings.append(f"{slot_name}: missing buf path")
        return None
    base_offset = _infer_slot_base_offset(buf_path, header, slot_name, slot_index, elements, header_paths)
    if base_offset != header.byte_offset:
        warnings.append(f"{slot_name}: corrected byte offset by {base_offset - int(header.byte_offset)} bytes")
    return _SlotSlice(
        slot_name=slot_name,
        slot_index=slot_index,
        buf_path=buf_path,
        header=header,
        elements=elements,
        base_offset=base_offset,
    )


def _existing_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    existing: list[str] = []
    for path in paths:
        normalized = os.path.abspath(path)
        if normalized in seen or not os.path.exists(normalized):
            continue
        seen.add(normalized)
        existing.append(normalized)
    return existing


def _first_valid_header(paths: list[str]) -> BufferHeader:
    for path in paths:
        header = _parse_buffer_header(path)
        if header.stride > 0 and header.vertex_count > 0:
            return header
    return _parse_buffer_header(paths[0])


def _merge_slot_elements(paths: list[str], slot_index: int) -> dict[tuple[str, int], HeaderElement]:
    elements: dict[tuple[str, int], HeaderElement] = {}
    for path in paths:
        header = _parse_buffer_header(path)
        for element in header.elements:
            if element.input_slot != slot_index:
                continue
            key = (element.semantic_name.upper(), int(element.semantic_index))
            current = elements.get(key)
            if current is None or _element_rank(element) > _element_rank(current):
                elements[key] = element
    return elements


def _element_rank(element: HeaderElement) -> int:
    rank = 0
    if element.fmt:
        rank += 1
    if element.aligned_byte_offset >= 0:
        rank += 1
    return rank


def _infer_slot_base_offset(
    buf_path: str,
    header: BufferHeader,
    slot_name: str,
    slot_index: int,
    elements: dict[tuple[str, int], HeaderElement],
    txt_paths: list[str],
) -> int:
    samples = _first_vertex_samples(slot_name, txt_paths)
    if not samples:
        return int(header.byte_offset)
    keyed_by_offset = {
        int(element.aligned_byte_offset): element
        for element in elements.values()
        if element.aligned_byte_offset >= 0
    }
    comparable_samples = [
        (field_offset, keyed_by_offset[field_offset], values)
        for field_offset, values in samples.items()
        if field_offset in keyed_by_offset
    ]
    if not comparable_samples:
        return int(header.byte_offset)
    with open(buf_path, "rb") as file_handle:
        for delta in range(-64, 65):
            base_offset = int(header.byte_offset) + delta
            if base_offset < 0:
                continue
            if all(
                _sample_matches(file_handle, base_offset + field_offset, element, values)
                for field_offset, element, values in comparable_samples
            ):
                return base_offset
    return int(header.byte_offset)


def _first_vertex_samples(slot_name: str, txt_paths: list[str]) -> dict[int, list[float]]:
    slot_number = int(slot_name[2:]) if slot_name.startswith("vb") and slot_name[2:].isdigit() else -1
    for path in txt_paths:
        samples: dict[int, list[float]] = {}
        in_vertex_data = False
        with open(path, "r", encoding="utf-8", errors="replace") as file_handle:
            for raw_line in file_handle:
                line = raw_line.strip()
                if line == "vertex-data:":
                    in_vertex_data = True
                    continue
                if not in_vertex_data:
                    continue
                match = _VERTEX_DATA_RE.match(line)
                if not match:
                    continue
                if int(match.group("slot")) != slot_number:
                    continue
                if int(match.group("vertex")) != 0:
                    break
                samples[int(match.group("offset"))] = _parse_numeric_values(match.group("values"))
        if samples:
            return samples
    return {}


def _parse_numeric_values(value_text: str) -> list[float]:
    values: list[float] = []
    for raw_value in value_text.split(","):
        text = raw_value.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            pass
    return values


def _sample_matches(file_handle, byte_offset: int, element: HeaderElement, expected: list[float]) -> bool:
    if not expected:
        return False
    try:
        file_handle.seek(byte_offset)
        data = file_handle.read(_format_size(element.fmt))
    except OSError:
        return False
    if len(data) < _format_size(element.fmt):
        return False
    actual = _unpack_format(data, 0, element.fmt)
    if len(actual) < len(expected):
        return False
    return all(_values_match(actual[index], expected[index]) for index in range(len(expected)))


def _values_match(actual: float, expected: float) -> bool:
    tolerance = 1e-5 if abs(expected) < 1.0 else 1e-4
    return abs(float(actual) - float(expected)) <= tolerance


def _find_element(slot: _SlotSlice, semantic_name: str, semantic_index: int) -> HeaderElement | None:
    return slot.elements.get((semantic_name.upper(), int(semantic_index)))


def _read_required_float3(
    slot: _SlotSlice,
    semantic_name: str,
    semantic_index: int,
    vertex_ids: list[int],
) -> list[tuple[float, float, float]]:
    element = _find_element(slot, semantic_name, semantic_index)
    if element is None:
        raise ValueError(f"{slot.slot_name}: {semantic_name}{semantic_index} is missing")
    values = _read_format_records(slot, element, vertex_ids)
    return [(float(record[0]), float(record[1]), float(record[2])) for record in values]


def _read_float2_or_default(
    slot: _SlotSlice,
    semantic_name: str,
    semantic_index: int,
    vertex_ids: list[int],
    default: tuple[float, float],
) -> list[tuple[float, float]]:
    element = _find_element(slot, semantic_name, semantic_index)
    if element is None:
        return [default for _ in vertex_ids]
    values = _read_format_records(slot, element, vertex_ids)
    return [(float(record[0]), float(record[1])) for record in values]


def _read_uint32_records(slot: _SlotSlice, element: HeaderElement, vertex_ids: list[int]) -> list[int]:
    raw_records = _read_raw_records(slot, element, vertex_ids, 4)
    return [int(struct.unpack("<I", record)[0]) for record in raw_records]


def _read_int8x4_records(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
) -> list[tuple[int, int, int, int]]:
    raw_records = _read_raw_records(slot, element, vertex_ids, 4)
    return [tuple(_signed_int8(byte_value) for byte_value in record) for record in raw_records]


def _read_blend_weights(
    slot: _SlotSlice,
    vertex_ids: list[int],
) -> list[tuple[float, float, float, float]]:
    element = _find_element(slot, "BLENDWEIGHTS", 0)
    if element is None:
        return [(0.0, 0.0, 0.0, 0.0) for _ in vertex_ids]
    values = _read_format_records(slot, element, vertex_ids)
    return [
        (float(record[0]), float(record[1]), float(record[2]), float(record[3]))
        for record in values
    ]


def _read_blend_indices(
    slot: _SlotSlice,
    vertex_ids: list[int],
) -> list[tuple[int, int, int, int]]:
    element = _find_element(slot, "BLENDINDICES", 0)
    if element is None:
        return [(0, 0, 0, 0) for _ in vertex_ids]
    values = _read_format_records(slot, element, vertex_ids)
    return [
        (int(record[0]), int(record[1]), int(record[2]), int(record[3]))
        for record in values
    ]


def _blend_indices_are_valid(blend_indices: list[tuple[int, int, int, int]]) -> bool:
    return all(0 <= int(value) <= 255 for record in blend_indices for value in record)


def _read_format_records(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
) -> list[tuple[float, ...]]:
    raw_records = _read_raw_records(slot, element, vertex_ids, _format_size(element.fmt))
    return [_unpack_format(record, 0, element.fmt) for record in raw_records]


def _read_raw_records(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
    size: int,
) -> list[bytes]:
    if not vertex_ids:
        return []
    max_vertex_id = max(vertex_ids)
    if max_vertex_id >= int(slot.header.vertex_count):
        raise ValueError(
            f"{slot.slot_name}: vertex id {max_vertex_id} exceeds vertex_count {slot.header.vertex_count}"
        )
    with open(slot.buf_path, "rb") as file_handle:
        file_handle.seek(int(slot.base_offset))
        data = file_handle.read(int(slot.header.vertex_count) * int(slot.header.stride))
    records: list[bytes] = []
    for vertex_id in vertex_ids:
        offset = int(vertex_id) * int(slot.header.stride) + int(element.aligned_byte_offset)
        end_offset = offset + size
        if end_offset > len(data):
            raise ValueError(f"{slot.slot_name}: record read exceeds loaded slice")
        records.append(data[offset:end_offset])
    return records


def _format_size(fmt: str) -> int:
    upper_fmt = str(fmt or "").upper()
    if upper_fmt in {"R32_FLOAT", "DXGI_FORMAT_R32_FLOAT"}:
        return 4
    if upper_fmt in {"R32G32_FLOAT", "DXGI_FORMAT_R32G32_FLOAT"}:
        return 8
    if upper_fmt in {"R32G32B32_FLOAT", "DXGI_FORMAT_R32G32B32_FLOAT"}:
        return 12
    if upper_fmt in {"R32G32B32A32_FLOAT", "DXGI_FORMAT_R32G32B32A32_FLOAT"}:
        return 16
    if upper_fmt in {"R32G32B32A32_UINT", "DXGI_FORMAT_R32G32B32A32_UINT"}:
        return 16
    if upper_fmt in {"R16G16B16A16_UNORM", "DXGI_FORMAT_R16G16B16A16_UNORM"}:
        return 8
    if upper_fmt in {"R8G8B8A8_UINT", "DXGI_FORMAT_R8G8B8A8_UINT"}:
        return 4
    if upper_fmt in {"R8G8B8A8_SNORM", "DXGI_FORMAT_R8G8B8A8_SNORM"}:
        return 4
    if upper_fmt in {"R8G8B8A8_UNORM", "DXGI_FORMAT_R8G8B8A8_UNORM"}:
        return 4
    raise ValueError(f"Unsupported vertex format: {fmt}")


def _is_format(fmt: str, name: str) -> bool:
    upper_fmt = str(fmt or "").upper()
    upper_name = str(name or "").upper()
    return upper_fmt == upper_name or upper_fmt == f"DXGI_FORMAT_{upper_name}"


def _unpack_format(data: bytes, offset: int, fmt: str) -> tuple[float, ...]:
    upper_fmt = str(fmt or "").upper()
    if upper_fmt in {"R32_FLOAT", "DXGI_FORMAT_R32_FLOAT"}:
        return (float(struct.unpack_from("<f", data, offset)[0]),)
    if upper_fmt in {"R32G32_FLOAT", "DXGI_FORMAT_R32G32_FLOAT"}:
        return tuple(float(value) for value in struct.unpack_from("<2f", data, offset))
    if upper_fmt in {"R32G32B32_FLOAT", "DXGI_FORMAT_R32G32B32_FLOAT"}:
        return tuple(float(value) for value in struct.unpack_from("<3f", data, offset))
    if upper_fmt in {"R32G32B32A32_FLOAT", "DXGI_FORMAT_R32G32B32A32_FLOAT"}:
        return tuple(float(value) for value in struct.unpack_from("<4f", data, offset))
    if upper_fmt in {"R32G32B32A32_UINT", "DXGI_FORMAT_R32G32B32A32_UINT"}:
        return tuple(float(value) for value in struct.unpack_from("<4I", data, offset))
    if upper_fmt in {"R16G16B16A16_UNORM", "DXGI_FORMAT_R16G16B16A16_UNORM"}:
        return tuple(float(value) / 65535.0 for value in struct.unpack_from("<4H", data, offset))
    if upper_fmt in {"R8G8B8A8_UINT", "DXGI_FORMAT_R8G8B8A8_UINT"}:
        return tuple(float(value) for value in struct.unpack_from("<4B", data, offset))
    if upper_fmt in {"R8G8B8A8_SNORM", "DXGI_FORMAT_R8G8B8A8_SNORM"}:
        return tuple(float(_signed_int8(value)) / 127.0 for value in struct.unpack_from("<4B", data, offset))
    if upper_fmt in {"R8G8B8A8_UNORM", "DXGI_FORMAT_R8G8B8A8_UNORM"}:
        return tuple(float(value) / 255.0 for value in struct.unpack_from("<4B", data, offset))
    raise ValueError(f"Unsupported vertex format: {fmt}")


def _signed_int8(value: int) -> int:
    return int(value) - 256 if int(value) >= 128 else int(value)


def _signed_10(value: int) -> int:
    masked = int(value) & 0x3FF
    return masked - 1024 if masked >= 512 else masked


def _normalize_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _apply_uv_layer(mesh, layer_name: str, values: list[tuple[float, float]]) -> None:
    uv_layer = mesh.uv_layers.new(name=layer_name)
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            if vertex_index < len(values):
                uv_layer.data[loop_index].uv = values[vertex_index]
    mesh.uv_layers.active = mesh.uv_layers.get("UV0")


def _apply_custom_normals(mesh, normals: list[tuple[float, float, float]]) -> None:
    if len(normals) != len(mesh.vertices):
        return
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    mesh.normals_split_custom_set_from_vertices(normals)


def _store_int_attribute(mesh, name: str, values: list[int]) -> None:
    attribute = mesh.attributes.new(name=name, type="INT", domain="POINT")
    for item, value in zip(attribute.data, values):
        item.value = int(value)


def _store_float_attribute(mesh, name: str, values: list[float]) -> None:
    attribute = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
    for item, value in zip(attribute.data, values):
        item.value = float(value)


def _store_uint32_split_attributes(mesh, base_name: str, values: list[int]) -> None:
    _store_int_attribute(mesh, f"{base_name}_lo16", [int(value) & 0xFFFF for value in values])
    _store_int_attribute(mesh, f"{base_name}_hi16", [(int(value) >> 16) & 0xFFFF for value in values])


def _assign_vertex_groups(
    imported_object,
    blend_indices: list[tuple[int, int, int, int]],
    blend_weights: list[tuple[float, float, float, float]],
) -> None:
    vertex_groups = {}
    for vertex_index, (index_record, weight_record) in enumerate(zip(blend_indices, blend_weights)):
        for palette_index, weight in zip(index_record, weight_record):
            if weight <= 0.0:
                continue
            group = vertex_groups.get(int(palette_index))
            if group is None:
                group = imported_object.vertex_groups.new(name=str(int(palette_index)))
                vertex_groups[int(palette_index)] = group
            group.add([vertex_index], float(weight), "ADD")
