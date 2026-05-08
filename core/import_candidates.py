"""Import redesigned candidate IB slices from FrameAnalysis buffers."""

from __future__ import annotations

import json
import math
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Iterable

from .main_analyze import BufferHeader, HeaderElement, _parse_buffer_header
from .texcoord_attrs import snorm_byte_to_color_component, texcoord_color_attr_names
from .uv_transform import DEFAULT_UV_FLIP_V, game_uv_to_blender
from .vertex_format import format_size as _shared_format_size, unpack_vertex_format


_VERTEX_DATA_RE = re.compile(
    r"^vb(?P<slot>\d+)\[(?P<vertex>\d+)\]\+(?P<offset>\d+)\s+[^:]+:\s*(?P<values>.+)$"
)
DEFAULT_MIRROR_FLIP = True


@dataclass(frozen=True)
class LoadedCandidateGeometry:
    display_name: str
    ib_hash: str
    match_first_index: int
    match_index_count: int
    positions: list[tuple[float, float, float]]
    triangles: list[tuple[int, int, int]]
    original_vertex_ids: list[int]
    uv0: list[tuple[float, float]] | None
    uv1: list[tuple[float, float]] | None
    normals: list[tuple[float, float, float]]
    normal_packed: list[int]
    texcoord4_raw: list[tuple[int, int, int, int]]
    texcoord_semantics: list[dict]
    vertex_layout: dict
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
    data: bytes


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

    warnings: list[str] = []
    vb_payload = dict(import_paths.get("vb", {}) or {})
    slot_slices: dict[str, _SlotSlice | None] = {}

    def get_slot(slot_name: str, slot_index: int) -> _SlotSlice | None:
        if slot_name not in slot_slices:
            slot_slices[slot_name] = _load_slot_slice(vb_payload, slot_name, slot_index, frameanalysis_dir, warnings)
        return slot_slices[slot_name]

    vb0 = get_slot("vb0", 0)
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

    vb1 = get_slot("vb1", 1)
    if vb1 is not None:
        uv0 = _read_float2_if_present(vb1, "TEXCOORD", 0, original_vertex_ids)
        uv1 = _read_float2_if_present(vb1, "TEXCOORD", 1, original_vertex_ids)
    else:
        warnings.append("vb1 is missing; no UV layers are imported from vb1")
        uv0 = None
        uv1 = None

    skin_slot_name, skin_slot_index = _skin_slot_from_candidate(candidate)
    skin_slot = get_slot(skin_slot_name, skin_slot_index)
    if skin_slot is not None:
        _validate_skin_format(skin_slot, dict(candidate.get("skin_format", {}) or {}), warnings)
        blend_weights = _read_blend_weights(skin_slot, original_vertex_ids)
        blend_indices = _read_blend_indices(skin_slot, original_vertex_ids)
        if not _blend_indices_are_valid(blend_indices):
            warnings.append(f"{skin_slot.slot_name}: BLENDINDICES0 contains values outside 0..255; weights skipped")
            blend_weights = [(0.0, 0.0, 0.0, 0.0) for _ in original_vertex_ids]
            blend_indices = [(0, 0, 0, 0) for _ in original_vertex_ids]
    else:
        warnings.append(f"{skin_slot_name} is missing; blend weights and indices use fallback values")
        blend_weights = [(0.0, 0.0, 0.0, 0.0) for _ in original_vertex_ids]
        blend_indices = [(0, 0, 0, 0) for _ in original_vertex_ids]

    for slot_name in sorted(vb_payload, key=_slot_sort_key):
        if re.match(r"^vb\d+$", str(slot_name)):
            get_slot(str(slot_name), int(str(slot_name)[2:]))
    loaded_slots = [slot for slot in slot_slices.values() if slot is not None]
    texcoord_semantics = _read_texcoord_semantics(loaded_slots, original_vertex_ids)
    texcoord4_raw = _first_raw_snorm_texcoord4(texcoord_semantics, len(original_vertex_ids))
    vertex_layout = _build_loaded_vertex_layout(loaded_slots)

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
        texcoord_semantics=texcoord_semantics,
        vertex_layout=vertex_layout,
        blend_indices=blend_indices,
        blend_weights=blend_weights,
        warnings=warnings,
    )


def import_selected_candidates(
    context,
    manifest: dict,
    selected_display_names: Iterable[str],
    target_collection,
    *,
    mirror_flip: bool = DEFAULT_MIRROR_FLIP,
    uv_flip_v: bool = DEFAULT_UV_FLIP_V,
):
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
            mirror_flip=mirror_flip,
            uv_flip_v=uv_flip_v,
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
    mirror_flip: bool = DEFAULT_MIRROR_FLIP,
    uv_flip_v: bool = DEFAULT_UV_FLIP_V,
):
    mesh = bpy_module.data.meshes.new(geometry.display_name)
    imported_object = bpy_module.data.objects.new(geometry.display_name, mesh)
    target_collection.objects.link(imported_object)

    blender_positions = _positions_for_blender(geometry.positions, mirror_flip=mirror_flip)
    blender_normals = _normals_for_blender(geometry.normals, mirror_flip=mirror_flip)
    blender_triangles = _triangles_for_blender(geometry.triangles, mirror_flip=mirror_flip)
    mesh.from_pydata(blender_positions, [], blender_triangles)
    mesh.validate(verbose=False, clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    if geometry.uv0 is not None:
        _apply_uv_layer(mesh, "UV0", geometry.uv0, uv_flip_v=uv_flip_v)
    if geometry.uv1 is not None:
        _apply_uv_layer(mesh, "UV1", geometry.uv1, uv_flip_v=uv_flip_v)
    _apply_custom_normals(mesh, blender_normals)
    _store_int_attribute(mesh, "bmc_orig_vertex_id", geometry.original_vertex_ids)
    _store_uint32_split_attributes(mesh, "bmc_normal_packed", geometry.normal_packed)
    _store_texcoord_semantic_attributes(mesh, geometry.texcoord_semantics)
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
    imported_object["bmc_mirror_flip"] = bool(mirror_flip)
    imported_object["modimp_mirror_flip"] = bool(mirror_flip)
    imported_object["bmc_uv_flip_v"] = bool(uv_flip_v)
    imported_object["bmc_uv0_present"] = geometry.uv0 is not None
    imported_object["bmc_uv1_present"] = geometry.uv1 is not None
    imported_object["bmc_vertex_layout_json"] = json.dumps(
        geometry.vertex_layout,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    imported_object["bmc_vertex_semantics_json"] = json.dumps(
        [_semantic_metadata(record) for record in geometry.texcoord_semantics],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if geometry.warnings:
        imported_object["bmc_import_warnings"] = "\n".join(geometry.warnings)
    return imported_object


def _mirror_x_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-float(vector[0]), float(vector[1]), float(vector[2]))


def _reverse_triangle_winding(triangle: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(triangle[0]), int(triangle[2]), int(triangle[1]))


def _positions_for_blender(
    positions: list[tuple[float, float, float]],
    *,
    mirror_flip: bool,
) -> list[tuple[float, float, float]]:
    if not mirror_flip:
        return list(positions)
    return [_mirror_x_vector(position) for position in positions]


def _normals_for_blender(
    normals: list[tuple[float, float, float]],
    *,
    mirror_flip: bool,
) -> list[tuple[float, float, float]]:
    if not mirror_flip:
        return list(normals)
    return [_mirror_x_vector(normal) for normal in normals]


def _triangles_for_blender(
    triangles: list[tuple[int, int, int]],
    *,
    mirror_flip: bool,
) -> list[tuple[int, int, int]]:
    # Direct3D and Blender disagree on the visible winding for these dumps.
    # Mirror Flip only changes coordinate handedness; Blender mesh faces still
    # need the source IB order reversed to display the same front side.
    return [_reverse_triangle_winding(triangle) for triangle in triangles]


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


def _skin_slot_from_candidate(candidate: dict) -> tuple[str, int]:
    skin_format = dict(candidate.get("skin_format", {}) or {})
    slot_name = str(skin_format.get("slot", "") or "vb2").lower()
    if not re.match(r"^vb\d+$", slot_name):
        slot_name = "vb2"
    return slot_name, int(slot_name[2:])


def _slot_sort_key(slot_name: str) -> tuple[int, str]:
    match = re.match(r"^vb(?P<index>\d+)$", str(slot_name))
    if not match:
        return 9999, str(slot_name)
    return int(match.group("index")), str(slot_name)


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
    return [int(record[0]) for record in struct.iter_unpack(unpack_format, data)]


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
    base_offset = _resolve_slot_base_offset(buf_path, header)
    slice_size = int(header.vertex_count) * int(header.stride)
    with open(buf_path, "rb") as file_handle:
        file_handle.seek(int(base_offset))
        data = file_handle.read(slice_size)
    if len(data) < slice_size:
        warnings.append(f"{slot_name}: buf slice is shorter than vertex_count*stride")
        return None
    return _SlotSlice(
        slot_name=slot_name,
        slot_index=slot_index,
        buf_path=buf_path,
        header=header,
        elements=elements,
        base_offset=base_offset,
        data=data,
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


def _validate_skin_format(slot: _SlotSlice, skin_format: dict, warnings: list[str]) -> None:
    if not skin_format:
        return
    expected_slot = str(skin_format.get("slot", "") or "").lower()
    if expected_slot and expected_slot != slot.slot_name:
        warnings.append(f"{slot.slot_name}: skin slot differs from manifest {expected_slot}")

    expectations = (
        ("BLENDWEIGHTS", 0, "blend_weights_format", "blend_weights_offset"),
        ("BLENDINDICES", 0, "blend_indices_format", "blend_indices_offset"),
    )
    for semantic, semantic_index, format_key, offset_key in expectations:
        expected_format = str(skin_format.get(format_key, "") or "").upper()
        expected_offset = int(skin_format.get(offset_key, -1) or -1)
        if not expected_format and expected_offset < 0:
            continue
        element = _find_element(slot, semantic, semantic_index)
        if element is None:
            warnings.append(f"{slot.slot_name}: {semantic}{semantic_index} missing for manifest skin format")
            continue
        actual_format = str(element.fmt or "").upper()
        if expected_format and actual_format != expected_format:
            warnings.append(
                f"{slot.slot_name}: {semantic}{semantic_index} format {actual_format} differs from manifest {expected_format}"
            )
        if expected_offset >= 0 and int(element.aligned_byte_offset) != expected_offset:
            warnings.append(
                f"{slot.slot_name}: {semantic}{semantic_index} offset {element.aligned_byte_offset} differs from manifest {expected_offset}"
            )


def _resolve_slot_base_offset(buf_path: str, header: BufferHeader) -> int:
    header_offset = max(0, int(header.byte_offset))
    required_size = header_offset + int(header.vertex_count) * int(header.stride)
    try:
        file_size = os.path.getsize(buf_path)
    except OSError:
        return header_offset
    if required_size <= file_size:
        return header_offset
    slice_size = int(header.vertex_count) * int(header.stride)
    if slice_size <= file_size:
        return 0
    return header_offset


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


def _read_float2_if_present(
    slot: _SlotSlice,
    semantic_name: str,
    semantic_index: int,
    vertex_ids: list[int],
) -> list[tuple[float, float]] | None:
    element = _find_element(slot, semantic_name, semantic_index)
    if element is None:
        return None
    if not _is_format(element.fmt, "R32G32_FLOAT"):
        return None
    values = _read_format_records(slot, element, vertex_ids)
    return [(float(record[0]), float(record[1])) for record in values]


def _read_texcoord_semantics(slots: list[_SlotSlice], vertex_ids: list[int]) -> list[dict]:
    records: list[dict] = []
    for slot in sorted(slots, key=lambda item: item.slot_index):
        for key in sorted(slot.elements, key=lambda item: (item[0], item[1])):
            element = slot.elements[key]
            if str(element.semantic_name).upper() != "TEXCOORD":
                continue
            records.append(_read_semantic_payload(slot, element, vertex_ids))
    return records


def _read_semantic_payload(slot: _SlotSlice, element: HeaderElement, vertex_ids: list[int]) -> dict:
    fmt = str(element.fmt or "").upper()
    if _is_format(fmt, "R8G8B8A8_SNORM"):
        values = _read_int8x4_records(slot, element, vertex_ids)
        storage = "sint8_raw"
    else:
        values = _read_format_records(slot, element, vertex_ids)
        storage = "float"
    component_count = len(values[0]) if values else max(1, _format_size(fmt) // 4)
    return {
        "slot_name": slot.slot_name,
        "slot_index": int(slot.slot_index),
        "semantic_name": str(element.semantic_name).upper(),
        "semantic_index": int(element.semantic_index),
        "format": fmt,
        "aligned_byte_offset": int(element.aligned_byte_offset),
        "component_count": int(component_count),
        "storage": storage,
        "values": [tuple(value) for value in values],
    }


def _first_raw_snorm_texcoord4(texcoord_semantics: list[dict], vertex_count: int) -> list[tuple[int, int, int, int]]:
    for record in texcoord_semantics:
        if (
            str(record.get("semantic_name", "")).upper() == "TEXCOORD"
            and int(record.get("semantic_index", -1)) == 4
            and str(record.get("storage", "")) == "sint8_raw"
            and int(record.get("component_count", 0)) == 4
        ):
            return [tuple(int(component) for component in value) for value in record.get("values", [])]
    return [(0, 0, 0, 0) for _ in range(vertex_count)]


def _build_loaded_vertex_layout(slots: list[_SlotSlice]) -> dict:
    return {
        "buffers": {
            slot.slot_name: {
                "slot": slot.slot_name,
                "slot_index": int(slot.slot_index),
                "stride": int(slot.header.stride),
                "vertex_count": int(slot.header.vertex_count),
                "byte_offset": int(slot.header.byte_offset),
                "base_offset": int(slot.base_offset),
                "elements": [
                    _element_layout_payload(element)
                    for _key, element in sorted(slot.elements.items(), key=lambda item: (item[1].aligned_byte_offset, item[1].semantic_name, item[1].semantic_index))
                ],
            }
            for slot in sorted(slots, key=lambda item: item.slot_index)
        }
    }


def _element_layout_payload(element: HeaderElement) -> dict:
    return {
        "semantic_name": str(element.semantic_name).upper(),
        "semantic_index": int(element.semantic_index),
        "format": str(element.fmt or "").upper(),
        "input_slot": int(element.input_slot),
        "aligned_byte_offset": int(element.aligned_byte_offset),
    }


def _read_uint32_records(slot: _SlotSlice, element: HeaderElement, vertex_ids: list[int]) -> list[int]:
    _validate_vertex_record_range(slot, element, vertex_ids, 4)
    stride = int(slot.header.stride)
    element_offset = int(element.aligned_byte_offset)
    data = slot.data
    return [int(struct.unpack_from("<I", data, int(vertex_id) * stride + element_offset)[0]) for vertex_id in vertex_ids]


def _read_int8x4_records(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
) -> list[tuple[int, int, int, int]]:
    _validate_vertex_record_range(slot, element, vertex_ids, 4)
    stride = int(slot.header.stride)
    element_offset = int(element.aligned_byte_offset)
    data = slot.data
    return [
        tuple(_signed_int8(byte_value) for byte_value in data[int(vertex_id) * stride + element_offset:int(vertex_id) * stride + element_offset + 4])
        for vertex_id in vertex_ids
    ]


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
    size = _format_size(element.fmt)
    _validate_vertex_record_range(slot, element, vertex_ids, size)
    stride = int(slot.header.stride)
    element_offset = int(element.aligned_byte_offset)
    data = slot.data
    return [
        _unpack_format(data, int(vertex_id) * stride + element_offset, element.fmt)
        for vertex_id in vertex_ids
    ]


def _read_raw_records(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
    size: int,
) -> list[bytes]:
    _validate_vertex_record_range(slot, element, vertex_ids, size)
    data = slot.data
    records: list[bytes] = []
    for vertex_id in vertex_ids:
        offset = int(vertex_id) * int(slot.header.stride) + int(element.aligned_byte_offset)
        end_offset = offset + size
        if end_offset > len(data):
            raise ValueError(f"{slot.slot_name}: record read exceeds loaded slice")
        records.append(data[offset:end_offset])
    return records


def _validate_vertex_record_range(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
    size: int,
) -> None:
    if not vertex_ids:
        return
    max_vertex_id = max(vertex_ids)
    if max_vertex_id >= int(slot.header.vertex_count):
        raise ValueError(
            f"{slot.slot_name}: vertex id {max_vertex_id} exceeds vertex_count {slot.header.vertex_count}"
        )
    max_offset = int(max_vertex_id) * int(slot.header.stride) + int(element.aligned_byte_offset) + int(size)
    if max_offset > len(slot.data):
        raise ValueError(f"{slot.slot_name}: record read exceeds loaded slice")


def _format_size(fmt: str) -> int:
    return _shared_format_size(fmt)


def _is_format(fmt: str, name: str) -> bool:
    upper_fmt = str(fmt or "").upper()
    upper_name = str(name or "").upper()
    return upper_fmt == upper_name or upper_fmt == f"DXGI_FORMAT_{upper_name}"


def _unpack_format(data: bytes, offset: int, fmt: str) -> tuple[float, ...]:
    return unpack_vertex_format(fmt, data, offset)


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


def _apply_uv_layer(mesh, layer_name: str, values: list[tuple[float, float]], *, uv_flip_v: bool = DEFAULT_UV_FLIP_V) -> None:
    uv_layer = mesh.uv_layers.new(name=layer_name)
    flat_uvs: list[float] = []
    for loop in mesh.loops:
        vertex_index = int(loop.vertex_index)
        uv = values[vertex_index] if vertex_index < len(values) else (0.0, 0.0)
        uv = game_uv_to_blender(uv, flip_v=uv_flip_v)
        flat_uvs.extend([float(uv[0]), float(uv[1])])
    if _foreach_set(uv_layer.data, "uv", flat_uvs):
        mesh.uv_layers.active = mesh.uv_layers.get("UV0")
        return
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            if vertex_index < len(values):
                uv_layer.data[loop_index].uv = game_uv_to_blender(values[vertex_index], flip_v=uv_flip_v)
    mesh.uv_layers.active = mesh.uv_layers.get("UV0")


def _apply_custom_normals(mesh, normals: list[tuple[float, float, float]]) -> None:
    if len(normals) != len(mesh.vertices):
        return
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    mesh.normals_split_custom_set_from_vertices(normals)


def _store_int_attribute(mesh, name: str, values: list[int]) -> None:
    attribute = mesh.attributes.new(name=name, type="INT", domain="POINT")
    if _foreach_set(attribute.data, "value", [int(value) for value in values]):
        return
    for item, value in zip(attribute.data, values):
        item.value = int(value)


def _store_float_attribute(mesh, name: str, values: list[float]) -> None:
    attribute = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
    if _foreach_set(attribute.data, "value", [float(value) for value in values]):
        return
    for item, value in zip(attribute.data, values):
        item.value = float(value)


def _foreach_set(data, attribute_name: str, values: list[float | int]) -> bool:
    setter = getattr(data, "foreach_set", None)
    if not callable(setter):
        return False
    try:
        setter(attribute_name, values)
        return True
    except Exception:
        return False


def _store_uint32_split_attributes(mesh, base_name: str, values: list[int]) -> None:
    _store_int_attribute(mesh, f"{base_name}_lo16", [int(value) & 0xFFFF for value in values])
    _store_int_attribute(mesh, f"{base_name}_hi16", [(int(value) >> 16) & 0xFFFF for value in values])


def _store_texcoord_semantic_attributes(mesh, records: list[dict]) -> None:
    for record in records:
        slot_index = int(record.get("slot_index", -1))
        semantic_index = int(record.get("semantic_index", -1))
        values = list(record.get("values", []) or [])
        component_count = int(record.get("component_count", 0) or 0)
        storage = str(record.get("storage", "") or "")
        if slot_index < 0 or semantic_index < 0 or component_count <= 0:
            continue
        base_name = f"bmc_vb{slot_index}_texcoord{semantic_index}"
        for component_index in range(component_count):
            component_values = [
                value[component_index] if component_index < len(value) else 0
                for value in values
            ]
            attr_name = f"{base_name}_{component_index}"
            if storage == "sint8_raw":
                _store_int_attribute(mesh, attr_name, [int(value) for value in component_values])
            else:
                _store_float_attribute(mesh, attr_name, [float(value) for value in component_values])
        if storage == "sint8_raw" and component_count == 4:
            _store_snorm_byte_color_attribute(
                mesh,
                texcoord_color_attr_names(f"vb{slot_index}", semantic_index)[0],
                values,
            )


def _store_snorm_byte_color_attribute(mesh, name: str, values: list[tuple[int, ...]]) -> None:
    color_attributes = getattr(mesh, "color_attributes", None)
    if color_attributes is None:
        return
    getter = getattr(color_attributes, "get", None)
    attribute = getter(name) if callable(getter) else None
    if attribute is None:
        creator = getattr(color_attributes, "new", None)
        if not callable(creator):
            return
        try:
            attribute = creator(name=name, type="BYTE_COLOR", domain="POINT")
        except Exception:
            try:
                attribute = creator(name=name, type="FLOAT_COLOR", domain="POINT")
            except Exception:
                return
    colors: list[float] = []
    for record in values:
        components = [int(record[index]) if index < len(record) else 0 for index in range(4)]
        colors.extend(snorm_byte_to_color_component(component) for component in components)
    data = getattr(attribute, "data", [])
    if _foreach_set(data, "color", colors):
        return
    for item, offset in zip(data, range(0, len(colors), 4)):
        try:
            item.color = tuple(colors[offset:offset + 4])
        except Exception:
            continue


def _semantic_metadata(record: dict) -> dict:
    return {
        "slot_name": str(record.get("slot_name", "") or ""),
        "slot_index": int(record.get("slot_index", -1)),
        "semantic_name": str(record.get("semantic_name", "") or ""),
        "semantic_index": int(record.get("semantic_index", -1)),
        "format": str(record.get("format", "") or ""),
        "aligned_byte_offset": int(record.get("aligned_byte_offset", -1)),
        "component_count": int(record.get("component_count", 0) or 0),
        "storage": str(record.get("storage", "") or ""),
    }


def _assign_vertex_groups(
    imported_object,
    blend_indices: list[tuple[int, int, int, int]],
    blend_weights: list[tuple[float, float, float, float]],
) -> None:
    assignments: dict[int, dict[float, list[int]]] = {}
    for vertex_index, (index_record, weight_record) in enumerate(zip(blend_indices, blend_weights)):
        for palette_index, weight in zip(index_record, weight_record):
            if weight <= 0.0:
                continue
            assignments.setdefault(int(palette_index), {}).setdefault(float(weight), []).append(vertex_index)

    for palette_index, weights_to_vertices in sorted(assignments.items()):
        group = imported_object.vertex_groups.new(name=str(int(palette_index)))
        for weight, vertex_indices in weights_to_vertices.items():
            group.add(vertex_indices, float(weight), "ADD")
