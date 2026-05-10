"""Import redesigned candidate IB slices from FrameAnalysis buffers."""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

from .draw_arrays import build_topology_arrays, read_index_array, require_numpy, row_tuple_list
from .main_analyze import BufferHeader, HeaderElement, _parse_buffer_header
from .numpy_buffers import read_interleaved_field, read_interleaved_fields
from .texcoord_attrs import texcoord_color_attr_names
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
    tangents: list[tuple[float, float, float]]
    bitangent_signs: list[float]
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


class _RowTupleArray:
    """NumPy-backed rows that keep the old tuple-like indexing contract."""

    def __init__(self, values) -> None:
        self._values = values

    @property
    def shape(self):
        return self._values.shape

    def __len__(self) -> int:
        return len(self._values)

    def __array__(self, dtype=None):
        np = require_numpy()
        return np.asarray(self._values, dtype=dtype) if dtype is not None else np.asarray(self._values)

    def __getitem__(self, index):
        value = self._values[index]
        if isinstance(index, slice):
            return _RowTupleArray(value)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return tuple(value.tolist())
        try:
            return value.item()
        except Exception:
            return value

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def _row_tuple_array(values):
    np = require_numpy()
    if values is None:
        return values
    array = np.asarray(values)
    if array.ndim >= 2:
        return _RowTupleArray(array)
    return values


def load_candidate_geometry(
    candidate: dict,
    frameanalysis_dir: str = "",
    *,
    performance: dict | None = None,
) -> LoadedCandidateGeometry:
    """Load one candidate from .buf slices into a compact CPU geometry payload."""

    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    import_paths = dict(candidate.get("import_paths", {}) or {})
    ib_txt_path = _resolve_path(str(import_paths.get("ib", "") or ""), frameanalysis_dir)
    ib_buf_path = _resolve_path(str(import_paths.get("ib_buf", "") or ""), frameanalysis_dir)
    if not ib_txt_path or not os.path.exists(ib_txt_path):
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no IB txt path")
    if not ib_buf_path or not os.path.exists(ib_buf_path):
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no IB buf path")

    stage_start = time.perf_counter()
    ib_header = _parse_buffer_header(ib_txt_path)
    indices = read_index_array(
        ib_buf_path,
        str(ib_header.fmt or ""),
        int(ib_header.index_count),
        byte_offset=int(ib_header.byte_offset),
        first_index=int(ib_header.first_index),
    )
    if int(indices.size) < 3:
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no triangles")
    timings["ib"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    triangle_array, original_vertex_id_array, _source_triangles = build_topology_arrays(indices)
    original_vertex_ids = [int(value) for value in original_vertex_id_array.tolist()]
    triangles = row_tuple_list(triangle_array, dtype="int64")
    timings["topology"] = time.perf_counter() - stage_start

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
    stage_start = time.perf_counter()
    positions, normals, normal_packed, tangents, bitangent_signs = _read_vb0_position_normal(vb0, original_vertex_ids, warnings)
    vb0_seconds = time.perf_counter() - stage_start
    timings["positions"] = vb0_seconds * 0.5
    timings["normals"] = vb0_seconds * 0.5

    stage_start = time.perf_counter()
    vb1 = get_slot("vb1", 1)
    if vb1 is not None:
        uv0, uv1 = _read_vb1_uv_layers(vb1, original_vertex_ids)
    else:
        warnings.append("vb1 is missing; no UV layers are imported from vb1")
        uv0 = None
        uv1 = None
    timings["uv"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    skin_slot_name, skin_slot_index = _skin_slot_from_candidate(candidate)
    skin_slot = get_slot(skin_slot_name, skin_slot_index)
    if skin_slot is not None:
        _validate_skin_format(skin_slot, dict(candidate.get("skin_format", {}) or {}), warnings)
        blend_weights, blend_indices = _read_skin_weights_indices(skin_slot, original_vertex_ids)
        if not _blend_indices_are_valid(blend_indices):
            warnings.append(f"{skin_slot.slot_name}: BLENDINDICES0 contains values outside 0..255; weights skipped")
            blend_weights = [(0.0, 0.0, 0.0, 0.0) for _ in original_vertex_ids]
            blend_indices = [(0, 0, 0, 0) for _ in original_vertex_ids]
    else:
        warnings.append(f"{skin_slot_name} is missing; blend weights and indices use fallback values")
        blend_weights = [(0.0, 0.0, 0.0, 0.0) for _ in original_vertex_ids]
        blend_indices = [(0, 0, 0, 0) for _ in original_vertex_ids]
    timings["skin"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    for slot_name in sorted(vb_payload, key=_slot_sort_key):
        if re.match(r"^vb\d+$", str(slot_name)):
            get_slot(str(slot_name), int(str(slot_name)[2:]))
    loaded_slots = [slot for slot in slot_slices.values() if slot is not None]
    texcoord_semantics = _read_texcoord_semantics(loaded_slots, original_vertex_ids)
    texcoord4_raw = _first_raw_snorm_texcoord4(texcoord_semantics, len(original_vertex_ids))
    timings["texcoord_semantics"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    vertex_layout = _build_loaded_vertex_layout(loaded_slots)
    timings["layout"] = time.perf_counter() - stage_start
    timings["total"] = time.perf_counter() - total_start

    geometry = LoadedCandidateGeometry(
        display_name=str(candidate.get("display_name", "") or _candidate_display_name(candidate)),
        ib_hash=str(candidate.get("ib_hash", "") or "").lower(),
        match_first_index=int(candidate.get("match_first_index", 0) or 0),
        match_index_count=int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0),
        positions=_row_tuple_array(positions),
        triangles=triangles,
        original_vertex_ids=original_vertex_ids,
        uv0=_row_tuple_array(uv0),
        uv1=_row_tuple_array(uv1),
        normals=_row_tuple_array(normals),
        normal_packed=normal_packed,
        tangents=_row_tuple_array(tangents),
        bitangent_signs=bitangent_signs,
        texcoord4_raw=texcoord4_raw,
        texcoord_semantics=texcoord_semantics,
        vertex_layout=vertex_layout,
        blend_indices=_row_tuple_array(blend_indices),
        blend_weights=_row_tuple_array(blend_weights),
        warnings=warnings,
    )
    if performance is not None:
        performance.clear()
        performance.update(timings)
    return geometry


def import_selected_candidates(
    context,
    manifest: dict,
    selected_display_names: Iterable[str],
    target_collection,
    *,
    mirror_flip: bool = DEFAULT_MIRROR_FLIP,
    uv_flip_v: bool = DEFAULT_UV_FLIP_V,
    performance: dict | None = None,
):
    """Create Blender mesh objects for selected candidate display names."""

    import bpy  # type: ignore

    total_start = time.perf_counter()
    object_reports: list[dict] = []
    selected_names = {str(value) for value in selected_display_names if str(value)}
    if not selected_names:
        return []
    frameanalysis_dir = str(manifest.get("frameanalysis_dir", "") or "")
    imported_objects = []
    for candidate in manifest.get("candidate_ibs", []) or []:
        display_name = str(candidate.get("display_name", "") or _candidate_display_name(candidate))
        if display_name not in selected_names:
            continue
        object_start = time.perf_counter()
        stage_start = time.perf_counter()
        load_report: dict[str, float] = {}
        geometry = load_candidate_geometry(candidate, frameanalysis_dir, performance=load_report)
        load_seconds = time.perf_counter() - stage_start
        create_report: dict[str, float] = {}
        stage_start = time.perf_counter()
        imported_object = create_blender_object_from_geometry(
            bpy,
            geometry,
            target_collection,
            draw_indices=list(candidate.get("draw_indices", []) or []),
            shadow_draw_indices=list(candidate.get("shadow_draw_indices", []) or []),
            mirror_flip=mirror_flip,
            uv_flip_v=uv_flip_v,
            performance=create_report,
        )
        create_seconds = time.perf_counter() - stage_start
        imported_objects.append(imported_object)
        object_reports.append(
            {
                "name": display_name,
                "vertices": len(geometry.positions),
                "triangles": len(geometry.triangles),
                "load": load_seconds,
                "create": create_seconds,
                "total": time.perf_counter() - object_start,
                "load_stages": load_report,
                "create_stages": create_report,
            }
        )

    if imported_objects:
        for selected_object in context.selected_objects:
            selected_object.select_set(False)
        for imported_object in imported_objects:
            imported_object.select_set(True)
        context.view_layer.objects.active = imported_objects[0]
    if performance is not None:
        performance.clear()
        performance.update(
            {
                "total": time.perf_counter() - total_start,
                "objects": object_reports,
            }
        )
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
    performance: dict | None = None,
):
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    stage_start = time.perf_counter()
    mesh = bpy_module.data.meshes.new(geometry.display_name)
    imported_object = bpy_module.data.objects.new(geometry.display_name, mesh)
    target_collection.objects.link(imported_object)
    timings["setup"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    blender_positions = _positions_for_blender(geometry.positions, mirror_flip=mirror_flip)
    blender_normals = _normals_for_blender(geometry.normals, mirror_flip=mirror_flip)
    blender_tangents = _normals_for_blender(geometry.tangents, mirror_flip=mirror_flip)
    blender_bitangent_signs = _bitangent_signs_for_blender(
        geometry.bitangent_signs,
        mirror_flip=mirror_flip,
        uv_flip_v=uv_flip_v,
    )
    blender_triangles = _triangles_for_blender(geometry.triangles, mirror_flip=mirror_flip)
    timings["transform"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    mesh.from_pydata(blender_positions, [], blender_triangles)
    mesh.validate(verbose=False, clean_customdata=False)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    timings["mesh"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    if geometry.uv0 is not None:
        _apply_uv_layer(mesh, "UV0", geometry.uv0, uv_flip_v=uv_flip_v)
    if geometry.uv1 is not None:
        _apply_uv_layer(mesh, "UV1", geometry.uv1, uv_flip_v=uv_flip_v)
    timings["uv"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _apply_custom_normals(mesh, blender_normals)
    timings["normals"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _store_int_attribute(mesh, "bmc_orig_vertex_id", geometry.original_vertex_ids)
    if _has_packed_tangent_frame(geometry.normal_packed):
        _store_tangent_frame_attributes(mesh, blender_tangents, blender_bitangent_signs)
    _store_texcoord_semantic_attributes(mesh, geometry.texcoord_semantics)
    for channel_index in range(4):
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
    timings["attributes"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _assign_vertex_groups(imported_object, geometry.blend_indices, geometry.blend_weights)
    timings["vertex_groups"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
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
    timings["metadata"] = time.perf_counter() - stage_start
    timings["total"] = time.perf_counter() - total_start
    if performance is not None:
        performance.clear()
        performance.update(timings)
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
    np = require_numpy()
    if hasattr(positions, "shape"):
        values = np.asarray(positions, dtype=np.float32)
        if mirror_flip and values.size:
            values = values.copy()
            values[:, 0] *= -1.0
        return values.tolist()
    if not mirror_flip:
        return list(positions)
    return [_mirror_x_vector(position) for position in positions]


def _normals_for_blender(
    normals: list[tuple[float, float, float]],
    *,
    mirror_flip: bool,
) -> list[tuple[float, float, float]]:
    np = require_numpy()
    if hasattr(normals, "shape"):
        values = np.asarray(normals, dtype=np.float32)
        if mirror_flip and values.size:
            values = values.copy()
            values[:, 0] *= -1.0
        return values.tolist()
    if not mirror_flip:
        return list(normals)
    return [_mirror_x_vector(normal) for normal in normals]


def _bitangent_signs_for_blender(
    signs: list[float],
    *,
    mirror_flip: bool,
    uv_flip_v: bool,
) -> list[float]:
    flip_sign = bool(mirror_flip) ^ bool(uv_flip_v)
    np = require_numpy()
    if hasattr(signs, "shape"):
        values = np.asarray(signs, dtype=np.float32)
        if flip_sign and values.size:
            values = -values
        return values
    if not flip_sign:
        return list(signs)
    return [-float(value) for value in signs]


def _triangles_for_blender(
    triangles: list[tuple[int, int, int]],
    *,
    mirror_flip: bool,
) -> list[tuple[int, int, int]]:
    # Direct3D and Blender disagree on the visible winding for these dumps.
    # Mirror Flip only changes coordinate handedness; Blender mesh faces still
    # need the source IB order reversed to display the same front side.
    return [_reverse_triangle_winding(triangle) for triangle in triangles]


def _decode_game_packed_normals(values):
    np = require_numpy()
    packed = np.asarray(values, dtype=np.uint32).reshape(-1)
    valid = (packed & np.uint32(0x40000000)) != 0
    raw_x = (packed & np.uint32(0x3FF)).astype(np.int32)
    raw_y = ((packed >> np.uint32(10)) & np.uint32(0x3FF)).astype(np.int32)
    signed_x = np.where(raw_x >= 512, raw_x - 1024, raw_x).astype(np.float32) / 511.0
    signed_y = np.where(raw_y >= 512, raw_y - 1024, raw_y).astype(np.float32) / 511.0
    z_values = 1.0 - np.abs(signed_x) - np.abs(signed_y)
    normal_x = signed_x.copy()
    normal_y = signed_y.copy()
    fold = z_values < 0.0
    if np.any(fold):
        sign_x = np.where(signed_x >= 0.0, 1.0, -1.0)
        sign_y = np.where(signed_y >= 0.0, 1.0, -1.0)
        normal_x = np.where(fold, (1.0 - np.abs(signed_y)) * sign_x, normal_x)
        normal_y = np.where(fold, (1.0 - np.abs(signed_x)) * sign_y, normal_y)
    normals = np.stack((normal_x, normal_y, z_values), axis=1).astype(np.float32)
    normals[~valid] = (0.0, 0.0, 1.0)
    return _normalize_vectors(normals)


def _decode_game_packed_tangent_frames(values):
    np = require_numpy()
    packed = np.asarray(values, dtype=np.uint32).reshape(-1)
    if packed.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    valid = (packed & np.uint32(0x40000000)) != 0
    normals = np.asarray(_decode_game_packed_normals(packed), dtype=np.float32)
    roll_raw = ((packed >> np.uint32(20)) & np.uint32(0x3FF)).astype(np.int32)
    roll = np.where(roll_raw >= 512, roll_raw - 1024, roll_raw).astype(np.float32) / 511.0
    basis_u, basis_v = _packed_tangent_basis_numpy(normals)
    roll_x = 1.0 - np.abs(roll)
    roll_y = np.where(roll < 0.0, -1.0, 1.0) * (1.0 - np.abs(roll_x))
    roll_length = np.sqrt((roll_x * roll_x) + (roll_y * roll_y))
    roll_length = np.where(roll_length <= 1e-12, 1.0, roll_length)
    cosine = roll_x / roll_length
    sine = roll_y / roll_length
    tangents = (basis_u * cosine[:, None]) + (basis_v * sine[:, None])
    tangents = _normalize_vectors(tangents)
    tangents = np.where(valid[:, None], tangents, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    signs = np.where((packed & np.uint32(0x80000000)) != 0, 1.0, -1.0).astype(np.float32)
    signs = np.where(valid, signs, 1.0).astype(np.float32)
    return tangents, signs


def _packed_tangent_basis_numpy(normals):
    np = require_numpy()
    normal_values = _normalize_vectors(normals)
    basis_u = np.stack(
        (
            normal_values[:, 1] - normal_values[:, 2],
            normal_values[:, 2] - normal_values[:, 0],
            normal_values[:, 0] - normal_values[:, 1],
        ),
        axis=1,
    ).astype(np.float32)
    projection = np.sum(basis_u * normal_values, axis=1, keepdims=True)
    basis_u = basis_u - projection
    lengths = np.linalg.norm(basis_u, axis=1)
    fallback = lengths <= 1e-6
    if np.any(fallback):
        basis_u = basis_u.copy()
        basis_u[fallback] = _fallback_tangents_numpy(normal_values[fallback])
    basis_u = _normalize_vectors(basis_u)
    basis_v = _normalize_vectors(np.cross(normal_values, basis_u))
    return basis_u, basis_v


def _fallback_tangents_numpy(normals):
    np = require_numpy()
    normal_values = _normalize_vectors(normals)
    axes = np.zeros_like(normal_values)
    use_x = np.abs(normal_values[:, 0]) < 0.9
    axes[use_x, 0] = 1.0
    axes[~use_x, 1] = 1.0
    tangents = np.cross(axes, normal_values)
    lengths = np.linalg.norm(tangents, axis=1)
    fallback = lengths <= 1e-6
    if np.any(fallback):
        z_axes = np.zeros_like(normal_values[fallback])
        z_axes[:, 2] = 1.0
        tangents[fallback] = np.cross(z_axes, normal_values[fallback])
    return _normalize_vectors(tangents)


def _normalize_vectors(values):
    np = require_numpy()
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.size == 0:
        return vectors.reshape((0, 3))
    lengths = np.linalg.norm(vectors[:, :3], axis=1, keepdims=True)
    lengths = np.where(lengths <= 1e-12, 1.0, lengths)
    return vectors[:, :3] / lengths


def _default_normals(vertex_ids: list[int]):
    np = require_numpy()
    values = np.zeros((len(vertex_ids), 3), dtype=np.float32)
    values[:, 2] = 1.0
    return values


def _default_tangents(vertex_ids: list[int]):
    np = require_numpy()
    values = np.zeros((len(vertex_ids), 3), dtype=np.float32)
    values[:, 0] = 1.0
    return values


def _default_bitangent_signs(vertex_ids: list[int]):
    np = require_numpy()
    return np.ones((len(vertex_ids),), dtype=np.float32)


def _has_packed_tangent_frame(values) -> bool:
    np = require_numpy()
    packed = np.asarray(values, dtype=np.uint32)
    return bool(packed.size and np.any((packed & np.uint32(0x40000000)) != 0))


def _zeros_like_vertex_ids(vertex_ids: list[int], *, dtype: str):
    np = require_numpy()
    return np.zeros((len(vertex_ids),), dtype=np.dtype(dtype))


def _default_blend_weights(vertex_ids: list[int]):
    np = require_numpy()
    return np.zeros((len(vertex_ids), 4), dtype=np.float32)


def _default_blend_indices(vertex_ids: list[int]):
    np = require_numpy()
    return np.zeros((len(vertex_ids), 4), dtype=np.uint32)


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
    values = read_index_array(
        path,
        str(header.fmt or ""),
        index_count,
        byte_offset=int(header.byte_offset),
        first_index=int(header.first_index),
    )
    return [int(value) for value in values.tolist()]


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


def _read_vb0_position_normal(
    slot: _SlotSlice,
    vertex_ids: list[int],
    warnings: list[str],
):
    position_element = _find_element(slot, "POSITION", 0)
    if position_element is None:
        raise ValueError(f"{slot.slot_name}: POSITION0 is missing")
    normal_element = _find_element(slot, "NORMAL", 0)
    fields = [
        ("positions", int(position_element.aligned_byte_offset), str(position_element.fmt), True),
    ]
    normal_mode = ""
    if normal_element is not None and _is_format(normal_element.fmt, "R32_FLOAT"):
        fields.append(("normal_packed", int(normal_element.aligned_byte_offset), "R32_UINT", False))
        normal_mode = "packed"
    elif normal_element is not None and _is_format(normal_element.fmt, "R32G32B32_FLOAT"):
        fields.append(("normals", int(normal_element.aligned_byte_offset), str(normal_element.fmt), True))
        normal_mode = "float3"
    batch = read_interleaved_fields(
        slot.data,
        vertex_ids,
        stride=int(slot.header.stride),
        fields=fields,
        vertex_count=int(slot.header.vertex_count),
    )
    if batch is None:
        raise ValueError(f"{slot.slot_name}: failed to read POSITION/NORMAL fields with numpy")
    positions = batch["positions"]
    if normal_mode == "packed":
        normal_packed = batch["normal_packed"].reshape(-1)
        normals = _decode_game_packed_normals(normal_packed)
        tangents, bitangent_signs = _decode_game_packed_tangent_frames(normal_packed)
        return positions, normals, normal_packed, tangents, bitangent_signs
    if normal_mode == "float3":
        normals = _normalize_vectors(batch["normals"])
        normal_packed = _zeros_like_vertex_ids(vertex_ids, dtype="uint32")
        tangents = _default_tangents(vertex_ids)
        bitangent_signs = _default_bitangent_signs(vertex_ids)
        return positions, normals, normal_packed, tangents, bitangent_signs
    warnings.append(f"{slot.slot_name}: NORMAL0 is missing; imported normals use +Z fallback")
    return (
        positions,
        _default_normals(vertex_ids),
        _zeros_like_vertex_ids(vertex_ids, dtype="uint32"),
        _default_tangents(vertex_ids),
        _default_bitangent_signs(vertex_ids),
    )


def _read_vb1_uv_layers(slot: _SlotSlice, vertex_ids: list[int]):
    uv0_element = _find_element(slot, "TEXCOORD", 0)
    uv1_element = _find_element(slot, "TEXCOORD", 1)
    fields = []
    if uv0_element is not None and _is_format(uv0_element.fmt, "R32G32_FLOAT"):
        fields.append(("uv0", int(uv0_element.aligned_byte_offset), str(uv0_element.fmt), True))
    if uv1_element is not None and _is_format(uv1_element.fmt, "R32G32_FLOAT"):
        fields.append(("uv1", int(uv1_element.aligned_byte_offset), str(uv1_element.fmt), True))
    if fields:
        batch = read_interleaved_fields(
            slot.data,
            vertex_ids,
            stride=int(slot.header.stride),
            fields=fields,
            vertex_count=int(slot.header.vertex_count),
        )
        if batch is None:
            raise ValueError(f"{slot.slot_name}: failed to read TEXCOORD UV fields with numpy")
        return batch.get("uv0"), batch.get("uv1")
    return None, None


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
    semantic_index = int(element.semantic_index)
    if semantic_index in {0, 1} and _is_format(fmt, "R32G32_FLOAT"):
        values = []
        storage = "uv_layer"
        component_count = 2
    elif _is_format(fmt, "R8G8B8A8_SNORM"):
        values = None
        np = require_numpy()
        raw_values = read_interleaved_field(
            slot.data,
            vertex_ids,
            stride=int(slot.header.stride),
            offset=int(element.aligned_byte_offset),
            fmt=str(element.fmt or ""),
            vertex_count=int(slot.header.vertex_count),
            converted=False,
        )
        if raw_values is not None:
            signed = raw_values.astype(np.int16)
            values = np.where(signed >= 128, signed - 256, signed)
        if values is None:
            values = _read_int8x4_records(slot, element, vertex_ids)
        storage = "sint8_raw"
        component_count = _record_component_count(values, fmt)
    else:
        values = read_interleaved_field(
            slot.data,
            vertex_ids,
            stride=int(slot.header.stride),
            offset=int(element.aligned_byte_offset),
            fmt=str(element.fmt or ""),
            vertex_count=int(slot.header.vertex_count),
            converted=True,
        )
        if values is None:
            values = _read_format_records(slot, element, vertex_ids)
        storage = "float"
        component_count = _record_component_count(values, fmt)
    return {
        "slot_name": slot.slot_name,
        "slot_index": int(slot.slot_index),
        "semantic_name": str(element.semantic_name).upper(),
        "semantic_index": semantic_index,
        "format": fmt,
        "aligned_byte_offset": int(element.aligned_byte_offset),
        "component_count": int(component_count),
        "storage": storage,
        "values": values,
    }


def _record_component_count(values, fmt: str) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    try:
        if len(values) > 0:
            return len(values[0])
    except TypeError:
        pass
    return max(1, _format_size(fmt) // 4)


def _first_raw_snorm_texcoord4(texcoord_semantics: list[dict], vertex_count: int) -> list[tuple[int, int, int, int]]:
    for record in texcoord_semantics:
        if (
            str(record.get("semantic_name", "")).upper() == "TEXCOORD"
            and int(record.get("semantic_index", -1)) == 4
            and str(record.get("storage", "")) == "sint8_raw"
            and int(record.get("component_count", 0)) == 4
        ):
            values = record.get("values", [])
            tolist = getattr(values, "tolist", None)
            if callable(tolist):
                values = tolist()
            return [tuple(int(component) for component in value) for value in values]
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


def _read_int8x4_records(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
) -> list[tuple[int, int, int, int]]:
    _validate_vertex_record_range(slot, element, vertex_ids, 4)
    numpy_values = _read_numpy_vector_records(slot, element, vertex_ids, "u1", 4)
    if numpy_values is None:
        raise ValueError(f"{slot.slot_name}: failed to read int8x4 records for {element.semantic_name}{element.semantic_index}")
    np = require_numpy()
    signed = numpy_values.astype(np.int16)
    signed = np.where(signed >= 128, signed - 256, signed)
    return [tuple(int(component) for component in row) for row in signed.tolist()]


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


def _read_skin_weights_indices(slot: _SlotSlice, vertex_ids: list[int]):
    weight_element = _find_element(slot, "BLENDWEIGHTS", 0)
    index_element = _find_element(slot, "BLENDINDICES", 0)
    fields = []
    if weight_element is not None:
        fields.append(("weights", int(weight_element.aligned_byte_offset), str(weight_element.fmt), True))
    if index_element is not None:
        fields.append(("indices", int(index_element.aligned_byte_offset), str(index_element.fmt), True))
    if fields:
        batch = read_interleaved_fields(
            slot.data,
            vertex_ids,
            stride=int(slot.header.stride),
            fields=fields,
            vertex_count=int(slot.header.vertex_count),
        )
        if batch is None:
            raise ValueError(f"{slot.slot_name}: failed to read BLEND fields with numpy")
        weights = batch.get("weights")
        indices = batch.get("indices")
        if weights is None:
            weights = _default_blend_weights(vertex_ids)
        if indices is None:
            indices = _default_blend_indices(vertex_ids)
        return weights, indices
    return _default_blend_weights(vertex_ids), _default_blend_indices(vertex_ids)


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
    np = require_numpy()
    values = np.asarray(blend_indices)
    if values.size == 0:
        return True
    return bool(int(values.min()) >= 0 and int(values.max()) <= 255)


def _read_format_records(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
) -> list[tuple[float, ...]]:
    size = _format_size(element.fmt)
    _validate_vertex_record_range(slot, element, vertex_ids, size)
    numpy_records = _read_format_records_numpy(slot, element, vertex_ids)
    if numpy_records is None:
        raise ValueError(f"{slot.slot_name}: failed to read {element.semantic_name}{element.semantic_index} with numpy")
    return numpy_records


def _read_format_records_numpy(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
) -> list[tuple[float, ...]] | None:
    values = read_interleaved_field(
        slot.data,
        vertex_ids,
        stride=int(slot.header.stride),
        offset=int(element.aligned_byte_offset),
        fmt=str(element.fmt or ""),
        vertex_count=int(slot.header.vertex_count),
        converted=True,
    )
    if values is None:
        return None
    return [tuple(float(component) for component in row) for row in values.tolist()]


def _read_numpy_scalar_records(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
    dtype_name: str,
):
    values = _read_numpy_vector_records(slot, element, vertex_ids, dtype_name, 1)
    if values is None:
        return None
    try:
        return values[:, 0]
    except Exception:
        return None


def _read_numpy_vector_records(
    slot: _SlotSlice,
    element: HeaderElement,
    vertex_ids: list[int],
    dtype_name: str,
    component_count: int,
):
    np = require_numpy()
    fmt_by_dtype = {
        ("<u4", 1): "R32_UINT",
        ("<u4", 4): "R32G32B32A32_UINT",
        ("<f4", 1): "R32_FLOAT",
        ("<f4", 2): "R32G32_FLOAT",
        ("<f4", 3): "R32G32B32_FLOAT",
        ("<f4", 4): "R32G32B32A32_FLOAT",
        ("<u2", 4): "R16G16B16A16_UNORM",
        ("u1", 4): "R8G8B8A8_UINT",
    }
    fmt = fmt_by_dtype.get((str(dtype_name), int(component_count)))
    if fmt is not None:
        values = read_interleaved_field(
            slot.data,
            vertex_ids,
            stride=int(slot.header.stride),
            offset=int(element.aligned_byte_offset),
            fmt=fmt,
            vertex_count=int(slot.header.vertex_count),
            converted=False,
        )
        if values is not None:
            return values
    field_dtype = np.dtype(dtype_name)
    record_dtype = np.dtype(
        {
            "names": ["field"],
            "formats": [(field_dtype, (int(component_count),))],
            "offsets": [int(element.aligned_byte_offset)],
            "itemsize": int(slot.header.stride),
        }
    )
    records = np.frombuffer(slot.data, dtype=record_dtype, count=int(slot.header.vertex_count))
    indices = np.asarray(vertex_ids, dtype=np.intp)
    return records["field"][indices]


def _normalize_dxgi_format(fmt: str) -> str:
    normalized = str(fmt or "").upper()
    if normalized.startswith("DXGI_FORMAT_"):
        normalized = normalized[len("DXGI_FORMAT_"):]
    return normalized


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
    if _apply_uv_layer_numpy(mesh, uv_layer, values, uv_flip_v=uv_flip_v):
        mesh.uv_layers.active = mesh.uv_layers.get("UV0")
        return
    raise ValueError(f"{layer_name}: numpy UV import path failed")


def _apply_uv_layer_numpy(mesh, uv_layer, values, *, uv_flip_v: bool = DEFAULT_UV_FLIP_V) -> bool:
    np = require_numpy()
    loops = getattr(mesh, "loops", [])
    loop_count = len(loops)
    if loop_count <= 0:
        return False
    foreach_get = getattr(loops, "foreach_get", None)
    if not callable(foreach_get):
        return False
    try:
        vertex_indices = np.empty(loop_count, dtype=np.int32)
        foreach_get("vertex_index", vertex_indices)
        uv_values = np.asarray(values, dtype=np.float32)
        if uv_values.ndim < 2 or uv_values.shape[1] < 2:
            return False
        output = np.zeros((loop_count, 2), dtype=np.float32)
        valid = (vertex_indices >= 0) & (vertex_indices < len(uv_values))
        if np.any(valid):
            output[valid] = uv_values[vertex_indices[valid], :2]
        if uv_flip_v:
            output[:, 1] = 1.0 - output[:, 1]
        return _foreach_set(uv_layer.data, "uv", output.reshape(-1))
    except Exception:
        return False


def _apply_custom_normals(mesh, normals: list[tuple[float, float, float]]) -> None:
    if len(normals) != len(mesh.vertices):
        return
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    mesh.normals_split_custom_set_from_vertices(normals)


def _store_int_attribute(mesh, name: str, values: list[int]) -> None:
    attribute = mesh.attributes.new(name=name, type="INT", domain="POINT")
    np = require_numpy()
    payload = np.asarray(values, dtype=np.int32)
    if _foreach_set(attribute.data, "value", payload):
        return
    raise ValueError(f"{name}: numpy INT attribute write failed")


def _store_float_attribute(mesh, name: str, values: list[float]) -> None:
    attribute = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
    np = require_numpy()
    payload = np.asarray(values, dtype=np.float32)
    if _foreach_set(attribute.data, "value", payload):
        return
    raise ValueError(f"{name}: numpy FLOAT attribute write failed")


def _store_tangent_frame_attributes(mesh, tangents, bitangent_signs) -> None:
    np = require_numpy()
    tangent_values = np.asarray(tangents, dtype=np.float32)
    if tangent_values.ndim < 2 or tangent_values.shape[1] < 3:
        raise ValueError("packed tangent attribute payload is not an Nx3 array")
    _store_float_attribute(mesh, "bmc_tangent_x", tangent_values[:, 0])
    _store_float_attribute(mesh, "bmc_tangent_y", tangent_values[:, 1])
    _store_float_attribute(mesh, "bmc_tangent_z", tangent_values[:, 2])
    _store_float_attribute(mesh, "bmc_bitangent_sign", bitangent_signs)


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
    np = require_numpy()
    array = np.asarray(values, dtype=np.uint32)
    _store_int_attribute(mesh, f"{base_name}_lo16", (array & 0xFFFF).astype(np.int32))
    _store_int_attribute(mesh, f"{base_name}_hi16", ((array >> 16) & 0xFFFF).astype(np.int32))


def _store_texcoord_semantic_attributes(mesh, records: list[dict]) -> None:
    for record in records:
        slot_index = int(record.get("slot_index", -1))
        semantic_index = int(record.get("semantic_index", -1))
        component_count = int(record.get("component_count", 0) or 0)
        storage = str(record.get("storage", "") or "")
        if storage == "uv_layer":
            continue
        values = record.get("values", [])
        if values is None:
            values = []
        if slot_index < 0 or semantic_index < 0 or component_count <= 0:
            continue
        base_name = f"bmc_vb{slot_index}_texcoord{semantic_index}"
        if storage == "sint8_raw" and component_count == 4:
            _store_snorm_byte_color_attribute(
                mesh,
                texcoord_color_attr_names(f"vb{slot_index}", semantic_index)[0],
                values,
            )
            continue
        for component_index in range(component_count):
            component_values = _component_values(values, component_index)
            attr_name = f"{base_name}_{component_index}"
            _store_float_attribute(mesh, attr_name, component_values)


def _component_values(values, component_index: int):
    np = require_numpy()
    array = np.asarray(values)
    if array.ndim < 2 or int(component_index) >= int(array.shape[1]):
        return np.zeros((len(values),), dtype=np.float32)
    return array[:, int(component_index)]


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
    np = require_numpy()
    array = np.asarray(values, dtype=np.int16)
    if array.ndim < 2 or array.shape[1] < 4:
        colors = np.zeros((0,), dtype=np.float32)
    else:
        colors = ((array[:, :4].astype(np.int16) & 0xFF).astype(np.float32) / 255.0).reshape(-1)
    data = getattr(attribute, "data", [])
    if _foreach_set(data, "color", colors):
        return
    raise ValueError(f"{name}: numpy color attribute write failed")


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
