"""Blender-side mesh operations for remap and alias application."""

from __future__ import annotations

import struct
from math import floor

from ..constants import (
    BMC_EXPORT_CHUNK_PROP,
    BMC_EXPORT_PALETTE_PROP,
    BMC_GLOBAL_REMAP_PROP,
    BMC_ORIGINAL_LOCAL_BONE_COUNT_PROP,
    BMC_VERTEX_GROUP_STATE_GLOBAL,
    BMC_VERTEX_GROUP_STATE_PROP,
    GLOBAL_RESERVED_ROWS,
)
from .frameanalysis import infer_mesh_identity_from_name
from .models import DuplicateMergeResult, RemapApplyResult

_CENTER_DISTANCE_NEAR = 0.0025
_CENTER_DISTANCE_CLOSE = 0.01
_SEAM_MATCH_TOLERANCE = 0.0015
_SEAM_WEIGHT_TOLERANCE = 0.001
_SEAM_MIN_VERTEX_PAIRS = 4
_SEAM_MIN_MAPPING_VOTES = 3
_SEAM_MAX_AVERAGE_DISTANCE = 0.001
_WEIGHT_EPSILON = 1.0e-8
_MAPPING_WEIGHT_FLOOR = 1.0e-4


def resolve_mesh_identity(mesh_obj) -> tuple[str, int] | None:
    autodetected = bool(getattr(mesh_obj, "merge_ib_autodetected", True))
    manual_hash = str(getattr(mesh_obj, "merge_ib_hash", "")).strip().lower()
    manual_count = int(getattr(mesh_obj, "merge_match_index_count", -1))
    if not autodetected and manual_hash and manual_count >= 0:
        mesh_obj.merge_ib_autodetected = False
        return manual_hash, manual_count

    inferred = infer_mesh_identity_from_name(mesh_obj.name)
    if inferred is None:
        mesh_obj.merge_ib_hash = ""
        mesh_obj.merge_match_index_count = -1
        mesh_obj.merge_ib_autodetected = True
        return None
    mesh_obj.merge_ib_hash = inferred[0]
    mesh_obj.merge_match_index_count = inferred[1]
    mesh_obj.merge_ib_autodetected = True
    return inferred


def infer_local_bone_count_from_mesh(mesh_obj) -> int:
    original_count = _read_int_prop(mesh_obj, BMC_ORIGINAL_LOCAL_BONE_COUNT_PROP)
    if original_count is not None and original_count > 0:
        return original_count

    global_remap = _read_int_sequence_prop(mesh_obj, BMC_GLOBAL_REMAP_PROP)
    if global_remap:
        return len(global_remap)

    localized_palette = _read_int_sequence_prop(mesh_obj, BMC_EXPORT_PALETTE_PROP)
    if localized_palette:
        return len(localized_palette)

    numeric_group_indices: list[int] = []
    for vertex_group in mesh_obj.vertex_groups:
        try:
            numeric_index = int(str(vertex_group.name).strip())
        except ValueError:
            continue
        if numeric_index < 0:
            continue
        numeric_group_indices.append(numeric_index)

    if not numeric_group_indices:
        raise ValueError(f"{mesh_obj.name}: no numeric local vertex groups found")
    return max(numeric_group_indices) + 1


def _read_int_prop(mesh_obj, prop_name: str) -> int | None:
    raw_value = mesh_obj.get(prop_name)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _read_int_sequence_prop(mesh_obj, prop_name: str) -> tuple[int, ...] | None:
    raw_value = mesh_obj.get(prop_name)
    if raw_value is None:
        return None
    try:
        return tuple(int(value) for value in raw_value)
    except (TypeError, ValueError):
        return None


def _normalize_local_to_global(local_to_global: dict[str, int]) -> dict[int, int]:
    normalized = {}
    for local_index, global_index in local_to_global.items():
        try:
            local_int = int(local_index)
            global_int = int(global_index)
        except (TypeError, ValueError):
            continue
        if local_int < 0 or global_int < 0:
            continue
        normalized[local_int] = global_int
    return normalized


def _dense_remap_sequence(local_to_global: dict[int, int]) -> tuple[int, ...]:
    if not local_to_global:
        return ()
    max_local = max(local_to_global)
    return tuple(int(local_to_global.get(local_index, -1)) for local_index in range(max_local + 1))


def _set_global_remap_metadata(mesh_obj, local_to_global: dict[int, int]) -> None:
    remap_sequence = _dense_remap_sequence(local_to_global)
    if remap_sequence:
        mesh_obj[BMC_GLOBAL_REMAP_PROP] = list(remap_sequence)
        mesh_obj[BMC_ORIGINAL_LOCAL_BONE_COUNT_PROP] = len(remap_sequence)
    mesh_obj[BMC_VERTEX_GROUP_STATE_PROP] = BMC_VERTEX_GROUP_STATE_GLOBAL
    _clear_export_local_metadata(mesh_obj)


def _clear_export_local_metadata(mesh_obj) -> None:
    for prop_name in (BMC_EXPORT_PALETTE_PROP, BMC_EXPORT_CHUNK_PROP):
        if prop_name in mesh_obj:
            del mesh_obj[prop_name]


def _mesh_has_expected_global_remap(mesh_obj, local_to_global: dict[int, int]) -> bool:
    expected = _dense_remap_sequence(local_to_global)
    current = _read_int_sequence_prop(mesh_obj, BMC_GLOBAL_REMAP_PROP)
    return bool(expected and current == expected and mesh_obj.get(BMC_VERTEX_GROUP_STATE_PROP) == BMC_VERTEX_GROUP_STATE_GLOBAL)


def annotate_alias_items_with_mesh_proximity(scene, alias_items) -> None:
    requested_groups_by_object: dict[str, set[str]] = {}
    for item in alias_items:
        requested_groups_by_object.setdefault(str(item.src_object_name), set()).add(str(item.src_global_bone))
        requested_groups_by_object.setdefault(str(item.canonical_object_name), set()).add(str(item.canonical_global_bone))

    center_cache: dict[tuple[str, str], object | None] = {}
    for object_name, group_names in requested_groups_by_object.items():
        mesh_obj = scene.objects.get(object_name)
        if mesh_obj is None or mesh_obj.type != "MESH":
            for group_name in group_names:
                center_cache[(object_name, group_name)] = None
            continue

        centers = _infer_group_weighted_centers_world_bulk(mesh_obj, group_names)
        for group_name in group_names:
            center_cache[(object_name, group_name)] = centers.get(group_name)

    for item in alias_items:
        base_confidence = str(item.confidence or "exact_current_previous")
        source_center = center_cache.get((str(item.src_object_name), str(item.src_global_bone)))
        canonical_center = center_cache.get((str(item.canonical_object_name), str(item.canonical_global_bone)))
        if source_center is None or canonical_center is None:
            item.confidence = base_confidence
            continue

        distance = float((source_center - canonical_center).length)
        if distance <= _CENTER_DISTANCE_NEAR:
            proximity_label = "near_center"
        elif distance <= _CENTER_DISTANCE_CLOSE:
            proximity_label = "close_center"
        else:
            proximity_label = "far_center"
        item.confidence = f"{base_confidence}|{proximity_label}|dist={distance:.6f}"


def mesh_objects_from_target_names(context, target_object_names: list[str]):
    mesh_objects = []
    missing_names = []
    for object_name in target_object_names:
        mesh_object = context.scene.objects.get(object_name)
        if mesh_object is None or mesh_object.type != "MESH":
            missing_names.append(object_name)
            continue
        mesh_objects.append(mesh_object)
    return mesh_objects, missing_names


def build_seam_filtered_aliases_from_manifest(context, manifest: dict, target_object_names: list[str]) -> list[dict]:
    mesh_objects, _missing_names = mesh_objects_from_target_names(context, target_object_names)
    mesh_by_name = {mesh_obj.name: mesh_obj for mesh_obj in mesh_objects}
    signature_by_global, metadata_by_global = _build_global_bone_signature_index(manifest)

    aliases_by_key: dict[tuple[int, int], dict] = {}
    object_names = [name for name in target_object_names if name in mesh_by_name]
    for source_index, source_name in enumerate(object_names):
        source_obj = mesh_by_name[source_name]
        source_seam_vertices = _collect_seam_vertices(source_obj)
        if not source_seam_vertices:
            continue

        for target_name in object_names[source_index + 1 :]:
            target_obj = mesh_by_name[target_name]
            target_seam_vertices = _collect_seam_vertices(target_obj)
            if not target_seam_vertices:
                continue

            matched_pairs = _build_vertex_pairs(source_seam_vertices, target_seam_vertices, _SEAM_MATCH_TOLERANCE)
            if len(matched_pairs) < _SEAM_MIN_VERTEX_PAIRS:
                continue

            candidates = _build_mapping_candidates_from_seams(
                source_obj,
                target_obj,
                matched_pairs,
                _SEAM_WEIGHT_TOLERANCE,
            )
            for candidate in candidates:
                source_group = _parse_numeric_group(candidate["group_a"])
                target_group = _parse_numeric_group(candidate["group_b"])
                if source_group is None or target_group is None or source_group == target_group:
                    continue
                if signature_by_global.get(source_group) != signature_by_global.get(target_group):
                    continue

                source_meta = metadata_by_global.get(source_group)
                target_meta = metadata_by_global.get(target_group)
                if source_meta is None or target_meta is None:
                    continue

                canonical_group, canonical_meta, duplicate_group, duplicate_meta = _choose_canonical_bone(
                    source_group,
                    source_meta,
                    target_group,
                    target_meta,
                )
                alias_key = (duplicate_group, canonical_group)
                existing_alias = aliases_by_key.get(alias_key)
                if existing_alias is not None and existing_alias.get("_score", 0.0) >= candidate["score"]:
                    continue

                aliases_by_key[alias_key] = {
                    "src_draw_index": duplicate_meta["draw_index"],
                    "src_object_name": duplicate_meta["object_name"],
                    "src_ib_hash": duplicate_meta["ib_hash"],
                    "src_local_bone": duplicate_meta["local_bone"],
                    "src_global_bone": duplicate_group,
                    "canonical_draw_index": canonical_meta["draw_index"],
                    "canonical_object_name": canonical_meta["object_name"],
                    "canonical_ib_hash": canonical_meta["ib_hash"],
                    "canonical_local_bone": canonical_meta["local_bone"],
                    "canonical_global_bone": canonical_group,
                    "confidence": (
                        "seam_weight_exact_matrix"
                        f"|votes={candidate['vote_count']}"
                        f"|score={candidate['score']:.6f}"
                        f"|avg_dist={candidate['average_distance']:.6f}"
                        f"|avg_wdiff={candidate['average_weight_difference']:.6f}"
                    ),
                    "_score": candidate["score"],
                }

    aliases = []
    for alias in aliases_by_key.values():
        alias.pop("_score", None)
        aliases.append(alias)
    aliases.sort(
        key=lambda item: (
            int(item["canonical_draw_index"]),
            int(item["src_draw_index"]),
            int(item["src_global_bone"]),
        )
    )
    return aliases


def apply_group_remaps_to_meshes(mesh_objects, manifest: dict) -> RemapApplyResult:
    remap_index = {}
    for entry in manifest.get("object_remaps", []):
        remap_index[(entry.get("object_name", ""), entry["ib_hash"].lower(), int(entry["match_index_count"]))] = entry
        remap_index[("", entry["ib_hash"].lower(), int(entry["match_index_count"]))] = entry

    updated_objects = 0
    renamed_groups = 0
    skipped_objects: list[str] = []

    for mesh_obj in mesh_objects:
        mesh_identity = resolve_mesh_identity(mesh_obj)
        if mesh_identity is None:
            skipped_objects.append(f"{mesh_obj.name}: cannot infer ib_hash/index_count")
            continue

        remap_entry = remap_index.get((mesh_obj.name, mesh_identity[0], mesh_identity[1]))
        if remap_entry is None:
            remap_entry = remap_index.get(("", mesh_identity[0], mesh_identity[1]))
        if remap_entry is None:
            skipped_objects.append(f"{mesh_obj.name}: no remap entry for {mesh_identity[0]}-{mesh_identity[1]}")
            continue

        local_to_global = _normalize_local_to_global(remap_entry.get("local_group_to_global_group", {}))
        renamed = _apply_group_rename(mesh_obj, local_to_global)
        if renamed or _mesh_has_expected_global_remap(mesh_obj, local_to_global):
            updated_objects += 1
            renamed_groups += renamed
        else:
            skipped_objects.append(f"{mesh_obj.name}: no matching numeric groups")

    return RemapApplyResult(
        target_objects=len(mesh_objects),
        updated_objects=updated_objects,
        renamed_groups=renamed_groups,
        skipped_objects=tuple(skipped_objects),
    )


def _apply_group_rename(mesh_obj, local_to_global: dict[int, int]) -> int:
    if not local_to_global:
        return 0

    if _mesh_has_expected_global_remap(mesh_obj, local_to_global):
        return 0

    rename_pairs = _build_rename_pairs_for_current_state(mesh_obj, local_to_global)
    if not rename_pairs:
        existing_global_names = {str(global_index) for global_index in local_to_global.values()}
        current_numeric_names = {
            str(vertex_group.name).strip()
            for vertex_group in mesh_obj.vertex_groups
            if _parse_numeric_group(vertex_group.name) is not None
        }
        if existing_global_names and existing_global_names.issubset(current_numeric_names):
            _set_global_remap_metadata(mesh_obj, local_to_global)
        return 0

    temp_name_by_source: dict[str, str] = {}
    for source_name, _target_name in rename_pairs:
        vertex_group = mesh_obj.vertex_groups.get(source_name)
        if vertex_group is None:
            continue
        temp_name = f"__bmc_tmp__{vertex_group.index}__{source_name}"
        vertex_group.name = temp_name
        temp_name_by_source[source_name] = temp_name

    renamed_count = 0
    for source_name, target_name in rename_pairs:
        temp_name = temp_name_by_source.get(source_name, "")
        if not temp_name:
            continue
        mesh_obj.vertex_groups[temp_name].name = target_name
        renamed_count += 1
    _set_global_remap_metadata(mesh_obj, local_to_global)
    return renamed_count


def _build_rename_pairs_for_current_state(mesh_obj, local_to_global: dict[int, int]) -> list[tuple[str, str]]:
    current_remap = _read_int_sequence_prop(mesh_obj, BMC_GLOBAL_REMAP_PROP)
    localized_palette = _read_int_sequence_prop(mesh_obj, BMC_EXPORT_PALETTE_PROP)
    global_to_original_local = _invert_remap_sequence(current_remap)

    rename_by_source: dict[str, str] = {}
    for vertex_group in mesh_obj.vertex_groups:
        numeric_name = _parse_numeric_group(vertex_group.name)
        if numeric_name is None:
            continue

        original_local = None
        if localized_palette is not None and 0 <= numeric_name < len(localized_palette):
            original_local = global_to_original_local.get(int(localized_palette[numeric_name]))
        elif current_remap is not None:
            original_local = global_to_original_local.get(numeric_name)
        else:
            original_local = numeric_name

        if original_local is None:
            continue
        target_global = local_to_global.get(int(original_local))
        if target_global is None:
            continue

        source_name = str(vertex_group.name)
        target_name = str(int(target_global))
        if source_name == target_name:
            continue
        rename_by_source[source_name] = target_name

    return sorted(rename_by_source.items(), key=lambda item: (_parse_numeric_group(item[0]) or 0, item[0]))


def _invert_remap_sequence(remap_sequence: tuple[int, ...] | None) -> dict[int, int]:
    if not remap_sequence:
        return {}
    return {
        int(global_index): local_index
        for local_index, global_index in enumerate(remap_sequence)
        if int(global_index) >= 0
    }


def merge_duplicate_alias_weights(mesh_objects, alias_entries: list[dict]) -> DuplicateMergeResult:
    updated_objects = 0
    merged_aliases = 0
    skipped_objects: list[str] = []

    active_alias_entries = [
        entry
        for entry in alias_entries
        if entry.get("enabled", True) and _is_safe_alias_entry(entry)
    ]
    if not active_alias_entries:
        return DuplicateMergeResult(
            target_objects=len(mesh_objects),
            updated_objects=0,
            merged_aliases=0,
            skipped_objects=("No enabled duplicate-bone aliases configured",),
        )

    aliases_by_object: dict[str, list[dict]] = {}
    aliases_by_ib_hash: dict[str, list[dict]] = {}
    for alias_entry in active_alias_entries:
        aliases_by_object.setdefault(str(alias_entry.get("src_object_name", "")).strip(), []).append(alias_entry)
        src_ib_hash = str(alias_entry.get("src_ib_hash", "")).strip().lower()
        if src_ib_hash:
            aliases_by_ib_hash.setdefault(src_ib_hash, []).append(alias_entry)

    for mesh_obj in mesh_objects:
        relevant_alias_entries = aliases_by_object.get(mesh_obj.name, [])
        if not relevant_alias_entries:
            mesh_identity = resolve_mesh_identity(mesh_obj)
            if mesh_identity is not None:
                relevant_alias_entries = aliases_by_ib_hash.get(mesh_identity[0].lower(), [])
        if not relevant_alias_entries:
            skipped_objects.append(f"{mesh_obj.name}: no duplicate alias groups present")
            continue

        relevant_source_group_names = {
            str(int(alias_entry["src_global_bone"])) for alias_entry in relevant_alias_entries
        }
        source_group_entries = _collect_group_member_weights(mesh_obj, relevant_source_group_names)
        groups_by_global_name = _build_global_name_to_vertex_groups(mesh_obj)
        changed = False
        for alias_entry in relevant_alias_entries:
            source_group_name = str(int(alias_entry["src_global_bone"]))
            canonical_group_name = str(int(alias_entry["canonical_global_bone"]))
            source_group = _first_group_for_global(groups_by_global_name, source_group_name)
            if source_group is None:
                continue
            canonical_group = _first_group_for_global(groups_by_global_name, canonical_group_name)
            if canonical_group is None:
                canonical_group = mesh_obj.vertex_groups.new(name=canonical_group_name)
                groups_by_global_name.setdefault(canonical_group_name, []).append(canonical_group)
            moved_vertices = _move_vertex_group_weights(
                mesh_obj,
                source_group,
                canonical_group,
                source_group_entries.get(source_group_name, ()),
            )
            if moved_vertices <= 0:
                continue
            merged_aliases += 1
            changed = True
        if changed:
            updated_objects += 1
        else:
            skipped_objects.append(f"{mesh_obj.name}: no duplicate alias groups present")

    return DuplicateMergeResult(
        target_objects=len(mesh_objects),
        updated_objects=updated_objects,
        merged_aliases=merged_aliases,
        skipped_objects=tuple(skipped_objects),
    )


def _move_vertex_group_weights(mesh_obj, source_group, target_group, source_entries) -> int:
    moved_vertices = 0
    if source_group.index == target_group.index:
        return 0
    for vertex_index, source_weight in source_entries:
        if source_weight <= 0.0:
            continue
        target_weight = _safe_weight(target_group, vertex_index)
        target_group.add([vertex_index], min(1.0, target_weight + source_weight), "REPLACE")
        source_group.remove([vertex_index])
        moved_vertices += 1
    return moved_vertices


def _safe_weight(vertex_group, vertex_index: int) -> float:
    try:
        return float(vertex_group.weight(vertex_index))
    except RuntimeError:
        return 0.0


def _is_safe_alias_entry(alias_entry: dict) -> bool:
    confidence = str(alias_entry.get("confidence", "") or "")
    if confidence.startswith("exact_current_previous") and "seam_weight" not in confidence:
        return False
    return True


def _build_group_index_to_global_name_map(mesh_obj) -> dict[int, str]:
    localized_palette = _read_int_sequence_prop(mesh_obj, BMC_EXPORT_PALETTE_PROP)
    group_index_to_global_name = {}
    for vertex_group in mesh_obj.vertex_groups:
        numeric_group = _parse_numeric_group(vertex_group.name)
        if numeric_group is None:
            continue
        if localized_palette is not None and 0 <= numeric_group < len(localized_palette):
            global_group = int(localized_palette[numeric_group])
        else:
            global_group = numeric_group
        if global_group < 0:
            continue
        group_index_to_global_name[int(vertex_group.index)] = str(global_group)
    return group_index_to_global_name


def _build_global_name_to_vertex_groups(mesh_obj) -> dict[str, list[object]]:
    group_index_to_global_name = _build_group_index_to_global_name_map(mesh_obj)
    groups_by_global_name: dict[str, list[object]] = {}
    for vertex_group in mesh_obj.vertex_groups:
        global_name = group_index_to_global_name.get(int(vertex_group.index))
        if global_name is None:
            continue
        groups_by_global_name.setdefault(global_name, []).append(vertex_group)
    return groups_by_global_name


def _first_group_for_global(groups_by_global_name: dict[str, list[object]], global_group_name: str):
    groups = groups_by_global_name.get(str(global_group_name), [])
    if groups:
        return groups[0]
    return None


def _infer_group_weighted_center_world(mesh_obj, group_name: str):
    return _infer_group_weighted_centers_world_bulk(mesh_obj, {str(group_name)}).get(str(group_name))


def _infer_group_weighted_centers_world_bulk(mesh_obj, group_names: set[str]) -> dict[str, object]:
    requested_names = {str(group_name) for group_name in group_names}
    group_index_to_global_name = _build_group_index_to_global_name_map(mesh_obj)
    relevant_group_indices = {
        group_index
        for group_index, global_name in group_index_to_global_name.items()
        if global_name in requested_names
    }
    if not relevant_group_indices:
        return {}

    weighted_sums = {}
    total_weights = {}

    for vertex in mesh_obj.data.vertices:
        if not vertex.groups:
            continue
        world_position = mesh_obj.matrix_world @ vertex.co
        for group_element in vertex.groups:
            group_index = int(group_element.group)
            if group_index not in relevant_group_indices:
                continue
            global_name = group_index_to_global_name.get(group_index)
            if global_name is None:
                continue
            weight = float(group_element.weight)
            if weight <= 0.0:
                continue
            weighted_position = world_position * weight
            weighted_sums[global_name] = (
                weighted_position
                if global_name not in weighted_sums
                else weighted_sums[global_name] + weighted_position
            )
            total_weights[global_name] = total_weights.get(global_name, 0.0) + weight

    centers = {}
    for group_name in requested_names:
        total_weight = total_weights.get(group_name, 0.0)
        if total_weight <= 0.0:
            continue
        centers[group_name] = weighted_sums[group_name] / total_weight
    return centers


def _collect_group_member_weights(mesh_obj, group_names: set[str]) -> dict[str, tuple[tuple[int, float], ...]]:
    requested_names = {str(group_name) for group_name in group_names}
    group_index_to_global_name = _build_group_index_to_global_name_map(mesh_obj)
    relevant_group_indices = {
        group_index
        for group_index, global_name in group_index_to_global_name.items()
        if global_name in requested_names
    }
    if not relevant_group_indices:
        return {}

    members = {group_name: [] for group_name in requested_names}
    for vertex in mesh_obj.data.vertices:
        for group_element in vertex.groups:
            group_index = int(group_element.group)
            if group_index not in relevant_group_indices:
                continue
            weight = float(group_element.weight)
            if weight <= 0.0:
                continue
            global_name = group_index_to_global_name.get(group_index)
            if global_name is None:
                continue
            members[global_name].append((vertex.index, weight))

    return {group_name: tuple(entries) for group_name, entries in members.items()}


def _build_global_bone_signature_index(manifest: dict):
    signature_by_global = {}
    metadata_by_global = {}
    for part in manifest.get("part_records", []):
        global_bone_base = int(part["global_bone_base"])
        capture_bone_count = int(part["capture_bone_count"])
        current_base, previous_base = _read_palette_bases_from_vs_cb1(part["vs_cb1_path"])
        with open(part["vs_t0_path"], "rb") as file_handle:
            vs_t0_blob = file_handle.read()
        total_rows = len(vs_t0_blob) // 16

        for local_bone in range(capture_bone_count):
            global_bone = global_bone_base + local_bone
            signature_by_global[global_bone] = _build_bone_signature_from_blob(
                vs_t0_blob,
                total_rows,
                current_base,
                previous_base,
                local_bone,
            )
            metadata_by_global[global_bone] = {
                "draw_index": int(part["draw_index"]),
                "object_name": str(part["object_name"]),
                "ib_hash": str(part["ib_hash"]),
                "local_bone": local_bone,
            }
    return signature_by_global, metadata_by_global


def _read_palette_bases_from_vs_cb1(vs_cb1_path: str) -> tuple[int, int]:
    with open(vs_cb1_path, "rb") as file_handle:
        file_handle.seek(5 * 16)
        row_bytes = file_handle.read(16)
    if len(row_bytes) != 16:
        raise ValueError(f"vs-cb1 buffer too small: {vs_cb1_path}")
    x_value, y_value, _z_value, _w_value = struct.unpack("<4I", row_bytes)
    return x_value, y_value


def _build_bone_signature_from_blob(
    vs_t0_blob: bytes,
    total_rows: int,
    current_base: int,
    previous_base: int,
    local_bone: int,
) -> bytes:
    current_row = current_base + GLOBAL_RESERVED_ROWS + local_bone * 3
    previous_row = previous_base + GLOBAL_RESERVED_ROWS + local_bone * 3
    return _read_three_rows_from_blob(vs_t0_blob, current_row, total_rows) + _read_three_rows_from_blob(
        vs_t0_blob,
        previous_row,
        total_rows,
    )


def _read_three_rows_from_blob(vs_t0_blob: bytes, row_index: int, total_rows: int) -> bytes:
    blobs: list[bytes] = []
    for row_offset in range(3):
        wrapped_row_index = (int(row_index) + row_offset) % int(total_rows)
        byte_offset = wrapped_row_index * 16
        row_blob = vs_t0_blob[byte_offset : byte_offset + 16]
        if len(row_blob) != 16:
            raise ValueError(f"vs-t0 buffer too small for row {wrapped_row_index}")
        blobs.append(row_blob)
    return b"".join(blobs)


def _choose_canonical_bone(source_group: int, source_meta: dict, target_group: int, target_meta: dict):
    source_key = (int(source_meta["draw_index"]), int(source_group))
    target_key = (int(target_meta["draw_index"]), int(target_group))
    if source_key <= target_key:
        return source_group, source_meta, target_group, target_meta
    return target_group, target_meta, source_group, source_meta


def _collect_seam_vertices(obj):
    boundary_indices = _resolve_boundary_vertex_indices(obj)
    matrix_world = obj.matrix_world
    vertices = []
    for vertex_index in boundary_indices:
        vertex = obj.data.vertices[vertex_index]
        vertices.append((vertex.index, matrix_world @ vertex.co))
    vertices.sort(key=lambda item: (item[1][0], item[1][1], item[1][2], item[0]))
    return vertices


def _resolve_boundary_vertex_indices(obj):
    mesh = obj.data
    edge_face_counts = {}

    for polygon in mesh.polygons:
        for edge_key in polygon.edge_keys:
            edge_face_counts[edge_key] = edge_face_counts.get(edge_key, 0) + 1

    boundary_indices = set()
    for edge in mesh.edges:
        edge_key = tuple(sorted(edge.vertices))
        if edge.is_loose or edge_face_counts.get(edge_key, 0) == 1:
            boundary_indices.update(edge.vertices)
    return boundary_indices


def _cell_key(world_co, tolerance):
    inverse_tolerance = 1.0 / tolerance
    return (
        floor(world_co[0] * inverse_tolerance),
        floor(world_co[1] * inverse_tolerance),
        floor(world_co[2] * inverse_tolerance),
    )


def _build_spatial_hash(vertices, tolerance):
    spatial_hash = {}
    for vertex_index, world_co in vertices:
        key = _cell_key(world_co, tolerance)
        spatial_hash.setdefault(key, []).append((vertex_index, world_co))
    return spatial_hash


def _neighbor_keys(base_key):
    base_x, base_y, base_z = base_key
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                yield (base_x + offset_x, base_y + offset_y, base_z + offset_z)


def _build_nearest_vertex_map(source_vertices, target_vertices, tolerance):
    target_spatial_hash = _build_spatial_hash(target_vertices, tolerance)
    tolerance_squared = tolerance * tolerance
    nearest_by_source = {}

    for source_index, source_world_co in source_vertices:
        nearest_match = None
        nearest_distance_squared = None
        for key in _neighbor_keys(_cell_key(source_world_co, tolerance)):
            for target_index, target_world_co in target_spatial_hash.get(key, ()):
                distance_squared = (source_world_co - target_world_co).length_squared
                if distance_squared > tolerance_squared:
                    continue
                if nearest_distance_squared is None or distance_squared < nearest_distance_squared:
                    nearest_match = target_index
                    nearest_distance_squared = distance_squared

        if nearest_match is None:
            continue

        nearest_by_source[source_index] = (nearest_match, nearest_distance_squared ** 0.5)
    return nearest_by_source


def _build_vertex_pairs(source_vertices, target_vertices, tolerance):
    source_to_target = _build_nearest_vertex_map(source_vertices, target_vertices, tolerance)
    target_to_source = _build_nearest_vertex_map(target_vertices, source_vertices, tolerance)
    matched_pairs = []

    for source_index, (target_index, distance) in source_to_target.items():
        reverse = target_to_source.get(target_index)
        if reverse is None:
            continue
        reverse_source_index, _reverse_distance = reverse
        if reverse_source_index != source_index:
            continue
        matched_pairs.append((source_index, target_index, distance))
    return matched_pairs


def _build_mapping_candidates_from_seams(source_obj, target_obj, matched_pairs, weight_tolerance):
    candidate_stats = {}
    for source_index, target_index, pair_distance in matched_pairs:
        source_items = _sorted_weight_items(_read_vertex_weights(source_obj, source_index))
        target_items = _sorted_weight_items(_read_vertex_weights(target_obj, target_index))
        if not source_items or not target_items:
            continue
        _accumulate_mapping_candidates(candidate_stats, source_items, target_items, weight_tolerance, pair_distance)

    candidates = []
    for (group_a, group_b), stats in candidate_stats.items():
        if stats["votes"] < _SEAM_MIN_MAPPING_VOTES:
            continue
        average_distance = stats["distance_sum"] / max(1, stats["votes"])
        average_weight_difference = stats["weight_difference_sum"] / max(1, stats["votes"])
        if average_distance > _SEAM_MAX_AVERAGE_DISTANCE:
            continue
        candidates.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "score": stats["score"],
                "vote_count": stats["votes"],
                "average_distance": average_distance,
                "average_weight_difference": average_weight_difference,
            }
        )
    candidates.sort(key=lambda item: (item["vote_count"], item["score"]), reverse=True)

    used_group_a = set()
    used_group_b = set()
    mappings = []
    for candidate in candidates:
        if candidate["group_a"] in used_group_a or candidate["group_b"] in used_group_b:
            continue
        used_group_a.add(candidate["group_a"])
        used_group_b.add(candidate["group_b"])
        mappings.append(candidate)
    return mappings


def _read_vertex_weights(obj, vertex_index):
    weight_map = {}
    group_index_to_global_name = _build_group_index_to_global_name_map(obj)
    vertex = obj.data.vertices[vertex_index]
    for assignment in vertex.groups:
        group_name = group_index_to_global_name.get(int(assignment.group))
        if group_name is None:
            continue
        if assignment.weight > _WEIGHT_EPSILON:
            weight_map[group_name] = min(1.0, weight_map.get(group_name, 0.0) + float(assignment.weight))
    return weight_map


def _sorted_weight_items(weight_map):
    filtered_items = [
        (group_name, weight)
        for group_name, weight in weight_map.items()
        if weight > _MAPPING_WEIGHT_FLOOR
    ]
    filtered_items.sort(key=lambda item: (-item[1], item[0]))
    return filtered_items


def _accumulate_mapping_candidates(candidate_stats, source_items, target_items, weight_tolerance, pair_distance):
    used_target_indices = set()

    for source_group, source_weight in source_items:
        best_match_index = None
        best_match_score = -1.0
        best_match_group = None

        for target_index, (target_group, target_weight) in enumerate(target_items):
            if target_index in used_target_indices:
                continue

            difference = abs(source_weight - target_weight)
            if difference > weight_tolerance:
                continue

            similarity = 1.0 - (difference / max(weight_tolerance, _WEIGHT_EPSILON))
            match_score = min(source_weight, target_weight) * similarity
            if match_score > best_match_score:
                best_match_index = target_index
                best_match_score = match_score
                best_match_group = target_group

        if best_match_index is None or best_match_group is None:
            continue

        used_target_indices.add(best_match_index)
        mapping_key = (source_group, best_match_group)
        stats = candidate_stats.setdefault(
            mapping_key,
            {"score": 0.0, "votes": 0, "distance_sum": 0.0, "weight_difference_sum": 0.0},
        )
        stats["score"] += best_match_score
        stats["votes"] += 1
        stats["distance_sum"] += float(pair_distance)
        stats["weight_difference_sum"] += abs(float(source_weight) - float(target_items[best_match_index][1]))


def _parse_numeric_group(group_name: str) -> int | None:
    try:
        numeric_group = int(str(group_name).strip())
    except ValueError:
        return None
    if numeric_group < 0:
        return None
    return numeric_group
