"""Blender-side mesh operations for global vertex-group remapping."""

from __future__ import annotations

from ..constants import (
    BMC_EXPORT_CHUNK_PROP,
    BMC_EXPORT_PALETTE_PROP,
    BMC_GLOBAL_POOL_GENERATION_PROP,
    BMC_GLOBAL_REMAP_PROP,
    BMC_GLOBAL_SOURCE_KEY_PROP,
    BMC_ORIGINAL_LOCAL_BONE_COUNT_PROP,
    BMC_VERTEX_GROUP_STATE_GLOBAL,
    BMC_VERTEX_GROUP_STATE_PROP,
)
from .identity import infer_mesh_identity_from_name
from .models import RemapApplyResult


def resolve_mesh_identity(mesh_obj):
    autodetected = bool(getattr(mesh_obj, "merge_ib_autodetected", True))
    manual_hash = str(getattr(mesh_obj, "merge_ib_hash", "")).strip().lower()
    if not autodetected and manual_hash:
        mesh_obj.merge_ib_autodetected = False
        return manual_hash, 0

    inferred = infer_mesh_identity_from_name(mesh_obj.name)
    if inferred is None:
        mesh_obj.merge_ib_hash = ""
        mesh_obj.merge_match_index_count = 0
        mesh_obj.merge_ib_autodetected = True
        return None
    mesh_obj.merge_ib_hash = inferred[0]
    mesh_obj.merge_match_index_count = 0
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
        numeric_index = _parse_numeric_group(vertex_group.name)
        if numeric_index is not None:
            numeric_group_indices.append(numeric_index)

    if not numeric_group_indices:
        raise ValueError(f"{mesh_obj.name}: no numeric local vertex groups found")
    return max(numeric_group_indices) + 1


def apply_group_remaps_to_meshes(mesh_objects, manifest: dict, identity_resolver=None) -> RemapApplyResult:
    remap_index = {}
    for entry in manifest.get("object_remaps", []):
        ib_hash = str(entry["ib_hash"]).lower()
        remap_index[(str(entry.get("object_name", "")), ib_hash)] = entry
        remap_index[("", ib_hash)] = entry

    resolver = identity_resolver or resolve_mesh_identity
    updated_objects = 0
    renamed_groups = 0
    skipped_objects: list[str] = []

    for mesh_obj in mesh_objects:
        mesh_identity = resolver(mesh_obj)
        if mesh_identity is None:
            skipped_objects.append(f"{mesh_obj.name}: cannot infer ib_hash")
            continue

        mesh_ib_hash = str(mesh_identity[0]).lower()
        remap_entry = remap_index.get((mesh_obj.name, mesh_ib_hash))
        if remap_entry is None:
            remap_entry = remap_index.get(("", mesh_ib_hash))
        if remap_entry is None:
            skipped_objects.append(f"{mesh_obj.name}: no remap entry for {mesh_ib_hash}")
            continue

        local_to_global = _normalize_local_to_global(remap_entry.get("local_group_to_global_group", {}))
        source_key = str(remap_entry.get("source_key", "") or "")
        generation_id = str(manifest.get("global_pool_generation", "") or manifest.get("generation_id", "") or "")
        renamed = _apply_group_rename(
            mesh_obj,
            local_to_global,
            source_key=source_key,
            generation_id=generation_id,
        )
        if renamed or _mesh_has_expected_global_remap(
            mesh_obj,
            local_to_global,
            source_key=source_key,
            generation_id=generation_id,
        ):
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


def _apply_group_rename(
    mesh_obj,
    local_to_global: dict[int, int],
    *,
    source_key: str = "",
    generation_id: str = "",
) -> int:
    if not local_to_global:
        return 0

    if _mesh_has_expected_global_remap(
        mesh_obj,
        local_to_global,
        source_key=source_key,
        generation_id=generation_id,
    ):
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
            _set_global_remap_metadata(
                mesh_obj,
                local_to_global,
                source_key=source_key,
                generation_id=generation_id,
            )
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
    _set_global_remap_metadata(
        mesh_obj,
        local_to_global,
        source_key=source_key,
        generation_id=generation_id,
    )
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
        if source_name != target_name:
            rename_by_source[source_name] = target_name

    return sorted(rename_by_source.items(), key=lambda item: (_parse_numeric_group(item[0]) or 0, item[0]))


def _mesh_has_expected_global_remap(
    mesh_obj,
    local_to_global: dict[int, int],
    *,
    source_key: str = "",
    generation_id: str = "",
) -> bool:
    expected = _dense_remap_sequence(local_to_global)
    current = _read_int_sequence_prop(mesh_obj, BMC_GLOBAL_REMAP_PROP)
    if not expected or current != expected or mesh_obj.get(BMC_VERTEX_GROUP_STATE_PROP) != BMC_VERTEX_GROUP_STATE_GLOBAL:
        return False

    current_source_key = str(mesh_obj.get(BMC_GLOBAL_SOURCE_KEY_PROP, "") or getattr(mesh_obj, BMC_GLOBAL_SOURCE_KEY_PROP, "") or "")
    if source_key and current_source_key != str(source_key):
        return False

    current_generation_id = str(
        mesh_obj.get(BMC_GLOBAL_POOL_GENERATION_PROP, "") or getattr(mesh_obj, BMC_GLOBAL_POOL_GENERATION_PROP, "") or ""
    )
    if generation_id and current_generation_id != str(generation_id):
        return False

    expected_global_names = {int(global_index) for global_index in expected if int(global_index) >= 0}
    current_numeric_names = {
        numeric_group
        for vertex_group in mesh_obj.vertex_groups
        if (numeric_group := _parse_numeric_group(vertex_group.name)) is not None
    }
    # Metadata can survive copies or manual edits. Treat an object as already
    # global only when its visible vertex-group names actually contain globals.
    return bool(current_numeric_names and current_numeric_names.intersection(expected_global_names))


def _set_global_remap_metadata(
    mesh_obj,
    local_to_global: dict[int, int],
    *,
    source_key: str = "",
    generation_id: str = "",
) -> None:
    remap_sequence = _dense_remap_sequence(local_to_global)
    if remap_sequence:
        mesh_obj[BMC_GLOBAL_REMAP_PROP] = list(remap_sequence)
        mesh_obj[BMC_ORIGINAL_LOCAL_BONE_COUNT_PROP] = len(remap_sequence)
    mesh_obj[BMC_VERTEX_GROUP_STATE_PROP] = BMC_VERTEX_GROUP_STATE_GLOBAL
    if source_key:
        setattr(mesh_obj, BMC_GLOBAL_SOURCE_KEY_PROP, str(source_key))
        mesh_obj[BMC_GLOBAL_SOURCE_KEY_PROP] = str(source_key)
    if generation_id:
        setattr(mesh_obj, BMC_GLOBAL_POOL_GENERATION_PROP, str(generation_id))
        mesh_obj[BMC_GLOBAL_POOL_GENERATION_PROP] = str(generation_id)
    _clear_export_local_metadata(mesh_obj)


def _clear_export_local_metadata(mesh_obj) -> None:
    for prop_name in (BMC_EXPORT_PALETTE_PROP, BMC_EXPORT_CHUNK_PROP):
        if prop_name in mesh_obj:
            del mesh_obj[prop_name]


def _normalize_local_to_global(local_to_global: dict[str, int]) -> dict[int, int]:
    normalized = {}
    for local_index, global_index in local_to_global.items():
        try:
            local_int = int(local_index)
            global_int = int(global_index)
        except (TypeError, ValueError):
            continue
        if local_int >= 0 and global_int >= 0:
            normalized[local_int] = global_int
    return normalized


def _dense_remap_sequence(local_to_global: dict[int, int]) -> tuple[int, ...]:
    if not local_to_global:
        return ()
    max_local = max(local_to_global)
    return tuple(int(local_to_global.get(local_index, -1)) for local_index in range(max_local + 1))


def _invert_remap_sequence(remap_sequence: tuple[int, ...] | None) -> dict[int, int]:
    if not remap_sequence:
        return {}
    return {
        int(global_index): local_index
        for local_index, global_index in enumerate(remap_sequence)
        if int(global_index) >= 0
    }


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


def _parse_numeric_group(group_name: str) -> int | None:
    try:
        group_index = int(str(group_name).strip())
    except ValueError:
        return None
    if group_index < 0:
        return None
    return group_index
