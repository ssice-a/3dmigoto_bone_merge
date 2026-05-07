"""Blender operators for the Bone Merge Capture plugin."""

from __future__ import annotations

import json
import os
import re

import bpy
from bpy.app.handlers import persistent

from .constants import (
    BI4_MAX_BONE_COUNT,
    BMC_GLOBAL_POOL_GENERATION_PROP,
    BMC_GLOBAL_REMAP_PROP,
    BMC_GLOBAL_SOURCE_KEY_PROP,
    BMC_VERTEX_GROUP_STATE_GLOBAL,
    BMC_VERTEX_GROUP_STATE_PROP,
)
from .core.blender_ops import (
    apply_group_remaps_to_meshes,
    annotate_alias_items_with_mesh_proximity,
    build_seam_filtered_aliases_from_manifest,
    infer_local_bone_count_from_mesh,
    merge_duplicate_alias_weights,
    resolve_mesh_identity,
)
from .core.mapping_payload import (
    apply_mapping_payload_to_scene,
    build_mapping_payload,
    load_mapping_payload_from_scene,
    serialize_lod_mapping_items,
    serialize_alias_items,
    store_mapping_payload_on_scene,
)
from .core.presets import delete_preset, load_preset, save_preset
from .core.export_prepare import prepare_export_collection, regenerate_bonestore_runtime_files
from .core.export_package import build_export_plan
from .core.frameanalysis import infer_mesh_identity_from_name
from .core.io import read_json, write_json
from .core.import_candidates import import_selected_candidates
from .core.lod_analyze import analyze_lod_for_manifest, review_lod_global_pool_coverage
from .core.lod_fallback import apply_lod_fallbacks_to_manifest, preview_lod_fallbacks_for_export
from .core.lod_runtime import build_lod_mapping, scan_lod_targets_and_generate_manifest
from .core.main_analyze import build_bone_pool_order, write_main_analysis_manifest
from .core.seam_matcher import apply_seam_mapping, build_and_apply_seam_mapping, build_seam_mapping
from .core.workflow import scan_targets_and_generate_outputs
from .core.models import TargetObjectSpec

DEFAULT_TARGET_COLLECTION_NAME = "BMC Bone Palette Targets"
DEFAULT_EXPORT_COLLECTION_NAME = "BMC Export Sources"
_EXPORT_NORMALIZE_GUARD = False
_EXPORT_REGION_NAME_RE = re.compile(r"(?P<hash>[0-9A-Fa-f]{8})[-_](?P<count>\d+)[-_](?P<first>\d+)")


def _ensure_target_collection(context):
    scene = context.scene
    collection = scene.bmc_target_collection
    if collection is None:
        collection = bpy.data.collections.get(DEFAULT_TARGET_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(DEFAULT_TARGET_COLLECTION_NAME)
        scene.collection.children.link(collection)
    scene.bmc_target_collection = collection
    return collection


def _ensure_export_collection(context):
    scene = context.scene
    collection = scene.bmc_export_collection
    if collection is None:
        collection = bpy.data.collections.get(DEFAULT_EXPORT_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(DEFAULT_EXPORT_COLLECTION_NAME)
        scene.collection.children.link(collection)
    scene.bmc_export_collection = collection
    return collection


def _export_region_collection_name(
    ib_hash: str,
    match_index_count: int = 0,
    match_first_index: int = 0,
    *,
    bone_capture_available: bool = True,
    lod_match_excluded: bool = False,
) -> str:
    region_name = f"{ib_hash.lower()}-{int(match_index_count)}-{int(match_first_index)}"
    if not bone_capture_available:
        region_name += "-NO_CAPTURE_BONES"
    if lod_match_excluded:
        region_name += "-NO_LOD_DYNAMIC_VB0"
    return region_name


def _ensure_export_region_collection(
    parent_collection,
    ib_hash: str,
    match_index_count: int = 0,
    match_first_index: int = 0,
    *,
    bone_capture_available: bool = True,
    lod_match_excluded: bool = False,
):
    region_name = _export_region_collection_name(
        ib_hash,
        match_index_count,
        match_first_index,
        bone_capture_available=bone_capture_available,
        lod_match_excluded=lod_match_excluded,
    )
    for child in parent_collection.children:
        identity = _resolve_export_region_collection_identity(child)
        if identity == (str(ib_hash or "").lower(), int(match_index_count), int(match_first_index)):
            if child.name != region_name and bpy.data.collections.get(region_name) is None:
                child.name = region_name
            child["bmc_bone_capture_available"] = bool(bone_capture_available)
            child["bmc_lod_match_excluded"] = bool(lod_match_excluded)
            return child
    collection = bpy.data.collections.get(region_name)
    if collection is None:
        collection = bpy.data.collections.new(region_name)
    if all(child.name != collection.name for child in parent_collection.children):
        parent_collection.children.link(collection)
    collection["bmc_bone_capture_available"] = bool(bone_capture_available)
    collection["bmc_lod_match_excluded"] = bool(lod_match_excluded)
    return collection


def _ensure_export_region_collections_from_targets(context, parent_collection) -> int:
    created_count = 0
    seen_keys: set[tuple[str, int, int]] = set()
    scene = context.scene

    for item in scene.bmc_target_items:
        if not (item.enabled and item.ib_hash):
            continue
        match_index_count = _resolve_target_match_index_count(scene, item.ib_hash, item.object_name, int(item.match_index_count))
        if match_index_count <= 0:
            continue
        key = (item.ib_hash.lower(), int(match_index_count), 0)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if _ensure_export_region_collection_if_missing(parent_collection, key[0], key[1], key[2]):
            created_count += 1

    if seen_keys:
        return created_count

    for mesh_obj in context.selected_objects:
        if mesh_obj.type != "MESH":
            continue
        identity = resolve_mesh_identity(mesh_obj)
        if identity is None:
            continue
        match_index_count = _resolve_target_match_index_count(scene, identity[0], mesh_obj.name, int(identity[1]))
        if match_index_count <= 0:
            continue
        first_index = int(mesh_obj.get("bmc_match_first_index", 0) or 0)
        key = (identity[0].lower(), int(match_index_count), first_index)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if _ensure_export_region_collection_if_missing(parent_collection, key[0], key[1], key[2]):
            created_count += 1

    return created_count


def _ensure_export_region_collection_if_missing(
    parent_collection,
    ib_hash: str,
    match_index_count: int,
    match_first_index: int = 0,
    *,
    bone_capture_available: bool = True,
    lod_match_excluded: bool = False,
) -> bool:
    identity = (str(ib_hash or "").lower(), int(match_index_count), int(match_first_index))
    existed_under_parent = any(_resolve_export_region_collection_identity(child) == identity for child in parent_collection.children)
    _ensure_export_region_collection(
        parent_collection,
        ib_hash,
        match_index_count,
        match_first_index,
        bone_capture_available=bone_capture_available,
        lod_match_excluded=lod_match_excluded,
    )
    return not existed_under_parent


def _resolve_export_region_collection_identity(collection) -> tuple[str, int, int] | None:
    match = _EXPORT_REGION_NAME_RE.search(str(getattr(collection, "name", "") or ""))
    if not match:
        return None
    return match.group("hash").lower(), int(match.group("count")), int(match.group("first"))


def _link_object_to_collection(mesh_obj, collection) -> None:
    if any(obj.name == mesh_obj.name for obj in collection.objects):
        return
    collection.objects.link(mesh_obj)


def _materialize_auto_export_part_collections(export_collection) -> dict[str, object]:
    plan = build_export_plan(
        export_collection,
        _collect_weighted_numeric_vertex_groups_for_export,
        max_bones_per_part=BI4_MAX_BONE_COUNT,
    )
    generated_region_keys = {part.region.key for part in plan.parts if bool(part.generated)}
    if not generated_region_keys:
        return {"created": 0, "linked": 0, "unlinked": 0, "regions": 0, "parts": 0, "warnings": []}

    region_collections = {
        identity: child
        for child in export_collection.children
        if (identity := _resolve_export_region_collection_identity(child)) is not None
    }
    created_count = 0
    linked_count = 0
    unlinked_count = 0
    materialized_parts = 0

    for part in plan.parts:
        if part.region.key not in generated_region_keys:
            continue
        region_collection = region_collections.get(
            (part.region.ib_hash, int(part.region.match_index_count), int(part.region.match_first_index))
        )
        if region_collection is None:
            continue
        part_collection, created = _ensure_export_part_collection(region_collection, part.part_name)
        if created:
            created_count += 1
        materialized_parts += 1
        for mesh_obj in part.mesh_objects:
            if _link_object_to_collection_counted(mesh_obj, part_collection):
                linked_count += 1
            unlinked_count += _unlink_object_from_other_region_part_locations(region_collection, mesh_obj, part_collection)

    return {
        "created": created_count,
        "linked": linked_count,
        "unlinked": unlinked_count,
        "regions": len(generated_region_keys),
        "parts": materialized_parts,
        "warnings": list(plan.warnings),
    }


def _collect_weighted_numeric_vertex_groups_for_export(mesh_obj) -> set[int]:
    group_index_to_global = {}
    for vertex_group in getattr(mesh_obj, "vertex_groups", []) or []:
        raw_name = str(getattr(vertex_group, "name", "") or "").strip()
        if not raw_name.isdigit():
            continue
        group_index_to_global[int(vertex_group.index)] = int(raw_name)

    used_groups: set[int] = set()
    for vertex in getattr(getattr(mesh_obj, "data", None), "vertices", []) or []:
        for group_element in getattr(vertex, "groups", []) or []:
            global_group = group_index_to_global.get(int(getattr(group_element, "group", -1)))
            if global_group is None:
                continue
            if float(getattr(group_element, "weight", 0.0)) <= 0.0:
                continue
            used_groups.add(int(global_group))
    return used_groups


def _ensure_export_part_collection(region_collection, part_name: str):
    requested_name = str(part_name or "part00")
    requested_index = _parse_export_part_index(requested_name)
    for child in region_collection.children:
        if _parse_export_part_index(child.name) == requested_index:
            return child, False
    collection = bpy.data.collections.new(requested_name)
    region_collection.children.link(collection)
    return collection, True


def _parse_export_part_index(collection_name: str) -> int | None:
    match = re.match(r"^part(?P<index>\d+)(?:\D.*)?$", str(collection_name or ""), re.IGNORECASE)
    if not match:
        return None
    return int(match.group("index"))


def _link_object_to_collection_counted(mesh_obj, collection) -> bool:
    if any(obj.name == mesh_obj.name for obj in collection.objects):
        return False
    collection.objects.link(mesh_obj)
    return True


def _unlink_object_from_other_region_part_locations(region_collection, mesh_obj, target_part_collection) -> int:
    region_subtree_ids = {collection.as_pointer() for collection in _iter_collection_subtree(region_collection)}
    removed = 0
    for collection in tuple(getattr(mesh_obj, "users_collection", []) or []):
        if collection == target_part_collection:
            continue
        if collection.as_pointer() not in region_subtree_ids:
            continue
        if any(obj.name == mesh_obj.name for obj in collection.objects):
            collection.objects.unlink(mesh_obj)
            removed += 1
    return removed


def _resolve_target_match_index_count(scene, ib_hash: str, object_name: str = "", fallback: int = 0) -> int:
    normalized_hash = str(ib_hash or "").lower()
    normalized_name = str(object_name or "")
    if not normalized_hash:
        return int(fallback)

    for item in scene.bmc_target_items:
        if str(getattr(item, "ib_hash", "") or "").lower() != normalized_hash:
            continue
        item_object = getattr(item, "object_ref", None)
        item_name = item_object.name if item_object is not None else str(getattr(item, "object_name", "") or "")
        if normalized_name and item_name and item_name != normalized_name:
            continue
        item_count = int(getattr(item, "match_index_count", 0))
        if item_count > 0:
            return item_count

    manifest_path = bpy.path.abspath(str(getattr(scene, "bmc_manifest_path", "") or ""))
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as file_handle:
                manifest = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            manifest = {}
        fallback_count = 0
        identity_records = list(manifest.get("part_records", []) or [])
        identity_records.extend(manifest.get("candidate_ibs", []) or [])
        for part_record in identity_records:
            if str(part_record.get("ib_hash", "")).lower() != normalized_hash:
                continue
            count = int(part_record.get("match_index_count", 0))
            if count <= 0:
                continue
            if normalized_name and str(part_record.get("object_name", "")) == normalized_name:
                return count
            if fallback_count <= 0:
                fallback_count = count
        if fallback_count > 0:
            return fallback_count

    return max(0, int(fallback))


def _sync_target_match_counts_from_manifest(scene, manifest_path: str, collection_name: str = "bmc_target_items") -> None:
    normalized_manifest_path = bpy.path.abspath(str(manifest_path or ""))
    if not normalized_manifest_path or not os.path.exists(normalized_manifest_path):
        return
    try:
        with open(normalized_manifest_path, "r", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return

    records_by_name: dict[str, dict] = {}
    records_by_hash: dict[str, dict] = {}
    identity_records = list(manifest.get("part_records", []) or [])
    identity_records.extend(manifest.get("candidate_ibs", []) or [])
    for part_record in identity_records:
        ib_hash = str(part_record.get("ib_hash", "")).lower()
        if not ib_hash:
            continue
        object_name = str(part_record.get("object_name", ""))
        if object_name:
            records_by_name[object_name] = part_record
        records_by_hash.setdefault(ib_hash, part_record)

    items = getattr(scene, collection_name, None)
    if items is None:
        return
    for item in items:
        ib_hash = str(getattr(item, "ib_hash", "") or "").lower()
        if not ib_hash:
            continue
        mesh_obj = _resolve_lod_target_item_object(scene, item) if collection_name == "bmc_lod_target_items" else _resolve_target_item_object(scene, item)
        object_name = mesh_obj.name if mesh_obj is not None else str(getattr(item, "object_name", "") or "")
        part_record = records_by_name.get(object_name) or records_by_hash.get(ib_hash)
        if not part_record:
            continue
        item.match_index_count = int(part_record.get("match_index_count", 0))


def _resolve_target_item_object(scene, item):
    mesh_obj = getattr(item, "object_ref", None)
    if mesh_obj is not None and mesh_obj.type == "MESH":
        return mesh_obj

    object_name = str(getattr(item, "object_name", "") or "")
    if not object_name:
        return None
    mesh_obj = scene.objects.get(object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        return None
    item.object_ref = mesh_obj
    return mesh_obj


def _refresh_target_item(scene, item):
    mesh_obj = _resolve_target_item_object(scene, item)
    if mesh_obj is None:
        return None

    item.object_name = mesh_obj.name
    if not item.ib_hash:
        identity = resolve_mesh_identity(mesh_obj)
        if identity is not None:
            item.ib_hash = identity[0]
            item.match_index_count = 0
        else:
            item.ib_hash = ""
            item.match_index_count = 0
    if int(getattr(item, "local_bone_count", 0)) <= 0:
        try:
            item.local_bone_count = int(infer_local_bone_count_from_mesh(mesh_obj))
        except ValueError:
            pass
    item.autodetected = bool(getattr(mesh_obj, "merge_ib_autodetected", True))
    return mesh_obj


def _refresh_target_items(scene) -> int:
    refreshed_count = 0
    for item in scene.bmc_target_items:
        if _refresh_target_item(scene, item) is not None:
            refreshed_count += 1
    return refreshed_count


def _iter_collection_subtree(root_collection):
    yield root_collection
    for child_collection in root_collection.children:
        yield from _iter_collection_subtree(child_collection)


def _iter_mesh_objects_in_collection_subtree(root_collection):
    seen_names: set[str] = set()
    for collection in _iter_collection_subtree(root_collection):
        for mesh_obj in collection.objects:
            if mesh_obj.type != "MESH" or mesh_obj.name in seen_names:
                continue
            seen_names.add(mesh_obj.name)
            yield mesh_obj


def _normalize_export_collection_membership(scene) -> dict[str, int]:
    export_collection = scene.bmc_export_collection
    if export_collection is None:
        return {"moved": 0, "created": 0, "skipped": 0}

    subtree_collections = tuple(_iter_collection_subtree(export_collection))
    subtree_ids = {collection.as_pointer() for collection in subtree_collections}
    existing_region_names = {child.name for child in export_collection.children}
    moved_count = 0
    skipped_count = 0
    created_count = 0

    for mesh_obj in tuple(_iter_mesh_objects_in_collection_subtree(export_collection)):
        identity = resolve_mesh_identity(mesh_obj)
        if identity is None:
            skipped_count += 1
            continue

        match_index_count = _resolve_target_match_index_count(scene, identity[0], mesh_obj.name, int(identity[1]))
        if match_index_count <= 0:
            skipped_count += 1
            continue
        first_index = int(mesh_obj.get("bmc_match_first_index", 0) or 0)
        region_name = f"{identity[0].lower()}-{int(match_index_count)}-{first_index}"
        if region_name not in existing_region_names:
            created_count += 1
            existing_region_names.add(region_name)
        region_collection = _ensure_export_region_collection(export_collection, identity[0], match_index_count, first_index)

        moved_here = False
        if all(obj.name != mesh_obj.name for obj in region_collection.objects):
            region_collection.objects.link(mesh_obj)
            moved_here = True

        for collection in tuple(mesh_obj.users_collection):
            if collection.as_pointer() not in subtree_ids or collection == region_collection:
                continue
            if any(obj.name == mesh_obj.name for obj in collection.objects):
                collection.objects.unlink(mesh_obj)
                moved_here = True

        if moved_here:
            moved_count += 1

    return {
        "moved": moved_count,
        "created": created_count,
        "skipped": skipped_count,
    }


def _should_refresh_for_depsgraph_update(scene, depsgraph) -> bool:
    if scene is None:
        return False
    if not scene.bmc_target_items and scene.bmc_target_collection is None:
        return False

    target_object_ids: set[int] = set()
    for item in scene.bmc_target_items:
        mesh_obj = getattr(item, "object_ref", None)
        if mesh_obj is None and getattr(item, "object_name", ""):
            mesh_obj = scene.objects.get(item.object_name)
        if mesh_obj is not None and mesh_obj.type == "MESH":
            target_object_ids.add(mesh_obj.as_pointer())

    for update in depsgraph.updates:
        update_id = getattr(update, "id", None)
        if isinstance(update_id, bpy.types.Object):
            pointer = update_id.as_pointer()
            if pointer in target_object_ids:
                return True
            continue
        if isinstance(update_id, bpy.types.Collection):
            if scene.bmc_target_collection is not None and update_id == scene.bmc_target_collection:
                return True
    return False


@persistent
def _bmc_depsgraph_update_post(scene, depsgraph):
    global _EXPORT_NORMALIZE_GUARD

    if _EXPORT_NORMALIZE_GUARD or not _should_refresh_for_depsgraph_update(scene, depsgraph):
        return

    _EXPORT_NORMALIZE_GUARD = True
    try:
        _refresh_target_items(scene)
    finally:
        _EXPORT_NORMALIZE_GUARD = False


def register_runtime_handlers():
    if _bmc_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_bmc_depsgraph_update_post)


def unregister_runtime_handlers():
    if _bmc_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_bmc_depsgraph_update_post)


def _add_mesh_object_to_target_list(scene, mesh_obj, existing_names: set[str]) -> bool:
    identity = resolve_mesh_identity(mesh_obj)
    if identity is None:
        return False
    try:
        local_bone_count = infer_local_bone_count_from_mesh(mesh_obj)
    except ValueError:
        return False
    if mesh_obj.name in existing_names:
        return False
    item = scene.bmc_target_items.add()
    item.object_name = mesh_obj.name
    item.object_ref = mesh_obj
    item.ib_hash = identity[0]
    item.match_index_count = 0
    item.local_bone_count = int(local_bone_count)
    item.autodetected = bool(getattr(mesh_obj, "merge_ib_autodetected", True))
    item.enabled = True
    existing_names.add(mesh_obj.name)
    return True


def _resolve_lod_target_item_object(scene, item):
    mesh_obj = getattr(item, "object_ref", None)
    if mesh_obj is not None and mesh_obj.type == "MESH":
        return mesh_obj

    object_name = str(getattr(item, "object_name", "") or "")
    if not object_name:
        return None
    mesh_obj = scene.objects.get(object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        return None
    item.object_ref = mesh_obj
    return mesh_obj


def _refresh_lod_target_item(scene, item):
    mesh_obj = _resolve_lod_target_item_object(scene, item)
    if mesh_obj is None:
        return None
    item.object_name = mesh_obj.name
    if not item.ib_hash:
        identity = resolve_mesh_identity(mesh_obj)
        if identity is not None:
            item.ib_hash = identity[0]
            item.match_index_count = 0
        else:
            item.ib_hash = ""
            item.match_index_count = 0
    if int(getattr(item, "local_bone_count", 0)) <= 0:
        try:
            item.local_bone_count = int(infer_local_bone_count_from_mesh(mesh_obj))
        except ValueError:
            pass
    return mesh_obj


def _add_mesh_object_to_lod_target_list(scene, mesh_obj, existing_names: set[str]) -> bool:
    identity = resolve_mesh_identity(mesh_obj)
    if identity is None:
        return False
    try:
        local_bone_count = infer_local_bone_count_from_mesh(mesh_obj)
    except ValueError:
        return False
    if mesh_obj.name in existing_names:
        return False
    item = scene.bmc_lod_target_items.add()
    item.object_name = mesh_obj.name
    item.object_ref = mesh_obj
    item.ib_hash = identity[0]
    item.match_index_count = 0
    item.local_bone_count = int(local_bone_count)
    item.enabled = True
    existing_names.add(mesh_obj.name)
    return True


def _enabled_lod_target_specs(context) -> list[TargetObjectSpec]:
    scene = context.scene
    target_specs: list[TargetObjectSpec] = []
    seen_hashes: set[str] = set()
    for item in scene.bmc_lod_target_items:
        if not (item.enabled and item.ib_hash and item.object_name):
            continue
        normalized_hash = str(item.ib_hash).lower()
        if normalized_hash in seen_hashes:
            continue
        mesh_obj = _refresh_lod_target_item(scene, item)
        if mesh_obj is None:
            raise ValueError(f"{item.object_name}: LOD mesh object not found in current scene")
        local_bone_count = int(getattr(item, "local_bone_count", 0))
        if local_bone_count <= 0:
            raise ValueError(f"{mesh_obj.name}: frozen LOD local bone count is missing")
        target_specs.append(
            TargetObjectSpec(
                object_name=mesh_obj.name,
                ib_hash=normalized_hash,
                match_index_count=-1,
                local_bone_count=local_bone_count,
            )
        )
        seen_hashes.add(normalized_hash)
    return target_specs


def _enabled_lod_target_mesh_objects(scene) -> list[object]:
    mesh_objects: list[object] = []
    for item in scene.bmc_lod_target_items:
        if not item.enabled:
            continue
        mesh_obj = _resolve_lod_target_item_object(scene, item)
        if mesh_obj is None or mesh_obj.type != "MESH":
            continue
        mesh_objects.append(mesh_obj)
    return mesh_objects


def _select_lod_object_remap_entries(scene, payload: dict) -> tuple[list[object], list[dict], list[str]]:
    lod_variant = dict(payload.get("lod_variant", {}) or {})
    entries_by_identity: dict[str, list[dict]] = {}
    for entry in lod_variant.get("object_remaps", []) or []:
        key = str(entry.get("ib_hash", "")).lower()
        entries_by_identity.setdefault(key, []).append(entry)

    matched_meshes: list[object] = []
    selected_entries: list[dict] = []
    skipped_messages: list[str] = []
    for item in scene.bmc_lod_target_items:
        if not item.enabled:
            continue
        mesh_obj = _resolve_lod_target_item_object(scene, item)
        if mesh_obj is None or mesh_obj.type != "MESH":
            skipped_messages.append(f"{item.object_name}: LOD target mesh object not found in scene")
            continue
        identity = str(item.ib_hash or "").lower()
        remap_entries = entries_by_identity.get(identity, [])
        if not remap_entries:
            skipped_messages.append(f"{mesh_obj.name}: no LOD mapping entry for {identity}")
            continue
        chosen_entry = _choose_identity_remap_entry(remap_entries, mesh_obj.name)
        if chosen_entry is None:
            skipped_messages.append(f"{mesh_obj.name}: multiple LOD mapping entries exist for {identity}")
            continue
        matched_meshes.append(mesh_obj)
        selected_entries.append(chosen_entry)
    return matched_meshes, selected_entries, skipped_messages


def _replace_lod_mapping_items(scene, mapping_entries: list[dict]) -> None:
    scene.bmc_lod_mapping_items.clear()
    for mapping_entry in mapping_entries:
        item = scene.bmc_lod_mapping_items.add()
        item.enabled = bool(mapping_entry.get("enabled", True))
        item.canonical_global_bone = int(mapping_entry.get("canonical_global_bone", 0))
        item.mapped_lod_global_bone = int(mapping_entry.get("mapped_lod_global_bone", mapping_entry.get("lod_local_bone", -1)))
        item.lod_record_key = str(mapping_entry.get("lod_record_key", "") or "")
        item.lod_local_bone = int(mapping_entry.get("lod_local_bone", -1))
        item.votes = int(mapping_entry.get("votes", 0) or 0)
        item.average_distance = float(mapping_entry.get("average_distance", 0.0) or 0.0)
        item.status = str(mapping_entry.get("status", ""))
        item.score = float(mapping_entry.get("score", 0.0))
        item.note = str(mapping_entry.get("note", "") or _lod_mapping_note(mapping_entry))
    scene.bmc_lod_mapping_index = min(scene.bmc_lod_mapping_index, max(0, len(scene.bmc_lod_mapping_items) - 1))


def _replace_lod_fallback_items(scene, fallback_entries: list[dict]) -> None:
    scene.bmc_lod_fallback_items.clear()
    for fallback in fallback_entries:
        item = scene.bmc_lod_fallback_items.add()
        item.enabled = bool(fallback.get("enabled", True))
        item.canonical_global_bone = int(fallback.get("canonical_global_bone", 0) or 0)
        item.donor_global_bone = int(fallback.get("donor_global_bone", -1) or -1)
        item.lod_record_key = str(fallback.get("lod_record_key", "") or "")
        item.lod_local_bone = int(fallback.get("lod_local_bone", -1) or -1)
        item.method = str(fallback.get("method", "") or fallback.get("fallback_method", "") or "")
        item.confidence = float(fallback.get("confidence", fallback.get("fallback_confidence", 0.0)) or 0.0)
        item.status = str(fallback.get("status", "") or "")
        item.note = str(fallback.get("note", "") or "")
    scene.bmc_lod_fallback_index = min(scene.bmc_lod_fallback_index, max(0, len(scene.bmc_lod_fallback_items) - 1))


def _store_lod_fallback_preview_on_scene(scene, preview: dict, *, applied: bool = False) -> None:
    summary = dict(preview.get("summary", {}) or {})
    fallback_count = int(summary.get("fallback_count", 0) or 0)
    unresolved_count = int(summary.get("unresolved_count", 0) or 0)
    unmatched_used_count = int(summary.get("unmatched_used_count", 0) or 0)
    unused_unmatched_count = int(summary.get("unused_unmatched_count", 0) or 0)
    prefix = "Applied" if applied else "Preview"
    scene.bmc_lod_fallback_summary = (
        f"{prefix}: used unmatched {unmatched_used_count}, fallback {fallback_count}, "
        f"unresolved {unresolved_count}, unused unmatched {unused_unmatched_count}"
    )
    scene.bmc_lod_fallback_warning = ""
    if unresolved_count:
        scene.bmc_lod_fallback_warning = "Some used unmatched bones still have no donor; export remains blocked for LOD."
    elif fallback_count and not applied:
        scene.bmc_lod_fallback_warning = "Fallbacks are inherited donor bones; inspect before applying."
    _replace_lod_fallback_items(scene, list(preview.get("fallbacks", []) or []) + list(preview.get("unresolved", []) or []))


def _lod_export_blocking_preview(scene, manifest: dict) -> dict:
    export_collection = scene.bmc_export_collection
    preview = preview_lod_fallbacks_for_export(export_collection, manifest, use_export_plan=True)
    _store_lod_fallback_preview_on_scene(scene, preview, applied=False)
    return preview


def _raise_if_lod_unmatched_used_by_export(scene, manifest: dict) -> None:
    if not manifest.get("lod_mapping"):
        return
    preview = _lod_export_blocking_preview(scene, manifest)
    unmatched_used = list(preview.get("unmatched_used_global_bones", []) or [])
    if not unmatched_used:
        return
    shown = ", ".join(f"G{int(value)}" for value in unmatched_used[:12])
    if len(unmatched_used) > 12:
        shown += ", ..."
    raise ValueError(
        f"LOD has {len(unmatched_used)} unmatched global bone(s) used by the export palette: {shown}. "
        "Open LOD Repair, preview carefully, then apply fallbacks if the inherited result is acceptable."
    )


def _lod_mapping_note(mapping_entry: dict) -> str:
    status = str(mapping_entry.get("status", ""))
    if status == "matched":
        return (
            f"{mapping_entry.get('lod_record_key', '')}: local {int(mapping_entry.get('lod_local_bone', -1))}, "
            f"votes {int(mapping_entry.get('votes', 0) or 0)}, score {float(mapping_entry.get('score', 0.0) or 0.0):.3f}"
        )
    if status == "ignored_lod_match_excluded":
        return "ignored: dynamic/pre-skinned vb0 source"
    return "unmatched"


def _canonical_mesh_entries_for_lod_build(scene, payload: dict) -> list[tuple[object, dict]]:
    matched_meshes, selected_entries, _skipped_messages = _select_object_remap_entries_for_targets(scene, payload)
    return list(zip(matched_meshes, selected_entries))


def _lod_mesh_entries_for_build(scene, payload: dict) -> list[tuple[object, dict]]:
    matched_meshes, selected_entries, _skipped_messages = _select_lod_object_remap_entries(scene, payload)
    return list(zip(matched_meshes, selected_entries))


def _sync_scene_payload(scene) -> dict:
    payload = build_mapping_payload(scene, manifest_payload=_read_manifest_payload(scene.bmc_manifest_path), base_payload=load_mapping_payload_from_scene(scene))
    store_mapping_payload_on_scene(scene, payload)
    return payload


def _enabled_target_specs(context) -> list[TargetObjectSpec]:
    scene = context.scene
    target_specs: list[TargetObjectSpec] = []
    seen_hashes: set[str] = set()
    for item in scene.bmc_target_items:
        if not (item.enabled and item.ib_hash and item.object_name):
            continue
        normalized_hash = str(item.ib_hash).lower()
        if normalized_hash in seen_hashes:
            continue
        mesh_obj = _refresh_target_item(scene, item)
        if mesh_obj is None:
            raise ValueError(f"{item.object_name}: mesh object not found in current scene")
        display_name = mesh_obj.name
        manifest_bone_count = _lookup_capture_bone_count_from_manifest(
            scene,
            item.ib_hash.lower(),
            -1,
            display_name,
        )
        local_bone_count = int(getattr(item, "local_bone_count", 0))
        if local_bone_count <= 0:
            if manifest_bone_count is None:
                raise ValueError(
                    f"{display_name}: frozen local bone count is missing. "
                    "Use Refresh Target Identity once before Scan."
                )
            local_bone_count = int(manifest_bone_count)
            item.local_bone_count = local_bone_count
        if local_bone_count > BI4_MAX_BONE_COUNT and manifest_bone_count is not None:
            local_bone_count = manifest_bone_count
        if local_bone_count > BI4_MAX_BONE_COUNT:
            raise ValueError(
                f"{display_name}: local bone count {local_bone_count} exceeds BI4 limit {BI4_MAX_BONE_COUNT}; "
                "this workflow requires each final export part to stay within 256 bones"
            )
        target_specs.append(
            TargetObjectSpec(
                object_name=display_name,
                ib_hash=normalized_hash,
                match_index_count=-1,
                local_bone_count=local_bone_count,
            )
        )
        seen_hashes.add(normalized_hash)
    return target_specs


def _lookup_capture_bone_count_from_manifest(scene, ib_hash: str, match_index_count: int, object_name: str) -> int | None:
    manifest_path = bpy.path.abspath(str(getattr(scene, "bmc_manifest_path", "") or ""))
    if not manifest_path or not os.path.exists(manifest_path):
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return None

    normalized_hash = str(ib_hash).lower()
    fallback_count = None
    for part_record in manifest.get("part_records", []):
        if str(part_record.get("ib_hash", "")).lower() != normalized_hash:
            continue
        bone_count = int(part_record.get("capture_bone_count", part_record.get("bone_count", 0)))
        if bone_count <= 0:
            continue
        if str(part_record.get("object_name", "")) == str(object_name):
            return bone_count
        if fallback_count is None:
            fallback_count = bone_count
    return fallback_count


def _enabled_target_names(scene) -> list[str]:
    target_names: list[str] = []
    for item in scene.bmc_target_items:
        if not item.enabled:
            continue
        mesh_obj = getattr(item, "object_ref", None)
        if mesh_obj is None and getattr(item, "object_name", ""):
            mesh_obj = scene.objects.get(item.object_name)
        if mesh_obj is None:
            continue
        if mesh_obj.type != "MESH":
            continue
        target_names.append(mesh_obj.name)
    return target_names


def _enabled_target_mesh_objects(scene) -> list[object]:
    mesh_objects: list[object] = []
    for item in scene.bmc_target_items:
        if not item.enabled:
            continue
        mesh_obj = _resolve_target_item_object(scene, item)
        if mesh_obj is None or mesh_obj.type != "MESH":
            continue
        mesh_objects.append(mesh_obj)
    return mesh_objects


def _enabled_alias_payload(scene) -> list[dict]:
    return serialize_alias_items(scene)


def _replace_alias_items_from_manifest(scene, manifest: dict):
    scene.bmc_alias_items.clear()
    for alias in manifest.get("bone_aliases", []):
        item = scene.bmc_alias_items.add()
        item.enabled = True
        item.src_draw_index = int(alias.get("src_draw_index", 0))
        item.src_object_name = str(alias.get("src_object_name", ""))
        item.src_ib_hash = str(alias.get("src_ib_hash", ""))
        item.src_local_bone = int(alias.get("src_local_bone", 0))
        item.src_global_bone = int(alias.get("src_global_bone", 0))
        item.canonical_draw_index = int(alias.get("canonical_draw_index", 0))
        item.canonical_object_name = str(alias.get("canonical_object_name", ""))
        item.canonical_ib_hash = str(alias.get("canonical_ib_hash", ""))
        item.canonical_local_bone = int(alias.get("canonical_local_bone", 0))
        item.canonical_global_bone = int(alias.get("canonical_global_bone", 0))
        item.confidence = str(alias.get("confidence", ""))
    scene.bmc_alias_index = min(scene.bmc_alias_index, max(0, len(scene.bmc_alias_items) - 1))


def _analyze_same_bone_aliases(context, manifest_path: str, target_object_names: list[str]) -> int:
    normalized_manifest_path = bpy.path.abspath(manifest_path)
    if not normalized_manifest_path or not os.path.exists(normalized_manifest_path):
        raise ValueError("Manifest path does not exist; run Scan and Generate first")

    with open(normalized_manifest_path, "r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)
    manifest["bone_aliases"] = build_seam_filtered_aliases_from_manifest(
        context,
        manifest,
        target_object_names,
    )
    with open(normalized_manifest_path, "w", encoding="utf-8", newline="\n") as file_handle:
        json.dump(manifest, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")
    _replace_alias_items_from_manifest(context.scene, manifest)
    annotate_alias_items_with_mesh_proximity(context.scene, context.scene.bmc_alias_items)
    return len(context.scene.bmc_alias_items)


def _read_manifest_payload(manifest_path: str) -> dict:
    normalized_manifest_path = bpy.path.abspath(str(manifest_path or ""))
    if not normalized_manifest_path or not os.path.exists(normalized_manifest_path):
        return {}
    with open(normalized_manifest_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _parse_hash_list(raw_value: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for token in str(raw_value or "").replace(",", " ").replace(";", " ").split():
        normalized = token.strip().lower()
        if not normalized:
            continue
        if len(normalized) != 8 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError(f"Invalid IB hash: {token}")
        if normalized in seen:
            continue
        values.append(normalized)
        seen.add(normalized)
    return values


def _main_analyze_target_hashes(scene) -> list[str]:
    explicit_hashes = _parse_hash_list(str(getattr(scene, "bmc_target_ib_hash", "") or ""))
    if explicit_hashes:
        return explicit_hashes

    hashes: list[str] = []
    seen: set[str] = set()
    for item in scene.bmc_target_items:
        if not item.enabled:
            continue
        normalized = str(getattr(item, "ib_hash", "") or "").strip().lower()
        if not normalized:
            continue
        if len(normalized) != 8 or any(character not in "0123456789abcdef" for character in normalized):
            continue
        if normalized in seen:
            continue
        hashes.append(normalized)
        seen.add(normalized)
    return hashes


def _replace_candidate_items_from_manifest(scene, manifest: dict) -> None:
    scene.bmc_candidate_items.clear()
    for candidate in manifest.get("candidate_ibs", []):
        item = scene.bmc_candidate_items.add()
        item.enabled = bool(candidate.get("enabled", True))
        item.display_name = str(candidate.get("display_name", "") or "")
        item.ib_hash = str(candidate.get("ib_hash", "") or "")
        item.match_first_index = int(candidate.get("match_first_index", 0))
        item.match_index_count = int(candidate.get("match_index_count", 0))
        item.import_draw_index = int(candidate.get("import_draw_index", -1))
        item.local_bone_count = int(candidate.get("local_bone_count", 0))
        item.draw_count = len(candidate.get("draw_indices", []) or [])
        item.shadow_draw_count = len(candidate.get("shadow_draw_indices", []) or [])
        item.shadow_capture_ready = bool(candidate.get("shadow_capture_ready", bool(candidate.get("shadow_draw_indices"))))
        item.lod_match_excluded = bool(candidate.get("lod_match_excluded", False))
        item.lod_match_excluded_reason = str(candidate.get("lod_match_excluded_reason", "") or "")
        item.status = str(candidate.get("status", "") or ("capture_ready" if item.shadow_capture_ready else "import_only_no_early_shadow"))
        item.manual = False
    scene.bmc_candidate_index = min(scene.bmc_candidate_index, max(0, len(scene.bmc_candidate_items) - 1))


def _candidate_display_name_from_values(ib_hash: str, match_index_count: int, match_first_index: int) -> str:
    return f"{str(ib_hash or '').lower()}-{int(match_index_count)}-{int(match_first_index)}"


def _candidate_key_from_item(item) -> tuple[str, int, int]:
    return (
        str(getattr(item, "ib_hash", "") or "").strip().lower(),
        int(getattr(item, "match_first_index", 0) or 0),
        int(getattr(item, "match_index_count", 0) or 0),
    )


def _candidate_key_from_payload(candidate: dict) -> tuple[str, int, int]:
    return (
        str(candidate.get("ib_hash", "") or "").strip().lower(),
        int(candidate.get("match_first_index", 0) or 0),
        int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0),
    )


def _candidate_item_exists(scene, candidate: dict) -> bool:
    candidate_key = _candidate_key_from_payload(candidate)
    candidate_name = str(candidate.get("display_name", "") or "")
    for item in scene.bmc_candidate_items:
        if candidate_name and str(item.display_name) == candidate_name:
            return True
        if _candidate_key_from_item(item) == candidate_key:
            return True
    return False


def _apply_candidate_payload_to_item(item, candidate: dict, *, manual: bool = False) -> None:
    ib_hash, first_index, index_count = _candidate_key_from_payload(candidate)
    item.enabled = bool(candidate.get("enabled", True))
    item.ib_hash = ib_hash
    item.match_first_index = int(first_index)
    item.match_index_count = int(index_count)
    item.display_name = str(candidate.get("display_name", "") or _candidate_display_name_from_values(ib_hash, index_count, first_index))
    item.import_draw_index = int(candidate.get("import_draw_index", -1))
    item.local_bone_count = int(candidate.get("local_bone_count", 0))
    item.draw_count = len(candidate.get("draw_indices", []) or [])
    item.shadow_draw_count = len(candidate.get("shadow_draw_indices", []) or [])
    item.shadow_capture_ready = bool(candidate.get("shadow_capture_ready", bool(candidate.get("shadow_draw_indices"))))
    item.lod_match_excluded = bool(candidate.get("lod_match_excluded", False))
    item.lod_match_excluded_reason = str(candidate.get("lod_match_excluded_reason", "") or "")
    item.status = str(candidate.get("status", "") or ("capture_ready" if item.shadow_capture_ready else "manual_or_import_only"))
    item.manual = bool(manual)


def _manifest_candidates_by_hash(manifest: dict, ib_hash: str) -> list[dict]:
    normalized_hash = str(ib_hash or "").strip().lower()
    return [
        candidate
        for candidate in manifest.get("candidate_ibs", []) or []
        if str(candidate.get("ib_hash", "") or "").strip().lower() == normalized_hash
    ]


def _find_manifest_candidate_for_item(manifest: dict, item) -> dict | None:
    item_key = _candidate_key_from_item(item)
    item_name = str(getattr(item, "display_name", "") or "")
    for candidate in manifest.get("candidate_ibs", []) or []:
        if item_name and str(candidate.get("display_name", "") or "") == item_name:
            return dict(candidate)
        if _candidate_key_from_payload(candidate) == item_key:
            return dict(candidate)
    return None


def _candidate_payload_from_item(item, manifest: dict | None = None) -> dict:
    candidate = _find_manifest_candidate_for_item(manifest or {}, item) if manifest else None
    if candidate is None:
        ib_hash, first_index, index_count = _candidate_key_from_item(item)
        candidate = {
            "ib_hash": ib_hash,
            "match_first_index": int(first_index),
            "match_index_count": int(index_count),
            "display_name": str(getattr(item, "display_name", "") or _candidate_display_name_from_values(ib_hash, index_count, first_index)),
            "draw_indices": [],
            "shadow_draw_indices": [],
            "local_bone_count": int(getattr(item, "local_bone_count", 0) or 0),
            "import_paths": {"ib": "", "vb": {}, "layout": ""},
        }
    candidate["enabled"] = bool(getattr(item, "enabled", True))
    candidate["shadow_capture_ready"] = bool(getattr(item, "shadow_capture_ready", False))
    candidate["lod_match_excluded"] = bool(getattr(item, "lod_match_excluded", candidate.get("lod_match_excluded", False)))
    candidate["lod_match_excluded_reason"] = str(
        getattr(item, "lod_match_excluded_reason", candidate.get("lod_match_excluded_reason", "")) or ""
    )
    if candidate["lod_match_excluded"] and not candidate["lod_match_excluded_reason"]:
        candidate["lod_match_excluded_reason"] = "manual_lod_match_excluded"
    candidate["status"] = str(getattr(item, "status", "") or candidate.get("status", ""))
    return candidate


def _candidate_payloads_from_ui(scene, manifest: dict | None = None) -> list[dict]:
    payloads = []
    seen: set[tuple[str, int, int]] = set()
    for item in scene.bmc_candidate_items:
        key = _candidate_key_from_item(item)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        payloads.append(_candidate_payload_from_item(item, manifest))
    return payloads


def _add_candidate_from_payload(scene, candidate: dict, *, manual: bool = False) -> bool:
    if _candidate_item_exists(scene, candidate):
        return False
    item = scene.bmc_candidate_items.add()
    _apply_candidate_payload_to_item(item, candidate, manual=manual)
    scene.bmc_candidate_index = len(scene.bmc_candidate_items) - 1
    return True


def _replace_candidate_items_from_payloads(scene, candidates: list[dict], *, previous_state: dict | None = None) -> None:
    previous_state = previous_state or {}
    previous_index = int(getattr(scene, "bmc_candidate_index", 0) or 0)
    scene.bmc_candidate_items.clear()
    for candidate in candidates:
        item = scene.bmc_candidate_items.add()
        manual = bool(candidate.get("_manual", candidate.get("manual", False)))
        _apply_candidate_payload_to_item(item, candidate, manual=manual)
        key = _candidate_key_from_item(item)
        state = previous_state.get(key)
        if state is not None:
            item.enabled = bool(state.get("enabled", item.enabled))
    scene.bmc_candidate_index = min(previous_index, max(0, len(scene.bmc_candidate_items) - 1))


def _candidate_payloads_from_collection(scene, source_collection, manifest: dict) -> tuple[list[dict], dict]:
    payloads: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    stats = {
        "scanned_meshes": 0,
        "recognized_meshes": 0,
        "duplicate_candidates": 0,
    }
    for mesh_obj in _iter_collection_objects_recursive(source_collection):
        if getattr(mesh_obj, "type", "") != "MESH":
            continue
        stats["scanned_meshes"] += 1
        object_candidates = _manifest_candidates_for_object(manifest, mesh_obj)
        if not object_candidates:
            continue
        stats["recognized_meshes"] += 1
        for candidate in object_candidates:
            key = _candidate_key_from_payload(candidate)
            if not key[0] or key in seen:
                stats["duplicate_candidates"] += 1
                continue
            seen.add(key)
            payloads.append(dict(candidate))
    payloads.sort(
        key=lambda candidate: (
            -int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0),
            str(candidate.get("ib_hash", "") or ""),
            int(candidate.get("match_first_index", 0) or 0),
        )
    )
    return payloads, stats


def _iter_collection_objects_recursive(collection):
    if collection is None:
        return
    for mesh_obj in collection.objects:
        yield mesh_obj
    for child in collection.children:
        yield from _iter_collection_objects_recursive(child)


def _object_candidate_identity(mesh_obj) -> tuple[str, int, int] | None:
    ib_hash = str(mesh_obj.get("bmc_source_ib_hash", "") or getattr(mesh_obj, "merge_ib_hash", "") or "").strip().lower()
    if not ib_hash:
        inferred = infer_mesh_identity_from_name(mesh_obj.name)
        if inferred:
            ib_hash = inferred[0].lower()
    if not ib_hash or len(ib_hash) != 8 or any(character not in "0123456789abcdef" for character in ib_hash):
        return None
    first_index = int(mesh_obj.get("bmc_match_first_index", 0) or 0)
    index_count = int(mesh_obj.get("bmc_match_index_count", 0) or getattr(mesh_obj, "merge_match_index_count", 0) or 0)
    return ib_hash, first_index, max(0, index_count)


def _object_source_hash(mesh_obj) -> str:
    identity = _object_candidate_identity(mesh_obj)
    if identity is not None:
        return identity[0]
    inferred = infer_mesh_identity_from_name(mesh_obj.name)
    return inferred[0].lower() if inferred else ""


def _global_pool_generation_id(manifest: dict, bone_pool_order: list[dict] | None = None) -> str:
    existing = str(manifest.get("global_pool_generation", "") or "")
    if existing:
        return existing
    order = bone_pool_order if bone_pool_order is not None else list(manifest.get("bone_pool_order", []) or [])
    signature_parts = [
        (
            str(item.get("ib_hash", "") or "").lower(),
            str(int(item.get("match_index_count", 0) or 0)),
            str(int(item.get("match_first_index", 0) or 0)),
            str(int(item.get("global_bone_base", 0) or 0)),
            str(int(item.get("local_bone_count", 0) or 0)),
            ",".join(str(int(value)) for value in item.get("used_local_bone_indices", []) or []),
            "capture" if bool(item.get("bone_capture_available", item.get("shadow_capture_ready", False))) else "mapping",
            "no_lod" if bool(item.get("lod_match_excluded", False)) else "lod",
        )
        for item in order
    ]
    return "|".join("-".join(part) for part in signature_parts)


def _source_key_from_values(ib_hash: str, match_index_count: int, match_first_index: int = 0) -> str:
    return f"{str(ib_hash or '').lower()}-{int(match_index_count)}-{int(match_first_index)}"


def _object_remaps_from_bone_pool_order(bone_pool_order: list[dict]) -> list[dict]:
    remaps: list[dict] = []
    for record in bone_pool_order:
        ib_hash = str(record.get("ib_hash", "") or "").lower()
        match_index_count = int(record.get("match_index_count", 0) or 0)
        match_first_index = int(record.get("match_first_index", 0) or 0)
        used_local_bone_indices = _used_local_bone_indices_from_pool_record(record)
        local_bone_count = len(used_local_bone_indices)
        global_bone_base = int(record.get("global_bone_base", record.get("capture_store_base", 0)) or 0)
        if not ib_hash or match_index_count <= 0 or local_bone_count <= 0:
            continue
        source_key = _source_key_from_values(ib_hash, match_index_count, match_first_index)
        remaps.append(
            {
                "object_name": "",
                "ib_hash": ib_hash,
                "match_first_index": match_first_index,
                "match_index_count": match_index_count,
                "source_key": source_key,
                "bone_capture_available": bool(record.get("bone_capture_available", record.get("shadow_capture_ready", False))),
                "local_group_to_global_group": {
                    str(local_index): int(global_bone_base + compact_index)
                    for compact_index, local_index in enumerate(used_local_bone_indices)
                },
            }
        )
    return remaps


def _used_local_bone_indices_from_pool_record(record: dict) -> list[int]:
    raw_indices = record.get("used_local_bone_indices")
    if isinstance(raw_indices, (list, tuple)) and raw_indices:
        return sorted({int(value) for value in raw_indices if int(value) >= 0})
    local_bone_count = int(record.get("local_bone_count", 0) or 0)
    return list(range(max(0, local_bone_count)))


def _remap_index_from_manifest(manifest: dict) -> dict[tuple[str, int, int], dict]:
    index: dict[tuple[str, int, int], dict] = {}
    for remap in manifest.get("object_remaps", []) or []:
        ib_hash = str(remap.get("ib_hash", "") or "").lower()
        if not ib_hash:
            continue
        match_first_index = int(remap.get("match_first_index", 0) or 0)
        match_index_count = int(remap.get("match_index_count", 0) or 0)
        index[(ib_hash, match_first_index, match_index_count)] = dict(remap)
    return index


def _remap_entries_for_hash(manifest: dict, ib_hash: str) -> list[dict]:
    normalized_hash = str(ib_hash or "").lower()
    return [
        dict(remap)
        for remap in manifest.get("object_remaps", []) or []
        if str(remap.get("ib_hash", "") or "").lower() == normalized_hash
    ]


def _iter_object_name_hashes(object_name: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"[0-9A-Fa-f]{8}", str(object_name or "")):
        value = match.group(0).lower()
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def _resolve_remap_for_object_name(manifest: dict, object_name: str) -> tuple[dict | None, str]:
    normalized_name = str(object_name or "").lower()
    remaps = [dict(remap) for remap in manifest.get("object_remaps", []) or []]
    for remap in remaps:
        source_key = str(remap.get("source_key", "") or _source_key_from_values(
            str(remap.get("ib_hash", "") or ""),
            int(remap.get("match_index_count", 0) or 0),
            int(remap.get("match_first_index", 0) or 0),
        )).lower()
        if source_key and source_key in normalized_name:
            return remap, ""

    matched_entries: list[dict] = []
    matched_keys: set[str] = set()
    for ib_hash in _iter_object_name_hashes(normalized_name):
        for remap in _remap_entries_for_hash(manifest, ib_hash):
            source_key = str(remap.get("source_key", "") or _source_key_from_values(
                str(remap.get("ib_hash", "") or ""),
                int(remap.get("match_index_count", 0) or 0),
                int(remap.get("match_first_index", 0) or 0),
            ))
            if source_key in matched_keys:
                continue
            matched_entries.append(remap)
            matched_keys.add(source_key)

    if not matched_entries:
        return None, f"{object_name}: no global-pool hash in object name"
    if len(matched_entries) > 1:
        hashes = ", ".join(sorted({str(entry.get("ib_hash", "")).lower() for entry in matched_entries}))
        return None, f"{object_name}: ambiguous global-pool hash {hashes}"
    return matched_entries[0], ""


def _hash_rename_target_meshes(context) -> list[object]:
    selected_meshes = [obj for obj in context.selected_objects if getattr(obj, "type", "") == "MESH"]
    if selected_meshes:
        return selected_meshes
    return [obj for obj in context.scene.objects if getattr(obj, "type", "") == "MESH"]


def _resolve_remap_for_region_identity(
    manifest: dict,
    ib_hash: str,
    match_index_count: int,
    match_first_index: int,
) -> tuple[dict | None, str]:
    matches = [
        remap
        for remap in _remap_entries_for_hash(manifest, ib_hash)
        if int(remap.get("match_index_count", 0) or 0) == int(match_index_count)
        and int(remap.get("match_first_index", 0) or 0) == int(match_first_index)
    ]
    if not matches:
        return None, f"{ib_hash}-{match_index_count}-{match_first_index}: no global-pool mapping"
    source_keys = {
        str(remap.get("source_key", "") or _source_key_from_values(
            str(remap.get("ib_hash", "") or ""),
            int(remap.get("match_index_count", 0) or 0),
            int(remap.get("match_first_index", 0) or 0),
        ))
        for remap in matches
    }
    if len(source_keys) > 1:
        return None, f"{ib_hash}-{match_index_count}-{match_first_index}: ambiguous_source_hash"
    return dict(matches[0]), ""


def _apply_remap_to_meshes(meshes_and_remaps: list[tuple[object, dict]], manifest: dict):
    if not meshes_and_remaps:
        return None
    mesh_objects = [mesh_obj for mesh_obj, _remap in meshes_and_remaps]
    selected_entries = []
    for mesh_obj, remap in meshes_and_remaps:
        remap_entry = dict(remap)
        remap_entry["object_name"] = mesh_obj.name
        selected_entries.append(remap_entry)
    return apply_group_remaps_to_meshes(
        mesh_objects,
        {
            "object_remaps": selected_entries,
            "global_pool_generation": _global_pool_generation_id(manifest),
        },
        identity_resolver=_identity_resolver_from_entries(mesh_objects, selected_entries),
    )


def _collect_mesh_remap_pairs(mesh_objects, remap_resolver) -> tuple[int, list[tuple[object, dict]], list[str]]:
    scanned_objects = 0
    meshes_and_remaps: list[tuple[object, dict]] = []
    skipped: list[str] = []
    for mesh_obj in mesh_objects:
        if getattr(mesh_obj, "type", "") != "MESH":
            continue
        scanned_objects += 1
        remap, error = remap_resolver(mesh_obj)
        if remap is None:
            if error:
                skipped.append(error)
            continue
        meshes_and_remaps.append((mesh_obj, remap))
    return scanned_objects, meshes_and_remaps, skipped


def _apply_global_names_for_meshes(mesh_objects, manifest: dict, remap_resolver) -> tuple[int, int, int, list[str]]:
    scanned_objects, meshes_and_remaps, skipped = _collect_mesh_remap_pairs(mesh_objects, remap_resolver)
    if not meshes_and_remaps:
        return scanned_objects, 0, 0, skipped
    result = _apply_remap_to_meshes(meshes_and_remaps, manifest)
    if result is None:
        return scanned_objects, 0, 0, skipped
    skipped.extend(result.skipped_objects)
    return scanned_objects, int(result.updated_objects), int(result.renamed_groups), skipped


def _revert_global_names_for_meshes(mesh_objects, remap_resolver) -> tuple[int, int, int, list[str]]:
    scanned_objects, meshes_and_remaps, skipped = _collect_mesh_remap_pairs(mesh_objects, remap_resolver)
    updated_objects = 0
    renamed_groups = 0
    for mesh_obj, remap in meshes_and_remaps:
        renamed, rename_error = _rename_global_groups_to_local(mesh_obj, remap)
        if rename_error:
            skipped.append(rename_error)
            continue
        if renamed > 0:
            updated_objects += 1
            renamed_groups += renamed
    return scanned_objects, updated_objects, renamed_groups, skipped


def _candidate_source_meshes_and_remaps(context, manifest: dict) -> list[tuple[object, dict]]:
    remap_index = _remap_index_from_manifest(manifest)
    meshes_and_remaps: list[tuple[object, dict]] = []
    export_collection = context.scene.bmc_export_collection
    if export_collection is None:
        return meshes_and_remaps
    for mesh_obj in _iter_collection_objects_recursive(export_collection):
        if getattr(mesh_obj, "type", "") != "MESH":
            continue
        identity = _object_candidate_identity(mesh_obj)
        if identity is None:
            continue
        remap = remap_index.get(identity)
        if remap is None:
            continue
        meshes_and_remaps.append((mesh_obj, remap))
    return meshes_and_remaps


def _apply_global_names_to_candidate_source_objects(
    context,
    manifest: dict,
    meshes_and_remaps: list[tuple[object, dict]] | None = None,
) -> tuple[int, int, list[str]]:
    if meshes_and_remaps is None:
        meshes_and_remaps = _candidate_source_meshes_and_remaps(context, manifest)
    result = _apply_remap_to_meshes(meshes_and_remaps, manifest)
    if result is None:
        return 0, 0, []
    return int(result.updated_objects), int(result.renamed_groups), list(result.skipped_objects)


def _merge_candidate_source_seam_groups(
    context,
    manifest: dict,
    meshes_and_remaps: list[tuple[object, dict]] | None = None,
):
    if meshes_and_remaps is None:
        meshes_and_remaps = _candidate_source_meshes_and_remaps(context, manifest)
    meshes = [mesh_obj for mesh_obj, _remap in meshes_and_remaps]
    if len(meshes) < 2:
        return None
    return build_and_apply_seam_mapping(meshes)


def _merge_selected_seam_groups(context):
    mesh_objects = [obj for obj in context.selected_objects if getattr(obj, "type", "") == "MESH"]
    if len(mesh_objects) < 2:
        raise ValueError("Select at least two mesh objects")
    return build_and_apply_seam_mapping(mesh_objects)


def _export_region_collections_by_hash(export_collection) -> dict[str, list[object]]:
    by_hash: dict[str, list[object]] = {}
    if export_collection is None:
        return by_hash
    for child in export_collection.children:
        identity = _resolve_export_region_collection_identity(child)
        if identity is None:
            continue
        by_hash.setdefault(identity[0], []).append(child)
    return by_hash


def _child_collection_contains_object(collection, mesh_obj) -> bool:
    return any(obj.name == mesh_obj.name for obj in collection.objects)


def _unlink_object_from_export_sibling_regions(export_collection, mesh_obj, keep_collection) -> None:
    for child in export_collection.children:
        if child == keep_collection:
            continue
        if _resolve_export_region_collection_identity(child) is None:
            continue
        if _child_collection_contains_object(child, mesh_obj):
            child.objects.unlink(mesh_obj)


def _apply_global_names_in_export_collection(context, manifest: dict) -> tuple[int, int, int, list[str]]:
    export_collection = context.scene.bmc_export_collection
    if export_collection is None:
        raise ValueError("Export source collection is not set")
    skipped: list[str] = []
    scanned_objects = 0
    objects_by_name: dict[str, tuple[object, dict, str]] = {}
    conflict_names: set[str] = set()

    for child in export_collection.children:
        identity = _resolve_export_region_collection_identity(child)
        if identity is None:
            continue
        remap, error = _resolve_remap_for_region_identity(manifest, identity[0], identity[1], identity[2])
        if remap is None:
            skipped.append(error)
            continue
        for mesh_obj in _iter_collection_objects_recursive(child):
            if getattr(mesh_obj, "type", "") != "MESH":
                continue
            scanned_objects += 1
            if mesh_obj.name in conflict_names:
                continue
            previous = objects_by_name.get(mesh_obj.name)
            if previous is not None and previous[2] != child.name:
                skipped.append(f"{mesh_obj.name}: multiple export regions")
                objects_by_name.pop(mesh_obj.name, None)
                conflict_names.add(mesh_obj.name)
                continue
            objects_by_name[mesh_obj.name] = (mesh_obj, remap, child.name)

    result = _apply_remap_to_meshes(
        [(mesh_obj, remap) for mesh_obj, remap, _child_name in objects_by_name.values()],
        manifest,
    )
    if result is None:
        return scanned_objects, 0, 0, skipped
    skipped.extend(result.skipped_objects)
    return scanned_objects, int(result.updated_objects), int(result.renamed_groups), skipped


def _apply_global_names_by_object_hash(context, manifest: dict) -> tuple[int, int, int, list[str]]:
    return _apply_global_names_for_meshes(
        _hash_rename_target_meshes(context),
        manifest,
        lambda mesh_obj: _resolve_remap_for_object_name(manifest, mesh_obj.name),
    )


def _revert_global_names_by_object_hash(context, manifest: dict) -> tuple[int, int, int, list[str]]:
    return _revert_global_names_for_meshes(
        _hash_rename_target_meshes(context),
        lambda mesh_obj: _resolve_remap_for_object_name(manifest, mesh_obj.name),
    )


def _rename_global_groups_to_local(mesh_obj, remap: dict) -> tuple[int, str]:
    local_to_global = {
        int(local_index): int(global_index)
        for local_index, global_index in dict(remap.get("local_group_to_global_group", {}) or {}).items()
    }
    if not local_to_global:
        return 0, f"{mesh_obj.name}: no local/global remap"
    global_to_local = {global_index: local_index for local_index, global_index in local_to_global.items()}

    rename_pairs: list[tuple[str, str]] = []
    current_names = {str(vertex_group.name) for vertex_group in mesh_obj.vertex_groups}
    source_names: set[str] = set()
    for vertex_group in mesh_obj.vertex_groups:
        numeric_name = _parse_vertex_group_int(vertex_group.name)
        if numeric_name is None:
            continue
        local_index = global_to_local.get(numeric_name)
        if local_index is None:
            continue
        source_name = str(vertex_group.name)
        target_name = str(int(local_index))
        if source_name == target_name:
            continue
        rename_pairs.append((source_name, target_name))
        source_names.add(source_name)

    if not rename_pairs:
        return 0, ""

    for _source_name, target_name in rename_pairs:
        if target_name in current_names and target_name not in source_names:
            return 0, f"{mesh_obj.name}: target local group {target_name} already exists"

    temp_name_by_source: dict[str, str] = {}
    for source_name, _target_name in rename_pairs:
        vertex_group = mesh_obj.vertex_groups.get(source_name)
        if vertex_group is None:
            continue
        temp_name = f"__bmc_tmp_local__{vertex_group.index}__{source_name}"
        vertex_group.name = temp_name
        temp_name_by_source[source_name] = temp_name

    renamed_count = 0
    for source_name, target_name in rename_pairs:
        temp_name = temp_name_by_source.get(source_name, "")
        if not temp_name:
            continue
        mesh_obj.vertex_groups[temp_name].name = target_name
        renamed_count += 1

    for prop_name in (
        BMC_GLOBAL_REMAP_PROP,
        BMC_GLOBAL_SOURCE_KEY_PROP,
        BMC_GLOBAL_POOL_GENERATION_PROP,
        BMC_VERTEX_GROUP_STATE_PROP,
    ):
        if prop_name in mesh_obj:
            del mesh_obj[prop_name]
    return renamed_count, ""


def _parse_vertex_group_int(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _manifest_candidates_for_object(manifest: dict, mesh_obj) -> list[dict]:
    identity = _object_candidate_identity(mesh_obj)
    if identity is None:
        return []
    ib_hash, first_index, index_count = identity
    hash_matches = _manifest_candidates_by_hash(manifest, ib_hash)
    if index_count > 0:
        exact_matches = [
            candidate
            for candidate in hash_matches
            if int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0) == index_count
            and int(candidate.get("match_first_index", 0) or 0) == first_index
        ]
        if exact_matches:
            return exact_matches
    if hash_matches:
        return hash_matches if index_count <= 0 else [
            candidate
            for candidate in hash_matches
            if int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0) == index_count
        ] or hash_matches
    return [
        {
            "enabled": True,
            "ib_hash": ib_hash,
            "match_first_index": int(first_index),
            "match_index_count": int(index_count),
            "display_name": _candidate_display_name_from_values(ib_hash, index_count, first_index),
            "draw_indices": [],
            "shadow_draw_indices": [],
            "shadow_capture_ready": False,
            "local_bone_count": int(infer_local_bone_count_from_mesh(mesh_obj)),
            "status": "manual_from_collection",
            "import_paths": {"ib": "", "vb": {}, "layout": ""},
        }
    ]


def _update_scene_mapping_payload(scene, manifest_path: str | None = None) -> dict:
    manifest_payload = _read_manifest_payload(manifest_path or scene.bmc_manifest_path)
    base_payload = load_mapping_payload_from_scene(scene)
    payload = build_mapping_payload(scene, manifest_payload=manifest_payload, base_payload=base_payload)
    store_mapping_payload_on_scene(scene, payload)
    return payload


def _resolve_active_mapping_payload(scene) -> dict:
    payload = load_mapping_payload_from_scene(scene)
    if payload.get("object_remaps"):
        return payload
    payload = _update_scene_mapping_payload(scene)
    if payload.get("object_remaps"):
        return payload
    raise ValueError("No mapping payload is loaded. Run Scan first or load a Mapping Preset.")


def _selected_mesh_objects(context) -> list[object]:
    return [mesh_obj for mesh_obj in context.selected_objects if mesh_obj.type == "MESH"]


def _resolve_seam_item_object(scene, item):
    mesh_obj = getattr(item, "object_ref", None)
    if mesh_obj is not None and mesh_obj.type == "MESH":
        return mesh_obj
    object_name = str(getattr(item, "object_name", "") or "")
    if not object_name:
        return None
    mesh_obj = scene.objects.get(object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        return None
    item.object_ref = mesh_obj
    return mesh_obj


def _enabled_seam_mesh_objects(scene, sync_names: bool = False) -> list[object]:
    mesh_objects = []
    seen_names: set[str] = set()
    for item in scene.bmc_seam_match_items:
        if not item.enabled:
            continue
        mesh_obj = _resolve_seam_item_object(scene, item)
        if mesh_obj is None or mesh_obj.name in seen_names:
            continue
        if sync_names:
            item.object_name = mesh_obj.name
        mesh_objects.append(mesh_obj)
        seen_names.add(mesh_obj.name)
    return mesh_objects


def _replace_seam_alias_items(scene, build_result) -> None:
    scene.bmc_seam_alias_items.clear()
    for alias in build_result.aliases:
        item = scene.bmc_seam_alias_items.add()
        item.enabled = True
        item.src_object_name = alias.src_object_name
        item.src_group = int(alias.src_group)
        item.dst_object_name = alias.dst_object_name
        item.dst_group = int(alias.dst_group)
        item.votes = int(alias.votes)
        item.score = float(alias.score)
        item.average_distance = float(alias.average_distance)
        item.average_weight_difference = float(alias.average_weight_difference)
    scene.bmc_seam_alias_index = min(scene.bmc_seam_alias_index, max(0, len(scene.bmc_seam_alias_items) - 1))
    scene.bmc_seam_pair_summary = "\n".join(build_result.pair_summaries)


def _seam_alias_payload(scene) -> list[dict]:
    return [
        {
            "enabled": bool(item.enabled),
            "src_object_name": str(item.src_object_name),
            "src_group": int(item.src_group),
            "dst_object_name": str(item.dst_object_name),
            "dst_group": int(item.dst_group),
            "votes": int(item.votes),
            "score": float(item.score),
            "average_distance": float(item.average_distance),
            "average_weight_difference": float(item.average_weight_difference),
        }
        for item in scene.bmc_seam_alias_items
    ]


def _mesh_identity_from_name(mesh_obj) -> tuple[str, int] | None:
    inferred = infer_mesh_identity_from_name(mesh_obj.name)
    if inferred is None:
        return None
    return inferred[0].lower(), int(inferred[1])


def _choose_identity_remap_entry(remap_entries: list[dict], mesh_name: str) -> dict | None:
    if not remap_entries:
        return None
    exact_matches = [entry for entry in remap_entries if str(entry.get("object_name", "")) == str(mesh_name)]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(remap_entries) == 1:
        return remap_entries[0]
    return None


def _select_object_remap_entries_for_meshes(mesh_objects: list[object], payload: dict) -> tuple[list[object], list[dict], list[str]]:
    entries_by_identity: dict[str, list[dict]] = {}
    for entry in payload.get("object_remaps", []):
        key = str(entry.get("ib_hash", "")).lower()
        entries_by_identity.setdefault(key, []).append(entry)

    matched_meshes: list[object] = []
    selected_entries: list[dict] = []
    skipped_messages: list[str] = []

    for mesh_obj in mesh_objects:
        identity = _mesh_identity_from_name(mesh_obj)
        if identity is None:
            skipped_messages.append(f"{mesh_obj.name}: object name does not contain ib hash")
            continue
        remap_entries = entries_by_identity.get(identity[0], [])
        if not remap_entries:
            skipped_messages.append(f"{mesh_obj.name}: no mapping entry for {identity[0]}")
            continue
        chosen_entry = _choose_identity_remap_entry(remap_entries, mesh_obj.name)
        if chosen_entry is None:
            skipped_messages.append(f"{mesh_obj.name}: multiple mapping entries exist for {identity[0]}")
            continue
        matched_meshes.append(mesh_obj)
        selected_entries.append(chosen_entry)

    return matched_meshes, selected_entries, skipped_messages


def _select_object_remap_entries_for_targets(scene, payload: dict) -> tuple[list[object], list[dict], list[str]]:
    entries_by_identity: dict[str, list[dict]] = {}
    for entry in payload.get("object_remaps", []):
        key = str(entry.get("ib_hash", "")).lower()
        entries_by_identity.setdefault(key, []).append(entry)

    matched_meshes: list[object] = []
    selected_entries: list[dict] = []
    skipped_messages: list[str] = []

    for item in scene.bmc_target_items:
        if not item.enabled:
            continue
        mesh_obj = _resolve_target_item_object(scene, item)
        if mesh_obj is None or mesh_obj.type != "MESH":
            skipped_messages.append(f"{item.object_name}: target mesh object not found in scene")
            continue

        identity = str(item.ib_hash or "").lower()
        if not identity:
            skipped_messages.append(f"{mesh_obj.name}: frozen target identity is incomplete")
            continue

        remap_entries = entries_by_identity.get(identity, [])
        if not remap_entries:
            skipped_messages.append(f"{mesh_obj.name}: no mapping entry for {identity}")
            continue
        chosen_entry = _choose_identity_remap_entry(remap_entries, mesh_obj.name)
        if chosen_entry is None:
            skipped_messages.append(f"{mesh_obj.name}: multiple mapping entries exist for {identity}")
            continue
        matched_meshes.append(mesh_obj)
        selected_entries.append(chosen_entry)

    return matched_meshes, selected_entries, skipped_messages


def _identity_resolver_from_entries(mesh_objects: list[object], selected_entries: list[dict]):
    identity_by_name = {
        mesh_obj.name: (str(entry.get("ib_hash", "")).lower(), 0)
        for mesh_obj, entry in zip(mesh_objects, selected_entries)
    }

    def _resolver(mesh_obj):
        identity = identity_by_name.get(mesh_obj.name)
        if identity is not None and identity[0]:
            return identity
        return resolve_mesh_identity(mesh_obj)

    return _resolver


def _identity_resolver_from_targets(scene):
    identity_by_name: dict[str, tuple[str, int]] = {}
    for item in scene.bmc_target_items:
        if not item.enabled:
            continue
        mesh_obj = _resolve_target_item_object(scene, item)
        if mesh_obj is None:
            continue
        ib_hash = str(item.ib_hash or "").lower()
        if ib_hash:
            identity_by_name[mesh_obj.name] = (ib_hash, 0)

    def _resolver(mesh_obj):
        identity = identity_by_name.get(mesh_obj.name)
        if identity is not None:
            return identity
        return resolve_mesh_identity(mesh_obj)

    return _resolver


def _skipped_name_set(messages: tuple[str, ...] | list[str]) -> set[str]:
    skipped_names: set[str] = set()
    for message in messages:
        if ":" not in message:
            continue
        skipped_names.add(message.split(":", 1)[0].strip())
    return skipped_names


def _apply_mapping_bundle(
    mesh_objects: list[object],
    selected_entries: list[dict],
    alias_entries: list[dict],
    identity_resolver,
    merge_same_bone_groups: bool,
):
    remap_result = apply_group_remaps_to_meshes(
        mesh_objects,
        {"object_remaps": selected_entries},
        identity_resolver=identity_resolver,
    )
    merged_aliases = 0
    merge_messages: list[str] = []
    if merge_same_bone_groups and alias_entries:
        merge_meshes = [
            mesh_obj
            for mesh_obj in mesh_objects
            if mesh_obj.name not in _skipped_name_set(remap_result.skipped_objects)
        ]
        if merge_meshes:
            merge_result = merge_duplicate_alias_weights(
                merge_meshes,
                alias_entries,
                identity_resolver=identity_resolver,
            )
            merged_aliases = int(merge_result.merged_aliases)
            merge_messages.extend(list(merge_result.skipped_objects))
    return remap_result, merged_aliases, merge_messages


class BMC_OT_add_selected_targets(bpy.types.Operator):
    bl_idname = "object.bmc_add_selected_targets"
    bl_label = "Add Selected Objects"
    bl_description = "Add the current selected mesh objects to the target list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context):
        scene = context.scene
        _refresh_target_items(scene)
        existing_names = {
            (item.object_ref.name if getattr(item, "object_ref", None) else item.object_name)
            for item in scene.bmc_target_items
            if item.object_name or getattr(item, "object_ref", None)
        }
        added_count = 0
        collection = _ensure_target_collection(context)

        for mesh_obj in context.selected_objects:
            if mesh_obj.type != "MESH":
                continue
            _link_object_to_collection(mesh_obj, collection)
            if mesh_obj.name in existing_names:
                continue
            if not _add_mesh_object_to_target_list(scene, mesh_obj, existing_names):
                self.report({"WARNING"}, f"{mesh_obj.name}: cannot infer ib_hash")
                continue
            added_count += 1

        if added_count == 0:
            self.report({"WARNING"}, "No new mesh targets were added")
            return {"CANCELLED"}

        scene.bmc_target_index = len(scene.bmc_target_items) - 1
        self.report({"INFO"}, f"Added {added_count} target objects")
        return {"FINISHED"}


class BMC_OT_create_target_collection(bpy.types.Operator):
    bl_idname = "object.bmc_create_target_collection"
    bl_label = "Create Target Collection"
    bl_description = "Create or select the collection that stores Bone Merge Capture target meshes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        collection = _ensure_target_collection(context)
        self.report({"INFO"}, f"Target collection: {collection.name}")
        return {"FINISHED"}


class BMC_OT_sync_targets_from_collection(bpy.types.Operator):
    bl_idname = "object.bmc_sync_targets_from_collection"
    bl_label = "Sync Targets From Collection"
    bl_description = "Replace the target list with mesh objects from the target collection"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_target_collection", None))

    def execute(self, context):
        scene = context.scene
        collection = scene.bmc_target_collection
        scene.bmc_target_items.clear()
        existing_names: set[str] = set()
        added_count = 0
        skipped_count = 0

        for mesh_obj in collection.objects:
            if mesh_obj.type != "MESH":
                continue
            if _add_mesh_object_to_target_list(scene, mesh_obj, existing_names):
                added_count += 1
            else:
                skipped_count += 1

        scene.bmc_target_index = min(scene.bmc_target_index, max(0, len(scene.bmc_target_items) - 1))
        message = f"Synced {added_count} target object(s) from {collection.name}"
        if skipped_count:
            message += f"; skipped {skipped_count}"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_refresh_target_identity(bpy.types.Operator):
    bl_idname = "object.bmc_refresh_target_identity"
    bl_label = "Refresh Target Identity"
    bl_description = "Explicitly re-freeze the active target's IB hash and local bone count from the current object"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        target_items = getattr(scene, "bmc_target_items", None)
        target_index = int(getattr(scene, "bmc_target_index", -1))
        return bool(scene and target_items and 0 <= target_index < len(target_items))

    def execute(self, context):
        scene = context.scene
        item = scene.bmc_target_items[scene.bmc_target_index]
        mesh_obj = _resolve_target_item_object(scene, item)
        if mesh_obj is None:
            self.report({"ERROR"}, f"{item.object_name}: mesh object not found in current scene")
            return {"CANCELLED"}

        identity = resolve_mesh_identity(mesh_obj)
        if identity is None:
            self.report({"ERROR"}, f"{mesh_obj.name}: cannot infer ib_hash")
            return {"CANCELLED"}

        try:
            local_bone_count = int(infer_local_bone_count_from_mesh(mesh_obj))
        except ValueError as exc:
            self.report({"ERROR"}, f"{mesh_obj.name}: {exc}")
            return {"CANCELLED"}

        item.object_ref = mesh_obj
        item.object_name = mesh_obj.name
        item.ib_hash = identity[0]
        item.match_index_count = 0
        item.local_bone_count = local_bone_count
        item.autodetected = bool(getattr(mesh_obj, "merge_ib_autodetected", True))
        self.report(
            {"INFO"},
            f"Frozen target {mesh_obj.name} as {item.ib_hash} with {item.local_bone_count} local bones",
        )
        return {"FINISHED"}


class BMC_OT_create_export_collection(bpy.types.Operator):
    bl_idname = "object.bmc_create_export_collection"
    bl_label = "Create Export Collection"
    bl_description = "Create or select the collection that stores final meshes to prepare for export"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        source_collection = _ensure_export_collection(context)
        message = f"Export root: {source_collection.name}; build global pool to create IB region collections"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_add_selected_export_objects(bpy.types.Operator):
    bl_idname = "object.bmc_add_selected_export_objects"
    bl_label = "Add Selected To Export"
    bl_description = "Add selected meshes to the unique existing export child collection with the same source IB hash"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context):
        added_count = 0
        moved_count = 0
        skipped_messages: list[str] = []
        collection = _ensure_export_collection(context)
        regions_by_hash = _export_region_collections_by_hash(collection)

        for mesh_obj in context.selected_objects:
            if mesh_obj.type != "MESH":
                continue
            ib_hash = _object_source_hash(mesh_obj)
            if not ib_hash:
                skipped_messages.append(f"{mesh_obj.name}: no source hash")
                continue
            matching_collections = regions_by_hash.get(ib_hash, [])
            if not matching_collections:
                skipped_messages.append(f"{mesh_obj.name}: no export child for {ib_hash}")
                continue
            if len(matching_collections) > 1:
                skipped_messages.append(f"{mesh_obj.name}: ambiguous_source_hash {ib_hash}")
                continue
            region_collection = matching_collections[0]
            already_in_region = _child_collection_contains_object(region_collection, mesh_obj)
            _link_object_to_collection(mesh_obj, region_collection)
            _unlink_object_from_export_sibling_regions(collection, mesh_obj, region_collection)
            if already_in_region:
                moved_count += 1
            else:
                added_count += 1

        touched_count = added_count + moved_count
        if touched_count == 0:
            if skipped_messages:
                self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
            else:
                self.report({"WARNING"}, "No selected mesh objects matched an export child collection")
            return {"CANCELLED"}

        message = f"Placed {touched_count} selected object(s) into matching export collection(s)"
        if skipped_messages:
            message += f"; skipped {len(skipped_messages)}"
        self.report({"INFO"}, message)
        if skipped_messages:
            self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_apply_export_collection_global_names(bpy.types.Operator):
    bl_idname = "object.bmc_apply_export_collection_global_names"
    bl_label = "Apply Export Global Names"
    bl_description = "Rename mesh vertex groups under export child collections using the child collection's global-pool mapping"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_export_collection", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            scanned_count, updated_count, renamed_groups, skipped_messages = _apply_global_names_in_export_collection(
                context,
                manifest,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Apply export global names failed: {exc}")
            return {"CANCELLED"}

        if scanned_count <= 0:
            self.report({"WARNING"}, "No mesh objects found under export child collections")
            return {"CANCELLED"}
        message = f"Applied global names to {updated_count}/{scanned_count} export mesh object(s); renamed {renamed_groups} group(s)"
        if skipped_messages:
            message += f"; skipped {len(skipped_messages)}"
        self.report({"INFO"}, message)
        if skipped_messages:
            self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_apply_global_names_by_object_hash(bpy.types.Operator):
    bl_idname = "object.bmc_apply_global_names_by_object_hash"
    bl_label = "Rename To Global Groups"
    bl_description = "Rename selected meshes to global vertex-group indices by matching object-name hashes to the global pool"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_manifest_path", ""))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            scanned_count, updated_count, renamed_groups, skipped_messages = _apply_global_names_by_object_hash(
                context,
                manifest,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Rename To Global failed: {exc}")
            return {"CANCELLED"}
        if scanned_count <= 0:
            self.report({"WARNING"}, "No mesh objects found")
            return {"CANCELLED"}
        message = f"Rename To Global updated {updated_count}/{scanned_count} mesh object(s); renamed {renamed_groups} group(s)"
        if skipped_messages:
            message += f"; skipped {len(skipped_messages)}"
        self.report({"INFO"}, message)
        if skipped_messages:
            self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_revert_global_names_by_object_hash(bpy.types.Operator):
    bl_idname = "object.bmc_revert_global_names_by_object_hash"
    bl_label = "Rename To Local Groups"
    bl_description = "Rename selected meshes from global vertex-group indices back to source-local indices by object-name hash"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_manifest_path", ""))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            scanned_count, updated_count, renamed_groups, skipped_messages = _revert_global_names_by_object_hash(
                context,
                manifest,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Rename To Local failed: {exc}")
            return {"CANCELLED"}
        if scanned_count <= 0:
            self.report({"WARNING"}, "No mesh objects found")
            return {"CANCELLED"}
        message = f"Rename To Local updated {updated_count}/{scanned_count} mesh object(s); renamed {renamed_groups} group(s)"
        if skipped_messages:
            message += f"; skipped {len(skipped_messages)}"
        self.report({"INFO"}, message)
        if skipped_messages:
            self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_merge_selected_seam_groups(bpy.types.Operator):
    bl_idname = "object.bmc_merge_selected_seam_groups"
    bl_label = "Merge Seam Groups"
    bl_description = "Build and apply seam vertex-group rename mapping for selected mesh objects"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        selected_meshes = [obj for obj in context.selected_objects if getattr(obj, "type", "") == "MESH"]
        return len(selected_meshes) >= 2

    def execute(self, context):
        try:
            result = _merge_selected_seam_groups(context)
        except Exception as exc:
            self.report({"ERROR"}, f"Merge Seam Groups failed: {exc}")
            return {"CANCELLED"}

        message = (
            f"Merge Seam Groups matched {len(result.aliases)} mapping(s); "
            f"renamed {result.renamed_groups} group(s) on {result.updated_objects} object(s)"
        )
        self.report({"INFO"}, message)
        if result.skipped_messages:
            self.report({"WARNING"}, " | ".join(result.skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_prepare_export_collection(bpy.types.Operator):
    bl_idname = "object.bmc_prepare_export_collection"
    bl_label = "Export"
    bl_description = "Scan the export root, build per-part palettes, and write buffers according to the selected export mode"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene)

    def execute(self, context):
        scene = context.scene
        try:
            source_collection = _ensure_export_collection(context)
            part_materialize_result = _materialize_auto_export_part_collections(source_collection)
            generate_ini = str(getattr(scene, "bmc_export_mode", "BUFFER_ONLY") or "BUFFER_ONLY") == "BUFFER_AND_INI"
            manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
            if manifest_path and os.path.exists(manifest_path):
                _raise_if_lod_unmatched_used_by_export(scene, read_json(manifest_path))
            result = prepare_export_collection(
                context=context,
                source_collection=source_collection,
                build_collection=None,
                output_dir=scene.bmc_output_dir,
                internal_manifest_dir=None,
                capture_manifest_path=scene.bmc_manifest_path,
                generate_ini=generate_ini,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Export failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_export_manifest_path = result["manifest_path"]
        if result.get("bonestore_ini_path"):
            scene.bmc_ini_path = result["bonestore_ini_path"]
        try:
            export_manifest = read_json(result["manifest_path"])
            payload = load_mapping_payload_from_scene(scene)
            payload["lod_host_map"] = list(export_manifest.get("lod_host_map", []) or [])
            store_mapping_payload_on_scene(scene, payload)
        except Exception:
            pass
        mode_label = "buffers + INI" if generate_ini else "buffers"
        message = f"Exported {mode_label}: {result['objects']} object(s), {result['palettes']} palette(s) to {result['output_dir']}"
        if int(part_materialize_result.get("parts", 0) or 0) > 0:
            message += (
                f"; auto-parted {int(part_materialize_result.get('regions', 0) or 0)} region(s) "
                f"into {int(part_materialize_result.get('parts', 0) or 0)} part collection(s)"
            )
        self.report({"INFO"}, message)
        if int(part_materialize_result.get("parts", 0) or 0) > 0:
            self.report(
                {"WARNING"},
                (
                    "Weighted global bones exceeded 256 in at least one export region; "
                    "partNN child collections were created from actual mesh weight usage."
                ),
            )
        materialize_warnings = list(part_materialize_result.get("warnings", []) or [])
        if materialize_warnings:
            self.report({"WARNING"}, " | ".join(str(item) for item in materialize_warnings[:3]))
        return {"FINISHED"}


class BMC_OT_remove_target(bpy.types.Operator):
    bl_idname = "object.bmc_remove_target"
    bl_label = "Remove Target"
    bl_description = "Remove the active target object from the list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.bmc_target_items)

    def execute(self, context):
        scene = context.scene
        scene.bmc_target_items.remove(scene.bmc_target_index)
        scene.bmc_target_index = min(scene.bmc_target_index, max(0, len(scene.bmc_target_items) - 1))
        return {"FINISHED"}


class BMC_OT_clear_targets(bpy.types.Operator):
    bl_idname = "object.bmc_clear_targets"
    bl_label = "Clear Targets"
    bl_description = "Clear the current target object list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.bmc_target_items)

    def execute(self, context):
        scene = context.scene
        scene.bmc_target_items.clear()
        scene.bmc_target_index = 0
        return {"FINISHED"}


class BMC_OT_add_selected_lod_targets(bpy.types.Operator):
    bl_idname = "object.bmc_add_selected_lod_targets"
    bl_label = "Add Selected LOD Objects"
    bl_description = "Add the current selected mesh objects to the LOD target list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context):
        scene = context.scene
        existing_names = {
            (item.object_ref.name if getattr(item, "object_ref", None) else item.object_name)
            for item in scene.bmc_lod_target_items
        }
        added_count = 0
        for mesh_obj in context.selected_objects:
            if mesh_obj.type != "MESH":
                continue
            if _add_mesh_object_to_lod_target_list(scene, mesh_obj, existing_names):
                added_count += 1
        if added_count:
            scene.bmc_lod_target_index = len(scene.bmc_lod_target_items) - 1
            self.report({"INFO"}, f"Added {added_count} LOD target object(s)")
        else:
            self.report({"WARNING"}, "No new mesh objects were added to the LOD target list")
        return {"FINISHED"}


class BMC_OT_remove_lod_target(bpy.types.Operator):
    bl_idname = "object.bmc_remove_lod_target"
    bl_label = "Remove LOD Target"
    bl_description = "Remove the active object from the LOD target list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_lod_target_items", None))

    def execute(self, context):
        scene = context.scene
        scene.bmc_lod_target_items.remove(scene.bmc_lod_target_index)
        scene.bmc_lod_target_index = min(scene.bmc_lod_target_index, max(0, len(scene.bmc_lod_target_items) - 1))
        return {"FINISHED"}


class BMC_OT_clear_lod_targets(bpy.types.Operator):
    bl_idname = "object.bmc_clear_lod_targets"
    bl_label = "Clear LOD Targets"
    bl_description = "Clear the LOD target list and mapping table"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_lod_target_items", None))

    def execute(self, context):
        scene = context.scene
        scene.bmc_lod_target_items.clear()
        scene.bmc_lod_target_index = 0
        scene.bmc_lod_mapping_items.clear()
        scene.bmc_lod_mapping_index = 0
        return {"FINISHED"}


class BMC_OT_scan_lod_targets(bpy.types.Operator):
    bl_idname = "object.bmc_scan_lod_targets"
    bl_label = "Scan LOD"
    bl_description = "Scan the chosen LOD FrameAnalysis directory for the frozen LOD target list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(
            scene
            and scene.bmc_lod_frameanalysis_dir
            and any(
                item.enabled and item.ib_hash and item.object_name
                for item in scene.bmc_lod_target_items
            )
        )

    def execute(self, context):
        scene = context.scene
        try:
            target_specs = _enabled_lod_target_specs(context)
            result = scan_lod_targets_and_generate_manifest(
                frameanalysis_dir=scene.bmc_lod_frameanalysis_dir,
                target_specs=target_specs,
                output_dir=scene.bmc_output_dir,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Scan LOD failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_lod_manifest_path = result["manifest_path"]
        _sync_target_match_counts_from_manifest(scene, scene.bmc_lod_manifest_path, collection_name="bmc_lod_target_items")
        scene.bmc_lod_mapping_items.clear()
        scene.bmc_lod_mapping_index = 0
        payload = dict(result["payload"])
        scene.bmc_lod_shadow_host_hash = str(payload.get("shadow_host_hash", "") or "")
        scene.bmc_lod_shadow_host_match_index_count = int(payload.get("shadow_host_match_index_count", -1))
        scene.bmc_lod_shadow_host_vs_hash = str(payload.get("shadow_host_vs_hash", "") or "")

        base_payload = load_mapping_payload_from_scene(scene)
        base_payload["canonical_global_to_lod_global"] = []
        base_payload["lod_host_map"] = []
        mapping_payload = build_mapping_payload(
            scene,
            manifest_payload=_read_manifest_payload(scene.bmc_manifest_path),
            base_payload=base_payload,
        )
        store_mapping_payload_on_scene(scene, mapping_payload)

        if scene.bmc_manifest_path:
            try:
                scene.bmc_ini_path = regenerate_bonestore_runtime_files(
                    output_dir=scene.bmc_output_dir or os.path.dirname(bpy.path.abspath(scene.bmc_manifest_path)),
                    capture_manifest_path=bpy.path.abspath(scene.bmc_manifest_path),
                    export_manifest_path=(
                        bpy.path.abspath(scene.bmc_export_manifest_path) if scene.bmc_export_manifest_path else ""
                    ),
                    mapping_payload=mapping_payload,
                )
            except Exception as exc:
                self.report({"WARNING"}, f"LOD runtime refresh skipped: {exc}")

        info_message = (
            f"Scanned {int(result['scanned_parts'])} LOD parts; total LOD globals {int(result['total_lod_global_bones'])}"
        )
        if scene.bmc_lod_shadow_host_hash:
            info_message += (
                f"; LOD shadow host {scene.bmc_lod_shadow_host_hash}-{scene.bmc_lod_shadow_host_match_index_count}"
                f" vs={scene.bmc_lod_shadow_host_vs_hash or '?'}"
            )
        self.report({"INFO"}, info_message)
        if result.get("warnings"):
            self.report({"WARNING"}, " | ".join(result["warnings"][:3]))
        if result.get("shadow_host_warning"):
            self.report({"WARNING"}, str(result["shadow_host_warning"]))
        return {"FINISHED"}


class BMC_OT_apply_lod_vertex_group_remap(bpy.types.Operator):
    bl_idname = "object.bmc_apply_lod_vertex_group_remap"
    bl_label = "Rename LOD Groups"
    bl_description = "Rename enabled LOD target objects' local vertex groups to LOD global indices"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_lod_target_items", None))

    def execute(self, context):
        scene = context.scene
        try:
            payload = _sync_scene_payload(scene)
            matched_meshes, selected_entries, skipped_before_apply = _select_lod_object_remap_entries(scene, payload)
            if not matched_meshes:
                raise ValueError("No enabled LOD Target objects matched any LOD remap entry")
            remap_result = apply_group_remaps_to_meshes(
                matched_meshes,
                {"object_remaps": selected_entries},
                identity_resolver=_identity_resolver_from_entries(matched_meshes, selected_entries),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"LOD remap failed: {exc}")
            return {"CANCELLED"}

        skipped_after_apply = list(remap_result.skipped_objects)
        skipped_after_apply.extend(skipped_before_apply)
        self.report(
            {"INFO"},
            f"Renamed {remap_result.updated_objects}/{len(matched_meshes)} LOD target meshes; renamed {remap_result.renamed_groups} groups",
        )
        if skipped_after_apply:
            self.report({"WARNING"}, " | ".join(skipped_after_apply[:3]))
        return {"FINISHED"}


class BMC_OT_build_lod_mapping(bpy.types.Operator):
    bl_idname = "object.bmc_build_lod_mapping"
    bl_label = "Build LOD Mapping"
    bl_description = "Build canonical_global -> lod_global mappings from main and LOD meshes"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(
            scene
            and scene.bmc_manifest_path
            and scene.bmc_lod_manifest_path
            and scene.bmc_target_items
            and scene.bmc_lod_target_items
        )

    def execute(self, context):
        scene = context.scene
        try:
            payload = _sync_scene_payload(scene)
            canonical_manifest = dict(payload.get("capture_manifest", {}) or {})
            lod_variant = dict(payload.get("lod_variant", {}) or {})
            canonical_mesh_entries = _canonical_mesh_entries_for_lod_build(scene, payload)
            lod_mesh_entries = _lod_mesh_entries_for_build(scene, payload)
            if not canonical_mesh_entries:
                raise ValueError("No enabled main Target objects matched the frozen canonical remap entries")
            if not lod_mesh_entries:
                raise ValueError("No enabled LOD Target objects matched the frozen LOD remap entries")
            mapping_records = build_lod_mapping(
                canonical_manifest=canonical_manifest,
                lod_variant=lod_variant,
                canonical_mesh_entries=canonical_mesh_entries,
                lod_mesh_entries=lod_mesh_entries,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Build LOD mapping failed: {exc}")
            return {"CANCELLED"}

        mapping_entries = [
            {
                "enabled": True,
                "canonical_global_bone": int(record.canonical_global_bone),
                "mapped_lod_global_bone": int(record.mapped_lod_global_bone),
                "status": str(record.status),
                "score": float(record.score),
                "note": str(record.note),
            }
            for record in mapping_records
        ]
        _replace_lod_mapping_items(scene, mapping_entries)
        payload = _sync_scene_payload(scene)
        exact_count = sum(1 for item in mapping_entries if str(item["status"]) == "exact")
        grouped_count = sum(1 for item in mapping_entries if str(item["status"]) == "grouped")
        unmatched_count = sum(1 for item in mapping_entries if int(item["mapped_lod_global_bone"]) < 0)
        self.report(
            {"INFO"},
            f"Built LOD mapping: exact {exact_count}; grouped {grouped_count}; unmatched {unmatched_count}",
        )
        if unmatched_count:
            unresolved_notes = [
                entry
                for entry in mapping_entries
                if int(entry["mapped_lod_global_bone"]) < 0 and str(entry["note"])
            ]
            if unresolved_notes:
                self.report({"WARNING"}, " | ".join(str(entry["note"]) for entry in unresolved_notes[:3]))
        return {"FINISHED"}


class BMC_OT_generate_lod_runtime_map(bpy.types.Operator):
    bl_idname = "object.bmc_generate_lod_runtime_map"
    bl_label = "Generate LOD Runtime Map"
    bl_description = "Regenerate BoneStore.ini and export manifest LOD resources from the current canonical->LOD mapping"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_ini_path", ""))

    def execute(self, context):
        scene = context.scene
        try:
            payload = _sync_scene_payload(scene)
            output_dir = scene.bmc_output_dir or os.path.dirname(bpy.path.abspath(scene.bmc_ini_path or scene.bmc_manifest_path))
            ini_path = regenerate_bonestore_runtime_files(
                output_dir=output_dir,
                capture_manifest_path=bpy.path.abspath(scene.bmc_manifest_path),
                export_manifest_path=(
                    bpy.path.abspath(scene.bmc_export_manifest_path) if scene.bmc_export_manifest_path else ""
                ),
                mapping_payload=payload,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Generate LOD runtime map failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_ini_path = ini_path
        if scene.bmc_export_manifest_path:
            try:
                export_manifest = read_json(bpy.path.abspath(scene.bmc_export_manifest_path))
                payload = load_mapping_payload_from_scene(scene)
                payload["lod_host_map"] = list(export_manifest.get("lod_host_map", []) or [])
                store_mapping_payload_on_scene(scene, payload)
            except Exception:
                pass
        self.report({"INFO"}, f"LOD runtime resources regenerated: {ini_path}")
        return {"FINISHED"}


class BMC_OT_seam_add_selected_objects(bpy.types.Operator):
    bl_idname = "object.bmc_seam_add_selected_objects"
    bl_label = "Add Selected Seam Objects"
    bl_description = "Add selected mesh objects to the independent seam matcher list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context):
        scene = context.scene
        existing_names = {
            (item.object_ref.name if getattr(item, "object_ref", None) else item.object_name)
            for item in scene.bmc_seam_match_items
            if item.object_name or getattr(item, "object_ref", None)
        }
        added_count = 0
        for mesh_obj in context.selected_objects:
            if mesh_obj.type != "MESH" or mesh_obj.name in existing_names:
                continue
            item = scene.bmc_seam_match_items.add()
            item.enabled = True
            item.object_name = mesh_obj.name
            item.object_ref = mesh_obj
            existing_names.add(mesh_obj.name)
            added_count += 1
        if added_count:
            scene.bmc_seam_match_index = len(scene.bmc_seam_match_items) - 1
            self.report({"INFO"}, f"Added {added_count} seam matcher object(s)")
        else:
            self.report({"WARNING"}, "No new mesh objects were added")
        return {"FINISHED"}


class BMC_OT_seam_remove_object(bpy.types.Operator):
    bl_idname = "object.bmc_seam_remove_object"
    bl_label = "Remove Seam Object"
    bl_description = "Remove the active object from the independent seam matcher list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_seam_match_items", None))

    def execute(self, context):
        scene = context.scene
        scene.bmc_seam_match_items.remove(scene.bmc_seam_match_index)
        scene.bmc_seam_match_index = min(scene.bmc_seam_match_index, max(0, len(scene.bmc_seam_match_items) - 1))
        return {"FINISHED"}


class BMC_OT_seam_clear_objects(bpy.types.Operator):
    bl_idname = "object.bmc_seam_clear_objects"
    bl_label = "Clear Seam Objects"
    bl_description = "Clear seam matcher objects and mappings"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_seam_match_items", None))

    def execute(self, context):
        scene = context.scene
        scene.bmc_seam_match_items.clear()
        scene.bmc_seam_match_index = 0
        scene.bmc_seam_alias_items.clear()
        scene.bmc_seam_alias_index = 0
        scene.bmc_seam_pair_summary = ""
        return {"FINISHED"}


class BMC_OT_seam_build_mapping(bpy.types.Operator):
    bl_idname = "object.bmc_seam_build_mapping"
    bl_label = "Build Seam Mapping"
    bl_description = "Build a seam-only vertex-group rename map from the independent object list"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and len(_enabled_seam_mesh_objects(context.scene, sync_names=False)) >= 2)

    def execute(self, context):
        scene = context.scene
        mesh_objects = _enabled_seam_mesh_objects(scene, sync_names=True)
        try:
            result = build_seam_mapping(mesh_objects)
            _replace_seam_alias_items(scene, result)
        except Exception as exc:
            self.report({"ERROR"}, f"Build seam mapping failed: {exc}")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Built {len(result.aliases)} seam mapping(s); tested {result.matched_pairs + result.skipped_pairs} pair(s)",
        )
        return {"FINISHED"}


class BMC_OT_seam_apply_mapping(bpy.types.Operator):
    bl_idname = "object.bmc_seam_apply_mapping"
    bl_label = "Apply Seam Mapping"
    bl_description = "Apply the current seam mapping by renaming source vertex groups to canonical groups"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_seam_alias_items", None))

    def execute(self, context):
        scene = context.scene
        mesh_objects = _enabled_seam_mesh_objects(scene, sync_names=True)
        if not mesh_objects:
            self.report({"ERROR"}, "No enabled seam matcher objects found")
            return {"CANCELLED"}
        try:
            result = apply_seam_mapping(mesh_objects, _seam_alias_payload(scene))
        except Exception as exc:
            self.report({"ERROR"}, f"Apply seam mapping failed: {exc}")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Applied seam mapping to {result.updated_objects} object(s); renamed {result.renamed_groups} group(s)",
        )
        if result.skipped_messages:
            self.report({"WARNING"}, " | ".join(result.skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_candidate_add_hash(bpy.types.Operator):
    bl_idname = "object.bmc_candidate_add_hash"
    bl_label = "Add Candidate IB"
    bl_description = "Add the typed IB hash to the Candidate IB list, restoring matching analyzed slices when available"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene)

    def execute(self, context):
        scene = context.scene
        try:
            hashes = _parse_hash_list(str(getattr(scene, "bmc_candidate_add_hash", "") or ""))
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not hashes:
            self.report({"ERROR"}, "Type an 8-hex IB hash before pressing Add")
            return {"CANCELLED"}

        manifest = _read_manifest_payload(scene.bmc_manifest_path)
        added_count = 0
        for ib_hash in hashes:
            matches = _manifest_candidates_by_hash(manifest, ib_hash)
            if matches:
                for candidate in matches:
                    if _add_candidate_from_payload(scene, candidate, manual=False):
                        added_count += 1
                continue
            manual_candidate = {
                "enabled": True,
                "ib_hash": ib_hash,
                "match_first_index": 0,
                "match_index_count": 0,
                "display_name": _candidate_display_name_from_values(ib_hash, 0, 0),
                "draw_indices": [],
                "shadow_draw_indices": [],
                "shadow_capture_ready": False,
                "local_bone_count": 0,
                "status": "manual_hash_missing_analysis",
                "import_paths": {"ib": "", "vb": {}, "layout": ""},
            }
            if _add_candidate_from_payload(scene, manual_candidate, manual=True):
                added_count += 1

        if added_count:
            self.report({"INFO"}, f"Added {added_count} candidate IB entr{'y' if added_count == 1 else 'ies'}")
            return {"FINISHED"}
        self.report({"INFO"}, "Candidate IB already exists in the list")
        return {"FINISHED"}


class BMC_OT_candidate_remove(bpy.types.Operator):
    bl_idname = "object.bmc_candidate_remove"
    bl_label = "Remove Candidate IB"
    bl_description = "Remove the selected Candidate IB list entry"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_candidate_items", None))

    def execute(self, context):
        scene = context.scene
        index = int(scene.bmc_candidate_index)
        if index < 0 or index >= len(scene.bmc_candidate_items):
            self.report({"ERROR"}, "No candidate IB is selected")
            return {"CANCELLED"}
        removed_name = str(scene.bmc_candidate_items[index].display_name or scene.bmc_candidate_items[index].ib_hash)
        scene.bmc_candidate_items.remove(index)
        scene.bmc_candidate_index = min(index, max(0, len(scene.bmc_candidate_items) - 1))
        self.report({"INFO"}, f"Removed candidate {removed_name}")
        return {"FINISHED"}


class BMC_OT_candidate_refresh_from_collection(bpy.types.Operator):
    bl_idname = "object.bmc_candidate_refresh_from_collection"
    bl_label = "Refresh Candidates From Collection"
    bl_description = "Replace the Candidate IB list with IB identities from objects under the export/root collection"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene)

    def execute(self, context):
        scene = context.scene
        manifest = _read_manifest_payload(scene.bmc_manifest_path)
        source_collection = scene.bmc_export_collection or context.collection
        if source_collection is None:
            self.report({"ERROR"}, "No collection is available for candidate refresh")
            return {"CANCELLED"}
        previous_state = {
            _candidate_key_from_item(item): {"enabled": bool(getattr(item, "enabled", True))}
            for item in scene.bmc_candidate_items
        }
        previous_keys = set(previous_state)
        candidates, stats = _candidate_payloads_from_collection(scene, source_collection, manifest)
        next_keys = {_candidate_key_from_payload(candidate) for candidate in candidates}
        _replace_candidate_items_from_payloads(scene, candidates, previous_state=previous_state)
        synced_count = len(scene.bmc_candidate_items)
        added_count = len(next_keys - previous_keys)
        removed_count = len(previous_keys - next_keys)
        self.report(
            {"INFO"},
            (
                f"Synced {synced_count} candidate IB(s) from {stats['recognized_meshes']}/"
                f"{stats['scanned_meshes']} mesh object(s); added {added_count}; removed {removed_count}"
            ),
        )
        return {"FINISHED"}


class BMC_OT_build_global_bone_pool(bpy.types.Operator):
    bl_idname = "object.bmc_build_global_bone_pool"
    bl_label = "Build Global Bone Pool"
    bl_description = "Build a compact global bone pool from enabled Candidate IBs; capture-ready entries are ordered first"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_candidate_items", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; run Analyze Main first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            candidates = _candidate_payloads_from_ui(scene, manifest)
            bone_pool_order = build_bone_pool_order(candidates)
            if not bone_pool_order:
                self.report({"ERROR"}, "No enabled candidate has usable bone weights for the global pool")
                return {"CANCELLED"}
            object_remaps = _object_remaps_from_bone_pool_order(bone_pool_order)
            generation_id = _global_pool_generation_id({"bone_pool_order": bone_pool_order}, bone_pool_order)
            manifest["candidate_ibs"] = candidates
            manifest["bone_pool_order"] = bone_pool_order
            manifest["object_remaps"] = object_remaps
            manifest["global_pool_generation"] = generation_id
            manifest.setdefault("buffer_tables", {})["global_bone_count"] = sum(
                int(item.get("local_bone_count", 0)) for item in bone_pool_order
            )
            manifest.setdefault("validation", []).append(
                {
                    "severity": "info",
                    "code": "global_pool_built_from_candidate_list",
                    "message": f"Built compact global pool from {len(bone_pool_order)} enabled candidate(s).",
                    "draw_indices": [],
                }
            )
            write_json(manifest_path, manifest)
            export_collection = _ensure_export_collection(context)
            created_count = 0
            unavailable_count = 0
            lod_excluded_count = 0
            for record in bone_pool_order:
                capture_available = bool(record.get("bone_capture_available", record.get("shadow_capture_ready", False)))
                lod_match_excluded = bool(record.get("lod_match_excluded", False))
                if not capture_available:
                    unavailable_count += 1
                if lod_match_excluded:
                    lod_excluded_count += 1
                if _ensure_export_region_collection_if_missing(
                    export_collection,
                    str(record.get("ib_hash", "") or ""),
                    int(record.get("match_index_count", 0) or 0),
                    int(record.get("match_first_index", 0) or 0),
                    bone_capture_available=capture_available,
                    lod_match_excluded=lod_match_excluded,
                ):
                    created_count += 1
            _update_scene_mapping_payload(scene, manifest_path)
        except Exception as exc:
            self.report({"ERROR"}, f"Build Global Bone Pool failed: {exc}")
            return {"CANCELLED"}

        skipped_count = len([item for item in candidates if item.get("enabled", True)]) - len(bone_pool_order)
        message = f"Built global pool with {len(bone_pool_order)} source IB(s)"
        message += f"; collections {created_count}"
        message += f"; compact bones {sum(int(item.get('local_bone_count', 0)) for item in bone_pool_order)}"
        if unavailable_count > 0:
            message += f"; {unavailable_count} mapping-only IB(s)"
        if lod_excluded_count > 0:
            message += f"; {lod_excluded_count} no-LOD IB(s)"
        if skipped_count > 0:
            message += f"; skipped {skipped_count} invalid IB(s)"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_apply_global_bone_pool(bpy.types.Operator):
    bl_idname = "object.bmc_apply_global_bone_pool"
    bl_label = "Apply Global Bone Pool"
    bl_description = "Apply the built global bone pool to candidate source meshes by renaming groups and merging seam groups"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", ""))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        seam_result = None
        seam_warnings: list[str] = []
        seam_skipped_reason = ""
        try:
            manifest = read_json(manifest_path)
            if not manifest.get("bone_pool_order") or not manifest.get("object_remaps"):
                self.report({"ERROR"}, "Global bone pool is empty; run Build Global Bone Pool first")
                return {"CANCELLED"}
            _update_scene_mapping_payload(scene, manifest_path)
            meshes_and_remaps = _candidate_source_meshes_and_remaps(context, manifest)
            updated_objects, renamed_groups, rename_warnings = _apply_global_names_to_candidate_source_objects(
                context,
                manifest,
                meshes_and_remaps,
            )
            if renamed_groups > 0:
                try:
                    seam_result = _merge_candidate_source_seam_groups(context, manifest, meshes_and_remaps)
                except Exception as seam_exc:
                    seam_warnings.append(f"Seam merge skipped: {seam_exc}")
            else:
                seam_skipped_reason = "unchanged groups"
        except Exception as exc:
            self.report({"ERROR"}, f"Apply Global Bone Pool failed: {exc}")
            return {"CANCELLED"}

        message = "Applied global bone pool"
        message += f"; global-renamed {updated_objects} object(s), {renamed_groups} group(s)"
        if seam_result is not None:
            message += f"; seam-merged {seam_result.updated_objects} object(s), {seam_result.renamed_groups} group(s)"
        elif seam_skipped_reason:
            message += f"; seam-skipped {seam_skipped_reason}"
        else:
            message += "; seam-merged 0 object(s), 0 group(s)"
        self.report({"INFO"}, message)
        warnings = list(rename_warnings)
        if seam_result is not None:
            warnings.extend(str(message) for message in seam_result.skipped_messages)
        warnings.extend(seam_warnings)
        if warnings:
            self.report({"WARNING"}, " | ".join(warnings[:3]))
        return {"FINISHED"}


class BMC_OT_analyze_lod_frameanalysis(bpy.types.Operator):
    bl_idname = "object.bmc_analyze_lod_frameanalysis"
    bl_label = "Analyze LOD"
    bl_description = "Analyze the LOD FrameAnalysis folder and write canonical global-bone scatter mappings into the manifest"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_lod_frameanalysis_dir", ""))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            lod_result = analyze_lod_for_manifest(
                manifest,
                bpy.path.abspath(str(scene.bmc_lod_frameanalysis_dir or "")),
                lod_level=1,
            )
            manifest["lod_frameanalysis"] = list(lod_result.get("lod_frameanalysis", []) or [])
            manifest["lod_links"] = list(lod_result.get("lod_links", []) or [])
            manifest["lod_capture_records"] = list(lod_result.get("lod_capture_records", []) or [])
            manifest["lod_mapping"] = list(lod_result.get("lod_mapping", []) or [])
            manifest["lod_review"] = dict(lod_result.get("lod_review", {}) or {})
            manifest["lod_validation"] = list(lod_result.get("validation", []) or [])
            manifest["lod_manifest_snapshot"] = dict(lod_result.get("lod_manifest_snapshot", {}) or {})
            manifest.setdefault("validation", []).append(
                {
                    "severity": "info",
                    "code": "lod_analysis_built",
                    "message": f"Built LOD scatter mapping with {len(manifest['lod_capture_records'])} capture record(s).",
                    "draw_indices": [],
                }
            )
            write_json(manifest_path, manifest)
            _update_scene_mapping_payload(scene, manifest_path)
            _replace_lod_mapping_items(scene, manifest["lod_mapping"])
        except Exception as exc:
            self.report({"ERROR"}, f"Analyze LOD failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_lod_manifest_path = manifest_path
        frame_records = list(lod_result.get("lod_frameanalysis", []) or [])
        frame_record = frame_records[0] if frame_records else {}
        shadow_stage = dict(frame_record.get("shadow_stage", {}) or {})
        scene.bmc_lod_shadow_host_hash = str(shadow_stage.get("host_ib_hash", "") or "")
        scene.bmc_lod_shadow_host_match_index_count = int(shadow_stage.get("host_match_index_count", -1) or -1)
        scene.bmc_lod_shadow_host_vs_hash = str(shadow_stage.get("transparent_vs_hash", "") or shadow_stage.get("normal_vs_hash", "") or "")
        matched_count = int(frame_record.get("matched_global_bone_count", 0) or 0)
        total_count = int(frame_record.get("total_global_bone_count", 0) or 0)
        required_count = int(frame_record.get("required_global_bone_count", total_count) or 0)
        ignored_count = int(frame_record.get("ignored_lod_global_bone_count", 0) or 0)
        capture_count = len(manifest.get("lod_capture_records", []) or [])
        review = dict(manifest.get("lod_review", {}) or {})
        runtime_safe = bool(review.get("runtime_safe", False))
        missing_count = int(review.get("missing_global_bone_count", 0) or 0)
        coverage = (matched_count / required_count * 100.0) if required_count > 0 else 0.0
        scene.bmc_lod_match_summary = (
            f"LOD {'OK' if runtime_safe else 'BLOCKED'}: matched {matched_count}/{required_count} "
            f"required bones ({coverage:.1f}%), ignored {ignored_count}, missing {missing_count}, "
            f"{capture_count} capture records"
        )
        self.report(
            {"INFO"},
            f"Analyzed LOD: matched {matched_count}/{required_count} required global bone(s); ignored {ignored_count}; capture records {capture_count}",
        )
        lod_warnings = list(manifest.get("lod_validation", []) or [])
        warning_messages = [
            str(item.get("message", ""))
            for item in lod_warnings
            if str(item.get("severity", "")).lower() in {"warning", "error"} and str(item.get("message", ""))
        ]
        scene.bmc_lod_match_warning = warning_messages[0] if warning_messages else ""
        if warning_messages:
            self.report({"WARNING"}, " | ".join(warning_messages[:3]))
        return {"FINISHED"}


class BMC_OT_preview_lod_fallbacks(bpy.types.Operator):
    bl_idname = "object.bmc_preview_lod_fallbacks"
    bl_label = "Preview LOD Fallbacks"
    bl_description = "Preview inherited donor mappings for LOD globals that are unmatched but used by the export meshes"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_export_collection", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            preview = _lod_export_blocking_preview(scene, manifest)
        except Exception as exc:
            self.report({"ERROR"}, f"Preview LOD fallbacks failed: {exc}")
            return {"CANCELLED"}

        summary = dict(preview.get("summary", {}) or {})
        message = (
            f"Previewed LOD fallbacks: used unmatched {int(summary.get('unmatched_used_count', 0) or 0)}, "
            f"fallback {int(summary.get('fallback_count', 0) or 0)}, "
            f"unresolved {int(summary.get('unresolved_count', 0) or 0)}"
        )
        self.report({"INFO"}, message)
        if scene.bmc_lod_fallback_warning:
            self.report({"WARNING"}, scene.bmc_lod_fallback_warning)
        return {"FINISHED"}


class BMC_OT_apply_lod_fallbacks(bpy.types.Operator):
    bl_idname = "object.bmc_apply_lod_fallbacks"
    bl_label = "Apply LOD Fallbacks"
    bl_description = "Cautiously apply inherited donor mappings for currently used unmatched LOD globals"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_export_collection", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            preview = preview_lod_fallbacks_for_export(scene.bmc_export_collection, manifest, use_export_plan=True)
            fallbacks = list(preview.get("fallbacks", []) or [])
            if not fallbacks:
                _store_lod_fallback_preview_on_scene(scene, preview, applied=False)
                self.report({"WARNING"}, "No LOD fallback donor could be applied")
                return {"CANCELLED"}
            apply_result = apply_lod_fallbacks_to_manifest(manifest, preview)
            manifest["lod_review"] = review_lod_global_pool_coverage(
                manifest,
                list(manifest.get("lod_capture_records", []) or []),
            )
            write_json(manifest_path, manifest)
            _replace_lod_mapping_items(scene, list(manifest.get("lod_mapping", []) or []))
            _store_lod_fallback_preview_on_scene(scene, preview, applied=True)
            _update_scene_mapping_payload(scene, manifest_path)
        except Exception as exc:
            self.report({"ERROR"}, f"Apply LOD fallbacks failed: {exc}")
            return {"CANCELLED"}

        applied_count = int(apply_result.get("applied_count", 0) or 0)
        unresolved_count = len(preview.get("unresolved", []) or [])
        message = f"Applied {applied_count} LOD fallback mapping(s)"
        if unresolved_count:
            message += f"; {unresolved_count} unresolved"
        self.report({"INFO"}, message)
        if scene.bmc_lod_fallback_warning:
            self.report({"WARNING"}, scene.bmc_lod_fallback_warning)
        return {"FINISHED"}


class BMC_OT_analyze_main_frameanalysis(bpy.types.Operator):
    bl_idname = "object.bmc_analyze_main_frameanalysis"
    bl_label = "Analyze Main"
    bl_description = "Analyze Main FrameAnalysis and build the redesigned candidate IB manifest"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_frameanalysis_dir", ""))

    def execute(self, context):
        scene = context.scene
        try:
            payload, manifest_path = write_main_analysis_manifest(
                frameanalysis_dir=bpy.path.abspath(scene.bmc_frameanalysis_dir),
                output_dir=bpy.path.abspath(scene.bmc_output_dir) if scene.bmc_output_dir else "",
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Analyze Main failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_manifest_path = manifest_path
        shadow_stage = payload.get("shadow_stage", {})
        scene.bmc_shadow_host_hash = str(shadow_stage.get("host_ib_hash", "") or "")
        scene.bmc_shadow_host_match_index_count = int(shadow_stage.get("host_match_index_count", -1))
        shadow_vs_hashes = list(shadow_stage.get("shadow_vs_hashes", []) or [])
        scene.bmc_shadow_host_vs_hash = shadow_vs_hashes[-1] if shadow_vs_hashes else ""
        _replace_candidate_items_from_manifest(scene, payload)

        warning_count = sum(1 for item in payload.get("validation", []) if item.get("severity") == "warning")
        message = (
            f"Analyzed {len(payload.get('candidate_ibs', []))} candidate IB(s); "
            f"shadow VS {len(shadow_vs_hashes)}; manifest {os.path.basename(manifest_path)}"
        )
        if warning_count:
            message += f"; warnings {warning_count}"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_import_selected_candidates(bpy.types.Operator):
    bl_idname = "object.bmc_import_selected_candidates"
    bl_label = "Import Selected IBs"
    bl_description = "Import enabled redesigned candidate IBs from the current capture manifest"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_candidate_items", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; run Analyze Main first")
            return {"CANCELLED"}
        selected_names = [
            str(item.display_name)
            for item in scene.bmc_candidate_items
            if item.enabled and str(item.display_name)
        ]
        if not selected_names:
            self.report({"ERROR"}, "No enabled candidate IBs selected")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            target_collection = _ensure_export_collection(context)
            imported_objects = import_selected_candidates(
                context,
                manifest,
                selected_names,
                target_collection,
                mirror_flip=bool(getattr(scene, "bmc_mirror_flip", True)),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Import Selected IBs failed: {exc}")
            return {"CANCELLED"}
        if not imported_objects:
            self.report({"ERROR"}, "No candidate IBs matched the enabled UI list")
            return {"CANCELLED"}
        warning_count = sum(1 for obj in imported_objects if obj.get("bmc_import_warnings"))
        message = f"Imported {len(imported_objects)} candidate IB object(s) into {target_collection.name}"
        if warning_count:
            message += f"; {warning_count} with warnings"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_scan_targets(bpy.types.Operator):
    bl_idname = "object.bmc_scan_targets"
    bl_label = "Scan / Freeze Mapping"
    bl_description = "Scan the chosen FrameAnalysis directory for the frozen target list and generate capture_manifest.json plus BoneStore.ini without modifying source meshes"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(
            scene
            and scene.bmc_frameanalysis_dir
            and any(
                item.enabled and item.ib_hash and item.object_name
                for item in scene.bmc_target_items
            )
        )

    def execute(self, context):
        scene = context.scene
        alias_count = 0
        auto_apply_warning = ""
        auto_apply_updated = 0
        auto_apply_renamed = 0
        auto_apply_merged = 0
        try:
            target_specs = _enabled_target_specs(context)
            result = scan_targets_and_generate_outputs(
                frameanalysis_dir=scene.bmc_frameanalysis_dir,
                target_specs=target_specs,
                output_dir=scene.bmc_output_dir,
                mapping_payload=load_mapping_payload_from_scene(scene),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Scan failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_manifest_path = result.manifest_path
        scene.bmc_ini_path = result.ini_path
        _sync_target_match_counts_from_manifest(scene, scene.bmc_manifest_path)
        scene.bmc_shadow_host_hash = str(result.shadow_host_hash or "")
        scene.bmc_shadow_host_match_index_count = int(result.shadow_host_match_index_count)
        scene.bmc_shadow_host_vs_hash = str(result.shadow_host_vs_hash or "")

        alias_warning = ""
        scene.bmc_alias_items.clear()
        if not scene.bmc_scan_auto_apply_mapping:
            scene.bmc_mapping_payload_json = ""

        _update_scene_mapping_payload(scene, result.manifest_path)
        if scene.bmc_scan_auto_apply_mapping:
            try:
                payload = _resolve_active_mapping_payload(scene)
                matched_meshes, selected_entries, skipped_before_apply = _select_object_remap_entries_for_targets(
                    scene,
                    payload,
                )
                if matched_meshes:
                    identity_resolver = _identity_resolver_from_entries(matched_meshes, selected_entries)
                    remap_result = apply_group_remaps_to_meshes(
                        matched_meshes,
                        {"object_remaps": selected_entries},
                        identity_resolver=identity_resolver,
                    )
                    auto_apply_updated = int(remap_result.updated_objects)
                    auto_apply_renamed = int(remap_result.renamed_groups)
                    merge_messages: list[str] = []
                    remapped_meshes = [
                        mesh_obj
                        for mesh_obj in matched_meshes
                        if mesh_obj.name not in _skipped_name_set(remap_result.skipped_objects)
                    ]
                    if remapped_meshes:
                        try:
                            alias_count = _analyze_same_bone_aliases(
                                context,
                                result.manifest_path,
                                _enabled_target_names(scene),
                            )
                            _update_scene_mapping_payload(scene, result.manifest_path)
                            alias_payload = _enabled_alias_payload(scene) or list(
                                _resolve_active_mapping_payload(scene).get("bone_aliases", [])
                            )
                            if alias_payload:
                                merge_result = merge_duplicate_alias_weights(
                                    remapped_meshes,
                                    alias_payload,
                                    identity_resolver=identity_resolver,
                                )
                                auto_apply_merged = int(merge_result.merged_aliases)
                                merge_messages.extend(list(merge_result.skipped_objects))
                        except Exception as exc:
                            scene.bmc_alias_items.clear()
                            alias_warning = f"Same-bone alias analysis failed after global rename: {exc}"
                    combined_messages = list(skipped_before_apply) + list(remap_result.skipped_objects) + list(merge_messages)
                    if combined_messages:
                        auto_apply_warning = " | ".join(combined_messages[:3])
                elif skipped_before_apply:
                    auto_apply_warning = " | ".join(skipped_before_apply[:3])
            except Exception as exc:
                auto_apply_warning = f"Auto apply after Scan failed: {exc}"

        info_message = f"Scanned {result.scanned_parts} parts; total global bones {result.total_global_bones}"
        if scene.bmc_scan_auto_apply_mapping:
            info_message += (
                f"; auto remap {auto_apply_updated}"
                f"; renamed {auto_apply_renamed}"
                f"; merged {auto_apply_merged}"
                f"; aliases {alias_count}"
            )
        if scene.bmc_shadow_host_hash:
            info_message += (
                f"; shadow host {scene.bmc_shadow_host_hash}-{scene.bmc_shadow_host_match_index_count}"
                f" vs={scene.bmc_shadow_host_vs_hash or '?'}"
            )
        if scene.bmc_scan_auto_apply_mapping:
            info_message += "; target meshes auto-updated"
        else:
            info_message += "; source meshes unchanged"
        self.report({"INFO"}, info_message)
        if result.warnings:
            self.report({"WARNING"}, " | ".join(result.warnings[:3]))
        if alias_warning:
            self.report({"WARNING"}, alias_warning)
        if auto_apply_warning:
            self.report({"WARNING"}, auto_apply_warning)
        return {"FINISHED"}


class BMC_OT_analyze_duplicate_bones(bpy.types.Operator):
    bl_idname = "object.bmc_analyze_duplicate_bones"
    bl_label = "Rebuild Same-Bone Aliases"
    bl_description = "Rebuild seam-filtered same-bone recommendations from the current manifest and frozen target meshes"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and _enabled_target_names(scene))

    def execute(self, context):
        scene = context.scene
        try:
            alias_count = _analyze_same_bone_aliases(
                context,
                scene.bmc_manifest_path,
                _enabled_target_names(scene),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Same-bone analysis failed: {exc}")
            return {"CANCELLED"}

        _update_scene_mapping_payload(scene, scene.bmc_manifest_path)
        self.report({"INFO"}, f"Found {alias_count} same-bone alias candidate(s)")
        return {"FINISHED"}


class BMC_OT_apply_vertex_group_remap(bpy.types.Operator):
    bl_idname = "object.bmc_apply_vertex_group_remap"
    bl_label = "Rename Target Vertex Groups"
    bl_description = "Rename enabled Target objects' local vertex groups to global bone indices using the frozen mapping"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and _enabled_target_names(context.scene))

    def execute(self, context):
        try:
            payload = _resolve_active_mapping_payload(context.scene)
            matched_meshes, selected_entries, skipped_before_apply = _select_object_remap_entries_for_targets(
                context.scene,
                payload,
            )
            if not matched_meshes:
                raise ValueError("No enabled Target objects matched any object_remap entry in the loaded mapping payload")
            result, _merged_aliases, _merge_warning_messages = _apply_mapping_bundle(
                matched_meshes,
                selected_entries,
                [],
                identity_resolver=_identity_resolver_from_entries(matched_meshes, selected_entries),
                merge_same_bone_groups=False,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Remap failed: {exc}")
            return {"CANCELLED"}

        skipped_after_apply = list(result.skipped_objects)
        skipped_after_apply.extend(skipped_before_apply)

        self.report(
            {"INFO"},
            f"Renamed {result.updated_objects}/{len(matched_meshes)} target meshes; renamed {result.renamed_groups} groups",
        )
        if skipped_after_apply:
            self.report({"WARNING"}, " | ".join(skipped_after_apply[:3]))
        return {"FINISHED"}


class BMC_OT_merge_duplicate_bones(bpy.types.Operator):
    bl_idname = "object.bmc_merge_duplicate_bones"
    bl_label = "Fast Merge Same-Bone Groups"
    bl_description = "Fast-merge same-bone groups on enabled Target objects by renaming source groups to canonical groups and leaving empty placeholders"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and _enabled_target_names(context.scene))

    def execute(self, context):
        target_meshes = _enabled_target_mesh_objects(context.scene)
        if not target_meshes:
            self.report({"ERROR"}, "No enabled Target mesh objects found")
            return {"CANCELLED"}
        try:
            rebuilt_alias_count = None
            if context.scene.bmc_manifest_path:
                rebuilt_alias_count = _analyze_same_bone_aliases(
                    context,
                    context.scene.bmc_manifest_path,
                    _enabled_target_names(context.scene),
                )
                _update_scene_mapping_payload(context.scene, context.scene.bmc_manifest_path)
            payload = _resolve_active_mapping_payload(context.scene)
            result = merge_duplicate_alias_weights(
                target_meshes,
                _enabled_alias_payload(context.scene) or list(payload.get("bone_aliases", [])),
                identity_resolver=_identity_resolver_from_targets(context.scene),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Duplicate merge failed: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            (
                f"Fast-merged {result.merged_aliases} aliases across "
                f"{result.updated_objects}/{len(target_meshes)} target meshes"
                + (f"; rebuilt {rebuilt_alias_count} seam aliases" if rebuilt_alias_count is not None else "")
            ),
        )
        if result.skipped_objects:
            self.report({"WARNING"}, " | ".join(result.skipped_objects[:3]))
        return {"FINISHED"}


class BMC_OT_alias_add(bpy.types.Operator):
    bl_idname = "object.bmc_alias_add"
    bl_label = "Add Alias"
    bl_description = "Add a manual duplicate-bone alias entry"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        scene.bmc_alias_items.add()
        scene.bmc_alias_index = len(scene.bmc_alias_items) - 1
        return {"FINISHED"}


class BMC_OT_alias_remove(bpy.types.Operator):
    bl_idname = "object.bmc_alias_remove"
    bl_label = "Remove Alias"
    bl_description = "Remove the active duplicate-bone alias entry"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.bmc_alias_items)

    def execute(self, context):
        scene = context.scene
        scene.bmc_alias_items.remove(scene.bmc_alias_index)
        scene.bmc_alias_index = min(scene.bmc_alias_index, max(0, len(scene.bmc_alias_items) - 1))
        return {"FINISHED"}


class BMC_OT_save_preset(bpy.types.Operator):
    bl_idname = "object.bmc_save_preset"
    bl_label = "Save Mapping Preset"
    bl_description = "Save the current frozen scan mapping, alias list, and shadow host as a single JSON preset"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        try:
            base_payload = load_mapping_payload_from_scene(scene)
            payload = build_mapping_payload(scene, manifest_payload={}, base_payload=base_payload)
            if not payload.get("object_remaps"):
                raise ValueError("No frozen scan mapping is loaded. Run Scan first or load a Mapping Preset.")
            preset_name = scene.bmc_preset_name or scene.bmc_preset_choice
            preset_path = save_preset(preset_name, payload)
            store_mapping_payload_on_scene(scene, payload)
            scene.bmc_preset_name = os.path.splitext(os.path.basename(preset_path))[0]
            scene.bmc_preset_choice = scene.bmc_preset_name
        except Exception as exc:
            self.report({"ERROR"}, f"Save preset failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Preset saved: {preset_path}")
        return {"FINISHED"}


class BMC_OT_load_preset(bpy.types.Operator):
    bl_idname = "object.bmc_load_preset"
    bl_label = "Load Mapping Preset"
    bl_description = "Load the selected mapping preset without overwriting runtime export paths"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        preset_name = scene.bmc_preset_choice
        if not preset_name or preset_name == "__NONE__":
            self.report({"ERROR"}, "No preset selected")
            return {"CANCELLED"}
        try:
            payload = load_preset(preset_name)
            apply_mapping_payload_to_scene(scene, payload, preset_name=preset_name)
        except Exception as exc:
            self.report({"ERROR"}, f"Load preset failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Preset loaded: {preset_name}")
        return {"FINISHED"}


class BMC_OT_delete_preset(bpy.types.Operator):
    bl_idname = "object.bmc_delete_preset"
    bl_label = "Delete Preset"
    bl_description = "Delete the selected preset"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        preset_name = scene.bmc_preset_choice
        if not preset_name or preset_name == "__NONE__":
            self.report({"ERROR"}, "No preset selected")
            return {"CANCELLED"}
        try:
            delete_preset(preset_name)
        except Exception as exc:
            self.report({"ERROR"}, f"Delete preset failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Preset deleted: {preset_name}")
        return {"FINISHED"}
