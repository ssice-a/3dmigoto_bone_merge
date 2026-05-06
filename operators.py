"""Blender operators for the Bone Merge Capture plugin."""

from __future__ import annotations

import json
import os

import bpy
from bpy.app.handlers import persistent

from .constants import BI4_MAX_BONE_COUNT
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
from .core.frameanalysis import infer_mesh_identity_from_name
from .core.io import read_json
from .core.import_candidates import import_selected_candidates
from .core.lod_runtime import build_lod_mapping, scan_lod_targets_and_generate_manifest
from .core.main_analyze import write_main_analysis_manifest
from .core.shadow_split import generate_shadow_split
from .core.seam_matcher import apply_seam_mapping, build_seam_mapping
from .core.workflow import scan_targets_and_generate_outputs
from .core.models import TargetObjectSpec

DEFAULT_TARGET_COLLECTION_NAME = "BMC Bone Palette Targets"
DEFAULT_EXPORT_COLLECTION_NAME = "BMC Export Sources"
DEFAULT_EXPORT_BUILD_COLLECTION_NAME = "BMC Export Build"
_EXPORT_NORMALIZE_GUARD = False


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


def _ensure_export_build_collection(context):
    scene = context.scene
    collection = scene.bmc_export_build_collection
    if collection is None:
        collection = bpy.data.collections.get(DEFAULT_EXPORT_BUILD_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(DEFAULT_EXPORT_BUILD_COLLECTION_NAME)
        scene.collection.children.link(collection)
    scene.bmc_export_build_collection = collection
    return collection


def _ensure_export_chunk_collection(parent_collection, ib_hash: str, match_index_count: int = 0, chunk_index: int = 0):
    chunk_name = f"{ib_hash.lower()}-{int(match_index_count)}-{int(chunk_index)}"
    collection = bpy.data.collections.get(chunk_name)
    if collection is None:
        collection = bpy.data.collections.new(chunk_name)
    if all(child.name != collection.name for child in parent_collection.children):
        parent_collection.children.link(collection)
    return collection


def _ensure_export_chunk_collections_from_targets(context, parent_collection) -> int:
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
        if _ensure_export_chunk_collection_if_missing(parent_collection, key[0], key[1], key[2]):
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
        key = (identity[0].lower(), int(match_index_count), 0)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if _ensure_export_chunk_collection_if_missing(parent_collection, key[0], key[1], key[2]):
            created_count += 1

    return created_count


def _ensure_export_chunk_collection_if_missing(
    parent_collection,
    ib_hash: str,
    match_index_count: int,
    chunk_index: int = 0,
) -> bool:
    chunk_name = f"{ib_hash.lower()}-{int(match_index_count)}-{int(chunk_index)}"
    existed_under_parent = any(child.name == chunk_name for child in parent_collection.children)
    _ensure_export_chunk_collection(parent_collection, ib_hash, match_index_count, chunk_index)
    return not existed_under_parent


def _link_object_to_collection(mesh_obj, collection) -> None:
    if any(obj.name == mesh_obj.name for obj in collection.objects):
        return
    collection.objects.link(mesh_obj)


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
    existing_chunk_names = {child.name for child in export_collection.children}
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
        chunk_name = f"{identity[0].lower()}-{int(match_index_count)}-0"
        if chunk_name not in existing_chunk_names:
            created_count += 1
            existing_chunk_names.add(chunk_name)
        chunk_collection = _ensure_export_chunk_collection(export_collection, identity[0], match_index_count, 0)

        moved_here = False
        if all(obj.name != mesh_obj.name for obj in chunk_collection.objects):
            chunk_collection.objects.link(mesh_obj)
            moved_here = True

        for collection in tuple(mesh_obj.users_collection):
            if collection.as_pointer() not in subtree_ids or collection == chunk_collection:
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
        item.mapped_lod_global_bone = int(mapping_entry.get("mapped_lod_global_bone", -1))
        item.status = str(mapping_entry.get("status", ""))
        item.score = float(mapping_entry.get("score", 0.0))
        item.note = str(mapping_entry.get("note", ""))
    scene.bmc_lod_mapping_index = min(scene.bmc_lod_mapping_index, max(0, len(scene.bmc_lod_mapping_items) - 1))


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
                "this workflow requires each final object/draw chunk to stay within 256 bones"
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
        item.local_bone_count = int(candidate.get("local_bone_count", 0))
        item.draw_count = len(candidate.get("draw_indices", []) or [])
        item.shadow_draw_count = len(candidate.get("shadow_draw_indices", []) or [])
        item.manual = False
    scene.bmc_candidate_index = min(scene.bmc_candidate_index, max(0, len(scene.bmc_candidate_items) - 1))


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
        return bool(context.scene and context.scene.bmc_target_collection)

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
        return bool(scene and scene.bmc_target_items and 0 <= scene.bmc_target_index < len(scene.bmc_target_items))

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
        build_collection = _ensure_export_build_collection(context)
        _refresh_target_items(context.scene)
        created_count = _ensure_export_chunk_collections_from_targets(context, source_collection)
        message = f"Export source: {source_collection.name}; build: {build_collection.name}"
        if created_count:
            message += f"; created {created_count} chunk collection(s)"
        else:
            message += "; no new chunk collections"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_add_selected_export_objects(bpy.types.Operator):
    bl_idname = "object.bmc_add_selected_export_objects"
    bl_label = "Add Selected To Export"
    bl_description = "Add selected meshes to a host chunk child collection inside the export collection"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context):
        added_count = 0
        skipped_names: list[str] = []
        collection = _ensure_export_collection(context)

        for mesh_obj in context.selected_objects:
            if mesh_obj.type != "MESH":
                continue
            identity = resolve_mesh_identity(mesh_obj)
            if identity is None:
                skipped_names.append(mesh_obj.name)
                continue
            match_index_count = _resolve_target_match_index_count(context.scene, identity[0], mesh_obj.name, int(identity[1]))
            if match_index_count <= 0:
                skipped_names.append(mesh_obj.name)
                continue
            chunk_collection = _ensure_export_chunk_collection(collection, identity[0], match_index_count, 0)
            _link_object_to_collection(mesh_obj, chunk_collection)
            added_count += 1

        if added_count == 0:
            if skipped_names:
                self.report({"WARNING"}, f"Skipped {len(skipped_names)} mesh(es): cannot infer ib_hash or scanned match count")
            else:
                self.report({"WARNING"}, "No new mesh objects were added to export source collection")
            return {"CANCELLED"}

        message = f"Added {added_count} object(s) to export source collection"
        if skipped_names:
            message += f"; skipped {len(skipped_names)}"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_prepare_export_collection(bpy.types.Operator):
    bl_idname = "object.bmc_prepare_export_collection"
    bl_label = "Prepare Export / Palette"
    bl_description = "Rebuild Export Build from Export Source, localize only the build copies, and write Palette.buf / export_manifest.json"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and context.scene.bmc_export_collection)

    def execute(self, context):
        scene = context.scene
        try:
            source_collection = _ensure_export_collection(context)
            build_collection = _ensure_export_build_collection(context)
            result = prepare_export_collection(
                context=context,
                source_collection=source_collection,
                build_collection=build_collection,
                output_dir=scene.bmc_output_dir,
                internal_manifest_dir=None,
                capture_manifest_path=scene.bmc_manifest_path,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Prepare export failed: {exc}")
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
        self.report(
            {"INFO"},
            f"Prepared {result['objects']} object(s), {result['palettes']} palette(s); wrote 3Dmigoto files to {result['output_dir']}",
        )
        return {"FINISHED"}


class BMC_OT_generate_shadow_split(bpy.types.Operator):
    bl_idname = "object.bmc_generate_shadow_split"
    bl_label = "Modify Main INI"
    bl_description = "Rewrite the source mod INI so vs==200 shadow draws move to the last shadow host and consume the latest exported BoneStore local palettes"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        has_manual_shadow_host = bool(
            scene
            and scene.bmc_shadow_host_hash
            and int(scene.bmc_shadow_host_match_index_count) > 0
        )
        return bool(
            scene
            and (scene.bmc_frameanalysis_dir or has_manual_shadow_host)
            and scene.bmc_export_manifest_path
            and scene.bmc_ini_path
            and scene.bmc_source_ini_path
        )

    def execute(self, context):
        scene = context.scene
        try:
            result = generate_shadow_split(
                frameanalysis_dir=bpy.path.abspath(scene.bmc_frameanalysis_dir),
                export_manifest_path=bpy.path.abspath(scene.bmc_export_manifest_path),
                bonestore_ini_path=bpy.path.abspath(scene.bmc_ini_path),
                source_ini_path=bpy.path.abspath(scene.bmc_source_ini_path),
                shadow_host_hash=scene.bmc_shadow_host_hash,
                shadow_host_match_index_count=int(scene.bmc_shadow_host_match_index_count),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Shadow split failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_source_ini_path = result.source_ini_path
        scene.bmc_shadow_host_hash = result.shadow_host_hash
        scene.bmc_shadow_host_match_index_count = int(result.shadow_host_match_index_count)
        scene.bmc_shadow_host_vs_hash = result.shadow_host_vs_hash
        self.report(
            {"INFO"},
            f"Shadow split updated {result.rewritten_sections} section(s); migrated {result.migrated_chunks} chunk(s) via host {result.shadow_host_hash}-{result.shadow_host_match_index_count}",
        )
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
        return bool(context.scene and context.scene.bmc_lod_target_items)

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
        return bool(context.scene and context.scene.bmc_lod_target_items)

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
        return bool(context.scene and context.scene.bmc_lod_target_items)

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
        return bool(scene and scene.bmc_manifest_path and scene.bmc_ini_path)

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
        return bool(context.scene and context.scene.bmc_seam_match_items)

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
        return bool(context.scene and context.scene.bmc_seam_match_items)

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
        return bool(context.scene and context.scene.bmc_seam_alias_items)

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


class BMC_OT_analyze_main_frameanalysis(bpy.types.Operator):
    bl_idname = "object.bmc_analyze_main_frameanalysis"
    bl_label = "Analyze Main"
    bl_description = "Analyze Main FrameAnalysis and build the redesigned candidate IB manifest"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and scene.bmc_frameanalysis_dir)

    def execute(self, context):
        scene = context.scene
        try:
            target_hashes = _main_analyze_target_hashes(scene)
            payload, manifest_path = write_main_analysis_manifest(
                frameanalysis_dir=bpy.path.abspath(scene.bmc_frameanalysis_dir),
                target_ib_hashes=target_hashes,
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
        return bool(scene and scene.bmc_manifest_path and scene.bmc_candidate_items)

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
            imported_objects = import_selected_candidates(context, manifest, selected_names, target_collection)
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
        return bool(scene and scene.bmc_manifest_path and _enabled_target_names(scene))

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
