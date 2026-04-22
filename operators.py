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
    serialize_alias_items,
    store_mapping_payload_on_scene,
)
from .core.presets import delete_preset, load_preset, save_preset
from .core.export_prepare import prepare_export_collection
from .core.frameanalysis import detect_last_shadow_host, infer_mesh_identity_from_name
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


def _ensure_export_chunk_collection(parent_collection, ib_hash: str, match_index_count: int, chunk_index: int = 0):
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
        if not (item.enabled and item.ib_hash and int(item.match_index_count) >= 0):
            continue
        key = (item.ib_hash.lower(), int(item.match_index_count), 0)
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
        key = (identity[0].lower(), int(identity[1]), 0)
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
    if not item.ib_hash or int(item.match_index_count) < 0:
        identity = resolve_mesh_identity(mesh_obj)
        if identity is not None:
            item.ib_hash = identity[0]
            item.match_index_count = int(identity[1])
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

        chunk_name = f"{identity[0].lower()}-{int(identity[1])}-0"
        if chunk_name not in existing_chunk_names:
            created_count += 1
            existing_chunk_names.add(chunk_name)
        chunk_collection = _ensure_export_chunk_collection(export_collection, identity[0], int(identity[1]), 0)

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
    item.match_index_count = identity[1]
    item.local_bone_count = int(local_bone_count)
    item.autodetected = bool(getattr(mesh_obj, "merge_ib_autodetected", True))
    item.enabled = True
    existing_names.add(mesh_obj.name)
    return True


def _enabled_target_specs(context) -> list[TargetObjectSpec]:
    scene = context.scene
    target_specs: list[TargetObjectSpec] = []
    for item in scene.bmc_target_items:
        if not (item.enabled and item.ib_hash and int(item.match_index_count) >= 0 and item.object_name):
            continue
        mesh_obj = _refresh_target_item(scene, item)
        if mesh_obj is None:
            raise ValueError(f"{item.object_name}: mesh object not found in current scene")
        display_name = mesh_obj.name
        manifest_bone_count = _lookup_capture_bone_count_from_manifest(
            scene,
            item.ib_hash.lower(),
            int(item.match_index_count),
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
                ib_hash=item.ib_hash.lower(),
                match_index_count=int(item.match_index_count),
                local_bone_count=local_bone_count,
            )
        )
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
    normalized_count = int(match_index_count)
    fallback_count = None
    for part_record in manifest.get("part_records", []):
        if str(part_record.get("ib_hash", "")).lower() != normalized_hash:
            continue
        if int(part_record.get("match_index_count", -1)) != normalized_count:
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
    entries_by_identity: dict[tuple[str, int], list[dict]] = {}
    for entry in payload.get("object_remaps", []):
        key = (str(entry.get("ib_hash", "")).lower(), int(entry.get("match_index_count", -1)))
        entries_by_identity.setdefault(key, []).append(entry)

    matched_meshes: list[object] = []
    selected_entries: list[dict] = []
    skipped_messages: list[str] = []

    for mesh_obj in mesh_objects:
        identity = _mesh_identity_from_name(mesh_obj)
        if identity is None:
            skipped_messages.append(f"{mesh_obj.name}: object name does not contain hash-count")
            continue
        remap_entries = entries_by_identity.get(identity, [])
        if not remap_entries:
            skipped_messages.append(f"{mesh_obj.name}: no mapping entry for {identity[0]}-{identity[1]}")
            continue
        chosen_entry = _choose_identity_remap_entry(remap_entries, mesh_obj.name)
        if chosen_entry is None:
            skipped_messages.append(f"{mesh_obj.name}: multiple mapping entries exist for {identity[0]}-{identity[1]}")
            continue
        matched_meshes.append(mesh_obj)
        selected_entries.append(chosen_entry)

    return matched_meshes, selected_entries, skipped_messages


def _select_object_remap_entries_for_targets(scene, payload: dict) -> tuple[list[object], list[dict], list[str]]:
    entries_by_identity: dict[tuple[str, int], list[dict]] = {}
    for entry in payload.get("object_remaps", []):
        key = (str(entry.get("ib_hash", "")).lower(), int(entry.get("match_index_count", -1)))
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

        identity = (str(item.ib_hash or "").lower(), int(item.match_index_count))
        if not identity[0] or identity[1] < 0:
            skipped_messages.append(f"{mesh_obj.name}: frozen target identity is incomplete")
            continue

        remap_entries = entries_by_identity.get(identity, [])
        if not remap_entries:
            skipped_messages.append(f"{mesh_obj.name}: no mapping entry for {identity[0]}-{identity[1]}")
            continue
        chosen_entry = _choose_identity_remap_entry(remap_entries, mesh_obj.name)
        if chosen_entry is None:
            skipped_messages.append(f"{mesh_obj.name}: multiple mapping entries exist for {identity[0]}-{identity[1]}")
            continue
        matched_meshes.append(mesh_obj)
        selected_entries.append(chosen_entry)

    return matched_meshes, selected_entries, skipped_messages


def _identity_resolver_from_entries(mesh_objects: list[object], selected_entries: list[dict]):
    identity_by_name = {
        mesh_obj.name: (str(entry.get("ib_hash", "")).lower(), int(entry.get("match_index_count", -1)))
        for mesh_obj, entry in zip(mesh_objects, selected_entries)
    }

    def _resolver(mesh_obj):
        identity = identity_by_name.get(mesh_obj.name)
        if identity is not None and identity[0] and identity[1] >= 0:
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
        match_index_count = int(item.match_index_count)
        if ib_hash and match_index_count >= 0:
            identity_by_name[mesh_obj.name] = (ib_hash, match_index_count)

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
                self.report({"WARNING"}, f"{mesh_obj.name}: cannot infer ib_hash/index_count")
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
    bl_description = "Explicitly re-freeze the active target's IB hash, match count, and local bone count from the current object"
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
            self.report({"ERROR"}, f"{mesh_obj.name}: cannot infer ib_hash/index_count")
            return {"CANCELLED"}

        try:
            local_bone_count = int(infer_local_bone_count_from_mesh(mesh_obj))
        except ValueError as exc:
            self.report({"ERROR"}, f"{mesh_obj.name}: {exc}")
            return {"CANCELLED"}

        item.object_ref = mesh_obj
        item.object_name = mesh_obj.name
        item.ib_hash = identity[0]
        item.match_index_count = int(identity[1])
        item.local_bone_count = local_bone_count
        item.autodetected = bool(getattr(mesh_obj, "merge_ib_autodetected", True))
        self.report(
            {"INFO"},
            f"Frozen target {mesh_obj.name} as {item.ib_hash}-{item.match_index_count} with {item.local_bone_count} local bones",
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
            chunk_collection = _ensure_export_chunk_collection(collection, identity[0], int(identity[1]), 0)
            _link_object_to_collection(mesh_obj, chunk_collection)
            added_count += 1

        if added_count == 0:
            if skipped_names:
                self.report({"WARNING"}, f"Skipped {len(skipped_names)} mesh(es): cannot infer ib_hash/index_count")
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
                item.enabled and item.ib_hash and int(item.match_index_count) >= 0 and item.object_name
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
                merge_same_bone_groups=False,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Scan failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_manifest_path = result.manifest_path
        scene.bmc_ini_path = result.ini_path
        shadow_host_warning = ""
        scene.bmc_shadow_host_hash = ""
        scene.bmc_shadow_host_match_index_count = -1
        scene.bmc_shadow_host_vs_hash = ""
        try:
            shadow_host = detect_last_shadow_host(scene.bmc_frameanalysis_dir)
            scene.bmc_shadow_host_hash = shadow_host.ib_hash
            scene.bmc_shadow_host_match_index_count = int(shadow_host.match_index_count)
            scene.bmc_shadow_host_vs_hash = shadow_host.vs_hash
        except Exception as exc:
            shadow_host_warning = f"Last shadow host detection failed: {exc}"

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
        if shadow_host_warning:
            self.report({"WARNING"}, shadow_host_warning)
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
