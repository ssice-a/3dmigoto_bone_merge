"""Blender operators for the Bone Merge Capture plugin."""

from __future__ import annotations

import json
import os

import bpy
from bpy.app.handlers import persistent

from .constants import BI4_MAX_BONE_COUNT
from .core.blender_ops import (
    annotate_alias_items_with_mesh_proximity,
    build_seam_filtered_aliases_from_manifest,
    infer_local_bone_count_from_mesh,
    resolve_mesh_identity,
)
from .core.presets import delete_preset, load_preset, save_preset
from .core.export_prepare import prepare_export_collection
from .core.workflow import (
    apply_vertex_group_remap_for_target_names,
    merge_duplicate_bone_weights_for_target_names,
    scan_targets_and_generate_outputs,
)
from .core.models import TargetObjectSpec

DEFAULT_TARGET_COLLECTION_NAME = "BMC Bone Palette Targets"
DEFAULT_EXPORT_COLLECTION_NAME = "BMC Export Sources"
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
    identity = resolve_mesh_identity(mesh_obj)
    if identity is not None:
        item.ib_hash = identity[0]
        item.match_index_count = int(identity[1])
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
    created_chunk_names = {child.name for child in export_collection.children}
    moved_count = 0
    skipped_count = 0

    for mesh_obj in tuple(_iter_mesh_objects_in_collection_subtree(export_collection)):
        identity = resolve_mesh_identity(mesh_obj)
        if identity is None:
            skipped_count += 1
            continue

        chunk_name = f"{identity[0].lower()}-{int(identity[1])}-0"
        chunk_collection = _ensure_export_chunk_collection(export_collection, identity[0], int(identity[1]), 0)
        if chunk_name not in created_chunk_names:
            created_chunk_names.add(chunk_name)

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
        "created": len(created_chunk_names),
        "skipped": skipped_count,
    }


def _add_mesh_object_to_target_list(scene, mesh_obj, existing_names: set[str]) -> bool:
    identity = resolve_mesh_identity(mesh_obj)
    if identity is None:
        return False
    if mesh_obj.name in existing_names:
        return False
    item = scene.bmc_target_items.add()
    item.object_name = mesh_obj.name
    item.object_ref = mesh_obj
    item.ib_hash = identity[0]
    item.match_index_count = identity[1]
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
        local_bone_count = infer_local_bone_count_from_mesh(mesh_obj)
        if local_bone_count > BI4_MAX_BONE_COUNT:
            raise ValueError(
                f"{item.object_name}: local bone count {local_bone_count} exceeds BI4 limit {BI4_MAX_BONE_COUNT}; "
                "this workflow requires each final object/draw chunk to stay within 256 bones"
            )
        target_specs.append(
            TargetObjectSpec(
                object_name=item.object_name,
                ib_hash=item.ib_hash.lower(),
                match_index_count=int(item.match_index_count),
                local_bone_count=local_bone_count,
            )
        )
    return target_specs


def _enabled_target_names(scene) -> list[str]:
    target_names: list[str] = []
    for item in scene.bmc_target_items:
        if not item.enabled:
            continue
        mesh_obj = _refresh_target_item(scene, item)
        if mesh_obj is None:
            continue
        target_names.append(mesh_obj.name)
    return target_names


def _enabled_alias_payload(scene) -> list[dict]:
    payload = []
    for item in scene.bmc_alias_items:
        payload.append(
            {
                "enabled": bool(item.enabled),
                "src_draw_index": int(item.src_draw_index),
                "src_object_name": item.src_object_name,
                "src_ib_hash": item.src_ib_hash.lower(),
                "src_local_bone": int(item.src_local_bone),
                "src_global_bone": int(item.src_global_bone),
                "canonical_draw_index": int(item.canonical_draw_index),
                "canonical_object_name": item.canonical_object_name,
                "canonical_ib_hash": item.canonical_ib_hash.lower(),
                "canonical_local_bone": int(item.canonical_local_bone),
                "canonical_global_bone": int(item.canonical_global_bone),
                "confidence": item.confidence,
            }
        )
    return payload


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


def _serialize_scene_preset(scene) -> dict:
    _refresh_target_items(scene)
    return {
        "merge_same_bone_groups": bool(scene.bmc_merge_same_bone_groups),
        "targets": [
            {
                "object_name": item.object_name,
                "ib_hash": item.ib_hash,
                "match_index_count": int(item.match_index_count),
                "autodetected": bool(item.autodetected),
                "enabled": bool(item.enabled),
            }
            for item in scene.bmc_target_items
        ],
        "aliases": _enabled_alias_payload(scene),
    }


def _apply_loaded_preset(scene, payload: dict):
    scene.bmc_merge_same_bone_groups = bool(payload.get("merge_same_bone_groups", False))
    scene.bmc_target_items.clear()
    for target in payload.get("targets", []):
        item = scene.bmc_target_items.add()
        item.object_name = str(target.get("object_name", ""))
        item.ib_hash = str(target.get("ib_hash", ""))
        item.match_index_count = int(target.get("match_index_count", 0))
        item.autodetected = bool(target.get("autodetected", False))
        item.enabled = bool(target.get("enabled", True))
    scene.bmc_target_index = min(scene.bmc_target_index, max(0, len(scene.bmc_target_items) - 1))

    scene.bmc_alias_items.clear()
    for alias in payload.get("aliases", []):
        item = scene.bmc_alias_items.add()
        item.enabled = bool(alias.get("enabled", True))
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
        existing_names = {item.object_name for item in scene.bmc_target_items if item.object_name}
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

        _refresh_target_items(scene)
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

        _refresh_target_items(scene)
        scene.bmc_target_index = min(scene.bmc_target_index, max(0, len(scene.bmc_target_items) - 1))
        message = f"Synced {added_count} target object(s) from {collection.name}"
        if skipped_count:
            message += f"; skipped {skipped_count}"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_create_export_collection(bpy.types.Operator):
    bl_idname = "object.bmc_create_export_collection"
    bl_label = "Create Export Collection"
    bl_description = "Create or select the collection that stores final meshes to prepare for export"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        collection = _ensure_export_collection(context)
        _refresh_target_items(context.scene)
        created_count = _ensure_export_chunk_collections_from_targets(context, collection)
        normalize_result = _normalize_export_collection_membership(context.scene)
        message = f"Export collection: {collection.name}"
        if created_count:
            message += f"; created {created_count} chunk collection(s)"
        else:
            message += "; no new chunk collections"
        if normalize_result["moved"]:
            message += f"; normalized {normalize_result['moved']} object(s)"
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

        normalize_result = _normalize_export_collection_membership(context.scene)
        if added_count == 0 and not normalize_result["moved"]:
            if skipped_names:
                self.report({"WARNING"}, f"Skipped {len(skipped_names)} mesh(es): cannot infer ib_hash/index_count")
            else:
                self.report({"WARNING"}, "No new mesh objects were added to export collection")
            return {"CANCELLED"}

        message = f"Added {added_count} object(s) to export collection"
        if normalize_result["moved"]:
            message += f"; normalized {normalize_result['moved']} object(s)"
        if skipped_names:
            message += f"; skipped {len(skipped_names)}"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_prepare_export_collection(bpy.types.Operator):
    bl_idname = "object.bmc_prepare_export_collection"
    bl_label = "Prepare Export / Palette"
    bl_description = "Localize export chunk vertex groups in place and write Palette.buf files"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and context.scene.bmc_export_collection)

    def execute(self, context):
        scene = context.scene
        try:
            _refresh_target_items(scene)
            _normalize_export_collection_membership(scene)
            result = prepare_export_collection(
                context=context,
                export_collection=scene.bmc_export_collection,
                output_dir=scene.bmc_output_dir,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Prepare export failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_export_manifest_path = result["manifest_path"]
        if result.get("bonestore_ini_path"):
            scene.bmc_ini_path = result["bonestore_ini_path"]
        self.report(
            {"INFO"},
            f"Prepared {result['objects']} object(s), {result['palettes']} palette(s) in {result['export_collection_name']}",
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


class BMC_OT_scan_targets(bpy.types.Operator):
    bl_idname = "object.bmc_scan_targets"
    bl_label = "Scan and Generate"
    bl_description = "Scan the chosen FrameAnalysis directory for the listed objects and generate capture_manifest.json plus BoneStore.ini"
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

        remap_result = None
        try:
            remap_result = apply_vertex_group_remap_for_target_names(
                context=context,
                manifest_path=result.manifest_path,
                target_object_names=_enabled_target_names(scene),
            )
        except Exception as exc:
            self.report({"WARNING"}, f"Scan succeeded but auto-remap failed: {exc}")

        scene.bmc_alias_items.clear()
        auto_alias_count = 0
        auto_merge_result = None
        auto_merge_warning = ""
        if scene.bmc_merge_same_bone_groups:
            try:
                target_names = _enabled_target_names(scene)
                auto_alias_count = _analyze_same_bone_aliases(context, result.manifest_path, target_names)
                if auto_alias_count:
                    auto_merge_result = merge_duplicate_bone_weights_for_target_names(
                        context=context,
                        target_object_names=target_names,
                        alias_entries=_enabled_alias_payload(scene),
                    )
            except Exception as exc:
                auto_merge_warning = f"Auto same-bone merge failed: {exc}"

        info_message = f"Scanned {result.scanned_parts} parts; total global bones {result.total_global_bones}"
        if remap_result is not None:
            info_message += f"; auto-renamed {remap_result.renamed_groups} groups on {remap_result.updated_objects} object(s)"
        if scene.bmc_merge_same_bone_groups:
            merged_aliases = auto_merge_result.merged_aliases if auto_merge_result is not None else 0
            info_message += f"; same-bone aliases {auto_alias_count}, merged {merged_aliases}"
        self.report({"INFO"}, info_message)
        if result.warnings:
            self.report({"WARNING"}, " | ".join(result.warnings[:3]))
        if remap_result is not None and remap_result.skipped_objects:
            self.report({"WARNING"}, " | ".join(remap_result.skipped_objects[:3]))
        if auto_merge_warning:
            self.report({"WARNING"}, auto_merge_warning)
        if auto_merge_result is not None and auto_merge_result.skipped_objects:
            self.report({"WARNING"}, " | ".join(auto_merge_result.skipped_objects[:3]))
        return {"FINISHED"}


class BMC_OT_analyze_duplicate_bones(bpy.types.Operator):
    bl_idname = "object.bmc_analyze_duplicate_bones"
    bl_label = "Analyze Same-Bone Groups"
    bl_description = "Build seam-filtered same-bone recommendations from the current manifest and target meshes"
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

        self.report({"INFO"}, f"Found {alias_count} same-bone alias candidate(s)")
        return {"FINISHED"}


class BMC_OT_apply_vertex_group_remap(bpy.types.Operator):
    bl_idname = "object.bmc_apply_vertex_group_remap"
    bl_label = "Apply Vertex Group Remap"
    bl_description = "Rename listed target mesh vertex groups from local indices to the generated global bone indices"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and scene.bmc_manifest_path and _enabled_target_names(scene))

    def execute(self, context):
        try:
            result = apply_vertex_group_remap_for_target_names(
                context=context,
                manifest_path=context.scene.bmc_manifest_path,
                target_object_names=_enabled_target_names(context.scene),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Remap failed: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Remapped {result.updated_objects}/{result.target_objects} listed meshes; renamed {result.renamed_groups} groups",
        )
        if result.skipped_objects:
            self.report({"WARNING"}, " | ".join(result.skipped_objects[:3]))
        return {"FINISHED"}


class BMC_OT_merge_duplicate_bones(bpy.types.Operator):
    bl_idname = "object.bmc_merge_duplicate_bones"
    bl_label = "Merge Duplicate Bones"
    bl_description = "Merge duplicate seam/alias bone weights for the listed target meshes using the configured alias list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and _enabled_target_names(scene))

    def execute(self, context):
        try:
            result = merge_duplicate_bone_weights_for_target_names(
                context=context,
                target_object_names=_enabled_target_names(context.scene),
                alias_entries=_enabled_alias_payload(context.scene),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Duplicate merge failed: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Merged {result.merged_aliases} aliases across {result.updated_objects}/{result.target_objects} listed meshes",
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
    bl_label = "Save Preset"
    bl_description = "Save the current target list, duplicate-bone settings, and alias list as a preset"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        try:
            preset_path = save_preset(scene.bmc_preset_name or scene.bmc_preset_choice, _serialize_scene_preset(scene))
        except Exception as exc:
            self.report({"ERROR"}, f"Save preset failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Preset saved: {preset_path}")
        return {"FINISHED"}


class BMC_OT_load_preset(bpy.types.Operator):
    bl_idname = "object.bmc_load_preset"
    bl_label = "Load Preset"
    bl_description = "Load the selected preset into the current scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        preset_name = scene.bmc_preset_choice
        if not preset_name or preset_name == "__NONE__":
            self.report({"ERROR"}, "No preset selected")
            return {"CANCELLED"}
        try:
            payload = load_preset(preset_name)
            _apply_loaded_preset(scene, payload)
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
