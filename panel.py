"""Sidebar panel for the Bone Merge Capture plugin."""

import hashlib
from pathlib import Path

import bpy

from .core.export_names import ini_filename_from_collection_name
from .core.texture_converter import TextureConversionError, convert_dds_to_png_preview, load_image_for_blender

_PREVIEW_COLLECTION = None


def unregister_preview_cache():
    global _PREVIEW_COLLECTION  # pylint: disable=global-statement
    if _PREVIEW_COLLECTION is None:
        return
    import bpy.utils.previews

    bpy.utils.previews.remove(_PREVIEW_COLLECTION)
    _PREVIEW_COLLECTION = None


def _preview_collection():
    global _PREVIEW_COLLECTION  # pylint: disable=global-statement
    if _PREVIEW_COLLECTION is None:
        import bpy.utils.previews

        _PREVIEW_COLLECTION = bpy.utils.previews.new()
    return _PREVIEW_COLLECTION


def _image_preview_icon(source_path: str) -> int | None:
    path = Path(str(source_path or ""))
    if not path.is_file():
        return None
    preview_path = path
    if path.suffix.lower() == ".dds":
        try:
            preview_path = convert_dds_to_png_preview(path)
        except (FileNotFoundError, TextureConversionError):
            preview_path = path

    try:
        stat = preview_path.stat()
        key_payload = f"{preview_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8", errors="ignore")
        preview_key = hashlib.sha1(key_payload).hexdigest()
        previews = _preview_collection()
        if preview_key in previews:
            return int(previews[preview_key].icon_id)
        thumbnail = previews.load(preview_key, str(preview_path), "IMAGE")
        return int(thumbnail.icon_id)
    except Exception:
        pass

    try:
        image = load_image_for_blender(path)
        preview = image.preview_ensure()
        return int(preview.icon_id)
    except Exception:
        return None


def _semantic_label(semantic: str, semantic_index: int = 0) -> str:
    labels = {
        "base_color": "Base",
        "normal": "Normal",
        "material": "Material",
        "effect": "Effect",
    }
    label = labels.get(str(semantic or ""), "Unmarked")
    if semantic in {"material", "effect"}:
        label += f" {int(semantic_index)}"
    return label


def _export_ini_filename_label(collection) -> str:
    collection_name = str(getattr(collection, "name", "") or "")
    return ini_filename_from_collection_name(collection_name)


def _has_scene_props(scene, *names: str) -> bool:
    return all(hasattr(scene, name) for name in names)


class BMC_UL_candidate_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=item.display_name or item.ib_hash or "(candidate)")
            row.label(text=f"idx {int(item.match_index_count)}")
            row.label(text=f"bones {int(item.local_bone_count)}")
            if not item.replacement_supported:
                row.label(text="CPU / no replace", icon="ERROR")
            if item.lod_match_excluded:
                row.label(text="noLOD")
            else:
                row.label(text="capture" if item.shadow_capture_ready else "map")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.ib_hash or "?")


class BMC_UL_lod_profiles(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=item.label or f"LOD {int(item.lod_level)}")
            row.label(text=item.status or "not_analyzed")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=f"L{int(item.lod_level)}")


class BMC_UL_lod_mapping_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            if item.lod_record_key:
                row.label(text=f"L{int(item.lod_level)} G{item.canonical_global_bone} -> {item.lod_record_key}:{item.lod_local_bone}")
            else:
                row.label(text=f"L{int(item.lod_level)} G{item.canonical_global_bone} -> unmatched")
            row.label(text=item.status or "unmatched")
            row.label(text=f"v{int(item.votes)}")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=str(item.canonical_global_bone))


class BMC_UL_lod_fallback_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            if item.status == "unresolved":
                row.label(text="", icon="ERROR")
            else:
                row.prop(item, "enabled", text="")
            row.label(text=f"L{int(item.lod_level)} G{item.canonical_global_bone} <- G{item.donor_global_bone}")
            row.label(text=item.method or item.status or "fallback")
            row.label(text=f"{float(item.confidence):.2f}")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=str(item.canonical_global_bone))


class BMC_UL_texture_mark_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            icon_id = _image_preview_icon(item.source_path)
            if icon_id is not None:
                row.label(text="", icon_value=icon_id)
            else:
                row.label(text="", icon="TEXTURE")
            row.label(text=f"{item.slot}  {item.hash_value[:8]}  {_semantic_label(item.semantic, item.semantic_index)}")
            row.label(text=item.filename or "(missing)")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.hash_value[:8] or "?")


class BMC_UL_toggle_draw_sets(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            label = item.label or item.toggle_id or "(toggle)"
            row.label(text=label, icon="DRIVER")
            row.label(text=item.key or "no key")
            row.label(text=f"{len(item.values)} value(s)")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.label[:3] or "?")


class VIEW3D_PT_bone_merge_capture(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bone Merge Capture"
    bl_label = "Bone Merge Capture"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if not _has_scene_props(
            scene,
            "bmc_frameanalysis_dir",
            "bmc_lod_frameanalysis_dir",
            "bmc_mirror_flip",
            "bmc_uv_flip_v",
            "bmc_candidate_items",
            "bmc_candidate_index",
            "bmc_lod_profiles",
            "bmc_lod_profile_index",
            "bmc_export_collection",
            "bmc_export_mode",
            "bmc_output_dir",
        ):
            layout.label(text="Bone Merge properties are not registered. Reload the addon.", icon="ERROR")
            return

        scan_box = layout.box()
        scan_box.label(text="Main Analyze", icon="VIEWZOOM")
        scan_box.prop(scene, "bmc_frameanalysis_dir")
        scan_box.prop(scene, "bmc_mirror_flip")
        scan_box.prop(scene, "bmc_uv_flip_v")
        scan_actions = scan_box.row(align=True)
        scan_actions.operator("object.bmc_analyze_main_frameanalysis", icon="VIEWZOOM")
        scan_actions.operator("object.bmc_import_selected_candidates", icon="IMPORT")
        pool_actions = scan_box.row(align=True)
        pool_actions.operator("object.bmc_build_global_bone_pool", icon="MOD_ARMATURE", text="Build Pool")
        pool_actions.operator("object.bmc_apply_global_bone_pool", icon="GROUP_VERTEX", text="Apply Pool")
        scan_box.label(text="LOD Profiles", icon="MESH_GRID")
        lod_profiles = scan_box.row()
        lod_profiles.template_list(
            "BMC_UL_lod_profiles",
            "",
            scene,
            "bmc_lod_profiles",
            scene,
            "bmc_lod_profile_index",
            rows=3,
        )
        lod_profile_buttons = lod_profiles.column(align=True)
        lod_profile_buttons.operator("object.bmc_lod_profile_add", icon="ADD", text="")
        lod_profile_buttons.operator("object.bmc_lod_profile_remove", icon="REMOVE", text="")
        lod_profile_buttons.operator("object.bmc_sync_lod_profiles", icon="FILE_REFRESH", text="")
        if scene.bmc_lod_profiles:
            profile_index = min(int(scene.bmc_lod_profile_index), len(scene.bmc_lod_profiles) - 1)
            lod_profile = scene.bmc_lod_profiles[profile_index]
            lod_detail = scan_box.column(align=True)
            lod_detail.prop(lod_profile, "label")
            lod_detail.prop(lod_profile, "lod_level")
            lod_detail.prop(lod_profile, "frameanalysis_dir")
            lod_actions = scan_box.row(align=True)
            lod_actions.operator("object.bmc_analyze_lod_frameanalysis", icon="MESH_GRID", text="Analyze Active")
            if lod_profile.summary:
                lod_detail.label(text=lod_profile.summary, icon="INFO")
            if lod_profile.warning:
                lod_detail.label(text=lod_profile.warning, icon="ERROR")
        if getattr(scene, "bmc_lod_mapping_items", None):
            row = scan_box.row()
            row.template_list(
                "BMC_UL_lod_mapping_items",
                "",
                scene,
                "bmc_lod_mapping_items",
                scene,
                "bmc_lod_mapping_index",
                rows=4,
            )
        add_row = scan_box.row(align=True)
        add_row.prop(scene, "bmc_candidate_add_hash", text="")
        row = scan_box.row()
        row.template_list(
            "BMC_UL_candidate_items",
            "",
            scene,
            "bmc_candidate_items",
            scene,
            "bmc_candidate_index",
            rows=6,
        )
        col = row.column(align=True)
        col.operator("object.bmc_candidate_add_hash", icon="ADD", text="")
        col.operator("object.bmc_candidate_remove", icon="REMOVE", text="")
        col.operator("object.bmc_candidate_refresh_from_collection", icon="FILE_REFRESH", text="")
        if scene.bmc_candidate_items and 0 <= scene.bmc_candidate_index < len(scene.bmc_candidate_items):
            candidate = scene.bmc_candidate_items[scene.bmc_candidate_index]
            scan_box.prop(candidate, "ib_hash")
            scan_box.prop(candidate, "match_first_index")
            scan_box.prop(candidate, "match_index_count")
            scan_box.prop(candidate, "import_draw_index")
            scan_box.prop(candidate, "local_bone_count")
            scan_box.prop(candidate, "shadow_capture_ready")
            scan_box.prop(candidate, "lod_match_excluded")
            scan_box.prop(candidate, "replacement_supported")
            scan_box.prop(candidate, "skinning_mode")
            scan_box.prop(candidate, "status")

        export_box = layout.box()
        export_box.label(text="3Dmigoto Export", icon="EXPORT")
        export_box.prop(scene, "bmc_output_dir", text="Output Dir")
        export_box.label(text="Big Export Collection", icon="OUTLINER_COLLECTION")
        export_box.prop(scene, "bmc_export_collection", text="Collection")
        export_box.label(text=f"INI: {_export_ini_filename_label(scene.bmc_export_collection)}", icon="TEXT")
        export_box.operator("object.bmc_add_selected_export_objects", icon="ADD", text="Add Selected")
        export_box.prop(scene, "bmc_export_mode", text="")
        export_box.operator("object.bmc_prepare_export_collection", icon="EXPORT", text="Export")


class VIEW3D_PT_bone_merge_toggle_draw_sets(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bone Merge Capture"
    bl_label = "Toggle Draw Sets"
    bl_parent_id = "VIEW3D_PT_bone_merge_capture"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if not _has_scene_props(scene, "bmc_toggle_draw_sets", "bmc_toggle_draw_set_index"):
            layout.label(text="Toggle properties are not registered.", icon="ERROR")
            return

        row = layout.row()
        row.template_list(
            "BMC_UL_toggle_draw_sets",
            "",
            scene,
            "bmc_toggle_draw_sets",
            scene,
            "bmc_toggle_draw_set_index",
            rows=3,
        )
        buttons = row.column(align=True)
        buttons.operator("object.bmc_toggle_draw_set_add", icon="ADD", text="")
        buttons.operator("object.bmc_toggle_draw_set_remove", icon="REMOVE", text="")

        if not scene.bmc_toggle_draw_sets:
            layout.label(text="Objects not assigned here always draw.", icon="INFO")
            return
        group_index = min(int(scene.bmc_toggle_draw_set_index), len(scene.bmc_toggle_draw_sets) - 1)
        toggle = scene.bmc_toggle_draw_sets[group_index]

        detail = layout.box()
        top = detail.row(align=True)
        top.prop(toggle, "enabled", text="")
        top.prop(toggle, "label", text="Label")
        detail.prop(toggle, "toggle_id", text="ID")
        key_row = detail.row(align=True)
        key_row.prop(toggle, "key", text="Key")
        op = key_row.operator("object.bmc_toggle_draw_set_record_key", icon="REC", text="Record")
        op.group_index = group_index
        detail.prop(toggle, "default_value", text="Default Value")

        value_header = detail.row(align=True)
        value_header.label(text="Values", icon="DOT")
        value_header.operator("object.bmc_toggle_draw_value_add", icon="ADD", text="")
        for value_index, value_item in enumerate(toggle.values):
            value_box = detail.box()
            value_row = value_box.row(align=True)
            value_row.prop(value_item, "value", text="Value")
            value_row.prop(value_item, "label", text="Label")
            remove_op = value_row.operator("object.bmc_toggle_draw_value_remove", icon="REMOVE", text="")
            remove_op.group_index = group_index
            remove_op.value_index = value_index

            objects = [item.object_name for item in value_item.objects if item.object_name]
            value_box.label(text=f"Draws {len(objects)} object(s)")
            for object_name in objects[:6]:
                value_box.label(text=object_name, icon="MESH_DATA")
            if len(objects) > 6:
                value_box.label(text=f"... {len(objects) - 6} more")

            actions = value_box.row(align=True)
            add_op = actions.operator("object.bmc_toggle_draw_value_add_selected", icon="ADD", text="Add Selected")
            add_op.group_index = group_index
            add_op.value_index = value_index
            remove_selected_op = actions.operator("object.bmc_toggle_draw_value_remove_selected", icon="REMOVE", text="Remove Selected")
            remove_selected_op.group_index = group_index
            remove_selected_op.value_index = value_index
            select_op = actions.operator("object.bmc_toggle_draw_value_select_objects", icon="RESTRICT_SELECT_OFF", text="Select Objects")
            select_op.group_index = group_index
            select_op.value_index = value_index


class VIEW3D_PT_bone_merge_texture_tools(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bone Merge Capture"
    bl_label = "Texture Tools"
    bl_parent_id = "VIEW3D_PT_bone_merge_capture"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if not _has_scene_props(scene, "bmc_texture_region", "bmc_texture_draw", "bmc_texture_mark_items"):
            layout.label(text="Texture properties are not registered.", icon="ERROR")
            return
        layout.prop(scene, "bmc_texture_region", text="Region")
        layout.prop(scene, "bmc_texture_draw", text="Draw")
        row = layout.row()
        row.template_list(
            "BMC_UL_texture_mark_items",
            "",
            scene,
            "bmc_texture_mark_items",
            scene,
            "bmc_texture_mark_index",
            rows=5,
        )
        if scene.bmc_texture_mark_items and 0 <= scene.bmc_texture_mark_index < len(scene.bmc_texture_mark_items):
            item = scene.bmc_texture_mark_items[scene.bmc_texture_mark_index]
            detail = layout.box()
            row = detail.row(align=True)
            icon_id = _image_preview_icon(item.source_path)
            if icon_id is not None:
                row.template_icon(icon_value=icon_id, scale=6.0)
            else:
                row.label(text="", icon="TEXTURE")
            info = row.column(align=True)
            info.label(text=f"{item.slot}  {item.hash_value[:8]}  {_semantic_label(item.semantic, item.semantic_index)}")
            info.label(text=item.filename or "(missing)")
            buttons = detail.row(align=True)
            for semantic, label in (
                ("base_color", "Base"),
                ("normal", "Normal"),
                ("material", "Material"),
                ("effect", "Effect"),
            ):
                op = buttons.operator(
                    "object.bmc_mark_texture_semantic",
                    text=label,
                    depress=item.semantic == semantic,
                )
                op.slot = item.slot
                op.semantic = semantic
            op = buttons.operator("object.bmc_mark_texture_semantic", text="Clear")
            op.slot = item.slot
            op.semantic = "clear"
        layout.operator("object.bmc_apply_texture_marks_to_models", icon="MATERIAL", text="Apply To Models")


class VIEW3D_PT_bone_merge_hash_tools(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bone Merge Capture"
    bl_label = "Vertex Group Tools"
    bl_parent_id = "VIEW3D_PT_bone_merge_capture"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, _context):
        layout = self.layout
        layout.operator("object.bmc_apply_global_names_by_object_hash", icon="GROUP_VERTEX", text="Rename To Global")
        layout.operator("object.bmc_revert_global_names_by_object_hash", icon="FILE_REFRESH", text="Rename To Local")
        layout.operator("object.bmc_merge_selected_seam_groups", icon="AUTOMERGE_ON", text="Merge Seam Groups")


class VIEW3D_PT_bone_merge_lod_repair(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bone Merge Capture"
    bl_label = "LOD Repair"
    bl_parent_id = "VIEW3D_PT_bone_merge_capture"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        row = layout.row(align=True)
        row.operator("object.bmc_preview_lod_fallbacks", icon="VIEWZOOM", text="Preview Fallbacks")
        row.operator("object.bmc_apply_lod_fallbacks", icon="CHECKMARK", text="Apply Fallbacks")
        if scene.bmc_lod_fallback_summary:
            icon = "CHECKMARK" if not scene.bmc_lod_fallback_warning else "ERROR"
            layout.label(text=scene.bmc_lod_fallback_summary, icon=icon)
        if scene.bmc_lod_fallback_warning:
            layout.label(text=scene.bmc_lod_fallback_warning, icon="ERROR")
        if scene.bmc_lod_fallback_items:
            layout.template_list(
                "BMC_UL_lod_fallback_items",
                "",
                scene,
                "bmc_lod_fallback_items",
                scene,
                "bmc_lod_fallback_index",
                rows=5,
            )
