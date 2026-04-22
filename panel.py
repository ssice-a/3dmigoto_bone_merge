"""Sidebar panel for the Bone Merge Capture plugin."""

import os

import bpy


def _file_label(path: str, fallback: str = "(not set)") -> str:
    normalized = str(path or "")
    if not normalized:
        return fallback
    return os.path.basename(normalized) or normalized


def _path_label(path: str, fallback: str = "(not set)") -> str:
    return str(path or "") or fallback


class BMC_UL_target_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        object_label = item.object_ref.name if getattr(item, "object_ref", None) else item.object_name
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=object_label or "(unnamed)")
            row.label(text=f"{item.ib_hash}-{item.match_index_count}")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=object_label or "?")


class BMC_UL_alias_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=f"{item.src_object_name}:{item.src_local_bone} -> {item.canonical_object_name}:{item.canonical_local_bone}")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=str(item.src_local_bone))


class VIEW3D_PT_bone_merge_capture(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bone Merge Capture"
    bl_label = "Bone Merge Capture"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        target_box = layout.box()
        target_box.label(text="Target Objects", icon="OUTLINER_OB_MESH")
        target_box.prop(scene, "bmc_target_collection")
        collection = scene.bmc_target_collection
        target_box.label(
            text=f"Collection: {collection.name if collection else '(not set)'}",
            icon="OUTLINER_COLLECTION",
        )
        collection_row = target_box.row(align=True)
        collection_row.operator("object.bmc_create_target_collection", icon="COLLECTION_NEW")
        collection_row.operator("object.bmc_sync_targets_from_collection", icon="FILE_REFRESH")
        row = target_box.row()
        row.template_list(
            "BMC_UL_target_items",
            "",
            scene,
            "bmc_target_items",
            scene,
            "bmc_target_index",
            rows=6,
        )
        col = row.column(align=True)
        col.operator("object.bmc_add_selected_targets", icon="ADD", text="")
        col.operator("object.bmc_remove_target", icon="REMOVE", text="")
        col.operator("object.bmc_clear_targets", icon="TRASH", text="")

        if scene.bmc_target_items and 0 <= scene.bmc_target_index < len(scene.bmc_target_items):
            target = scene.bmc_target_items[scene.bmc_target_index]
            target_box.prop(target, "object_ref")
            target_box.label(
                text=f"Object: {(target.object_ref.name if target.object_ref else target.object_name) or '(missing)'}",
                icon="OBJECT_DATA",
            )
            target_box.prop(target, "ib_hash")
            target_box.prop(target, "match_index_count")
            target_box.prop(target, "local_bone_count")
            target_box.prop(target, "autodetected")
            target_box.operator("object.bmc_refresh_target_identity", icon="FILE_REFRESH")

        scan_box = layout.box()
        scan_box.label(text="FrameAnalysis Scan", icon="FILE_FOLDER")
        scan_box.prop(scene, "bmc_frameanalysis_dir")
        scan_box.prop(scene, "bmc_output_dir")
        scan_box.operator("object.bmc_scan_targets", icon="VIEWZOOM")
        scan_box.label(text="Scan now freezes capture data only; it no longer remaps or merges source meshes.", icon="INFO")

        preset_box = layout.box()
        preset_box.label(text="Presets", icon="BOOKMARKS")
        preset_box.prop(scene, "bmc_preset_choice")
        preset_box.prop(scene, "bmc_preset_name")
        row = preset_box.row(align=True)
        row.operator("object.bmc_save_preset", icon="FILE_TICK")
        row.operator("object.bmc_load_preset", icon="IMPORT")
        row.operator("object.bmc_delete_preset", icon="TRASH")

        output_box = layout.box()
        output_box.label(text="Current Run State", icon="TEXT")
        output_box.label(text=f"Capture Manifest: {_path_label(scene.bmc_manifest_path)}")
        output_box.label(text=f"BoneStore: {_path_label(scene.bmc_ini_path)}")
        output_box.label(text="These paths come from the current Scan / Prepare Export run.", icon="INFO")

        export_box = layout.box()
        export_box.label(text="3Dmigoto Export", icon="EXPORT")
        export_box.prop(scene, "bmc_output_dir", text="3Dmigoto Output Dir")
        export_box.prop(scene, "bmc_export_collection", text="Export Source")
        export_box.prop(scene, "bmc_export_build_collection", text="Export Build")
        export_collection = scene.bmc_export_collection
        export_build_collection = scene.bmc_export_build_collection
        export_box.label(
            text=f"Source: {export_collection.name if export_collection else '(not set)'}",
            icon="OUTLINER_COLLECTION",
        )
        export_box.label(
            text=f"Build: {export_build_collection.name if export_build_collection else '(not set)'}",
            icon="OUTLINER_COLLECTION",
        )
        row = export_box.row(align=True)
        row.operator("object.bmc_create_export_collection", icon="COLLECTION_NEW")
        row.operator("object.bmc_add_selected_export_objects", icon="ADD")
        export_box.operator("object.bmc_prepare_export_collection", icon="EXPORT")
        export_box.label(text="Edit meshes only in Export Source child collections.", icon="INFO")
        export_box.label(text="Prepare Export rebuilds Export Build from source, then localizes only the build copies.", icon="INFO")

        shadow_box = layout.box()
        shadow_box.label(text="Advanced Shadow Split", icon="SHADING_RENDERED")
        shadow_box.label(text="Optional: move vs==200 shadow payloads to the last shadow host.", icon="INFO")
        shadow_box.label(text="Uses the latest Prepare Export result automatically.", icon="INFO")
        shadow_box.prop(scene, "bmc_source_ini_path")
        shadow_box.prop(scene, "bmc_shadow_host_hash")
        shadow_box.prop(scene, "bmc_shadow_host_match_index_count")
        shadow_box.prop(scene, "bmc_shadow_host_vs_hash")
        shadow_box.operator("object.bmc_generate_shadow_split", icon="SHADING_RENDERED")

        alias_box = layout.box()
        alias_box.label(text="Merge Duplicate Bone", icon="AUTOMERGE_ON")
        alias_box.operator("object.bmc_analyze_duplicate_bones", icon="VIEWZOOM")
        alias_box.label(text="Run after scan; this is intentionally separate because it can be slow.", icon="INFO")
        row = alias_box.row()
        row.template_list(
            "BMC_UL_alias_items",
            "",
            scene,
            "bmc_alias_items",
            scene,
            "bmc_alias_index",
            rows=6,
        )
        col = row.column(align=True)
        col.operator("object.bmc_alias_add", icon="ADD", text="")
        col.operator("object.bmc_alias_remove", icon="REMOVE", text="")

        if scene.bmc_alias_items and 0 <= scene.bmc_alias_index < len(scene.bmc_alias_items):
            alias = scene.bmc_alias_items[scene.bmc_alias_index]
            alias_box.prop(alias, "enabled")
            alias_box.prop(alias, "src_object_name")
            alias_box.prop(alias, "src_ib_hash")
            alias_box.prop(alias, "src_local_bone")
            alias_box.prop(alias, "src_global_bone")
            alias_box.prop(alias, "canonical_object_name")
            alias_box.prop(alias, "canonical_ib_hash")
            alias_box.prop(alias, "canonical_local_bone")
            alias_box.prop(alias, "canonical_global_bone")
            alias_box.prop(alias, "confidence")

        apply_box = layout.box()
        apply_box.label(text="Apply To Listed Objects", icon="MODIFIER")
        apply_box.operator("object.bmc_apply_vertex_group_remap", icon="DRIVER")
        apply_box.operator("object.bmc_merge_duplicate_bones", icon="AUTOMERGE_ON")

        active_object = context.active_object
        if active_object and active_object.type == "MESH":
            debug_box = layout.box()
            debug_box.label(text=f"Active Mesh: {active_object.name}", icon="INFO")
            debug_box.prop(active_object, "merge_ib_hash")
            debug_box.prop(active_object, "merge_match_index_count")
            debug_box.prop(active_object, "merge_ib_autodetected")
