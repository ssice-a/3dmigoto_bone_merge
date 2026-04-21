"""Sidebar panel for the Bone Merge Capture plugin."""

import bpy


class BMC_UL_target_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=item.object_name or "(unnamed)")
            row.label(text=f"{item.ib_hash}-{item.match_index_count}")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.object_name or "?")


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
            target_box.prop(target, "object_name")
            target_box.prop(target, "ib_hash")
            target_box.prop(target, "match_index_count")
            target_box.prop(target, "autodetected")

        scan_box = layout.box()
        scan_box.label(text="FrameAnalysis Scan", icon="FILE_FOLDER")
        scan_box.prop(scene, "bmc_frameanalysis_dir")
        scan_box.prop(scene, "bmc_output_dir")
        scan_box.prop(scene, "bmc_merge_same_bone_groups")
        scan_box.operator("object.bmc_scan_targets", icon="VIEWZOOM")

        preset_box = layout.box()
        preset_box.label(text="Presets", icon="BOOKMARKS")
        preset_box.prop(scene, "bmc_preset_choice")
        preset_box.prop(scene, "bmc_preset_name")
        row = preset_box.row(align=True)
        row.operator("object.bmc_save_preset", icon="FILE_TICK")
        row.operator("object.bmc_load_preset", icon="IMPORT")
        row.operator("object.bmc_delete_preset", icon="TRASH")

        output_box = layout.box()
        output_box.label(text="Generated Files", icon="TEXT")
        output_box.prop(scene, "bmc_manifest_path")
        output_box.prop(scene, "bmc_ini_path")

        export_box = layout.box()
        export_box.label(text="Export Preparation", icon="EXPORT")
        export_box.prop(scene, "bmc_export_collection")
        export_collection = scene.bmc_export_collection
        export_box.label(
            text=f"Export Collection: {export_collection.name if export_collection else '(not set)'}",
            icon="OUTLINER_COLLECTION",
        )
        row = export_box.row(align=True)
        row.operator("object.bmc_create_export_collection", icon="COLLECTION_NEW")
        row.operator("object.bmc_add_selected_export_objects", icon="ADD")
        export_box.operator("object.bmc_prepare_export_collection", icon="EXPORT")
        export_box.label(text="Child collections are final chunks named like fe47dc61-7014-0; Prepare localizes groups in place.", icon="INFO")
        export_box.prop(scene, "bmc_export_manifest_path")

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
