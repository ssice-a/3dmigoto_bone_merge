"""Sidebar panel for the Bone Merge Capture plugin."""

import bpy


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


class BMC_UL_seam_match_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        object_label = item.object_ref.name if getattr(item, "object_ref", None) else item.object_name
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=object_label or "(missing)")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=object_label or "?")


class BMC_UL_seam_alias_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=f"{item.src_object_name}:{item.src_group} -> {item.dst_object_name}:{item.dst_group}")
            row.label(text=f"votes={item.votes}")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=str(item.src_group))


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
            target_box.prop(target, "ib_hash")
            target_box.prop(target, "match_index_count")
            target_box.prop(target, "local_bone_count")
            target_box.prop(target, "autodetected")
            target_box.operator("object.bmc_refresh_target_identity", icon="FILE_REFRESH")

        seam_box = layout.box()
        seam_box.label(text="Seam Group Matcher", icon="AUTOMERGE_ON")
        row = seam_box.row()
        row.template_list(
            "BMC_UL_seam_match_items",
            "",
            scene,
            "bmc_seam_match_items",
            scene,
            "bmc_seam_match_index",
            rows=5,
        )
        col = row.column(align=True)
        col.operator("object.bmc_seam_add_selected_objects", icon="ADD", text="")
        col.operator("object.bmc_seam_remove_object", icon="REMOVE", text="")
        col.operator("object.bmc_seam_clear_objects", icon="TRASH", text="")

        action_row = seam_box.row(align=True)
        action_row.operator("object.bmc_seam_build_mapping", icon="VIEWZOOM")
        action_row.operator("object.bmc_seam_apply_mapping", icon="FILE_TICK")
        if scene.bmc_seam_pair_summary:
            for summary_line in str(scene.bmc_seam_pair_summary).splitlines()[:6]:
                seam_box.label(text=summary_line, icon="LINKED")
        row = seam_box.row()
        row.template_list(
            "BMC_UL_seam_alias_items",
            "",
            scene,
            "bmc_seam_alias_items",
            scene,
            "bmc_seam_alias_index",
            rows=6,
        )

        scan_box = layout.box()
        scan_box.label(text="Scan / Freeze Mapping", icon="FILE_FOLDER")
        scan_box.prop(scene, "bmc_frameanalysis_dir")
        scan_box.prop(scene, "bmc_output_dir")
        scan_box.prop(scene, "bmc_scan_auto_apply_mapping")
        scan_box.operator("object.bmc_scan_targets", icon="VIEWZOOM")

        mapping_box = layout.box()
        mapping_box.label(text="Legacy Target Remap", icon="MODIFIER")
        mapping_box.operator("object.bmc_apply_vertex_group_remap", icon="DRIVER")

        preset_box = layout.box()
        preset_box.label(text="Mapping Preset", icon="BOOKMARKS")
        preset_box.prop(scene, "bmc_preset_choice")
        preset_box.prop(scene, "bmc_preset_name")
        row = preset_box.row(align=True)
        row.operator("object.bmc_save_preset", icon="FILE_TICK")
        row.operator("object.bmc_load_preset", icon="IMPORT")
        row.operator("object.bmc_delete_preset", icon="TRASH")

        export_box = layout.box()
        export_box.label(text="3Dmigoto Export", icon="EXPORT")
        export_box.prop(scene, "bmc_output_dir", text="3Dmigoto Output Dir")
        export_box.prop(scene, "bmc_export_collection", text="Export Source")
        export_box.prop(scene, "bmc_export_build_collection", text="Export Build")
        row = export_box.row(align=True)
        row.operator("object.bmc_create_export_collection", icon="COLLECTION_NEW")
        row.operator("object.bmc_add_selected_export_objects", icon="ADD")
        export_box.operator("object.bmc_prepare_export_collection", icon="EXPORT")

        shadow_box = layout.box()
        shadow_box.label(text="Modify Main INI", icon="SHADING_RENDERED")
        shadow_box.prop(scene, "bmc_source_ini_path")
        shadow_box.prop(scene, "bmc_shadow_host_hash")
        shadow_box.prop(scene, "bmc_shadow_host_match_index_count")
        shadow_box.prop(scene, "bmc_shadow_host_vs_hash")
        shadow_box.operator("object.bmc_generate_shadow_split", icon="SHADING_RENDERED")
