"""Sidebar panel for the Bone Merge Capture plugin."""

import bpy


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
            if item.lod_match_excluded:
                row.label(text="noLOD")
            else:
                row.label(text="capture" if item.shadow_capture_ready else "map")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.ib_hash or "?")


class BMC_UL_lod_mapping_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            if item.lod_record_key:
                row.label(text=f"G{item.canonical_global_bone} -> {item.lod_record_key}:{item.lod_local_bone}")
            else:
                row.label(text=f"G{item.canonical_global_bone} -> unmatched")
            row.label(text=item.status or "unmatched")
            row.label(text=f"v{int(item.votes)}")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=str(item.canonical_global_bone))


class BMC_UL_lod_fallback_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=f"G{item.canonical_global_bone} <- G{item.donor_global_bone}")
            row.label(text=item.method or item.status or "fallback")
            row.label(text=f"{float(item.confidence):.2f}")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=str(item.canonical_global_bone))


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
            "bmc_candidate_items",
            "bmc_candidate_index",
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
        scan_actions = scan_box.row(align=True)
        scan_actions.operator("object.bmc_analyze_main_frameanalysis", icon="VIEWZOOM")
        scan_actions.operator("object.bmc_import_selected_candidates", icon="IMPORT")
        pool_actions = scan_box.row(align=True)
        pool_actions.operator("object.bmc_build_global_bone_pool", icon="MOD_ARMATURE", text="Build Pool")
        pool_actions.operator("object.bmc_apply_global_bone_pool", icon="GROUP_VERTEX", text="Apply Pool")
        lod_row = scan_box.row(align=True)
        lod_row.prop(scene, "bmc_lod_frameanalysis_dir", text="LOD")
        lod_row.operator("object.bmc_analyze_lod_frameanalysis", icon="MESH_GRID", text="Analyze LOD")
        lod_match_summary = getattr(scene, "bmc_lod_match_summary", "")
        lod_match_warning = getattr(scene, "bmc_lod_match_warning", "")
        if lod_match_summary:
            icon = "CHECKMARK" if not lod_match_warning else "ERROR"
            scan_box.label(text=lod_match_summary, icon=icon)
        if lod_match_warning:
            scan_box.label(text=lod_match_warning, icon="ERROR")
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
            scan_box.prop(candidate, "status")

        export_box = layout.box()
        export_box.label(text="3Dmigoto Export", icon="EXPORT")
        export_box.prop(scene, "bmc_output_dir", text="Output Dir")
        export_box.prop(scene, "bmc_export_collection", text="Export Root")
        export_box.operator("object.bmc_add_selected_export_objects", icon="ADD", text="Add Selected")
        export_box.prop(scene, "bmc_export_mode", text="")
        export_box.operator("object.bmc_prepare_export_collection", icon="EXPORT", text="Export")


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
