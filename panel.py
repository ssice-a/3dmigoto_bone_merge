"""Sidebar panel for the Bone Merge Capture plugin."""

import bpy


class BMC_UL_target_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        object_label = item.object_ref.name if getattr(item, "object_ref", None) else item.object_name
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=object_label or "(unnamed)")
            row.label(text=item.ib_hash)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=object_label or "?")


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


class BMC_UL_lod_target_items(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        object_label = item.object_ref.name if getattr(item, "object_ref", None) else item.object_name
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=object_label or "(unnamed)")
            row.label(text=item.ib_hash)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=object_label or "?")


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


class VIEW3D_PT_bone_merge_capture(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bone Merge Capture"
    bl_label = "Bone Merge Capture"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        scan_box = layout.box()
        scan_box.label(text="Main Analyze", icon="VIEWZOOM")
        scan_box.prop(scene, "bmc_frameanalysis_dir")
        scan_box.prop(scene, "bmc_output_dir")
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
        if scene.bmc_lod_match_summary:
            icon = "CHECKMARK" if not scene.bmc_lod_match_warning else "ERROR"
            scan_box.label(text=scene.bmc_lod_match_summary, icon=icon)
        if scene.bmc_lod_match_warning:
            scan_box.label(text=scene.bmc_lod_match_warning, icon="ERROR")
        if scene.bmc_lod_mapping_items:
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
        export_box.prop(scene, "bmc_output_dir", text="3Dmigoto Output Dir")
        export_box.prop(scene, "bmc_export_collection", text="Export Source")
        export_box.prop(scene, "bmc_export_build_collection", text="Export Build")
        row = export_box.row(align=True)
        row.operator("object.bmc_create_export_collection", icon="COLLECTION_NEW")
        row.operator("object.bmc_add_selected_export_objects", icon="ADD")
        export_box.operator("object.bmc_apply_export_collection_global_names", icon="GROUP_VERTEX")
        export_box.operator("object.bmc_prepare_export_collection", icon="EXPORT")

        shadow_box = layout.box()
        shadow_box.label(text="Modify Main INI", icon="SHADING_RENDERED")
        shadow_box.prop(scene, "bmc_source_ini_path")
        shadow_box.prop(scene, "bmc_shadow_host_hash")
        shadow_box.prop(scene, "bmc_shadow_host_match_index_count")
        shadow_box.prop(scene, "bmc_shadow_host_vs_hash")
        shadow_box.operator("object.bmc_generate_shadow_split", icon="SHADING_RENDERED")


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
