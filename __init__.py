"""Bone Merge Capture plugin entrypoint."""

bl_info = {
    "name": "Bone Merge Capture",
    "author": "OpenAI Codex",
    "version": (0, 3, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Bone Merge Capture",
    "description": "Analyze FrameAnalysis captures, import candidate IBs, and build global bone pools for Bone Merge workflows.",
    "category": "Animation",
}

try:
    import bpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - unavailable outside Blender
    bpy = None


REGISTERED_CLASSES = ()
_PROPERTIES_MODULE = None


def register():
    if bpy is None:
        raise RuntimeError("Bone Merge Capture can only be registered inside Blender")

    from . import operators, panel, properties

    global REGISTERED_CLASSES, _PROPERTIES_MODULE
    REGISTERED_CLASSES = (
        properties.BMC_TargetItem,
        properties.BMC_CandidateItem,
        properties.BMC_AliasItem,
        properties.BMC_LodTargetItem,
        properties.BMC_LodMappingItem,
        properties.BMC_SeamMatchItem,
        properties.BMC_SeamAliasItem,
        panel.BMC_UL_target_items,
        panel.BMC_UL_candidate_items,
        panel.BMC_UL_alias_items,
        panel.BMC_UL_lod_target_items,
        panel.BMC_UL_lod_mapping_items,
        panel.BMC_UL_seam_match_items,
        panel.BMC_UL_seam_alias_items,
        operators.BMC_OT_add_selected_targets,
        operators.BMC_OT_add_selected_lod_targets,
        operators.BMC_OT_create_target_collection,
        operators.BMC_OT_sync_targets_from_collection,
        operators.BMC_OT_refresh_target_identity,
        operators.BMC_OT_create_export_collection,
        operators.BMC_OT_add_selected_export_objects,
        operators.BMC_OT_apply_export_collection_global_names,
        operators.BMC_OT_apply_global_names_by_object_hash,
        operators.BMC_OT_revert_global_names_by_object_hash,
        operators.BMC_OT_merge_selected_seam_groups,
        operators.BMC_OT_prepare_export_collection,
        operators.BMC_OT_generate_shadow_split,
        operators.BMC_OT_remove_target,
        operators.BMC_OT_clear_targets,
        operators.BMC_OT_remove_lod_target,
        operators.BMC_OT_clear_lod_targets,
        operators.BMC_OT_analyze_main_frameanalysis,
        operators.BMC_OT_candidate_add_hash,
        operators.BMC_OT_candidate_remove,
        operators.BMC_OT_candidate_refresh_from_collection,
        operators.BMC_OT_import_selected_candidates,
        operators.BMC_OT_build_global_bone_pool,
        operators.BMC_OT_scan_targets,
        operators.BMC_OT_scan_lod_targets,
        operators.BMC_OT_analyze_duplicate_bones,
        operators.BMC_OT_apply_vertex_group_remap,
        operators.BMC_OT_apply_lod_vertex_group_remap,
        operators.BMC_OT_merge_duplicate_bones,
        operators.BMC_OT_build_lod_mapping,
        operators.BMC_OT_generate_lod_runtime_map,
        operators.BMC_OT_alias_add,
        operators.BMC_OT_alias_remove,
        operators.BMC_OT_seam_add_selected_objects,
        operators.BMC_OT_seam_remove_object,
        operators.BMC_OT_seam_clear_objects,
        operators.BMC_OT_seam_build_mapping,
        operators.BMC_OT_seam_apply_mapping,
        operators.BMC_OT_save_preset,
        operators.BMC_OT_load_preset,
        operators.BMC_OT_delete_preset,
        panel.VIEW3D_PT_bone_merge_capture,
        panel.VIEW3D_PT_bone_merge_hash_tools,
    )
    _PROPERTIES_MODULE = properties

    for blender_class in REGISTERED_CLASSES:
        bpy.utils.register_class(blender_class)
    properties.register_addon_properties()
    operators.register_runtime_handlers()


def unregister():
    if bpy is None:
        return

    from . import operators

    operators.unregister_runtime_handlers()
    if _PROPERTIES_MODULE is not None:
        _PROPERTIES_MODULE.unregister_addon_properties()
    for blender_class in reversed(REGISTERED_CLASSES):
        bpy.utils.unregister_class(blender_class)


if __name__ == "__main__":
    register()
