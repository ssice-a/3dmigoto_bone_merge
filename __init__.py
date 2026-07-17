"""Bone Merge Capture plugin entrypoint."""

bl_info = {
    "name": "Bone Merge Capture",
    "author": "OpenAI Codex",
    "version": (0, 3, 2),
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
PROPERTY_GROUP_CLASSES = ()
RUNTIME_CLASSES = ()
_PROPERTIES_MODULE = None


def register():
    if bpy is None:
        raise RuntimeError("Bone Merge Capture can only be registered inside Blender")

    from . import operators, panel, properties

    global REGISTERED_CLASSES, PROPERTY_GROUP_CLASSES, RUNTIME_CLASSES, _PROPERTIES_MODULE
    PROPERTY_GROUP_CLASSES = (
        properties.BMC_CandidateItem,
        properties.BMC_LodMappingItem,
        properties.BMC_LodFallbackItem,
        properties.BMC_TextureMarkItem,
        properties.BMC_ToggleDrawObjectItem,
        properties.BMC_ToggleDrawValueItem,
        properties.BMC_ToggleDrawSetItem,
    )
    RUNTIME_CLASSES = (
        panel.BMC_UL_candidate_items,
        panel.BMC_UL_lod_mapping_items,
        panel.BMC_UL_lod_fallback_items,
        panel.BMC_UL_texture_mark_items,
        panel.BMC_UL_toggle_draw_sets,
        operators.BMC_OT_create_export_collection,
        operators.BMC_OT_add_selected_export_objects,
        operators.BMC_OT_toggle_draw_set_add,
        operators.BMC_OT_toggle_draw_set_remove,
        operators.BMC_OT_toggle_draw_value_add,
        operators.BMC_OT_toggle_draw_value_remove,
        operators.BMC_OT_toggle_draw_value_add_selected,
        operators.BMC_OT_toggle_draw_value_remove_selected,
        operators.BMC_OT_toggle_draw_value_select_objects,
        operators.BMC_OT_toggle_draw_set_record_key,
        operators.BMC_OT_apply_export_collection_global_names,
        operators.BMC_OT_apply_global_names_by_object_hash,
        operators.BMC_OT_revert_global_names_by_object_hash,
        operators.BMC_OT_merge_selected_seam_groups,
        operators.BMC_OT_mark_texture_semantic,
        operators.BMC_OT_apply_texture_marks_to_models,
        operators.BMC_OT_prepare_export_collection,
        operators.BMC_OT_analyze_main_frameanalysis,
        operators.BMC_OT_candidate_add_hash,
        operators.BMC_OT_candidate_remove,
        operators.BMC_OT_candidate_refresh_from_collection,
        operators.BMC_OT_import_selected_candidates,
        operators.BMC_OT_build_global_bone_pool,
        operators.BMC_OT_apply_global_bone_pool,
        operators.BMC_OT_analyze_lod_frameanalysis,
        operators.BMC_OT_preview_lod_fallbacks,
        operators.BMC_OT_apply_lod_fallbacks,
        panel.VIEW3D_PT_bone_merge_capture,
        panel.VIEW3D_PT_bone_merge_toggle_draw_sets,
        panel.VIEW3D_PT_bone_merge_texture_tools,
        panel.VIEW3D_PT_bone_merge_hash_tools,
        panel.VIEW3D_PT_bone_merge_lod_repair,
    )
    REGISTERED_CLASSES = PROPERTY_GROUP_CLASSES + RUNTIME_CLASSES
    _PROPERTIES_MODULE = properties

    for blender_class in PROPERTY_GROUP_CLASSES:
        bpy.utils.register_class(blender_class)
    properties.register_addon_properties()
    for blender_class in RUNTIME_CLASSES:
        bpy.utils.register_class(blender_class)


def unregister():
    if bpy is None:
        return

    if _PROPERTIES_MODULE is not None:
        _PROPERTIES_MODULE.unregister_addon_properties()
    try:
        from . import panel

        panel.unregister_preview_cache()
    except Exception:
        pass
    for blender_class in reversed(REGISTERED_CLASSES):
        bpy.utils.unregister_class(blender_class)


if __name__ == "__main__":
    register()
