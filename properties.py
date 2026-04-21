"""Register Blender properties used by the Bone Merge Capture plugin."""

from __future__ import annotations

import bpy

from .core.presets import list_preset_names, load_preset, resolve_preset_workspace_paths


def _preset_enum_items(_self, _context):
    names = list_preset_names()
    if not names:
        return [("__NONE__", "(No Presets)", "No saved presets available")]
    return [(name, name, f"Load preset {name}") for name in names]


def _apply_preset_payload_to_scene(scene, preset_name: str, payload: dict) -> None:
    scene.bmc_preset_name = preset_name
    scene.bmc_merge_same_bone_groups = bool(payload.get("merge_same_bone_groups", False))

    workspace_paths = resolve_preset_workspace_paths(preset_name)
    if workspace_paths:
        scene.bmc_frameanalysis_dir = str(workspace_paths.get("frameanalysis_dir", "") or "")
        scene.bmc_output_dir = str(workspace_paths.get("output_dir", "") or "")
        scene.bmc_manifest_path = ""
        scene.bmc_ini_path = ""
        scene.bmc_export_manifest_path = ""
        scene.bmc_source_ini_path = str(workspace_paths.get("source_ini_path", "") or "")
        scene.bmc_shadow_host_hash = ""
        scene.bmc_shadow_host_match_index_count = -1
        scene.bmc_shadow_host_vs_hash = ""

    workspace = payload.get("workspace", {})
    if isinstance(workspace, dict):
        target_collection_name = str(workspace.get("target_collection_name", "") or "").strip()
        if target_collection_name:
            target_collection = bpy.data.collections.get(target_collection_name)
            if target_collection is not None:
                scene.bmc_target_collection = target_collection

        export_collection_name = str(workspace.get("export_collection_name", "") or "").strip()
        if export_collection_name:
            export_collection = bpy.data.collections.get(export_collection_name)
            if export_collection is not None:
                scene.bmc_export_collection = export_collection

    scene.bmc_target_items.clear()
    for target in payload.get("targets", []):
        item = scene.bmc_target_items.add()
        item.object_name = str(target.get("object_name", ""))
        item.ib_hash = str(target.get("ib_hash", ""))
        item.match_index_count = int(target.get("match_index_count", 0))
        item.autodetected = bool(target.get("autodetected", True))
        item.enabled = bool(target.get("enabled", True))
        mesh_obj = scene.objects.get(item.object_name)
        if mesh_obj is not None and mesh_obj.type == "MESH":
            item.object_ref = mesh_obj
            item.object_name = mesh_obj.name
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


def _preset_choice_update(self, _context):
    preset_name = str(getattr(self, "bmc_preset_choice", "") or "")
    if not preset_name or preset_name == "__NONE__":
        return
    try:
        payload = load_preset(preset_name)
        _apply_preset_payload_to_scene(self, preset_name, payload)
    except Exception:
        # Enum update callbacks cannot report cleanly; the explicit Load button still reports details.
        return


class BMC_TargetItem(bpy.types.PropertyGroup):
    object_name: bpy.props.StringProperty(name="Object", default="")
    object_ref: bpy.props.PointerProperty(name="Object Ref", type=bpy.types.Object)
    ib_hash: bpy.props.StringProperty(name="IB Hash", default="")
    match_index_count: bpy.props.IntProperty(name="Match Index Count", default=0, min=0)
    autodetected: bpy.props.BoolProperty(name="Auto", default=True)
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)


class BMC_AliasItem(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    src_draw_index: bpy.props.IntProperty(name="Src Draw", default=0, min=0)
    src_object_name: bpy.props.StringProperty(name="Src Object", default="")
    src_ib_hash: bpy.props.StringProperty(name="Src IB", default="")
    src_local_bone: bpy.props.IntProperty(name="Src Local Bone", default=0, min=0)
    src_global_bone: bpy.props.IntProperty(name="Src Global Bone", default=0, min=0)
    canonical_draw_index: bpy.props.IntProperty(name="Canonical Draw", default=0, min=0)
    canonical_object_name: bpy.props.StringProperty(name="Canonical Object", default="")
    canonical_ib_hash: bpy.props.StringProperty(name="Canonical IB", default="")
    canonical_local_bone: bpy.props.IntProperty(name="Canonical Local Bone", default=0, min=0)
    canonical_global_bone: bpy.props.IntProperty(name="Canonical Global Bone", default=0, min=0)
    confidence: bpy.props.StringProperty(name="Confidence", default="")


REGISTERED_PROPERTY_PATHS = (
    (bpy.types.Object, "merge_ib_hash"),
    (bpy.types.Object, "merge_match_index_count"),
    (bpy.types.Object, "merge_ib_autodetected"),
    (bpy.types.Scene, "bmc_frameanalysis_dir"),
    (bpy.types.Scene, "bmc_output_dir"),
    (bpy.types.Scene, "bmc_manifest_path"),
    (bpy.types.Scene, "bmc_ini_path"),
    (bpy.types.Scene, "bmc_target_collection"),
    (bpy.types.Scene, "bmc_export_collection"),
    (bpy.types.Scene, "bmc_export_manifest_path"),
    (bpy.types.Scene, "bmc_source_ini_path"),
    (bpy.types.Scene, "bmc_shadow_host_hash"),
    (bpy.types.Scene, "bmc_shadow_host_match_index_count"),
    (bpy.types.Scene, "bmc_shadow_host_vs_hash"),
    (bpy.types.Scene, "bmc_merge_same_bone_groups"),
    (bpy.types.Scene, "bmc_target_items"),
    (bpy.types.Scene, "bmc_target_index"),
    (bpy.types.Scene, "bmc_alias_items"),
    (bpy.types.Scene, "bmc_alias_index"),
    (bpy.types.Scene, "bmc_preset_name"),
    (bpy.types.Scene, "bmc_preset_choice"),
)


def register_addon_properties():
    bpy.types.Object.merge_ib_hash = bpy.props.StringProperty(
        name="IB Hash",
        default="",
        description="Manual override for the mesh IB hash used by Bone Merge Capture.",
    )
    bpy.types.Object.merge_match_index_count = bpy.props.IntProperty(
        name="Match Index Count",
        default=-1,
        min=-1,
        description="Manual override for the mesh match_index_count used by Bone Merge Capture.",
    )
    bpy.types.Object.merge_ib_autodetected = bpy.props.BoolProperty(
        name="IB Auto",
        default=True,
        description="Whether IB hash and match_index_count are currently inferred from the object name.",
    )

    bpy.types.Scene.bmc_frameanalysis_dir = bpy.props.StringProperty(
        name="FrameAnalysis Dir",
        default="",
        subtype="DIR_PATH",
        description="FrameAnalysis directory that contains log.txt and dumped buffers.",
    )
    bpy.types.Scene.bmc_output_dir = bpy.props.StringProperty(
        name="Output Dir",
        default="",
        subtype="DIR_PATH",
        description="Optional output directory for capture_manifest.json and BoneStore.ini. Leave empty to use the FrameAnalysis directory.",
    )
    bpy.types.Scene.bmc_manifest_path = bpy.props.StringProperty(
        name="Manifest Path",
        default="",
        subtype="FILE_PATH",
        description="Generated capture manifest path.",
    )
    bpy.types.Scene.bmc_ini_path = bpy.props.StringProperty(
        name="BoneStore INI",
        default="",
        subtype="FILE_PATH",
        description="Generated BoneStore.ini path.",
    )
    bpy.types.Scene.bmc_target_collection = bpy.props.PointerProperty(
        name="Target Collection",
        type=bpy.types.Collection,
        description="Collection containing mesh objects that should participate in Bone Merge Capture scanning.",
    )
    bpy.types.Scene.bmc_export_collection = bpy.props.PointerProperty(
        name="Export Collection",
        type=bpy.types.Collection,
        description="Root collection containing final draw chunk child collections for BI4 export.",
    )
    bpy.types.Scene.bmc_export_manifest_path = bpy.props.StringProperty(
        name="Export Manifest",
        default="",
        subtype="FILE_PATH",
        description="Generated export manifest path.",
    )
    bpy.types.Scene.bmc_source_ini_path = bpy.props.StringProperty(
        name="Main INI",
        default="",
        subtype="FILE_PATH",
        description="Source mod INI that should receive the generated last-shadow-host split logic.",
    )
    bpy.types.Scene.bmc_shadow_host_hash = bpy.props.StringProperty(
        name="Shadow Host Hash",
        default="",
        description="Optional manual override for the final vs==200 shadow host IB hash. Leave empty for auto-detect.",
    )
    bpy.types.Scene.bmc_shadow_host_match_index_count = bpy.props.IntProperty(
        name="Shadow Host Count",
        default=-1,
        min=-1,
        description="Optional manual override for the final vs==200 shadow host match_index_count. Leave at -1 for auto-detect.",
    )
    bpy.types.Scene.bmc_shadow_host_vs_hash = bpy.props.StringProperty(
        name="Last Shadow VS",
        default="",
        description="VS hash of the detected final vs==200 shadow host draw. Filled by Scan for reference.",
    )
    bpy.types.Scene.bmc_merge_same_bone_groups = bpy.props.BoolProperty(
        name="Scan 后自动合并同骨顶点组",
        default=False,
        description="When enabled, Scan and Generate will run seam-filtered same-bone analysis and merge matching vertex-group weights after remapping. This can be slow.",
    )
    bpy.types.Scene.bmc_target_items = bpy.props.CollectionProperty(type=BMC_TargetItem)
    bpy.types.Scene.bmc_target_index = bpy.props.IntProperty(name="Target Index", default=0, min=0)
    bpy.types.Scene.bmc_alias_items = bpy.props.CollectionProperty(type=BMC_AliasItem)
    bpy.types.Scene.bmc_alias_index = bpy.props.IntProperty(name="Alias Index", default=0, min=0)
    bpy.types.Scene.bmc_preset_name = bpy.props.StringProperty(
        name="Preset Name",
        default="",
        description="Preset name used when saving the current target/alias configuration.",
    )
    bpy.types.Scene.bmc_preset_choice = bpy.props.EnumProperty(
        name="Preset",
        items=_preset_enum_items,
        description="Saved target/duplicate-bone presets.",
        update=_preset_choice_update,
    )


def unregister_addon_properties():
    for owner, attribute_name in REGISTERED_PROPERTY_PATHS:
        if hasattr(owner, attribute_name):
            delattr(owner, attribute_name)
