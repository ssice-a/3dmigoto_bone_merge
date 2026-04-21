"""Register Blender properties used by the Bone Merge Capture plugin."""

from __future__ import annotations

import bpy

from .core.presets import list_preset_names


def _preset_enum_items(_self, _context):
    names = list_preset_names()
    if not names:
        return [("__NONE__", "(No Presets)", "No saved presets available")]
    return [(name, name, f"Load preset {name}") for name in names]


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
    )


def unregister_addon_properties():
    for owner, attribute_name in REGISTERED_PROPERTY_PATHS:
        if hasattr(owner, attribute_name):
            delattr(owner, attribute_name)
