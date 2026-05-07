"""Register Blender properties used by the Bone Merge Capture plugin."""

from __future__ import annotations

import bpy

from .constants import BMC_GLOBAL_POOL_GENERATION_PROP, BMC_GLOBAL_SOURCE_KEY_PROP


EXPORT_MODE_ITEMS = (
    ("BUFFER_ONLY", "Buffer Only", "Export all runtime buffer files and export_manifest.json without regenerating BoneStore.ini"),
    ("BUFFER_AND_INI", "Buffer + INI", "Export buffer files, HLSL assets, export_manifest.json, and BoneStore.ini"),
)


class BMC_CandidateItem(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    display_name: bpy.props.StringProperty(name="Candidate", default="")
    ib_hash: bpy.props.StringProperty(name="IB Hash", default="")
    match_first_index: bpy.props.IntProperty(name="First Index", default=0, min=0)
    match_index_count: bpy.props.IntProperty(name="Index Count", default=0, min=0)
    import_draw_index: bpy.props.IntProperty(name="Import Draw", default=-1, min=-1)
    local_bone_count: bpy.props.IntProperty(name="Compact Bones", default=0, min=0)
    draw_count: bpy.props.IntProperty(name="Draws", default=0, min=0)
    shadow_draw_count: bpy.props.IntProperty(name="Shadow Draws", default=0, min=0)
    shadow_capture_ready: bpy.props.BoolProperty(name="Shadow Capture", default=False)
    lod_match_excluded: bpy.props.BoolProperty(name="Skip LOD Match", default=False)
    lod_match_excluded_reason: bpy.props.StringProperty(name="LOD Skip Reason", default="")
    status: bpy.props.StringProperty(name="Status", default="")
    manual: bpy.props.BoolProperty(name="Manual", default=False)


class BMC_LodMappingItem(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    canonical_global_bone: bpy.props.IntProperty(name="Canonical Global", default=0, min=0)
    mapped_lod_global_bone: bpy.props.IntProperty(name="LOD Global", default=-1, min=-1)
    lod_record_key: bpy.props.StringProperty(name="LOD Source", default="")
    lod_local_bone: bpy.props.IntProperty(name="LOD Local", default=-1, min=-1)
    votes: bpy.props.IntProperty(name="Votes", default=0, min=0)
    average_distance: bpy.props.FloatProperty(name="Avg Distance", default=0.0, min=0.0)
    status: bpy.props.StringProperty(name="Status", default="")
    score: bpy.props.FloatProperty(name="Score", default=0.0)
    note: bpy.props.StringProperty(name="Note", default="")


class BMC_LodFallbackItem(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    canonical_global_bone: bpy.props.IntProperty(name="Missing Global", default=0, min=0)
    donor_global_bone: bpy.props.IntProperty(name="Donor Global", default=-1, min=-1)
    lod_record_key: bpy.props.StringProperty(name="LOD Source", default="")
    lod_local_bone: bpy.props.IntProperty(name="LOD Local", default=-1, min=-1)
    method: bpy.props.StringProperty(name="Method", default="")
    confidence: bpy.props.FloatProperty(name="Confidence", default=0.0, min=0.0, max=1.0)
    status: bpy.props.StringProperty(name="Status", default="")
    note: bpy.props.StringProperty(name="Note", default="")


REGISTERED_PROPERTY_PATHS = (
    (bpy.types.Object, "merge_ib_hash"),
    (bpy.types.Object, "merge_match_index_count"),
    (bpy.types.Object, "merge_ib_autodetected"),
    (bpy.types.Object, BMC_GLOBAL_POOL_GENERATION_PROP),
    (bpy.types.Object, BMC_GLOBAL_SOURCE_KEY_PROP),
    (bpy.types.Scene, "bmc_frameanalysis_dir"),
    (bpy.types.Scene, "bmc_lod_frameanalysis_dir"),
    (bpy.types.Scene, "bmc_output_dir"),
    (bpy.types.Scene, "bmc_manifest_path"),
    (bpy.types.Scene, "bmc_lod_manifest_path"),
    (bpy.types.Scene, "bmc_ini_path"),
    (bpy.types.Scene, "bmc_export_collection"),
    (bpy.types.Scene, "bmc_export_mode"),
    (bpy.types.Scene, "bmc_export_manifest_path"),
    (bpy.types.Scene, "bmc_shadow_host_hash"),
    (bpy.types.Scene, "bmc_shadow_host_match_index_count"),
    (bpy.types.Scene, "bmc_shadow_host_vs_hash"),
    (bpy.types.Scene, "bmc_lod_shadow_host_hash"),
    (bpy.types.Scene, "bmc_lod_shadow_host_match_index_count"),
    (bpy.types.Scene, "bmc_lod_shadow_host_vs_hash"),
    (bpy.types.Scene, "bmc_lod_match_summary"),
    (bpy.types.Scene, "bmc_lod_match_warning"),
    (bpy.types.Scene, "bmc_lod_fallback_summary"),
    (bpy.types.Scene, "bmc_lod_fallback_warning"),
    (bpy.types.Scene, "bmc_mapping_payload_json"),
    (bpy.types.Scene, "bmc_candidate_items"),
    (bpy.types.Scene, "bmc_candidate_index"),
    (bpy.types.Scene, "bmc_candidate_add_hash"),
    (bpy.types.Scene, "bmc_mirror_flip"),
    (bpy.types.Scene, "bmc_lod_mapping_items"),
    (bpy.types.Scene, "bmc_lod_mapping_index"),
    (bpy.types.Scene, "bmc_lod_fallback_items"),
    (bpy.types.Scene, "bmc_lod_fallback_index"),
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
        description="Cached match_index_count inferred from the mesh or collection name.",
    )
    bpy.types.Object.merge_ib_autodetected = bpy.props.BoolProperty(
        name="IB Auto",
        default=True,
        description="Whether IB hash is currently inferred from the object name.",
    )
    setattr(
        bpy.types.Object,
        BMC_GLOBAL_POOL_GENERATION_PROP,
        bpy.props.StringProperty(
            name="Global Pool Generation",
            default="",
            description="Generation id of the global bone pool used to rename this object's vertex groups.",
        ),
    )
    setattr(
        bpy.types.Object,
        BMC_GLOBAL_SOURCE_KEY_PROP,
        bpy.props.StringProperty(
            name="Global Source Key",
            default="",
            description="IB/count/first source key used to rename this object's vertex groups.",
        ),
    )

    bpy.types.Scene.bmc_frameanalysis_dir = bpy.props.StringProperty(
        name="FrameAnalysis Dir",
        default="",
        subtype="DIR_PATH",
        description="FrameAnalysis directory that contains log.txt and dumped buffers.",
    )
    bpy.types.Scene.bmc_lod_frameanalysis_dir = bpy.props.StringProperty(
        name="LOD FrameAnalysis Dir",
        default="",
        subtype="DIR_PATH",
        description="LOD FrameAnalysis directory used for source capture variant scanning.",
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
        description="Generated capture manifest path from the latest Analyze Main run.",
    )
    bpy.types.Scene.bmc_lod_manifest_path = bpy.props.StringProperty(
        name="LOD Manifest Path",
        default="",
        subtype="FILE_PATH",
        description="Generated LOD capture manifest path from the latest Analyze LOD run.",
    )
    bpy.types.Scene.bmc_ini_path = bpy.props.StringProperty(
        name="BoneStore INI",
        default="",
        subtype="FILE_PATH",
        description="Generated BoneStore.ini path from the latest Export run.",
    )
    bpy.types.Scene.bmc_export_collection = bpy.props.PointerProperty(
        name="Export Root Collection",
        type=bpy.types.Collection,
        description="Single editable root collection containing final IB region and part collections for export.",
    )
    bpy.types.Scene.bmc_export_mode = bpy.props.EnumProperty(
        name="Export Mode",
        items=EXPORT_MODE_ITEMS,
        default="BUFFER_ONLY",
        description="Choose whether this export writes only buffers or also regenerates BoneStore.ini.",
    )
    bpy.types.Scene.bmc_export_manifest_path = bpy.props.StringProperty(
        name="Export Manifest",
        default="",
        subtype="FILE_PATH",
        description="Authoritative export manifest path produced by the latest Prepare Export run.",
    )
    bpy.types.Scene.bmc_shadow_host_hash = bpy.props.StringProperty(
        name="Shadow Host Hash",
        default="",
        description="Optional manual override for the final vs==200 shadow host IB hash. Leave empty for the latest Analyze Main value.",
    )
    bpy.types.Scene.bmc_shadow_host_match_index_count = bpy.props.IntProperty(
        name="Shadow Host Count",
        default=-1,
        min=-1,
        description="Optional manual override for the final vs==200 shadow host match_index_count. Leave at -1 for the latest Analyze Main value.",
    )
    bpy.types.Scene.bmc_shadow_host_vs_hash = bpy.props.StringProperty(
        name="Last Shadow VS",
        default="",
        description="VS hash of the detected final vs==200 shadow host draw. Filled by Analyze Main for reference.",
    )
    bpy.types.Scene.bmc_lod_shadow_host_hash = bpy.props.StringProperty(
        name="LOD Shadow Host Hash",
        default="",
        description="Detected final vs==200 LOD shadow host IB hash.",
    )
    bpy.types.Scene.bmc_lod_shadow_host_match_index_count = bpy.props.IntProperty(
        name="LOD Shadow Host Count",
        default=-1,
        min=-1,
        description="Detected final vs==200 LOD shadow host match_index_count.",
    )
    bpy.types.Scene.bmc_lod_shadow_host_vs_hash = bpy.props.StringProperty(
        name="LOD Last Shadow VS",
        default="",
        description="VS hash of the detected final vs==200 LOD shadow host draw.",
    )
    bpy.types.Scene.bmc_lod_match_summary = bpy.props.StringProperty(
        name="LOD Match Summary",
        default="",
        description="Compact summary of the latest Analyze LOD result.",
    )
    bpy.types.Scene.bmc_lod_match_warning = bpy.props.StringProperty(
        name="LOD Match Warning",
        default="",
        description="Most important warning from the latest Analyze LOD result.",
    )
    bpy.types.Scene.bmc_lod_fallback_summary = bpy.props.StringProperty(
        name="LOD Fallback Summary",
        default="",
        description="Compact summary of the latest LOD fallback preview/apply operation.",
    )
    bpy.types.Scene.bmc_lod_fallback_warning = bpy.props.StringProperty(
        name="LOD Fallback Warning",
        default="",
        description="Most important warning from the latest LOD fallback operation.",
    )
    bpy.types.Scene.bmc_mapping_payload_json = bpy.props.StringProperty(
        name="Mapping Payload JSON",
        default="",
        options={"HIDDEN"},
        description="Compact cache of the current analysis state used by runtime tools.",
    )
    bpy.types.Scene.bmc_candidate_items = bpy.props.CollectionProperty(type=BMC_CandidateItem)
    bpy.types.Scene.bmc_candidate_index = bpy.props.IntProperty(name="Candidate Index", default=0, min=0)
    bpy.types.Scene.bmc_candidate_add_hash = bpy.props.StringProperty(
        name="Candidate IB",
        default="",
        description="IB hash to add to the Candidate IB list. If the latest manifest contains several slices for the hash, all missing slices are added.",
    )
    bpy.types.Scene.bmc_mirror_flip = bpy.props.BoolProperty(
        name="Mirror Flip",
        default=True,
        description="Mirror imported geometry on the X axis. This matches the default game-to-Blender import orientation.",
    )
    bpy.types.Scene.bmc_lod_mapping_items = bpy.props.CollectionProperty(type=BMC_LodMappingItem)
    bpy.types.Scene.bmc_lod_mapping_index = bpy.props.IntProperty(name="LOD Mapping Index", default=0, min=0)
    bpy.types.Scene.bmc_lod_fallback_items = bpy.props.CollectionProperty(type=BMC_LodFallbackItem)
    bpy.types.Scene.bmc_lod_fallback_index = bpy.props.IntProperty(name="LOD Fallback Index", default=0, min=0)

def unregister_addon_properties():
    for owner, attribute_name in REGISTERED_PROPERTY_PATHS:
        if hasattr(owner, attribute_name):
            delattr(owner, attribute_name)
