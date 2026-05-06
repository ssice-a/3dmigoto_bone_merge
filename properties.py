"""Register Blender properties used by the Bone Merge Capture plugin."""

from __future__ import annotations

import bpy

from .constants import BMC_GLOBAL_POOL_GENERATION_PROP, BMC_GLOBAL_SOURCE_KEY_PROP
from .core.mapping_payload import apply_mapping_payload_to_scene
from .core.presets import list_preset_names


def _preset_enum_items(_self, _context):
    names = list_preset_names()
    if not names:
        return [("__NONE__", "(No Presets)", "No saved presets available")]
    return [(name, name, f"Load preset {name}") for name in names]


def _apply_preset_payload_to_scene(scene, preset_name: str, payload: dict) -> None:
    apply_mapping_payload_to_scene(scene, payload, preset_name=preset_name)


def _preset_choice_update(self, _context):
    return


class BMC_TargetItem(bpy.types.PropertyGroup):
    object_name: bpy.props.StringProperty(name="Object", default="")
    object_ref: bpy.props.PointerProperty(name="Object Ref", type=bpy.types.Object)
    ib_hash: bpy.props.StringProperty(name="IB Hash", default="")
    match_index_count: bpy.props.IntProperty(name="Match Index Count", default=0, min=0)
    local_bone_count: bpy.props.IntProperty(name="Local Bone Count", default=0, min=0)
    autodetected: bpy.props.BoolProperty(name="Auto", default=True)
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)


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
    status: bpy.props.StringProperty(name="Status", default="")
    manual: bpy.props.BoolProperty(name="Manual", default=False)


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


class BMC_SeamMatchItem(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    object_name: bpy.props.StringProperty(name="Object", default="")
    object_ref: bpy.props.PointerProperty(name="Object Ref", type=bpy.types.Object)


class BMC_SeamAliasItem(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    src_object_name: bpy.props.StringProperty(name="Src Object", default="")
    src_group: bpy.props.IntProperty(name="Src Group", default=0, min=0)
    dst_object_name: bpy.props.StringProperty(name="Dst Object", default="")
    dst_group: bpy.props.IntProperty(name="Dst Group", default=0, min=0)
    votes: bpy.props.IntProperty(name="Votes", default=0, min=0)
    score: bpy.props.FloatProperty(name="Score", default=0.0)
    average_distance: bpy.props.FloatProperty(name="Avg Distance", default=0.0, min=0.0)
    average_weight_difference: bpy.props.FloatProperty(name="Avg Weight Diff", default=0.0, min=0.0)


class BMC_LodTargetItem(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    object_name: bpy.props.StringProperty(name="Object", default="")
    object_ref: bpy.props.PointerProperty(name="Object Ref", type=bpy.types.Object)
    ib_hash: bpy.props.StringProperty(name="IB Hash", default="")
    match_index_count: bpy.props.IntProperty(name="Match Index Count", default=0, min=0)
    local_bone_count: bpy.props.IntProperty(name="Local Bone Count", default=0, min=0)


class BMC_LodMappingItem(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    canonical_global_bone: bpy.props.IntProperty(name="Canonical Global", default=0, min=0)
    mapped_lod_global_bone: bpy.props.IntProperty(name="LOD Global", default=-1, min=-1)
    status: bpy.props.StringProperty(name="Status", default="")
    score: bpy.props.FloatProperty(name="Score", default=0.0)
    note: bpy.props.StringProperty(name="Note", default="")


REGISTERED_PROPERTY_PATHS = (
    (bpy.types.Object, "merge_ib_hash"),
    (bpy.types.Object, "merge_match_index_count"),
    (bpy.types.Object, "merge_ib_autodetected"),
    (bpy.types.Object, BMC_GLOBAL_POOL_GENERATION_PROP),
    (bpy.types.Object, BMC_GLOBAL_SOURCE_KEY_PROP),
    (bpy.types.Scene, "bmc_frameanalysis_dir"),
    (bpy.types.Scene, "bmc_target_ib_hash"),
    (bpy.types.Scene, "bmc_lod_frameanalysis_dir"),
    (bpy.types.Scene, "bmc_output_dir"),
    (bpy.types.Scene, "bmc_manifest_path"),
    (bpy.types.Scene, "bmc_lod_manifest_path"),
    (bpy.types.Scene, "bmc_ini_path"),
    (bpy.types.Scene, "bmc_target_collection"),
    (bpy.types.Scene, "bmc_export_collection"),
    (bpy.types.Scene, "bmc_export_build_collection"),
    (bpy.types.Scene, "bmc_export_manifest_path"),
    (bpy.types.Scene, "bmc_source_ini_path"),
    (bpy.types.Scene, "bmc_shadow_host_hash"),
    (bpy.types.Scene, "bmc_shadow_host_match_index_count"),
    (bpy.types.Scene, "bmc_shadow_host_vs_hash"),
    (bpy.types.Scene, "bmc_lod_shadow_host_hash"),
    (bpy.types.Scene, "bmc_lod_shadow_host_match_index_count"),
    (bpy.types.Scene, "bmc_lod_shadow_host_vs_hash"),
    (bpy.types.Scene, "bmc_scan_auto_apply_mapping"),
    (bpy.types.Scene, "bmc_mapping_payload_json"),
    (bpy.types.Scene, "bmc_target_items"),
    (bpy.types.Scene, "bmc_target_index"),
    (bpy.types.Scene, "bmc_candidate_items"),
    (bpy.types.Scene, "bmc_candidate_index"),
    (bpy.types.Scene, "bmc_candidate_add_hash"),
    (bpy.types.Scene, "bmc_mirror_flip"),
    (bpy.types.Scene, "bmc_lod_target_items"),
    (bpy.types.Scene, "bmc_lod_target_index"),
    (bpy.types.Scene, "bmc_alias_items"),
    (bpy.types.Scene, "bmc_alias_index"),
    (bpy.types.Scene, "bmc_lod_mapping_items"),
    (bpy.types.Scene, "bmc_lod_mapping_index"),
    (bpy.types.Scene, "bmc_seam_match_items"),
    (bpy.types.Scene, "bmc_seam_match_index"),
    (bpy.types.Scene, "bmc_seam_alias_items"),
    (bpy.types.Scene, "bmc_seam_alias_index"),
    (bpy.types.Scene, "bmc_seam_pair_summary"),
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
        description="Legacy cached match_index_count from older Bone Merge Capture presets.",
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
    bpy.types.Scene.bmc_target_ib_hash = bpy.props.StringProperty(
        name="Target IB Hash",
        default="",
        description="Starting IB hash for the redesigned Main Analyze flow. If empty, enabled Target Objects are used.",
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
        description="Generated capture manifest path from the latest Scan.",
    )
    bpy.types.Scene.bmc_lod_manifest_path = bpy.props.StringProperty(
        name="LOD Manifest Path",
        default="",
        subtype="FILE_PATH",
        description="Generated LOD capture manifest path from the latest LOD Scan.",
    )
    bpy.types.Scene.bmc_ini_path = bpy.props.StringProperty(
        name="BoneStore INI",
        default="",
        subtype="FILE_PATH",
        description="Generated BoneStore.ini path from the latest Scan / Prepare Export run.",
    )
    bpy.types.Scene.bmc_target_collection = bpy.props.PointerProperty(
        name="Target Collection",
        type=bpy.types.Collection,
        description="Collection containing mesh objects that should participate in Bone Merge Capture scanning.",
    )
    bpy.types.Scene.bmc_export_collection = bpy.props.PointerProperty(
        name="Export Source Collection",
        type=bpy.types.Collection,
        description="Editable source collection containing final draw chunk child collections for export.",
    )
    bpy.types.Scene.bmc_export_build_collection = bpy.props.PointerProperty(
        name="Export Build Collection",
        type=bpy.types.Collection,
        description="Disposable generated collection rebuilt from Export Source Collection during Prepare Export.",
    )
    bpy.types.Scene.bmc_export_manifest_path = bpy.props.StringProperty(
        name="Export Manifest",
        default="",
        subtype="FILE_PATH",
        description="Authoritative export manifest path produced by the latest Prepare Export run.",
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
        description="Optional manual override for the final vs==200 shadow host IB hash. Leave empty for the latest Scan / Preset value.",
    )
    bpy.types.Scene.bmc_shadow_host_match_index_count = bpy.props.IntProperty(
        name="Shadow Host Count",
        default=-1,
        min=-1,
        description="Optional manual override for the final vs==200 shadow host match_index_count. Leave at -1 for the latest Scan / Preset value.",
    )
    bpy.types.Scene.bmc_shadow_host_vs_hash = bpy.props.StringProperty(
        name="Last Shadow VS",
        default="",
        description="VS hash of the detected final vs==200 shadow host draw. Filled by Scan / Load Mapping Preset for reference.",
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
    bpy.types.Scene.bmc_scan_auto_apply_mapping = bpy.props.BoolProperty(
        name="Auto remap after Scan",
        default=False,
        description="When enabled, Scan will additionally rebuild same-bone aliases, remap target meshes, and merge same-bone weights. This is slower.",
    )
    bpy.types.Scene.bmc_mapping_payload_json = bpy.props.StringProperty(
        name="Mapping Payload JSON",
        default="",
        options={"HIDDEN"},
        description="Frozen scan mapping payload used by presets and Apply Mapping.",
    )
    bpy.types.Scene.bmc_target_items = bpy.props.CollectionProperty(type=BMC_TargetItem)
    bpy.types.Scene.bmc_target_index = bpy.props.IntProperty(name="Target Index", default=0, min=0)
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
    bpy.types.Scene.bmc_lod_target_items = bpy.props.CollectionProperty(type=BMC_LodTargetItem)
    bpy.types.Scene.bmc_lod_target_index = bpy.props.IntProperty(name="LOD Target Index", default=0, min=0)
    bpy.types.Scene.bmc_alias_items = bpy.props.CollectionProperty(type=BMC_AliasItem)
    bpy.types.Scene.bmc_alias_index = bpy.props.IntProperty(name="Alias Index", default=0, min=0)
    bpy.types.Scene.bmc_lod_mapping_items = bpy.props.CollectionProperty(type=BMC_LodMappingItem)
    bpy.types.Scene.bmc_lod_mapping_index = bpy.props.IntProperty(name="LOD Mapping Index", default=0, min=0)
    bpy.types.Scene.bmc_seam_match_items = bpy.props.CollectionProperty(type=BMC_SeamMatchItem)
    bpy.types.Scene.bmc_seam_match_index = bpy.props.IntProperty(name="Seam Object Index", default=0, min=0)
    bpy.types.Scene.bmc_seam_alias_items = bpy.props.CollectionProperty(type=BMC_SeamAliasItem)
    bpy.types.Scene.bmc_seam_alias_index = bpy.props.IntProperty(name="Seam Mapping Index", default=0, min=0)
    bpy.types.Scene.bmc_seam_pair_summary = bpy.props.StringProperty(
        name="Matched Pairs",
        default="",
        description="Summary of object pairs that contributed seam vertex-group mappings.",
    )
    bpy.types.Scene.bmc_preset_name = bpy.props.StringProperty(
        name="Preset Name",
        default="",
        description="Mapping preset name used when saving the current frozen scan snapshot.",
    )
    bpy.types.Scene.bmc_preset_choice = bpy.props.EnumProperty(
        name="Preset",
        items=_preset_enum_items,
        description="Saved mapping presets built from frozen scan results.",
    )


def unregister_addon_properties():
    for owner, attribute_name in REGISTERED_PROPERTY_PATHS:
        if hasattr(owner, attribute_name):
            delattr(owner, attribute_name)
