"""Register Blender properties used by the Bone Merge Capture plugin."""

from __future__ import annotations

import os

import bpy

from .constants import BMC_GLOBAL_POOL_GENERATION_PROP, BMC_GLOBAL_SOURCE_KEY_PROP, BMC_TEXTURE_MARKS_PROP
from .core.io import read_json
from .core.texture_marks import (
    build_texture_mark_payload,
    dump_texture_mark_payload,
    load_texture_mark_payload,
    region_label,
    slot_sort_key,
    texture_candidates_for_draw,
)


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


class BMC_TextureMarkItem(bpy.types.PropertyGroup):
    slot: bpy.props.StringProperty(name="Slot", default="")
    hash_value: bpy.props.StringProperty(name="Hash", default="")
    source_path: bpy.props.StringProperty(name="Source", default="", subtype="FILE_PATH")
    filename: bpy.props.StringProperty(name="Filename", default="")
    semantic: bpy.props.StringProperty(name="Semantic", default="")
    semantic_index: bpy.props.IntProperty(name="Semantic Index", default=0, min=0)
    draw_index: bpy.props.IntProperty(name="Draw", default=0, min=0)
    ps_hash: bpy.props.StringProperty(name="PS Hash", default="")
    rt_count: bpy.props.IntProperty(name="RT Count", default=-1, min=-1)


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
    (bpy.types.Scene, "bmc_uv_flip_v"),
    (bpy.types.Scene, "bmc_lod_mapping_items"),
    (bpy.types.Scene, "bmc_lod_mapping_index"),
    (bpy.types.Scene, "bmc_lod_fallback_items"),
    (bpy.types.Scene, "bmc_lod_fallback_index"),
    (bpy.types.Scene, "bmc_texture_marks_json"),
    (bpy.types.Scene, "bmc_texture_region"),
    (bpy.types.Scene, "bmc_texture_draw"),
    (bpy.types.Scene, "bmc_texture_mark_items"),
    (bpy.types.Scene, "bmc_texture_mark_index"),
)


def _read_manifest_payload(scene) -> dict:
    manifest_path = bpy.path.abspath(str(getattr(scene, "bmc_manifest_path", "") or ""))
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _store_texture_mark_payload_on_scene(scene, payload: dict) -> None:
    serialized = dump_texture_mark_payload(payload)
    scene.bmc_texture_marks_json = serialized
    if getattr(scene, "bmc_export_collection", None) is not None:
        scene.bmc_export_collection[BMC_TEXTURE_MARKS_PROP] = serialized


def texture_mark_payload_from_scene(scene) -> dict:
    stored = str(getattr(scene, "bmc_texture_marks_json", "") or "").strip()
    if not stored and getattr(scene, "bmc_export_collection", None) is not None:
        stored = str(scene.bmc_export_collection.get(BMC_TEXTURE_MARKS_PROP, "") or "")
    existing = load_texture_mark_payload(stored)
    manifest = _read_manifest_payload(scene)
    if manifest.get("texture_candidates"):
        payload = build_texture_mark_payload(manifest, existing)
    else:
        payload = existing
    if payload and dump_texture_mark_payload(payload) != stored:
        _store_texture_mark_payload_on_scene(scene, payload)
    return payload


def store_texture_mark_payload_on_scene(scene, payload: dict) -> None:
    _store_texture_mark_payload_on_scene(scene, payload)


def _texture_region_items(self, context):  # pylint: disable=unused-argument
    payload = texture_mark_payload_from_scene(context.scene)
    candidates = payload.get("candidates", {})
    if not isinstance(candidates, dict) or not candidates:
        return [("__none__", "No texture candidates", "Run Analyze Main first")]
    return [
        (str(region_key), region_label(str(region_key)), "Texture candidate region")
        for region_key in sorted(candidates)
    ]


def _texture_draw_items(self, context):  # pylint: disable=unused-argument
    scene = context.scene
    payload = texture_mark_payload_from_scene(scene)
    region_key = str(getattr(scene, "bmc_texture_region", "") or "")
    candidates = payload.get("candidates", {})
    region_candidates = candidates.get(region_key, {}) if isinstance(candidates, dict) else {}
    if not isinstance(region_candidates, dict) or not region_candidates:
        return [("__none__", "No draw candidates", "Select another texture region")]
    draws = payload.get("draws", {}).get(region_key, {}) if isinstance(payload.get("draws", {}), dict) else {}
    items = []
    for draw_key in sorted(region_candidates, key=lambda value: int(value) if str(value).isdigit() else 0):
        meta = draws.get(str(draw_key), {}) if isinstance(draws, dict) else {}
        slot_count = len(region_candidates.get(draw_key, {}) or {})
        rt_count = int(meta.get("rt_count", -1) or -1) if isinstance(meta, dict) else -1
        ps_hash = str(meta.get("ps_hash", "") or "") if isinstance(meta, dict) else ""
        label = f"{int(draw_key):06d}  textures={slot_count}  RT={rt_count}"
        if ps_hash:
            label += f"  ps={ps_hash[:8]}"
        items.append((str(draw_key), label, "Draw texture candidates"))
    return items or [("__none__", "No draw candidates", "Select another texture region")]


def sync_texture_mark_items(scene):
    if not hasattr(scene, "bmc_texture_mark_items"):
        return
    payload = texture_mark_payload_from_scene(scene)
    region_key = str(getattr(scene, "bmc_texture_region", "") or "")
    draw_key = str(getattr(scene, "bmc_texture_draw", "") or "")
    draw_candidates, draw_marks = texture_candidates_for_draw(payload, region_key, draw_key)
    scene.bmc_texture_mark_items.clear()
    for slot, binding in sorted(draw_candidates.items(), key=lambda item: slot_sort_key(item[0])):
        item = scene.bmc_texture_mark_items.add()
        item.slot = str(slot)
        item.hash_value = str(binding.get("hash", "") or "")
        item.source_path = str(binding.get("source_path", "") or "")
        item.filename = os.path.basename(item.source_path)
        item.draw_index = int(binding.get("draw_index", 0) or 0)
        item.ps_hash = str(binding.get("ps_hash", "") or "")
        item.rt_count = int(binding.get("rt_count", -1) or -1)
        mark = draw_marks.get(slot, {})
        if isinstance(mark, dict):
            item.semantic = str(mark.get("semantic", "") or "")
            item.semantic_index = int(mark.get("semantic_index", 0) or 0)
    if scene.bmc_texture_mark_index >= len(scene.bmc_texture_mark_items):
        scene.bmc_texture_mark_index = max(0, len(scene.bmc_texture_mark_items) - 1)


def _update_texture_draw(self, context):  # pylint: disable=unused-argument
    sync_texture_mark_items(context.scene)


def _update_texture_region(self, context):  # pylint: disable=unused-argument
    scene = context.scene
    payload = texture_mark_payload_from_scene(scene)
    default_draws = payload.get("default_draws", {})
    region_key = str(getattr(scene, "bmc_texture_region", "") or "")
    draw_key = str(default_draws.get(region_key, "") or "") if isinstance(default_draws, dict) else ""
    if draw_key:
        try:
            scene.bmc_texture_draw = draw_key
        except TypeError:
            sync_texture_mark_items(scene)
    else:
        sync_texture_mark_items(scene)


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
        name="Big Export Collection",
        type=bpy.types.Collection,
        description="Single editable big collection containing final IB region and part collections. The generated INI uses this collection name.",
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
    bpy.types.Scene.bmc_uv_flip_v = bpy.props.BoolProperty(
        name="Flip UV V",
        default=True,
        description="Flip UV V on import and flip it back on export so Blender view matches the game texture orientation.",
    )
    bpy.types.Scene.bmc_lod_mapping_items = bpy.props.CollectionProperty(type=BMC_LodMappingItem)
    bpy.types.Scene.bmc_lod_mapping_index = bpy.props.IntProperty(name="LOD Mapping Index", default=0, min=0)
    bpy.types.Scene.bmc_lod_fallback_items = bpy.props.CollectionProperty(type=BMC_LodFallbackItem)
    bpy.types.Scene.bmc_lod_fallback_index = bpy.props.IntProperty(name="LOD Fallback Index", default=0, min=0)
    bpy.types.Scene.bmc_texture_marks_json = bpy.props.StringProperty(
        name="Texture Marks JSON",
        default="",
        options={"HIDDEN"},
        description="Hash-style texture replacement marks cached from the latest Analyze Main run.",
    )
    bpy.types.Scene.bmc_texture_region = bpy.props.EnumProperty(
        name="Texture Region",
        items=_texture_region_items,
        update=_update_texture_region,
        description="IB region whose texture candidates are shown.",
    )
    bpy.types.Scene.bmc_texture_draw = bpy.props.EnumProperty(
        name="Texture Draw",
        items=_texture_draw_items,
        update=_update_texture_draw,
        description="Draw whose PS texture candidates are shown.",
    )
    bpy.types.Scene.bmc_texture_mark_items = bpy.props.CollectionProperty(type=BMC_TextureMarkItem)
    bpy.types.Scene.bmc_texture_mark_index = bpy.props.IntProperty(name="Texture Mark Index", default=0, min=0)

def unregister_addon_properties():
    for owner, attribute_name in REGISTERED_PROPERTY_PATHS:
        if hasattr(owner, attribute_name):
            delattr(owner, attribute_name)
