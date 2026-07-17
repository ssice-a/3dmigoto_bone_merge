"""Blender operators for the Bone Merge Capture plugin."""

from __future__ import annotations

import json
import os
import re
import time

import bpy

from .constants import (
    BMC_GLOBAL_POOL_GENERATION_PROP,
    BMC_GLOBAL_REMAP_PROP,
    BMC_GLOBAL_SOURCE_KEY_PROP,
    BMC_TEXTURE_SLOTS_PROP,
    BMC_TOGGLE_DRAW_SETS_PROP,
    BMC_VERTEX_GROUP_STATE_PROP,
)
from .core.blender_ops import (
    apply_group_remaps_to_meshes,
    infer_local_bone_count_from_mesh,
    resolve_mesh_identity,
)
from .core.mapping_payload import (
    build_mapping_payload,
    load_mapping_payload_from_scene,
    store_mapping_payload_on_scene,
)
from .core.export_prepare import prepare_export_collection
from .core.identity import infer_mesh_identity_from_name
from .core.io import read_json, write_json
from .core.import_candidates import import_selected_candidates
from .core.lod_analyze import analyze_lod_for_manifest, review_lod_global_pool_coverage
from .core.lod_fallback import (
    apply_lod_fallbacks_to_manifest,
    filter_lod_fallback_preview,
    preview_lod_fallbacks_for_export,
)
from .core.lod_profiles import (
    assert_lod_profiles_exportable,
    ensure_lod_profiles,
    first_lod_profile_issue,
    invalidate_lod_profiles,
    lod_profile_is_stale,
    profile_id_for,
    rebuild_lod_aggregate,
    remove_lod_profile,
    sync_lod_profile_settings,
    upsert_lod_profile,
)
from .core.value_utils import int_or_default
from .core.main_analyze import build_bone_pool_order, write_main_analysis_manifest
from .core.numpy_compat import numpy_status
from .core.seam_matcher import build_and_apply_seam_mapping
from .core.texture_marks import (
    dump_texture_mark_payload,
    marked_texture_bindings,
    slot_sort_key,
    texture_candidates_for_draw,
    validate_texture_hash,
)
from .core.texture_materials import apply_material_from_texture_bindings
from .core.toggle_draw_sets import normalize_key_binding

DEFAULT_EXPORT_COLLECTION_NAME = "BMC Export Sources"
_EXPORT_REGION_NAME_RE = re.compile(r"(?P<hash>[0-9A-Fa-f]{8})[-_](?P<count>\d+)[-_](?P<first>\d+)")
_KEY_EVENT_ALIASES = {
    "ZERO": "0",
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
    "UP_ARROW": "VK_UP",
    "DOWN_ARROW": "VK_DOWN",
    "LEFT_ARROW": "VK_LEFT",
    "RIGHT_ARROW": "VK_RIGHT",
    "NUMPAD_0": "VK_NUMPAD0",
    "NUMPAD_1": "VK_NUMPAD1",
    "NUMPAD_2": "VK_NUMPAD2",
    "NUMPAD_3": "VK_NUMPAD3",
    "NUMPAD_4": "VK_NUMPAD4",
    "NUMPAD_5": "VK_NUMPAD5",
    "NUMPAD_6": "VK_NUMPAD6",
    "NUMPAD_7": "VK_NUMPAD7",
    "NUMPAD_8": "VK_NUMPAD8",
    "NUMPAD_9": "VK_NUMPAD9",
    "RET": "ENTER",
    "SPACE": "SPACE",
    "TAB": "TAB",
    "MINUS": "MINUS",
    "EQUAL": "EQUAL",
    "LEFT_BRACKET": "LEFT_BRACKET",
    "RIGHT_BRACKET": "RIGHT_BRACKET",
    "SEMI_COLON": "SEMICOLON",
    "QUOTE": "QUOTE",
    "COMMA": "COMMA",
    "PERIOD": "PERIOD",
    "SLASH": "SLASH",
    "BACK_SLASH": "BACKSLASH",
    "ACCENT_GRAVE": "GRAVE",
}
_MODIFIER_EVENT_TYPES = {"LEFT_SHIFT", "RIGHT_SHIFT", "LEFT_CTRL", "RIGHT_CTRL", "LEFT_ALT", "RIGHT_ALT", "OSKEY"}




def _ensure_export_collection(context):
    scene = context.scene
    collection = scene.bmc_export_collection
    if collection is None:
        collection = bpy.data.collections.get(DEFAULT_EXPORT_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(DEFAULT_EXPORT_COLLECTION_NAME)
        scene.collection.children.link(collection)
    scene.bmc_export_collection = collection
    return collection


def _export_region_collection_name(
    ib_hash: str,
    match_index_count: int = 0,
    match_first_index: int = 0,
    *,
    bone_capture_available: bool = True,
    lod_match_excluded: bool = False,
) -> str:
    region_name = f"{ib_hash.lower()}-{int(match_index_count)}-{int(match_first_index)}"
    if not bone_capture_available:
        region_name += "-NO_CAPTURE_BONES"
    if lod_match_excluded:
        region_name += "-NO_LOD_DYNAMIC_VB0"
    return region_name


def _ensure_export_region_collection(
    parent_collection,
    ib_hash: str,
    match_index_count: int = 0,
    match_first_index: int = 0,
    *,
    bone_capture_available: bool = True,
    lod_match_excluded: bool = False,
):
    region_name = _export_region_collection_name(
        ib_hash,
        match_index_count,
        match_first_index,
        bone_capture_available=bone_capture_available,
        lod_match_excluded=lod_match_excluded,
    )
    for child in parent_collection.children:
        identity = _resolve_export_region_collection_identity(child)
        if identity == (str(ib_hash or "").lower(), int(match_index_count), int(match_first_index)):
            if child.name != region_name and bpy.data.collections.get(region_name) is None:
                child.name = region_name
            child["bmc_bone_capture_available"] = bool(bone_capture_available)
            child["bmc_lod_match_excluded"] = bool(lod_match_excluded)
            return child
    collection = bpy.data.collections.get(region_name)
    if collection is None:
        collection = bpy.data.collections.new(region_name)
    if all(child.name != collection.name for child in parent_collection.children):
        parent_collection.children.link(collection)
    collection["bmc_bone_capture_available"] = bool(bone_capture_available)
    collection["bmc_lod_match_excluded"] = bool(lod_match_excluded)
    return collection




def _ensure_export_region_collection_if_missing(
    parent_collection,
    ib_hash: str,
    match_index_count: int,
    match_first_index: int = 0,
    *,
    bone_capture_available: bool = True,
    lod_match_excluded: bool = False,
) -> bool:
    identity = (str(ib_hash or "").lower(), int(match_index_count), int(match_first_index))
    existed_under_parent = any(_resolve_export_region_collection_identity(child) == identity for child in parent_collection.children)
    _ensure_export_region_collection(
        parent_collection,
        ib_hash,
        match_index_count,
        match_first_index,
        bone_capture_available=bone_capture_available,
        lod_match_excluded=lod_match_excluded,
    )
    return not existed_under_parent


def _resolve_export_region_collection_identity(collection) -> tuple[str, int, int] | None:
    match = _EXPORT_REGION_NAME_RE.search(str(getattr(collection, "name", "") or ""))
    if not match:
        return None
    return match.group("hash").lower(), int(match.group("count")), int(match.group("first"))


def _link_object_to_collection(mesh_obj, collection) -> None:
    if any(obj.name == mesh_obj.name for obj in collection.objects):
        return
    collection.objects.link(mesh_obj)


def _parse_export_part_index(collection_name: str) -> int | None:
    match = re.match(r"^part(?P<index>\d+)(?:\D.*)?$", str(collection_name or ""), re.IGNORECASE)
    if not match:
        return None
    return int(match.group("index"))





def _replace_lod_mapping_items(scene, mapping_entries: list[dict]) -> None:
    scene.bmc_lod_mapping_items.clear()
    for mapping_entry in mapping_entries:
        item = scene.bmc_lod_mapping_items.add()
        item.lod_profile_id = str(mapping_entry.get("lod_profile_id", "") or "")
        item.lod_level = max(1, int(mapping_entry.get("lod_level", 1) or 1))
        item.canonical_global_bone = int(mapping_entry.get("canonical_global_bone", 0))
        item.mapped_lod_global_bone = int_or_default(
            mapping_entry.get("mapped_lod_global_bone", mapping_entry.get("lod_local_bone")),
            -1,
        )
        item.lod_record_key = str(mapping_entry.get("lod_record_key", "") or "")
        item.lod_local_bone = int_or_default(mapping_entry.get("lod_local_bone"), -1)
        item.votes = int(mapping_entry.get("votes", 0) or 0)
        item.average_distance = float(mapping_entry.get("average_distance", 0.0) or 0.0)
        item.status = str(mapping_entry.get("status", ""))
        item.score = float(mapping_entry.get("score", 0.0))
        item.note = str(mapping_entry.get("note", "") or _lod_mapping_note(mapping_entry))
    scene.bmc_lod_mapping_index = min(scene.bmc_lod_mapping_index, max(0, len(scene.bmc_lod_mapping_items) - 1))


def _replace_lod_fallback_items(scene, fallback_entries: list[dict]) -> None:
    scene.bmc_lod_fallback_items.clear()
    for fallback in fallback_entries:
        item = scene.bmc_lod_fallback_items.add()
        item.enabled = bool(fallback.get("enabled", True))
        item.lod_profile_id = str(fallback.get("lod_profile_id", "") or "")
        item.lod_level = max(1, int(fallback.get("lod_level", 1) or 1))
        item.canonical_global_bone = int(fallback.get("canonical_global_bone", 0) or 0)
        item.donor_global_bone = int_or_default(fallback.get("donor_global_bone"), -1)
        item.lod_record_key = str(fallback.get("lod_record_key", "") or "")
        item.lod_local_bone = int_or_default(fallback.get("lod_local_bone"), -1)
        item.method = str(fallback.get("method", "") or fallback.get("fallback_method", "") or "")
        item.confidence = float(fallback.get("confidence", fallback.get("fallback_confidence", 0.0)) or 0.0)
        item.status = str(fallback.get("status", "") or "")
        item.note = str(fallback.get("note", "") or "")
    scene.bmc_lod_fallback_index = min(scene.bmc_lod_fallback_index, max(0, len(scene.bmc_lod_fallback_items) - 1))


def _replace_lod_profile_items(scene, manifest: dict) -> None:
    previous_index = int(getattr(scene, "bmc_lod_profile_index", 0) or 0)
    profiles = ensure_lod_profiles(manifest)
    scene.bmc_lod_profiles.clear()
    current_generation = str(manifest.get("global_pool_generation", "") or "")
    for profile in profiles:
        item = scene.bmc_lod_profiles.add()
        item.enabled = bool(profile.get("enabled", True))
        item.profile_id = str(profile.get("profile_id", "") or "")
        item.label = str(profile.get("label", "") or f"LOD {int(profile.get('lod_level', 1) or 1)}")
        item.lod_level = max(1, int(profile.get("lod_level", 1) or 1))
        item.frameanalysis_dir = str(profile.get("frameanalysis_dir", "") or "")
        item.global_pool_generation = str(profile.get("global_pool_generation", "") or "")
        result = dict(profile.get("result", {}) or {})
        stale = lod_profile_is_stale(profile, current_generation)
        review = dict(result.get("lod_review", {}) or {})
        item.status = "stale" if stale else ("ok" if bool(review.get("runtime_safe", False)) else ("blocked" if result else "not_analyzed"))
        frame_records = list(result.get("lod_frameanalysis", []) or [])
        frame_record = dict(frame_records[0] or {}) if frame_records else {}
        matched = int(frame_record.get("matched_global_bone_count", 0) or 0)
        required = int(frame_record.get("required_global_bone_count", frame_record.get("total_global_bone_count", 0)) or 0)
        missing = int(review.get("missing_global_bone_count", 0) or 0)
        item.summary = f"matched {matched}/{required}, missing {missing}" if result else ""
        item.warning = str(profile.get("stale_reason", "") or "") if stale else first_lod_profile_issue(result)
    scene.bmc_lod_profile_index = min(previous_index, max(0, len(scene.bmc_lod_profiles) - 1))


def _lod_profile_settings_from_scene(scene) -> list[dict]:
    return [
        {
            "profile_id": str(item.profile_id or ""),
            "label": str(item.label or ""),
            "lod_level": max(1, int(item.lod_level)),
            "frameanalysis_dir": bpy.path.abspath(str(item.frameanalysis_dir or "")),
            "enabled": bool(item.enabled),
        }
        for item in scene.bmc_lod_profiles
    ]


def _sync_lod_profiles_from_scene(scene, manifest: dict) -> None:
    sync_lod_profile_settings(manifest, _lod_profile_settings_from_scene(scene))


def _active_lod_profile_item(scene):
    if not scene.bmc_lod_profiles:
        return None
    index = min(max(0, int(scene.bmc_lod_profile_index)), len(scene.bmc_lod_profiles) - 1)
    return scene.bmc_lod_profiles[index]


def _store_lod_fallback_preview_on_scene(scene, preview: dict, *, applied: bool = False) -> None:
    summary = dict(preview.get("summary", {}) or {})
    fallback_count = int(summary.get("fallback_count", 0) or 0)
    unresolved_count = int(summary.get("unresolved_count", 0) or 0)
    unmatched_used_count = int(summary.get("unmatched_used_count", 0) or 0)
    unused_unmatched_count = int(summary.get("unused_unmatched_count", 0) or 0)
    prefix = "Remaining" if applied and unmatched_used_count else ("Applied" if applied else "Preview")
    scene.bmc_lod_fallback_summary = (
        f"{prefix}: used unmatched {unmatched_used_count}, fallback {fallback_count}, "
        f"unresolved {unresolved_count}, unused unmatched {unused_unmatched_count}"
    )
    scene.bmc_lod_fallback_warning = ""
    if unresolved_count or (applied and unmatched_used_count):
        scene.bmc_lod_fallback_warning = "Some used unmatched bones remain unresolved or unaccepted; export remains blocked for LOD."
    elif fallback_count and not applied:
        scene.bmc_lod_fallback_warning = "Fallbacks are inherited donor bones; inspect before applying."
    _replace_lod_fallback_items(scene, list(preview.get("fallbacks", []) or []) + list(preview.get("unresolved", []) or []))


def _clear_lod_fallback_preview(scene) -> None:
    scene.bmc_lod_fallback_items.clear()
    scene.bmc_lod_fallback_index = 0
    scene.bmc_lod_fallback_summary = ""
    scene.bmc_lod_fallback_warning = ""


def _lod_export_blocking_preview(scene, manifest: dict) -> dict:
    export_collection = scene.bmc_export_collection
    preview = preview_lod_fallbacks_for_export(export_collection, manifest, use_export_plan=True)
    _store_lod_fallback_preview_on_scene(scene, preview, applied=False)
    return preview


def _raise_if_lod_unmatched_used_by_export(scene, manifest: dict) -> None:
    if not manifest.get("lod_mapping"):
        return
    preview = _lod_export_blocking_preview(scene, manifest)
    unmatched_used = list(preview.get("unmatched_used_global_bones", []) or [])
    if not unmatched_used:
        return
    shown = ", ".join(f"G{int(value)}" for value in unmatched_used[:12])
    if len(unmatched_used) > 12:
        shown += ", ..."
    raise ValueError(
        f"LOD has {len(unmatched_used)} unmatched global bone(s) used by the export palette: {shown}. "
        "Open LOD Repair, preview carefully, then apply fallbacks if the inherited result is acceptable."
    )


def _lod_mapping_note(mapping_entry: dict) -> str:
    status = str(mapping_entry.get("status", ""))
    if status == "matched":
        return (
            f"{mapping_entry.get('lod_record_key', '')}: local {int(mapping_entry.get('lod_local_bone', -1))}, "
            f"votes {int(mapping_entry.get('votes', 0) or 0)}, score {float(mapping_entry.get('score', 0.0) or 0.0):.3f}"
        )
    if status == "ignored_lod_match_excluded":
        return "ignored: dynamic/pre-skinned vb0 source"
    return "unmatched"






















def _read_manifest_payload(manifest_path: str) -> dict:
    normalized_manifest_path = bpy.path.abspath(str(manifest_path or ""))
    if not normalized_manifest_path or not os.path.exists(normalized_manifest_path):
        return {}
    with open(normalized_manifest_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _parse_hash_list(raw_value: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for token in str(raw_value or "").replace(",", " ").replace(";", " ").split():
        normalized = token.strip().lower()
        if not normalized:
            continue
        if len(normalized) != 8 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError(f"Invalid IB hash: {token}")
        if normalized in seen:
            continue
        values.append(normalized)
        seen.add(normalized)
    return values




def _replace_candidate_items_from_manifest(scene, manifest: dict) -> None:
    scene.bmc_candidate_items.clear()
    for candidate in manifest.get("candidate_ibs", []):
        item = scene.bmc_candidate_items.add()
        item.enabled = bool(candidate.get("enabled", True))
        item.display_name = str(candidate.get("display_name", "") or "")
        item.ib_hash = str(candidate.get("ib_hash", "") or "")
        item.match_first_index = int(candidate.get("match_first_index", 0))
        item.match_index_count = int(candidate.get("match_index_count", 0))
        item.import_draw_index = int(candidate.get("import_draw_index", -1))
        item.local_bone_count = int(candidate.get("local_bone_count", 0))
        item.draw_count = len(candidate.get("draw_indices", []) or [])
        item.shadow_draw_count = len(candidate.get("shadow_draw_indices", []) or [])
        item.shadow_capture_ready = bool(candidate.get("shadow_capture_ready", bool(candidate.get("shadow_draw_indices"))))
        item.lod_match_excluded = bool(candidate.get("lod_match_excluded", False))
        item.lod_match_excluded_reason = str(candidate.get("lod_match_excluded_reason", "") or "")
        item.replacement_supported = bool(candidate.get("replacement_supported", True))
        item.skinning_mode = str(dict(candidate.get("skinning_mode", {}) or {}).get("kind", "") or "shader_skinned")
        item.status = str(candidate.get("status", "") or ("capture_ready" if item.shadow_capture_ready else "import_only_no_early_shadow"))
        item.manual = False
    scene.bmc_candidate_index = min(scene.bmc_candidate_index, max(0, len(scene.bmc_candidate_items) - 1))


def _candidate_display_name_from_values(ib_hash: str, match_index_count: int, match_first_index: int) -> str:
    return f"{str(ib_hash or '').lower()}-{int(match_index_count)}-{int(match_first_index)}"


def _candidate_key_from_item(item) -> tuple[str, int, int]:
    return (
        str(getattr(item, "ib_hash", "") or "").strip().lower(),
        int(getattr(item, "match_first_index", 0) or 0),
        int(getattr(item, "match_index_count", 0) or 0),
    )


def _candidate_key_from_payload(candidate: dict) -> tuple[str, int, int]:
    return (
        str(candidate.get("ib_hash", "") or "").strip().lower(),
        int(candidate.get("match_first_index", 0) or 0),
        int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0),
    )


def _candidate_item_exists(scene, candidate: dict) -> bool:
    candidate_key = _candidate_key_from_payload(candidate)
    candidate_name = str(candidate.get("display_name", "") or "")
    for item in scene.bmc_candidate_items:
        if candidate_name and str(item.display_name) == candidate_name:
            return True
        if _candidate_key_from_item(item) == candidate_key:
            return True
    return False


def _apply_candidate_payload_to_item(item, candidate: dict, *, manual: bool = False) -> None:
    ib_hash, first_index, index_count = _candidate_key_from_payload(candidate)
    item.enabled = bool(candidate.get("enabled", True))
    item.ib_hash = ib_hash
    item.match_first_index = int(first_index)
    item.match_index_count = int(index_count)
    item.display_name = str(candidate.get("display_name", "") or _candidate_display_name_from_values(ib_hash, index_count, first_index))
    item.import_draw_index = int(candidate.get("import_draw_index", -1))
    item.local_bone_count = int(candidate.get("local_bone_count", 0))
    item.draw_count = len(candidate.get("draw_indices", []) or [])
    item.shadow_draw_count = len(candidate.get("shadow_draw_indices", []) or [])
    item.shadow_capture_ready = bool(candidate.get("shadow_capture_ready", bool(candidate.get("shadow_draw_indices"))))
    item.lod_match_excluded = bool(candidate.get("lod_match_excluded", False))
    item.lod_match_excluded_reason = str(candidate.get("lod_match_excluded_reason", "") or "")
    item.replacement_supported = bool(candidate.get("replacement_supported", True))
    item.skinning_mode = str(dict(candidate.get("skinning_mode", {}) or {}).get("kind", "") or "shader_skinned")
    item.status = str(candidate.get("status", "") or ("capture_ready" if item.shadow_capture_ready else "manual_or_import_only"))
    item.manual = bool(manual)


def _manifest_candidates_by_hash(manifest: dict, ib_hash: str) -> list[dict]:
    normalized_hash = str(ib_hash or "").strip().lower()
    return [
        candidate
        for candidate in manifest.get("candidate_ibs", []) or []
        if str(candidate.get("ib_hash", "") or "").strip().lower() == normalized_hash
    ]


def _find_manifest_candidate_for_item(manifest: dict, item) -> dict | None:
    item_key = _candidate_key_from_item(item)
    item_name = str(getattr(item, "display_name", "") or "")
    for candidate in manifest.get("candidate_ibs", []) or []:
        if item_name and str(candidate.get("display_name", "") or "") == item_name:
            return dict(candidate)
        if _candidate_key_from_payload(candidate) == item_key:
            return dict(candidate)
    return None


def _candidate_payload_from_item(item, manifest: dict | None = None) -> dict:
    candidate = _find_manifest_candidate_for_item(manifest or {}, item) if manifest else None
    if candidate is None:
        ib_hash, first_index, index_count = _candidate_key_from_item(item)
        candidate = {
            "ib_hash": ib_hash,
            "match_first_index": int(first_index),
            "match_index_count": int(index_count),
            "display_name": str(getattr(item, "display_name", "") or _candidate_display_name_from_values(ib_hash, index_count, first_index)),
            "draw_indices": [],
            "shadow_draw_indices": [],
            "local_bone_count": int(getattr(item, "local_bone_count", 0) or 0),
            "import_paths": {"ib": "", "vb": {}, "layout": ""},
        }
    candidate["enabled"] = bool(getattr(item, "enabled", True))
    candidate["shadow_capture_ready"] = bool(getattr(item, "shadow_capture_ready", False))
    candidate["replacement_supported"] = bool(
        getattr(item, "replacement_supported", candidate.get("replacement_supported", True))
    )
    candidate["skinning_mode"] = {
        **dict(candidate.get("skinning_mode", {}) or {}),
        "kind": str(getattr(item, "skinning_mode", "") or dict(candidate.get("skinning_mode", {}) or {}).get("kind", "shader_skinned")),
    }
    candidate["lod_match_excluded"] = bool(getattr(item, "lod_match_excluded", candidate.get("lod_match_excluded", False)))
    candidate["lod_match_excluded_reason"] = str(
        getattr(item, "lod_match_excluded_reason", candidate.get("lod_match_excluded_reason", "")) or ""
    )
    if candidate["lod_match_excluded"] and not candidate["lod_match_excluded_reason"]:
        candidate["lod_match_excluded_reason"] = "manual_lod_match_excluded"
    candidate["status"] = str(getattr(item, "status", "") or candidate.get("status", ""))
    return candidate


def _candidate_payloads_from_ui(scene, manifest: dict | None = None) -> list[dict]:
    payloads = []
    seen: set[tuple[str, int, int]] = set()
    for item in scene.bmc_candidate_items:
        key = _candidate_key_from_item(item)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        payloads.append(_candidate_payload_from_item(item, manifest))
    return payloads


def _add_candidate_from_payload(scene, candidate: dict, *, manual: bool = False) -> bool:
    if _candidate_item_exists(scene, candidate):
        return False
    item = scene.bmc_candidate_items.add()
    _apply_candidate_payload_to_item(item, candidate, manual=manual)
    scene.bmc_candidate_index = len(scene.bmc_candidate_items) - 1
    return True


def _replace_candidate_items_from_payloads(scene, candidates: list[dict], *, previous_state: dict | None = None) -> None:
    previous_state = previous_state or {}
    previous_index = int(getattr(scene, "bmc_candidate_index", 0) or 0)
    scene.bmc_candidate_items.clear()
    for candidate in candidates:
        item = scene.bmc_candidate_items.add()
        manual = bool(candidate.get("_manual", candidate.get("manual", False)))
        _apply_candidate_payload_to_item(item, candidate, manual=manual)
        key = _candidate_key_from_item(item)
        state = previous_state.get(key)
        if state is not None:
            item.enabled = bool(state.get("enabled", item.enabled))
    scene.bmc_candidate_index = min(previous_index, max(0, len(scene.bmc_candidate_items) - 1))


def _candidate_payloads_from_collection(scene, source_collection, manifest: dict) -> tuple[list[dict], dict]:
    payloads: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    stats = {
        "scanned_meshes": 0,
        "recognized_meshes": 0,
        "duplicate_candidates": 0,
    }
    for mesh_obj in _iter_collection_objects_recursive(source_collection):
        if getattr(mesh_obj, "type", "") != "MESH":
            continue
        stats["scanned_meshes"] += 1
        object_candidates = _manifest_candidates_for_object(manifest, mesh_obj)
        if not object_candidates:
            continue
        stats["recognized_meshes"] += 1
        for candidate in object_candidates:
            key = _candidate_key_from_payload(candidate)
            if not key[0] or key in seen:
                stats["duplicate_candidates"] += 1
                continue
            seen.add(key)
            payloads.append(dict(candidate))
    payloads.sort(
        key=lambda candidate: (
            -int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0),
            str(candidate.get("ib_hash", "") or ""),
            int(candidate.get("match_first_index", 0) or 0),
        )
    )
    return payloads, stats


def _iter_collection_objects_recursive(collection):
    if collection is None:
        return
    for mesh_obj in collection.objects:
        yield mesh_obj
    for child in collection.children:
        yield from _iter_collection_objects_recursive(child)


def _toggle_payloads_from_scene(scene) -> list[dict]:
    payloads: list[dict] = []
    for toggle in getattr(scene, "bmc_toggle_draw_sets", []) or []:
        values = []
        for value_item in getattr(toggle, "values", []) or []:
            values.append(
                {
                    "value": int(getattr(value_item, "value", 0) or 0),
                    "label": str(getattr(value_item, "label", "") or ""),
                    "objects": [
                        {"object_name": str(getattr(object_item, "object_name", "") or "")}
                        for object_item in getattr(value_item, "objects", []) or []
                    ],
                }
            )
        payloads.append(
            {
                "enabled": bool(getattr(toggle, "enabled", True)),
                "toggle_id": str(getattr(toggle, "toggle_id", "") or ""),
                "label": str(getattr(toggle, "label", "") or ""),
                "key": str(getattr(toggle, "key", "") or ""),
                "default_value": int(getattr(toggle, "default_value", 0) or 0),
                "values": values,
            }
        )
    return payloads


def _sync_toggle_draw_sets_to_collection(scene) -> None:
    payloads = _toggle_payloads_from_scene(scene)
    collection = getattr(scene, "bmc_export_collection", None)
    if collection is not None:
        collection[BMC_TOGGLE_DRAW_SETS_PROP] = json.dumps(payloads, ensure_ascii=True, separators=(",", ":"))


def _active_toggle(scene):
    toggles = getattr(scene, "bmc_toggle_draw_sets", None)
    if not toggles:
        return None
    index = int(getattr(scene, "bmc_toggle_draw_set_index", 0) or 0)
    if index < 0 or index >= len(toggles):
        return None
    return toggles[index]


def _active_toggle_value(toggle):
    if toggle is None or not getattr(toggle, "values", None):
        return None
    index = int(getattr(toggle, "active_value_index", 0) or 0)
    if index < 0 or index >= len(toggle.values):
        return None
    return toggle.values[index]


def _next_toggle_id(scene) -> str:
    used = {str(getattr(item, "toggle_id", "") or "").strip() for item in getattr(scene, "bmc_toggle_draw_sets", []) or []}
    index = 1
    while f"toggle_{index}" in used:
        index += 1
    return f"toggle_{index}"


def _next_toggle_value(toggle) -> int:
    used = {int(getattr(item, "value", 0) or 0) for item in getattr(toggle, "values", []) or []}
    value = 0
    while value in used:
        value += 1
    return value


def _add_object_name_to_toggle_value(value_item, object_name: str) -> bool:
    object_name = str(object_name or "")
    if not object_name:
        return False
    for existing in value_item.objects:
        if existing.object_name == object_name:
            return False
    object_item = value_item.objects.add()
    object_item.object_name = object_name
    return True


def _event_to_key_binding(event) -> str:
    event_type = str(getattr(event, "type", "") or "")
    if not event_type or event_type in _MODIFIER_EVENT_TYPES:
        return ""
    key = _KEY_EVENT_ALIASES.get(event_type, event_type)
    tokens: list[str] = []
    if bool(getattr(event, "ctrl", False)):
        tokens.append("ctrl")
    if bool(getattr(event, "shift", False)):
        tokens.append("shift")
    if bool(getattr(event, "alt", False)):
        tokens.append("alt")
    tokens.append(key)
    return normalize_key_binding(" ".join(tokens))


def _object_candidate_identity(mesh_obj) -> tuple[str, int, int] | None:
    ib_hash = str(mesh_obj.get("bmc_source_ib_hash", "") or getattr(mesh_obj, "merge_ib_hash", "") or "").strip().lower()
    if not ib_hash:
        inferred = infer_mesh_identity_from_name(mesh_obj.name)
        if inferred:
            ib_hash = inferred[0].lower()
    if not ib_hash or len(ib_hash) != 8 or any(character not in "0123456789abcdef" for character in ib_hash):
        return None
    first_index = int(mesh_obj.get("bmc_match_first_index", 0) or 0)
    index_count = int(mesh_obj.get("bmc_match_index_count", 0) or getattr(mesh_obj, "merge_match_index_count", 0) or 0)
    return ib_hash, first_index, max(0, index_count)


def _object_source_hash(mesh_obj) -> str:
    identity = _object_candidate_identity(mesh_obj)
    if identity is not None:
        return identity[0]
    inferred = infer_mesh_identity_from_name(mesh_obj.name)
    return inferred[0].lower() if inferred else ""


def _global_pool_generation_id(manifest: dict, bone_pool_order: list[dict] | None = None) -> str:
    existing = str(manifest.get("global_pool_generation", "") or "")
    if existing:
        return existing
    order = bone_pool_order if bone_pool_order is not None else list(manifest.get("bone_pool_order", []) or [])
    signature_parts = [
        (
            str(item.get("ib_hash", "") or "").lower(),
            str(int(item.get("match_index_count", 0) or 0)),
            str(int(item.get("match_first_index", 0) or 0)),
            str(int(item.get("global_bone_base", 0) or 0)),
            str(int(item.get("local_bone_count", 0) or 0)),
            ",".join(str(int(value)) for value in item.get("used_local_bone_indices", []) or []),
            "capture" if bool(item.get("bone_capture_available", item.get("shadow_capture_ready", False))) else "mapping",
            "no_lod" if bool(item.get("lod_match_excluded", False)) else "lod",
        )
        for item in order
    ]
    return "|".join("-".join(part) for part in signature_parts)


def _source_key_from_values(ib_hash: str, match_index_count: int, match_first_index: int = 0) -> str:
    return f"{str(ib_hash or '').lower()}-{int(match_index_count)}-{int(match_first_index)}"


def _object_remaps_from_bone_pool_order(bone_pool_order: list[dict]) -> list[dict]:
    remaps: list[dict] = []
    for record in bone_pool_order:
        ib_hash = str(record.get("ib_hash", "") or "").lower()
        match_index_count = int(record.get("match_index_count", 0) or 0)
        match_first_index = int(record.get("match_first_index", 0) or 0)
        used_local_bone_indices = _used_local_bone_indices_from_pool_record(record)
        local_bone_count = len(used_local_bone_indices)
        global_bone_base = int(record.get("global_bone_base", record.get("capture_store_base", 0)) or 0)
        if not ib_hash or match_index_count <= 0 or local_bone_count <= 0:
            continue
        source_key = _source_key_from_values(ib_hash, match_index_count, match_first_index)
        remaps.append(
            {
                "object_name": "",
                "ib_hash": ib_hash,
                "match_first_index": match_first_index,
                "match_index_count": match_index_count,
                "source_key": source_key,
                "bone_capture_available": bool(record.get("bone_capture_available", record.get("shadow_capture_ready", False))),
                "local_group_to_global_group": {
                    str(local_index): int(global_bone_base + compact_index)
                    for compact_index, local_index in enumerate(used_local_bone_indices)
                },
            }
        )
    return remaps


def _used_local_bone_indices_from_pool_record(record: dict) -> list[int]:
    raw_indices = record.get("used_local_bone_indices")
    if isinstance(raw_indices, (list, tuple)) and raw_indices:
        return sorted({int(value) for value in raw_indices if int(value) >= 0})
    local_bone_count = int(record.get("local_bone_count", 0) or 0)
    return list(range(max(0, local_bone_count)))


def _remap_index_from_manifest(manifest: dict) -> dict[tuple[str, int, int], dict]:
    index: dict[tuple[str, int, int], dict] = {}
    for remap in manifest.get("object_remaps", []) or []:
        ib_hash = str(remap.get("ib_hash", "") or "").lower()
        if not ib_hash:
            continue
        match_first_index = int(remap.get("match_first_index", 0) or 0)
        match_index_count = int(remap.get("match_index_count", 0) or 0)
        index[(ib_hash, match_first_index, match_index_count)] = dict(remap)
    return index


def _remap_entries_for_hash(manifest: dict, ib_hash: str) -> list[dict]:
    normalized_hash = str(ib_hash or "").lower()
    return [
        dict(remap)
        for remap in manifest.get("object_remaps", []) or []
        if str(remap.get("ib_hash", "") or "").lower() == normalized_hash
    ]


def _iter_object_name_hashes(object_name: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"[0-9A-Fa-f]{8}", str(object_name or "")):
        value = match.group(0).lower()
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def _resolve_remap_for_object_name(manifest: dict, object_name: str) -> tuple[dict | None, str]:
    normalized_name = str(object_name or "").lower()
    remaps = [dict(remap) for remap in manifest.get("object_remaps", []) or []]
    for remap in remaps:
        source_key = str(remap.get("source_key", "") or _source_key_from_values(
            str(remap.get("ib_hash", "") or ""),
            int(remap.get("match_index_count", 0) or 0),
            int(remap.get("match_first_index", 0) or 0),
        )).lower()
        if source_key and source_key in normalized_name:
            return remap, ""

    matched_entries: list[dict] = []
    matched_keys: set[str] = set()
    for ib_hash in _iter_object_name_hashes(normalized_name):
        for remap in _remap_entries_for_hash(manifest, ib_hash):
            source_key = str(remap.get("source_key", "") or _source_key_from_values(
                str(remap.get("ib_hash", "") or ""),
                int(remap.get("match_index_count", 0) or 0),
                int(remap.get("match_first_index", 0) or 0),
            ))
            if source_key in matched_keys:
                continue
            matched_entries.append(remap)
            matched_keys.add(source_key)

    if not matched_entries:
        return None, f"{object_name}: no global-pool hash in object name"
    if len(matched_entries) > 1:
        hashes = ", ".join(sorted({str(entry.get("ib_hash", "")).lower() for entry in matched_entries}))
        return None, f"{object_name}: ambiguous global-pool hash {hashes}"
    return matched_entries[0], ""


def _hash_rename_target_meshes(context) -> list[object]:
    selected_meshes = [obj for obj in context.selected_objects if getattr(obj, "type", "") == "MESH"]
    if selected_meshes:
        return selected_meshes
    return [obj for obj in context.scene.objects if getattr(obj, "type", "") == "MESH"]


def _resolve_remap_for_region_identity(
    manifest: dict,
    ib_hash: str,
    match_index_count: int,
    match_first_index: int,
) -> tuple[dict | None, str]:
    matches = [
        remap
        for remap in _remap_entries_for_hash(manifest, ib_hash)
        if int(remap.get("match_index_count", 0) or 0) == int(match_index_count)
        and int(remap.get("match_first_index", 0) or 0) == int(match_first_index)
    ]
    if not matches:
        return None, f"{ib_hash}-{match_index_count}-{match_first_index}: no global-pool mapping"
    source_keys = {
        str(remap.get("source_key", "") or _source_key_from_values(
            str(remap.get("ib_hash", "") or ""),
            int(remap.get("match_index_count", 0) or 0),
            int(remap.get("match_first_index", 0) or 0),
        ))
        for remap in matches
    }
    if len(source_keys) > 1:
        return None, f"{ib_hash}-{match_index_count}-{match_first_index}: ambiguous_source_hash"
    return dict(matches[0]), ""


def _apply_remap_to_meshes(meshes_and_remaps: list[tuple[object, dict]], manifest: dict):
    if not meshes_and_remaps:
        return None
    mesh_objects = [mesh_obj for mesh_obj, _remap in meshes_and_remaps]
    selected_entries = []
    for mesh_obj, remap in meshes_and_remaps:
        remap_entry = dict(remap)
        remap_entry["object_name"] = mesh_obj.name
        selected_entries.append(remap_entry)
    return apply_group_remaps_to_meshes(
        mesh_objects,
        {
            "object_remaps": selected_entries,
            "global_pool_generation": _global_pool_generation_id(manifest),
        },
        identity_resolver=_identity_resolver_from_entries(mesh_objects, selected_entries),
    )


def _collect_mesh_remap_pairs(mesh_objects, remap_resolver) -> tuple[int, list[tuple[object, dict]], list[str]]:
    scanned_objects = 0
    meshes_and_remaps: list[tuple[object, dict]] = []
    skipped: list[str] = []
    for mesh_obj in mesh_objects:
        if getattr(mesh_obj, "type", "") != "MESH":
            continue
        scanned_objects += 1
        remap, error = remap_resolver(mesh_obj)
        if remap is None:
            if error:
                skipped.append(error)
            continue
        meshes_and_remaps.append((mesh_obj, remap))
    return scanned_objects, meshes_and_remaps, skipped


def _apply_global_names_for_meshes(mesh_objects, manifest: dict, remap_resolver) -> tuple[int, int, int, list[str]]:
    scanned_objects, meshes_and_remaps, skipped = _collect_mesh_remap_pairs(mesh_objects, remap_resolver)
    if not meshes_and_remaps:
        return scanned_objects, 0, 0, skipped
    result = _apply_remap_to_meshes(meshes_and_remaps, manifest)
    if result is None:
        return scanned_objects, 0, 0, skipped
    skipped.extend(result.skipped_objects)
    return scanned_objects, int(result.updated_objects), int(result.renamed_groups), skipped


def _revert_global_names_for_meshes(mesh_objects, remap_resolver) -> tuple[int, int, int, list[str]]:
    scanned_objects, meshes_and_remaps, skipped = _collect_mesh_remap_pairs(mesh_objects, remap_resolver)
    updated_objects = 0
    renamed_groups = 0
    for mesh_obj, remap in meshes_and_remaps:
        renamed, rename_error = _rename_global_groups_to_local(mesh_obj, remap)
        if rename_error:
            skipped.append(rename_error)
            continue
        if renamed > 0:
            updated_objects += 1
            renamed_groups += renamed
    return scanned_objects, updated_objects, renamed_groups, skipped


def _candidate_source_meshes_and_remaps(context, manifest: dict) -> list[tuple[object, dict]]:
    remap_index = _remap_index_from_manifest(manifest)
    meshes_and_remaps: list[tuple[object, dict]] = []
    export_collection = context.scene.bmc_export_collection
    if export_collection is None:
        return meshes_and_remaps
    for mesh_obj in _iter_collection_objects_recursive(export_collection):
        if getattr(mesh_obj, "type", "") != "MESH":
            continue
        identity = _object_candidate_identity(mesh_obj)
        if identity is None:
            continue
        remap = remap_index.get(identity)
        if remap is None:
            continue
        meshes_and_remaps.append((mesh_obj, remap))
    return meshes_and_remaps


def _apply_global_names_to_candidate_source_objects(
    context,
    manifest: dict,
    meshes_and_remaps: list[tuple[object, dict]] | None = None,
) -> tuple[int, int, list[str]]:
    if meshes_and_remaps is None:
        meshes_and_remaps = _candidate_source_meshes_and_remaps(context, manifest)
    result = _apply_remap_to_meshes(meshes_and_remaps, manifest)
    if result is None:
        return 0, 0, []
    return int(result.updated_objects), int(result.renamed_groups), list(result.skipped_objects)


def _merge_candidate_source_seam_groups(
    context,
    manifest: dict,
    meshes_and_remaps: list[tuple[object, dict]] | None = None,
):
    if meshes_and_remaps is None:
        meshes_and_remaps = _candidate_source_meshes_and_remaps(context, manifest)
    meshes = [mesh_obj for mesh_obj, _remap in meshes_and_remaps]
    if len(meshes) < 2:
        return None
    return build_and_apply_seam_mapping(meshes)


def _merge_selected_seam_groups(context):
    mesh_objects = [obj for obj in context.selected_objects if getattr(obj, "type", "") == "MESH"]
    if len(mesh_objects) < 2:
        raise ValueError("Select at least two mesh objects")
    return build_and_apply_seam_mapping(mesh_objects)


def _export_region_collections_by_hash(export_collection) -> dict[str, list[object]]:
    by_hash: dict[str, list[object]] = {}
    if export_collection is None:
        return by_hash
    for child in export_collection.children:
        identity = _resolve_export_region_collection_identity(child)
        if identity is None:
            continue
        by_hash.setdefault(identity[0], []).append(child)
    return by_hash


def _child_collection_contains_object(collection, mesh_obj) -> bool:
    return any(obj.name == mesh_obj.name for obj in collection.objects)


def _unlink_object_from_export_sibling_regions(export_collection, mesh_obj, keep_collection) -> None:
    for child in export_collection.children:
        if child == keep_collection:
            continue
        if _resolve_export_region_collection_identity(child) is None:
            continue
        if _child_collection_contains_object(child, mesh_obj):
            child.objects.unlink(mesh_obj)


def _apply_global_names_in_export_collection(context, manifest: dict) -> tuple[int, int, int, list[str]]:
    export_collection = context.scene.bmc_export_collection
    if export_collection is None:
        raise ValueError("Export source collection is not set")
    skipped: list[str] = []
    scanned_objects = 0
    objects_by_name: dict[str, tuple[object, dict, str]] = {}
    conflict_names: set[str] = set()

    for child in export_collection.children:
        identity = _resolve_export_region_collection_identity(child)
        if identity is None:
            continue
        remap, error = _resolve_remap_for_region_identity(manifest, identity[0], identity[1], identity[2])
        if remap is None:
            skipped.append(error)
            continue
        for mesh_obj in _iter_collection_objects_recursive(child):
            if getattr(mesh_obj, "type", "") != "MESH":
                continue
            scanned_objects += 1
            if mesh_obj.name in conflict_names:
                continue
            previous = objects_by_name.get(mesh_obj.name)
            if previous is not None and previous[2] != child.name:
                skipped.append(f"{mesh_obj.name}: multiple export regions")
                objects_by_name.pop(mesh_obj.name, None)
                conflict_names.add(mesh_obj.name)
                continue
            objects_by_name[mesh_obj.name] = (mesh_obj, remap, child.name)

    result = _apply_remap_to_meshes(
        [(mesh_obj, remap) for mesh_obj, remap, _child_name in objects_by_name.values()],
        manifest,
    )
    if result is None:
        return scanned_objects, 0, 0, skipped
    skipped.extend(result.skipped_objects)
    return scanned_objects, int(result.updated_objects), int(result.renamed_groups), skipped


def _apply_global_names_by_object_hash(context, manifest: dict) -> tuple[int, int, int, list[str]]:
    return _apply_global_names_for_meshes(
        _hash_rename_target_meshes(context),
        manifest,
        lambda mesh_obj: _resolve_remap_for_object_name(manifest, mesh_obj.name),
    )


def _revert_global_names_by_object_hash(context, manifest: dict) -> tuple[int, int, int, list[str]]:
    return _revert_global_names_for_meshes(
        _hash_rename_target_meshes(context),
        lambda mesh_obj: _resolve_remap_for_object_name(manifest, mesh_obj.name),
    )


def _rename_global_groups_to_local(mesh_obj, remap: dict) -> tuple[int, str]:
    local_to_global = {
        int(local_index): int(global_index)
        for local_index, global_index in dict(remap.get("local_group_to_global_group", {}) or {}).items()
    }
    if not local_to_global:
        return 0, f"{mesh_obj.name}: no local/global remap"
    global_to_local = {global_index: local_index for local_index, global_index in local_to_global.items()}

    rename_pairs: list[tuple[str, str]] = []
    current_names = {str(vertex_group.name) for vertex_group in mesh_obj.vertex_groups}
    source_names: set[str] = set()
    for vertex_group in mesh_obj.vertex_groups:
        numeric_name = _parse_vertex_group_int(vertex_group.name)
        if numeric_name is None:
            continue
        local_index = global_to_local.get(numeric_name)
        if local_index is None:
            continue
        source_name = str(vertex_group.name)
        target_name = str(int(local_index))
        if source_name == target_name:
            continue
        rename_pairs.append((source_name, target_name))
        source_names.add(source_name)

    if not rename_pairs:
        return 0, ""

    for _source_name, target_name in rename_pairs:
        if target_name in current_names and target_name not in source_names:
            return 0, f"{mesh_obj.name}: target local group {target_name} already exists"

    temp_name_by_source: dict[str, str] = {}
    for source_name, _target_name in rename_pairs:
        vertex_group = mesh_obj.vertex_groups.get(source_name)
        if vertex_group is None:
            continue
        temp_name = f"__bmc_tmp_local__{vertex_group.index}__{source_name}"
        vertex_group.name = temp_name
        temp_name_by_source[source_name] = temp_name

    renamed_count = 0
    for source_name, target_name in rename_pairs:
        temp_name = temp_name_by_source.get(source_name, "")
        if not temp_name:
            continue
        mesh_obj.vertex_groups[temp_name].name = target_name
        renamed_count += 1

    for prop_name in (
        BMC_GLOBAL_REMAP_PROP,
        BMC_GLOBAL_SOURCE_KEY_PROP,
        BMC_GLOBAL_POOL_GENERATION_PROP,
        BMC_VERTEX_GROUP_STATE_PROP,
    ):
        if prop_name in mesh_obj:
            del mesh_obj[prop_name]
    return renamed_count, ""


def _parse_vertex_group_int(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _manifest_candidates_for_object(manifest: dict, mesh_obj) -> list[dict]:
    identity = _object_candidate_identity(mesh_obj)
    if identity is None:
        return []
    ib_hash, first_index, index_count = identity
    hash_matches = _manifest_candidates_by_hash(manifest, ib_hash)
    if index_count > 0:
        exact_matches = [
            candidate
            for candidate in hash_matches
            if int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0) == index_count
            and int(candidate.get("match_first_index", 0) or 0) == first_index
        ]
        if exact_matches:
            return exact_matches
    if hash_matches:
        return hash_matches if index_count <= 0 else [
            candidate
            for candidate in hash_matches
            if int(candidate.get("match_index_count", candidate.get("source_index_count", 0)) or 0) == index_count
        ] or hash_matches
    return [
        {
            "enabled": True,
            "ib_hash": ib_hash,
            "match_first_index": int(first_index),
            "match_index_count": int(index_count),
            "display_name": _candidate_display_name_from_values(ib_hash, index_count, first_index),
            "draw_indices": [],
            "shadow_draw_indices": [],
            "shadow_capture_ready": False,
            "local_bone_count": int(infer_local_bone_count_from_mesh(mesh_obj)),
            "status": "manual_from_collection",
            "import_paths": {"ib": "", "vb": {}, "layout": ""},
        }
    ]


def _update_scene_mapping_payload(scene, manifest_path: str | None = None) -> dict:
    manifest_payload = _read_manifest_payload(manifest_path or scene.bmc_manifest_path)
    base_payload = load_mapping_payload_from_scene(scene)
    payload = build_mapping_payload(scene, manifest_payload=manifest_payload, base_payload=base_payload)
    store_mapping_payload_on_scene(scene, payload)
    return payload


def _texture_payload_from_scene(scene) -> dict:
    from . import properties as addon_properties

    return addon_properties.texture_mark_payload_from_scene(scene)


def _store_texture_payload_on_scene(scene, payload: dict) -> None:
    from . import properties as addon_properties

    addon_properties.store_texture_mark_payload_on_scene(scene, payload)


def _sync_texture_mark_items(scene) -> None:
    from . import properties as addon_properties

    addon_properties.sync_texture_mark_items(scene)


def _refresh_texture_payload_after_analyze(scene) -> None:
    payload = _texture_payload_from_scene(scene)
    _select_default_texture_region(scene, payload)
    _sync_texture_mark_items(scene)


def _select_default_texture_region(scene, payload: dict) -> None:
    candidates = payload.get("candidates", {})
    if not isinstance(candidates, dict) or not candidates:
        return
    current_region = str(getattr(scene, "bmc_texture_region", "") or "")
    region_key = current_region if current_region in candidates else next(iter(sorted(candidates)))
    try:
        scene.bmc_texture_region = region_key
    except TypeError:
        return
    default_draws = payload.get("default_draws", {})
    draw_key = str(default_draws.get(region_key, "") or "") if isinstance(default_draws, dict) else ""
    if draw_key:
        try:
            scene.bmc_texture_draw = draw_key
        except TypeError:
            pass


def _remove_unique_region_texture_semantic(region_marks: dict[str, object], semantic: str, *, keep_draw_key: str, keep_slot: str) -> None:
    if semantic not in {"base_color", "normal"}:
        return
    for draw_key, slot_marks in list(region_marks.items()):
        if not isinstance(slot_marks, dict):
            continue
        for slot, mark in list(slot_marks.items()):
            if str(draw_key) == str(keep_draw_key) and str(slot) == str(keep_slot):
                continue
            if isinstance(mark, dict) and str(mark.get("semantic", "") or "") == semantic:
                slot_marks.pop(slot, None)


def _used_region_semantic_indices(region_marks: dict[str, object], semantic: str, *, skip_draw_key: str, skip_slot: str) -> set[int]:
    used: set[int] = set()
    for draw_key, slot_marks in region_marks.items():
        if not isinstance(slot_marks, dict):
            continue
        for slot, mark in slot_marks.items():
            if str(draw_key) == str(skip_draw_key) and str(slot) == str(skip_slot):
                continue
            if isinstance(mark, dict) and str(mark.get("semantic", "") or "") == semantic:
                used.add(int(mark.get("semantic_index", 0) or 0))
    return used


def _texture_bindings_for_region(payload: dict, region_key: str) -> dict[str, dict[str, object]]:
    bindings: dict[str, dict[str, object]] = {}
    for binding in marked_texture_bindings(payload):
        if str(binding.get("region_key", "") or "") != str(region_key):
            continue
        slot = str(binding.get("slot", "") or "").strip().lower()
        if slot:
            bindings[slot] = dict(binding)
    return dict(sorted(bindings.items(), key=lambda item: slot_sort_key(item[0])))


def _region_key_from_object(mesh_obj) -> str:
    identity = _object_candidate_identity(mesh_obj)
    if identity is None:
        return ""
    ib_hash, first_index, index_count = identity
    if not ib_hash or int(index_count) <= 0:
        return ""
    return f"{ib_hash}-{int(index_count)}-{int(first_index)}"


def _apply_texture_payload_to_object(mesh_obj, payload: dict, *, region_key: str = "") -> bool:
    target_region = str(region_key or _region_key_from_object(mesh_obj) or "")
    if not target_region:
        return False
    bindings = _texture_bindings_for_region(payload, target_region)
    if not bindings:
        return False
    mesh_obj[BMC_TEXTURE_SLOTS_PROP] = dump_texture_mark_payload(bindings)
    return apply_material_from_texture_bindings(mesh_obj, bindings, clear_existing=True)


def _selected_mesh_objects(context) -> list[object]:
    return [obj for obj in context.selected_objects if getattr(obj, "type", "") == "MESH"]


def _apply_texture_marks_to_mesh_scope(context, payload: dict) -> tuple[int, int]:
    objects = _selected_mesh_objects(context)
    if not objects:
        export_collection = context.scene.bmc_export_collection
        if export_collection is not None:
            seen: set[str] = set()
            objects = []
            for mesh_obj in _iter_collection_objects_recursive(export_collection):
                if getattr(mesh_obj, "type", "") != "MESH" or mesh_obj.name in seen:
                    continue
                seen.add(mesh_obj.name)
                objects.append(mesh_obj)

    matched_count = 0
    applied_count = 0
    fallback_region = str(getattr(context.scene, "bmc_texture_region", "") or "")
    if fallback_region == "__none__":
        fallback_region = ""
    for mesh_obj in objects:
        region_key = _region_key_from_object(mesh_obj)
        if not region_key and fallback_region:
            region_key = fallback_region
        if not region_key or not _texture_bindings_for_region(payload, region_key):
            continue
        matched_count += 1
        if _apply_texture_payload_to_object(mesh_obj, payload, region_key=region_key):
            applied_count += 1
    return matched_count, applied_count






















def _identity_resolver_from_entries(mesh_objects: list[object], selected_entries: list[dict]):
    identity_by_name = {
        mesh_obj.name: (str(entry.get("ib_hash", "")).lower(), 0)
        for mesh_obj, entry in zip(mesh_objects, selected_entries)
    }

    def _resolver(mesh_obj):
        identity = identity_by_name.get(mesh_obj.name)
        if identity is not None and identity[0]:
            return identity
        return resolve_mesh_identity(mesh_obj)

    return _resolver








class BMC_OT_toggle_draw_set_add(bpy.types.Operator):
    bl_idname = "object.bmc_toggle_draw_set_add"
    bl_label = "Add Toggle Draw Set"
    bl_description = "Create a key-controlled draw set. Objects assigned to a value draw only when that value is active"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        toggle = scene.bmc_toggle_draw_sets.add()
        toggle_id = _next_toggle_id(scene)
        toggle.enabled = True
        toggle.toggle_id = toggle_id
        toggle.label = toggle_id.replace("_", " ").title()
        toggle.key = ""
        toggle.default_value = 0
        value_item = toggle.values.add()
        value_item.value = 0
        value_item.label = "Value 0"
        toggle.active_value_index = 0
        scene.bmc_toggle_draw_set_index = len(scene.bmc_toggle_draw_sets) - 1
        _sync_toggle_draw_sets_to_collection(scene)
        return {"FINISHED"}


class BMC_OT_toggle_draw_set_remove(bpy.types.Operator):
    bl_idname = "object.bmc_toggle_draw_set_remove"
    bl_label = "Remove Toggle Draw Set"
    bl_description = "Remove the active key-controlled draw set"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(getattr(context.scene, "bmc_toggle_draw_sets", None))

    def execute(self, context):
        scene = context.scene
        index = int(getattr(scene, "bmc_toggle_draw_set_index", 0) or 0)
        if index < 0 or index >= len(scene.bmc_toggle_draw_sets):
            return {"CANCELLED"}
        scene.bmc_toggle_draw_sets.remove(index)
        scene.bmc_toggle_draw_set_index = min(index, max(0, len(scene.bmc_toggle_draw_sets) - 1))
        _sync_toggle_draw_sets_to_collection(scene)
        return {"FINISHED"}


class BMC_OT_toggle_draw_value_add(bpy.types.Operator):
    bl_idname = "object.bmc_toggle_draw_value_add"
    bl_label = "Add Toggle Value"
    bl_description = "Add a value slot to the active toggle draw set"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _active_toggle(context.scene) is not None

    def execute(self, context):
        scene = context.scene
        toggle = _active_toggle(scene)
        if toggle is None:
            return {"CANCELLED"}
        value_number = _next_toggle_value(toggle)
        value_item = toggle.values.add()
        value_item.value = value_number
        value_item.label = f"Value {value_number}"
        toggle.active_value_index = len(toggle.values) - 1
        _sync_toggle_draw_sets_to_collection(scene)
        return {"FINISHED"}


class BMC_OT_toggle_draw_value_remove(bpy.types.Operator):
    bl_idname = "object.bmc_toggle_draw_value_remove"
    bl_label = "Remove Toggle Value"
    bl_description = "Remove a value slot from the active toggle draw set"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(default=-1)
    value_index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        return _active_toggle(context.scene) is not None

    def execute(self, context):
        scene = context.scene
        group_index = int(self.group_index)
        if group_index < 0:
            group_index = int(getattr(scene, "bmc_toggle_draw_set_index", 0) or 0)
        if group_index < 0 or group_index >= len(scene.bmc_toggle_draw_sets):
            return {"CANCELLED"}
        toggle = scene.bmc_toggle_draw_sets[group_index]
        value_index = int(self.value_index)
        if value_index < 0:
            value_index = int(getattr(toggle, "active_value_index", 0) or 0)
        if value_index < 0 or value_index >= len(toggle.values):
            return {"CANCELLED"}
        toggle.values.remove(value_index)
        if not toggle.values:
            value_item = toggle.values.add()
            value_item.value = 0
            value_item.label = "Value 0"
        toggle.active_value_index = min(value_index, max(0, len(toggle.values) - 1))
        valid_values = {int(item.value) for item in toggle.values}
        if int(toggle.default_value) not in valid_values:
            toggle.default_value = int(toggle.values[0].value)
        _sync_toggle_draw_sets_to_collection(scene)
        return {"FINISHED"}


class BMC_OT_toggle_draw_value_add_selected(bpy.types.Operator):
    bl_idname = "object.bmc_toggle_draw_value_add_selected"
    bl_label = "Add Selected To Toggle Value"
    bl_description = "Assign selected mesh objects to this toggle value. Existing toggle assignments for those objects are replaced"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(default=-1)
    value_index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        return bool(_selected_mesh_objects(context))

    def execute(self, context):
        scene = context.scene
        group_index = int(self.group_index)
        value_index = int(self.value_index)
        if group_index < 0 or group_index >= len(scene.bmc_toggle_draw_sets):
            return {"CANCELLED"}
        toggle = scene.bmc_toggle_draw_sets[group_index]
        if value_index < 0 or value_index >= len(toggle.values):
            return {"CANCELLED"}
        value_item = toggle.values[value_index]
        added = 0
        for mesh_obj in _selected_mesh_objects(context):
            if _add_object_name_to_toggle_value(value_item, mesh_obj.name):
                added += 1
        _sync_toggle_draw_sets_to_collection(scene)
        self.report({"INFO"}, f"Assigned {added} selected object(s)")
        return {"FINISHED"}


class BMC_OT_toggle_draw_value_remove_selected(bpy.types.Operator):
    bl_idname = "object.bmc_toggle_draw_value_remove_selected"
    bl_label = "Remove Selected From Toggle Value"
    bl_description = "Remove selected mesh objects from this toggle value"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(default=-1)
    value_index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        return bool(_selected_mesh_objects(context))

    def execute(self, context):
        scene = context.scene
        group_index = int(self.group_index)
        value_index = int(self.value_index)
        if group_index < 0 or group_index >= len(scene.bmc_toggle_draw_sets):
            return {"CANCELLED"}
        toggle = scene.bmc_toggle_draw_sets[group_index]
        if value_index < 0 or value_index >= len(toggle.values):
            return {"CANCELLED"}
        selected_names = {mesh_obj.name for mesh_obj in _selected_mesh_objects(context)}
        value_item = toggle.values[value_index]
        removed = 0
        for index in reversed(range(len(value_item.objects))):
            if value_item.objects[index].object_name in selected_names:
                value_item.objects.remove(index)
                removed += 1
        _sync_toggle_draw_sets_to_collection(scene)
        self.report({"INFO"}, f"Removed {removed} selected object(s)")
        return {"FINISHED"}


class BMC_OT_toggle_draw_value_select_objects(bpy.types.Operator):
    bl_idname = "object.bmc_toggle_draw_value_select_objects"
    bl_label = "Select Toggle Value Objects"
    bl_description = "Select Blender objects assigned to this toggle value"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(default=-1)
    value_index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        scene = context.scene
        group_index = int(self.group_index)
        value_index = int(self.value_index)
        if group_index < 0 or group_index >= len(scene.bmc_toggle_draw_sets):
            return {"CANCELLED"}
        toggle = scene.bmc_toggle_draw_sets[group_index]
        if value_index < 0 or value_index >= len(toggle.values):
            return {"CANCELLED"}
        names = {item.object_name for item in toggle.values[value_index].objects if item.object_name}
        selected_count = 0
        active_obj = None
        for obj in context.scene.objects:
            should_select = obj.name in names
            obj.select_set(should_select)
            if should_select:
                selected_count += 1
                active_obj = active_obj or obj
        if active_obj is not None:
            context.view_layer.objects.active = active_obj
        self.report({"INFO"}, f"Selected {selected_count} object(s)")
        return {"FINISHED"}


class BMC_OT_toggle_draw_set_record_key(bpy.types.Operator):
    bl_idname = "object.bmc_toggle_draw_set_record_key"
    bl_label = "Record Toggle Key"
    bl_description = "Press a key or key combination to assign it to the active toggle draw set"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(default=-1)

    def invoke(self, context, _event):
        if self.group_index < 0 or self.group_index >= len(context.scene.bmc_toggle_draw_sets):
            return {"CANCELLED"}
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, "Press a key for this toggle; Esc cancels")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            return {"CANCELLED"}
        if event.value != "PRESS":
            return {"RUNNING_MODAL"}
        key_binding = _event_to_key_binding(event)
        if not key_binding:
            return {"RUNNING_MODAL"}
        toggle = context.scene.bmc_toggle_draw_sets[int(self.group_index)]
        toggle.key = key_binding
        _sync_toggle_draw_sets_to_collection(context.scene)
        self.report({"INFO"}, f"Recorded key: {key_binding}")
        return {"FINISHED"}


class BMC_OT_create_export_collection(bpy.types.Operator):
    bl_idname = "object.bmc_create_export_collection"
    bl_label = "Create Export Collection"
    bl_description = "Create or select the collection that stores final meshes to prepare for export"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        source_collection = _ensure_export_collection(context)
        message = f"Export root: {source_collection.name}; build global pool to create IB region collections"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_add_selected_export_objects(bpy.types.Operator):
    bl_idname = "object.bmc_add_selected_export_objects"
    bl_label = "Add Selected To Export"
    bl_description = "Add selected meshes to the unique existing export child collection with the same source IB hash"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context):
        added_count = 0
        moved_count = 0
        skipped_messages: list[str] = []
        collection = _ensure_export_collection(context)
        regions_by_hash = _export_region_collections_by_hash(collection)

        for mesh_obj in context.selected_objects:
            if mesh_obj.type != "MESH":
                continue
            ib_hash = _object_source_hash(mesh_obj)
            if not ib_hash:
                skipped_messages.append(f"{mesh_obj.name}: no source hash")
                continue
            matching_collections = regions_by_hash.get(ib_hash, [])
            if not matching_collections:
                skipped_messages.append(f"{mesh_obj.name}: no export child for {ib_hash}")
                continue
            if len(matching_collections) > 1:
                skipped_messages.append(f"{mesh_obj.name}: ambiguous_source_hash {ib_hash}")
                continue
            region_collection = matching_collections[0]
            already_in_region = _child_collection_contains_object(region_collection, mesh_obj)
            _link_object_to_collection(mesh_obj, region_collection)
            _unlink_object_from_export_sibling_regions(collection, mesh_obj, region_collection)
            if already_in_region:
                moved_count += 1
            else:
                added_count += 1

        touched_count = added_count + moved_count
        if touched_count == 0:
            if skipped_messages:
                self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
            else:
                self.report({"WARNING"}, "No selected mesh objects matched an export child collection")
            return {"CANCELLED"}

        message = f"Placed {touched_count} selected object(s) into matching export collection(s)"
        if skipped_messages:
            message += f"; skipped {len(skipped_messages)}"
        self.report({"INFO"}, message)
        if skipped_messages:
            self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_apply_export_collection_global_names(bpy.types.Operator):
    bl_idname = "object.bmc_apply_export_collection_global_names"
    bl_label = "Apply Export Global Names"
    bl_description = "Rename mesh vertex groups under export child collections using the child collection's global-pool mapping"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_export_collection", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            scanned_count, updated_count, renamed_groups, skipped_messages = _apply_global_names_in_export_collection(
                context,
                manifest,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Apply export global names failed: {exc}")
            return {"CANCELLED"}

        if scanned_count <= 0:
            self.report({"WARNING"}, "No mesh objects found under export child collections")
            return {"CANCELLED"}
        message = f"Applied global names to {updated_count}/{scanned_count} export mesh object(s); renamed {renamed_groups} group(s)"
        if skipped_messages:
            message += f"; skipped {len(skipped_messages)}"
        self.report({"INFO"}, message)
        if skipped_messages:
            self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_apply_global_names_by_object_hash(bpy.types.Operator):
    bl_idname = "object.bmc_apply_global_names_by_object_hash"
    bl_label = "Rename To Global Groups"
    bl_description = "Rename selected meshes to global vertex-group indices by matching object-name hashes to the global pool"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_manifest_path", ""))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            scanned_count, updated_count, renamed_groups, skipped_messages = _apply_global_names_by_object_hash(
                context,
                manifest,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Rename To Global failed: {exc}")
            return {"CANCELLED"}
        if scanned_count <= 0:
            self.report({"WARNING"}, "No mesh objects found")
            return {"CANCELLED"}
        message = f"Rename To Global updated {updated_count}/{scanned_count} mesh object(s); renamed {renamed_groups} group(s)"
        if skipped_messages:
            message += f"; skipped {len(skipped_messages)}"
        self.report({"INFO"}, message)
        if skipped_messages:
            self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_revert_global_names_by_object_hash(bpy.types.Operator):
    bl_idname = "object.bmc_revert_global_names_by_object_hash"
    bl_label = "Rename To Local Groups"
    bl_description = "Rename selected meshes from global vertex-group indices back to source-local indices by object-name hash"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_manifest_path", ""))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            scanned_count, updated_count, renamed_groups, skipped_messages = _revert_global_names_by_object_hash(
                context,
                manifest,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Rename To Local failed: {exc}")
            return {"CANCELLED"}
        if scanned_count <= 0:
            self.report({"WARNING"}, "No mesh objects found")
            return {"CANCELLED"}
        message = f"Rename To Local updated {updated_count}/{scanned_count} mesh object(s); renamed {renamed_groups} group(s)"
        if skipped_messages:
            message += f"; skipped {len(skipped_messages)}"
        self.report({"INFO"}, message)
        if skipped_messages:
            self.report({"WARNING"}, " | ".join(skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_merge_selected_seam_groups(bpy.types.Operator):
    bl_idname = "object.bmc_merge_selected_seam_groups"
    bl_label = "Merge Seam Groups"
    bl_description = "Build and apply seam vertex-group rename mapping for selected mesh objects"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        selected_meshes = [obj for obj in context.selected_objects if getattr(obj, "type", "") == "MESH"]
        return len(selected_meshes) >= 2

    def execute(self, context):
        try:
            result = _merge_selected_seam_groups(context)
        except Exception as exc:
            self.report({"ERROR"}, f"Merge Seam Groups failed: {exc}")
            return {"CANCELLED"}

        message = (
            f"Merge Seam Groups matched {len(result.aliases)} mapping(s); "
            f"renamed {result.renamed_groups} group(s) on {result.updated_objects} object(s)"
        )
        self.report({"INFO"}, message)
        if result.skipped_messages:
            self.report({"WARNING"}, " | ".join(result.skipped_messages[:3]))
        return {"FINISHED"}


class BMC_OT_mark_texture_semantic(bpy.types.Operator):
    bl_idname = "object.bmc_mark_texture_semantic"
    bl_label = "Mark Texture"
    bl_description = "Mark the selected texture candidate; export uses hash-style TextureOverride replacement"
    bl_options = {"REGISTER", "UNDO"}

    slot: bpy.props.StringProperty(name="PS Slot", default="")
    semantic: bpy.props.EnumProperty(
        name="Semantic",
        items=(
            ("base_color", "Base Color", "Base color texture"),
            ("normal", "Normal", "Normal map texture"),
            ("material", "Material", "Material/ORM/detail texture"),
            ("effect", "Effect", "Effect/emission/special texture"),
            ("clear", "Clear", "Remove manual mark"),
        ),
        default="base_color",
    )

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", ""))

    def execute(self, context):
        scene = context.scene
        try:
            payload = _texture_payload_from_scene(scene)
            region_key = str(getattr(scene, "bmc_texture_region", "") or "").strip()
            draw_key = str(getattr(scene, "bmc_texture_draw", "") or "").strip()
            slot = str(self.slot or "").strip().lower()
            if not region_key or region_key == "__none__":
                raise ValueError("Select a texture region first")
            if not draw_key or draw_key == "__none__":
                raise ValueError("Select a texture draw first")
            draw_candidates, _draw_marks = texture_candidates_for_draw(payload, region_key, draw_key)
            if slot not in draw_candidates:
                raise ValueError(f"Texture candidate not found: {region_key} draw {draw_key} {slot}")
            candidate_hash = str(draw_candidates[slot].get("hash", "") or "")
            if not validate_texture_hash(candidate_hash):
                raise ValueError(f"{slot}: texture hash is not a valid 8-hex TextureOverride hash: {candidate_hash}")

            marks = payload.setdefault("marks", {})
            if not isinstance(marks, dict):
                marks = {}
                payload["marks"] = marks
            region_marks = marks.setdefault(region_key, {})
            if not isinstance(region_marks, dict):
                region_marks = {}
                marks[region_key] = region_marks
            draw_marks = region_marks.setdefault(draw_key, {})
            if not isinstance(draw_marks, dict):
                draw_marks = {}
                region_marks[draw_key] = draw_marks

            if self.semantic == "clear":
                draw_marks.pop(slot, None)
            else:
                existing = draw_marks.get(slot, {})
                _remove_unique_region_texture_semantic(
                    region_marks,
                    self.semantic,
                    keep_draw_key=draw_key,
                    keep_slot=slot,
                )
                if isinstance(existing, dict) and existing.get("semantic") == self.semantic:
                    semantic_index = int(existing.get("semantic_index", 0) or 0)
                elif self.semantic in {"base_color", "normal"}:
                    semantic_index = 0
                else:
                    used_indices = _used_region_semantic_indices(
                        region_marks,
                        self.semantic,
                        skip_draw_key=draw_key,
                        skip_slot=slot,
                    )
                    semantic_index = 0
                    while semantic_index in used_indices:
                        semantic_index += 1
                draw_marks[slot] = {"semantic": self.semantic, "semantic_index": semantic_index}

            _store_texture_payload_on_scene(scene, payload)
            _sync_texture_mark_items(scene)
            matched_count, applied_count = _apply_texture_marks_to_mesh_scope(context, payload)
        except Exception as exc:
            self.report({"ERROR"}, f"Mark Texture failed: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Texture mark updated: {slot} -> {self.semantic}; applied materials {applied_count}/{matched_count}",
        )
        return {"FINISHED"}


class BMC_OT_apply_texture_marks_to_models(bpy.types.Operator):
    bl_idname = "object.bmc_apply_texture_marks_to_models"
    bl_label = "Apply Texture Marks To Models"
    bl_description = "Create/update Blender materials on selected meshes or the export root from current texture marks"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", ""))

    def execute(self, context):
        scene = context.scene
        try:
            payload = _texture_payload_from_scene(scene)
            if not marked_texture_bindings(payload):
                raise ValueError("No marked textures. Mark at least one texture first.")
            matched_count, applied_count = _apply_texture_marks_to_mesh_scope(context, payload)
            _sync_texture_mark_items(scene)
        except Exception as exc:
            self.report({"ERROR"}, f"Apply Texture Marks failed: {exc}")
            return {"CANCELLED"}

        if matched_count <= 0:
            self.report({"WARNING"}, "No matching mesh objects found for current texture marks")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Applied texture marks to {applied_count}/{matched_count} mesh object(s)")
        return {"FINISHED"}


class BMC_OT_prepare_export_collection(bpy.types.Operator):
    bl_idname = "object.bmc_prepare_export_collection"
    bl_label = "Export"
    bl_description = "Scan the export root, build per-part palettes, and write buffers according to the selected export mode"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene)

    def execute(self, context):
        scene = context.scene
        export_start = time.perf_counter()
        try:
            source_collection = _ensure_export_collection(context)
            export_mode = str(getattr(scene, "bmc_export_mode", "BUFFER_ONLY") or "BUFFER_ONLY")
            generate_ini = export_mode in {"BUFFER_AND_INI", "SIMPLE_OVERRIDE"}
            simple_override = export_mode == "SIMPLE_OVERRIDE"
            manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
            if manifest_path and os.path.exists(manifest_path) and not simple_override:
                manifest = read_json(manifest_path)
                _sync_lod_profiles_from_scene(scene, manifest)
                assert_lod_profiles_exportable(manifest)
                write_json(manifest_path, manifest)
                _raise_if_lod_unmatched_used_by_export(scene, manifest)
            result = prepare_export_collection(
                context=context,
                source_collection=source_collection,
                build_collection=None,
                output_dir=scene.bmc_output_dir,
                internal_manifest_dir=None,
                capture_manifest_path=scene.bmc_manifest_path,
                generate_ini=generate_ini,
                simple_override=simple_override,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Export failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_export_manifest_path = result["manifest_path"]
        if result.get("bonestore_ini_path"):
            scene.bmc_ini_path = result["bonestore_ini_path"]
        try:
            export_manifest = read_json(result["manifest_path"])
            payload = load_mapping_payload_from_scene(scene)
            payload["lod_host_map"] = list(export_manifest.get("lod_host_map", []) or [])
            store_mapping_payload_on_scene(scene, payload)
        except Exception:
            pass
        mode_label = "simple override" if result.get("simple_override") else ("buffers + INI" if generate_ini else "buffers")
        message = f"Exported {mode_label}: {result['objects']} object(s), {result['palettes']} palette(s) to {result['output_dir']}"
        timings = dict(result.get("timings", {}) or {})
        if timings:
            total_seconds = time.perf_counter() - export_start
            message += (
                f"; time {total_seconds:.1f}s"
                f" (plan {float(timings.get('plan', 0.0)):.1f}s,"
                f" buffers {float(timings.get('geometry', 0.0)):.1f}s,"
                f" runtime {float(timings.get('runtime', 0.0)):.1f}s)"
            )
        _print_export_performance_report(result)
        self.report({"INFO"}, message)
        manifest_warnings = list(result.get("warnings", []) or [])
        if manifest_warnings:
            self.report({"WARNING"}, " | ".join(str(item) for item in manifest_warnings[:3]))
        return {"FINISHED"}


def _print_export_performance_report(result: dict) -> None:
    timings = dict(result.get("timings", {}) or {})
    performance = dict(result.get("performance", {}) or {})
    geometry = dict(performance.get("geometry", {}) or {})
    print("[BMC Export] Performance report")
    print(f"[BMC Export] numpy={numpy_status()}")
    if timings:
        print(
            "[BMC Export] "
            f"total={float(timings.get('total', 0.0)):.3f}s "
            f"setup={float(timings.get('setup', 0.0)):.3f}s "
            f"plan={float(timings.get('plan', 0.0)):.3f}s "
            f"capture_manifest={float(timings.get('capture_manifest', 0.0)):.3f}s "
            f"palettes={float(timings.get('palettes', 0.0)):.3f}s "
            f"geometry={float(timings.get('geometry', 0.0)):.3f}s "
            f"manifest={float(timings.get('manifest', 0.0)):.3f}s "
            f"runtime={float(timings.get('runtime', 0.0)):.3f}s"
        )
    if geometry:
        print(
            "[BMC Export] "
            f"geometry parts={int(geometry.get('part_count', 0) or 0)} "
            f"loops={int(geometry.get('loop_vertex_count', 0) or 0)} "
            f"indices={int(geometry.get('index_count', 0) or 0)} "
            f"vb_slots={int(geometry.get('vb_slot_count', 0) or 0)}"
        )
        slot_totals = dict(geometry.get("slot_totals", {}) or {})
        if slot_totals:
            formatted_slots = " ".join(
                f"{slot_name}={float(seconds or 0.0):.3f}s"
                for slot_name, seconds in list(slot_totals.items())[:8]
            )
            print(f"[BMC Export] slot totals: {formatted_slots}")
    slowest_parts = list(geometry.get("slowest_parts", []) or [])[:5]
    if slowest_parts:
        print("[BMC Export] Slowest geometry parts:")
        for index, part in enumerate(slowest_parts, start=1):
            slot_timings = dict(part.get("slot_timings", {}) or {})
            formatted_slots = " ".join(
                f"{slot_name}={float(seconds or 0.0):.3f}s"
                for slot_name, seconds in sorted(slot_timings.items(), key=lambda item: float(item[1] or 0.0), reverse=True)
            )
            print(
                "[BMC Export] "
                f"  {index}. {part.get('part', '')} "
                f"total={float(part.get('total_seconds', 0.0) or 0.0):.3f}s "
                f"write_vb={float(part.get('write_vb_seconds', 0.0) or 0.0):.3f}s "
                f"collect_loops={float(part.get('collect_loops_seconds', 0.0) or 0.0):.3f}s "
                f"loops={int(part.get('loop_vertex_count', 0) or 0)} "
                f"slots=[{formatted_slots}] "
                f"objects={', '.join(str(name) for name in list(part.get('objects', []) or [])[:4])}"
            )


class BMC_OT_candidate_add_hash(bpy.types.Operator):
    bl_idname = "object.bmc_candidate_add_hash"
    bl_label = "Add Candidate IB"
    bl_description = "Add the typed IB hash to the Candidate IB list, restoring matching analyzed slices when available"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene)

    def execute(self, context):
        scene = context.scene
        try:
            hashes = _parse_hash_list(str(getattr(scene, "bmc_candidate_add_hash", "") or ""))
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not hashes:
            self.report({"ERROR"}, "Type an 8-hex IB hash before pressing Add")
            return {"CANCELLED"}

        manifest = _read_manifest_payload(scene.bmc_manifest_path)
        added_count = 0
        for ib_hash in hashes:
            matches = _manifest_candidates_by_hash(manifest, ib_hash)
            if matches:
                for candidate in matches:
                    if _add_candidate_from_payload(scene, candidate, manual=False):
                        added_count += 1
                continue
            manual_candidate = {
                "enabled": True,
                "ib_hash": ib_hash,
                "match_first_index": 0,
                "match_index_count": 0,
                "display_name": _candidate_display_name_from_values(ib_hash, 0, 0),
                "draw_indices": [],
                "shadow_draw_indices": [],
                "shadow_capture_ready": False,
                "local_bone_count": 0,
                "status": "manual_hash_missing_analysis",
                "import_paths": {"ib": "", "vb": {}, "layout": ""},
            }
            if _add_candidate_from_payload(scene, manual_candidate, manual=True):
                added_count += 1

        if added_count:
            self.report({"INFO"}, f"Added {added_count} candidate IB entr{'y' if added_count == 1 else 'ies'}")
            return {"FINISHED"}
        self.report({"INFO"}, "Candidate IB already exists in the list")
        return {"FINISHED"}


class BMC_OT_candidate_remove(bpy.types.Operator):
    bl_idname = "object.bmc_candidate_remove"
    bl_label = "Remove Candidate IB"
    bl_description = "Remove the selected Candidate IB list entry"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_candidate_items", None))

    def execute(self, context):
        scene = context.scene
        index = int(scene.bmc_candidate_index)
        if index < 0 or index >= len(scene.bmc_candidate_items):
            self.report({"ERROR"}, "No candidate IB is selected")
            return {"CANCELLED"}
        removed_name = str(scene.bmc_candidate_items[index].display_name or scene.bmc_candidate_items[index].ib_hash)
        scene.bmc_candidate_items.remove(index)
        scene.bmc_candidate_index = min(index, max(0, len(scene.bmc_candidate_items) - 1))
        self.report({"INFO"}, f"Removed candidate {removed_name}")
        return {"FINISHED"}


class BMC_OT_candidate_refresh_from_collection(bpy.types.Operator):
    bl_idname = "object.bmc_candidate_refresh_from_collection"
    bl_label = "Refresh Candidates From Collection"
    bl_description = "Replace the Candidate IB list with IB identities from objects under the export/root collection"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene)

    def execute(self, context):
        scene = context.scene
        manifest = _read_manifest_payload(scene.bmc_manifest_path)
        source_collection = scene.bmc_export_collection or context.collection
        if source_collection is None:
            self.report({"ERROR"}, "No collection is available for candidate refresh")
            return {"CANCELLED"}
        previous_state = {
            _candidate_key_from_item(item): {"enabled": bool(getattr(item, "enabled", True))}
            for item in scene.bmc_candidate_items
        }
        previous_keys = set(previous_state)
        candidates, stats = _candidate_payloads_from_collection(scene, source_collection, manifest)
        next_keys = {_candidate_key_from_payload(candidate) for candidate in candidates}
        _replace_candidate_items_from_payloads(scene, candidates, previous_state=previous_state)
        synced_count = len(scene.bmc_candidate_items)
        added_count = len(next_keys - previous_keys)
        removed_count = len(previous_keys - next_keys)
        self.report(
            {"INFO"},
            (
                f"Synced {synced_count} candidate IB(s) from {stats['recognized_meshes']}/"
                f"{stats['scanned_meshes']} mesh object(s); added {added_count}; removed {removed_count}"
            ),
        )
        return {"FINISHED"}


class BMC_OT_build_global_bone_pool(bpy.types.Operator):
    bl_idname = "object.bmc_build_global_bone_pool"
    bl_label = "Build Global Bone Pool"
    bl_description = "Build a compact global bone pool from enabled Candidate IBs; capture-ready entries are ordered first"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_candidate_items", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; run Analyze Main first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            previous_generation = str(manifest.get("global_pool_generation", "") or "")
            candidates = _candidate_payloads_from_ui(scene, manifest)
            bone_pool_order = build_bone_pool_order(candidates)
            if not bone_pool_order:
                self.report({"ERROR"}, "No enabled candidate has usable bone weights for the global pool")
                return {"CANCELLED"}
            object_remaps = _object_remaps_from_bone_pool_order(bone_pool_order)
            generation_id = _global_pool_generation_id({"bone_pool_order": bone_pool_order}, bone_pool_order)
            manifest["candidate_ibs"] = candidates
            manifest["bone_pool_order"] = bone_pool_order
            manifest["object_remaps"] = object_remaps
            manifest["global_pool_generation"] = generation_id
            invalidated_lod_count = 0
            if previous_generation != generation_id:
                invalidated_lod_count = invalidate_lod_profiles(manifest, "global_bone_pool_changed")
            manifest.setdefault("buffer_tables", {})["global_bone_count"] = sum(
                int(item.get("local_bone_count", 0)) for item in bone_pool_order
            )
            manifest.setdefault("validation", []).append(
                {
                    "severity": "info",
                    "code": "global_pool_built_from_candidate_list",
                    "message": f"Built compact global pool from {len(bone_pool_order)} enabled candidate(s).",
                    "draw_indices": [],
                }
            )
            write_json(manifest_path, manifest)
            export_collection = _ensure_export_collection(context)
            created_count = 0
            unavailable_count = 0
            lod_excluded_count = 0
            for record in bone_pool_order:
                capture_available = bool(record.get("bone_capture_available", record.get("shadow_capture_ready", False)))
                lod_match_excluded = bool(record.get("lod_match_excluded", False))
                if not capture_available:
                    unavailable_count += 1
                if lod_match_excluded:
                    lod_excluded_count += 1
                if _ensure_export_region_collection_if_missing(
                    export_collection,
                    str(record.get("ib_hash", "") or ""),
                    int(record.get("match_index_count", 0) or 0),
                    int(record.get("match_first_index", 0) or 0),
                    bone_capture_available=capture_available,
                    lod_match_excluded=lod_match_excluded,
                ):
                    created_count += 1
            _update_scene_mapping_payload(scene, manifest_path)
            _replace_lod_profile_items(scene, manifest)
            _replace_lod_mapping_items(scene, list(manifest.get("lod_mapping", []) or []))
            if invalidated_lod_count > 0:
                _clear_lod_fallback_preview(scene)
        except Exception as exc:
            self.report({"ERROR"}, f"Build Global Bone Pool failed: {exc}")
            return {"CANCELLED"}

        skipped_count = len([item for item in candidates if item.get("enabled", True)]) - len(bone_pool_order)
        message = f"Built global pool with {len(bone_pool_order)} source IB(s)"
        message += f"; collections {created_count}"
        message += f"; compact bones {sum(int(item.get('local_bone_count', 0)) for item in bone_pool_order)}"
        if unavailable_count > 0:
            message += f"; {unavailable_count} mapping-only IB(s)"
        if lod_excluded_count > 0:
            message += f"; {lod_excluded_count} no-LOD IB(s)"
        if skipped_count > 0:
            message += f"; skipped {skipped_count} invalid IB(s)"
        if invalidated_lod_count > 0:
            message += f"; {invalidated_lod_count} LOD profile(s) need re-analysis"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_apply_global_bone_pool(bpy.types.Operator):
    bl_idname = "object.bmc_apply_global_bone_pool"
    bl_label = "Apply Global Bone Pool"
    bl_description = "Apply the built global bone pool to candidate source meshes by renaming groups and merging seam groups"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", ""))

    def execute(self, context):
        apply_start = time.perf_counter()
        timings: dict[str, float] = {}
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        seam_result = None
        seam_warnings: list[str] = []
        seam_skipped_reason = ""
        meshes_and_remaps = []
        try:
            stage_start = time.perf_counter()
            manifest = read_json(manifest_path)
            timings["manifest"] = time.perf_counter() - stage_start
            if not manifest.get("bone_pool_order") or not manifest.get("object_remaps"):
                self.report({"ERROR"}, "Global bone pool is empty; run Build Global Bone Pool first")
                return {"CANCELLED"}
            stage_start = time.perf_counter()
            _update_scene_mapping_payload(scene, manifest_path)
            _replace_lod_profile_items(scene, manifest)
            timings["scene_payload"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            meshes_and_remaps = _candidate_source_meshes_and_remaps(context, manifest)
            timings["collect_meshes"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            updated_objects, renamed_groups, rename_warnings = _apply_global_names_to_candidate_source_objects(
                context,
                manifest,
                meshes_and_remaps,
            )
            timings["rename_groups"] = time.perf_counter() - stage_start
            if renamed_groups > 0:
                try:
                    stage_start = time.perf_counter()
                    seam_result = _merge_candidate_source_seam_groups(context, manifest, meshes_and_remaps)
                    timings["seam_merge"] = time.perf_counter() - stage_start
                except Exception as seam_exc:
                    timings["seam_merge"] = time.perf_counter() - stage_start
                    seam_warnings.append(f"Seam merge skipped: {seam_exc}")
            else:
                seam_skipped_reason = "unchanged groups"
                timings["seam_merge"] = 0.0
            timings["total"] = time.perf_counter() - apply_start
        except Exception as exc:
            self.report({"ERROR"}, f"Apply Global Bone Pool failed: {exc}")
            return {"CANCELLED"}

        message = "Applied global bone pool"
        message += f"; global-renamed {updated_objects} object(s), {renamed_groups} group(s)"
        if seam_result is not None:
            message += f"; seam-merged {seam_result.updated_objects} object(s), {seam_result.renamed_groups} group(s)"
        elif seam_skipped_reason:
            message += f"; seam-skipped {seam_skipped_reason}"
        else:
            message += "; seam-merged 0 object(s), 0 group(s)"
        self.report({"INFO"}, message)
        warnings = list(rename_warnings)
        if seam_result is not None:
            warnings.extend(str(message) for message in seam_result.skipped_messages)
        warnings.extend(seam_warnings)
        if warnings:
            self.report({"WARNING"}, " | ".join(warnings[:3]))
        print("[BMC Apply Pool] Performance report")
        print(f"[BMC Apply Pool] numpy={numpy_status()}")
        print(
            "[BMC Apply Pool] "
            f"total={timings.get('total', 0.0):.3f}s "
            f"manifest={timings.get('manifest', 0.0):.3f}s "
            f"scene_payload={timings.get('scene_payload', 0.0):.3f}s "
            f"collect_meshes={timings.get('collect_meshes', 0.0):.3f}s "
            f"rename_groups={timings.get('rename_groups', 0.0):.3f}s "
            f"seam_merge={timings.get('seam_merge', 0.0):.3f}s"
        )
        print(
            "[BMC Apply Pool] "
            f"objects={len(meshes_and_remaps)} "
            f"renamed_objects={updated_objects} "
            f"renamed_groups={renamed_groups}"
        )
        if seam_result is not None:
            print(
                "[BMC Apply Pool] "
                f"seam aliases={len(seam_result.aliases)} "
                f"tested_pairs={seam_result.matched_pairs} "
                f"skipped_pairs={seam_result.skipped_pairs}"
            )
        return {"FINISHED"}


class BMC_OT_lod_profile_add(bpy.types.Operator):
    bl_idname = "object.bmc_lod_profile_add"
    bl_label = "Add LOD Profile"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        existing_levels = {int(item.lod_level) for item in scene.bmc_lod_profiles}
        lod_level = 1
        while lod_level in existing_levels:
            lod_level += 1
        had_profiles = bool(scene.bmc_lod_profiles)
        used_ids = {str(existing.profile_id or "") for existing in scene.bmc_lod_profiles}
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        try:
            manifest = read_json(manifest_path) if manifest_path and os.path.exists(manifest_path) else None
        except Exception as exc:
            self.report({"ERROR"}, f"Add LOD Profile failed: {exc}")
            return {"CANCELLED"}
        if manifest is not None:
            used_ids.update(
                str(profile.get("profile_id", "") or "")
                for profile in ensure_lod_profiles(manifest)
            )
        profile_id = f"lod{lod_level}"
        suffix = 2
        while profile_id in used_ids:
            profile_id = f"lod{lod_level}_{suffix}"
            suffix += 1
        frameanalysis_dir = ""
        if not had_profiles and str(getattr(scene, "bmc_lod_frameanalysis_dir", "") or ""):
            frameanalysis_dir = str(scene.bmc_lod_frameanalysis_dir)
        if manifest is not None:
            profiles = ensure_lod_profiles(manifest)
            profiles.append(
                {
                    "profile_id": profile_id,
                    "label": f"LOD {lod_level}",
                    "lod_level": lod_level,
                    "frameanalysis_dir": bpy.path.abspath(frameanalysis_dir) if frameanalysis_dir else "",
                    "enabled": True,
                    "stale": False,
                    "global_pool_generation": str(manifest.get("global_pool_generation", "") or ""),
                    "result": {},
                }
            )
            rebuild_lod_aggregate(manifest)
            try:
                write_json(manifest_path, manifest)
            except Exception as exc:
                self.report({"ERROR"}, f"Add LOD Profile failed: {exc}")
                return {"CANCELLED"}
        item = scene.bmc_lod_profiles.add()
        item.enabled = True
        item.profile_id = profile_id
        item.label = f"LOD {lod_level}"
        item.lod_level = lod_level
        item.frameanalysis_dir = frameanalysis_dir
        item.status = "not_analyzed"
        scene.bmc_lod_profile_index = len(scene.bmc_lod_profiles) - 1
        if manifest is not None:
            _update_scene_mapping_payload(scene, manifest_path)
        return {"FINISHED"}


class BMC_OT_lod_profile_remove(bpy.types.Operator):
    bl_idname = "object.bmc_lod_profile_remove"
    bl_label = "Remove LOD Profile"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_lod_profiles", None))

    def execute(self, context):
        scene = context.scene
        item = _active_lod_profile_item(scene)
        if item is None:
            return {"CANCELLED"}
        profile_id = str(item.profile_id or "")
        index = min(max(0, int(scene.bmc_lod_profile_index)), len(scene.bmc_lod_profiles) - 1)
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        manifest = None
        if profile_id and manifest_path and os.path.exists(manifest_path):
            try:
                manifest = read_json(manifest_path)
                remove_lod_profile(manifest, profile_id)
                write_json(manifest_path, manifest)
            except Exception as exc:
                self.report({"ERROR"}, f"Remove LOD Profile failed: {exc}")
                return {"CANCELLED"}
        scene.bmc_lod_profiles.remove(index)
        scene.bmc_lod_profile_index = min(index, max(0, len(scene.bmc_lod_profiles) - 1))
        if manifest is not None:
            _replace_lod_mapping_items(scene, list(manifest.get("lod_mapping", []) or []))
            _clear_lod_fallback_preview(scene)
            _update_scene_mapping_payload(scene, manifest_path)
        return {"FINISHED"}


class BMC_OT_sync_lod_profiles(bpy.types.Operator):
    bl_idname = "object.bmc_sync_lod_profiles"
    bl_label = "Update LOD Profiles"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and getattr(context.scene, "bmc_manifest_path", ""))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            _sync_lod_profiles_from_scene(scene, manifest)
            write_json(manifest_path, manifest)
            _replace_lod_profile_items(scene, manifest)
            _replace_lod_mapping_items(scene, list(manifest.get("lod_mapping", []) or []))
            _clear_lod_fallback_preview(scene)
            _update_scene_mapping_payload(scene, manifest_path)
        except Exception as exc:
            self.report({"ERROR"}, f"Update LOD Profiles failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Updated enabled LOD profiles")
        return {"FINISHED"}


class BMC_OT_analyze_lod_frameanalysis(bpy.types.Operator):
    bl_idname = "object.bmc_analyze_lod_frameanalysis"
    bl_label = "Analyze LOD"
    bl_description = "Analyze the active LOD profile and merge its capture mappings into the manifest"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        profile = _active_lod_profile_item(scene) if scene else None
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and profile and profile.frameanalysis_dir)

    def execute(self, context):
        scene = context.scene
        profile_item = _active_lod_profile_item(scene)
        if profile_item is None:
            self.report({"ERROR"}, "Add an LOD profile first")
            return {"CANCELLED"}
        profile_label = str(profile_item.label or f"LOD {int(profile_item.lod_level)}")
        profile_level = max(1, int(profile_item.lod_level))
        profile_enabled = bool(profile_item.enabled)
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; build the global pool first")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            _sync_lod_profiles_from_scene(scene, manifest)
            frameanalysis_dir = bpy.path.abspath(str(profile_item.frameanalysis_dir or ""))
            profile_id = str(profile_item.profile_id or "") or profile_id_for(
                profile_level,
                frameanalysis_dir,
            )
            profile_item.profile_id = profile_id
            lod_result = analyze_lod_for_manifest(
                manifest,
                frameanalysis_dir,
                lod_level=profile_level,
            )
            profile = upsert_lod_profile(
                manifest,
                profile_id=profile_id,
                label=profile_label,
                lod_level=profile_level,
                frameanalysis_dir=frameanalysis_dir,
                enabled=profile_enabled,
                result=lod_result,
            )
            manifest.setdefault("validation", []).append(
                {
                    "severity": "info",
                    "code": "lod_profile_analysis_built",
                    "message": f"Built {profile_label} with {len(lod_result.get('lod_capture_records', []) or [])} capture record(s).",
                    "draw_indices": [],
                }
            )
            write_json(manifest_path, manifest)
            _update_scene_mapping_payload(scene, manifest_path)
            _replace_lod_mapping_items(scene, list(manifest.get("lod_mapping", []) or []))
            _replace_lod_profile_items(scene, manifest)
            _clear_lod_fallback_preview(scene)
        except Exception as exc:
            self.report({"ERROR"}, f"Analyze LOD failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_lod_manifest_path = manifest_path
        scene.bmc_lod_frameanalysis_dir = frameanalysis_dir
        frame_records = list(lod_result.get("lod_frameanalysis", []) or [])
        frame_record = frame_records[0] if frame_records else {}
        shadow_stage = dict(frame_record.get("shadow_stage", {}) or {})
        scene.bmc_lod_shadow_host_hash = str(shadow_stage.get("host_ib_hash", "") or "")
        scene.bmc_lod_shadow_host_match_index_count = int_or_default(shadow_stage.get("host_match_index_count"), -1)
        scene.bmc_lod_shadow_host_vs_hash = str(shadow_stage.get("transparent_vs_hash", "") or shadow_stage.get("normal_vs_hash", "") or "")
        matched_count = int(frame_record.get("matched_global_bone_count", 0) or 0)
        total_count = int(frame_record.get("total_global_bone_count", 0) or 0)
        required_count = int(frame_record.get("required_global_bone_count", total_count) or 0)
        ignored_count = int(frame_record.get("ignored_lod_global_bone_count", 0) or 0)
        capture_count = len(lod_result.get("lod_capture_records", []) or [])
        review = dict(dict(profile.get("result", {}) or {}).get("lod_review", {}) or {})
        runtime_safe = bool(review.get("runtime_safe", False))
        missing_count = int(review.get("missing_global_bone_count", 0) or 0)
        coverage = (matched_count / required_count * 100.0) if required_count > 0 else 0.0
        scene.bmc_lod_match_summary = (
            f"{profile_label} {'OK' if runtime_safe else 'BLOCKED'}: matched {matched_count}/{required_count} "
            f"({coverage:.1f}%), ignored {ignored_count}, missing {missing_count}, {capture_count} capture records"
        )
        self.report(
            {"INFO"},
            f"Analyzed {profile_label}: matched {matched_count}/{required_count}; ignored {ignored_count}; capture records {capture_count}",
        )
        lod_warnings = list(dict(profile.get("result", {}) or {}).get("validation", []) or [])
        warning_messages = [
            str(item.get("message", ""))
            for item in lod_warnings
            if str(item.get("severity", "")).lower() in {"warning", "error"} and str(item.get("message", ""))
        ]
        scene.bmc_lod_match_warning = warning_messages[0] if warning_messages else ""
        if warning_messages:
            self.report({"WARNING"}, " | ".join(warning_messages[:3]))
        return {"FINISHED"}


class BMC_OT_preview_lod_fallbacks(bpy.types.Operator):
    bl_idname = "object.bmc_preview_lod_fallbacks"
    bl_label = "Preview LOD Fallbacks"
    bl_description = "Preview inherited donor mappings for LOD globals that are unmatched but used by the export meshes"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_export_collection", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            _sync_lod_profiles_from_scene(scene, manifest)
            assert_lod_profiles_exportable(manifest)
            write_json(manifest_path, manifest)
            preview = _lod_export_blocking_preview(scene, manifest)
        except Exception as exc:
            self.report({"ERROR"}, f"Preview LOD fallbacks failed: {exc}")
            return {"CANCELLED"}

        summary = dict(preview.get("summary", {}) or {})
        message = (
            f"Previewed LOD fallbacks: used unmatched {int(summary.get('unmatched_used_count', 0) or 0)}, "
            f"fallback {int(summary.get('fallback_count', 0) or 0)}, "
            f"unresolved {int(summary.get('unresolved_count', 0) or 0)}"
        )
        self.report({"INFO"}, message)
        if scene.bmc_lod_fallback_warning:
            self.report({"WARNING"}, scene.bmc_lod_fallback_warning)
        return {"FINISHED"}


class BMC_OT_apply_lod_fallbacks(bpy.types.Operator):
    bl_idname = "object.bmc_apply_lod_fallbacks"
    bl_label = "Apply LOD Fallbacks"
    bl_description = "Cautiously apply inherited donor mappings for currently used unmatched LOD globals"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_export_collection", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist")
            return {"CANCELLED"}
        enabled_keys = {
            (str(item.lod_profile_id or ""), int(item.canonical_global_bone))
            for item in scene.bmc_lod_fallback_items
            if bool(item.enabled) and str(item.status or "") != "unresolved"
        }
        if not scene.bmc_lod_fallback_items:
            self.report({"WARNING"}, "Preview LOD fallbacks before applying")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            _sync_lod_profiles_from_scene(scene, manifest)
            assert_lod_profiles_exportable(manifest)
            preview = preview_lod_fallbacks_for_export(scene.bmc_export_collection, manifest, use_export_plan=True)
            preview = filter_lod_fallback_preview(preview, enabled_keys)
            fallbacks = list(preview.get("fallbacks", []) or [])
            if not fallbacks:
                _store_lod_fallback_preview_on_scene(scene, preview, applied=False)
                self.report({"WARNING"}, "No LOD fallback donor could be applied")
                return {"CANCELLED"}
            apply_result = apply_lod_fallbacks_to_manifest(manifest, preview)
            for profile in ensure_lod_profiles(manifest):
                result = dict(profile.get("result", {}) or {})
                if not result:
                    continue
                result["lod_review"] = review_lod_global_pool_coverage(
                    manifest,
                    list(result.get("lod_capture_records", []) or []),
                )
                profile["result"] = result
            rebuild_lod_aggregate(manifest)
            write_json(manifest_path, manifest)
            _replace_lod_mapping_items(scene, list(manifest.get("lod_mapping", []) or []))
            _replace_lod_profile_items(scene, manifest)
            post_preview = preview_lod_fallbacks_for_export(
                scene.bmc_export_collection,
                manifest,
                use_export_plan=True,
            )
            _store_lod_fallback_preview_on_scene(scene, post_preview, applied=True)
            _update_scene_mapping_payload(scene, manifest_path)
        except Exception as exc:
            self.report({"ERROR"}, f"Apply LOD fallbacks failed: {exc}")
            return {"CANCELLED"}

        applied_count = int(apply_result.get("applied_count", 0) or 0)
        unresolved_count = int(dict(post_preview.get("summary", {}) or {}).get("unmatched_used_count", 0) or 0)
        message = f"Applied {applied_count} LOD fallback mapping(s)"
        if unresolved_count:
            message += f"; {unresolved_count} unresolved"
        self.report({"INFO"}, message)
        if scene.bmc_lod_fallback_warning:
            self.report({"WARNING"}, scene.bmc_lod_fallback_warning)
        return {"FINISHED"}


class BMC_OT_analyze_main_frameanalysis(bpy.types.Operator):
    bl_idname = "object.bmc_analyze_main_frameanalysis"
    bl_label = "Analyze Main"
    bl_description = "Analyze Main FrameAnalysis and build the redesigned candidate IB manifest"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_frameanalysis_dir", ""))

    def execute(self, context):
        scene = context.scene
        try:
            payload, manifest_path = write_main_analysis_manifest(
                frameanalysis_dir=bpy.path.abspath(scene.bmc_frameanalysis_dir),
                output_dir=bpy.path.abspath(scene.bmc_output_dir) if scene.bmc_output_dir else "",
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Analyze Main failed: {exc}")
            return {"CANCELLED"}

        scene.bmc_manifest_path = manifest_path
        if scene.bmc_lod_profiles:
            payload["lod_profiles"] = [
                {
                    "profile_id": str(item.profile_id or ""),
                    "label": str(item.label or f"LOD {int(item.lod_level)}"),
                    "lod_level": max(1, int(item.lod_level)),
                    "frameanalysis_dir": bpy.path.abspath(str(item.frameanalysis_dir or "")) if item.frameanalysis_dir else "",
                    "enabled": bool(item.enabled),
                    "stale": False,
                    "global_pool_generation": "",
                    "result": {},
                }
                for item in scene.bmc_lod_profiles
            ]
            rebuild_lod_aggregate(payload)
            write_json(manifest_path, payload)
        shadow_stage = payload.get("shadow_stage", {})
        scene.bmc_shadow_host_hash = str(shadow_stage.get("host_ib_hash", "") or "")
        scene.bmc_shadow_host_match_index_count = int_or_default(shadow_stage.get("host_match_index_count"), -1)
        shadow_vs_hashes = list(shadow_stage.get("shadow_vs_hashes", []) or [])
        scene.bmc_shadow_host_vs_hash = shadow_vs_hashes[-1] if shadow_vs_hashes else ""
        _replace_candidate_items_from_manifest(scene, payload)
        _replace_lod_profile_items(scene, payload)
        _replace_lod_mapping_items(scene, [])
        _clear_lod_fallback_preview(scene)
        _refresh_texture_payload_after_analyze(scene)

        warning_count = sum(1 for item in payload.get("validation", []) if item.get("severity") == "warning")
        message = (
            f"Analyzed {len(payload.get('candidate_ibs', []))} candidate IB(s); "
            f"shadow VS {len(shadow_vs_hashes)}; manifest {os.path.basename(manifest_path)}"
        )
        if warning_count:
            message += f"; warnings {warning_count}"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class BMC_OT_import_selected_candidates(bpy.types.Operator):
    bl_idname = "object.bmc_import_selected_candidates"
    bl_label = "Import Selected IBs"
    bl_description = "Import enabled redesigned candidate IBs from the current capture manifest"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return bool(scene and getattr(scene, "bmc_manifest_path", "") and getattr(scene, "bmc_candidate_items", None))

    def execute(self, context):
        scene = context.scene
        manifest_path = bpy.path.abspath(str(scene.bmc_manifest_path or ""))
        if not manifest_path or not os.path.exists(manifest_path):
            self.report({"ERROR"}, "Capture manifest does not exist; run Analyze Main first")
            return {"CANCELLED"}
        selected_names = [
            str(item.display_name)
            for item in scene.bmc_candidate_items
            if item.enabled and str(item.display_name)
        ]
        if not selected_names:
            self.report({"ERROR"}, "No enabled candidate IBs selected")
            return {"CANCELLED"}
        try:
            manifest = read_json(manifest_path)
            target_collection = _ensure_export_collection(context)
            performance: dict = {}
            imported_objects = import_selected_candidates(
                context,
                manifest,
                selected_names,
                target_collection,
                mirror_flip=bool(getattr(scene, "bmc_mirror_flip", True)),
                uv_flip_v=bool(getattr(scene, "bmc_uv_flip_v", True)),
                performance=performance,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Import Selected IBs failed: {exc}")
            return {"CANCELLED"}
        if not imported_objects:
            self.report({"ERROR"}, "No candidate IBs matched the enabled UI list")
            return {"CANCELLED"}
        warning_count = sum(1 for obj in imported_objects if obj.get("bmc_import_warnings"))
        message = f"Imported {len(imported_objects)} candidate IB object(s) into {target_collection.name}"
        if warning_count:
            message += f"; {warning_count} with warnings"
        _print_import_performance_report(performance)
        self.report({"INFO"}, message)
        return {"FINISHED"}


def _print_import_performance_report(performance: dict) -> None:
    if not performance:
        return
    print("[BMC Import] Performance report")
    print(f"[BMC Import] numpy={numpy_status()}")
    print(
        "[BMC Import] "
        f"total={float(performance.get('total', 0.0) or 0.0):.3f}s "
        f"objects={len(list(performance.get('objects', []) or []))}"
    )
    objects = list(performance.get("objects", []) or [])
    if not objects:
        return
    slowest = sorted(objects, key=lambda item: float(item.get("total", 0.0) or 0.0), reverse=True)[:5]
    print("[BMC Import] Slowest objects:")
    for index, item in enumerate(slowest, start=1):
        stages = dict(item.get("create_stages", {}) or {})
        load_stages = dict(item.get("load_stages", {}) or {})
        formatted_stages = " ".join(
            f"{name}={float(seconds or 0.0):.3f}s"
            for name, seconds in sorted(stages.items(), key=lambda pair: float(pair[1] or 0.0), reverse=True)
            if name != "total"
        )
        formatted_load_stages = " ".join(
            f"{name}={float(seconds or 0.0):.3f}s"
            for name, seconds in sorted(load_stages.items(), key=lambda pair: float(pair[1] or 0.0), reverse=True)
            if name != "total"
        )
        print(
            "[BMC Import] "
            f"  {index}. {item.get('name', '')} "
            f"total={float(item.get('total', 0.0) or 0.0):.3f}s "
            f"load={float(item.get('load', 0.0) or 0.0):.3f}s "
            f"create={float(item.get('create', 0.0) or 0.0):.3f}s "
            f"vertices={int(item.get('vertices', 0) or 0)} "
            f"triangles={int(item.get('triangles', 0) or 0)} "
            f"load_stages=[{formatted_load_stages}] "
            f"stages=[{formatted_stages}]"
        )


