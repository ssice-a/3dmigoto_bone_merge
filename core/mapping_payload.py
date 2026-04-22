"""Helpers for frozen mapping payloads used by scan, presets, and apply mapping."""

from __future__ import annotations

import json
import os

import bpy


def serialize_target_items(scene) -> list[dict]:
    return [
        {
            "object_name": (item.object_ref.name if getattr(item, "object_ref", None) else item.object_name),
            "ib_hash": str(item.ib_hash or "").lower(),
            "match_index_count": int(item.match_index_count),
            "local_bone_count": int(getattr(item, "local_bone_count", 0)),
            "autodetected": bool(item.autodetected),
            "enabled": bool(item.enabled),
        }
        for item in scene.bmc_target_items
    ]


def serialize_alias_items(scene) -> list[dict]:
    return [
        {
            "enabled": bool(item.enabled),
            "src_draw_index": int(item.src_draw_index),
            "src_object_name": str(item.src_object_name),
            "src_ib_hash": str(item.src_ib_hash or "").lower(),
            "src_local_bone": int(item.src_local_bone),
            "src_global_bone": int(item.src_global_bone),
            "canonical_draw_index": int(item.canonical_draw_index),
            "canonical_object_name": str(item.canonical_object_name),
            "canonical_ib_hash": str(item.canonical_ib_hash or "").lower(),
            "canonical_local_bone": int(item.canonical_local_bone),
            "canonical_global_bone": int(item.canonical_global_bone),
            "confidence": str(item.confidence or ""),
        }
        for item in scene.bmc_alias_items
    ]


def load_mapping_payload_from_scene(scene) -> dict:
    raw_payload = str(getattr(scene, "bmc_mapping_payload_json", "") or "").strip()
    if raw_payload:
        try:
            return json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Stored mapping payload is invalid JSON: {exc}") from exc
    return {}


def store_mapping_payload_on_scene(scene, payload: dict) -> None:
    scene.bmc_mapping_payload_json = json.dumps(payload, ensure_ascii=False)


def build_mapping_payload(scene, manifest_payload: dict | None = None, base_payload: dict | None = None) -> dict:
    base_payload = dict(base_payload or {})
    if manifest_payload is None:
        manifest_payload = _load_manifest_snapshot(scene)
    manifest_payload = dict(manifest_payload)

    object_remaps = manifest_payload.get("object_remaps")
    if not isinstance(object_remaps, list) or not object_remaps:
        object_remaps = list(base_payload.get("object_remaps", []))

    capture_manifest = _extract_capture_manifest_snapshot(manifest_payload)
    if not capture_manifest:
        capture_manifest = dict(base_payload.get("capture_manifest", {}))

    alias_payload = serialize_alias_items(scene)
    if not alias_payload:
        alias_payload = list(manifest_payload.get("bone_aliases", []))
    if not alias_payload:
        alias_payload = list(base_payload.get("bone_aliases", []))

    target_payload = serialize_target_items(scene)
    if not target_payload:
        target_payload = list(base_payload.get("targets", []))

    shadow_host_hash = str(scene.bmc_shadow_host_hash or base_payload.get("shadow_host_hash", "") or "").lower()
    shadow_host_match_index_count = int(
        getattr(scene, "bmc_shadow_host_match_index_count", -1)
        if getattr(scene, "bmc_shadow_host_match_index_count", -1) >= 0
        else base_payload.get("shadow_host_match_index_count", -1)
    )
    shadow_host_vs_hash = str(scene.bmc_shadow_host_vs_hash or base_payload.get("shadow_host_vs_hash", "") or "").lower()

    return {
        "preset_type": "bmc_mapping",
        "version": 1,
        "workspace": {
            "frameanalysis_dir": str(scene.bmc_frameanalysis_dir or base_payload.get("workspace", {}).get("frameanalysis_dir", "") or ""),
            "output_dir": str(scene.bmc_output_dir or base_payload.get("workspace", {}).get("output_dir", "") or ""),
            "source_ini_path": str(scene.bmc_source_ini_path or base_payload.get("workspace", {}).get("source_ini_path", "") or ""),
            "target_collection_name": scene.bmc_target_collection.name if scene.bmc_target_collection else str(base_payload.get("workspace", {}).get("target_collection_name", "") or ""),
            "export_collection_name": scene.bmc_export_collection.name if scene.bmc_export_collection else str(base_payload.get("workspace", {}).get("export_collection_name", "") or ""),
            "export_build_collection_name": scene.bmc_export_build_collection.name if scene.bmc_export_build_collection else str(base_payload.get("workspace", {}).get("export_build_collection_name", "") or ""),
        },
        "targets": target_payload,
        "capture_manifest": capture_manifest,
        "object_remaps": object_remaps,
        "bone_aliases": alias_payload,
        "shadow_host_hash": shadow_host_hash,
        "shadow_host_match_index_count": shadow_host_match_index_count,
        "shadow_host_vs_hash": shadow_host_vs_hash,
    }


def apply_mapping_payload_to_scene(scene, payload: dict, preset_name: str = "") -> None:
    normalized_payload = {
        "preset_type": str(payload.get("preset_type", "bmc_mapping") or "bmc_mapping"),
        "version": int(payload.get("version", 1)),
        "workspace": dict(payload.get("workspace", {}) or {}),
        "targets": list(payload.get("targets", payload.get("frozen_targets", [])) or []),
        "capture_manifest": dict(payload.get("capture_manifest", {}) or {}),
        "object_remaps": list(payload.get("object_remaps", []) or []),
        "bone_aliases": list(payload.get("bone_aliases", payload.get("aliases", [])) or []),
        "shadow_host_hash": str(payload.get("shadow_host_hash", "") or ""),
        "shadow_host_match_index_count": int(payload.get("shadow_host_match_index_count", -1)),
        "shadow_host_vs_hash": str(payload.get("shadow_host_vs_hash", "") or ""),
    }

    if preset_name:
        scene.bmc_preset_name = preset_name
        scene.bmc_preset_choice = preset_name

    workspace = normalized_payload.get("workspace", {})
    if isinstance(workspace, dict):
        scene.bmc_frameanalysis_dir = str(workspace.get("frameanalysis_dir", "") or scene.bmc_frameanalysis_dir or "")
        scene.bmc_output_dir = str(workspace.get("output_dir", "") or scene.bmc_output_dir or "")
        scene.bmc_source_ini_path = str(workspace.get("source_ini_path", "") or scene.bmc_source_ini_path or "")

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

        export_build_collection_name = str(workspace.get("export_build_collection_name", "") or "").strip()
        if export_build_collection_name:
            export_build_collection = bpy.data.collections.get(export_build_collection_name)
            if export_build_collection is not None:
                scene.bmc_export_build_collection = export_build_collection

    scene.bmc_shadow_host_hash = str(normalized_payload.get("shadow_host_hash", "") or "")
    scene.bmc_shadow_host_match_index_count = int(normalized_payload.get("shadow_host_match_index_count", -1))
    scene.bmc_shadow_host_vs_hash = str(normalized_payload.get("shadow_host_vs_hash", "") or "")

    scene.bmc_target_items.clear()
    for target in normalized_payload.get("targets", []):
        item = scene.bmc_target_items.add()
        item.object_name = str(target.get("object_name", ""))
        item.ib_hash = str(target.get("ib_hash", ""))
        item.match_index_count = int(target.get("match_index_count", 0))
        item.local_bone_count = int(target.get("local_bone_count", 0))
        item.autodetected = bool(target.get("autodetected", True))
        item.enabled = bool(target.get("enabled", True))
        mesh_obj = scene.objects.get(item.object_name)
        if mesh_obj is not None and mesh_obj.type == "MESH":
            item.object_ref = mesh_obj
            item.object_name = mesh_obj.name
    scene.bmc_target_index = min(scene.bmc_target_index, max(0, len(scene.bmc_target_items) - 1))

    scene.bmc_alias_items.clear()
    for alias in normalized_payload.get("bone_aliases", []):
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

    store_mapping_payload_on_scene(scene, normalized_payload)


def _load_manifest_snapshot(scene) -> dict:
    manifest_path = bpy.path.abspath(str(getattr(scene, "bmc_manifest_path", "") or ""))
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _extract_capture_manifest_snapshot(manifest_payload: dict) -> dict:
    if not manifest_payload:
        return {}
    return {
        "frameanalysis_dir": str(manifest_payload.get("frameanalysis_dir", "") or ""),
        "bonestore_namespace": str(manifest_payload.get("bonestore_namespace", "") or ""),
        "selected_vs_hashes": list(manifest_payload.get("selected_vs_hashes", []) or []),
        "part_records": list(manifest_payload.get("part_records", []) or []),
    }
