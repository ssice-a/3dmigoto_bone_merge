"""Small scene-side cache for the current Bone Merge analysis state."""

from __future__ import annotations

import json
import os


def load_mapping_payload_from_scene(scene) -> dict:
    raw_payload = str(getattr(scene, "bmc_mapping_payload_json", "") or "").strip()
    if not raw_payload:
        return {}
    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Stored mapping payload is invalid JSON: {exc}") from exc


def store_mapping_payload_on_scene(scene, payload: dict) -> None:
    scene.bmc_mapping_payload_json = json.dumps(payload, ensure_ascii=False)


def build_mapping_payload(scene, manifest_payload: dict | None = None, base_payload: dict | None = None) -> dict:
    base_payload = dict(base_payload or {})
    if manifest_payload is None:
        manifest_payload = _load_manifest_snapshot(scene)
    manifest_payload = dict(manifest_payload or {})

    object_remaps = manifest_payload.get("object_remaps")
    if not isinstance(object_remaps, list) or not object_remaps:
        object_remaps = list(base_payload.get("object_remaps", []) or [])

    return {
        "version": 2,
        "workspace": {
            "frameanalysis_dir": str(getattr(scene, "bmc_frameanalysis_dir", "") or base_payload.get("workspace", {}).get("frameanalysis_dir", "") or ""),
            "lod_frameanalysis_dir": str(getattr(scene, "bmc_lod_frameanalysis_dir", "") or base_payload.get("workspace", {}).get("lod_frameanalysis_dir", "") or ""),
            "output_dir": str(getattr(scene, "bmc_output_dir", "") or base_payload.get("workspace", {}).get("output_dir", "") or ""),
            "export_collection_name": _collection_name(getattr(scene, "bmc_export_collection", None), base_payload),
        },
        "capture_manifest": _extract_capture_manifest_snapshot(manifest_payload)
        or dict(base_payload.get("capture_manifest", {}) or {}),
        "object_remaps": object_remaps,
        "lod_mapping": list(manifest_payload.get("lod_mapping", base_payload.get("lod_mapping", [])) or []),
        "lod_profiles": list(manifest_payload.get("lod_profiles", base_payload.get("lod_profiles", [])) or []),
        "lod_chains": list(manifest_payload.get("lod_chains", base_payload.get("lod_chains", [])) or []),
        "lod_capture_records": list(manifest_payload.get("lod_capture_records", base_payload.get("lod_capture_records", [])) or []),
        "lod_links": list(manifest_payload.get("lod_links", base_payload.get("lod_links", [])) or []),
        "lod_review": dict(manifest_payload.get("lod_review", base_payload.get("lod_review", {})) or {}),
        "shadow_stage": dict(manifest_payload.get("shadow_stage", base_payload.get("shadow_stage", {})) or {}),
        "lod_manifest_snapshot": dict(manifest_payload.get("lod_manifest_snapshot", base_payload.get("lod_manifest_snapshot", {})) or {}),
        "lod_manifest_snapshots": list(manifest_payload.get("lod_manifest_snapshots", base_payload.get("lod_manifest_snapshots", [])) or []),
    }


def _collection_name(collection, base_payload: dict) -> str:
    if collection is not None:
        return str(collection.name)
    return str(base_payload.get("workspace", {}).get("export_collection_name", "") or "")


def _load_manifest_snapshot(scene) -> dict:
    manifest_path = os.path.abspath(str(getattr(scene, "bmc_manifest_path", "") or ""))
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
        "selected_vs_hashes": list(manifest_payload.get("selected_vs_hashes", []) or []),
        "shadow_stage": dict(manifest_payload.get("shadow_stage", {}) or {}),
        "bone_pool_order": list(manifest_payload.get("bone_pool_order", []) or []),
        "object_remaps": list(manifest_payload.get("object_remaps", []) or []),
        "global_pool_generation": str(manifest_payload.get("global_pool_generation", "") or ""),
    }
