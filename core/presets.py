"""Preset persistence helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path


def get_presets_dir() -> str:
    presets_dir = Path(__file__).resolve().parent.parent / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    return str(presets_dir)


def list_preset_names() -> list[str]:
    presets_dir = Path(get_presets_dir())
    names: list[str] = []
    for path in sorted(presets_dir.glob("*.json")):
        names.append(path.stem)
    return names


def save_preset(preset_name: str, payload: dict) -> str:
    normalized_name = sanitize_preset_name(preset_name)
    preset_path = Path(get_presets_dir()) / f"{normalized_name}.json"
    with preset_path.open("w", encoding="utf-8", newline="\n") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
    return str(preset_path)


def load_preset(preset_name: str) -> dict:
    normalized_name = sanitize_preset_name(preset_name)
    preset_path = Path(get_presets_dir()) / f"{normalized_name}.json"
    if not preset_path.exists():
        raise ValueError(f"Preset not found: {normalized_name}")
    with preset_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def resolve_preset_workspace_paths(preset_name: str) -> dict[str, str | int]:
    payload = load_preset(preset_name)
    workspace = payload.get("workspace", {})
    if not isinstance(workspace, dict):
        return {}

    return {
        "frameanalysis_dir": str(workspace.get("frameanalysis_dir", "") or ""),
        "output_dir": str(workspace.get("output_dir", "") or ""),
        "manifest_path": "",
        "ini_path": "",
        "export_manifest_path": "",
        "shadow_host_hash": "",
        "shadow_host_match_index_count": -1,
        "shadow_host_vs_hash": "",
        "preset_cache_dir": "",
    }


def delete_preset(preset_name: str) -> None:
    normalized_name = sanitize_preset_name(preset_name)
    preset_path = Path(get_presets_dir()) / f"{normalized_name}.json"
    if not preset_path.exists():
        raise ValueError(f"Preset not found: {normalized_name}")
    os.remove(preset_path)


def sanitize_preset_name(raw_name: str) -> str:
    normalized = (raw_name or "").strip()
    if not normalized:
        raise ValueError("Preset name cannot be empty")
    safe_chars = []
    for char in normalized:
        if char.isalnum() or char in ("-", "_", " "):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    result = "".join(safe_chars).strip().replace(" ", "_")
    if not result:
        raise ValueError("Preset name cannot be empty")
    return result
