"""Texture candidate and mark helpers for hash-based texture replacement."""

from __future__ import annotations

import json
import re
from pathlib import Path

_PS_SLOT_RE = re.compile(r"^ps-t(?P<index>\d+)$")
_HASH8_RE = re.compile(r"^[0-9a-fA-F]{8}$")


def slot_sort_key(slot: str) -> int:
    match = _PS_SLOT_RE.match(str(slot or "").strip().lower())
    return int(match.group("index")) if match else 999


def region_label(region_key: str) -> str:
    parts = str(region_key or "").replace("_", "-").split("-")
    if len(parts) >= 3:
        return f"{parts[0]} count={parts[1]} first={parts[2]}"
    return str(region_key or "")


def load_texture_mark_payload(raw_payload: str | dict | None) -> dict:
    if isinstance(raw_payload, dict):
        return dict(raw_payload)
    text = str(raw_payload or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Stored texture mark payload is invalid JSON: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def dump_texture_mark_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_texture_mark_payload(manifest: dict, existing_payload: dict | None = None) -> dict:
    """Build UI-ready texture candidates from capture_manifest.texture_candidates."""

    old_marks = {}
    if isinstance(existing_payload, dict) and isinstance(existing_payload.get("marks"), dict):
        old_marks = existing_payload["marks"]

    candidates: dict[str, dict[str, dict[str, dict]]] = {}
    draw_meta: dict[str, dict[str, dict]] = {}
    for raw_candidate in manifest.get("texture_candidates", []) or []:
        if not isinstance(raw_candidate, dict):
            continue
        region_key = str(raw_candidate.get("region_key", "") or "").strip()
        draw_key = str(int(raw_candidate.get("draw_index", 0) or 0))
        slot = str(raw_candidate.get("slot", "") or "").strip().lower()
        texture_hash = str(raw_candidate.get("hash", "") or "").strip().lower()
        source_path = str(raw_candidate.get("source_path", "") or "").strip()
        if not region_key or not draw_key or not slot or not texture_hash or not source_path:
            continue
        extension = Path(source_path).suffix.lower().lstrip(".") or str(raw_candidate.get("extension", "") or "dds")
        binding = {
            "slot": slot,
            "hash": texture_hash,
            "source_path": source_path,
            "extension": extension,
            "draw_index": int(raw_candidate.get("draw_index", 0) or 0),
            "ps_hash": str(raw_candidate.get("ps_hash", "") or "").strip().lower(),
            "rt_count": int(raw_candidate.get("rt_count", -1) or -1),
            "semantic_hint": str(raw_candidate.get("semantic_hint", "") or ""),
        }
        candidates.setdefault(region_key, {}).setdefault(draw_key, {})[slot] = binding
        draw_meta.setdefault(region_key, {}).setdefault(
            draw_key,
            {
                "draw_index": int(raw_candidate.get("draw_index", 0) or 0),
                "ps_hash": str(raw_candidate.get("ps_hash", "") or "").strip().lower(),
                "rt_count": int(raw_candidate.get("rt_count", -1) or -1),
            },
        )

    default_draws: dict[str, str] = {}
    for region_key, draw_candidates in candidates.items():
        best_key = max(
            draw_candidates,
            key=lambda draw_key: (
                len(draw_candidates.get(draw_key, {})),
                int(draw_meta.get(region_key, {}).get(draw_key, {}).get("rt_count", -1) or -1),
                int(draw_key),
            ),
        )
        default_draws[region_key] = best_key

    return {
        "version": 1,
        "binding_mode": "texture_hash_override",
        "candidates": candidates,
        "draws": draw_meta,
        "default_draws": default_draws,
        "marks": old_marks,
    }


def texture_candidates_for_draw(payload: dict, region_key: str, draw_key: str) -> tuple[dict, dict]:
    candidates = payload.get("candidates", {})
    marks = payload.get("marks", {})
    if not isinstance(candidates, dict):
        candidates = {}
    if not isinstance(marks, dict):
        marks = {}
    region_candidates = candidates.get(str(region_key), {})
    region_marks = marks.get(str(region_key), {})
    if not isinstance(region_candidates, dict):
        region_candidates = {}
    if not isinstance(region_marks, dict):
        region_marks = {}
    draw_candidates = region_candidates.get(str(draw_key), {})
    draw_marks = region_marks.get(str(draw_key), {})
    return (
        draw_candidates if isinstance(draw_candidates, dict) else {},
        draw_marks if isinstance(draw_marks, dict) else {},
    )


def marked_texture_bindings(payload: dict) -> list[dict]:
    """Return marked texture bindings, preserving UI semantic metadata."""

    candidates = payload.get("candidates", {})
    marks = payload.get("marks", {})
    if not isinstance(candidates, dict) or not isinstance(marks, dict):
        return []

    bindings: list[dict] = []
    for region_key, region_marks in sorted(marks.items()):
        if not isinstance(region_marks, dict):
            continue
        region_candidates = candidates.get(str(region_key), {})
        if not isinstance(region_candidates, dict):
            continue
        for draw_key, draw_marks in sorted(region_marks.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0):
            if not isinstance(draw_marks, dict):
                continue
            draw_candidates = region_candidates.get(str(draw_key), {})
            if not isinstance(draw_candidates, dict):
                continue
            for slot, mark in sorted(draw_marks.items(), key=lambda item: slot_sort_key(item[0])):
                if not isinstance(mark, dict):
                    continue
                candidate = draw_candidates.get(str(slot))
                if not isinstance(candidate, dict):
                    continue
                binding = dict(candidate)
                binding["region_key"] = str(region_key)
                binding["draw_key"] = str(draw_key)
                binding["slot"] = str(slot)
                binding["semantic"] = str(mark.get("semantic", "") or "").strip()
                binding["semantic_index"] = int(mark.get("semantic_index", 0) or 0)
                bindings.append(binding)
    return bindings


def validate_texture_hash(hash_value: str) -> bool:
    return _HASH8_RE.fullmatch(str(hash_value or "").strip()) is not None
