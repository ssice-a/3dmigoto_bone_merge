"""Known shader input contracts and vertex layout profiles.

The analyzer records the raw FrameAnalysis layout as the source of truth.
This package only adds small, auditable annotations from disassembled shaders
and previously observed slot structures.
"""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path


_DATA_DIR = Path(__file__).resolve().parent
_SEMANTIC_RE = re.compile(r"^(?P<name>[A-Za-z_]+)(?P<index>\d*)$")


@lru_cache(maxsize=None)
def _load_json(name: str) -> dict:
    path = _DATA_DIR / name
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def get_vs_input_contract(vs_hash: str) -> dict:
    """Return the known VS input contract for a shader hash, if present."""

    normalized = str(vs_hash or "").lower()
    if not normalized:
        return {}
    contracts = _load_json("vs_input_contracts.json").get("contracts", {})
    payload = contracts.get(normalized, {})
    return copy.deepcopy(payload) if payload else {}


def get_vertex_layout_profiles() -> dict:
    """Return known vertex layout profile definitions keyed by profile id."""

    profiles = _load_json("vertex_layout_profiles.json").get("profiles", {})
    return copy.deepcopy(profiles)


def annotate_vertex_layout(layout: dict, vs_hash: str = "") -> dict:
    """Attach known VS/layout facts to a manifest vertex layout payload.

    The returned payload still keeps the raw FrameAnalysis fields unchanged.
    Annotations are advisory and should not be used as a replacement for the
    actual recorded offset/stride/format data.
    """

    annotated = copy.deepcopy(layout)
    normalized_vs_hash = str(vs_hash or annotated.get("import_vs_hash", "") or "").lower()
    if normalized_vs_hash:
        annotated["import_vs_hash"] = normalized_vs_hash

    contract = get_vs_input_contract(normalized_vs_hash)
    if contract:
        annotated["vs_input_contract"] = contract

    input_roles = _input_roles(contract)
    for slot_payload in dict(annotated.get("vertex_buffers", {}) or {}).values():
        profile_id = classify_vertex_slot_layout(slot_payload)
        if profile_id:
            slot_payload["layout_profile"] = profile_id
        semantic_roles: dict[str, dict] = {}
        for field in list(slot_payload.get("fields", []) or []):
            semantic_key = _field_semantic_key(field)
            if not semantic_key:
                continue
            if input_roles:
                role = input_roles.get(semantic_key, "")
                semantic_roles[semantic_key] = {
                    "used_by_import_vs": semantic_key in input_roles,
                    "vs_role": role,
                }
        if semantic_roles:
            slot_payload["vs_semantic_roles"] = semantic_roles
    return annotated


def classify_vertex_slot_layout(slot_layout: dict) -> str:
    """Return the first known profile matching a vertex slot layout."""

    slot_name = str(slot_layout.get("slot", "") or "").lower()
    stride = int(slot_layout.get("stride", 0) or 0)
    fields = {_field_signature(field) for field in slot_layout.get("fields", []) or []}
    for profile_id, profile in get_vertex_layout_profiles().items():
        expected_slot = str(profile.get("slot", "") or "").lower()
        expected_stride = int(profile.get("stride", 0) or 0)
        if expected_slot and expected_slot != slot_name:
            continue
        if expected_stride and expected_stride != stride:
            continue
        required = {_profile_field_signature(field) for field in profile.get("fields", []) or []}
        if required and required.issubset(fields):
            return str(profile_id)
    return ""


def _input_roles(contract: dict) -> dict[str, str]:
    roles: dict[str, str] = {}
    for item in contract.get("inputs", []) or []:
        semantic = _normalize_semantic(str(item.get("semantic", "") or ""))
        if not semantic:
            continue
        roles[semantic] = str(item.get("role", "") or "")
    return roles


def _field_semantic_key(field: dict) -> str:
    semantic = str(field.get("semantic", "") or "")
    if semantic:
        return _normalize_semantic(semantic)
    name = str(field.get("semantic_name", "") or "")
    index = int(field.get("semantic_index", 0) or 0)
    return _normalize_semantic(f"{name}{index}")


def _field_signature(field: dict) -> tuple[str, str, int]:
    return (
        _field_semantic_key(field),
        str(field.get("format", "") or "").upper(),
        int(field.get("aligned_byte_offset", field.get("offset", -1)) or -1),
    )


def _profile_field_signature(field: dict) -> tuple[str, str, int]:
    return (
        _normalize_semantic(str(field.get("semantic", "") or "")),
        str(field.get("format", "") or "").upper(),
        int(field.get("offset", -1) or -1),
    )


def _normalize_semantic(semantic: str) -> str:
    match = _SEMANTIC_RE.match(str(semantic or "").strip())
    if not match:
        return ""
    name = match.group("name").upper()
    index = match.group("index")
    if index == "":
        index = "0"
    return f"{name}{int(index)}"
