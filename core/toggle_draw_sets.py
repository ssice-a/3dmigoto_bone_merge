"""Toggle draw-set normalization and export helpers."""

from __future__ import annotations

import re
from copy import deepcopy


_SAFE_ID_RE = re.compile(r"[^0-9A-Za-z_]+")
_VALID_KEY_TOKEN_RE = re.compile(r"^[A-Za-z0-9_]+$")
_MODIFIER_ALIASES = {
    "CTRL": "ctrl",
    "CONTROL": "ctrl",
    "ALT": "alt",
    "SHIFT": "shift",
    "NO_MODIFIERS": "no_modifiers",
}


def normalize_toggle_draw_sets(raw_groups) -> list[dict]:
    """Return deterministic toggle draw-set payloads.

    Input can be Blender PropertyGroups or plain dictionaries.  Output is the
    only schema consumed by export/runtime generation.
    """

    groups: list[dict] = []
    used_ids: set[str] = set()
    for group_index, raw_group in enumerate(raw_groups or []):
        if not _raw_get_bool(raw_group, "enabled", True):
            continue
        raw_values = _raw_get_collection(raw_group, "values")
        values = _normalize_values(raw_values)
        if not values:
            continue
        raw_id = _raw_get_str(raw_group, "toggle_id") or _raw_get_str(raw_group, "id")
        label = _raw_get_str(raw_group, "label") or _raw_get_str(raw_group, "name") or raw_id or f"Toggle {group_index + 1}"
        group_id = _unique_id(_safe_identifier(raw_id or label), used_ids)
        key = normalize_key_binding(_raw_get_str(raw_group, "key"))
        variable = _normalize_variable(_raw_get_str(raw_group, "variable"), group_id)
        default_value = _raw_get_int(raw_group, "default_value", values[0]["value"])
        value_numbers = {int(value["value"]) for value in values}
        if default_value not in value_numbers:
            default_value = values[0]["value"]
        groups.append(
            {
                "id": group_id,
                "label": str(label),
                "key": key,
                "variable": variable,
                "default_value": int(default_value),
                "values": values,
            }
        )
    return groups


def normalize_key_binding(raw_key: str) -> str:
    """Normalize user key text to a 3DMigoto-friendly key binding string."""

    tokens = [
        token.strip()
        for token in str(raw_key or "").replace("+", " ").replace(",", " ").split()
        if token.strip()
    ]
    normalized: list[str] = []
    for token in tokens:
        upper = token.upper()
        token_value = _MODIFIER_ALIASES.get(upper, upper)
        if not _VALID_KEY_TOKEN_RE.match(token_value):
            continue
        normalized.append(token_value)
    return " ".join(normalized)


def apply_toggle_draw_sets_to_geometry(geometry_records: list[dict], toggle_groups: list[dict]) -> tuple[list[dict], list[str]]:
    """Attach toggle metadata to object draw ranges by Blender object name."""

    if not toggle_groups:
        return list(geometry_records or []), []
    lookup = _object_toggle_lookup(toggle_groups)
    warnings: list[str] = []
    seen_objects: set[str] = set()
    output_records: list[dict] = []
    for record in geometry_records or []:
        new_record = deepcopy(record)
        object_draws = []
        for raw_draw in new_record.get("object_draws", []) or []:
            draw = dict(raw_draw or {})
            object_name = str(draw.get("object_name", "") or "")
            toggle = lookup.get(object_name)
            if toggle is not None:
                seen_objects.add(object_name)
                draw.update(toggle)
            object_draws.append(draw)
        new_record["object_draws"] = object_draws
        output_records.append(new_record)

    missing_objects = sorted(set(lookup).difference(seen_objects))
    if missing_objects:
        shown = ", ".join(missing_objects[:12])
        if len(missing_objects) > 12:
            shown += ", ..."
        warnings.append(f"Toggle draw set references object(s) not exported as draw ranges: {shown}")
    return output_records, warnings


def _object_toggle_lookup(toggle_groups: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for group in toggle_groups or []:
        group_id = str(group.get("id", "") or "")
        variable = str(group.get("variable", "") or "")
        group_label = str(group.get("label", "") or group_id)
        if not group_id or not variable:
            continue
        for value in group.get("values", []) or []:
            value_number = int(value.get("value", 0) or 0)
            value_label = str(value.get("label", "") or f"Value {value_number}")
            for object_name in value.get("objects", []) or []:
                object_name = str(object_name or "")
                if not object_name:
                    continue
                if object_name in lookup:
                    previous = lookup[object_name]
                    raise ValueError(
                        f"{object_name}: object is assigned to multiple toggle values "
                        f"({previous.get('toggle_label')}={previous.get('toggle_value')} and {group_label}={value_number})"
                    )
                lookup[object_name] = {
                    "toggle_group_id": group_id,
                    "toggle_label": group_label,
                    "toggle_variable": variable,
                    "toggle_value": value_number,
                    "toggle_value_label": value_label,
                    "toggle_condition": f"{variable} == {value_number}",
                }
    return lookup


def _normalize_values(raw_values) -> list[dict]:
    values: list[dict] = []
    used_values: set[int] = set()
    for raw_value in raw_values or []:
        value_number = _raw_get_int(raw_value, "value", 0)
        if value_number in used_values:
            continue
        used_values.add(value_number)
        label = _raw_get_str(raw_value, "label") or f"Value {value_number}"
        objects = []
        seen_objects: set[str] = set()
        for raw_object in _raw_get_collection(raw_value, "objects"):
            object_name = str(raw_object).strip() if isinstance(raw_object, str) else ""
            if not object_name:
                object_name = _raw_get_str(raw_object, "object_name") or _raw_get_str(raw_object, "name")
            if not object_name or object_name in seen_objects:
                continue
            objects.append(object_name)
            seen_objects.add(object_name)
        values.append(
            {
                "value": int(value_number),
                "label": label,
                "objects": objects,
            }
        )
    values.sort(key=lambda item: int(item["value"]))
    return values


def _normalize_variable(raw_variable: str, group_id: str) -> str:
    variable = str(raw_variable or "").strip()
    if variable and not variable.startswith("$"):
        variable = f"${variable}"
    if not variable:
        variable = f"$bmc_toggle_{group_id}"
    variable = "$" + _safe_identifier(variable[1:])
    return variable


def _unique_id(base: str, used_ids: set[str]) -> str:
    base = base or "toggle"
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _safe_identifier(value: str) -> str:
    safe = _SAFE_ID_RE.sub("_", str(value or "").strip()).strip("_").lower()
    if not safe:
        return "toggle"
    if safe[0].isdigit():
        safe = f"toggle_{safe}"
    return safe


def _raw_get_collection(raw, name: str):
    value = _raw_get(raw, name, [])
    return value if value is not None else []


def _raw_get_str(raw, name: str, default: str = "") -> str:
    return str(_raw_get(raw, name, default) or "").strip()


def _raw_get_int(raw, name: str, default: int = 0) -> int:
    try:
        return int(_raw_get(raw, name, default) or 0)
    except (TypeError, ValueError):
        return int(default)


def _raw_get_bool(raw, name: str, default: bool = False) -> bool:
    value = _raw_get(raw, name, default)
    return bool(value)


def _raw_get(raw, name: str, default=None):
    if isinstance(raw, dict):
        return raw.get(name, default)
    return getattr(raw, name, default)
