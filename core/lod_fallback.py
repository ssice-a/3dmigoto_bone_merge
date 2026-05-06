"""Fallback helpers for unresolved LOD global bones used by export meshes."""

from __future__ import annotations

import math
from collections import defaultdict

from ..constants import BMC_EXPORT_PALETTE_PROP

_WEIGHT_EPSILON = 1.0e-6


def preview_lod_fallbacks_for_export(export_collection, manifest: dict) -> dict:
    """Build non-mutating fallback suggestions for unmatched LOD globals used by export meshes."""

    mapping_entries = list(manifest.get("lod_mapping", []) or [])
    if not mapping_entries:
        return _empty_preview("no_lod_mapping")

    usage = collect_export_global_group_usage(export_collection)
    used_globals = set(usage["used_global_bones"])
    unmatched_required = find_unmatched_required_lod_globals(manifest)
    unmatched_used = sorted(used_globals & unmatched_required)
    unused_unmatched = sorted(unmatched_required - used_globals)
    if not unmatched_used:
        return {
            **usage,
            "unmatched_used_global_bones": [],
            "unused_unmatched_global_bones": unused_unmatched,
            "fallbacks": [],
            "unresolved": [],
            "summary": {
                "used_global_bone_count": len(used_globals),
                "unmatched_used_count": 0,
                "unused_unmatched_count": len(unused_unmatched),
                "fallback_count": 0,
                "unresolved_count": 0,
            },
        }

    mapping_by_global = _mapping_by_global(mapping_entries)
    resolved_mapping_by_global = {
        global_bone: entry
        for global_bone, entry in mapping_by_global.items()
        if _is_lod_mapping_resolved(entry)
    }
    matched_export_globals = used_globals & set(resolved_mapping_by_global)
    fallbacks = []
    unresolved = []
    for missing_global in unmatched_used:
        fallback = _build_fallback_for_global(
            missing_global,
            usage,
            matched_export_globals,
            resolved_mapping_by_global,
        )
        if fallback is None:
            unresolved.append(
                {
                    "canonical_global_bone": int(missing_global),
                    "status": "unresolved",
                    "note": "No matched donor global bone was found in the current export weight distribution.",
                }
            )
            continue
        fallbacks.append(fallback)

    return {
        **usage,
        "unmatched_used_global_bones": unmatched_used,
        "unused_unmatched_global_bones": unused_unmatched,
        "fallbacks": fallbacks,
        "unresolved": unresolved,
        "summary": {
            "used_global_bone_count": len(used_globals),
            "unmatched_used_count": len(unmatched_used),
            "unused_unmatched_count": len(unused_unmatched),
            "fallback_count": len(fallbacks),
            "unresolved_count": len(unresolved),
        },
    }


def apply_lod_fallbacks_to_manifest(manifest: dict, preview: dict) -> dict:
    """Apply previewed fallback entries into lod_mapping and lod_capture_records."""

    fallbacks = list(preview.get("fallbacks", []) or [])
    if not fallbacks:
        return {"applied_count": 0, "capture_record_count": len(manifest.get("lod_capture_records", []) or [])}

    mapping_entries = list(manifest.get("lod_mapping", []) or [])
    mapping_by_global = _mapping_by_global(mapping_entries)
    for fallback in fallbacks:
        canonical_global = int(fallback.get("canonical_global_bone", -1))
        if canonical_global < 0:
            continue
        entry = mapping_by_global.get(canonical_global)
        if entry is None:
            entry = {"canonical_global_bone": canonical_global}
            mapping_entries.append(entry)
            mapping_by_global[canonical_global] = entry
        entry.update(
            {
                "enabled": True,
                "lod_record_key": str(fallback.get("lod_record_key", "") or ""),
                "lod_local_bone": int(fallback.get("lod_local_bone", -1)),
                "mapped_lod_global_bone": int(fallback.get("mapped_lod_global_bone", -1)),
                "score": float(fallback.get("confidence", 0.0)),
                "votes": int(fallback.get("votes", 0) or 0),
                "average_distance": float(fallback.get("average_distance", 0.0) or 0.0),
                "status": "fallback_inherited",
                "donor_global_bone": int(fallback.get("donor_global_bone", -1)),
                "fallback_method": str(fallback.get("method", "") or ""),
                "fallback_confidence": float(fallback.get("confidence", 0.0)),
                "note": _fallback_note(fallback),
            }
        )

    manifest["lod_mapping"] = sorted(
        mapping_entries,
        key=lambda entry: int(entry.get("canonical_global_bone", 0) or 0),
    )
    manifest["lod_capture_records"] = _append_fallback_capture_pairs(
        list(manifest.get("lod_capture_records", []) or []),
        fallbacks,
    )
    manifest.setdefault("validation", []).append(
        {
            "severity": "warning",
            "code": "lod_fallback_inherited_missing_bones",
            "message": f"Applied {len(fallbacks)} LOD fallback mapping(s). These are inherited donors, not exact native matches.",
            "draw_indices": [],
        }
    )
    return {
        "applied_count": len(fallbacks),
        "capture_record_count": len(manifest.get("lod_capture_records", []) or []),
    }


def collect_export_global_group_usage(export_collection) -> dict:
    assignments_by_group: dict[int, list[dict]] = defaultdict(list)
    assignments_by_vertex: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    object_names_by_group: dict[int, set[str]] = defaultdict(set)

    for mesh_obj in _iter_mesh_objects_recursive(export_collection):
        group_index_to_global = _build_group_index_to_global_map(mesh_obj)
        if not group_index_to_global:
            continue
        for vertex in getattr(getattr(mesh_obj, "data", None), "vertices", []) or []:
            vertex_index = int(getattr(vertex, "index", 0))
            vertex_key = (str(getattr(mesh_obj, "name", "")), vertex_index)
            position = _vertex_position(mesh_obj, vertex)
            for group_element in getattr(vertex, "groups", []) or []:
                global_group = group_index_to_global.get(int(getattr(group_element, "group", -1)))
                if global_group is None:
                    continue
                weight = float(getattr(group_element, "weight", 0.0))
                if weight <= _WEIGHT_EPSILON:
                    continue
                assignments_by_vertex[vertex_key].append((global_group, weight))
                assignments_by_group[global_group].append(
                    {
                        "object_name": str(getattr(mesh_obj, "name", "")),
                        "vertex_index": vertex_index,
                        "vertex_key": vertex_key,
                        "weight": weight,
                        "position": position,
                    }
                )
                object_names_by_group[global_group].add(str(getattr(mesh_obj, "name", "")))

    group_infos = {
        global_group: _group_info(global_group, assignments, object_names_by_group[global_group])
        for global_group, assignments in assignments_by_group.items()
    }
    return {
        "used_global_bones": sorted(assignments_by_group),
        "group_infos": group_infos,
        "assignments_by_vertex": {
            f"{object_name}:{vertex_index}": tuple(values)
            for (object_name, vertex_index), values in assignments_by_vertex.items()
        },
    }


def find_unmatched_required_lod_globals(manifest: dict) -> set[int]:
    unmatched: set[int] = set()
    for entry in manifest.get("lod_mapping", []) or []:
        status = str(entry.get("status", "") or "")
        if status == "ignored_lod_match_excluded":
            continue
        canonical_global = int(entry.get("canonical_global_bone", -1))
        if canonical_global < 0:
            continue
        if not _is_lod_mapping_resolved(entry):
            unmatched.add(canonical_global)
    return unmatched


def _build_fallback_for_global(
    missing_global: int,
    usage: dict,
    matched_export_globals: set[int],
    resolved_mapping_by_global: dict[int, dict],
) -> dict | None:
    if not matched_export_globals:
        return None
    group_infos = dict(usage.get("group_infos", {}) or {})
    missing_info = group_infos.get(int(missing_global))
    if not missing_info:
        return None

    shared = _select_shared_vertex_donor(missing_global, usage, matched_export_globals)
    if shared is not None:
        donor_global, confidence, votes = shared
        return _fallback_payload(missing_global, donor_global, resolved_mapping_by_global[donor_global], "shared_vertex_weight", confidence, votes)

    same_object_candidates = {
        donor_global
        for donor_global in matched_export_globals
        if donor_global in group_infos
        and set(group_infos[donor_global].get("object_names", ())) & set(missing_info.get("object_names", ()))
    }
    nearest = _select_nearest_donor(missing_global, group_infos, same_object_candidates)
    if nearest is not None:
        donor_global, confidence, distance = nearest
        return _fallback_payload(
            missing_global,
            donor_global,
            resolved_mapping_by_global[donor_global],
            "same_object_nearest_weight_cloud",
            confidence,
            0,
            distance,
        )

    nearest = _select_nearest_donor(missing_global, group_infos, matched_export_globals)
    if nearest is not None:
        donor_global, confidence, distance = nearest
        return _fallback_payload(
            missing_global,
            donor_global,
            resolved_mapping_by_global[donor_global],
            "export_nearest_weight_cloud",
            confidence,
            0,
            distance,
        )
    return None


def _select_shared_vertex_donor(missing_global: int, usage: dict, matched_export_globals: set[int]) -> tuple[int, float, int] | None:
    group_infos = dict(usage.get("group_infos", {}) or {})
    missing_info = group_infos.get(int(missing_global), {})
    missing_total_weight = max(float(missing_info.get("total_weight", 0.0) or 0.0), _WEIGHT_EPSILON)
    by_vertex = dict(usage.get("assignments_by_vertex", {}) or {})
    scores: dict[int, float] = defaultdict(float)
    votes: dict[int, int] = defaultdict(int)
    for assignments in by_vertex.values():
        missing_weight = 0.0
        for global_group, weight in assignments:
            if int(global_group) == int(missing_global):
                missing_weight += float(weight)
        if missing_weight <= _WEIGHT_EPSILON:
            continue
        for global_group, weight in assignments:
            donor_global = int(global_group)
            if donor_global not in matched_export_globals or donor_global == int(missing_global):
                continue
            scores[donor_global] += min(float(weight), missing_weight)
            votes[donor_global] += 1
    if not scores:
        return None
    donor_global, score = max(scores.items(), key=lambda item: (float(item[1]), int(votes[item[0]]), -item[0]))
    return donor_global, min(1.0, float(score) / missing_total_weight), int(votes[donor_global])


def _select_nearest_donor(missing_global: int, group_infos: dict[int, dict], donor_globals: set[int]) -> tuple[int, float, float] | None:
    missing_info = group_infos.get(int(missing_global))
    if not missing_info:
        return None
    missing_center = tuple(float(value) for value in missing_info.get("center", (0.0, 0.0, 0.0)))
    missing_diag = float(missing_info.get("bounds_diag", 0.0) or 0.0)
    candidates = []
    for donor_global in donor_globals:
        donor_info = group_infos.get(int(donor_global))
        if not donor_info:
            continue
        donor_center = tuple(float(value) for value in donor_info.get("center", (0.0, 0.0, 0.0)))
        distance = _distance(missing_center, donor_center)
        scale = max(missing_diag, float(donor_info.get("bounds_diag", 0.0) or 0.0), 1.0e-5)
        confidence = 1.0 / (1.0 + distance / scale)
        candidates.append((int(donor_global), float(confidence), float(distance)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1], -item[2], -item[0]))


def _fallback_payload(
    missing_global: int,
    donor_global: int,
    donor_mapping: dict,
    method: str,
    confidence: float,
    votes: int = 0,
    average_distance: float = 0.0,
) -> dict:
    return {
        "canonical_global_bone": int(missing_global),
        "donor_global_bone": int(donor_global),
        "lod_record_key": str(donor_mapping.get("lod_record_key", "") or ""),
        "lod_local_bone": int(donor_mapping.get("lod_local_bone", -1)),
        "mapped_lod_global_bone": int(donor_mapping.get("mapped_lod_global_bone", -1)),
        "method": method,
        "confidence": float(confidence),
        "votes": int(votes),
        "average_distance": float(average_distance),
        "status": "fallback_preview",
        "note": f"G{int(missing_global)} inherits LOD mapping from G{int(donor_global)} via {method}.",
    }


def _append_fallback_capture_pairs(capture_records: list[dict], fallbacks: list[dict]) -> list[dict]:
    records_by_key = {str(record.get("lod_record_key", "") or ""): record for record in capture_records}
    for fallback in fallbacks:
        lod_key = str(fallback.get("lod_record_key", "") or "")
        if not lod_key:
            continue
        record = records_by_key.get(lod_key)
        if record is None:
            record = {"lod_record_key": lod_key, "scatter_pairs": []}
            capture_records.append(record)
            records_by_key[lod_key] = record
        canonical_global = int(fallback.get("canonical_global_bone", -1))
        pairs = list(record.get("scatter_pairs", []) or [])
        pairs = [
            pair
            for pair in pairs
            if int(pair.get("canonical_global_bone", -1)) != canonical_global
        ]
        pairs.append(
            {
                "lod_local_bone": int(fallback.get("lod_local_bone", -1)),
                "canonical_global_bone": canonical_global,
                "score": float(fallback.get("confidence", 0.0)),
                "votes": int(fallback.get("votes", 0) or 0),
                "status": "fallback_inherited",
                "donor_global_bone": int(fallback.get("donor_global_bone", -1)),
            }
        )
        record["scatter_pairs"] = sorted(
            pairs,
            key=lambda pair: (int(pair.get("lod_local_bone", -1)), int(pair.get("canonical_global_bone", -1))),
        )
    return capture_records


def _mapping_by_global(mapping_entries: list[dict]) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    for entry in mapping_entries:
        canonical_global = int(entry.get("canonical_global_bone", -1))
        if canonical_global >= 0:
            mapping[canonical_global] = entry
    return mapping


def _is_lod_mapping_resolved(entry: dict) -> bool:
    if str(entry.get("status", "") or "") == "ignored_lod_match_excluded":
        return False
    if str(entry.get("lod_record_key", "") or "") and int(entry.get("lod_local_bone", -1)) >= 0:
        return True
    return int(entry.get("mapped_lod_global_bone", -1)) >= 0


def _group_info(global_group: int, assignments: list[dict], object_names: set[str]) -> dict:
    total_weight = sum(float(item["weight"]) for item in assignments)
    if total_weight <= _WEIGHT_EPSILON:
        center = (0.0, 0.0, 0.0)
    else:
        center = tuple(
            sum(float(item["position"][axis]) * float(item["weight"]) for item in assignments) / total_weight
            for axis in range(3)
        )
    bounds_min = tuple(min(float(item["position"][axis]) for item in assignments) for axis in range(3))
    bounds_max = tuple(max(float(item["position"][axis]) for item in assignments) for axis in range(3))
    return {
        "global_bone": int(global_group),
        "total_weight": float(total_weight),
        "vertex_count": len({(item["object_name"], item["vertex_index"]) for item in assignments}),
        "center": center,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "bounds_diag": _distance(bounds_min, bounds_max),
        "object_names": tuple(sorted(object_names)),
    }


def _build_group_index_to_global_map(mesh_obj) -> dict[int, int]:
    localized_palette = _localized_palette(mesh_obj)
    group_index_to_global: dict[int, int] = {}
    for vertex_group in getattr(mesh_obj, "vertex_groups", []) or []:
        numeric = _parse_int(getattr(vertex_group, "name", ""))
        if numeric is None:
            continue
        if localized_palette:
            if 0 <= numeric < len(localized_palette):
                group_index_to_global[int(getattr(vertex_group, "index", numeric))] = int(localized_palette[numeric])
        else:
            group_index_to_global[int(getattr(vertex_group, "index", numeric))] = int(numeric)
    return group_index_to_global


def _localized_palette(mesh_obj) -> tuple[int, ...]:
    getter = getattr(mesh_obj, "get", None)
    raw = getter(BMC_EXPORT_PALETTE_PROP, None) if callable(getter) else None
    if not raw:
        return ()
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError):
        return ()


def _iter_mesh_objects_recursive(collection):
    if collection is None:
        return
    for mesh_obj in getattr(collection, "objects", []) or []:
        if getattr(mesh_obj, "type", "") == "MESH":
            yield mesh_obj
    for child in getattr(collection, "children", []) or []:
        yield from _iter_mesh_objects_recursive(child)


def _vertex_position(mesh_obj, vertex) -> tuple[float, float, float]:
    co = getattr(vertex, "co", (0.0, 0.0, 0.0))
    matrix = getattr(mesh_obj, "matrix_world", None)
    if matrix is not None:
        try:
            co = matrix @ co
        except Exception:
            pass
    return float(co[0]), float(co[1]), float(co[2])


def _parse_int(raw_value) -> int | None:
    value = str(raw_value).strip()
    if not value.isdigit():
        return None
    return int(value)


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(
        (float(left[0]) - float(right[0])) ** 2
        + (float(left[1]) - float(right[1])) ** 2
        + (float(left[2]) - float(right[2])) ** 2
    )


def _fallback_note(fallback: dict) -> str:
    return (
        f"LOD fallback: G{int(fallback.get('canonical_global_bone', -1))} inherits "
        f"G{int(fallback.get('donor_global_bone', -1))} by {fallback.get('method', '')}; "
        f"confidence {float(fallback.get('confidence', 0.0)):.3f}."
    )


def _empty_preview(reason: str) -> dict:
    return {
        "used_global_bones": [],
        "group_infos": {},
        "assignments_by_vertex": {},
        "unmatched_used_global_bones": [],
        "unused_unmatched_global_bones": [],
        "fallbacks": [],
        "unresolved": [],
        "summary": {
            "reason": reason,
            "used_global_bone_count": 0,
            "unmatched_used_count": 0,
            "unused_unmatched_count": 0,
            "fallback_count": 0,
            "unresolved_count": 0,
        },
    }
