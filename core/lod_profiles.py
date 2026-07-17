"""LOD profile ownership, aggregation, and export validation."""

from __future__ import annotations

import copy
import os
import re
import zlib


LOD_PROFILE_SCHEMA_VERSION = 1

_RESULT_LIST_FIELDS = (
    "lod_frameanalysis",
    "lod_links",
    "lod_chains",
    "lod_capture_records",
    "lod_mapping",
)
_LEGACY_CLEAR_FIELDS = (
    *_RESULT_LIST_FIELDS,
    "lod_validation",
    "lod_manifest_snapshots",
    "lod_profile_conflicts",
)


def profile_id_for(lod_level: int, frameanalysis_dir: str) -> str:
    level = max(1, int(lod_level))
    normalized = os.path.normcase(os.path.abspath(str(frameanalysis_dir or "")))
    stem = os.path.basename(normalized.rstrip("\\/")) or f"lod{level}"
    safe_stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_") or f"lod{level}"
    checksum = zlib.crc32(f"{level}|{normalized}".encode("utf-8")) & 0xFFFFFFFF
    return f"lod{level}_{safe_stem}_{checksum:08x}"


def ensure_lod_profiles(manifest: dict) -> list[dict]:
    profiles = manifest.get("lod_profiles")
    if isinstance(profiles, list):
        manifest["lod_profile_schema_version"] = LOD_PROFILE_SCHEMA_VERSION
        return profiles

    profiles = []
    if _has_legacy_lod_result(manifest):
        frame_records = list(manifest.get("lod_frameanalysis", []) or [])
        frame_record = dict(frame_records[0] or {}) if frame_records else {}
        lod_level = max(1, int(frame_record.get("lod_level", 1) or 1))
        frameanalysis_dir = str(
            frame_record.get("frameanalysis_dir", "")
            or dict(manifest.get("lod_manifest_snapshot", {}) or {}).get("frameanalysis_dir", "")
            or ""
        )
        profiles.append(
            {
                "profile_id": profile_id_for(lod_level, frameanalysis_dir),
                "label": f"LOD {lod_level}",
                "lod_level": lod_level,
                "frameanalysis_dir": frameanalysis_dir,
                "enabled": True,
                "stale": False,
                "global_pool_generation": str(manifest.get("global_pool_generation", "") or ""),
                "result": _legacy_result(manifest),
            }
        )
    manifest["lod_profiles"] = profiles
    manifest["lod_profile_schema_version"] = LOD_PROFILE_SCHEMA_VERSION
    return profiles


def upsert_lod_profile(
    manifest: dict,
    *,
    profile_id: str,
    label: str,
    lod_level: int,
    frameanalysis_dir: str,
    enabled: bool,
    result: dict,
) -> dict:
    profiles = ensure_lod_profiles(manifest)
    normalized_dir = os.path.abspath(str(frameanalysis_dir or ""))
    normalized_id = str(profile_id or "").strip() or profile_id_for(lod_level, normalized_dir)
    profile = next(
        (item for item in profiles if str(dict(item or {}).get("profile_id", "") or "") == normalized_id),
        None,
    )
    if profile is None:
        profile = next(
            (
                item
                for item in profiles
                if not str(dict(item or {}).get("profile_id", "") or "")
                and max(1, int(dict(item or {}).get("lod_level", 1) or 1)) == max(1, int(lod_level))
                and os.path.normcase(os.path.abspath(str(dict(item or {}).get("frameanalysis_dir", "") or "")))
                == os.path.normcase(normalized_dir)
            ),
            None,
        )
    if profile is None:
        profile = {}
        profiles.append(profile)
    profile.update(
        {
            "profile_id": normalized_id,
            "label": str(label or "").strip() or f"LOD {max(1, int(lod_level))}",
            "lod_level": max(1, int(lod_level)),
            "frameanalysis_dir": normalized_dir,
            "enabled": bool(enabled),
            "stale": False,
            "stale_reason": "",
            "global_pool_generation": str(manifest.get("global_pool_generation", "") or ""),
            "result": _normalize_result(result),
        }
    )
    rebuild_lod_aggregate(manifest)
    return profile


def remove_lod_profile(manifest: dict, profile_id: str) -> bool:
    profiles = ensure_lod_profiles(manifest)
    normalized_id = str(profile_id or "")
    kept = [
        profile
        for profile in profiles
        if str(dict(profile or {}).get("profile_id", "") or "") != normalized_id
    ]
    if len(kept) == len(profiles):
        return False
    manifest["lod_profiles"] = kept
    rebuild_lod_aggregate(manifest)
    return True


def sync_lod_profile_settings(manifest: dict, settings: list[dict]) -> None:
    profiles = ensure_lod_profiles(manifest)
    by_id = {
        str(dict(profile or {}).get("profile_id", "") or ""): profile
        for profile in profiles
    }
    for setting in settings or []:
        profile_id = str(dict(setting or {}).get("profile_id", "") or "")
        profile = by_id.get(profile_id)
        if profile is None:
            continue
        previous_level = max(1, int(profile.get("lod_level", 1) or 1))
        previous_dir = str(profile.get("frameanalysis_dir", "") or "")
        profile["enabled"] = bool(setting.get("enabled", profile.get("enabled", True)))
        profile["label"] = str(setting.get("label", profile.get("label", "")) or "")
        next_level = max(1, int(setting.get("lod_level", profile.get("lod_level", 1)) or 1))
        frameanalysis_dir = str(setting.get("frameanalysis_dir", profile.get("frameanalysis_dir", "")) or "")
        next_dir = os.path.abspath(frameanalysis_dir) if frameanalysis_dir else ""
        profile["lod_level"] = next_level
        profile["frameanalysis_dir"] = next_dir
        settings_changed = previous_level != next_level or os.path.normcase(os.path.abspath(previous_dir or "")) != os.path.normcase(os.path.abspath(next_dir or ""))
        if settings_changed and dict(profile.get("result", {}) or {}):
            profile["stale"] = True
            profile["stale_reason"] = "lod_profile_settings_changed"
    rebuild_lod_aggregate(manifest)


def invalidate_lod_profiles(manifest: dict, reason: str) -> int:
    profiles = ensure_lod_profiles(manifest)
    invalidated = 0
    for profile in profiles:
        if not dict(profile or {}).get("result"):
            continue
        profile["stale"] = True
        profile["stale_reason"] = str(reason or "global_bone_pool_changed")
        invalidated += 1
    rebuild_lod_aggregate(manifest)
    return invalidated


def rebuild_lod_aggregate(manifest: dict) -> dict:
    profiles = ensure_lod_profiles(manifest)
    current_generation = str(manifest.get("global_pool_generation", "") or "")
    active = [
        profile
        for profile in profiles
        if bool(dict(profile or {}).get("enabled", True))
        and not lod_profile_is_stale(profile, current_generation)
        and dict(profile or {}).get("result")
    ]

    for field in _LEGACY_CLEAR_FIELDS:
        manifest[field] = []
    manifest["lod_review"] = {}
    manifest["lod_manifest_snapshot"] = {}

    for profile in active:
        profile_id = str(profile.get("profile_id", "") or "")
        lod_level = max(1, int(profile.get("lod_level", 1) or 1))
        result = _normalize_result(dict(profile.get("result", {}) or {}))
        profile["result"] = result
        for field in _RESULT_LIST_FIELDS:
            manifest[field].extend(
                _annotate_payloads(
                    list(result.get(field, []) or []),
                    profile_id=profile_id,
                    lod_level=lod_level,
                    field=field,
                )
            )
        snapshot = _annotate_payload(
            dict(result.get("lod_manifest_snapshot", {}) or {}),
            profile_id=profile_id,
            lod_level=lod_level,
        )
        manifest["lod_manifest_snapshots"].append(snapshot)
        manifest["lod_validation"].extend(
            _annotate_payloads(
                list(result.get("validation", []) or []),
                profile_id=profile_id,
                lod_level=lod_level,
                field="validation",
            )
        )

    manifest["lod_manifest_snapshot"] = _combined_snapshot(manifest["lod_manifest_snapshots"])
    manifest["lod_profile_conflicts"] = _capture_target_conflicts(active)
    manifest["lod_review"] = _aggregate_review(active, manifest["lod_profile_conflicts"])
    return manifest


def active_lod_profiles(manifest: dict) -> list[dict]:
    rebuild_lod_aggregate(manifest)
    current_generation = str(manifest.get("global_pool_generation", "") or "")
    return [
        profile
        for profile in manifest.get("lod_profiles", []) or []
        if bool(dict(profile or {}).get("enabled", True))
        and not lod_profile_is_stale(profile, current_generation)
        and dict(profile or {}).get("result")
    ]


def lod_profile_by_id(manifest: dict, profile_id: str) -> dict | None:
    for profile in ensure_lod_profiles(manifest):
        if str(dict(profile or {}).get("profile_id", "") or "") == str(profile_id or ""):
            return profile
    return None


def lod_profile_is_stale(profile: dict, current_generation: str) -> bool:
    """Return whether a stored analysis is invalid for the current global pool."""

    return bool(dict(profile or {}).get("stale", False)) or not _profile_generation_matches(
        profile,
        str(current_generation or ""),
    )


def first_lod_profile_issue(result: dict) -> str:
    """Return the most important user-facing validation message for a profile."""

    validation = list(dict(result or {}).get("validation", []) or [])
    for severity in ("error", "warning"):
        for issue in validation:
            payload = dict(issue or {})
            if str(payload.get("severity", "") or "").lower() != severity:
                continue
            message = str(payload.get("message", "") or "")
            if message:
                return message
    return ""


def assert_lod_profiles_exportable(manifest: dict) -> None:
    profiles = ensure_lod_profiles(manifest)
    current_generation = str(manifest.get("global_pool_generation", "") or "")
    pending = [
        str(profile.get("label", profile.get("profile_id", "LOD")) or "LOD")
        for profile in profiles
        if bool(dict(profile or {}).get("enabled", True))
        and str(dict(profile or {}).get("frameanalysis_dir", "") or "")
        and not dict(profile or {}).get("result")
    ]
    if pending:
        raise ValueError(
            "Enabled LOD profiles have not been analyzed: "
            + ", ".join(pending)
            + ". Run Analyze Active or disable these profiles."
        )
    stale = [
        str(profile.get("label", profile.get("profile_id", "LOD")) or "LOD")
        for profile in profiles
        if bool(dict(profile or {}).get("enabled", True))
        and dict(profile or {}).get("result")
        and (
            lod_profile_is_stale(profile, current_generation)
        )
    ]
    if stale:
        raise ValueError(
            "LOD profile analysis is stale after the global bone pool changed: "
            + ", ".join(stale)
            + ". Re-run Analyze LOD for these profiles."
        )
    rebuild_lod_aggregate(manifest)
    conflicts = list(manifest.get("lod_profile_conflicts", []) or [])
    if conflicts:
        shown = "; ".join(
            f"{item['ib_hash']}-{item['match_index_count']}-{item['match_first_index']} G{item['canonical_global_bone']}"
            for item in conflicts[:8]
        )
        raise ValueError(
            "Enabled LOD profiles use the same override key but map one canonical bone "
            f"from different source-local bones: {shown}. The runtime cannot distinguish these profiles."
        )


def _has_legacy_lod_result(manifest: dict) -> bool:
    return any(manifest.get(field) for field in (*_RESULT_LIST_FIELDS, "lod_manifest_snapshot"))


def _legacy_result(manifest: dict) -> dict:
    return {
        **{field: copy.deepcopy(list(manifest.get(field, []) or [])) for field in _RESULT_LIST_FIELDS},
        "lod_review": copy.deepcopy(dict(manifest.get("lod_review", {}) or {})),
        "validation": copy.deepcopy(list(manifest.get("lod_validation", []) or [])),
        "lod_manifest_snapshot": copy.deepcopy(dict(manifest.get("lod_manifest_snapshot", {}) or {})),
    }


def _normalize_result(result: dict) -> dict:
    normalized = {
        field: copy.deepcopy(list(result.get(field, []) or []))
        for field in _RESULT_LIST_FIELDS
    }
    normalized["lod_review"] = copy.deepcopy(dict(result.get("lod_review", {}) or {}))
    normalized["validation"] = copy.deepcopy(
        list(result.get("validation", result.get("lod_validation", [])) or [])
    )
    normalized["lod_manifest_snapshot"] = copy.deepcopy(
        dict(result.get("lod_manifest_snapshot", {}) or {})
    )
    return normalized


def _profile_generation_matches(profile: dict, current_generation: str) -> bool:
    profile_generation = str(dict(profile or {}).get("global_pool_generation", "") or "")
    return not current_generation or profile_generation == current_generation


def _annotate_payloads(
    payloads: list[dict],
    *,
    profile_id: str,
    lod_level: int,
    field: str,
) -> list[dict]:
    annotated = []
    for payload in payloads:
        item = _annotate_payload(dict(payload or {}), profile_id=profile_id, lod_level=lod_level)
        if field == "lod_links":
            item["lod_sources"] = [
                _annotate_payload(dict(source or {}), profile_id=profile_id, lod_level=lod_level)
                for source in item.get("lod_sources", []) or []
            ]
        annotated.append(item)
    return annotated


def _annotate_payload(payload: dict, *, profile_id: str, lod_level: int) -> dict:
    payload["lod_profile_id"] = profile_id
    payload["lod_level"] = lod_level
    return payload


def _combined_snapshot(snapshots: list[dict]) -> dict:
    if not snapshots:
        return {}
    if len(snapshots) == 1:
        return copy.deepcopy(dict(snapshots[0] or {}))
    candidates = []
    validation = []
    for snapshot in snapshots:
        profile_id = str(dict(snapshot or {}).get("lod_profile_id", "") or "")
        lod_level = max(1, int(dict(snapshot or {}).get("lod_level", 1) or 1))
        candidates.extend(
            _annotate_payloads(
                list(dict(snapshot or {}).get("candidate_ibs", []) or []),
                profile_id=profile_id,
                lod_level=lod_level,
                field="candidate_ibs",
            )
        )
        validation.extend(
            _annotate_payloads(
                list(dict(snapshot or {}).get("validation", []) or []),
                profile_id=profile_id,
                lod_level=lod_level,
                field="validation",
            )
        )
    return {
        "schema_version": max(int(dict(snapshot or {}).get("schema_version", 1) or 1) for snapshot in snapshots),
        "frameanalysis_dir": "",
        "shadow_stage": {},
        "candidate_ibs": candidates,
        "validation": validation,
        "lod_profile_count": len(snapshots),
    }


def _aggregate_review(active_profiles: list[dict], conflicts: list[dict]) -> dict:
    reviews = [
        dict(dict(profile or {}).get("result", {}).get("lod_review", {}) or {})
        for profile in active_profiles
    ]
    return {
        "runtime_safe": bool(active_profiles)
        and not conflicts
        and all(bool(review.get("runtime_safe", False)) for review in reviews),
        "profile_count": len(active_profiles),
        "missing_global_bone_count": sum(
            int(review.get("missing_global_bone_count", 0) or 0)
            for review in reviews
        ),
        "profile_reviews": [
            {
                "lod_profile_id": str(profile.get("profile_id", "") or ""),
                "lod_level": max(1, int(profile.get("lod_level", 1) or 1)),
                **dict(dict(profile.get("result", {}) or {}).get("lod_review", {}) or {}),
            }
            for profile in active_profiles
        ],
        "profile_conflict_count": len(conflicts),
    }


def _capture_target_conflicts(active_profiles: list[dict]) -> list[dict]:
    target_sources: dict[tuple[tuple[str, int, int], int], dict[str, set[int]]] = {}
    for profile in active_profiles:
        profile_id = str(profile.get("profile_id", "") or "")
        result = dict(profile.get("result", {}) or {})
        for record in result.get("lod_capture_records", []) or []:
            key = _capture_override_key(dict(record or {}))
            if not key[0] or key[2] <= 0:
                continue
            for pair in dict(record or {}).get("scatter_pairs", []) or []:
                canonical_global = int(dict(pair or {}).get("canonical_global_bone", -1))
                source_local = int(dict(pair or {}).get("lod_local_bone", -1))
                if canonical_global < 0 or source_local < 0:
                    continue
                target_sources.setdefault((key, canonical_global), {}).setdefault(profile_id, set()).add(source_local)

    conflicts = []
    for (key, canonical_global), by_profile in sorted(target_sources.items()):
        distinct_sources = {source for sources in by_profile.values() for source in sources}
        if len(by_profile) < 2 or len(distinct_sources) <= 1:
            continue
        conflicts.append(
            {
                "ib_hash": key[0],
                "match_first_index": key[1],
                "match_index_count": key[2],
                "canonical_global_bone": canonical_global,
                "sources_by_profile": {
                    profile_id: sorted(sources)
                    for profile_id, sources in sorted(by_profile.items())
                },
            }
        )
    return conflicts


def _capture_override_key(record: dict) -> tuple[str, int, int]:
    return (
        str(record.get("lod_ib_hash", record.get("ib_hash", "")) or "").lower(),
        int(record.get("lod_match_first_index", record.get("match_first_index", 0)) or 0),
        int(record.get("lod_match_index_count", record.get("match_index_count", 0)) or 0),
    )
