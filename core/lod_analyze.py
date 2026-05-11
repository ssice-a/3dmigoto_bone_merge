"""LOD FrameAnalysis analyzer for canonical global-bone scatter mapping."""

from __future__ import annotations

import math
import os
import json
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone

from ..constants import CAPTURE_MANIFEST_FILE_NAME
from .draw_arrays import require_numpy, skin_signature
from .import_candidates import (
    _load_slot_slice,
    _read_blend_indices,
    _read_blend_weights,
    _read_index_buffer,
    _read_required_float3,
    _resolve_path,
    _skin_slot_from_candidate,
    _validate_skin_format,
)
from .main_analyze import ANALYZER_VERSION, analyze_main_frameanalysis
from .main_analyze import _parse_buffer_header
from .numpy_buffers import positions_diag
from .spatial_index import (
    build_spatial_hash as _generic_build_spatial_hash,
    cell_key as _generic_cell_key,
    neighbor_keys as _generic_neighbor_keys,
)


_WEIGHT_EPSILON = 1.0e-5
_MIN_PAIR_SCORE = 0.001
_TARGET_MATCH_POINT_COUNT = 25000
_MIN_LOD_LINK_COVERAGE_RATIO = 0.15
_MIN_LOD_LINK_GLOBALS = 4
_BONE_SAMPLE_TARGET = 24
_BONE_SAMPLE_TOP_WEIGHT_COUNT = 12
_MIN_BONE_MATCH_SCORE = 0.01
_MIN_BONE_MATCH_VOTES = 1
_LOD_BONE_MATCH_TOLERANCE_SCALES = (2.0, 4.0)
_LOD_BONE_MATCH_FALLBACK_RATIO = 0.95
_LOD_SLOT_DIRECT_ABS_TOLERANCE = 4
_LOD_SLOT_DIRECT_RATIO_TOLERANCE = 0.20
_LOD_SLOT_NEAR_RATIO_TOLERANCE = 0.35
_LOD_SHADOW_CHAIN_GAP = 12


@dataclass(frozen=True)
class WeightedPoint:
    position: tuple[float, float, float]
    weights: tuple[tuple[object, float], ...]


@dataclass(frozen=True)
class PointGeometry:
    positions: list[tuple[float, float, float]]
    blend_indices: list[tuple[int, int, int, int]]
    blend_weights: list[tuple[float, float, float, float]]


@dataclass(frozen=True)
class BoneSample:
    position: tuple[float, float, float]
    weight: float


@dataclass(frozen=True)
class LodBoneSample:
    position: tuple[float, float, float]
    weight: float
    lod_record_key: str
    lod_local_bone: int


_POINT_GEOMETRY_CACHE: dict[tuple, PointGeometry] = {}
_FRAMEANALYSIS_MANIFEST_CACHE: dict[tuple, dict] = {}
_POINT_CLOUD_CACHE: dict[tuple, tuple] = {}


def analyze_lod_for_manifest(
    canonical_manifest: dict,
    lod_frameanalysis_dir: str,
    *,
    lod_level: int = 1,
) -> dict:
    """Analyze one LOD FrameAnalysis folder and return manifest fields to merge."""

    normalized_lod_dir = os.path.abspath(lod_frameanalysis_dir)
    if not os.path.exists(os.path.join(normalized_lod_dir, "log.txt")):
        raise ValueError(f"log.txt not found in {normalized_lod_dir}")
    if not canonical_manifest.get("bone_pool_order"):
        raise ValueError("Build Global Bone Pool before Analyze LOD")

    main_points, canonical_global_count, main_records = _build_canonical_point_cloud(canonical_manifest)
    lod_manifest = _analyze_main_frameanalysis_cached(normalized_lod_dir)
    lod_points, lod_records, skipped_main_hash_count = _build_lod_point_cloud(
        lod_manifest,
    )
    if not main_points:
        raise ValueError("No canonical weighted point cloud could be built from the main manifest")
    if not lod_points:
        raise ValueError("No LOD weighted point cloud could be built from the LOD FrameAnalysis")

    ignored_global_bones = _lod_match_excluded_global_bones(canonical_manifest)
    mapping = build_lod_bone_cloud_mapping(
        main_points,
        lod_points,
        canonical_global_count,
        ignored_global_bones=ignored_global_bones,
    )
    global_candidates = _mapping_global_candidates(mapping)
    lod_chains = _build_lod_record_chains(lod_records)
    links = _build_lod_links(main_records, lod_records, global_candidates, lod_chains=lod_chains)
    capture_records = _build_lod_capture_records(
        lod_records,
        mapping["global_to_lod"],
        lod_links=links,
        global_candidates=global_candidates,
    )
    review = review_lod_global_pool_coverage(canonical_manifest, capture_records)
    variant_id = _build_lod_variant_id(normalized_lod_dir, capture_records)
    generated_at = datetime.now(timezone.utc).isoformat()
    validation = list(mapping["validation"])
    validation.extend(review["validation"])
    validation.extend(
        {
            **entry,
            "source": "lod_frameanalysis",
        }
        for entry in lod_manifest.get("validation", []) or []
    )

    return {
        "lod_frameanalysis": [
            {
                "lod_level": int(lod_level),
                "variant_id": variant_id,
                "frameanalysis_dir": normalized_lod_dir,
                "candidate_count": len(lod_manifest.get("candidate_ibs", []) or []),
                "capture_candidate_count": len(lod_records),
                "matched_global_bone_count": len(mapping["global_to_lod"]),
                "total_global_bone_count": int(canonical_global_count),
                "required_global_bone_count": int(mapping["required_global_bone_count"]),
                "ignored_lod_global_bone_count": int(mapping["ignored_global_bone_count"]),
                "match_tolerance": float(mapping["match_tolerance"]),
                "matched_vertex_count": int(mapping["matched_vertex_count"]),
                "lod_vertex_count": int(len(lod_points)),
                "lod_chain_count": int(len(lod_chains)),
                "skipped_main_hash_lod_candidate_count": int(skipped_main_hash_count),
                "canonical_match_point_count": int(mapping["canonical_match_point_count"]),
                "lod_match_point_count": int(mapping["lod_match_point_count"]),
                "shadow_stage": dict(lod_manifest.get("shadow_stage", {}) or {}),
                "generated_at": generated_at,
            }
        ],
        "lod_links": links,
        "lod_chains": lod_chains,
        "lod_capture_records": capture_records,
        "lod_mapping": mapping["mapping_entries"],
        "lod_review": review,
        "validation": validation,
        "lod_manifest_snapshot": _lod_manifest_snapshot(lod_manifest),
    }


def build_lod_scatter_mapping(
    canonical_points: list[WeightedPoint],
    lod_points: list[WeightedPoint],
    canonical_global_count: int,
    *,
    ignored_global_bones: set[int] | None = None,
) -> dict:
    """Vote canonical global bones to LOD local bones by nearest weighted vertices."""

    ignored_global_bones = {
        int(value)
        for value in (ignored_global_bones or set())
        if 0 <= int(value) < int(canonical_global_count)
    }
    tolerance = _initial_match_tolerance(canonical_points, lod_points)
    canonical_points = _compress_point_cloud(canonical_points, _TARGET_MATCH_POINT_COUNT, tolerance)
    lod_points = _compress_point_cloud(lod_points, _TARGET_MATCH_POINT_COUNT, tolerance)
    best_stats = {}
    best_matched_count = 0
    best_tolerance = tolerance
    for scale in _LOD_BONE_MATCH_TOLERANCE_SCALES:
        trial_tolerance = tolerance * scale
        stats, matched_count = _vote_lod_to_global(canonical_points, lod_points, trial_tolerance)
        if matched_count > best_matched_count or (matched_count == best_matched_count and len(stats) > len(best_stats)):
            best_stats = stats
            best_matched_count = matched_count
            best_tolerance = trial_tolerance
        if matched_count >= max(4, int(len(lod_points) * 0.2)):
            break

    candidates_by_global: dict[int, list[dict]] = {}
    best_by_global: dict[int, tuple[tuple[str, int], dict]] = {}
    for (lod_key, lod_local_bone, canonical_global), stats in best_stats.items():
        if int(canonical_global) < 0 or int(canonical_global) >= int(canonical_global_count):
            continue
        if float(stats["score"]) < _MIN_PAIR_SCORE:
            continue
        candidate = _lod_mapping_entry(canonical_global, lod_key, lod_local_bone, stats)
        candidates_by_global.setdefault(int(canonical_global), []).append(candidate)
        current = best_by_global.get(int(canonical_global))
        if current is None or _pair_rank(stats) > _pair_rank(current[1]):
            best_by_global[int(canonical_global)] = ((str(lod_key), int(lod_local_bone)), stats)

    mapping_entries = []
    global_to_lod: dict[int, dict] = {}
    for canonical_global in range(int(canonical_global_count)):
        if canonical_global in ignored_global_bones:
            mapping_entries.append(
                {
                    "canonical_global_bone": int(canonical_global),
                    "lod_record_key": "",
                    "lod_local_bone": -1,
                    "score": 0.0,
                    "votes": 0,
                    "status": "ignored_lod_match_excluded",
                }
            )
            continue
        selected = best_by_global.get(canonical_global)
        if selected is None:
            mapping_entries.append(
                {
                    "canonical_global_bone": int(canonical_global),
                    "lod_record_key": "",
                    "lod_local_bone": -1,
                    "score": 0.0,
                    "votes": 0,
                    "status": "unmatched",
                }
            )
            continue
        (lod_key, lod_local_bone), stats = selected
        entry = _lod_mapping_entry(canonical_global, lod_key, lod_local_bone, stats)
        mapping_entries.append(entry)
        global_to_lod[int(canonical_global)] = entry

    for canonical_global, candidates in list(candidates_by_global.items()):
        candidates_by_global[int(canonical_global)] = sorted(
            candidates,
            key=lambda entry: (
                -float(entry.get("score", 0.0)),
                -int(entry.get("votes", 0)),
                float(entry.get("average_distance", 0.0)),
                str(entry.get("lod_record_key", "")),
                int(entry.get("lod_local_bone", -1)),
            ),
        )

    unmatched_count = sum(1 for entry in mapping_entries if str(entry["status"]) == "unmatched")
    required_global_count = max(0, int(canonical_global_count) - len(ignored_global_bones))
    validation = []
    if unmatched_count:
        validation.append(
            {
                "severity": "warning",
                "code": "lod_unmatched_global_bones",
                "message": f"LOD mapping left {unmatched_count}/{required_global_count} required canonical global bone(s) unmatched.",
                "draw_indices": [],
            }
        )
    if best_matched_count <= 0:
        validation.append(
            {
                "severity": "error",
                "code": "lod_no_vertex_matches",
                "message": "LOD point cloud did not match any canonical vertices within the tested tolerances.",
                "draw_indices": [],
            }
        )

    return {
        "global_to_lod": global_to_lod,
        "global_candidates": candidates_by_global,
        "mapping_entries": mapping_entries,
        "validation": validation,
        "matched_vertex_count": best_matched_count,
        "match_tolerance": best_tolerance,
        "canonical_match_point_count": len(canonical_points),
        "lod_match_point_count": len(lod_points),
        "ignored_global_bone_count": len(ignored_global_bones),
        "required_global_bone_count": required_global_count,
    }


def build_lod_bone_cloud_mapping(
    canonical_points: list[WeightedPoint],
    lod_points: list[WeightedPoint],
    canonical_global_count: int,
    *,
    ignored_global_bones: set[int] | None = None,
) -> dict:
    """Match canonical global bones to LOD local bones using per-bone weighted point clouds."""

    ignored_global_bones = {
        int(value)
        for value in (ignored_global_bones or set())
        if 0 <= int(value) < int(canonical_global_count)
    }
    tolerance = _initial_match_tolerance(canonical_points, lod_points)
    canonical_clouds = _sample_canonical_bone_clouds(canonical_points, ignored_global_bones)
    lod_clouds = _sample_lod_bone_clouds(lod_points)

    best_stats: dict[tuple[str, int, int], dict] = {}
    best_matched_count = 0
    best_tolerance = tolerance
    for scale in (1.0, 2.0, 4.0):
        trial_tolerance = tolerance * scale
        stats = _score_bone_clouds(canonical_clouds, lod_clouds, trial_tolerance)
        matched_count = _count_matched_globals_from_stats(stats, canonical_global_count, ignored_global_bones)
        if matched_count > best_matched_count or (matched_count == best_matched_count and len(stats) > len(best_stats)):
            best_stats = stats
            best_matched_count = matched_count
            best_tolerance = trial_tolerance
        if matched_count >= max(1, int((canonical_global_count - len(ignored_global_bones)) * _LOD_BONE_MATCH_FALLBACK_RATIO)):
            break

    candidates_by_global: dict[int, list[dict]] = {}
    best_by_global: dict[int, tuple[tuple[str, int], dict]] = {}
    for (lod_key, lod_local_bone, canonical_global), stats in best_stats.items():
        if int(canonical_global) < 0 or int(canonical_global) >= int(canonical_global_count):
            continue
        if canonical_global in ignored_global_bones:
            continue
        if not _bone_match_stats_accepted(stats):
            continue
        candidate = _lod_mapping_entry(canonical_global, lod_key, lod_local_bone, stats)
        candidates_by_global.setdefault(int(canonical_global), []).append(candidate)
        current = best_by_global.get(int(canonical_global))
        if current is None or _pair_rank(stats) > _pair_rank(current[1]):
            best_by_global[int(canonical_global)] = ((str(lod_key), int(lod_local_bone)), stats)

    mapping_entries = []
    global_to_lod: dict[int, dict] = {}
    for canonical_global in range(int(canonical_global_count)):
        if canonical_global in ignored_global_bones:
            mapping_entries.append(
                {
                    "canonical_global_bone": int(canonical_global),
                    "lod_record_key": "",
                    "lod_local_bone": -1,
                    "score": 0.0,
                    "votes": 0,
                    "status": "ignored_lod_match_excluded",
                }
            )
            continue
        selected = best_by_global.get(canonical_global)
        if selected is None:
            mapping_entries.append(
                {
                    "canonical_global_bone": int(canonical_global),
                    "lod_record_key": "",
                    "lod_local_bone": -1,
                    "score": 0.0,
                    "votes": 0,
                    "status": "unmatched",
                }
            )
            continue
        (lod_key, lod_local_bone), stats = selected
        entry = _lod_mapping_entry(canonical_global, lod_key, lod_local_bone, stats)
        mapping_entries.append(entry)
        global_to_lod[int(canonical_global)] = entry

    for canonical_global, candidates in list(candidates_by_global.items()):
        candidates_by_global[int(canonical_global)] = sorted(
            candidates,
            key=lambda entry: (
                -float(entry.get("score", 0.0)),
                -int(entry.get("votes", 0)),
                float(entry.get("average_distance", 0.0)),
                str(entry.get("lod_record_key", "")),
                int(entry.get("lod_local_bone", -1)),
            ),
        )

    unmatched_count = sum(1 for entry in mapping_entries if str(entry["status"]) == "unmatched")
    required_global_count = max(0, int(canonical_global_count) - len(ignored_global_bones))
    validation = []
    if unmatched_count:
        validation.append(
            {
                "severity": "warning",
                "code": "lod_unmatched_global_bones",
                "message": f"LOD mapping left {unmatched_count}/{required_global_count} required canonical global bone(s) unmatched.",
                "draw_indices": [],
            }
        )
    if best_matched_count <= 0:
        validation.append(
            {
                "severity": "error",
                "code": "lod_no_vertex_matches",
                "message": "LOD point cloud did not match any canonical vertices within the tested tolerances.",
                "draw_indices": [],
            }
        )

    return {
        "global_to_lod": global_to_lod,
        "global_candidates": candidates_by_global,
        "mapping_entries": mapping_entries,
        "validation": validation,
        "matched_vertex_count": best_matched_count,
        "match_tolerance": best_tolerance,
        "canonical_match_point_count": sum(len(samples) for samples in canonical_clouds.values()),
        "lod_match_point_count": sum(len(samples) for samples in lod_clouds.values()),
        "ignored_global_bone_count": len(ignored_global_bones),
        "required_global_bone_count": required_global_count,
    }


def _sample_canonical_bone_clouds(
    points: list[WeightedPoint],
    ignored_global_bones: set[int],
) -> dict[int, list[BoneSample]]:
    clouds: dict[int, list[BoneSample]] = {}
    for point in points:
        for key, weight in point.weights:
            try:
                global_bone = int(key)
            except (TypeError, ValueError):
                continue
            if global_bone in ignored_global_bones:
                continue
            if float(weight) <= _WEIGHT_EPSILON:
                continue
            clouds.setdefault(global_bone, []).append(BoneSample(point.position, float(weight)))
    return {
        global_bone: _sample_bone_cloud(samples)
        for global_bone, samples in clouds.items()
        if samples
    }


def _sample_lod_bone_clouds(points: list[WeightedPoint]) -> dict[tuple[str, int], list[LodBoneSample]]:
    clouds: dict[tuple[str, int], list[BoneSample]] = {}
    for point in points:
        for key, weight in point.weights:
            parsed = _parse_lod_bone_key(key)
            if parsed is None or float(weight) <= _WEIGHT_EPSILON:
                continue
            clouds.setdefault(parsed, []).append(BoneSample(point.position, float(weight)))

    sampled_clouds: dict[tuple[str, int], list[LodBoneSample]] = {}
    for (lod_record_key, lod_local_bone), samples in clouds.items():
        sampled_clouds[(lod_record_key, lod_local_bone)] = [
            LodBoneSample(sample.position, sample.weight, lod_record_key, lod_local_bone)
            for sample in _sample_bone_cloud(samples)
        ]
    return sampled_clouds


def _parse_lod_bone_key(key) -> tuple[str, int] | None:
    if not isinstance(key, (tuple, list)) or len(key) != 2:
        return None
    lod_record_key, lod_local_bone = key
    try:
        return str(lod_record_key), int(lod_local_bone)
    except (TypeError, ValueError):
        return None


def _sample_bone_cloud(samples: list[BoneSample]) -> list[BoneSample]:
    if len(samples) <= _BONE_SAMPLE_TARGET:
        return list(samples)

    selected: list[BoneSample] = []
    selected_keys: set[tuple[tuple[float, float, float], int]] = set()

    def add(sample: BoneSample) -> None:
        key = (sample.position, round(float(sample.weight), 8))
        if key in selected_keys or len(selected) >= _BONE_SAMPLE_TARGET:
            return
        selected_keys.add(key)
        selected.append(sample)

    by_weight = sorted(samples, key=lambda item: (-float(item.weight), item.position))
    for sample in by_weight[: min(_BONE_SAMPLE_TOP_WEIGHT_COUNT, _BONE_SAMPLE_TARGET)]:
        add(sample)

    if len(selected) < _BONE_SAMPLE_TARGET:
        cell_size = max(_positions_diag([sample.position for sample in samples]) * 0.08, 1.0e-5)
        buckets: dict[tuple[int, int, int], BoneSample] = {}
        for sample in samples:
            key = _cell_key(sample.position, cell_size)
            current = buckets.get(key)
            if current is None or float(sample.weight) > float(current.weight):
                buckets[key] = sample
        for sample in sorted(buckets.values(), key=lambda item: (-float(item.weight), item.position)):
            add(sample)
            if len(selected) >= _BONE_SAMPLE_TARGET:
                break

    for sample in by_weight:
        add(sample)
        if len(selected) >= _BONE_SAMPLE_TARGET:
            break
    return selected


def _score_bone_clouds(
    canonical_clouds: dict[int, list[BoneSample]],
    lod_clouds: dict[tuple[str, int], list[LodBoneSample]],
    tolerance: float,
) -> dict[tuple[str, int, int], dict]:
    lod_samples = [sample for samples in lod_clouds.values() for sample in samples]
    lod_hash = _generic_build_spatial_hash(lod_samples, lambda sample: sample.position, tolerance)
    tolerance_squared = tolerance * tolerance
    inverse_tolerance = 1.0 / max(float(tolerance), 1.0e-6)
    lod_hash_get = lod_hash.get
    min_pair_score = _MIN_PAIR_SCORE
    stats: dict[tuple[str, int, int], dict] = {}

    for canonical_global, canonical_samples in canonical_clouds.items():
        for canonical_sample in canonical_samples:
            best_by_lod_bone: dict[tuple[str, int], tuple[float, float, float]] = {}
            base_key = _cell_key(canonical_sample.position, tolerance)
            canonical_x, canonical_y, canonical_z = canonical_sample.position
            canonical_weight = float(canonical_sample.weight)
            for cell_key in _neighbor_keys(base_key):
                for lod_sample in lod_hash_get(cell_key, ()):
                    lod_x, lod_y, lod_z = lod_sample.position
                    dx = canonical_x - lod_x
                    dy = canonical_y - lod_y
                    dz = canonical_z - lod_z
                    distance_squared = dx * dx + dy * dy + dz * dz
                    if distance_squared > tolerance_squared:
                        continue
                    distance = math.sqrt(distance_squared)
                    distance_score = 1.0 / (1.0 + distance * inverse_tolerance)
                    score = canonical_weight * float(lod_sample.weight) * distance_score
                    if score <= min_pair_score:
                        continue
                    lod_key = (lod_sample.lod_record_key, lod_sample.lod_local_bone)
                    current = best_by_lod_bone.get(lod_key)
                    if current is None or score > current[0]:
                        best_by_lod_bone[lod_key] = (score, distance, float(lod_sample.weight))
            for (lod_record_key, lod_local_bone), (score, distance, _lod_weight) in best_by_lod_bone.items():
                key = (str(lod_record_key), int(lod_local_bone), int(canonical_global))
                record = stats.setdefault(key, {"score": 0.0, "votes": 0, "distance_sum": 0.0})
                record["score"] += float(score)
                record["votes"] += 1
                record["distance_sum"] += float(distance)
    return stats


def _count_matched_globals_from_stats(
    stats: dict[tuple[str, int, int], dict],
    canonical_global_count: int,
    ignored_global_bones: set[int],
) -> int:
    matched = set()
    for (_lod_key, _lod_local_bone, canonical_global), record in stats.items():
        if int(canonical_global) < 0 or int(canonical_global) >= int(canonical_global_count):
            continue
        if int(canonical_global) in ignored_global_bones:
            continue
        if _bone_match_stats_accepted(record):
            matched.add(int(canonical_global))
    return len(matched)


def _bone_match_stats_accepted(stats: dict) -> bool:
    return float(stats.get("score", 0.0)) >= _MIN_BONE_MATCH_SCORE and int(stats.get("votes", 0)) >= _MIN_BONE_MATCH_VOTES


def _lod_mapping_entry(canonical_global: int, lod_key: str, lod_local_bone: int, stats: dict) -> dict:
    votes = max(1, int(stats.get("votes", 0)))
    return {
        "canonical_global_bone": int(canonical_global),
        "lod_record_key": str(lod_key),
        "lod_local_bone": int(lod_local_bone),
        "score": float(stats.get("score", 0.0)),
        "votes": int(stats.get("votes", 0)),
        "average_distance": float(stats.get("distance_sum", 0.0)) / votes,
        "status": "matched",
    }


def _analyze_main_frameanalysis_cached(frameanalysis_dir: str) -> dict:
    normalized_dir = os.path.abspath(frameanalysis_dir)
    log_path = os.path.join(normalized_dir, "log.txt")
    manifest_path = os.path.join(normalized_dir, CAPTURE_MANIFEST_FILE_NAME)
    cache_key = (normalized_dir, _file_fingerprint(log_path), _file_fingerprint(manifest_path))
    cached = _FRAMEANALYSIS_MANIFEST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    manifest = _read_existing_frameanalysis_manifest(normalized_dir, log_path, manifest_path)
    if manifest is None:
        manifest = analyze_main_frameanalysis(normalized_dir)
    _FRAMEANALYSIS_MANIFEST_CACHE.clear()
    _FRAMEANALYSIS_MANIFEST_CACHE[cache_key] = manifest
    return manifest


def _read_existing_frameanalysis_manifest(normalized_dir: str, log_path: str, manifest_path: str) -> dict | None:
    if not os.path.exists(manifest_path):
        return None
    try:
        manifest_stat = os.stat(manifest_path)
        log_stat = os.stat(log_path)
    except OSError:
        return None
    if int(manifest_stat.st_mtime_ns) < int(log_stat.st_mtime_ns):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("candidate_ibs"), list) or not isinstance(payload.get("shadow_stage"), dict):
        return None
    if int(payload.get("analyzer_version", 0) or 0) < ANALYZER_VERSION:
        return None
    payload_dir = str(payload.get("frameanalysis_dir", "") or "")
    if payload_dir and os.path.abspath(payload_dir) != normalized_dir:
        return None
    payload["frameanalysis_dir"] = normalized_dir
    return payload


def review_lod_global_pool_coverage(canonical_manifest: dict, lod_capture_records: list[dict]) -> dict:
    ignored_global_bones = _lod_match_excluded_global_bones(canonical_manifest)
    filled_globals = {
        int(pair.get("canonical_global_bone", -1))
        for record in lod_capture_records
        for pair in record.get("scatter_pairs", []) or []
        if int(pair.get("canonical_global_bone", -1)) >= 0
    } - ignored_global_bones
    total_globals = 0
    missing_by_record = []
    ignored_by_record = []
    missing_capture_ready_count = 0
    missing_mapping_only_count = 0
    for record in canonical_manifest.get("bone_pool_order", []) or []:
        global_base = int(record.get("global_bone_base", record.get("capture_store_base", 0)) or 0)
        local_bone_count = int(record.get("local_bone_count", 0) or 0)
        capture_available = bool(record.get("bone_capture_available", record.get("shadow_capture_ready", False)))
        lod_match_excluded = bool(record.get("lod_match_excluded", False))
        source_key = _source_key_from_candidate(record)
        total_globals = max(total_globals, global_base + local_bone_count)
        if lod_match_excluded:
            ignored_by_record.append(
                {
                    "source_key": source_key,
                    "ib_hash": str(record.get("ib_hash", "") or "").lower(),
                    "match_first_index": int(record.get("match_first_index", 0) or 0),
                    "match_index_count": int(record.get("match_index_count", 0) or 0),
                    "global_bone_base": int(global_base),
                    "local_bone_count": int(local_bone_count),
                    "ignored_count": int(local_bone_count),
                    "reason": str(record.get("lod_match_excluded_reason", "") or "lod_match_excluded"),
                }
            )
            continue
        missing_globals = [
            global_bone
            for global_bone in range(global_base, global_base + local_bone_count)
            if global_bone not in filled_globals
        ]
        if not missing_globals:
            continue
        if capture_available:
            missing_capture_ready_count += len(missing_globals)
        else:
            missing_mapping_only_count += len(missing_globals)
        missing_by_record.append(
            {
                "source_key": source_key,
                "ib_hash": str(record.get("ib_hash", "") or "").lower(),
                "match_first_index": int(record.get("match_first_index", 0) or 0),
                "match_index_count": int(record.get("match_index_count", 0) or 0),
                "global_bone_base": int(global_base),
                "local_bone_count": int(local_bone_count),
                "bone_capture_available": capture_available,
                "missing_count": len(missing_globals),
                "missing_global_bones": missing_globals,
                "missing_local_bones": [global_bone - global_base for global_bone in missing_globals],
                "status": "blocking_capture_ready_missing" if capture_available else "mapping_only_missing",
            }
        )

    missing_total = sum(int(record["missing_count"]) for record in missing_by_record)
    runtime_safe = missing_total == 0
    validation = []
    if missing_total:
        required_total = max(0, total_globals - len(ignored_global_bones))
        validation.append(
            {
                "severity": "error",
                "code": "lod_global_pool_not_fully_filled",
                "message": (
                    f"LOD scatter fills {len(filled_globals)}/{required_total} required canonical global bones; "
                    f"{missing_total} global bone(s) are missing."
                ),
                "draw_indices": [],
            }
        )
    if missing_capture_ready_count:
        validation.append(
            {
                "severity": "error",
                "code": "lod_capture_ready_sources_missing",
                "message": f"LOD scatter misses {missing_capture_ready_count} bone(s) from capture-ready global-pool records.",
                "draw_indices": [],
            }
        )

    return {
        "runtime_safe": runtime_safe,
        "filled_global_bone_count": len(filled_globals),
        "total_global_bone_count": int(total_globals),
        "required_global_bone_count": int(max(0, total_globals - len(ignored_global_bones))),
        "ignored_lod_global_bone_count": int(len(ignored_global_bones)),
        "missing_global_bone_count": int(missing_total),
        "missing_capture_ready_count": int(missing_capture_ready_count),
        "missing_mapping_only_count": int(missing_mapping_only_count),
        "missing_by_record": missing_by_record,
        "ignored_by_record": ignored_by_record,
        "validation": validation,
    }


def _lod_match_excluded_global_bones(canonical_manifest: dict) -> set[int]:
    ignored: set[int] = set()
    for record in canonical_manifest.get("bone_pool_order", []) or []:
        if not _record_lod_match_excluded(record):
            continue
        global_base = int(record.get("global_bone_base", record.get("capture_store_base", 0)) or 0)
        local_bone_count = int(record.get("local_bone_count", 0) or 0)
        ignored.update(range(global_base, global_base + max(0, local_bone_count)))
    return ignored


def _record_lod_match_excluded(record: dict) -> bool:
    return bool(record.get("lod_match_excluded", False))


def _compress_point_cloud(points: list[WeightedPoint], target_count: int, tolerance: float) -> list[WeightedPoint]:
    if len(points) <= int(target_count):
        return points
    cell_size = max(float(tolerance) * 0.5, _point_cloud_diag(points) * 0.006, 1.0e-5)
    compressed = points
    for _iteration in range(6):
        compressed = _aggregate_points_by_cell(points, cell_size)
        if len(compressed) <= int(target_count):
            return compressed
        cell_size *= 1.35
    return compressed


def _aggregate_points_by_cell(points: list[WeightedPoint], cell_size: float) -> list[WeightedPoint]:
    buckets: dict[tuple[int, int, int, object], dict] = {}
    np = require_numpy()
    cell_keys = None
    if points:
        positions = np.asarray([point.position for point in points], dtype=np.float64)
        cell_keys = np.floor(positions / max(float(cell_size), 1.0e-6)).astype(np.int64)
    for point_index, point in enumerate(points):
        dominant_key = point.weights[0][0] if point.weights else None
        if cell_keys is not None:
            cell = cell_keys[point_index]
            bucket_key = (int(cell[0]), int(cell[1]), int(cell[2]), dominant_key)
        else:
            bucket_key = (*_cell_key(point.position, cell_size), dominant_key)
        bucket = buckets.get(bucket_key)
        if bucket is None:
            bucket = {
                "position_sum": [0.0, 0.0, 0.0],
                "count": 0,
                "weights": {},
            }
            buckets[bucket_key] = bucket
        bucket["position_sum"][0] += float(point.position[0])
        bucket["position_sum"][1] += float(point.position[1])
        bucket["position_sum"][2] += float(point.position[2])
        bucket["count"] += 1
        weight_map = bucket["weights"]
        for weight_key, weight in point.weights:
            weight_map[weight_key] = weight_map.get(weight_key, 0.0) + float(weight)

    compressed: list[WeightedPoint] = []
    for bucket in buckets.values():
        count = max(1, int(bucket["count"]))
        normalized_weights = _normalize_weights(list(bucket["weights"].items()))
        if not normalized_weights:
            continue
        compressed.append(
            WeightedPoint(
                position=(
                    float(bucket["position_sum"][0]) / count,
                    float(bucket["position_sum"][1]) / count,
                    float(bucket["position_sum"][2]) / count,
                ),
                weights=tuple(normalized_weights),
            )
        )
    return compressed


def _build_canonical_point_cloud(canonical_manifest: dict) -> tuple[list[WeightedPoint], int, list[dict]]:
    cache_key = _canonical_point_cloud_cache_key(canonical_manifest)
    cached = _POINT_CLOUD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    frameanalysis_dir = str(canonical_manifest.get("frameanalysis_dir", "") or "")
    candidates_by_key = {
        _source_key_from_candidate(candidate): candidate
        for candidate in canonical_manifest.get("candidate_ibs", []) or []
    }
    remaps_by_key = _object_remaps_by_key(canonical_manifest)
    points: list[WeightedPoint] = []
    main_records: list[dict] = []
    canonical_global_count = 0
    for record in canonical_manifest.get("bone_pool_order", []) or []:
        source_key = _source_key_from_candidate(record)
        candidate = candidates_by_key.get(source_key)
        remap = remaps_by_key.get(source_key) or _object_remap_from_pool_record(record)
        global_base = int(record.get("global_bone_base", record.get("capture_store_base", 0)) or 0)
        local_count = int(record.get("local_bone_count", 0) or 0)
        canonical_global_count = max(canonical_global_count, global_base + local_count)
        if _record_lod_match_excluded(record):
            continue
        if candidate is None or not remap:
            continue
        geometry = _load_candidate_point_geometry(candidate, frameanalysis_dir)
        vb2_signature = _vb2_signature_from_geometry(geometry, record.get("used_local_bone_indices"))
        points.extend(_geometry_points(geometry, _canonical_weight_resolver(remap)))
        main_records.append(
            {
                "source_key": source_key,
                "ib_hash": str(record.get("ib_hash", "") or "").lower(),
                "match_first_index": int(record.get("match_first_index", 0) or 0),
                "match_index_count": int(record.get("match_index_count", 0) or 0),
                "global_bone_base": global_base,
                "local_bone_count": local_count,
                "used_local_bone_indices": [int(value) for value in record.get("used_local_bone_indices", []) or []],
                "vb2_signature": vb2_signature,
            }
        )
    result = (points, canonical_global_count, main_records)
    _POINT_CLOUD_CACHE[cache_key] = result
    return result


def _build_lod_point_cloud(
    lod_manifest: dict,
    *,
    excluded_ib_hashes: set[str] | None = None,
) -> tuple[list[WeightedPoint], dict[str, dict], int]:
    normalized_excluded_ib_hashes = {str(value).lower() for value in (excluded_ib_hashes or set()) if str(value)}
    cache_key = _lod_point_cloud_cache_key(lod_manifest, normalized_excluded_ib_hashes)
    cached = _POINT_CLOUD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    frameanalysis_dir = str(lod_manifest.get("frameanalysis_dir", "") or "")
    points: list[WeightedPoint] = []
    lod_records: dict[str, dict] = {}
    skipped_main_hash_count = 0
    for candidate in lod_manifest.get("candidate_ibs", []) or []:
        if not bool(candidate.get("enabled", True)):
            continue
        if not bool(candidate.get("shadow_capture_ready", False)):
            continue
        if bool(candidate.get("lod_match_excluded", False)):
            continue
        if int(candidate.get("local_bone_count", 0) or 0) <= 0:
            continue
        ib_hash = str(candidate.get("ib_hash", "") or "").lower()
        if ib_hash in normalized_excluded_ib_hashes:
            skipped_main_hash_count += 1
            continue
        source_key = _source_key_from_candidate(candidate)
        geometry = _load_candidate_point_geometry(candidate, frameanalysis_dir)
        vb2_signature = _vb2_signature_from_geometry(geometry, candidate.get("used_local_bone_indices"))
        points.extend(_geometry_points(geometry, _lod_weight_resolver(source_key)))
        lod_records[source_key] = _lod_record_payload(candidate, source_key, vb2_signature=vb2_signature)
    result = (points, lod_records, skipped_main_hash_count)
    _POINT_CLOUD_CACHE[cache_key] = result
    return result


def _canonical_point_cloud_cache_key(canonical_manifest: dict) -> tuple:
    frameanalysis_dir = str(canonical_manifest.get("frameanalysis_dir", "") or "")
    candidates_by_key = {
        _source_key_from_candidate(candidate): candidate
        for candidate in canonical_manifest.get("candidate_ibs", []) or []
    }
    remaps_by_key = _object_remaps_by_key(canonical_manifest)
    records = []
    for record in canonical_manifest.get("bone_pool_order", []) or []:
        source_key = _source_key_from_candidate(record)
        candidate = candidates_by_key.get(source_key)
        remap = remaps_by_key.get(source_key) or _object_remap_from_pool_record(record)
        local_to_global = tuple(
            sorted(
                (int(local), int(global_bone))
                for local, global_bone in dict(remap.get("local_group_to_global_group", {}) or {}).items()
            )
        )
        records.append(
            (
                source_key,
                int(record.get("global_bone_base", record.get("capture_store_base", 0)) or 0),
                int(record.get("local_bone_count", 0) or 0),
                bool(_record_lod_match_excluded(record)),
                local_to_global,
                _candidate_file_cache_key(candidate, frameanalysis_dir) if candidate is not None else (),
            )
        )
    return (
        "canonical",
        frameanalysis_dir,
        str(canonical_manifest.get("global_pool_generation", "") or ""),
        tuple(records),
    )


def _lod_point_cloud_cache_key(lod_manifest: dict, excluded_ib_hashes: set[str] | None = None) -> tuple:
    frameanalysis_dir = str(lod_manifest.get("frameanalysis_dir", "") or "")
    normalized_excluded_ib_hashes = {str(value).lower() for value in (excluded_ib_hashes or set()) if str(value)}
    records = []
    for candidate in lod_manifest.get("candidate_ibs", []) or []:
        if not bool(candidate.get("enabled", True)):
            continue
        if not bool(candidate.get("shadow_capture_ready", False)):
            continue
        if bool(candidate.get("lod_match_excluded", False)):
            continue
        if int(candidate.get("local_bone_count", 0) or 0) <= 0:
            continue
        ib_hash = str(candidate.get("ib_hash", "") or "").lower()
        if ib_hash in normalized_excluded_ib_hashes:
            continue
        source_key = _source_key_from_candidate(candidate)
        records.append(
            (
                source_key,
                int(candidate.get("local_bone_count", 0) or 0),
                _candidate_file_cache_key(candidate, frameanalysis_dir),
            )
        )
    return ("lod", frameanalysis_dir, tuple(sorted(normalized_excluded_ib_hashes)), tuple(records))


def _candidate_file_cache_key(candidate: dict, frameanalysis_dir: str) -> tuple:
    import_paths = dict(candidate.get("import_paths", {}) or {})
    vb_payload = dict(import_paths.get("vb", {}) or {})
    ib_txt_path = _resolve_path(str(import_paths.get("ib", "") or ""), frameanalysis_dir)
    ib_buf_path = _resolve_path(str(import_paths.get("ib_buf", "") or ""), frameanalysis_dir)
    skin_slot_name, skin_slot_index = _skin_slot_from_candidate(candidate)
    skin_format = dict(candidate.get("skin_format", {}) or {})
    paths = [
        ib_txt_path,
        ib_buf_path,
        *_resolved_payload_paths(dict(vb_payload.get("vb0", {}) or {}), frameanalysis_dir),
        *_resolved_payload_paths(dict(vb_payload.get(skin_slot_name, {}) or {}), frameanalysis_dir),
    ]
    return (
        str(skin_slot_name),
        int(skin_slot_index),
        str(skin_format.get("blend_indices_format", "") or "").upper(),
        str(skin_format.get("blend_weights_format", "") or "").upper(),
        int(skin_format.get("blend_indices_offset", -1) or -1),
        int(skin_format.get("blend_weights_offset", -1) or -1),
        tuple(_file_fingerprint(path) for path in paths if path),
    )


def _load_candidate_point_geometry(candidate: dict, frameanalysis_dir: str) -> PointGeometry:
    import_paths = dict(candidate.get("import_paths", {}) or {})
    ib_txt_path = _resolve_path(str(import_paths.get("ib", "") or ""), frameanalysis_dir)
    ib_buf_path = _resolve_path(str(import_paths.get("ib_buf", "") or ""), frameanalysis_dir)
    if not ib_txt_path or not os.path.exists(ib_txt_path):
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no IB txt path")
    if not ib_buf_path or not os.path.exists(ib_buf_path):
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no IB buf path")

    cache_key = _point_geometry_cache_key(candidate, frameanalysis_dir, ib_txt_path, ib_buf_path)
    cached = _POINT_GEOMETRY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ib_header = _parse_buffer_header(ib_txt_path)
    indices = _read_index_buffer(ib_buf_path, ib_header)
    original_vertex_ids = sorted({int(vertex_index) for vertex_index in indices})

    vb_payload = dict(import_paths.get("vb", {}) or {})
    warnings: list[str] = []
    vb0 = _load_slot_slice(vb_payload, "vb0", 0, frameanalysis_dir, warnings)
    if vb0 is None:
        raise ValueError(f"Candidate {candidate.get('display_name') or candidate.get('ib_hash')} has no vb0 slice")
    positions = _read_required_float3(vb0, "POSITION", 0, original_vertex_ids)

    skin_slot_name, skin_slot_index = _skin_slot_from_candidate(candidate)
    skin_slot = _load_slot_slice(vb_payload, skin_slot_name, skin_slot_index, frameanalysis_dir, warnings)
    if skin_slot is None:
        geometry = PointGeometry(
            positions=positions,
            blend_indices=[(0, 0, 0, 0) for _ in original_vertex_ids],
            blend_weights=[(0.0, 0.0, 0.0, 0.0) for _ in original_vertex_ids],
        )
        _POINT_GEOMETRY_CACHE[cache_key] = geometry
        return geometry
    _validate_skin_format(skin_slot, dict(candidate.get("skin_format", {}) or {}), warnings)
    geometry = PointGeometry(
        positions=positions,
        blend_indices=_read_blend_indices(skin_slot, original_vertex_ids),
        blend_weights=_read_blend_weights(skin_slot, original_vertex_ids),
    )
    _POINT_GEOMETRY_CACHE[cache_key] = geometry
    return geometry


def _vb2_signature_from_geometry(geometry: PointGeometry, declared_used_slots=None) -> dict:
    return skin_signature(
        geometry.positions,
        geometry.blend_indices,
        geometry.blend_weights,
        declared_used_slots=declared_used_slots,
        epsilon=_WEIGHT_EPSILON,
    )


def _point_geometry_cache_key(candidate: dict, frameanalysis_dir: str, ib_txt_path: str, ib_buf_path: str) -> tuple:
    import_paths = dict(candidate.get("import_paths", {}) or {})
    vb_payload = dict(import_paths.get("vb", {}) or {})
    vb0_payload = dict(vb_payload.get("vb0", {}) or {})
    skin_slot_name, skin_slot_index = _skin_slot_from_candidate(candidate)
    skin_payload = dict(vb_payload.get(skin_slot_name, {}) or {})
    skin_format = dict(candidate.get("skin_format", {}) or {})
    paths = [
        ib_txt_path,
        ib_buf_path,
        *_resolved_payload_paths(vb0_payload, frameanalysis_dir),
        *_resolved_payload_paths(skin_payload, frameanalysis_dir),
    ]
    return (
        _source_key_from_candidate(candidate),
        str(skin_slot_name),
        int(skin_slot_index),
        str(skin_format.get("blend_indices_format", "") or "").upper(),
        str(skin_format.get("blend_weights_format", "") or "").upper(),
        int(skin_format.get("blend_indices_offset", -1) or -1),
        int(skin_format.get("blend_weights_offset", -1) or -1),
        tuple(_file_fingerprint(path) for path in paths if path),
    )


def _resolved_payload_paths(payload: dict, frameanalysis_dir: str) -> list[str]:
    paths = [
        _resolve_path(str(payload.get("buf", "") or ""), frameanalysis_dir),
        *(
            _resolve_path(str(path), frameanalysis_dir)
            for path in list(payload.get("txt", []) or [])
            if str(path)
        ),
        *(
            _resolve_path(str(path), frameanalysis_dir)
            for path in list(payload.get("layout_txt", []) or [])
            if str(path)
        ),
    ]
    return [path for path in paths if path]


def _file_fingerprint(path: str) -> tuple[str, int, int]:
    normalized = os.path.abspath(path)
    try:
        stat = os.stat(normalized)
    except OSError:
        return normalized, -1, -1
    return normalized, int(stat.st_mtime_ns), int(stat.st_size)


def _geometry_points(geometry: PointGeometry, resolver) -> list[WeightedPoint]:
    points: list[WeightedPoint] = []
    for vertex_index, position in enumerate(geometry.positions):
        resolved_weights: list[tuple[object, float]] = []
        blend_indices = geometry.blend_indices[vertex_index] if vertex_index < len(geometry.blend_indices) else ()
        blend_weights = geometry.blend_weights[vertex_index] if vertex_index < len(geometry.blend_weights) else ()
        for local_bone, weight in zip(blend_indices[:4], blend_weights[:4]):
            if float(weight) <= _WEIGHT_EPSILON:
                continue
            resolved = resolver(int(local_bone), float(weight))
            if resolved is not None:
                resolved_weights.append(resolved)
        normalized = _normalize_vertex_weights(resolved_weights)
        if normalized:
            points.append(WeightedPoint(position=_position_tuple(position), weights=tuple(normalized)))
    return points


def _normalize_vertex_weights(weights: list[tuple[object, float]]) -> list[tuple[object, float]]:
    if not weights:
        return []
    merged: list[tuple[object, float]] = []
    for key, weight in weights:
        if float(weight) <= _WEIGHT_EPSILON:
            continue
        for index, (existing_key, existing_weight) in enumerate(merged):
            if existing_key == key:
                merged[index] = (existing_key, existing_weight + float(weight))
                break
        else:
            merged.append((key, float(weight)))
    total = sum(weight for _key, weight in merged)
    if total <= _WEIGHT_EPSILON:
        return []
    normalized = [(key, float(weight) / total) for key, weight in merged]
    normalized.sort(key=lambda item: (-float(item[1]), str(item[0])))
    return normalized


def _canonical_weight_resolver(remap: dict):
    local_to_global = {
        int(local): int(global_bone)
        for local, global_bone in dict(remap.get("local_group_to_global_group", {}) or {}).items()
        if int(global_bone) >= 0
    }

    def resolve(local_bone: int, weight: float):
        global_bone = local_to_global.get(int(local_bone))
        if global_bone is None:
            return None
        return int(global_bone), float(weight)

    return resolve


def _lod_weight_resolver(source_key: str):
    def resolve(local_bone: int, weight: float):
        return (source_key, int(local_bone)), float(weight)

    return resolve


def _normalize_weights(weights: list[tuple[object, float]]) -> list[tuple[object, float]]:
    merged: dict[object, float] = {}
    for key, weight in weights:
        merged[key] = merged.get(key, 0.0) + float(weight)
    total = sum(weight for weight in merged.values() if weight > _WEIGHT_EPSILON)
    if total <= _WEIGHT_EPSILON:
        return []
    return [
        (key, float(weight) / total)
        for key, weight in sorted(merged.items(), key=lambda item: (-float(item[1]), str(item[0])))
        if float(weight) > _WEIGHT_EPSILON
    ]


def _vote_lod_to_global(
    canonical_points: list[WeightedPoint],
    lod_points: list[WeightedPoint],
    tolerance: float,
) -> tuple[dict[tuple[str, int, int], dict], int]:
    canonical_hash = _build_spatial_hash(canonical_points, tolerance)
    stats: dict[tuple[str, int, int], dict] = {}
    matched_count = 0
    for lod_point in lod_points:
        nearest = _nearest_point(lod_point.position, canonical_hash, tolerance)
        if nearest is None:
            continue
        canonical_point, distance = nearest
        distance_score = 1.0 / (1.0 + distance / max(tolerance, 1.0e-6))
        matched_count += 1
        for lod_key, lod_weight in lod_point.weights:
            lod_record_key, lod_local_bone = lod_key
            for canonical_global, canonical_weight in canonical_point.weights:
                score = float(lod_weight) * float(canonical_weight) * distance_score
                if score <= _MIN_PAIR_SCORE:
                    continue
                key = (str(lod_record_key), int(lod_local_bone), int(canonical_global))
                record = stats.setdefault(key, {"score": 0.0, "votes": 0, "distance_sum": 0.0})
                record["score"] += score
                record["votes"] += 1
                record["distance_sum"] += float(distance)
    return stats, matched_count


def _nearest_point(position: tuple[float, float, float], spatial_hash: dict, tolerance: float):
    tolerance_squared = tolerance * tolerance
    best_point = None
    best_distance_squared = None
    for key in _neighbor_keys(_cell_key(position, tolerance)):
        for point in spatial_hash.get(key, ()):
            distance_squared = _distance_squared(position, point.position)
            if distance_squared > tolerance_squared:
                continue
            if best_distance_squared is None or distance_squared < best_distance_squared:
                best_point = point
                best_distance_squared = distance_squared
    if best_point is None or best_distance_squared is None:
        return None
    return best_point, math.sqrt(best_distance_squared)


def _build_spatial_hash(points: list[WeightedPoint], tolerance: float) -> dict[tuple[int, int, int], list[WeightedPoint]]:
    return _generic_build_spatial_hash(points, lambda point: point.position, tolerance)


def _cell_key(position: tuple[float, float, float], tolerance: float) -> tuple[int, int, int]:
    return _generic_cell_key(position, tolerance)


def _neighbor_keys(base_key: tuple[int, int, int]):
    yield from _generic_neighbor_keys(base_key)


def _initial_match_tolerance(canonical_points: list[WeightedPoint], lod_points: list[WeightedPoint]) -> float:
    positions = [point.position for point in canonical_points[:20000]] + [point.position for point in lod_points[:20000]]
    if not positions:
        return 0.02
    diag = _positions_diag(positions)
    return max(0.015, diag * 0.015)


def _point_cloud_diag(points: list[WeightedPoint]) -> float:
    if not points:
        return 0.0
    return _positions_diag([point.position for point in points[:40000]])


def _positions_diag(positions: list[tuple[float, float, float]]) -> float:
    return positions_diag(positions)


def _distance_squared(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return (
        (left[0] - right[0]) * (left[0] - right[0])
        + (left[1] - right[1]) * (left[1] - right[1])
        + (left[2] - right[2]) * (left[2] - right[2])
    )


def _pair_rank(stats: dict) -> tuple[float, int, float]:
    return float(stats["score"]), int(stats["votes"]), -float(stats["distance_sum"]) / max(1, int(stats["votes"]))


def _build_lod_capture_records(
    lod_records: dict[str, dict],
    global_to_lod: dict[int, dict],
    *,
    lod_links: list[dict] | None = None,
    global_candidates: dict[int, list[dict]] | None = None,
) -> list[dict]:
    pairs_by_lod: dict[str, dict[tuple[int, int], dict]] = {}

    def add_pair(lod_key: str, lod_local_bone: int, canonical_global: int, mapping: dict | None = None) -> None:
        lod_key = str(lod_key or "")
        lod_local_bone = int(lod_local_bone)
        canonical_global = int(canonical_global)
        if not lod_key or lod_local_bone < 0 or canonical_global < 0:
            return
        pair = {
            "lod_local_bone": lod_local_bone,
            "canonical_global_bone": canonical_global,
            "score": float(dict(mapping or {}).get("score", 0.0)),
            "votes": int(dict(mapping or {}).get("votes", 0)),
        }
        bucket = pairs_by_lod.setdefault(lod_key, {})
        pair_key = (lod_local_bone, canonical_global)
        current = bucket.get(pair_key)
        if current is None or _mapping_entry_rank(pair) > _mapping_entry_rank(current):
            bucket[pair_key] = pair

    for canonical_global, mapping in sorted(global_to_lod.items()):
        lod_key = str(mapping.get("lod_record_key", "") or "")
        if not lod_key:
            continue
        add_pair(lod_key, int(mapping.get("lod_local_bone", -1)), int(canonical_global), mapping)

    candidates = global_candidates or {}
    for link in lod_links or []:
        required_globals = _link_required_globals(link)
        if not required_globals:
            continue
        for source in link.get("lod_sources", []) or []:
            lod_key = str(dict(source or {}).get("lod_record_key", "") or "")
            if not lod_key:
                continue
            for canonical_global in required_globals:
                mapping = _best_candidate_by_lod(_iter_mapping_candidates(candidates.get(int(canonical_global)))).get(lod_key)
                if mapping is not None:
                    add_pair(lod_key, int(mapping.get("lod_local_bone", -1)), int(canonical_global), mapping)
                    continue
                local_bone = _same_identity_lod_local_bone(link, lod_records.get(lod_key), int(canonical_global))
                if local_bone >= 0:
                    add_pair(
                        lod_key,
                        local_bone,
                        int(canonical_global),
                        {"score": float(source.get("score", 0.0) or 0.0), "votes": int(source.get("votes", 0) or 0)},
                    )

    records = []
    for lod_key, pair_map in sorted(pairs_by_lod.items(), key=lambda item: (-len(item[1]), item[0])):
        lod_record = lod_records.get(lod_key)
        if not lod_record:
            continue
        pairs = list(pair_map.values())
        records.append({**lod_record, "scatter_pairs": sorted(pairs, key=lambda item: (item["lod_local_bone"], item["canonical_global_bone"]))})
    return records


def _link_required_globals(link: dict) -> list[int]:
    base = int(link.get("global_bone_base", 0) or 0)
    count = int(link.get("local_bone_count", 0) or 0)
    if count <= 0:
        return []
    return list(range(base, base + count))


def _same_identity_lod_local_bone(link: dict, lod_record: dict | None, canonical_global: int) -> int:
    if not lod_record:
        return -1
    if str(link.get("ib_hash", "") or "").lower() != str(lod_record.get("lod_ib_hash", "") or "").lower():
        return -1
    if int(link.get("match_first_index", 0) or 0) != int(lod_record.get("lod_match_first_index", 0) or 0):
        return -1
    if int(link.get("match_index_count", 0) or 0) != int(lod_record.get("lod_match_index_count", 0) or 0):
        return -1
    base = int(link.get("global_bone_base", 0) or 0)
    compact_index = int(canonical_global) - base
    if compact_index < 0:
        return -1
    used_indices = [int(value) for value in link.get("used_local_bone_indices", []) or []]
    if used_indices:
        if compact_index >= len(used_indices):
            return -1
        local_bone = int(used_indices[compact_index])
    else:
        local_bone = compact_index
    lod_local_count = int(lod_record.get("lod_source_local_bone_count", lod_record.get("lod_local_bone_count", 0)) or 0)
    if lod_local_count > 0 and local_bone >= lod_local_count:
        return -1
    return local_bone


def _mapping_global_candidates(mapping: dict) -> dict[int, list[dict]]:
    candidates = mapping.get("global_candidates")
    if isinstance(candidates, dict):
        return {
            int(canonical_global): [
                dict(candidate)
                for candidate in _iter_mapping_candidates(raw_candidates)
                if str(candidate.get("lod_record_key", "") or "")
            ]
            for canonical_global, raw_candidates in candidates.items()
        }
    return {
        int(canonical_global): [dict(entry)]
        for canonical_global, entry in dict(mapping.get("global_to_lod", {}) or {}).items()
        if str(dict(entry).get("lod_record_key", "") or "")
    }


def _iter_mapping_candidates(raw_candidates) -> list[dict]:
    if raw_candidates is None:
        return []
    if isinstance(raw_candidates, dict):
        return [raw_candidates]
    if isinstance(raw_candidates, (list, tuple)):
        return [candidate for candidate in raw_candidates if isinstance(candidate, dict)]
    return []


def _best_candidate_by_lod(candidates: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for candidate in candidates:
        lod_key = str(candidate.get("lod_record_key", "") or "")
        if not lod_key:
            continue
        current = best.get(lod_key)
        if current is None or _mapping_entry_rank(candidate) > _mapping_entry_rank(current):
            best[lod_key] = dict(candidate)
    return best


def _mapping_entry_rank(entry: dict) -> tuple[float, int, float]:
    return (
        float(entry.get("score", 0.0)),
        int(entry.get("votes", 0)),
        -float(entry.get("average_distance", 0.0)),
    )


def _build_lod_record_chains(lod_records: dict[str, dict]) -> list[dict]:
    entries: list[tuple[int, str]] = []
    for lod_key, lod_record in lod_records.items():
        for draw_index in lod_record.get("lod_capture_draw_indices", lod_record.get("capture_draw_indices", [])) or []:
            try:
                draw = int(draw_index)
            except (TypeError, ValueError):
                continue
            if draw >= 0:
                entries.append((draw, str(lod_key)))
    if not entries:
        return []

    entries.sort(key=lambda item: (int(item[0]), item[1]))
    segments: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    previous_draw: int | None = None
    for entry in entries:
        draw = int(entry[0])
        if current and previous_draw is not None and draw - previous_draw > _LOD_SHADOW_CHAIN_GAP:
            segments.append(current)
            current = []
        current.append(entry)
        previous_draw = draw
    if current:
        segments.append(current)

    chains: list[dict] = []
    for chain_index, segment in enumerate(segments):
        draw_start = min(draw for draw, _lod_key in segment)
        draw_end = max(draw for draw, _lod_key in segment)
        start_draw, start_lod_key = min(segment, key=lambda item: (int(item[0]), item[1]))
        host_draw, host_lod_key = max(segment, key=lambda item: (int(item[0]), item[1]))
        chain_lod_keys = sorted({lod_key for _draw, lod_key in segment})
        host_record = lod_records.get(host_lod_key, {})
        start_record = lod_records.get(start_lod_key, {})
        chains.append(
            {
                "chain_index": int(chain_index),
                "draw_start": int(draw_start),
                "draw_end": int(draw_end),
                "start_draw_index": int(start_draw),
                "start_lod_record_key": str(start_lod_key),
                "start_key": _lod_override_key_payload(start_record),
                "host_draw_index": int(host_draw),
                "host_lod_record_key": str(host_lod_key),
                "host_key": _lod_override_key_payload(host_record),
                "lod_record_keys": chain_lod_keys,
            }
        )
    return chains


def _build_lod_links(
    main_records: list[dict],
    lod_records: dict[str, dict],
    global_to_lod: dict[int, object],
    *,
    lod_chains: list[dict] | None = None,
) -> list[dict]:
    chains = list(lod_chains or [])
    if not chains:
        return _build_lod_links_global(main_records, lod_records, global_to_lod)

    links: list[dict] = []
    matched_main_keys: set[str] = set()
    for chain in chains:
        chain_keys = {
            str(value)
            for value in chain.get("lod_record_keys", []) or []
            if str(value) in lod_records
        }
        if not chain_keys:
            continue
        chain_lod_records = {lod_key: lod_records[lod_key] for lod_key in sorted(chain_keys)}
        chain_links = _build_lod_links_global(main_records, chain_lod_records, global_to_lod)
        for link in chain_links:
            sources = [dict(source or {}) for source in link.get("lod_sources", []) or []]
            if not sources:
                continue
            matched_main_keys.add(str(link.get("source_key", "") or ""))
            for source in sources:
                source["lod_chain_index"] = _int_default(chain.get("chain_index"), -1)
                source["lod_chain_draw_start"] = _int_default(chain.get("draw_start"), -1)
                source["lod_chain_draw_end"] = _int_default(chain.get("draw_end"), -1)
                source["lod_chain_host_draw_index"] = _int_default(chain.get("host_draw_index"), -1)
                source["lod_chain_host_key"] = dict(chain.get("host_key", {}) or {})
                links.append(
                    {
                        **{key: value for key, value in link.items() if key != "lod_sources"},
                        "lod_sources": [source],
                        "status": "matched",
                        "lod_chain_index": _int_default(chain.get("chain_index"), -1),
                        "lod_chain_draw_start": _int_default(chain.get("draw_start"), -1),
                        "lod_chain_draw_end": _int_default(chain.get("draw_end"), -1),
                        "lod_chain_host_draw_index": _int_default(chain.get("host_draw_index"), -1),
                        "lod_chain_host_key": dict(chain.get("host_key", {}) or {}),
                    }
                )

    for main_index, main_record in enumerate(main_records):
        main_key = str(main_record.get("source_key", "") or f"__main_{main_index}")
        if main_key not in matched_main_keys:
            links.append({**main_record, "lod_sources": [], "status": "unmatched"})
    return links


def _build_lod_links_global(main_records: list[dict], lod_records: dict[str, dict], global_to_lod: dict[int, object]) -> list[dict]:
    direct_links = _build_lod_links_from_vb2_signatures(main_records, lod_records)
    used_lod_keys = {
        str(source.get("lod_record_key", "") or "")
        for link in direct_links
        for source in link.get("lod_sources", []) or []
        if str(source.get("lod_record_key", "") or "")
    }
    fallback_allowed = set(lod_records).difference(used_lod_keys)
    fallback_links = _build_lod_links_from_bone_candidates(
        main_records,
        lod_records,
        global_to_lod,
        allowed_lod_keys=fallback_allowed if used_lod_keys else None,
    )
    fallback_by_main = {str(link.get("source_key", "") or ""): link for link in fallback_links}

    links: list[dict] = []
    for direct_link in direct_links:
        if direct_link.get("lod_sources"):
            links.append(direct_link)
            continue
        fallback = fallback_by_main.get(str(direct_link.get("source_key", "") or ""))
        if fallback is not None and fallback.get("lod_sources"):
            links.append(fallback)
        else:
            links.append(direct_link)
    return links


def _build_lod_links_from_vb2_signatures(main_records: list[dict], lod_records: dict[str, dict]) -> list[dict]:
    candidate_pairs: list[tuple[dict, str, dict, str, dict, int, int]] = []
    for main_index, main_record in enumerate(main_records):
        main_key = str(main_record.get("source_key", "") or f"__main_{main_index}")
        main_slot_count = _relation_slot_count(main_record)
        if main_slot_count > 0:
            for lod_key, lod_record in lod_records.items():
                lod_slot_count = _relation_slot_count(lod_record)
                score = _slot_relation_score(main_record, lod_record)
                if score is None:
                    continue
                candidate_pairs.append((score, main_key, main_record, str(lod_key), lod_record, main_slot_count, lod_slot_count))

    candidate_pairs.sort(
        key=lambda item: (
            -float(item[0]["score"]),
            int(item[0]["slot_delta"]),
            -int(item[6]),
            str(item[3]),
        )
    )

    assigned_by_main: dict[str, tuple[dict, str, dict, int, int]] = {}
    used_lod_keys: set[str] = set()
    for score, main_key, _main_record, lod_key, lod_record, main_slot_count, lod_slot_count in candidate_pairs:
        if main_key in assigned_by_main or lod_key in used_lod_keys:
            continue
        assigned_by_main[main_key] = (score, lod_key, lod_record, main_slot_count, lod_slot_count)
        used_lod_keys.add(lod_key)

    links = []
    for main_index, main_record in enumerate(main_records):
        main_key = str(main_record.get("source_key", "") or f"__main_{main_index}")
        assigned = assigned_by_main.get(main_key)
        if assigned is None:
            links.append({**main_record, "lod_sources": [], "status": "unmatched"})
            continue
        best_score, best_lod_key, best_lod_record, main_slot_count, best_lod_slot_count = assigned
        lod_sources = [
            {
                "lod_record_key": best_lod_key,
                "lod_ib_hash": str(best_lod_record.get("lod_ib_hash", "") or ""),
                "lod_match_first_index": int(best_lod_record.get("lod_match_first_index", 0) or 0),
                "lod_match_index_count": int(best_lod_record.get("lod_match_index_count", 0) or 0),
                "mapped_global_count": int(min(main_slot_count, best_lod_slot_count)),
                "score": float(best_score["score"]),
                "votes": int(min(main_slot_count, best_lod_slot_count)),
                "relation_method": "vb2_slot_signature",
                "main_slot_count": int(main_slot_count),
                "lod_slot_count": int(best_lod_slot_count),
                "slot_delta": int(best_score["slot_delta"]),
                "slot_delta_ratio": float(best_score["slot_delta_ratio"]),
            }
        ]
        links.append({**main_record, "lod_sources": lod_sources, "status": "matched"})
    return links


def _build_lod_links_from_bone_candidates(
    main_records: list[dict],
    lod_records: dict[str, dict],
    global_to_lod: dict[int, object],
    *,
    allowed_lod_keys: set[str] | None = None,
) -> list[dict]:
    allowed = {str(value) for value in allowed_lod_keys} if allowed_lod_keys is not None else None
    links = []
    for main_record in main_records:
        base = int(main_record.get("global_bone_base", 0) or 0)
        count = int(main_record.get("local_bone_count", 0) or 0)
        by_lod: dict[str, dict] = {}
        for canonical_global in range(base, base + count):
            for lod_key, mapping in _best_candidate_by_lod(_iter_mapping_candidates(global_to_lod.get(canonical_global))).items():
                if allowed is not None and str(lod_key) not in allowed:
                    continue
                bucket = by_lod.setdefault(lod_key, {"mapped_global_count": 0, "score": 0.0, "votes": 0})
                bucket["mapped_global_count"] += 1
                bucket["score"] += float(mapping.get("score", 0.0))
                bucket["votes"] += int(mapping.get("votes", 0))
        if not by_lod:
            links.append({**main_record, "lod_sources": [], "status": "unmatched"})
            continue
        lod_sources = []
        for lod_key, bucket in sorted(by_lod.items(), key=lambda item: (-int(item[1]["mapped_global_count"]), -float(item[1]["score"]), item[0])):
            if int(bucket["mapped_global_count"]) < _minimum_lod_link_global_count(count):
                continue
            lod_record = lod_records.get(lod_key, {})
            lod_sources.append(
                {
                    "lod_record_key": lod_key,
                    "lod_ib_hash": str(lod_record.get("lod_ib_hash", "") or ""),
                    "lod_match_first_index": int(lod_record.get("lod_match_first_index", 0) or 0),
                    "lod_match_index_count": int(lod_record.get("lod_match_index_count", 0) or 0),
                    "mapped_global_count": int(bucket["mapped_global_count"]),
                    "score": float(bucket["score"]),
                    "votes": int(bucket["votes"]),
                    "relation_method": "bone_cloud_fallback",
                }
            )
        links.append({**main_record, "lod_sources": lod_sources, "status": "matched" if lod_sources else "unmatched"})
    return links


def _relation_slot_count(record: dict) -> int:
    signature = dict(record.get("vb2_signature", {}) or {})
    signature_count = int(signature.get("slot_count", 0) or 0)
    if signature_count > 0:
        return signature_count
    used_slots = record.get("used_local_bone_indices") or record.get("lod_used_local_bone_indices") or []
    if used_slots:
        return len({int(value) for value in used_slots if int(value) >= 0})
    return int(record.get("local_bone_count", record.get("lod_local_bone_count", 0)) or 0)


def _slot_relation_score(main_record: dict, lod_record: dict) -> dict | None:
    main_slot_count = _relation_slot_count(main_record)
    lod_slot_count = _relation_slot_count(lod_record)
    if main_slot_count <= 0 or lod_slot_count <= 0:
        return None
    slot_delta = abs(int(main_slot_count) - int(lod_slot_count))
    base_count = max(int(main_slot_count), int(lod_slot_count), 1)
    slot_delta_ratio = float(slot_delta) / float(base_count)
    direct_limit = max(_LOD_SLOT_DIRECT_ABS_TOLERANCE, math.ceil(float(main_slot_count) * _LOD_SLOT_DIRECT_RATIO_TOLERANCE))
    if slot_delta > direct_limit and slot_delta_ratio > _LOD_SLOT_NEAR_RATIO_TOLERANCE:
        return None
    geometry_bonus = _signature_geometry_affinity(main_record, lod_record)
    slot_score = 1.0 - min(1.0, slot_delta_ratio)
    return {
        "score": float(slot_score * 100.0 + geometry_bonus),
        "slot_delta": int(slot_delta),
        "slot_delta_ratio": float(slot_delta_ratio),
        "geometry_bonus": float(geometry_bonus),
    }


def _signature_geometry_affinity(main_record: dict, lod_record: dict) -> float:
    main_signature = dict(main_record.get("vb2_signature", {}) or {})
    lod_signature = dict(lod_record.get("vb2_signature", {}) or {})
    main_center = main_signature.get("center")
    lod_center = lod_signature.get("center")
    if not isinstance(main_center, (list, tuple)) or not isinstance(lod_center, (list, tuple)):
        return 0.0
    if len(main_center) < 3 or len(lod_center) < 3:
        return 0.0
    main_diag = float(main_signature.get("diag", 0.0) or 0.0)
    lod_diag = float(lod_signature.get("diag", 0.0) or 0.0)
    diag = max(main_diag, lod_diag, 1.0e-5)
    center_distance = math.sqrt(
        (float(main_center[0]) - float(lod_center[0])) ** 2
        + (float(main_center[1]) - float(lod_center[1])) ** 2
        + (float(main_center[2]) - float(lod_center[2])) ** 2
    )
    center_score = 1.0 / (1.0 + center_distance / diag)
    diag_delta = abs(main_diag - lod_diag) / max(main_diag, lod_diag, 1.0e-5)
    diag_score = 1.0 - min(1.0, diag_delta)
    return float(center_score + diag_score)


def _int_default(value, default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _minimum_lod_link_global_count(local_bone_count: int) -> int:
    count = max(0, int(local_bone_count))
    if count <= 0:
        return 1
    return min(count, max(_MIN_LOD_LINK_GLOBALS, math.ceil(count * _MIN_LOD_LINK_COVERAGE_RATIO)))


def _lod_record_payload(candidate: dict, source_key: str, *, vb2_signature: dict | None = None) -> dict:
    return {
        "lod_record_key": source_key,
        "lod_ib_hash": str(candidate.get("ib_hash", "") or "").lower(),
        "lod_match_first_index": int(candidate.get("match_first_index", 0) or 0),
        "lod_match_index_count": int(candidate.get("match_index_count", 0) or 0),
        "lod_import_draw_index": int(candidate.get("import_draw_index", -1) or -1),
        "lod_capture_draw_indices": [int(value) for value in candidate.get("shadow_draw_indices", []) or []],
        "lod_local_bone_count": int(candidate.get("local_bone_count", 0) or 0),
        "lod_source_local_bone_count": int(candidate.get("source_local_bone_count", candidate.get("local_bone_count", 0)) or 0),
        "lod_used_local_bone_indices": [int(value) for value in candidate.get("used_local_bone_indices", []) or []],
        "lod_import_vs_hash": str(candidate.get("import_vs_hash", "") or "").lower(),
        "lod_import_ps_hash": str(candidate.get("import_ps_hash", "") or "").lower(),
        "vb2_signature": dict(vb2_signature or {}),
    }


def _lod_override_key_payload(lod_record: dict) -> dict:
    return {
        "ib_hash": str(lod_record.get("lod_ib_hash", "") or "").lower(),
        "match_first_index": int(lod_record.get("lod_match_first_index", 0) or 0),
        "match_index_count": int(lod_record.get("lod_match_index_count", 0) or 0),
    }


def _lod_manifest_snapshot(lod_manifest: dict) -> dict:
    lod_records = {
        _source_key_from_candidate(candidate): _lod_record_payload(candidate, _source_key_from_candidate(candidate))
        for candidate in lod_manifest.get("candidate_ibs", []) or []
        if bool(candidate.get("enabled", True))
        and bool(candidate.get("shadow_capture_ready", False))
        and int(candidate.get("local_bone_count", 0) or 0) > 0
    }
    return {
        "schema_version": int(lod_manifest.get("schema_version", 1) or 1),
        "analyzer_version": int(lod_manifest.get("analyzer_version", 0) or 0),
        "frameanalysis_dir": str(lod_manifest.get("frameanalysis_dir", "") or ""),
        "shadow_stage": dict(lod_manifest.get("shadow_stage", {}) or {}),
        "lod_chains": _build_lod_record_chains(lod_records),
        "candidate_ibs": list(lod_manifest.get("candidate_ibs", []) or []),
        "validation": list(lod_manifest.get("validation", []) or []),
    }


def _object_remaps_by_key(manifest: dict) -> dict[str, dict]:
    remaps = {}
    for remap in manifest.get("object_remaps", []) or []:
        source_key = str(remap.get("source_key", "") or _source_key_from_candidate(remap)).lower()
        if source_key:
            remaps[source_key] = dict(remap)
    return remaps


def _object_remap_from_pool_record(record: dict) -> dict:
    source_key = _source_key_from_candidate(record)
    used_local_bone_indices = [int(value) for value in record.get("used_local_bone_indices", []) or []]
    if not used_local_bone_indices:
        used_local_bone_indices = list(range(max(0, int(record.get("local_bone_count", 0) or 0))))
    global_base = int(record.get("global_bone_base", record.get("capture_store_base", 0)) or 0)
    return {
        "source_key": source_key,
        "ib_hash": str(record.get("ib_hash", "") or "").lower(),
        "match_first_index": int(record.get("match_first_index", 0) or 0),
        "match_index_count": int(record.get("match_index_count", 0) or 0),
        "local_group_to_global_group": {
            str(local_bone): int(global_base) + compact_index
            for compact_index, local_bone in enumerate(used_local_bone_indices)
        },
    }


def _source_key_from_candidate(candidate: dict) -> str:
    return (
        f"{str(candidate.get('ib_hash', '') or '').lower()}-"
        f"{int(candidate.get('match_index_count', candidate.get('source_index_count', 0)) or 0)}-"
        f"{int(candidate.get('match_first_index', 0) or 0)}"
    )


def _position_tuple(position) -> tuple[float, float, float]:
    return float(position[0]), float(position[1]), float(position[2])


def _build_lod_variant_id(frameanalysis_dir: str, capture_records: list[dict]) -> str:
    base_name = os.path.basename(os.path.abspath(frameanalysis_dir).rstrip("\\/")) or "lod"
    seed = "|".join(
        [
            base_name,
            *(
                f"{record.get('lod_record_key')}:{len(record.get('scatter_pairs', []) or [])}"
                for record in capture_records
            ),
        ]
    )
    checksum = zlib.crc32(seed.lower().encode("utf-8")) & 0xFFFFFFFF
    return f"{base_name}_{checksum:08x}"
