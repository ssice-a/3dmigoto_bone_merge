"""LOD FrameAnalysis analyzer for canonical global-bone scatter mapping."""

from __future__ import annotations

import math
import os
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor

from .import_candidates import LoadedCandidateGeometry, load_candidate_geometry
from .main_analyze import analyze_main_frameanalysis


_WEIGHT_EPSILON = 1.0e-5
_MIN_PAIR_SCORE = 0.001


@dataclass(frozen=True)
class WeightedPoint:
    position: tuple[float, float, float]
    weights: tuple[tuple[object, float], ...]


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

    lod_manifest = analyze_main_frameanalysis(normalized_lod_dir)
    main_points, canonical_global_count, main_records = _build_canonical_point_cloud(canonical_manifest)
    lod_points, lod_records = _build_lod_point_cloud(lod_manifest)
    if not main_points:
        raise ValueError("No canonical weighted point cloud could be built from the main manifest")
    if not lod_points:
        raise ValueError("No LOD weighted point cloud could be built from the LOD FrameAnalysis")

    mapping = build_lod_scatter_mapping(main_points, lod_points, canonical_global_count)
    capture_records = _build_lod_capture_records(lod_records, mapping["global_to_lod"])
    links = _build_lod_links(main_records, lod_records, mapping["global_to_lod"])
    variant_id = _build_lod_variant_id(normalized_lod_dir, capture_records)
    generated_at = datetime.now(timezone.utc).isoformat()
    validation = list(mapping["validation"])
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
                "match_tolerance": float(mapping["match_tolerance"]),
                "matched_vertex_count": int(mapping["matched_vertex_count"]),
                "lod_vertex_count": int(len(lod_points)),
                "shadow_stage": dict(lod_manifest.get("shadow_stage", {}) or {}),
                "generated_at": generated_at,
            }
        ],
        "lod_links": links,
        "lod_capture_records": capture_records,
        "lod_mapping": mapping["mapping_entries"],
        "validation": validation,
        "lod_manifest_snapshot": _lod_manifest_snapshot(lod_manifest),
    }


def build_lod_scatter_mapping(
    canonical_points: list[WeightedPoint],
    lod_points: list[WeightedPoint],
    canonical_global_count: int,
) -> dict:
    """Vote canonical global bones to LOD local bones by nearest weighted vertices."""

    tolerance = _initial_match_tolerance(canonical_points, lod_points)
    best_stats = {}
    best_matched_count = 0
    best_tolerance = tolerance
    for scale in (1.0, 2.0, 4.0):
        trial_tolerance = tolerance * scale
        stats, matched_count = _vote_lod_to_global(canonical_points, lod_points, trial_tolerance)
        if matched_count > best_matched_count or (matched_count == best_matched_count and len(stats) > len(best_stats)):
            best_stats = stats
            best_matched_count = matched_count
            best_tolerance = trial_tolerance
        if matched_count >= max(4, int(len(lod_points) * 0.2)):
            break

    best_by_global: dict[int, tuple[tuple[str, int], dict]] = {}
    for (lod_key, lod_local_bone, canonical_global), stats in best_stats.items():
        if int(canonical_global) < 0 or int(canonical_global) >= int(canonical_global_count):
            continue
        if float(stats["score"]) < _MIN_PAIR_SCORE:
            continue
        current = best_by_global.get(int(canonical_global))
        if current is None or _pair_rank(stats) > _pair_rank(current[1]):
            best_by_global[int(canonical_global)] = ((str(lod_key), int(lod_local_bone)), stats)

    mapping_entries = []
    global_to_lod: dict[int, dict] = {}
    for canonical_global in range(int(canonical_global_count)):
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
        entry = {
            "canonical_global_bone": int(canonical_global),
            "lod_record_key": lod_key,
            "lod_local_bone": int(lod_local_bone),
            "score": float(stats["score"]),
            "votes": int(stats["votes"]),
            "average_distance": float(stats["distance_sum"]) / max(1, int(stats["votes"])),
            "status": "matched",
        }
        mapping_entries.append(entry)
        global_to_lod[int(canonical_global)] = entry

    unmatched_count = sum(1 for entry in mapping_entries if str(entry["status"]) == "unmatched")
    validation = []
    if unmatched_count:
        validation.append(
            {
                "severity": "warning",
                "code": "lod_unmatched_global_bones",
                "message": f"LOD mapping left {unmatched_count}/{canonical_global_count} canonical global bone(s) unmatched.",
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
        "mapping_entries": mapping_entries,
        "validation": validation,
        "matched_vertex_count": best_matched_count,
        "match_tolerance": best_tolerance,
    }


def _build_canonical_point_cloud(canonical_manifest: dict) -> tuple[list[WeightedPoint], int, list[dict]]:
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
        if candidate is None or not remap:
            continue
        geometry = load_candidate_geometry(candidate, frameanalysis_dir)
        points.extend(_geometry_points(geometry, _canonical_weight_resolver(remap)))
        global_base = int(record.get("global_bone_base", record.get("capture_store_base", 0)) or 0)
        local_count = int(record.get("local_bone_count", 0) or 0)
        canonical_global_count = max(canonical_global_count, global_base + local_count)
        main_records.append(
            {
                "source_key": source_key,
                "ib_hash": str(record.get("ib_hash", "") or "").lower(),
                "match_first_index": int(record.get("match_first_index", 0) or 0),
                "match_index_count": int(record.get("match_index_count", 0) or 0),
                "global_bone_base": global_base,
                "local_bone_count": local_count,
            }
        )
    return points, canonical_global_count, main_records


def _build_lod_point_cloud(lod_manifest: dict) -> tuple[list[WeightedPoint], dict[str, dict]]:
    frameanalysis_dir = str(lod_manifest.get("frameanalysis_dir", "") or "")
    points: list[WeightedPoint] = []
    lod_records: dict[str, dict] = {}
    for candidate in lod_manifest.get("candidate_ibs", []) or []:
        if not bool(candidate.get("enabled", True)):
            continue
        if not bool(candidate.get("shadow_capture_ready", False)):
            continue
        if int(candidate.get("local_bone_count", 0) or 0) <= 0:
            continue
        source_key = _source_key_from_candidate(candidate)
        geometry = load_candidate_geometry(candidate, frameanalysis_dir)
        points.extend(_geometry_points(geometry, _lod_weight_resolver(source_key)))
        lod_records[source_key] = _lod_record_payload(candidate, source_key)
    return points, lod_records


def _geometry_points(geometry: LoadedCandidateGeometry, resolver) -> list[WeightedPoint]:
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
        normalized = _normalize_weights(resolved_weights)
        if normalized:
            points.append(WeightedPoint(position=_position_tuple(position), weights=tuple(normalized)))
    return points


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
    spatial_hash: dict[tuple[int, int, int], list[WeightedPoint]] = {}
    for point in points:
        spatial_hash.setdefault(_cell_key(point.position, tolerance), []).append(point)
    return spatial_hash


def _cell_key(position: tuple[float, float, float], tolerance: float) -> tuple[int, int, int]:
    inverse = 1.0 / max(tolerance, 1.0e-6)
    return floor(position[0] * inverse), floor(position[1] * inverse), floor(position[2] * inverse)


def _neighbor_keys(base_key: tuple[int, int, int]):
    base_x, base_y, base_z = base_key
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                yield base_x + offset_x, base_y + offset_y, base_z + offset_z


def _initial_match_tolerance(canonical_points: list[WeightedPoint], lod_points: list[WeightedPoint]) -> float:
    positions = [point.position for point in canonical_points[:20000]] + [point.position for point in lod_points[:20000]]
    if not positions:
        return 0.02
    min_x = min(position[0] for position in positions)
    min_y = min(position[1] for position in positions)
    min_z = min(position[2] for position in positions)
    max_x = max(position[0] for position in positions)
    max_y = max(position[1] for position in positions)
    max_z = max(position[2] for position in positions)
    diag = math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2)
    return max(0.015, diag * 0.015)


def _distance_squared(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return (
        (left[0] - right[0]) * (left[0] - right[0])
        + (left[1] - right[1]) * (left[1] - right[1])
        + (left[2] - right[2]) * (left[2] - right[2])
    )


def _pair_rank(stats: dict) -> tuple[float, int, float]:
    return float(stats["score"]), int(stats["votes"]), -float(stats["distance_sum"]) / max(1, int(stats["votes"]))


def _build_lod_capture_records(lod_records: dict[str, dict], global_to_lod: dict[int, dict]) -> list[dict]:
    pairs_by_lod: dict[str, list[dict]] = {}
    for canonical_global, mapping in sorted(global_to_lod.items()):
        lod_key = str(mapping.get("lod_record_key", "") or "")
        if not lod_key:
            continue
        pairs_by_lod.setdefault(lod_key, []).append(
            {
                "lod_local_bone": int(mapping.get("lod_local_bone", -1)),
                "canonical_global_bone": int(canonical_global),
                "score": float(mapping.get("score", 0.0)),
                "votes": int(mapping.get("votes", 0)),
            }
        )

    records = []
    for lod_key, pairs in sorted(pairs_by_lod.items(), key=lambda item: (-len(item[1]), item[0])):
        lod_record = lod_records.get(lod_key)
        if not lod_record:
            continue
        records.append({**lod_record, "scatter_pairs": sorted(pairs, key=lambda item: (item["lod_local_bone"], item["canonical_global_bone"]))})
    return records


def _build_lod_links(main_records: list[dict], lod_records: dict[str, dict], global_to_lod: dict[int, dict]) -> list[dict]:
    links = []
    for main_record in main_records:
        base = int(main_record.get("global_bone_base", 0) or 0)
        count = int(main_record.get("local_bone_count", 0) or 0)
        by_lod: dict[str, dict] = {}
        for canonical_global in range(base, base + count):
            mapping = global_to_lod.get(canonical_global)
            if not mapping:
                continue
            lod_key = str(mapping.get("lod_record_key", "") or "")
            bucket = by_lod.setdefault(lod_key, {"mapped_global_count": 0, "score": 0.0, "votes": 0})
            bucket["mapped_global_count"] += 1
            bucket["score"] += float(mapping.get("score", 0.0))
            bucket["votes"] += int(mapping.get("votes", 0))
        if not by_lod:
            links.append({**main_record, "lod_sources": [], "status": "unmatched"})
            continue
        lod_sources = []
        for lod_key, bucket in sorted(by_lod.items(), key=lambda item: (-int(item[1]["mapped_global_count"]), -float(item[1]["score"]), item[0])):
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
                }
            )
        links.append({**main_record, "lod_sources": lod_sources, "status": "matched"})
    return links


def _lod_record_payload(candidate: dict, source_key: str) -> dict:
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
    }


def _lod_manifest_snapshot(lod_manifest: dict) -> dict:
    return {
        "schema_version": int(lod_manifest.get("schema_version", 1) or 1),
        "frameanalysis_dir": str(lod_manifest.get("frameanalysis_dir", "") or ""),
        "shadow_stage": dict(lod_manifest.get("shadow_stage", {}) or {}),
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
