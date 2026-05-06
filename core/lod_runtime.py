"""LOD capture scanning, matching, and runtime asset helpers."""

from __future__ import annotations

import os
import zlib

from ..constants import (
    BUFFER_EXPORT_DIR_NAME,
    LOD_CAPTURE_MANIFEST_FILE_NAME,
)
from .blender_ops import _infer_group_spatial_info_world, _normalize_local_to_global
from .frameanalysis import detect_last_shadow_host, find_draw_records_for_targets, resolve_output_dir
from .io import ensure_directory, write_json
from .models import LocalPaletteRecord, LodMappingRecord, LodPartRecord, LodRuntimePartRecord, PartRecord, TargetObjectSpec


def scan_lod_targets_and_generate_manifest(
    frameanalysis_dir: str,
    target_specs: list[TargetObjectSpec],
    output_dir: str | None = None,
) -> dict:
    normalized_frameanalysis_dir = os.path.abspath(frameanalysis_dir)
    normalized_output_dir = resolve_output_dir(normalized_frameanalysis_dir, output_dir)

    draw_records, warnings = find_draw_records_for_targets(normalized_frameanalysis_dir, target_specs)
    if not draw_records:
        details = "; ".join(warnings[:5])
        if details:
            raise ValueError(f"No matching LOD draw records found for current target list: {details}")
        raise ValueError("No matching LOD draw records found for current target list")

    part_records = _build_lod_part_records(draw_records)
    object_remaps = _build_lod_object_remaps(part_records)
    total_lod_global_bones = sum(int(part.bone_count) for part in part_records)

    shadow_host_hash = ""
    shadow_host_match_index_count = -1
    shadow_host_vs_hash = ""
    shadow_host_warning = ""
    try:
        shadow_host = detect_last_shadow_host(normalized_frameanalysis_dir)
        shadow_host_hash = shadow_host.ib_hash
        shadow_host_match_index_count = int(shadow_host.match_index_count)
        shadow_host_vs_hash = str(shadow_host.vs_hash or "")
    except Exception as exc:  # pragma: no cover - depends on external dump completeness
        shadow_host_warning = str(exc)

    variant_id = _build_lod_variant_id(normalized_frameanalysis_dir, part_records)
    payload = {
        "variant_id": variant_id,
        "frameanalysis_dir": normalized_frameanalysis_dir,
        "selected_vs_hashes": sorted({str(part.vs_hash).lower() for part in part_records}),
        "shadow_host_hash": shadow_host_hash,
        "shadow_host_match_index_count": shadow_host_match_index_count,
        "shadow_host_vs_hash": shadow_host_vs_hash,
        "part_records": [
            {
                "draw_index": int(part.draw_index),
                "object_name": str(part.object_name),
                "vs_hash": str(part.vs_hash).lower(),
                "ib_hash": str(part.ib_hash).lower(),
                "match_index_count": int(part.match_index_count),
                "capture_bone_count": int(part.bone_count),
                "lod_global_bone_base": int(part.lod_global_bone_base),
                "vb2_path": str(part.vb2_path),
                "vs_t0_path": str(part.vs_t0_path),
                "vs_cb1_path": str(part.vs_cb1_path),
                "vs_cb1_first_constant": int(part.vs_cb1_first_constant),
                "vs_cb1_num_constants": int(part.vs_cb1_num_constants),
            }
            for part in part_records
        ],
        "object_remaps": [
            {
                "object_name": str(remap["object_name"]),
                "ib_hash": str(remap["ib_hash"]).lower(),
                "match_index_count": int(remap["match_index_count"]),
                "local_group_to_global_group": {
                    str(local_group): int(global_group)
                    for local_group, global_group in remap["local_group_to_global_group"].items()
                },
            }
            for remap in object_remaps
        ],
        "warnings": list(warnings),
        "shadow_host_warning": shadow_host_warning,
        "total_lod_global_bones": total_lod_global_bones,
    }
    manifest_path = write_json(os.path.join(normalized_output_dir, LOD_CAPTURE_MANIFEST_FILE_NAME), payload)
    return {
        "manifest_path": manifest_path,
        "payload": payload,
        "scanned_parts": len(part_records),
        "total_lod_global_bones": total_lod_global_bones,
        "warnings": tuple(warnings),
        "shadow_host_warning": shadow_host_warning,
    }


def build_lod_mapping(
    canonical_manifest: dict,
    lod_variant: dict,
    canonical_mesh_entries: list[tuple[object, dict]],
    lod_mesh_entries: list[tuple[object, dict]],
) -> list[LodMappingRecord]:
    canonical_total = _canonical_total_global_bones(canonical_manifest.get("part_records", []))
    if canonical_total <= 0:
        raise ValueError("Main canonical manifest does not contain any global bones")

    lod_part_records = list(lod_variant.get("part_records", []) or [])
    if not lod_part_records:
        raise ValueError("LOD variant does not contain any scanned part records")

    canonical_infos = _aggregate_group_infos(canonical_mesh_entries)
    lod_infos = _aggregate_group_infos(lod_mesh_entries)
    if not canonical_infos:
        raise ValueError("No canonical global group spatial data could be inferred from the current main meshes")
    if not lod_infos:
        raise ValueError("No LOD global group spatial data could be inferred from the current LOD meshes")

    exact_matches, exact_notes = _build_exact_matches(canonical_infos, lod_infos)
    grouped_matches, grouped_notes = _build_grouped_matches(
        canonical_infos,
        lod_infos,
        exact_matches,
    )

    notes_by_canonical: dict[int, str] = {}
    notes_by_canonical.update(exact_notes)
    for canonical_global, note in grouped_notes.items():
        notes_by_canonical[canonical_global] = note

    records: list[LodMappingRecord] = []
    for canonical_global in range(canonical_total):
        mapped_lod_global = -1
        status = "unmatched"
        score = 0.0
        note = notes_by_canonical.get(canonical_global, "")

        if canonical_global in exact_matches:
            mapped_lod_global, score = exact_matches[canonical_global]
            status = "exact"
        elif canonical_global in grouped_matches:
            mapped_lod_global, score = grouped_matches[canonical_global]
            status = "grouped"

        records.append(
            LodMappingRecord(
                canonical_global_bone=canonical_global,
                mapped_lod_global_bone=int(mapped_lod_global),
                status=status,
                score=float(score),
                note=str(note),
            )
        )
    return records


def materialize_lod_runtime_assets(
    output_dir: str,
    canonical_part_records: list[PartRecord],
    local_palette_records: list[LocalPaletteRecord],
    mapping_payload: dict | None,
) -> dict:
    payload = dict(mapping_payload or {})
    lod_variant = dict(payload.get("lod_variant", {}) or {})
    mapping_records = list(payload.get("canonical_global_to_lod_global", []) or [])
    if not lod_variant or not lod_variant.get("part_records"):
        return {"capture_parts": [], "palette_records": [], "variants": [], "lod_host_map": []}

    buffer_dir = ensure_directory(os.path.join(output_dir, BUFFER_EXPORT_DIR_NAME))
    variant_id = str(lod_variant.get("variant_id", "lod") or "lod")
    canonical_total = _canonical_total_global_bones(
        [
            {
                "global_bone_base": int(part.global_bone_base),
                "capture_bone_count": int(part.bone_count),
            }
            for part in canonical_part_records
        ]
    )

    capture_parts: list[LodRuntimePartRecord] = []
    palette_records: list[LocalPaletteRecord] = []
    lod_global_to_capture_store: dict[int, int] = {}
    next_capture_store_base = canonical_total
    for part_record in lod_variant.get("part_records", []):
        part = _lod_part_record_from_payload(part_record)
        resource_suffix = _build_lod_capture_resource_suffix(variant_id, part)
        capture_store_base = next_capture_store_base
        next_capture_store_base += int(part.bone_count)
        capture_parts.append(
            LodRuntimePartRecord(
                variant_id=variant_id,
                draw_index=int(part.draw_index),
                vs_hash=str(part.vs_hash).lower(),
                ib_hash=str(part.ib_hash).lower(),
                match_index_count=int(part.match_index_count),
                bone_count=int(part.bone_count),
                capture_store_base=int(capture_store_base),
                resource_suffix=resource_suffix,
            )
        )
        for local_bone in range(int(part.bone_count)):
            lod_global_to_capture_store[int(part.lod_global_bone_base) + local_bone] = int(capture_store_base) + local_bone

        native_palette = tuple(range(int(capture_store_base), int(capture_store_base) + int(part.bone_count)))
        palette_records.append(
            LocalPaletteRecord(
                object_name=str(part.object_name),
                ib_hash=str(part.ib_hash).lower(),
                match_index_count=int(part.match_index_count),
                chunk_index=int(part.draw_index),
                local_bone_count=int(part.bone_count),
                palette_values=native_palette,
                file_name=_build_lod_native_palette_file_name(variant_id, part),
                file_path=os.path.join(buffer_dir, _build_lod_native_palette_file_name(variant_id, part)),
                resource_suffix=f"{resource_suffix}_native",
                variant_id=f"{variant_id}__native",
            )
        )

    valid_lod_globals = set(lod_global_to_capture_store)
    canonical_to_lod = {
        int(entry.get("canonical_global_bone")): int(entry.get("mapped_lod_global_bone"))
        for entry in mapping_records
        if int(entry.get("mapped_lod_global_bone", -1)) in valid_lod_globals
    }

    effective_palettes = _effective_canonical_palettes(canonical_part_records, local_palette_records)
    eligible_chunk_suffixes: list[str] = []
    missing_chunks: list[dict] = []
    lod_host_map: list[dict] = []
    palette_overrides: list[dict] = []
    if canonical_to_lod:
        for palette_record in effective_palettes:
            translated_values: list[int] = []
            missing_globals: list[int] = []
            for canonical_global in tuple(int(value) for value in palette_record.palette_values):
                lod_global = canonical_to_lod.get(int(canonical_global), -1)
                capture_store_index = lod_global_to_capture_store.get(int(lod_global), -1)
                if capture_store_index < 0:
                    missing_globals.append(int(canonical_global))
                    continue
                translated_values.append(int(capture_store_index))

            covered = not missing_globals
            variant_palette_suffix = ""
            if covered:
                variant_palette_suffix = _build_lod_palette_resource_suffix(variant_id, palette_record)
                variant_file_name = _build_lod_palette_file_name(variant_id, palette_record)
                palette_records.append(
                    LocalPaletteRecord(
                        object_name=str(palette_record.object_name),
                        ib_hash=str(palette_record.ib_hash).lower(),
                        match_index_count=int(palette_record.match_index_count),
                        chunk_index=int(palette_record.chunk_index),
                        local_bone_count=int(palette_record.local_bone_count),
                        palette_values=tuple(translated_values),
                        file_name=variant_file_name,
                        file_path=os.path.join(buffer_dir, variant_file_name),
                        resource_suffix=variant_palette_suffix,
                        variant_id=variant_id,
                    )
                )
                eligible_chunk_suffixes.append(str(palette_record.resource_suffix))
                palette_overrides.append(
                    {
                        "base_resource_suffix": str(palette_record.resource_suffix),
                        "resource_suffix": variant_palette_suffix,
                        "ib_hash": str(palette_record.ib_hash).lower(),
                        "match_index_count": int(palette_record.match_index_count),
                        "chunk_index": int(palette_record.chunk_index),
                        "local_bone_count": int(palette_record.local_bone_count),
                        "file_name": variant_file_name,
                        "file_path": os.path.join(buffer_dir, variant_file_name),
                        "palette_values": translated_values,
                    }
                )
            else:
                missing_chunks.append(
                    {
                        "resource_suffix": str(palette_record.resource_suffix),
                        "ib_hash": str(palette_record.ib_hash).lower(),
                        "match_index_count": int(palette_record.match_index_count),
                        "chunk_index": int(palette_record.chunk_index),
                        "missing_globals": missing_globals,
                    }
                )

            lod_host_map.append(
                {
                    "variant_id": variant_id,
                    "base_resource_suffix": str(palette_record.resource_suffix),
                    "resource_suffix": variant_palette_suffix,
                    "ib_hash": str(palette_record.ib_hash).lower(),
                    "match_index_count": int(palette_record.match_index_count),
                    "chunk_index": int(palette_record.chunk_index),
                    "covered": covered,
                }
            )

    variants: list[dict] = []
    if canonical_to_lod:
        variants.append(
            {
                "variant_id": variant_id,
                "frameanalysis_dir": str(lod_variant.get("frameanalysis_dir", "") or ""),
                "shadow_host_hash": str(lod_variant.get("shadow_host_hash", "") or "").lower(),
                "shadow_host_match_index_count": int(lod_variant.get("shadow_host_match_index_count", -1)),
                "shadow_host_vs_hash": str(lod_variant.get("shadow_host_vs_hash", "") or "").lower(),
                "eligible_chunk_suffixes": eligible_chunk_suffixes,
                "missing_chunks": missing_chunks,
                "palette_overrides": palette_overrides,
            }
        )

    return {
        "capture_parts": capture_parts,
        "palette_records": palette_records,
        "variants": variants,
        "lod_host_map": lod_host_map,
    }


def _build_lod_part_records(draw_records) -> list[LodPartRecord]:
    part_records: list[LodPartRecord] = []
    next_lod_global_bone_base = 0
    for draw_record in sorted(draw_records, key=lambda item: item.draw_index):
        bone_count = int(draw_record.local_bone_count)
        if bone_count <= 0:
            raise ValueError(
                f"{draw_record.object_name}: invalid LOD local bone count {bone_count}; expected Blender numeric groups"
            )
        part_records.append(
            LodPartRecord(
                draw_index=int(draw_record.draw_index),
                object_name=str(draw_record.object_name),
                vs_hash=str(draw_record.vs_hash).lower(),
                ib_hash=str(draw_record.ib_hash).lower(),
                match_index_count=int(draw_record.match_index_count),
                bone_count=bone_count,
                lod_global_bone_base=next_lod_global_bone_base,
                vb2_path=str(draw_record.vb2_path),
                vs_t0_path=str(draw_record.vs_t0_path),
                vs_cb1_path=str(draw_record.vs_cb1_path),
                vs_cb1_first_constant=int(draw_record.vs_cb1_first_constant),
                vs_cb1_num_constants=int(draw_record.vs_cb1_num_constants),
            )
        )
        next_lod_global_bone_base += bone_count
    return part_records


def _build_lod_object_remaps(part_records: list[LodPartRecord]) -> list[dict]:
    remaps: list[dict] = []
    for part_record in part_records:
        remaps.append(
            {
                "object_name": str(part_record.object_name),
                "ib_hash": str(part_record.ib_hash).lower(),
                "match_index_count": int(part_record.match_index_count),
                "local_group_to_global_group": {
                    str(local_index): int(part_record.lod_global_bone_base) + local_index
                    for local_index in range(int(part_record.bone_count))
                },
            }
        )
    return remaps


def _build_lod_variant_id(frameanalysis_dir: str, part_records: list[LodPartRecord]) -> str:
    base_name = os.path.basename(os.path.abspath(frameanalysis_dir).rstrip("\\/")) or "lod"
    seed = "|".join(
        [
            base_name,
            *(f"{part.ib_hash}:{int(part.match_index_count)}:{int(part.draw_index)}" for part in part_records),
        ]
    )
    checksum = zlib.crc32(seed.lower().encode("utf-8")) & 0xFFFFFFFF
    return f"{base_name}_{checksum:08x}"


def _canonical_total_global_bones(part_records_payload: list[dict]) -> int:
    maximum = 0
    for part_record in part_records_payload:
        base = int(part_record.get("global_bone_base", 0))
        bone_count = int(part_record.get("capture_bone_count", part_record.get("bone_count", 0)))
        maximum = max(maximum, base + bone_count)
    return maximum


def _aggregate_group_infos(mesh_entries: list[tuple[object, dict]]) -> dict[int, dict]:
    aggregated: dict[int, dict] = {}
    for mesh_obj, remap_entry in mesh_entries:
        local_to_global = _normalize_local_to_global(remap_entry.get("local_group_to_global_group", {}))
        for global_bone in sorted({int(value) for value in local_to_global.values() if int(value) >= 0}):
            spatial_info = _infer_group_spatial_info_world(mesh_obj, str(global_bone), local_to_global)
            if spatial_info is None:
                continue
            aggregate = aggregated.get(global_bone)
            if aggregate is None:
                aggregated[global_bone] = {
                    "global_bone": int(global_bone),
                    "weighted_sum": spatial_info["center"] * float(spatial_info["total_weight"]),
                    "total_weight": float(spatial_info["total_weight"]),
                    "vertex_count": int(spatial_info["vertex_count"]),
                    "bounds_min": tuple(float(value) for value in spatial_info["bounds_min"]),
                    "bounds_max": tuple(float(value) for value in spatial_info["bounds_max"]),
                    "object_names": {str(mesh_obj.name)},
                }
                continue

            aggregate["weighted_sum"] = aggregate["weighted_sum"] + (
                spatial_info["center"] * float(spatial_info["total_weight"])
            )
            aggregate["total_weight"] += float(spatial_info["total_weight"])
            aggregate["vertex_count"] += int(spatial_info["vertex_count"])
            aggregate["bounds_min"] = (
                min(float(aggregate["bounds_min"][0]), float(spatial_info["bounds_min"][0])),
                min(float(aggregate["bounds_min"][1]), float(spatial_info["bounds_min"][1])),
                min(float(aggregate["bounds_min"][2]), float(spatial_info["bounds_min"][2])),
            )
            aggregate["bounds_max"] = (
                max(float(aggregate["bounds_max"][0]), float(spatial_info["bounds_max"][0])),
                max(float(aggregate["bounds_max"][1]), float(spatial_info["bounds_max"][1])),
                max(float(aggregate["bounds_max"][2]), float(spatial_info["bounds_max"][2])),
            )
            aggregate["object_names"].add(str(mesh_obj.name))

    for aggregate in aggregated.values():
        if float(aggregate["total_weight"]) > 0.0:
            aggregate["center"] = aggregate["weighted_sum"] / float(aggregate["total_weight"])
        else:
            aggregate["center"] = aggregate["weighted_sum"]
        aggregate["bounds_diag"] = _bounds_diag_length(aggregate["bounds_min"], aggregate["bounds_max"])
        aggregate["object_names"] = tuple(sorted(aggregate["object_names"]))
    return aggregated


def _build_exact_matches(
    canonical_infos: dict[int, dict],
    lod_infos: dict[int, dict],
) -> tuple[dict[int, tuple[int, float]], dict[int, str]]:
    best_lod_for_canonical: dict[int, tuple[int, float, float]] = {}
    best_canonical_for_lod: dict[int, tuple[int, float, float]] = {}

    for canonical_global, canonical_info in canonical_infos.items():
        best_lod = -1
        best_score = -1.0
        second_score = -1.0
        for lod_global, lod_info in lod_infos.items():
            score = _pair_score(canonical_info, lod_info)
            if score < 2.25:
                continue
            if score > best_score:
                second_score = best_score
                best_score = score
                best_lod = lod_global
            elif score > second_score:
                second_score = score
        if best_lod >= 0:
            best_lod_for_canonical[canonical_global] = (best_lod, best_score, second_score)

    for lod_global, lod_info in lod_infos.items():
        best_canonical = -1
        best_score = -1.0
        second_score = -1.0
        for canonical_global, canonical_info in canonical_infos.items():
            score = _pair_score(canonical_info, lod_info)
            if score < 2.25:
                continue
            if score > best_score:
                second_score = best_score
                best_score = score
                best_canonical = canonical_global
            elif score > second_score:
                second_score = score
        if best_canonical >= 0:
            best_canonical_for_lod[lod_global] = (best_canonical, best_score, second_score)

    accepted: dict[int, tuple[int, float]] = {}
    notes: dict[int, str] = {}
    for canonical_global, (lod_global, score, second_score) in best_lod_for_canonical.items():
        reverse = best_canonical_for_lod.get(lod_global)
        if reverse is None or int(reverse[0]) != canonical_global:
            continue
        reverse_second = float(reverse[2])
        if second_score >= 0.0 and abs(score - second_score) <= 0.15:
            notes[canonical_global] = "unresolved: exact pair is ambiguous on canonical side"
            continue
        if reverse_second >= 0.0 and abs(float(reverse[1]) - reverse_second) <= 0.15:
            notes[canonical_global] = "unresolved: exact pair is ambiguous on LOD side"
            continue
        accepted[canonical_global] = (lod_global, score)
        notes[canonical_global] = "exact spatial match"
    return accepted, notes


def _build_grouped_matches(
    canonical_infos: dict[int, dict],
    lod_infos: dict[int, dict],
    exact_matches: dict[int, tuple[int, float]],
) -> tuple[dict[int, tuple[int, float]], dict[int, str]]:
    grouped_matches: dict[int, tuple[int, float]] = {}
    notes: dict[int, str] = {}

    matched_canonical = set(exact_matches)
    matched_lod = {int(lod_global) for lod_global, _score in exact_matches.values()}
    remaining_lods = sorted(
        [lod_info for lod_global, lod_info in lod_infos.items() if lod_global not in matched_lod],
        key=lambda item: (-int(item["vertex_count"]), -float(item["total_weight"])),
    )

    for lod_info in remaining_lods:
        lod_global = int(lod_info["global_bone"])
        candidate_infos = []
        for canonical_global, canonical_info in canonical_infos.items():
            if canonical_global in matched_canonical:
                continue
            if not _is_group_candidate(canonical_info, lod_info):
                continue
            candidate_infos.append(canonical_info)
        candidate_infos.sort(
            key=lambda item: (
                -_bbox_overlap_score(item["bounds_min"], item["bounds_max"], lod_info["bounds_min"], lod_info["bounds_max"]),
                _center_distance(item["center"], lod_info["center"]),
                -_weight_similarity(item["total_weight"], lod_info["total_weight"]),
            )
        )
        selected: list[dict] = []
        best_fit = -1.0
        for candidate_info in candidate_infos:
            trial = selected + [candidate_info]
            trial_fit = _group_fit_score(lod_info, trial)
            if not selected:
                if trial_fit >= 1.85:
                    selected = trial
                    best_fit = trial_fit
                continue
            if trial_fit > best_fit + 0.08:
                selected = trial
                best_fit = trial_fit
        if not selected:
            continue
        if best_fit < 2.15:
            for candidate_info in selected:
                notes[int(candidate_info["global_bone"])] = "unresolved: grouped fit stayed too weak"
            continue
        selected_globals = sorted(int(candidate_info["global_bone"]) for candidate_info in selected)
        if len(selected_globals) == 1:
            grouped_matches[selected_globals[0]] = (lod_global, best_fit)
            notes[selected_globals[0]] = "grouped recovery selected a single fallback target"
            matched_canonical.add(selected_globals[0])
            matched_lod.add(lod_global)
            continue
        for canonical_global in selected_globals:
            grouped_matches[canonical_global] = (lod_global, best_fit)
            notes[canonical_global] = f"grouped bbox union -> lod {lod_global} ({len(selected_globals)} canonical groups)"
            matched_canonical.add(canonical_global)
        matched_lod.add(lod_global)
    return grouped_matches, notes


def _pair_score(canonical_info: dict, lod_info: dict) -> float:
    bbox_gap = _bbox_gap(canonical_info["bounds_min"], canonical_info["bounds_max"], lod_info["bounds_min"], lod_info["bounds_max"])
    diag = max(float(canonical_info["bounds_diag"]), float(lod_info["bounds_diag"]), 1.0e-6)
    if bbox_gap > max(0.01, diag * 0.75):
        return -1.0
    center_distance = _center_distance(canonical_info["center"], lod_info["center"])
    if center_distance > max(0.03, diag * 1.2):
        return -1.0
    overlap = _bbox_overlap_score(
        canonical_info["bounds_min"],
        canonical_info["bounds_max"],
        lod_info["bounds_min"],
        lod_info["bounds_max"],
    )
    weight_similarity = _weight_similarity(canonical_info["total_weight"], lod_info["total_weight"])
    vertex_similarity = _weight_similarity(canonical_info["vertex_count"], lod_info["vertex_count"])
    distance_score = 1.0 / (1.0 + center_distance * 40.0)
    gap_score = 1.0 / (1.0 + bbox_gap * 80.0)
    return overlap * 3.0 + weight_similarity + vertex_similarity * 0.75 + distance_score + gap_score


def _is_group_candidate(canonical_info: dict, lod_info: dict) -> bool:
    bbox_gap = _bbox_gap(canonical_info["bounds_min"], canonical_info["bounds_max"], lod_info["bounds_min"], lod_info["bounds_max"])
    diag = max(float(lod_info["bounds_diag"]), 1.0e-6)
    if bbox_gap > max(0.012, diag * 0.85):
        return False
    center_distance = _center_distance(canonical_info["center"], lod_info["center"])
    if center_distance > max(0.04, diag * 1.8):
        return False
    return True


def _group_fit_score(lod_info: dict, selected_infos: list[dict]) -> float:
    union_info = _combine_infos(selected_infos)
    overlap = _bbox_overlap_score(
        union_info["bounds_min"],
        union_info["bounds_max"],
        lod_info["bounds_min"],
        lod_info["bounds_max"],
    )
    weight_similarity = _weight_similarity(union_info["total_weight"], lod_info["total_weight"])
    center_distance = _center_distance(union_info["center"], lod_info["center"])
    distance_score = 1.0 / (1.0 + center_distance * 35.0)
    count_penalty = max(0.0, (len(selected_infos) - 1) * 0.08)
    return overlap * 3.5 + weight_similarity + distance_score - count_penalty


def _combine_infos(infos: list[dict]) -> dict:
    if not infos:
        raise ValueError("Cannot combine an empty info set")
    combined = {
        "bounds_min": infos[0]["bounds_min"],
        "bounds_max": infos[0]["bounds_max"],
        "weighted_sum": infos[0]["center"] * float(infos[0]["total_weight"]),
        "total_weight": float(infos[0]["total_weight"]),
    }
    for info in infos[1:]:
        combined["bounds_min"] = (
            min(float(combined["bounds_min"][0]), float(info["bounds_min"][0])),
            min(float(combined["bounds_min"][1]), float(info["bounds_min"][1])),
            min(float(combined["bounds_min"][2]), float(info["bounds_min"][2])),
        )
        combined["bounds_max"] = (
            max(float(combined["bounds_max"][0]), float(info["bounds_max"][0])),
            max(float(combined["bounds_max"][1]), float(info["bounds_max"][1])),
            max(float(combined["bounds_max"][2]), float(info["bounds_max"][2])),
        )
        combined["weighted_sum"] = combined["weighted_sum"] + (info["center"] * float(info["total_weight"]))
        combined["total_weight"] += float(info["total_weight"])
    combined["center"] = combined["weighted_sum"] / max(float(combined["total_weight"]), 1.0e-6)
    return combined


def _bbox_gap(bounds_min_a, bounds_max_a, bounds_min_b, bounds_max_b) -> float:
    gap_x = _axis_gap(bounds_min_a[0], bounds_max_a[0], bounds_min_b[0], bounds_max_b[0])
    gap_y = _axis_gap(bounds_min_a[1], bounds_max_a[1], bounds_min_b[1], bounds_max_b[1])
    gap_z = _axis_gap(bounds_min_a[2], bounds_max_a[2], bounds_min_b[2], bounds_max_b[2])
    return max(float(gap_x), float(gap_y), float(gap_z))


def _axis_gap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    if max_a < min_b:
        return min_b - max_a
    if max_b < min_a:
        return min_a - max_b
    return 0.0


def _bbox_overlap_score(bounds_min_a, bounds_max_a, bounds_min_b, bounds_max_b) -> float:
    return (
        _axis_overlap_score(bounds_min_a[0], bounds_max_a[0], bounds_min_b[0], bounds_max_b[0])
        + _axis_overlap_score(bounds_min_a[1], bounds_max_a[1], bounds_min_b[1], bounds_max_b[1])
        + _axis_overlap_score(bounds_min_a[2], bounds_max_a[2], bounds_min_b[2], bounds_max_b[2])
    ) / 3.0


def _axis_overlap_score(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    overlap = max(0.0, min(float(max_a), float(max_b)) - max(float(min_a), float(min_b)))
    span_a = max(1.0e-6, float(max_a) - float(min_a))
    span_b = max(1.0e-6, float(max_b) - float(min_b))
    return min(1.0, overlap / max(span_a, span_b))


def _bounds_diag_length(bounds_min, bounds_max) -> float:
    dx = float(bounds_max[0]) - float(bounds_min[0])
    dy = float(bounds_max[1]) - float(bounds_min[1])
    dz = float(bounds_max[2]) - float(bounds_min[2])
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _center_distance(center_a, center_b) -> float:
    return float((center_a - center_b).length)


def _weight_similarity(value_a: float | int, value_b: float | int) -> float:
    minimum = min(float(value_a), float(value_b))
    maximum = max(float(value_a), float(value_b), 1.0e-6)
    return minimum / maximum


def _effective_canonical_palettes(
    canonical_part_records: list[PartRecord],
    local_palette_records: list[LocalPaletteRecord],
) -> list[LocalPaletteRecord]:
    records_by_identity = {
        (str(record.ib_hash).lower(), int(record.match_index_count), int(record.chunk_index)): record
        for record in local_palette_records
        if not str(record.variant_id or "")
    }
    effective_records: list[LocalPaletteRecord] = []
    for part_record in canonical_part_records:
        key = (str(part_record.ib_hash).lower(), int(part_record.match_index_count), 0)
        record = records_by_identity.get(key)
        if record is None:
            palette_values = tuple(
                range(int(part_record.global_bone_base), int(part_record.global_bone_base) + int(part_record.bone_count))
            )
            record = LocalPaletteRecord(
                object_name=str(part_record.object_name),
                ib_hash=str(part_record.ib_hash).lower(),
                match_index_count=int(part_record.match_index_count),
                chunk_index=0,
                local_bone_count=len(palette_values),
                palette_values=palette_values,
                file_name=f"{str(part_record.ib_hash).lower()}-{int(part_record.match_index_count)}-0-Palette.buf",
                file_path="",
                resource_suffix=f"{str(part_record.ib_hash).lower()}_{int(part_record.match_index_count)}_0",
            )
        effective_records.append(record)
    return effective_records


def _lod_part_record_from_payload(payload: dict) -> LodPartRecord:
    return LodPartRecord(
        draw_index=int(payload.get("draw_index", 0)),
        object_name=str(payload.get("object_name", "")),
        vs_hash=str(payload.get("vs_hash", "")).lower(),
        ib_hash=str(payload.get("ib_hash", "")).lower(),
        match_index_count=int(payload.get("match_index_count", 0)),
        bone_count=int(payload.get("capture_bone_count", payload.get("bone_count", 0))),
        lod_global_bone_base=int(payload.get("lod_global_bone_base", 0)),
        vb2_path=str(payload.get("vb2_path", "")),
        vs_t0_path=str(payload.get("vs_t0_path", "")),
        vs_cb1_path=str(payload.get("vs_cb1_path", "")),
        vs_cb1_first_constant=int(payload.get("vs_cb1_first_constant", -1)),
        vs_cb1_num_constants=int(payload.get("vs_cb1_num_constants", -1)),
    )


def _build_lod_capture_resource_suffix(variant_id: str, part: LodPartRecord) -> str:
    return (
        f"{str(variant_id).lower()}_{str(part.ib_hash).lower()}_"
        f"{int(part.match_index_count)}_{int(part.draw_index)}_src"
    )


def _build_lod_native_palette_file_name(variant_id: str, part: LodPartRecord) -> str:
    return (
        f"{str(part.ib_hash).lower()}-{int(part.match_index_count)}-{int(part.draw_index)}-"
        f"{str(variant_id).lower()}-NativePalette.buf"
    )


def _build_lod_palette_resource_suffix(variant_id: str, palette_record: LocalPaletteRecord) -> str:
    return f"{str(variant_id).lower()}__{str(palette_record.resource_suffix)}"


def _build_lod_palette_file_name(variant_id: str, palette_record: LocalPaletteRecord) -> str:
    return (
        f"{str(palette_record.ib_hash).lower()}-{int(palette_record.match_index_count)}-"
        f"{int(palette_record.chunk_index)}-{str(variant_id).lower()}-Palette.buf"
    )
