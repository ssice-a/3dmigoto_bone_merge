"""High-level workflows for the Bone Merge Capture plugin."""

from __future__ import annotations

import os

from ..constants import BI4_MAX_BONE_COUNT, CAPTURE_MANIFEST_FILE_NAME, LOCAL_PREVIOUS_ROW_OFFSET
from .blender_ops import (
    apply_group_remaps_to_meshes,
    mesh_objects_from_target_names,
    merge_duplicate_alias_weights,
)
from .frameanalysis import build_part_records, find_draw_records_for_targets, resolve_output_dir
from .hlsl_assets import export_required_hlsl
from .ini_export import build_bonestore_namespace, write_bonestore_ini
from .io import read_json, write_json
from .models import ObjectRemap, ScanGenerateResult, TargetObjectSpec


def scan_targets_and_generate_outputs(
    frameanalysis_dir: str,
    target_specs: list[TargetObjectSpec],
    output_dir: str | None = None,
    merge_same_bone_groups: bool = True,
) -> ScanGenerateResult:
    normalized_frameanalysis_dir = os.path.abspath(frameanalysis_dir)
    normalized_output_dir = resolve_output_dir(normalized_frameanalysis_dir, output_dir)

    draw_records, warnings = find_draw_records_for_targets(normalized_frameanalysis_dir, target_specs)
    if not draw_records:
        raise ValueError("No matching draw records found for current target list")

    part_records = build_part_records(draw_records)
    bone_aliases = []
    object_remaps = _build_object_remaps(part_records)
    selected_vs_hashes = sorted({part.vs_hash.lower() for part in part_records})
    total_global_bones = sum(part.bone_count for part in part_records)

    warning_list = list(warnings)
    oversized_parts = [part for part in part_records if part.bone_count > BI4_MAX_BONE_COUNT]
    if oversized_parts:
        oversize_summary = ", ".join(
            f"{part.object_name}({part.bone_count})" for part in oversized_parts[:5]
        )
        raise ValueError(
            "One or more parts exceed the BI4 per-draw limit of 256 bones: "
            f"{oversize_summary}"
        )

    hlsl_output_dir = export_required_hlsl(normalized_output_dir)

    manifest_payload = {
        "frameanalysis_dir": normalized_frameanalysis_dir,
        "hlsl_dir": hlsl_output_dir,
        "bonestore_namespace": build_bonestore_namespace(normalized_output_dir),
        "selected_vs_hashes": selected_vs_hashes,
        "local_palette_protocol": {
            "palette_format": "R32_UINT",
            "palette_stride_bytes": 4,
            "palette_entry_meaning": "localBone -> globalBone",
            "local_palette_meta_format": "R32_FLOAT",
            "local_palette_meta_fields": ["local_bone_count"],
            "global_layout": {
                "reserved_rows": 3,
                "current_row_formula": "3 + globalBone*3 + rowInBone",
                "previous_row_formula": "100000 + 3 + globalBone*3 + rowInBone",
            },
            "local_layout": {
                "reserved_rows": 3,
                "current_row_formula": "3 + localBone*3 + rowInBone",
                "previous_row_formula": f"{LOCAL_PREVIOUS_ROW_OFFSET} + 3 + localBone*3 + rowInBone",
            },
            "bone_limit_per_draw": BI4_MAX_BONE_COUNT,
            "recommended_chain": {
                "stage1_source_capture": [
                    "CustomShader_ExtractCB1",
                    "cs-t2 = ResourceBoneMeta_<source ib>",
                    "CustomShader_RecordBones",
                ],
                "stage2_final_consuming_draw": [
                    "CustomShader_ExtractCB1",
                    "cs-t2 = ResourceLocalPalette_<chunk>",
                    "cs-t3 = ResourceLocalPaletteMeta_<chunk>",
                    "CustomShader_GatherBones",
                    "vs-t0 = ResourceLocalFakeT0_SRV",
                    "CustomShader_RedirectCB1",
                    "vs-cb1 = ResourceFakeCB1",
                ],
            },
        },
        "part_records": [
            {
                "draw_index": part.draw_index,
                "object_name": part.object_name,
                "vs_hash": part.vs_hash,
                "ib_hash": part.ib_hash,
                "match_index_count": part.match_index_count,
                "capture_bone_count": part.bone_count,
                "global_bone_base": part.global_bone_base,
                "vb2_path": part.vb2_path,
                "vs_t0_path": part.vs_t0_path,
                "vs_cb1_path": part.vs_cb1_path,
                "vs_cb1_first_constant": part.vs_cb1_first_constant,
                "vs_cb1_num_constants": part.vs_cb1_num_constants,
            }
            for part in part_records
        ],
        "object_remaps": [
            {
                "object_name": remap.object_name,
                "ib_hash": remap.ib_hash,
                "match_index_count": remap.match_index_count,
                "local_group_to_global_group": remap.local_group_to_global_group,
            }
            for remap in object_remaps
        ],
        "local_palettes": [],
        "bone_aliases": [
            {
                "src_draw_index": alias.src_draw_index,
                "src_object_name": alias.src_object_name,
                "src_ib_hash": alias.src_ib_hash,
                "src_local_bone": alias.src_local_bone,
                "src_global_bone": alias.src_global_bone,
                "canonical_draw_index": alias.canonical_draw_index,
                "canonical_object_name": alias.canonical_object_name,
                "canonical_ib_hash": alias.canonical_ib_hash,
                "canonical_local_bone": alias.canonical_local_bone,
                "canonical_global_bone": alias.canonical_global_bone,
                "confidence": alias.confidence,
            }
            for alias in bone_aliases
        ],
    }

    manifest_path = write_json(os.path.join(normalized_output_dir, CAPTURE_MANIFEST_FILE_NAME), manifest_payload)
    ini_path = write_bonestore_ini(normalized_output_dir, part_records, [])

    return ScanGenerateResult(
        frameanalysis_dir=normalized_frameanalysis_dir,
        manifest_path=manifest_path,
        ini_path=ini_path,
        scanned_parts=len(part_records),
        total_global_bones=total_global_bones,
        warnings=tuple(warning_list),
    )


def apply_vertex_group_remap_for_target_names(context, manifest_path: str, target_object_names: list[str]):
    manifest = read_json(manifest_path)
    mesh_objects, missing_names = mesh_objects_from_target_names(context, target_object_names)
    if not mesh_objects:
        raise ValueError("No listed mesh objects were found in the current Blender scene")
    result = apply_group_remaps_to_meshes(mesh_objects, manifest)
    skipped_objects = list(result.skipped_objects)
    skipped_objects.extend(f"{name}: object not found in scene" for name in missing_names)
    return result.__class__(
        target_objects=len(target_object_names),
        updated_objects=result.updated_objects,
        renamed_groups=result.renamed_groups,
        skipped_objects=tuple(skipped_objects),
    )


def merge_duplicate_bone_weights_for_target_names(context, target_object_names: list[str], alias_entries: list[dict]):
    mesh_objects, missing_names = mesh_objects_from_target_names(context, target_object_names)
    if not mesh_objects:
        raise ValueError("No listed mesh objects were found in the current Blender scene")
    result = merge_duplicate_alias_weights(mesh_objects, alias_entries)
    skipped_objects = list(result.skipped_objects)
    skipped_objects.extend(f"{name}: object not found in scene" for name in missing_names)
    return result.__class__(
        target_objects=len(target_object_names),
        updated_objects=result.updated_objects,
        merged_aliases=result.merged_aliases,
        skipped_objects=tuple(skipped_objects),
    )


def _build_object_remaps(part_records) -> list[ObjectRemap]:
    remaps: list[ObjectRemap] = []
    for part_record in part_records:
        remaps.append(
            ObjectRemap(
                object_name=part_record.object_name,
                ib_hash=part_record.ib_hash,
                match_index_count=part_record.match_index_count,
                local_group_to_global_group={
                    str(local_index): part_record.global_bone_base + local_index
                    for local_index in range(part_record.bone_count)
                },
            )
        )
    return remaps
