"""High-level workflows for the Bone Merge Capture plugin."""

from __future__ import annotations

from .blender_ops import (
    apply_group_remaps_to_meshes,
    mesh_objects_from_target_names,
    merge_duplicate_alias_weights,
)
from .io import read_json
from .models import ObjectRemap, TargetObjectSpec


def scan_targets_and_generate_outputs(
    frameanalysis_dir: str,
    target_specs: list[TargetObjectSpec],
    output_dir: str | None = None,
    mapping_payload: dict | None = None,
):
    raise ValueError(
        "Legacy target scanning was removed. Use Analyze Main -> Candidate IB list -> "
        "Build Global Bone Pool -> Prepare Export."
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
