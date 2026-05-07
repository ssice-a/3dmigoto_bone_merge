"""Prepare BoneMerge export region parts and local palettes."""

from __future__ import annotations

import os

from ..constants import (
    BI4_MAX_BONE_COUNT,
    BONESTORE_INI_FILE_NAME,
    BUFFER_EXPORT_DIR_NAME,
    CAPTURE_MANIFEST_FILE_NAME,
    EXPORT_MANIFEST_FILE_NAME,
    BMC_TEXTURE_MARKS_PROP,
)
from .export_names import ini_filename_from_collection_name
from .hlsl_assets import export_required_hlsl
from .ini_export import materialize_bonestore_runtime, write_bonestore_ini
from .io import ensure_directory, read_json, write_json
from .models import LocalPaletteRecord
from .export_buffers import write_part_geometry_buffers
from .export_package import build_export_plan, write_part_palette_files
from .texture_marks import load_texture_mark_payload
from .vertex_groups import collect_weighted_numeric_vertex_groups


def prepare_export_collection(
    context,
    source_collection,
    build_collection=None,
    output_dir: str | None = None,
    internal_manifest_dir: str | None = None,
    capture_manifest_path: str | None = None,
    generate_ini: bool = True,
):
    if source_collection is None:
        raise ValueError("Export source collection is not set")
    _ = build_collection

    normalized_output_dir = ensure_directory(output_dir or context.scene.bmc_output_dir or context.scene.bmc_frameanalysis_dir)
    buffer_dir = ensure_directory(os.path.join(normalized_output_dir, BUFFER_EXPORT_DIR_NAME))
    hlsl_dir = export_required_hlsl(normalized_output_dir) if generate_ini else ""

    export_plan = build_export_plan(
        source_collection,
        _collect_used_numeric_vertex_groups,
        max_bones_per_part=BI4_MAX_BONE_COUNT,
    )
    capture_manifest = _read_capture_manifest_for_export(normalized_output_dir, capture_manifest_path)
    palette_records = write_part_palette_files(buffer_dir, export_plan)
    geometry_records = write_part_geometry_buffers(
        buffer_dir,
        export_plan.parts,
        dict(capture_manifest.get("vertex_layout_table", {}) or {}),
    )
    local_palette_records = [_local_palette_record_from_export_record(record) for record in palette_records]
    texture_mark_payload = _read_texture_marks_for_export(context, source_collection)
    object_records = []
    for part in export_plan.parts:
        for usage in part.object_usages:
            object_records.append(
                {
                    "object": usage.name,
                    "region_collection": part.region.collection_name,
                    "part_name": part.part_name,
                    "host_ib_hash": part.region.ib_hash,
                    "host_match_first_index": part.region.match_first_index,
                    "host_match_index_count": part.region.match_index_count,
                    "part_index": part.part_index,
                    "palette_file": part.palette_file_name,
                    "local_bone_count": len(part.palette_values),
                    "used_global_groups": list(usage.used_global_groups),
                }
            )

    manifest = {
        "export_source_collection": source_collection.name,
        "export_collection": source_collection.name,
        "ini_file_name": ini_filename_from_collection_name(source_collection.name),
        "buffer_dir": buffer_dir,
        "bonestore_namespace": "",
        "palettes": palette_records,
        "geometry_buffers": geometry_records,
        "texture_marks": texture_mark_payload,
        "objects": object_records,
        "warnings": list(export_plan.warnings),
        "note": (
            "Export Root Collection child collections are final IB regions. "
            "Each region exports one implicit part00 or explicit partNN collections. "
            "Prepare Export writes per-part palettes without mutating source vertex groups."
        ),
    }
    manifest_dir = ensure_directory(internal_manifest_dir or normalized_output_dir)
    manifest_path = write_json(os.path.join(manifest_dir, EXPORT_MANIFEST_FILE_NAME), manifest)
    bonestore_ini_path = regenerate_bonestore_runtime_files(
        output_dir=normalized_output_dir,
        capture_manifest_path=capture_manifest_path,
        export_manifest_path=manifest_path,
        local_palette_records=local_palette_records,
        write_ini=generate_ini,
    )
    return {
        "manifest_path": manifest_path,
        "bonestore_ini_path": bonestore_ini_path,
        "export_collection_name": source_collection.name,
        "output_dir": normalized_output_dir,
        "buffer_dir": buffer_dir,
        "hlsl_dir": hlsl_dir,
        "objects": len(object_records),
        "palettes": len(palette_records),
    }


def _read_capture_manifest_for_export(output_dir: str, capture_manifest_path: str | None) -> dict:
    manifest_path = os.path.abspath(capture_manifest_path or "") if capture_manifest_path else ""
    if not manifest_path or not os.path.exists(manifest_path):
        manifest_path = os.path.join(output_dir, CAPTURE_MANIFEST_FILE_NAME)
    if not os.path.exists(manifest_path):
        raise ValueError("Missing capture_manifest.json; run Analyze Main/Build Pool before export")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("capture_manifest.json is not an object")
    return manifest


def _collect_used_numeric_vertex_groups(mesh_obj) -> set[int]:
    used_groups = collect_weighted_numeric_vertex_groups(mesh_obj)
    if not used_groups:
        raise ValueError(f"{mesh_obj.name}: no weighted numeric vertex groups found")
    return used_groups


def regenerate_bonestore_runtime_files(
    output_dir: str,
    capture_manifest_path: str | None = None,
    export_manifest_path: str | None = None,
    local_palette_records: list[LocalPaletteRecord] | None = None,
    mapping_payload: dict | None = None,
    write_ini: bool = True,
) -> str:
    _ = mapping_payload
    manifest_path = os.path.abspath(capture_manifest_path or "") if capture_manifest_path else ""
    if not manifest_path or not os.path.exists(manifest_path):
        manifest_path = os.path.join(output_dir, CAPTURE_MANIFEST_FILE_NAME)
    if not os.path.exists(manifest_path):
        return os.path.join(output_dir, BONESTORE_INI_FILE_NAME) if write_ini else ""

    manifest = read_json(manifest_path)
    if not manifest.get("bone_pool_order"):
        return os.path.join(output_dir, BONESTORE_INI_FILE_NAME) if write_ini else ""
    palette_records = list(local_palette_records or [])
    normalized_export_manifest_path = os.path.abspath(export_manifest_path or "") if export_manifest_path else ""
    export_manifest = {}
    if not palette_records and normalized_export_manifest_path and os.path.exists(normalized_export_manifest_path):
        export_manifest = read_json(normalized_export_manifest_path)
        palette_records = [
            _local_palette_record_from_export_record(record)
            for record in export_manifest.get("palettes", [])
        ]
    elif normalized_export_manifest_path and os.path.exists(normalized_export_manifest_path):
        export_manifest = read_json(normalized_export_manifest_path)
    geometry_records = list(export_manifest.get("geometry_buffers", []) or []) if isinstance(export_manifest, dict) else []
    _attach_object_names_to_geometry_records(geometry_records, list(export_manifest.get("objects", []) or []))
    texture_mark_payload = dict(export_manifest.get("texture_marks", {}) or {}) if isinstance(export_manifest, dict) else {}
    runtime_plan = materialize_bonestore_runtime(output_dir, manifest, palette_records, geometry_records, texture_mark_payload)
    ini_file_name = _ini_file_name_from_export_manifest(export_manifest)
    runtime_plan["ini_file_name"] = ini_file_name
    ini_path = write_bonestore_ini(output_dir, runtime_plan, ini_file_name=ini_file_name) if write_ini else ""

    if normalized_export_manifest_path and os.path.exists(normalized_export_manifest_path):
        export_manifest = read_json(normalized_export_manifest_path)
        export_manifest["runtime"] = {
            "schema_version": int(runtime_plan.get("schema_version", 2)),
            "namespace": str(runtime_plan.get("namespace", "")),
            "ini_file_name": ini_file_name,
            "global_bone_count": int(runtime_plan.get("global_bone_count", 0) or 0),
            "capture_records": list(runtime_plan.get("capture_records", []) or []),
            "lod_capture_records": list(runtime_plan.get("lod_capture_records", []) or []),
            "lod_replay_links": list(runtime_plan.get("lod_replay_links", []) or []),
            "lod_key_annotations": list(runtime_plan.get("lod_key_annotations", []) or []),
            "geometry": list(runtime_plan.get("geometry", []) or []),
            "textures": list(runtime_plan.get("textures", []) or []),
            "shadow_stage": dict(runtime_plan.get("shadow_stage", {}) or {}),
            "shadow_replay_plan": dict(runtime_plan.get("shadow_replay_plan", {}) or {}),
            "lod_shadow_replay_plan": dict(runtime_plan.get("lod_shadow_replay_plan", {}) or {}),
            "buffers": dict(runtime_plan.get("buffers", {}) or {}),
        }
        write_json(normalized_export_manifest_path, export_manifest)
    return ini_path


def _ini_file_name_from_export_manifest(export_manifest: dict) -> str:
    if not isinstance(export_manifest, dict):
        return BONESTORE_INI_FILE_NAME
    explicit_name = str(export_manifest.get("ini_file_name", "") or "").strip()
    if explicit_name:
        return ini_filename_from_collection_name(explicit_name)
    collection_name = str(
        export_manifest.get("export_source_collection", "")
        or export_manifest.get("export_collection", "")
        or ""
    )
    return ini_filename_from_collection_name(collection_name)


def _read_texture_marks_for_export(context, source_collection) -> dict:
    raw_payload = ""
    collection_get = getattr(source_collection, "get", None)
    if source_collection is not None and callable(collection_get):
        raw_payload = str(collection_get(BMC_TEXTURE_MARKS_PROP, "") or "")
    if not raw_payload and getattr(context.scene, "bmc_texture_marks_json", ""):
        raw_payload = str(context.scene.bmc_texture_marks_json or "")
    payload = load_texture_mark_payload(raw_payload)
    return payload if payload.get("marks") else {}


def _attach_object_names_to_geometry_records(geometry_records: list[dict], object_records: list[dict]) -> None:
    names_by_part: dict[tuple[str, int, int, int], list[str]] = {}
    for object_record in object_records:
        key = (
            str(object_record.get("host_ib_hash", "") or object_record.get("ib_hash", "") or "").lower(),
            int(object_record.get("host_match_first_index", object_record.get("match_first_index", 0)) or 0),
            int(object_record.get("host_match_index_count", object_record.get("match_index_count", 0)) or 0),
            int(object_record.get("part_index", object_record.get("chunk_index", 0)) or 0),
        )
        object_name = str(object_record.get("object", object_record.get("object_name", "")) or "").strip()
        if not key[0] or key[2] <= 0 or not object_name:
            continue
        bucket = names_by_part.setdefault(key, [])
        if object_name not in bucket:
            bucket.append(object_name)

    if not names_by_part:
        return

    for geometry_record in geometry_records:
        if geometry_record.get("object_names"):
            continue
        key = (
            str(geometry_record.get("ib_hash", "") or "").lower(),
            int(geometry_record.get("match_first_index", 0) or 0),
            int(geometry_record.get("match_index_count", 0) or 0),
            int(geometry_record.get("part_index", geometry_record.get("chunk_index", 0)) or 0),
        )
        names = names_by_part.get(key)
        if names:
            geometry_record["object_names"] = list(names)


def _local_palette_record_from_export_record(record: dict) -> LocalPaletteRecord:
    return LocalPaletteRecord(
        object_name=str(record.get("region_collection", record.get("chunk_collection", ""))),
        ib_hash=str(record.get("ib_hash", "")).lower(),
        match_index_count=int(record.get("match_index_count", 0)),
        chunk_index=int(record.get("part_index", record.get("chunk_index", 0))),
        local_bone_count=int(record.get("local_bone_count", 0)),
        palette_values=tuple(int(value) for value in record.get("palette_values", []) or []),
        file_name=str(record.get("file_name", "")),
        file_path=str(record.get("file_path", "")),
        resource_suffix=str(record.get("resource_suffix", "")),
        variant_id=str(record.get("variant_id", "")),
        match_first_index=int(record.get("match_first_index", 0) or 0),
    )
