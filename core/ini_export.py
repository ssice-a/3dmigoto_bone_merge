"""BoneStore runtime INI and static-buffer generation."""

from __future__ import annotations

import os
import re
import struct
from pathlib import Path

from ..constants import BONESTORE_INI_FILE_NAME, BUFFER_EXPORT_DIR_NAME
from .data_types import get_runtime_shader_filters
from .export_names import ini_filename_from_collection_name
from .io import ensure_directory, write_uint32_buffer
from .models import LocalPaletteRecord
from .texture_converter import write_game_texture
from .texture_marks import marked_texture_bindings, validate_texture_hash


_SAFE_SUFFIX_RE = re.compile(r"[^0-9A-Za-z_]+")
_SHADER_HASH_RE = re.compile(r"^[0-9a-fA-F]{16}$")

_MAIN_CAPTURE_BONE_MAP_FILE = "MainCaptureBoneMap.buf"
_LOD_CAPTURE_BONE_MAP_FILE = "LodCaptureBoneMap.buf"
_WHITE_SHADOW_TEXTURE_FILE = "Texture/white.dds"
_CAPTURE_BONE_MAP_HEADER_UINTS = 4
_CAPTURE_BONE_RECORD_STRIDE = 4
_CAPTURE_BONE_PAIR_STRIDE = 2
_CAPTURE_FLAG_MAIN = 0
_CAPTURE_FLAG_LOD = 1
_MAX_INSTANCE_SLOTS = 4
_GLOBAL_BONE_POOL_ROWS_PER_SLOT = 200000
_LOCAL_BONE_POOL_ROWS_PER_SLOT = 2048
_CB1_ROWS = 4096
_RUNTIME_STATE_ROWS = 64


def materialize_bonestore_runtime(
    output_directory: str,
    capture_manifest: dict,
    local_palette_records: list[LocalPaletteRecord],
    geometry_records: list[dict] | None = None,
    texture_mark_payload: dict | None = None,
    filter_residual: bool = True,
) -> dict:
    """Write static runtime buffers and return the normalized runtime plan."""

    normalized_output_dir = ensure_directory(output_directory)
    buffer_dir = ensure_directory(os.path.join(normalized_output_dir, BUFFER_EXPORT_DIR_NAME))

    capture_records, main_capture_bone_map = _build_main_capture_bone_map(capture_manifest)
    lod_records, lod_capture_bone_map = _build_lod_capture_bone_map(capture_manifest)
    palette_records = _normalize_palette_records(normalized_output_dir, local_palette_records)
    geometry_payloads = _normalize_geometry_records(normalized_output_dir, geometry_records or [])
    _attach_palette_metadata_to_geometry(geometry_payloads, palette_records)
    texture_records, texture_warnings = _materialize_texture_records(normalized_output_dir, texture_mark_payload or {})
    shader_filter_overrides = _runtime_shader_filter_overrides(
        capture_manifest,
        filter_residual=filter_residual,
    )
    lod_replay_links = _build_lod_replay_links(capture_manifest, geometry_payloads)
    lod_key_annotations = _build_lod_key_annotations(capture_manifest, geometry_payloads)
    shadow_replay_plan = _build_shadow_replay_plan(capture_manifest, geometry_payloads)
    lod_shadow_replay_plan = _build_lod_shadow_replay_plan(capture_manifest, geometry_payloads, lod_replay_links, lod_records)
    if _shadow_plan_needs_white_texture(shadow_replay_plan) or _shadow_plan_needs_white_texture(lod_shadow_replay_plan):
        _write_white_shadow_texture(os.path.join(normalized_output_dir, _WHITE_SHADOW_TEXTURE_FILE))
    _validate_palette_globals(capture_manifest, palette_records)

    main_capture_bone_map_path = write_uint32_buffer(
        os.path.join(buffer_dir, _MAIN_CAPTURE_BONE_MAP_FILE),
        main_capture_bone_map or [0, _CAPTURE_BONE_MAP_HEADER_UINTS, 0, 0],
    )
    lod_capture_bone_map_path = ""
    if lod_capture_bone_map:
        lod_capture_bone_map_path = write_uint32_buffer(
            os.path.join(buffer_dir, _LOD_CAPTURE_BONE_MAP_FILE),
            lod_capture_bone_map,
        )

    buffers = {
        "main_capture_bone_map": _resource_file_payload(
            main_capture_bone_map_path,
            normalized_output_dir,
            len(main_capture_bone_map or [0, _CAPTURE_BONE_MAP_HEADER_UINTS, 0, 0]),
        ),
    }
    if lod_capture_bone_map_path:
        buffers["lod_capture_bone_map"] = _resource_file_payload(
            lod_capture_bone_map_path,
            normalized_output_dir,
            len(lod_capture_bone_map),
        )

    return {
        "schema_version": 2,
        "namespace": "",
        "global_bone_count": _global_bone_count(capture_manifest),
        "shadow_vs_hashes": _runtime_shadow_vs_hashes(capture_manifest),
        "filter_residual": bool(filter_residual),
        "shader_filter_overrides": shader_filter_overrides,
        "visible_replay_excluded_filter_indices": _visible_replay_excluded_filter_indices(shader_filter_overrides),
        "capture_records": capture_records,
        "lod_capture_records": lod_records,
        "lod_replay_links": lod_replay_links,
        "lod_key_annotations": lod_key_annotations,
        "palettes": palette_records,
        "geometry": geometry_payloads,
        "textures": texture_records,
        "texture_warnings": texture_warnings,
        "shadow_stage": _normalize_shadow_stage(capture_manifest),
        "shadow_replay_plan": shadow_replay_plan,
        "lod_shadow_replay_plan": lod_shadow_replay_plan,
        "buffers": buffers,
    }


def write_bonestore_ini(output_directory: str, runtime_plan: dict, ini_file_name: str | None = None) -> str:
    os.makedirs(output_directory, exist_ok=True)
    file_name = ini_filename_from_collection_name(ini_file_name or runtime_plan.get("ini_file_name", "") or BONESTORE_INI_FILE_NAME)
    ini_path = os.path.join(output_directory, file_name)
    with open(ini_path, "w", encoding="utf-8", newline="\n") as file_handle:
        file_handle.write(build_bonestore_ini_content(runtime_plan))
    return ini_path


def build_bonestore_ini_content(runtime_plan: dict) -> str:
    lines: list[str] = []

    lines.extend(
        [
            "; Auto-generated by Bone Merge Capture",
            "; Runtime schema v2:",
            ";   main capture: record index -> MainCaptureBoneMap",
            ";   LOD capture:  record index -> LodCaptureBoneMap",
            ";   consume:      part local bone -> canonical global BoneStore pool",
            ";   runtime:      one INI, inline TextureOverride flow, shared HLSL algorithms",
            "",
        ]
    )
    lines.extend(_shader_override_sections(runtime_plan))
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(_resource_sections(runtime_plan))
    lines.append("")
    lines.extend(_custom_shader_sections(runtime_plan))
    lines.append("")
    lines.extend(_local_palette_sections(runtime_plan))
    if runtime_plan.get("palettes"):
        lines.append("")
    lines.extend(_geometry_resource_sections(runtime_plan))
    if runtime_plan.get("geometry"):
        lines.append("")
    lines.extend(_texture_resource_sections(runtime_plan))
    if runtime_plan.get("textures"):
        lines.append("")
    lines.extend(_texture_override_sections(runtime_plan))
    if runtime_plan.get("textures"):
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(_texture_hash_override_sections(runtime_plan))
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(_frame_lifecycle_sections())
    return "\n".join(lines).rstrip() + "\n"


def _build_main_capture_bone_map(capture_manifest: dict) -> tuple[list[dict], list[int]]:
    records: list[dict] = []
    record_uints: list[int] = []
    pair_uints: list[int] = []

    for pool_record in capture_manifest.get("bone_pool_order", []) or []:
        if not bool(pool_record.get("bone_capture_available", pool_record.get("shadow_capture_ready", False))):
            continue
        used_indices = _used_local_indices(pool_record)
        if not used_indices:
            continue

        record_index = len(records)
        pair_base = len(pair_uints) // _CAPTURE_BONE_PAIR_STRIDE
        capture_store_base = int(pool_record.get("capture_store_base", pool_record.get("global_bone_base", 0)) or 0)
        for compact_index, source_local_bone in enumerate(used_indices):
            pair_uints.extend([int(source_local_bone), int(capture_store_base) + compact_index])
        record_uints.extend([pair_base, len(used_indices), len(used_indices), _CAPTURE_FLAG_MAIN])

        records.append(
            {
                "record_index": record_index,
                "ib_hash": str(pool_record.get("ib_hash", "") or "").lower(),
                "match_first_index": int(pool_record.get("match_first_index", 0) or 0),
                "match_index_count": int(pool_record.get("match_index_count", 0) or 0),
                "capture_store_base": capture_store_base,
                "local_bone_count": len(used_indices),
                "source_local_indices": list(used_indices),
                "capture_pair_base": pair_base,
                "capture_pair_count": len(used_indices),
                "dispatch_rows": len(used_indices) * 3,
            }
        )

    pair_table_uint_base = _CAPTURE_BONE_MAP_HEADER_UINTS + len(record_uints)
    header = [len(records), pair_table_uint_base, len(pair_uints) // _CAPTURE_BONE_PAIR_STRIDE, _CAPTURE_FLAG_MAIN]
    return records, header + record_uints + pair_uints


def _build_lod_capture_bone_map(capture_manifest: dict) -> tuple[list[dict], list[int]]:
    records: list[dict] = []
    record_uints: list[int] = []
    pair_uints: list[int] = []

    for lod_record in capture_manifest.get("lod_capture_records", []) or []:
        raw_pairs = list(lod_record.get("scatter_pairs", []) or [])
        pairs = [
            (
                int(pair.get("lod_local_bone", -1)),
                int(pair.get("canonical_global_bone", -1)),
            )
            for pair in raw_pairs
            if int(pair.get("lod_local_bone", -1)) >= 0 and int(pair.get("canonical_global_bone", -1)) >= 0
        ]
        if not pairs:
            continue

        record_index = len(records)
        pair_base = len(pair_uints) // _CAPTURE_BONE_PAIR_STRIDE
        for lod_local_bone, canonical_global_bone in pairs:
            pair_uints.extend([lod_local_bone, canonical_global_bone])
        record_uints.extend([pair_base, len(pairs), int(lod_record.get("lod_local_bone_count", 0) or 0), _CAPTURE_FLAG_LOD])
        records.append(
            {
                "record_index": record_index,
                "lod_record_key": str(lod_record.get("lod_record_key", "") or ""),
                "ib_hash": str(lod_record.get("lod_ib_hash", "") or "").lower(),
                "match_first_index": int(lod_record.get("lod_match_first_index", 0) or 0),
                "match_index_count": int(lod_record.get("lod_match_index_count", 0) or 0),
                "capture_draw_indices": _int_list(lod_record.get("lod_capture_draw_indices", lod_record.get("capture_draw_indices", []))),
                "import_draw_index": int(lod_record.get("lod_import_draw_index", lod_record.get("import_draw_index", -1)) or -1),
                "canonical_global_bones": sorted({int(canonical_global_bone) for _lod_local_bone, canonical_global_bone in pairs}),
                "pair_base": pair_base,
                "pair_count": len(pairs),
                "dispatch_rows": len(pairs) * 3,
            }
        )

    if not records:
        return records, []
    pair_table_uint_base = _CAPTURE_BONE_MAP_HEADER_UINTS + len(record_uints)
    header = [len(records), pair_table_uint_base, len(pair_uints) // _CAPTURE_BONE_PAIR_STRIDE, _CAPTURE_FLAG_LOD]
    return records, header + record_uints + pair_uints


def _normalize_palette_records(output_directory: str, local_palette_records: list[LocalPaletteRecord]) -> list[dict]:
    normalized_records: list[dict] = []
    seen_suffixes: set[str] = set()
    buffer_dir = ensure_directory(os.path.join(output_directory, BUFFER_EXPORT_DIR_NAME))
    for record in local_palette_records:
        match_first_index = int(getattr(record, "match_first_index", 0) or 0)
        part_index = int(record.chunk_index)
        suffix = _safe_suffix(
            record.resource_suffix
            or f"{record.ib_hash}_{record.match_index_count}_{match_first_index}_part{part_index:02d}"
        )
        if suffix in seen_suffixes:
            continue
        seen_suffixes.add(suffix)
        palette_values = tuple(int(value) for value in record.palette_values)
        file_name = record.file_name or f"{suffix}-PartLocalToGlobalBoneMap.buf"
        file_path = record.file_path or os.path.join(buffer_dir, file_name)
        if not os.path.exists(file_path):
            write_uint32_buffer(file_path, palette_values)
        normalized_records.append(
            {
                "object_name": str(record.object_name),
                "ib_hash": str(record.ib_hash).lower(),
                "match_first_index": match_first_index,
                "match_index_count": int(record.match_index_count),
                "part_index": part_index,
                "chunk_index": part_index,
                "local_bone_count": int(record.local_bone_count),
                "palette_values": list(palette_values),
                "file_name": file_name,
                "file_path": file_path,
                "resource_suffix": suffix,
                "variant_id": str(record.variant_id or ""),
                "filename": _resource_filename(file_path, output_directory),
            }
        )
    return normalized_records


def _normalize_geometry_records(output_directory: str, geometry_records: list[dict]) -> list[dict]:
    normalized_records: list[dict] = []
    seen_suffixes: set[str] = set()
    for record in geometry_records:
        ib_hash = str(record.get("ib_hash", "") or "").lower()
        match_index_count = int(record.get("match_index_count", 0) or 0)
        match_first_index = int(record.get("match_first_index", 0) or 0)
        part_index = int(record.get("part_index", 0) or 0)
        suffix = _safe_suffix(f"{ib_hash}_{match_index_count}_{match_first_index}_part{part_index:02d}")
        if suffix in seen_suffixes:
            continue
        seen_suffixes.add(suffix)

        index_buffer = dict(record.get("index_buffer", {}) or {})
        vertex_buffers = {}
        for slot_name, vertex_buffer in sorted(dict(record.get("vertex_buffers", {}) or {}).items()):
            slot = str(slot_name or "").lower()
            role = str(vertex_buffer.get("role", _slot_resource_role(slot)) or _slot_resource_role(slot))
            file_path = str(vertex_buffer.get("file_path", "") or "")
            vertex_buffers[slot] = {
                "slot": slot,
                "role": role,
                "resource_name": f"ResourcePart_{suffix}_{role}",
                "file_name": str(vertex_buffer.get("file_name", "") or os.path.basename(file_path)),
                "file_path": file_path,
                "filename": _resource_filename(file_path, output_directory) if file_path else str(vertex_buffer.get("file_name", "") or ""),
                "stride": int(vertex_buffer.get("stride", 0) or 0),
                "vertex_count": int(vertex_buffer.get("vertex_count", 0) or 0),
            }

        index_file_path = str(index_buffer.get("file_path", "") or "")
        normalized_records.append(
            {
                "region_collection": str(record.get("region_collection", "") or ""),
                "part_name": str(record.get("part_name", f"part{part_index:02d}") or f"part{part_index:02d}"),
                "object_names": _normalize_object_names(record),
                "ib_hash": ib_hash,
                "match_first_index": match_first_index,
                "match_index_count": match_index_count,
                "part_index": part_index,
                "resource_suffix": suffix,
                "index_count": int(index_buffer.get("index_count", 0) or 0),
                "object_draws": _normalize_object_draws(record, int(index_buffer.get("index_count", 0) or 0)),
                "index_resource_name": f"ResourcePart_{suffix}_Index",
                "index_filename": _resource_filename(index_file_path, output_directory) if index_file_path else str(index_buffer.get("file_name", "") or ""),
                "vertex_buffers": vertex_buffers,
            }
        )
    return normalized_records


def _normalize_object_draws(record: dict, index_count: int) -> list[dict]:
    draws: list[dict] = []
    for raw_draw in record.get("object_draws", []) or []:
        draw = dict(raw_draw or {})
        count = int(draw.get("index_count", 0) or 0)
        if count <= 0:
            continue
        start_index = int(draw.get("start_index", 0) or 0)
        if start_index < 0 or start_index >= int(index_count):
            continue
        count = min(count, int(index_count) - start_index)
        draws.append(
            {
                "object_name": str(draw.get("object_name", "") or ""),
                "start_index": start_index,
                "index_count": count,
                "base_vertex": int(draw.get("base_vertex", 0) or 0),
                "start_vertex": int(draw.get("start_vertex", 0) or 0),
                "vertex_count": int(draw.get("vertex_count", 0) or 0),
            }
        )
    return draws


def _materialize_texture_records(output_directory: str, texture_mark_payload: dict) -> tuple[list[dict], list[str]]:
    """Copy/convert marked textures and return one hash-style replacement record per texture hash."""

    records_by_hash: dict[str, dict] = {}
    warnings: list[str] = []

    for binding in marked_texture_bindings(texture_mark_payload):
        texture_hash = str(binding.get("hash", "") or "").strip().lower()
        if not validate_texture_hash(texture_hash):
            warnings.append(f"Skipped texture candidate with invalid hash: {texture_hash or '<empty>'}")
            continue

        source_path = str(binding.get("source_path", "") or "").strip()
        if not source_path:
            warnings.append(f"Skipped texture {texture_hash}: missing source path")
            continue
        source = Path(source_path)
        if not source.is_file():
            warnings.append(f"Skipped texture {texture_hash}: source file not found: {source_path}")
            continue

        source_key = os.path.normcase(os.path.abspath(str(source)))
        slot = str(binding.get("slot", "") or "").strip().lower()
        semantic = str(binding.get("semantic", "") or "").strip().lower()
        semantic_index = int(binding.get("semantic_index", 0) or 0)

        existing = records_by_hash.get(texture_hash)
        if existing is not None:
            if existing["_source_key"] != source_key:
                raise ValueError(
                    f"Texture hash {texture_hash} has conflicting replacement sources: "
                    f"{existing.get('source_path')} and {source_path}"
                )
            _append_unique(existing["slots"], slot)
            _append_unique(existing["semantics"], _texture_semantic_label(semantic, semantic_index))
            _append_unique(existing["region_keys"], str(binding.get("region_key", "") or ""))
            _append_unique(existing["draw_keys"], str(binding.get("draw_key", "") or ""))
            continue

        semantic_slug = _texture_semantic_slug(semantic, semantic_index, slot)
        file_name = f"{texture_hash}_{semantic_slug}.dds"
        relative_filename = f"Texture/{file_name}"
        destination_path = os.path.join(output_directory, relative_filename)
        written_path = write_game_texture(
            source,
            destination_path,
            slot=slot,
            semantic=semantic,
        )
        records_by_hash[texture_hash] = {
            "_source_key": source_key,
            "hash": texture_hash,
            "resource_name": f"ResourceBMCTexture_{texture_hash}",
            "filename": _resource_filename(str(written_path), output_directory),
            "file_path": str(written_path),
            "source_path": source_path,
            "slot": slot,
            "semantic": semantic,
            "semantic_index": semantic_index,
            "slots": [slot] if slot else [],
            "semantics": [_texture_semantic_label(semantic, semantic_index)],
            "region_keys": [str(binding.get("region_key", "") or "")],
            "draw_keys": [str(binding.get("draw_key", "") or "")],
        }

    records = []
    for texture_hash in sorted(records_by_hash):
        record = dict(records_by_hash[texture_hash])
        record.pop("_source_key", None)
        record["slots"] = [value for value in record.get("slots", []) if value]
        record["semantics"] = [value for value in record.get("semantics", []) if value]
        record["region_keys"] = [value for value in record.get("region_keys", []) if value]
        record["draw_keys"] = [value for value in record.get("draw_keys", []) if value]
        records.append(record)
    return records, warnings


def _append_unique(values: list, value) -> None:
    if value in (None, "") or value in values:
        return
    values.append(value)


def _texture_semantic_label(semantic: str, semantic_index: int) -> str:
    normalized = str(semantic or "").strip().lower()
    if not normalized:
        return ""
    if normalized in {"material", "effect"}:
        return f"{normalized}{int(semantic_index)}"
    return normalized


def _texture_semantic_slug(semantic: str, semantic_index: int, slot: str) -> str:
    label = _texture_semantic_label(semantic, semantic_index)
    if label:
        return _safe_suffix(label)
    return _safe_suffix(str(slot or "texture").replace("-", "_"))


def _normalize_object_names(record: dict) -> list[str]:
    raw_names = record.get("object_names", [])
    if not raw_names:
        raw_names = record.get("objects", [])
    if not raw_names:
        raw_name = record.get("object_name", "")
        raw_names = [raw_name] if raw_name else []
    names: list[str] = []
    for raw_name in raw_names or []:
        name = str(raw_name or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _normalize_shadow_stage(capture_manifest: dict) -> dict:
    stage = dict(capture_manifest.get("shadow_stage", {}) or {})
    return {
        "host_ib_hash": str(stage.get("host_ib_hash", "") or "").lower(),
        "host_match_first_index": int(stage.get("host_match_first_index", 0) or 0),
        "host_match_index_count": int(stage.get("host_match_index_count", 0) or 0),
        "host_draw_index": int(stage.get("host_draw_index", -1) or -1),
        "stage_draw_start": int(stage.get("stage_draw_start", -1) or -1),
        "stage_draw_end": int(stage.get("stage_draw_end", -1) or -1),
        "normal_vs_hash": str(stage.get("normal_vs_hash", "") or "").lower(),
        "transparent_vs_hash": str(stage.get("transparent_vs_hash", "") or "").lower(),
    }


def _build_shadow_replay_plan(capture_manifest: dict, geometry_records: list[dict]) -> dict:
    stage = _normalize_shadow_stage(capture_manifest)
    host_key = (
        stage["host_ib_hash"],
        int(stage["host_match_first_index"]),
        int(stage["host_match_index_count"]),
    )
    if not host_key[0] or host_key[2] <= 0 or not geometry_records:
        return {"enabled": False, "reason": "missing_host_or_geometry"}

    roles_by_key = _shadow_roles_by_key(capture_manifest)
    transparent_parts: list[str] = []
    normal_parts: list[str] = []
    skipped_keys: set[tuple[str, int, int]] = set()

    for record in geometry_records:
        key = _override_key(record)
        roles = roles_by_key.get(key, set())
        if not roles.intersection({"transparent_shadow", "normal_shadow"}):
            continue
        skipped_keys.add(key)
        suffix = str(record.get("resource_suffix", "") or "")
        if "transparent_shadow" in roles:
            transparent_parts.append(suffix)
        if "normal_shadow" in roles:
            normal_parts.append(suffix)

    if not transparent_parts and not normal_parts:
        return {"enabled": False, "reason": "no_exported_shadow_parts"}

    return {
        "enabled": True,
        "host_key": _key_payload(host_key),
        "host_draw_index": int(stage["host_draw_index"]),
        "preserve_host_draw": host_key not in skipped_keys,
        "white_shadow_resource": "ResourceBMCWhiteShadow",
        "transparent_parts": transparent_parts,
        "normal_parts": normal_parts,
        "skip_keys": [_key_payload(key) for key in sorted(skipped_keys)],
    }


def _build_lod_replay_links(capture_manifest: dict, geometry_records: list[dict]) -> list[dict]:
    geometry_by_key = _geometry_records_by_key(geometry_records)
    if not geometry_by_key:
        return []

    links_by_lod_key: dict[tuple[str, int, int], dict] = {}
    for link in capture_manifest.get("lod_links", []) or []:
        main_key = _main_key_from_lod_link(dict(link or {}))
        main_geometry = geometry_by_key.get(main_key, [])
        if not main_geometry:
            continue
        lod_key = _lod_replay_host_key_from_link(dict(link or {}))
        if not _is_valid_override_key(lod_key):
            continue

        bucket = links_by_lod_key.setdefault(
            lod_key,
            {
                "lod_key": _key_payload(lod_key),
                "main_keys": [],
                "geometry": [],
                "geometry_suffixes": [],
            },
        )
        main_payload = _key_payload(main_key)
        _append_unique_payload(bucket["main_keys"], main_payload)
        for geometry_record in sorted(main_geometry, key=lambda item: int(item.get("part_index", 0) or 0)):
            suffix = str(geometry_record.get("resource_suffix", "") or "")
            if not suffix:
                continue
            if suffix not in bucket["geometry_suffixes"]:
                bucket["geometry_suffixes"].append(suffix)
            _append_unique_payload(
                bucket["geometry"],
                {
                    "resource_suffix": suffix,
                    "main_key": main_payload,
                },
            )

    return [
        links_by_lod_key[key]
        for key in sorted(links_by_lod_key)
        if links_by_lod_key[key].get("geometry_suffixes")
    ]


def _build_lod_key_annotations(capture_manifest: dict, geometry_records: list[dict]) -> list[dict]:
    geometry_by_key = _geometry_records_by_key(geometry_records)
    annotations_by_lod_key: dict[tuple[str, int, int], dict] = {}

    for link in capture_manifest.get("lod_links", []) or []:
        main_key = _main_key_from_lod_link(dict(link or {}))
        if not _is_valid_override_key(main_key):
            continue
        main_payload = _key_payload(main_key)
        geometry_suffixes = [
            str(record.get("resource_suffix", "") or "")
            for record in geometry_by_key.get(main_key, []) or []
            if str(record.get("resource_suffix", "") or "")
        ]
        for source in link.get("lod_sources", []) or []:
            lod_key = _lod_key_from_source(dict(source or {}))
            if not _is_valid_override_key(lod_key):
                continue
            bucket = annotations_by_lod_key.setdefault(
                lod_key,
                {
                    "lod_key": _key_payload(lod_key),
                    "main_keys": [],
                    "geometry_suffixes": [],
                },
            )
            _append_unique_payload(bucket["main_keys"], main_payload)
            for suffix in geometry_suffixes:
                if suffix not in bucket["geometry_suffixes"]:
                    bucket["geometry_suffixes"].append(suffix)

    return [
        annotations_by_lod_key[key]
        for key in sorted(annotations_by_lod_key)
        if annotations_by_lod_key[key].get("main_keys")
    ]


def _build_lod_shadow_replay_plan(
    capture_manifest: dict,
    geometry_records: list[dict],
    lod_replay_links: list[dict],
    lod_capture_records: list[dict],
) -> dict:
    lod_snapshot = dict(capture_manifest.get("lod_manifest_snapshot", {}) or {})
    stage = _normalize_shadow_stage(lod_snapshot)
    host_key, host_draw_index, host_source = _select_lod_shadow_host(stage, lod_capture_records)
    if not _is_valid_override_key(host_key) or not geometry_records or not lod_replay_links:
        return {"enabled": False, "reason": "missing_lod_host_or_geometry"}
    available_globals, coverage_records = _lod_shadow_available_globals_before_host(
        lod_capture_records,
        host_draw_index=int(host_draw_index),
        stage=stage,
    )

    roles_by_main_key = _shadow_roles_by_key(capture_manifest)
    geometry_by_suffix = {
        str(record.get("resource_suffix", "") or ""): record
        for record in geometry_records
    }
    transparent_parts: list[str] = []
    normal_parts: list[str] = []
    skipped_keys: set[tuple[str, int, int]] = set()
    missing_links: list[dict] = []

    for link in lod_replay_links:
        lod_key = _key_from_payload(dict(link.get("lod_key", {}) or {}))
        if not _is_valid_override_key(lod_key):
            continue
        link_transparent_parts: list[str] = []
        link_normal_parts: list[str] = []
        required_globals: set[int] = set()
        for geometry_item in _lod_link_geometry_items(link):
            suffix = str(geometry_item.get("resource_suffix", "") or "")
            geometry_record = geometry_by_suffix.get(suffix)
            if geometry_record is None:
                continue
            main_key = _key_from_payload(dict(geometry_item.get("main_key", {}) or {}))
            main_roles = roles_by_main_key.get(main_key, set())
            if not main_roles.intersection({"transparent_shadow", "normal_shadow"}):
                continue
            required_globals.update(_geometry_required_global_bones(geometry_record))
            if "transparent_shadow" in main_roles and suffix not in link_transparent_parts:
                link_transparent_parts.append(suffix)
            if "normal_shadow" in main_roles and suffix not in link_normal_parts:
                link_normal_parts.append(suffix)
        if not link_transparent_parts and not link_normal_parts:
            continue
        missing_globals = sorted(required_globals.difference(available_globals))
        if missing_globals:
            missing_links.append(
                {
                    "lod_key": _key_payload(lod_key),
                    "geometry_suffixes": [*link_transparent_parts, *[suffix for suffix in link_normal_parts if suffix not in link_transparent_parts]],
                    "required_global_count": len(required_globals),
                    "available_global_count": len(available_globals),
                    "missing_global_bones": missing_globals[:64],
                    "missing_global_count": len(missing_globals),
                }
            )
            continue
        for suffix in link_transparent_parts:
            if suffix not in transparent_parts:
                transparent_parts.append(suffix)
        for suffix in link_normal_parts:
            if suffix not in normal_parts:
                normal_parts.append(suffix)
        skipped_keys.add(lod_key)

    if not transparent_parts and not normal_parts:
        reason = "lod_shadow_coverage_incomplete" if missing_links else "no_lod_exported_shadow_parts"
        return {
            "enabled": False,
            "reason": reason,
            "host_key": _key_payload(host_key),
            "host_draw_index": int(host_draw_index),
            "host_source": host_source,
            "available_global_count": len(available_globals),
            "coverage_record_count": len(coverage_records),
            "missing_links": missing_links,
        }

    return {
        "enabled": True,
        "host_key": _key_payload(host_key),
        "host_draw_index": int(host_draw_index),
        "host_source": host_source,
        "preserve_host_draw": host_key not in skipped_keys,
        "white_shadow_resource": "ResourceBMCWhiteShadow",
        "transparent_parts": transparent_parts,
        "normal_parts": normal_parts,
        "skip_keys": [_key_payload(key) for key in sorted(skipped_keys)],
        "available_global_count": len(available_globals),
        "coverage_record_count": len(coverage_records),
        "missing_links": missing_links,
    }


def _select_lod_shadow_host(stage: dict, _lod_capture_records: list[dict]) -> tuple[tuple[str, int, int], int, str]:
    host_key = (
        str(stage.get("host_ib_hash", "") or "").lower(),
        int(stage.get("host_match_first_index", 0) or 0),
        int(stage.get("host_match_index_count", 0) or 0),
    )
    host_draw = int(stage.get("host_draw_index", -1) or -1)
    return host_key, host_draw, "lod_shadow_stage_host"


def _lod_shadow_available_globals_before_host(
    lod_capture_records: list[dict],
    *,
    host_draw_index: int,
    stage: dict,
) -> tuple[set[int], list[dict]]:
    if int(host_draw_index) < 0:
        return set(), []

    stage_start = int(stage.get("stage_draw_start", -1) or -1)
    stage_end = int(stage.get("stage_draw_end", -1) or -1)
    available_globals: set[int] = set()
    coverage_records: list[dict] = []

    for record in lod_capture_records or []:
        draw_indices = [
            int(value)
            for value in record.get("capture_draw_indices", []) or []
            if int(value) >= 0
            and int(value) <= int(host_draw_index)
            and _draw_index_in_optional_stage_window(int(value), stage_start, stage_end)
        ]
        if not draw_indices:
            continue

        globals_for_record = {
            int(value)
            for value in record.get("canonical_global_bones", []) or []
            if int(value) >= 0
        }
        if not globals_for_record:
            continue

        available_globals.update(globals_for_record)
        coverage_records.append(
            {
                "record_index": int(record.get("record_index", -1) or -1),
                "draw_index": max(draw_indices),
                "key": _key_payload(_override_key(record)),
                "global_count": len(globals_for_record),
            }
        )

    return available_globals, coverage_records


def _shadow_roles_by_key(capture_manifest: dict) -> dict[tuple[str, int, int], set[str]]:
    roles_by_key: dict[tuple[str, int, int], set[str]] = {}
    for hit in capture_manifest.get("draw_hits", []) or []:
        role = str(hit.get("pass_role", "") or "")
        if role not in {"transparent_shadow", "normal_shadow"}:
            continue
        key = (
            str(hit.get("ib_hash", "") or "").lower(),
            int(hit.get("first_index", 0) or 0),
            int(hit.get("index_count", 0) or 0),
        )
        if not key[0] or key[2] <= 0:
            continue
        roles_by_key.setdefault(key, set()).add(role)
    return roles_by_key


def _geometry_records_by_key(geometry_records: list[dict]) -> dict[tuple[str, int, int], list[dict]]:
    records_by_key: dict[tuple[str, int, int], list[dict]] = {}
    for record in geometry_records:
        records_by_key.setdefault(_override_key(record), []).append(record)
    return records_by_key


def _geometry_records_by_suffix(geometry_records: list[dict]) -> dict[str, dict]:
    return {
        str(record.get("resource_suffix", "") or ""): record
        for record in geometry_records
        if str(record.get("resource_suffix", "") or "")
    }


def _lod_geometry_records_by_key(runtime_plan: dict, geometry_by_suffix: dict[str, dict]) -> dict[tuple[str, int, int], list[dict]]:
    records_by_key: dict[tuple[str, int, int], list[dict]] = {}
    for link in runtime_plan.get("lod_replay_links", []) or []:
        key = _key_from_payload(dict(link.get("lod_key", {}) or {}))
        if not _is_valid_override_key(key):
            continue
        for suffix in link.get("geometry_suffixes", []) or []:
            record = geometry_by_suffix.get(str(suffix or ""))
            if record:
                records_by_key.setdefault(key, []).append(record)
    return records_by_key


def _lod_link_geometry_items(link: dict) -> list[dict]:
    items = [dict(item or {}) for item in link.get("geometry", []) or []]
    if items:
        return items
    main_keys = [dict(item or {}) for item in link.get("main_keys", []) or []]
    main_key = main_keys[0] if main_keys else {}
    return [
        {
            "resource_suffix": str(suffix or ""),
            "main_key": main_key,
        }
        for suffix in link.get("geometry_suffixes", []) or []
    ]


def _dedupe_geometry_records(geometry_records: list[dict]) -> list[dict]:
    records: list[dict] = []
    seen_suffixes: set[str] = set()
    for record in sorted(geometry_records, key=lambda item: (int(item.get("part_index", 0) or 0), str(item.get("resource_suffix", "") or ""))):
        suffix = str(record.get("resource_suffix", "") or "")
        if not suffix or suffix in seen_suffixes:
            continue
        seen_suffixes.add(suffix)
        records.append(record)
    return records


def _main_key_from_lod_link(link: dict) -> tuple[str, int, int]:
    return (
        str(link.get("ib_hash", link.get("main_ib_hash", "")) or "").lower(),
        int(link.get("match_first_index", link.get("main_match_first_index", link.get("main_first_index", 0))) or 0),
        int(link.get("match_index_count", link.get("main_match_index_count", 0)) or 0),
    )


def _lod_replay_host_key_from_link(link: dict) -> tuple[str, int, int]:
    sources = [
        dict(source or {})
        for source in link.get("lod_sources", []) or []
        if _is_valid_override_key(_lod_key_from_source(dict(source or {})))
    ]
    if sources:
        sources.sort(
            key=lambda source: (
                -int(source.get("mapped_global_count", 0) or 0),
                -float(source.get("score", 0.0) or 0.0),
                -int(source.get("votes", 0) or 0),
                str(source.get("lod_record_key", "") or ""),
            )
        )
        return _lod_key_from_source(sources[0])
    return _lod_key_from_source(link)


def _lod_key_from_source(source: dict) -> tuple[str, int, int]:
    return (
        str(source.get("lod_ib_hash", "") or "").lower(),
        int(source.get("lod_match_first_index", source.get("lod_first_index", 0)) or 0),
        int(source.get("lod_match_index_count", 0) or 0),
    )


def _key_from_payload(payload: dict) -> tuple[str, int, int]:
    return (
        str(payload.get("ib_hash", "") or "").lower(),
        int(payload.get("match_first_index", 0) or 0),
        int(payload.get("match_index_count", 0) or 0),
    )


def _is_valid_override_key(key: tuple[str, int, int]) -> bool:
    return bool(str(key[0] or "")) and int(key[2]) > 0


def _append_unique_payload(items: list[dict], payload: dict) -> None:
    if payload not in items:
        items.append(payload)


def _key_payload(key: tuple[str, int, int]) -> dict:
    return {
        "ib_hash": str(key[0]).lower(),
        "match_first_index": int(key[1]),
        "match_index_count": int(key[2]),
    }


def _attach_palette_metadata_to_geometry(geometry_records: list[dict], palette_records: list[dict]) -> None:
    metadata_by_suffix: dict[str, dict] = {}
    for record in palette_records:
        suffix = str(record.get("resource_suffix", "") or "")
        if not suffix:
            continue
        metadata_by_suffix[suffix] = {
            "local_bone_count": int(record.get("local_bone_count", 0) or 0),
            "palette_values": [int(value) for value in record.get("palette_values", []) or [] if int(value) >= 0],
        }
    missing_suffixes: list[str] = []
    for record in geometry_records:
        suffix = str(record.get("resource_suffix", "") or "")
        metadata = metadata_by_suffix.get(suffix)
        if metadata is None:
            missing_suffixes.append(suffix)
            continue
        record["local_bone_count"] = int(metadata["local_bone_count"])
        record["palette_values"] = list(metadata["palette_values"])
    if missing_suffixes:
        raise ValueError(
            "Geometry buffer(s) are missing matching PartLocalToGlobalBoneMap palette record(s): "
            + ", ".join(missing_suffixes[:8])
        )


def _geometry_required_global_bones(geometry_record: dict) -> set[int]:
    return {
        int(value)
        for value in geometry_record.get("palette_values", []) or []
        if int(value) >= 0
    }


def _validate_palette_globals(capture_manifest: dict, palette_records: list[dict]) -> None:
    unavailable = _capture_unavailable_global_bones(capture_manifest)
    if not unavailable:
        return

    errors: list[str] = []
    for palette_record in palette_records:
        used = sorted(unavailable.intersection(int(value) for value in palette_record.get("palette_values", []) or []))
        if not used:
            continue
        errors.append(
            f"{palette_record.get('resource_suffix')}: uses capture-unavailable global bone(s) "
            + ", ".join(str(value) for value in used[:12])
        )
    if errors:
        raise ValueError(
            "Export uses global bones that cannot be captured in the main shadow stage. "
            "Move those weights to captured globals or keep that source IB unmodified. "
            + " | ".join(errors[:3])
        )


def _capture_unavailable_global_bones(capture_manifest: dict) -> set[int]:
    unavailable: set[int] = set()
    for pool_record in capture_manifest.get("bone_pool_order", []) or []:
        if bool(pool_record.get("bone_capture_available", pool_record.get("shadow_capture_ready", False))):
            continue
        base = int(pool_record.get("global_bone_base", 0) or 0)
        count = int(pool_record.get("local_bone_count", 0) or 0)
        unavailable.update(range(base, base + max(0, count)))
    return unavailable


def _shader_override_sections(runtime_plan: dict) -> list[str]:
    lines: list[str] = []
    seen_sections: set[str] = set()
    seen_filter_keys: set[tuple[str, int]] = set()
    for vs_hash in runtime_plan.get("shadow_vs_hashes", []) or []:
        safe_hash = str(vs_hash).lower()
        if not safe_hash:
            continue
        section_name = f"ShaderOverrideBoneStoreVS_{safe_hash}"
        if section_name in seen_sections:
            continue
        seen_sections.add(section_name)
        seen_filter_keys.add((safe_hash, 200))
        lines.extend(
            [
                f"[{section_name}]",
                f"hash = {safe_hash}",
                "filter_index = 200",
                "allow_duplicate_hash = overrule",
                "",
            ]
        )
    for rule in runtime_plan.get("shader_filter_overrides", []) or []:
        safe_hash = str(rule.get("hash", "") or "").lower()
        if not safe_hash:
            continue
        filter_index = int(rule.get("filter_index", -1) or -1)
        if filter_index < 0:
            continue
        if (safe_hash, filter_index) in seen_filter_keys:
            continue
        seen_filter_keys.add((safe_hash, filter_index))
        section_name = str(rule.get("section_name", "") or "").strip()
        if not section_name:
            section_prefix = str(rule.get("section_prefix", "") or "ShaderOverrideBMCFilter")
            section_name = f"{_safe_suffix(section_prefix)}_{safe_hash}"
        section_name = _safe_suffix(section_name)
        if section_name in seen_sections:
            continue
        seen_sections.add(section_name)
        lines.extend(
            [
                f"[{section_name}]",
                f"hash = {safe_hash}",
                f"filter_index = {filter_index}",
                f"allow_duplicate_hash = {rule.get('allow_duplicate_hash', 'overrule')}",
                "",
            ]
        )
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _resource_sections(runtime_plan: dict) -> list[str]:
    buffers = dict(runtime_plan.get("buffers", {}) or {})
    global_pool_rows = _GLOBAL_BONE_POOL_ROWS_PER_SLOT * _MAX_INSTANCE_SLOTS
    local_pool_rows = _LOCAL_BONE_POOL_ROWS_PER_SLOT * _MAX_INSTANCE_SLOTS
    shadow_plan = dict(runtime_plan.get("shadow_replay_plan", {}) or {})
    lod_shadow_plan = dict(runtime_plan.get("lod_shadow_replay_plan", {}) or {})
    lines = [
        "; -------------------------------------------------",
        "; Static capture tables",
        "; -------------------------------------------------",
        *_buffer_resource("ResourceMainCaptureBoneMap", buffers.get("main_capture_bone_map", {})),
    ]
    if buffers.get("lod_capture_bone_map"):
        lines.extend(
            [
                "",
                *_buffer_resource("ResourceLodCaptureBoneMap", buffers.get("lod_capture_bone_map", {})),
            ]
        )
    lines.extend(
        [
            "",
            "; -------------------------------------------------",
            "; Shared runtime buffers",
            "; -------------------------------------------------",
            "[ResourceDumpedCB1_UAV]",
            "type = RWStructuredBuffer",
            "stride = 16",
            f"array = {_CB1_ROWS}",
            "",
            "[ResourceDumpedCB1_SRV]",
            "type = Buffer",
            "stride = 16",
            f"array = {_CB1_ROWS}",
            "",
            "[ResourceFakeCB1_UAV]",
            "type = RWStructuredBuffer",
            "stride = 16",
            f"array = {_CB1_ROWS}",
            "",
            "[ResourceFakeCB1]",
            "type = Buffer",
            "stride = 16",
            "format = R32G32B32A32_UINT",
            f"array = {_CB1_ROWS}",
            "",
            "[ResourceGlobalBonePool_UAV]",
            "type = RWStructuredBuffer",
            "stride = 16",
            f"array = {global_pool_rows}",
            "",
            "[ResourceGlobalBonePool_SRV]",
            "type = StructuredBuffer",
            "stride = 16",
            f"array = {global_pool_rows}",
            "",
            "[ResourceLocalBonePool_UAV]",
            "type = RWStructuredBuffer",
            "stride = 16",
            f"array = {local_pool_rows}",
            "",
            "[ResourceLocalBonePool_SRV]",
            "type = StructuredBuffer",
            "stride = 16",
            f"array = {local_pool_rows}",
            "",
            "[ResourceRuntimeState_UAV]",
            "type = RWStructuredBuffer",
            "stride = 16",
            f"array = {_RUNTIME_STATE_ROWS}",
        ]
    )
    if _shadow_plan_needs_white_texture(shadow_plan) or _shadow_plan_needs_white_texture(lod_shadow_plan):
        lines.extend(
            [
                "",
                "[ResourceBMCWhiteShadow]",
                f"filename = {_WHITE_SHADOW_TEXTURE_FILE.replace(os.sep, '/')}",
            ]
        )
    return lines


def _custom_shader_sections(runtime_plan: dict) -> list[str]:
    _ = runtime_plan
    return [
        "; -------------------------------------------------",
        "; Shaders",
        "; -------------------------------------------------",
        "[CustomShader_ExtractCB1]",
        "vs = hlsl\\extract_cb1_vs.hlsl",
        "ps = hlsl\\extract_cb1_ps.hlsl",
        "ps-u7 = ResourceDumpedCB1_UAV",
        "depth_enable = false",
        "blend = ADD SRC_ALPHA INV_SRC_ALPHA",
        "cull = none",
        "topology = point_list",
        "draw = 4096, 0",
        "ps-u7 = null",
        "ResourceDumpedCB1_SRV = copy ResourceDumpedCB1_UAV",
        "",
        "[CustomShader_RecordBones]",
        "cs = hlsl\\record_bones_cs.hlsl",
        "cs-t0 = vs-t0",
        "cs-t1 = ResourceDumpedCB1_SRV",
        "cs-u1 = ResourceGlobalBonePool_UAV",
        "cs-u2 = ResourceRuntimeState_UAV",
        "dispatch = 1, 1, 1",
        "cs-u1 = null",
        "cs-u2 = null",
        "cs-t0 = null",
        "cs-t1 = null",
        "cs-t2 = null",
        "ResourceGlobalBonePool_SRV = copy ResourceGlobalBonePool_UAV",
        "",
        "[CustomShader_GatherLocalBones]",
        "cs = hlsl\\gather_local_bones_cs.hlsl",
        "cs-t0 = ResourceGlobalBonePool_SRV",
        "cs-u1 = ResourceLocalBonePool_UAV",
        "cs-u2 = ResourceRuntimeState_UAV",
        "dispatch = 1, 1, 1",
        "cs-u1 = null",
        "cs-u2 = null",
        "cs-t0 = null",
        "cs-t2 = null",
        "ResourceLocalBonePool_SRV = copy ResourceLocalBonePool_UAV",
        "",
        "[CustomShader_RedirectCB1]",
        "cs = hlsl\\redirect_cb1_cs.hlsl",
        "cs-t0 = ResourceDumpedCB1_SRV",
        "cs-u0 = ResourceFakeCB1_UAV",
        "cs-u2 = ResourceRuntimeState_UAV",
        "dispatch = 1, 1, 1",
        "cs-u0 = null",
        "cs-u2 = null",
        "cs-t0 = null",
        "ResourceFakeCB1 = copy ResourceFakeCB1_UAV",
        "",
        "[CustomShader_ResetRuntimeState]",
        "cs = hlsl\\reset_runtime_state_cs.hlsl",
        "cs-u2 = ResourceRuntimeState_UAV",
        "dispatch = 1, 1, 1",
        "cs-u2 = null",
    ]


def _local_palette_sections(runtime_plan: dict) -> list[str]:
    palette_records = list(runtime_plan.get("palettes", []) or [])
    if not palette_records:
        return []

    lines: list[str] = [
        "; -------------------------------------------------",
        "; Export part local-to-global bone maps",
        "; -------------------------------------------------",
    ]
    for record in palette_records:
        suffix = str(record.get("resource_suffix", "") or "")
        lines.extend(
            [
                f"[ResourcePartLocalToGlobalBoneMap_{suffix}]",
                "type = Buffer",
                "format = R32_UINT",
                f"filename = {record.get('filename')}",
                "",
            ]
        )
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _geometry_resource_sections(runtime_plan: dict) -> list[str]:
    geometry_records = list(runtime_plan.get("geometry", []) or [])
    if not geometry_records:
        return []

    lines: list[str] = [
        "; -------------------------------------------------",
        "; Export geometry buffers",
        "; -------------------------------------------------",
    ]
    for record in geometry_records:
        for slot, vertex_buffer in sorted(dict(record.get("vertex_buffers", {}) or {}).items()):
            stride = int(vertex_buffer.get("stride", 0) or 0)
            if stride <= 0:
                continue
            lines.extend(
                [
                    f"[{vertex_buffer.get('resource_name')}]",
                    "type = Buffer",
                    f"stride = {stride}",
                    f"filename = {vertex_buffer.get('filename')}",
                    "",
                ]
            )
            if slot == "vb0":
                # vb3 aliases vb0 in the game layout; no extra resource is needed.
                pass
        lines.extend(
            [
                f"[{record.get('index_resource_name')}]",
                "type = Buffer",
                "format = R32_UINT",
                f"filename = {record.get('index_filename')}",
                "",
            ]
        )
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _texture_resource_sections(runtime_plan: dict) -> list[str]:
    texture_records = list(runtime_plan.get("textures", []) or [])
    if not texture_records:
        return []
    lines = [
        "; -------------------------------------------------",
        "; Hash-replaced textures",
        "; -------------------------------------------------",
    ]
    for record in texture_records:
        lines.extend(
            [
                f"[{record.get('resource_name')}]",
                f"filename = {record.get('filename')}",
                "",
            ]
        )
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _texture_hash_override_sections(runtime_plan: dict) -> list[str]:
    texture_records = list(runtime_plan.get("textures", []) or [])
    if not texture_records:
        return []
    lines = [
        "; -------------------------------------------------",
        "; Texture hash replacements",
        "; -------------------------------------------------",
    ]
    for record in texture_records:
        texture_hash = str(record.get("hash", "") or "").lower()
        suffix = _safe_suffix(f"{texture_hash}_{record.get('semantic', '')}_{record.get('slot', '')}")
        lines.extend(
            [
                f"[TextureOverride_BMCTexture_{suffix}]",
                f"hash = {texture_hash}",
                f"this = {record.get('resource_name')}",
                "",
            ]
        )
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _texture_override_sections(runtime_plan: dict) -> list[str]:
    capture_by_key: dict[tuple[str, int, int], dict[str, list[int]]] = {}
    for record in runtime_plan.get("capture_records", []) or []:
        key = _override_key(record)
        capture_by_key.setdefault(key, {"main": [], "lod": []})["main"].append(int(record["record_index"]))
    for record in runtime_plan.get("lod_capture_records", []) or []:
        key = _override_key(record)
        capture_by_key.setdefault(key, {"main": [], "lod": []})["lod"].append(int(record["record_index"]))

    geometry_records_all = list(runtime_plan.get("geometry", []) or [])
    geometry_by_key = _geometry_records_by_key(geometry_records_all)
    geometry_by_suffix = _geometry_records_by_suffix(geometry_records_all)
    lod_geometry_by_key = _lod_geometry_records_by_key(runtime_plan, geometry_by_suffix)

    shadow_plan = dict(runtime_plan.get("shadow_replay_plan", {}) or {})
    lod_shadow_plan = dict(runtime_plan.get("lod_shadow_replay_plan", {}) or {})
    shadow_plans = [plan for plan in (shadow_plan, lod_shadow_plan) if bool(plan.get("enabled", False))]
    shadow_host_keys = [_shadow_plan_host_key(plan) for plan in shadow_plans]
    shadow_skip_keys: set[tuple[str, int, int]] = set()
    for plan in shadow_plans:
        shadow_skip_keys.update(_shadow_plan_skip_keys(plan))
    shadow_hosts_by_key: dict[tuple[str, int, int], list[dict]] = {}
    for host_key, plan in zip(shadow_host_keys, shadow_plans):
        if host_key is not None:
            shadow_hosts_by_key.setdefault(host_key, []).append(plan)
    lod_role_keys = _lod_texture_override_keys(capture_by_key, lod_geometry_by_key, lod_shadow_plan)
    main_role_keys = _main_texture_override_keys(capture_by_key, geometry_by_key, shadow_plan)
    lod_annotations_by_key = _lod_annotations_by_key(runtime_plan)

    all_keys_set = set(capture_by_key).union(geometry_by_key).union(lod_geometry_by_key)
    all_keys_set.update(shadow_hosts_by_key)
    all_keys = sorted(all_keys_set)
    if not all_keys:
        return []

    lines: list[str] = [
        "; -------------------------------------------------",
        "; TextureOverrides",
        "; -------------------------------------------------",
    ]
    for key in all_keys:
        grouped_records = capture_by_key.get(key, {"main": [], "lod": []})
        geometry_records = _dedupe_geometry_records(
            [*geometry_by_key.get(key, []), *lod_geometry_by_key.get(key, [])]
        )
        ib_hash, match_first_index, match_index_count = key
        suffix = _safe_suffix(f"{ib_hash}_{match_index_count}_{match_first_index}")
        section_name = _texture_override_section_name(
            suffix,
            has_lod=key in lod_role_keys,
            has_main=key in main_role_keys,
        )
        lod_hash_annotation = _lod_hash_line_annotation(lod_annotations_by_key.get(key))
        lines.extend(
            [
                section_name,
            ]
        )
        if lod_hash_annotation:
            lines.append(f"; {lod_hash_annotation}")
        lines.extend(
            [
                f"hash = {ib_hash}",
                f"match_index_count = {match_index_count}",
            ]
        )
        has_capture_records = bool(grouped_records.get("main") or grouped_records.get("lod"))
        if has_capture_records:
            lines.extend(_capture_record_lines(grouped_records, indent=""))

        shadow_lines: list[str] = []
        if key in shadow_skip_keys:
            shadow_lines.append("  handling = skip")
        for plan in shadow_hosts_by_key.get(key, []):
            shadow_lines.extend(_shadow_host_replay_lines(plan, geometry_by_suffix, indent="  "))
        if shadow_lines:
            lines.append("if vs == 200")
            lines.extend(shadow_lines)
            lines.append("endif")

        if geometry_records:
            if has_capture_records or shadow_lines:
                lines.append("")
            lines.extend(_visible_replay_lines(geometry_records, runtime_plan))
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _lod_texture_override_keys(
    capture_by_key: dict[tuple[str, int, int], dict[str, list[int]]],
    lod_geometry_by_key: dict[tuple[str, int, int], list[dict]],
    lod_shadow_plan: dict,
) -> set[tuple[str, int, int]]:
    keys = {
        key
        for key, grouped_records in capture_by_key.items()
        if grouped_records.get("lod")
    }
    keys.update(lod_geometry_by_key)
    host_key = _shadow_plan_host_key(lod_shadow_plan)
    if host_key is not None:
        keys.add(host_key)
    keys.update(_shadow_plan_skip_keys(lod_shadow_plan))
    return keys


def _main_texture_override_keys(
    capture_by_key: dict[tuple[str, int, int], dict[str, list[int]]],
    geometry_by_key: dict[tuple[str, int, int], list[dict]],
    shadow_plan: dict,
) -> set[tuple[str, int, int]]:
    keys = {
        key
        for key, grouped_records in capture_by_key.items()
        if grouped_records.get("main")
    }
    keys.update(geometry_by_key)
    host_key = _shadow_plan_host_key(shadow_plan)
    if host_key is not None:
        keys.add(host_key)
    keys.update(_shadow_plan_skip_keys(shadow_plan))
    return keys


def _texture_override_section_name(suffix: str, *, has_lod: bool, has_main: bool) -> str:
    role_suffix = ""
    if has_lod and has_main:
        role_suffix = "_MAIN_LOD"
    elif has_lod:
        role_suffix = "_LOD"
    return f"[TextureOverride_BMC_{suffix}{role_suffix}]"


def _lod_annotations_by_key(runtime_plan: dict) -> dict[tuple[str, int, int], dict]:
    annotations: dict[tuple[str, int, int], dict] = {}
    for annotation in runtime_plan.get("lod_key_annotations", []) or []:
        key = _key_from_payload(dict(annotation.get("lod_key", {}) or {}))
        if _is_valid_override_key(key):
            annotations[key] = dict(annotation)
    return annotations


def _lod_hash_line_annotation(annotation: dict | None) -> str:
    if not annotation:
        return ""

    main_labels = [
        _hash_label(_key_from_payload(dict(payload or {})))
        for payload in annotation.get("main_keys", []) or []
        if _is_valid_override_key(_key_from_payload(dict(payload or {})))
    ]
    if main_labels:
        return "main:" + ",".join(main_labels)
    return ""


def _hash_label(key: tuple[str, int, int]) -> str:
    return str(key[0])


def _visible_replay_lines(geometry_records: list[dict], runtime_plan: dict) -> list[str]:
    lines = [
        _visible_replay_condition(runtime_plan),
        "  handling = skip",
        "  run = CustomShader_ExtractCB1",
    ]
    for record in geometry_records:
        lines.extend(_replay_part_lines(record, indent="  "))
    lines.append("endif")
    return lines


def _capture_record_lines(grouped_records: dict, *, indent: str) -> list[str]:
    lines: list[str] = []
    for record_index in grouped_records.get("main", []) or []:
        lines.extend(
            [
                f"{indent}run = CustomShader_ExtractCB1",
                f"{indent}x100 = {record_index}",
                f"{indent}cs-t2 = ResourceMainCaptureBoneMap",
                f"{indent}run = CustomShader_RecordBones",
            ]
        )
    for record_index in grouped_records.get("lod", []) or []:
        lines.extend(
            [
                f"{indent}run = CustomShader_ExtractCB1",
                f"{indent}x100 = {record_index}",
                f"{indent}cs-t2 = ResourceLodCaptureBoneMap",
                f"{indent}run = CustomShader_RecordBones",
            ]
        )
    return lines


def _visible_replay_condition(runtime_plan: dict) -> str:
    excluded_indices = [200]
    for value in runtime_plan.get("visible_replay_excluded_filter_indices", []) or []:
        try:
            filter_index = int(value)
        except (TypeError, ValueError):
            continue
        if filter_index not in excluded_indices:
            excluded_indices.append(filter_index)
    return "if " + " && ".join(f"vs != {filter_index}" for filter_index in excluded_indices)


def _shadow_host_replay_lines(shadow_plan: dict, parts_by_suffix: dict[str, dict], *, indent: str) -> list[str]:
    if not bool(shadow_plan.get("enabled", False)):
        return []
    lines: list[str] = []
    transparent_parts = [
        parts_by_suffix[suffix]
        for suffix in shadow_plan.get("transparent_parts", []) or []
        if suffix in parts_by_suffix
    ]
    normal_parts = [
        parts_by_suffix[suffix]
        for suffix in shadow_plan.get("normal_parts", []) or []
        if suffix in parts_by_suffix
    ]
    if _shadow_plan_preserves_host_draw(shadow_plan):
        lines.append(f"{indent}draw = from_caller")
    if transparent_parts:
        lines.append(f"{indent}; delayed transparent shadow replay")
        for record in transparent_parts:
            lines.extend(_replay_part_lines(record, indent=indent))
    if normal_parts:
        lines.append(f"{indent}ps-t0 = {shadow_plan.get('white_shadow_resource', 'ResourceBMCWhiteShadow')}")
        lines.append(f"{indent}; delayed normal shadow replay")
        for record in normal_parts:
            lines.extend(_replay_part_lines(record, indent=indent))
    return lines


def _replay_part_lines(record: dict, *, indent: str) -> list[str]:
    suffix = str(record.get("resource_suffix", "") or "")
    local_bone_count = _local_bone_count_for_resource_suffix(record)
    lines = [
        f"{indent}; replay {suffix}",
        f"{indent}x101 = {local_bone_count}",
        f"{indent}cs-t2 = ResourcePartLocalToGlobalBoneMap_{suffix}",
        f"{indent}run = CustomShader_GatherLocalBones",
        f"{indent}vs-t0 = ResourceLocalBonePool_SRV",
        f"{indent}run = CustomShader_RedirectCB1",
        f"{indent}vs-cb1 = ResourceFakeCB1",
        f"{indent}ib = {record.get('index_resource_name')}",
    ]
    vertex_buffers = dict(record.get("vertex_buffers", {}) or {})
    for slot_name in ("vb0", "vb1", "vb2"):
        vertex_buffer = vertex_buffers.get(slot_name)
        if vertex_buffer:
            lines.append(f"{indent}{slot_name} = {vertex_buffer.get('resource_name')}")
    vb0 = vertex_buffers.get("vb0")
    if vb0:
        lines.append(f"{indent}vb3 = {vb0.get('resource_name')}")
    for draw in _draw_ranges_for_record(record):
        object_comment = _draw_object_comment(draw, record)
        if object_comment:
            lines.append(f"{indent}; Blender objects: {object_comment}")
        index_count = int(draw.get("index_count", 0) or 0)
        start_index = int(draw.get("start_index", 0) or 0)
        base_vertex = int(draw.get("base_vertex", 0) or 0)
        lines.append(f"{indent}drawindexedinstanced = {index_count},INSTANCE_COUNT,{start_index},{base_vertex},FIRST_INSTANCE")
    return lines


def _draw_ranges_for_record(record: dict) -> list[dict]:
    ranges = [
        dict(draw or {})
        for draw in record.get("object_draws", []) or []
        if int(dict(draw or {}).get("index_count", 0) or 0) > 0
    ]
    if ranges:
        return ranges
    return [
        {
            "object_name": "",
            "start_index": 0,
            "index_count": int(record.get("index_count", 0) or 0),
            "base_vertex": 0,
        }
    ]


def _draw_object_comment(draw: dict, record: dict) -> str:
    object_name = str(draw.get("object_name", "") or "").replace("\n", " ").replace("\r", " ")
    if object_name:
        return object_name
    return _blender_object_comment(record)


def _blender_object_comment(record: dict) -> str:
    names = _normalize_object_names(record)
    if not names:
        return ""
    return ", ".join(name.replace("\n", " ").replace("\r", " ") for name in names)


def _local_bone_count_for_resource_suffix(record: dict) -> int:
    return int(record.get("local_bone_count", 0) or 0)


def _shadow_plan_host_key(shadow_plan: dict) -> tuple[str, int, int] | None:
    if not bool(shadow_plan.get("enabled", False)):
        return None
    host = dict(shadow_plan.get("host_key", {}) or {})
    key = (
        str(host.get("ib_hash", "") or "").lower(),
        int(host.get("match_first_index", 0) or 0),
        int(host.get("match_index_count", 0) or 0),
    )
    if not key[0] or key[2] <= 0:
        return None
    return key


def _shadow_plan_skip_keys(shadow_plan: dict) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    if not bool(shadow_plan.get("enabled", False)):
        return keys
    for payload in shadow_plan.get("skip_keys", []) or []:
        item = dict(payload or {})
        key = (
            str(item.get("ib_hash", "") or "").lower(),
            int(item.get("match_first_index", 0) or 0),
            int(item.get("match_index_count", 0) or 0),
        )
        if key[0] and key[2] > 0:
            keys.add(key)
    return keys


def _shadow_plan_preserves_host_draw(shadow_plan: dict) -> bool:
    if "preserve_host_draw" in shadow_plan:
        return bool(shadow_plan.get("preserve_host_draw", False))
    host_key = _shadow_plan_host_key(shadow_plan)
    return host_key is not None and host_key not in _shadow_plan_skip_keys(shadow_plan)


def _shadow_plan_needs_white_texture(shadow_plan: dict) -> bool:
    return bool(shadow_plan.get("enabled", False)) and bool(shadow_plan.get("normal_parts", []) or [])


def _write_white_shadow_texture(path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(path):
        return path

    ddpf_alphapixels = 0x1
    ddpf_rgb = 0x40
    ddsd_caps = 0x1
    ddsd_height = 0x2
    ddsd_width = 0x4
    ddsd_pitch = 0x8
    ddsd_pixelformat = 0x1000
    ddscaps_texture = 0x1000

    header_values = [
        124,
        ddsd_caps | ddsd_height | ddsd_width | ddsd_pitch | ddsd_pixelformat,
        1,
        1,
        4,
        0,
        0,
        *([0] * 11),
        32,
        ddpf_alphapixels | ddpf_rgb,
        0,
        32,
        0x00FF0000,
        0x0000FF00,
        0x000000FF,
        0xFF000000,
        ddscaps_texture,
        0,
        0,
        0,
        0,
    ]
    header = struct.pack("<" + "I" * len(header_values), *header_values)
    with open(path, "wb") as file_handle:
        file_handle.write(b"DDS ")
        file_handle.write(header)
        file_handle.write(b"\xff\xff\xff\xff")
    return path


def _frame_lifecycle_sections() -> list[str]:
    return [
        "; -------------------------------------------------",
        "; Frame lifecycle",
        "; -------------------------------------------------",
        "[Present]",
        "run = CommandList_BMC_FrameEndReset",
        "",
        "[CommandList_BMC_FrameEndReset]",
        "run = CustomShader_ResetRuntimeState",
    ]


def _buffer_resource(name: str, payload: dict) -> list[str]:
    return [
        f"[{name}]",
        "type = Buffer",
        "format = R32_UINT",
        f"filename = {payload.get('filename', '')}",
    ]


def _resource_file_payload(path: str, output_directory: str, count: int) -> dict:
    return {
        "file_name": os.path.basename(path),
        "file_path": path,
        "filename": _resource_filename(path, output_directory),
        "uint_count": int(count),
    }


def _resource_filename(path: str, output_directory: str) -> str:
    try:
        relpath = os.path.relpath(path, output_directory)
    except ValueError:
        relpath = path
    return relpath.replace("\\", "/")


def _runtime_shadow_vs_hashes(capture_manifest: dict) -> list[str]:
    hashes: list[str] = []

    def add_many(values) -> None:
        for value in values or []:
            normalized = str(value or "").strip().lower()
            if normalized and normalized not in hashes:
                hashes.append(normalized)

    shadow_stage = dict(capture_manifest.get("shadow_stage", {}) or {})
    add_many(shadow_stage.get("shadow_vs_hashes", []))
    add_many([shadow_stage.get("normal_vs_hash", ""), shadow_stage.get("transparent_vs_hash", "")])

    lod_snapshot = dict(capture_manifest.get("lod_manifest_snapshot", {}) or {})
    lod_shadow_stage = dict(lod_snapshot.get("shadow_stage", {}) or {})
    add_many(lod_shadow_stage.get("shadow_vs_hashes", []))
    add_many([lod_shadow_stage.get("normal_vs_hash", ""), lod_shadow_stage.get("transparent_vs_hash", "")])
    return hashes


def _runtime_shader_filter_overrides(capture_manifest: dict, *, filter_residual: bool) -> list[dict]:
    rules: list[dict] = []
    seen: set[tuple[str, int]] = set()

    for payload in _manifest_shader_filter_payloads(capture_manifest):
        rule = _normalize_shader_filter_rule(payload, source="manifest")
        if not rule:
            continue
        key = (str(rule["hash"]), int(rule["filter_index"]))
        if key in seen:
            continue
        seen.add(key)
        rules.append(rule)

    for payload in get_runtime_shader_filters():
        if not _shader_filter_rule_enabled(payload, filter_residual=filter_residual):
            continue
        rule = _normalize_shader_filter_rule(payload, source="runtime_shader_filters.json")
        if not rule:
            continue
        key = (str(rule["hash"]), int(rule["filter_index"]))
        if key in seen:
            continue
        seen.add(key)
        rules.append(rule)

    return rules


def _manifest_shader_filter_payloads(capture_manifest: dict) -> list[dict]:
    payloads: list[dict] = []

    def add_many(values) -> None:
        if isinstance(values, dict):
            if values.get("hash") or values.get("vs_hash"):
                payloads.append(dict(values))
                return
            values = values.get("shader_overrides", []) or values.get("filters", [])
        for value in values or []:
            if isinstance(value, dict):
                payloads.append(dict(value))

    for key in ("shader_filter_overrides", "runtime_shader_filters", "shader_overrides"):
        add_many(capture_manifest.get(key, []))
    runtime_stage = capture_manifest.get("runtime_stage", {})
    if isinstance(runtime_stage, dict):
        for key in ("shader_filter_overrides", "runtime_shader_filters", "shader_overrides"):
            add_many(runtime_stage.get(key, []))
    return payloads


def _normalize_shader_filter_rule(payload: dict, *, source: str) -> dict:
    safe_hash = str(payload.get("hash", "") or payload.get("vs_hash", "") or "").strip().lower()
    if not _SHADER_HASH_RE.fullmatch(safe_hash):
        return {}
    try:
        filter_index = int(payload.get("filter_index", -1))
    except (TypeError, ValueError):
        return {}
    if filter_index < 0:
        return {}

    section_prefix = str(payload.get("section_prefix", "") or "ShaderOverrideBMCFilter")
    section_name = str(payload.get("section_name", "") or "").strip()
    if not section_name:
        section_name = f"{_safe_suffix(section_prefix)}_{safe_hash}"

    return {
        "id": str(payload.get("id", "") or ""),
        "hash": safe_hash,
        "filter_index": filter_index,
        "section_prefix": _safe_suffix(section_prefix),
        "section_name": _safe_suffix(section_name),
        "allow_duplicate_hash": str(payload.get("allow_duplicate_hash", "") or "overrule"),
        "exclude_from_visible_replay": _truthy(payload.get("exclude_from_visible_replay", False)),
        "source": source,
    }


def _shader_filter_rule_enabled(payload: dict, *, filter_residual: bool) -> bool:
    if "enabled" in payload and not _truthy(payload.get("enabled")):
        return False
    enabled_by = str(payload.get("enabled_by", "") or "").strip()
    if not enabled_by:
        return True
    if enabled_by == "filter_residual":
        return bool(filter_residual)
    return True


def _visible_replay_excluded_filter_indices(shader_filter_overrides: list[dict]) -> list[int]:
    indices: list[int] = []
    for rule in shader_filter_overrides:
        if not bool(rule.get("exclude_from_visible_replay", False)):
            continue
        filter_index = int(rule.get("filter_index", -1) or -1)
        if filter_index >= 0 and filter_index not in indices:
            indices.append(filter_index)
    return indices


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _global_bone_count(capture_manifest: dict) -> int:
    count = 0
    for record in capture_manifest.get("bone_pool_order", []) or []:
        base = int(record.get("global_bone_base", 0) or 0)
        local_count = int(record.get("local_bone_count", 0) or 0)
        count = max(count, base + max(0, local_count))
    return count


def _used_local_indices(pool_record: dict) -> list[int]:
    raw_indices = pool_record.get("used_local_bone_indices", [])
    if raw_indices:
        return sorted({int(value) for value in raw_indices if int(value) >= 0})
    count = int(pool_record.get("local_bone_count", 0) or 0)
    return list(range(max(0, count)))


def _int_list(values) -> list[int]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        values = [values]
    result: list[int] = []
    for value in values or []:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _draw_index_in_optional_stage_window(draw_index: int, stage_start: int, stage_end: int) -> bool:
    if stage_start >= 0 and draw_index < stage_start:
        return False
    if stage_end >= 0 and draw_index > stage_end:
        return False
    return True


def _override_key(record: dict) -> tuple[str, int, int]:
    return (
        str(record.get("ib_hash", "") or "").lower(),
        int(record.get("match_first_index", 0) or 0),
        int(record.get("match_index_count", 0) or 0),
    )


def _slot_resource_role(slot_name: str) -> str:
    normalized = str(slot_name or "").lower()
    if normalized == "vb0":
        return "Position"
    if normalized == "vb1":
        return "Texcoord"
    if normalized == "vb2":
        return "Blend"
    if normalized.startswith("vb") and normalized[2:].isdigit():
        return f"VB{int(normalized[2:])}"
    return normalized or "Vertex"


def _safe_suffix(value: str) -> str:
    return _SAFE_SUFFIX_RE.sub("_", str(value)).strip("_") or "resource"
