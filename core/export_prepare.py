"""Prepare export meshes and local palettes for BI4-compatible output."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ..constants import (
    BI4_MAX_BONE_COUNT,
    BONESTORE_INI_FILE_NAME,
    BUFFER_EXPORT_DIR_NAME,
    CAPTURE_MANIFEST_FILE_NAME,
    EXPORT_MANIFEST_FILE_NAME,
)
from .ini_export import write_bonestore_ini
from .io import ensure_directory, read_json, write_json, write_uint32_buffer
from .models import LocalPaletteRecord, PartRecord

_NUMERIC_GROUP_RE = re.compile(r"^\d+$")
_CHUNK_COLLECTION_RE = re.compile(r"(?P<hash>[0-9A-Fa-f]{8})[-_](?P<count>\d+)(?:[-_](?P<chunk>\d+))?")
_LOCALIZED_PALETTE_PROP = "bmc_export_palette_values"
_LOCALIZED_CHUNK_PROP = "bmc_export_chunk"


@dataclass(frozen=True)
class ExportChunk:
    name: str
    ib_hash: str
    match_index_count: int
    chunk_index: int
    mesh_objects: tuple[object, ...]


def prepare_export_collection(context, export_collection, output_dir: str | None = None):
    if export_collection is None:
        raise ValueError("Export collection is not set")

    normalized_output_dir = ensure_directory(output_dir or context.scene.bmc_output_dir or context.scene.bmc_frameanalysis_dir)
    buffer_dir = ensure_directory(os.path.join(normalized_output_dir, BUFFER_EXPORT_DIR_NAME))

    chunks = _build_export_chunks(export_collection)
    _validate_single_chunk_membership(chunks)

    palette_records = []
    local_palette_records: list[LocalPaletteRecord] = []
    object_records = []

    for chunk in chunks:
        global_groups = _collect_used_groups_for_chunk(chunk)
        ib_hash = chunk.ib_hash
        match_index_count = chunk.match_index_count
        chunk_index = chunk.chunk_index
        palette = tuple(sorted(global_groups))
        if len(palette) > BI4_MAX_BONE_COUNT:
            raise ValueError(
                f"{chunk.name}: local palette has {len(palette)} bones, exceeds {BI4_MAX_BONE_COUNT}"
            )
        file_name = f"{ib_hash}-{match_index_count}-{chunk_index}-Palette.buf"
        file_path = os.path.join(buffer_dir, file_name)
        write_uint32_buffer(file_path, palette)
        resource_suffix = f"{ib_hash}_{match_index_count}_{chunk_index}"
        local_palette_records.append(
            LocalPaletteRecord(
                object_name=chunk.name,
                ib_hash=ib_hash,
                match_index_count=match_index_count,
                chunk_index=chunk_index,
                local_bone_count=len(palette),
                palette_values=palette,
                file_name=file_name,
                file_path=file_path,
                resource_suffix=resource_suffix,
            )
        )
        palette_records.append(
            {
                "ib_hash": ib_hash,
                "match_index_count": match_index_count,
                "chunk_index": chunk_index,
                "chunk_collection": chunk.name,
                "local_bone_count": len(palette),
                "file_name": file_name,
                "file_path": file_path,
                "resource_suffix": resource_suffix,
                "palette_values": list(palette),
            }
        )
        for mesh_obj in chunk.mesh_objects:
            localize_vertex_groups_for_palette(mesh_obj, palette, chunk.name)
            object_records.append(
                {
                    "object": mesh_obj.name,
                    "chunk_collection": chunk.name,
                    "host_ib_hash": ib_hash,
                    "host_match_index_count": match_index_count,
                    "chunk_index": chunk_index,
                    "palette_file": file_name,
                    "local_bone_count": len(palette),
                }
            )

    manifest = {
        "export_collection": export_collection.name,
        "buffer_dir": buffer_dir,
        "palettes": palette_records,
        "objects": object_records,
        "note": (
            "Export Collection child collections are final draw chunks. "
            "Each child collection name must contain <ib_hash>-<match_index_count>-<chunk_index>. "
            "Prepare Export localizes vertex groups in place, so export the same collection after running it."
        ),
    }
    manifest_path = write_json(os.path.join(normalized_output_dir, EXPORT_MANIFEST_FILE_NAME), manifest)
    bonestore_ini_path = _regenerate_bonestore_ini_if_possible(normalized_output_dir, local_palette_records)
    return {
        "manifest_path": manifest_path,
        "bonestore_ini_path": bonestore_ini_path,
        "export_collection_name": export_collection.name,
        "objects": len(object_records),
        "palettes": len(palette_records),
    }


def localize_vertex_groups_for_palette(mesh_obj, palette: tuple[int, ...], chunk_name: str = "") -> None:
    global_to_local = {global_group: local_index for local_index, global_group in enumerate(palette)}

    weights_by_local = {local_index: {} for local_index in range(len(palette))}
    for global_group, vertex_index, weight in _iter_weighted_global_assignments(mesh_obj):
        local_index = global_to_local.get(global_group)
        if local_index is None:
            continue
        previous_weight = weights_by_local[local_index].get(vertex_index, 0.0)
        weights_by_local[local_index][vertex_index] = max(previous_weight, weight)

    # Rebuild the groups instead of only renaming them. The 3Dmigoto exporter
    # writes BLENDINDICES from Blender's internal vertex-group indices, so the
    # internal order must be local 0..n-1 as well as the visible names.
    for vertex_group in list(mesh_obj.vertex_groups):
        mesh_obj.vertex_groups.remove(vertex_group)

    for local_index in range(len(palette)):
        vertex_group = mesh_obj.vertex_groups.new(name=str(local_index))
        assignments = weights_by_local.get(local_index, {})
        for vertex_index, weight in assignments.items():
            vertex_group.add([vertex_index], weight, "REPLACE")

    mesh_obj[_LOCALIZED_PALETTE_PROP] = list(palette)
    if chunk_name:
        mesh_obj[_LOCALIZED_CHUNK_PROP] = chunk_name


def _build_export_chunks(export_collection) -> tuple[ExportChunk, ...]:
    child_chunks: list[ExportChunk] = []
    for child_collection in export_collection.children:
        identity = _resolve_chunk_collection_identity(child_collection)
        if identity is None:
            continue
        mesh_objects = tuple(_iter_mesh_objects_recursive(child_collection))
        if not mesh_objects:
            continue
        child_chunks.append(
            ExportChunk(
                name=child_collection.name,
                ib_hash=identity[0],
                match_index_count=identity[1],
                chunk_index=identity[2],
                mesh_objects=mesh_objects,
            )
        )

    if child_chunks:
        direct_meshes = [obj.name for obj in export_collection.objects if obj.type == "MESH"]
        if direct_meshes:
            raise ValueError(
                f"{export_collection.name}: move direct mesh object(s) into a chunk child collection: "
                + ", ".join(direct_meshes[:5])
            )
        return tuple(sorted(child_chunks, key=lambda chunk: (chunk.ib_hash, chunk.match_index_count, chunk.chunk_index)))

    direct_meshes = [obj.name for obj in export_collection.objects if obj.type == "MESH"]
    extra_hint = ""
    if direct_meshes:
        extra_hint = " Direct mesh object(s) must be moved into a chunk child collection: " + ", ".join(
            direct_meshes[:5]
        )
    raise ValueError(
        f"{export_collection.name}: no export chunk child collections found. "
        "Create child collections named like fe47dc61-7014-0 and put meshes inside them."
        + extra_hint
    )


def _resolve_chunk_collection_identity(collection) -> tuple[str, int, int] | None:
    match = _CHUNK_COLLECTION_RE.search(str(collection.name or ""))
    if not match:
        return None
    return match.group("hash").lower(), int(match.group("count")), int(match.group("chunk") or 0)


def _iter_mesh_objects_recursive(collection):
    seen_names: set[str] = set()

    def walk(current_collection):
        for obj in current_collection.objects:
            if obj.type != "MESH" or obj.name in seen_names:
                continue
            seen_names.add(obj.name)
            yield obj
        for child in current_collection.children:
            yield from walk(child)

    yield from walk(collection)


def _validate_single_chunk_membership(chunks: tuple[ExportChunk, ...]) -> None:
    memberships: dict[str, list[str]] = {}
    for chunk in chunks:
        for mesh_obj in chunk.mesh_objects:
            memberships.setdefault(mesh_obj.name, []).append(chunk.name)
    duplicated = {name: names for name, names in memberships.items() if len(names) > 1}
    if duplicated:
        first_name, chunk_names = next(iter(duplicated.items()))
        raise ValueError(
            f"{first_name}: the same object is present in multiple export chunks: {', '.join(chunk_names)}. "
            "Direct in-place localization requires one object to belong to only one final chunk."
        )


def _collect_used_groups_for_chunk(chunk: ExportChunk) -> set[int]:
    used_groups: set[int] = set()
    for mesh_obj in chunk.mesh_objects:
        used_groups.update(_collect_used_numeric_vertex_groups(mesh_obj))
    if not used_groups:
        raise ValueError(f"{chunk.name}: no weighted numeric vertex groups found")
    return used_groups


def _collect_used_numeric_vertex_groups(mesh_obj) -> set[int]:
    used_groups = {global_group for global_group, _vertex_index, _weight in _iter_weighted_global_assignments(mesh_obj)}
    if not used_groups:
        raise ValueError(f"{mesh_obj.name}: no weighted numeric vertex groups found")
    return used_groups


def _iter_weighted_global_assignments(mesh_obj):
    group_index_to_global = _build_group_index_to_global_map(mesh_obj)
    for vertex in mesh_obj.data.vertices:
        for group_element in vertex.groups:
            global_group = group_index_to_global.get(int(group_element.group))
            if global_group is None:
                continue
            weight = float(group_element.weight)
            if weight <= 0.0:
                continue
            yield global_group, vertex.index, weight


def _build_group_index_to_global_map(mesh_obj) -> dict[int, int]:
    localized_palette = _get_existing_localized_palette(mesh_obj)
    group_index_to_global = {}
    for vertex_group in mesh_obj.vertex_groups:
        numeric_group = _parse_numeric_group(vertex_group.name)
        if numeric_group is None:
            continue
        if localized_palette is not None and 0 <= numeric_group < len(localized_palette):
            group_index_to_global[vertex_group.index] = int(localized_palette[numeric_group])
        else:
            group_index_to_global[vertex_group.index] = numeric_group
    return group_index_to_global


def _get_existing_localized_palette(mesh_obj) -> tuple[int, ...] | None:
    raw_palette = mesh_obj.get(_LOCALIZED_PALETTE_PROP)
    if not raw_palette:
        return None
    try:
        return tuple(int(value) for value in raw_palette)
    except (TypeError, ValueError):
        return None


def _parse_numeric_group(group_name: str) -> int | None:
    raw_name = str(group_name).strip()
    if not _NUMERIC_GROUP_RE.match(raw_name):
        return None
    return int(raw_name)


def _regenerate_bonestore_ini_if_possible(
    output_dir: str,
    local_palette_records: list[LocalPaletteRecord],
) -> str:
    manifest_path = os.path.join(output_dir, CAPTURE_MANIFEST_FILE_NAME)
    if not os.path.exists(manifest_path):
        return os.path.join(output_dir, BONESTORE_INI_FILE_NAME)

    manifest = read_json(manifest_path)
    part_records = [_part_record_from_manifest_record(record) for record in manifest.get("part_records", [])]
    if not part_records:
        return os.path.join(output_dir, BONESTORE_INI_FILE_NAME)
    return write_bonestore_ini(output_dir, part_records, local_palette_records)


def _part_record_from_manifest_record(record: dict) -> PartRecord:
    return PartRecord(
        draw_index=int(record.get("draw_index", 0)),
        object_name=str(record.get("object_name", "")),
        vs_hash=str(record.get("vs_hash", "")).lower(),
        ib_hash=str(record.get("ib_hash", "")).lower(),
        match_index_count=int(record.get("match_index_count", 0)),
        bone_count=int(record.get("capture_bone_count", record.get("bone_count", 0))),
        global_bone_base=int(record.get("global_bone_base", 0)),
        vb2_path=str(record.get("vb2_path", "")),
        vs_t0_path=str(record.get("vs_t0_path", "")),
        vs_cb1_path=str(record.get("vs_cb1_path", "")),
        vs_cb1_first_constant=int(record.get("vs_cb1_first_constant", -1)),
        vs_cb1_num_constants=int(record.get("vs_cb1_num_constants", -1)),
    )
