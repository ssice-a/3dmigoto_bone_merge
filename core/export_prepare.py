"""Prepare export meshes and local palettes for BI4-compatible output."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import bpy

from ..constants import (
    BI4_MAX_BONE_COUNT,
    BONESTORE_INI_FILE_NAME,
    BUFFER_EXPORT_DIR_NAME,
    BMC_EXPORT_CHUNK_PROP,
    BMC_EXPORT_PALETTE_PROP,
    BMC_VERTEX_GROUP_STATE_GLOBAL,
    BMC_VERTEX_GROUP_STATE_EXPORT_LOCAL,
    BMC_VERTEX_GROUP_STATE_PROP,
    CAPTURE_MANIFEST_FILE_NAME,
    EXPORT_MANIFEST_FILE_NAME,
)
from .hlsl_assets import export_required_hlsl
from .ini_export import build_bonestore_namespace, write_bonestore_ini
from .io import ensure_directory, read_json, write_json, write_uint32_buffer
from .models import LocalPaletteRecord, PartRecord

_NUMERIC_GROUP_RE = re.compile(r"^\d+$")
_CHUNK_COLLECTION_RE = re.compile(r"(?P<hash>[0-9A-Fa-f]{8})[-_](?P<count>\d+)(?:[-_](?P<chunk>\d+))?")


@dataclass(frozen=True)
class ExportChunk:
    name: str
    ib_hash: str
    match_index_count: int
    chunk_index: int
    mesh_objects: tuple[object, ...]


@dataclass(frozen=True)
class MeshPrepareState:
    mesh_obj: object
    is_localized: bool
    localized_palette: tuple[int, ...]
    export_chunk: str
    used_global_groups: tuple[int, ...]


def prepare_export_collection(
    context,
    source_collection,
    build_collection,
    output_dir: str | None = None,
    internal_manifest_dir: str | None = None,
    capture_manifest_path: str | None = None,
):
    if source_collection is None:
        raise ValueError("Export source collection is not set")
    if build_collection is None:
        raise ValueError("Export build collection is not set")

    normalized_output_dir = ensure_directory(output_dir or context.scene.bmc_output_dir or context.scene.bmc_frameanalysis_dir)
    buffer_dir = ensure_directory(os.path.join(normalized_output_dir, BUFFER_EXPORT_DIR_NAME))
    hlsl_dir = export_required_hlsl(normalized_output_dir)

    _rebuild_build_collection_from_source(source_collection, build_collection)
    chunks = _build_export_chunks(build_collection)
    _validate_single_chunk_membership(chunks)

    palette_records = []
    local_palette_records: list[LocalPaletteRecord] = []
    object_records = []

    for chunk in chunks:
        mesh_states = _inspect_chunk_prepare_states(chunk)
        global_groups = _collect_used_groups_for_chunk_states(chunk, mesh_states)
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
        for mesh_state in mesh_states:
            mesh_obj = mesh_state.mesh_obj
            if _can_reuse_localized_mesh(mesh_state, palette):
                mesh_obj[BMC_EXPORT_PALETTE_PROP] = list(palette)
                mesh_obj[BMC_VERTEX_GROUP_STATE_PROP] = BMC_VERTEX_GROUP_STATE_EXPORT_LOCAL
                mesh_obj[BMC_EXPORT_CHUNK_PROP] = chunk.name
            else:
                if mesh_state.is_localized:
                    _restore_mesh_to_global_state(mesh_obj, mesh_state.localized_palette)
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
        "export_source_collection": source_collection.name,
        "export_collection": build_collection.name,
        "buffer_dir": buffer_dir,
        "bonestore_namespace": build_bonestore_namespace(normalized_output_dir),
        "palettes": palette_records,
        "objects": object_records,
        "note": (
            "Export Source Collection child collections are final draw chunks. "
            "Prepare Export rebuilds a disposable Export Build Collection from those source chunks, "
            "then localizes only the build copies before writing runtime files."
        ),
    }
    manifest_dir = ensure_directory(internal_manifest_dir or normalized_output_dir)
    manifest_path = write_json(os.path.join(manifest_dir, EXPORT_MANIFEST_FILE_NAME), manifest)
    bonestore_ini_path = _regenerate_bonestore_ini_if_possible(
        normalized_output_dir,
        local_palette_records,
        capture_manifest_path=capture_manifest_path,
    )
    return {
        "manifest_path": manifest_path,
        "bonestore_ini_path": bonestore_ini_path,
        "export_collection_name": build_collection.name,
        "output_dir": normalized_output_dir,
        "buffer_dir": buffer_dir,
        "hlsl_dir": hlsl_dir,
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

    mesh_obj[BMC_EXPORT_PALETTE_PROP] = list(palette)
    mesh_obj[BMC_VERTEX_GROUP_STATE_PROP] = BMC_VERTEX_GROUP_STATE_EXPORT_LOCAL
    if chunk_name:
        mesh_obj[BMC_EXPORT_CHUNK_PROP] = chunk_name


def _inspect_chunk_prepare_states(chunk: ExportChunk) -> tuple[MeshPrepareState, ...]:
    return tuple(_inspect_mesh_prepare_state(mesh_obj) for mesh_obj in chunk.mesh_objects)


def _inspect_mesh_prepare_state(mesh_obj) -> MeshPrepareState:
    localized_palette = _get_existing_localized_palette(mesh_obj)
    state = str(mesh_obj.get(BMC_VERTEX_GROUP_STATE_PROP, "") or "")
    export_chunk = str(mesh_obj.get(BMC_EXPORT_CHUNK_PROP, "") or "")

    if localized_palette is None and state != BMC_VERTEX_GROUP_STATE_EXPORT_LOCAL and not export_chunk:
        used_groups = tuple(sorted(_collect_used_numeric_vertex_groups(mesh_obj)))
        return MeshPrepareState(
            mesh_obj=mesh_obj,
            is_localized=False,
            localized_palette=(),
            export_chunk="",
            used_global_groups=used_groups,
        )

    if localized_palette is None:
        raise ValueError(
            f"{mesh_obj.name}: export-local metadata is incomplete. "
            "Rebuild this mesh from a clean global-source object before Prepare Export."
        )

    used_local_groups = _collect_used_numeric_vertex_groups(mesh_obj)
    if any(local_index < 0 or local_index >= len(localized_palette) for local_index in used_local_groups):
        raise ValueError(
            f"{mesh_obj.name}: current local groups exceed saved palette range. "
            "Rebuild this mesh from a clean global-source object before Prepare Export."
        )

    used_global_groups = tuple(sorted({int(localized_palette[local_index]) for local_index in used_local_groups}))
    return MeshPrepareState(
        mesh_obj=mesh_obj,
        is_localized=True,
        localized_palette=localized_palette,
        export_chunk=export_chunk,
        used_global_groups=used_global_groups,
    )


def _collect_used_groups_for_chunk_states(chunk: ExportChunk, mesh_states: tuple[MeshPrepareState, ...]) -> set[int]:
    used_groups: set[int] = set()
    for mesh_state in mesh_states:
        used_groups.update(mesh_state.used_global_groups)
    if not used_groups:
        raise ValueError(f"{chunk.name}: no weighted numeric vertex groups found")
    return used_groups


def _can_reuse_localized_mesh(mesh_state: MeshPrepareState, target_palette: tuple[int, ...]) -> bool:
    if not mesh_state.is_localized or tuple(target_palette) != tuple(mesh_state.localized_palette):
        return False
    mesh_obj = mesh_state.mesh_obj
    if len(mesh_obj.vertex_groups) != len(target_palette):
        return False
    for expected_local_index in range(len(target_palette)):
        try:
            vertex_group = mesh_obj.vertex_groups[expected_local_index]
        except Exception:
            return False
        if vertex_group.index != expected_local_index:
            return False
    return True


def _restore_mesh_to_global_state(mesh_obj, localized_palette: tuple[int, ...]) -> None:
    if len(mesh_obj.vertex_groups) != len(localized_palette):
        raise ValueError(
            f"{mesh_obj.name}: export-local vertex groups no longer match the saved palette. "
            "Rebuild this mesh from a clean global-source object before changing export host."
        )

    temp_names: list[str] = []
    for local_index in range(len(localized_palette)):
        vertex_group = mesh_obj.vertex_groups[local_index]
        temp_name = f"__bmc_restore__{local_index}"
        vertex_group.name = temp_name
        temp_names.append(temp_name)

    for local_index, global_index in enumerate(localized_palette):
        mesh_obj.vertex_groups[temp_names[local_index]].name = str(int(global_index))

    if BMC_EXPORT_PALETTE_PROP in mesh_obj:
        del mesh_obj[BMC_EXPORT_PALETTE_PROP]
    if BMC_EXPORT_CHUNK_PROP in mesh_obj:
        del mesh_obj[BMC_EXPORT_CHUNK_PROP]
    mesh_obj[BMC_VERTEX_GROUP_STATE_PROP] = BMC_VERTEX_GROUP_STATE_GLOBAL


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


def _rebuild_build_collection_from_source(source_collection, build_collection) -> None:
    source_chunks = _build_export_chunks(source_collection)
    _clear_collection_tree(build_collection)

    for source_chunk in source_chunks:
        build_chunk_collection = bpy.data.collections.new(f"BMC_EXPORT__{source_chunk.name}")
        build_collection.children.link(build_chunk_collection)
        for mesh_obj in source_chunk.mesh_objects:
            _assert_mesh_is_global_source(mesh_obj)
            build_obj = _clone_mesh_object_for_build(mesh_obj)
            build_chunk_collection.objects.link(build_obj)


def _clear_collection_tree(root_collection) -> None:
    for child_collection in list(root_collection.children):
        _remove_collection_tree(child_collection)
    for obj in list(root_collection.objects):
        _remove_object_from_blender(obj)


def _remove_collection_tree(collection) -> None:
    for child_collection in list(collection.children):
        _remove_collection_tree(child_collection)
    for obj in list(collection.objects):
        _remove_object_from_blender(obj)
    bpy.data.collections.remove(collection)


def _remove_object_from_blender(obj) -> None:
    object_type = obj.type
    data_block = getattr(obj, "data", None)
    bpy.data.objects.remove(obj, do_unlink=True)
    if object_type == "MESH" and data_block is not None and data_block.users == 0:
        bpy.data.meshes.remove(data_block)


def _clone_mesh_object_for_build(mesh_obj):
    build_obj = mesh_obj.copy()
    if mesh_obj.data is not None:
        build_obj.data = mesh_obj.data.copy()
    build_obj.name = f"{mesh_obj.name}.BMC_EXPORT"
    for prop_name in (BMC_EXPORT_PALETTE_PROP, BMC_EXPORT_CHUNK_PROP):
        if prop_name in build_obj:
            del build_obj[prop_name]
    build_obj[BMC_VERTEX_GROUP_STATE_PROP] = BMC_VERTEX_GROUP_STATE_GLOBAL
    return build_obj


def _assert_mesh_is_global_source(mesh_obj) -> None:
    localized_palette = _get_existing_localized_palette(mesh_obj)
    state = str(mesh_obj.get(BMC_VERTEX_GROUP_STATE_PROP, "") or "")
    export_chunk = str(mesh_obj.get(BMC_EXPORT_CHUNK_PROP, "") or "")
    if localized_palette is None and state != BMC_VERTEX_GROUP_STATE_EXPORT_LOCAL and not export_chunk:
        return
    raise ValueError(
        f"{mesh_obj.name}: Export Source Collection must contain clean global-source meshes, "
        "not old export-local build copies."
    )


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
    group_index_to_global = {}
    for vertex_group in mesh_obj.vertex_groups:
        numeric_group = _parse_numeric_group(vertex_group.name)
        if numeric_group is None:
            continue
        group_index_to_global[vertex_group.index] = numeric_group
    return group_index_to_global


def _get_existing_localized_palette(mesh_obj) -> tuple[int, ...] | None:
    raw_palette = mesh_obj.get(BMC_EXPORT_PALETTE_PROP)
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
    capture_manifest_path: str | None = None,
) -> str:
    manifest_path = os.path.abspath(capture_manifest_path or "") if capture_manifest_path else ""
    if not manifest_path or not os.path.exists(manifest_path):
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
