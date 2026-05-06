"""Main FrameAnalysis analyzer for the redesigned BoneMerge workflow."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..constants import CAPTURE_MANIFEST_FILE_NAME
from .io import write_json


_HASH_RE = re.compile(r"^[0-9a-fA-F]{8}$")
_FILE_SHADER_TAG_RE = re.compile(r"-(?:vs|ps|cs)=[0-9a-fA-F]+")
_DRAW_PREFIX_RE = re.compile(r"^(?P<draw>\d{6})-(?P<body>.+)\.(?P<ext>[^.]+)$")
_RESOURCE_HASH_RE = re.compile(r"^(?P<hash>[0-9a-fA-F]{8})(?:\((?P<backing>[0-9a-fA-F]{8})\))?")
_VS_TAG_RE = re.compile(r"(?:^|-)vs=(?P<hash>[0-9a-fA-F]+)")
_PS_TAG_RE = re.compile(r"(?:^|-)ps=(?P<hash>[0-9a-fA-F]+)")
_DUMP_REDIRECT_RE = re.compile(r"Dumping (?:Buffer|Texture2D)\s+(?P<src>.+?)\s+->\s+(?P<dst>.+)$")
_VS_SET_RE = re.compile(r"^(?P<draw>\d{6}) VSSetShader\(.* hash=(?P<hash>[0-9a-fA-F]+)\s*$")
_PS_SET_RE = re.compile(r"^(?P<draw>\d{6}) PSSetShader\(.* hash=(?P<hash>[0-9a-fA-F]+)\s*$")
_OM_SET_RE = re.compile(r"^(?P<draw>\d{6}) OMSetRenderTargets\(NumViews:(?P<count>\d+),")
_IA_IB_RE = re.compile(
    r"^(?P<draw>\d{6}) IASetIndexBuffer\(.* Format:(?P<format>\d+), Offset:(?P<offset>\d+)\) "
    r"hash=(?P<hash>[0-9a-fA-F]+)\s*$"
)
_DRAW_INDEXED_RE = re.compile(
    r"^(?P<draw>\d{6}) DrawIndexed(?:Instanced)?\("
    r"(?:IndexCountPerInstance|IndexCount):(?P<count>\d+)"
    r"(?:,\s*InstanceCount:(?P<instances>\d+))?,\s*"
    r"StartIndexLocation:(?P<start>-?\d+),\s*"
    r"BaseVertexLocation:(?P<base>-?\d+)"
)
_HEADER_INT_RE = re.compile(r"^(?P<name>byte offset|first index|index count|stride|first vertex|vertex count):\s*(?P<value>-?\d+)")
_HEADER_TEXT_RE = re.compile(r"^(?P<name>topology|format):\s*(?P<value>.+)$")
_ELEMENT_RE = re.compile(r"^element\[(?P<index>\d+)\]:")
_CB1_RE = re.compile(r"^cb1\[(?P<row>\d+)\]\.(?P<component>[xyzw]):\s*(?P<value>[-+0-9.eE]+)")


@dataclass(frozen=True)
class DumpFile:
    draw_index: int
    slot: str
    resource_hash: str
    backing_hash: str
    vs_hash: str
    ps_hash: str
    extension: str
    path: str
    data_path: str


@dataclass(frozen=True)
class DrawState:
    draw_index: int
    vs_hash: str = ""
    ps_hash: str = ""
    rt_count: int = -1
    index_count: int = 0
    start_index: int = 0
    base_vertex: int = 0
    instance_count: int = 1
    ia_index_format: int = -1
    ia_index_offset: int = -1
    ia_backing_hash: str = ""


@dataclass
class HeaderElement:
    semantic_name: str = ""
    semantic_index: int = 0
    fmt: str = ""
    input_slot: int = -1
    aligned_byte_offset: int = -1

    def to_dict(self) -> dict:
        return {
            "semantic": f"{self.semantic_name}{self.semantic_index}",
            "semantic_name": self.semantic_name,
            "semantic_index": int(self.semantic_index),
            "format": self.fmt,
            "input_slot": int(self.input_slot),
            "aligned_byte_offset": int(self.aligned_byte_offset),
        }


@dataclass
class BufferHeader:
    byte_offset: int = 0
    first_index: int = 0
    index_count: int = 0
    stride: int = 0
    first_vertex: int = 0
    vertex_count: int = 0
    topology: str = ""
    fmt: str = ""
    elements: list[HeaderElement] = field(default_factory=list)


def write_main_analysis_manifest(
    frameanalysis_dir: str,
    target_ib_hashes: Iterable[str],
    output_dir: str | None = None,
) -> tuple[dict, str]:
    """Analyze the main FrameAnalysis folder and write the redesigned manifest."""

    normalized_frameanalysis_dir = os.path.abspath(frameanalysis_dir)
    normalized_output_dir = os.path.abspath(output_dir or normalized_frameanalysis_dir)
    payload = analyze_main_frameanalysis(normalized_frameanalysis_dir, target_ib_hashes)
    manifest_path = write_json(os.path.join(normalized_output_dir, CAPTURE_MANIFEST_FILE_NAME), payload)
    return payload, manifest_path


def analyze_main_frameanalysis(frameanalysis_dir: str, target_ib_hashes: Iterable[str]) -> dict:
    normalized_frameanalysis_dir = os.path.abspath(frameanalysis_dir)
    target_hashes = _normalize_target_hashes(target_ib_hashes)
    if not target_hashes:
        raise ValueError("Target IB hash is required for Main Analyze")
    log_path = os.path.join(normalized_frameanalysis_dir, "log.txt")
    if not os.path.exists(log_path):
        raise ValueError(f"log.txt not found in {normalized_frameanalysis_dir}")

    redirects = _parse_dump_redirects(log_path)
    files_by_draw = _scan_dump_files(normalized_frameanalysis_dir, redirects)
    draw_states = _parse_draw_states(log_path)
    warnings: list[dict] = []

    target_hits = [
        dump_file
        for dump_files in files_by_draw.values()
        for dump_file in dump_files
        if dump_file.slot == "ib" and dump_file.resource_hash in target_hashes and dump_file.extension == "txt"
    ]
    target_hits.sort(key=lambda item: item.draw_index)
    if not target_hits:
        raise ValueError(f"No target IB dump found for: {', '.join(target_hashes)}")

    shadow_vs_hashes = _first_distinct_vs_pair(target_hits, draw_states, warnings)
    if not shadow_vs_hashes:
        raise ValueError("Could not infer shadow VS hash pair from target IB hits")
    normal_vs_hash = shadow_vs_hashes[0]
    transparent_vs_hash = shadow_vs_hashes[1] if len(shadow_vs_hashes) > 1 else ""

    shadow_hits = [
        dump_file
        for dump_files in files_by_draw.values()
        for dump_file in dump_files
        if dump_file.slot == "ib"
        and dump_file.extension == "txt"
        and _state_vs_hash(draw_states, dump_file) in shadow_vs_hashes
    ]
    shadow_hits.sort(key=lambda item: item.draw_index)
    if not shadow_hits:
        raise ValueError("No IB draw hits matched the inferred shadow VS hash pair")

    candidate_keys = {_candidate_key_from_ib_dump(dump_file) for dump_file in shadow_hits}
    all_ib_dumps = [
        dump_file
        for dump_files in files_by_draw.values()
        for dump_file in dump_files
        if dump_file.slot == "ib" and dump_file.extension == "txt"
    ]
    all_hits_by_key: dict[tuple[str, int, int], list[DumpFile]] = {key: [] for key in candidate_keys}
    for dump_file in all_ib_dumps:
        key = _candidate_key_from_ib_dump(dump_file)
        if key in all_hits_by_key:
            all_hits_by_key[key].append(dump_file)

    shadow_draw_indices = sorted({dump_file.draw_index for dump_file in shadow_hits})
    host_draw_index = max(shadow_draw_indices)
    host_dump = next(dump_file for dump_file in reversed(shadow_hits) if dump_file.draw_index == host_draw_index)
    host_key = _candidate_key_from_ib_dump(host_dump)

    draw_hits = [
        _build_draw_hit_payload(
            frameanalysis_dir=normalized_frameanalysis_dir,
            dump_file=dump_file,
            draw_state=draw_states.get(dump_file.draw_index),
            draw_files=files_by_draw.get(dump_file.draw_index, []),
            pass_role=_pass_role_for_vs(_state_vs_hash(draw_states, dump_file), normal_vs_hash, transparent_vs_hash),
        )
        for dump_file in shadow_hits
    ]

    candidates = []
    for key in sorted(candidate_keys, key=lambda item: (-item[2], item[0], item[1])):
        ib_hash, first_index, index_count = key
        all_hits = sorted(all_hits_by_key.get(key, []), key=lambda item: item.draw_index)
        key_shadow_hits = [dump_file for dump_file in shadow_hits if _candidate_key_from_ib_dump(dump_file) == key]
        local_bone_count = _infer_candidate_local_bone_count(files_by_draw, all_hits, warnings)
        candidates.append(
            {
                "enabled": True,
                "ib_hash": ib_hash,
                "match_first_index": int(first_index),
                "match_index_count": int(index_count),
                "display_name": f"{ib_hash}-{index_count}-{first_index}",
                "draw_indices": [dump_file.draw_index for dump_file in all_hits],
                "shadow_draw_indices": [dump_file.draw_index for dump_file in key_shadow_hits],
                "source_index_count": int(index_count),
                "local_bone_count": int(local_bone_count),
                "texture_region_key": f"{ib_hash}-{index_count}-{first_index}",
                "import_paths": _build_import_paths_payload(files_by_draw, all_hits),
            }
        )

    texture_candidates = _build_texture_candidates(files_by_draw, all_hits_by_key)
    bone_pool_order = _build_bone_pool_order(candidates)
    payload = {
        "schema_version": 1,
        "frameanalysis_dir": normalized_frameanalysis_dir,
        "target": {
            "source_ib_hashes": target_hashes,
            "selection_mode": "manual_ib_hash_or_target_list",
        },
        "shadow_stage": {
            "shadow_vs_hashes": shadow_vs_hashes,
            "stage_draw_start": min(shadow_draw_indices),
            "stage_draw_end": max(shadow_draw_indices),
            "normal_vs_hash": normal_vs_hash,
            "transparent_vs_hash": transparent_vs_hash,
            "host_draw_index": int(host_draw_index),
            "host_ib_hash": host_key[0],
            "host_match_first_index": int(host_key[1]),
            "host_match_index_count": int(host_key[2]),
        },
        "candidate_ibs": candidates,
        "draw_hits": draw_hits,
        "texture_candidates": texture_candidates,
        "producer_dispatches": [],
        "bone_pool_order": bone_pool_order,
        "lod_frameanalysis": [],
        "lod_links": [],
        "lod_capture_records": [],
        "validation": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _validate_payload(payload)
    return payload


def _normalize_target_hashes(target_ib_hashes: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_hash in target_ib_hashes:
        value = str(raw_hash or "").strip().lower()
        if not value:
            continue
        if not _HASH_RE.match(value):
            raise ValueError(f"Invalid target IB hash: {raw_hash}")
        if value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _parse_dump_redirects(log_path: str) -> dict[str, str]:
    redirects: dict[str, str] = {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for raw_line in file_handle:
            match = _DUMP_REDIRECT_RE.search(raw_line.rstrip("\n"))
            if not match:
                continue
            src = os.path.normcase(os.path.abspath(match.group("src").strip()))
            dst = os.path.abspath(match.group("dst").strip())
            redirects[src] = dst
    return redirects


def _scan_dump_files(frameanalysis_dir: str, redirects: dict[str, str]) -> dict[int, list[DumpFile]]:
    files_by_draw: dict[int, list[DumpFile]] = {}
    for path in Path(frameanalysis_dir).iterdir():
        if not path.is_file():
            continue
        dump_file = _parse_dump_file(path, redirects)
        if dump_file is None:
            continue
        files_by_draw.setdefault(dump_file.draw_index, []).append(dump_file)
    for draw_files in files_by_draw.values():
        draw_files.sort(key=lambda item: (item.slot, item.extension, item.path))
    return files_by_draw


def _parse_dump_file(path: Path, redirects: dict[str, str]) -> DumpFile | None:
    match = _DRAW_PREFIX_RE.match(path.name)
    if not match:
        return None
    draw_index = int(match.group("draw"))
    body = match.group("body")
    extension = match.group("ext").lower()
    vs_match = _VS_TAG_RE.search(body)
    ps_match = _PS_TAG_RE.search(body)
    vs_hash = vs_match.group("hash").lower() if vs_match else ""
    ps_hash = ps_match.group("hash").lower() if ps_match else ""
    body_without_shaders = _FILE_SHADER_TAG_RE.sub("", body)
    if "=" in body_without_shaders:
        slot, resource_part = body_without_shaders.split("=", 1)
        resource_match = _RESOURCE_HASH_RE.match(resource_part)
        resource_hash = resource_match.group("hash").lower() if resource_match else ""
        backing_hash = (resource_match.group("backing") or "").lower() if resource_match else ""
    else:
        slot = body_without_shaders
        resource_hash = ""
        backing_hash = ""
    absolute_path = os.path.abspath(str(path))
    data_path = _resolve_data_path(absolute_path, redirects)
    return DumpFile(
        draw_index=draw_index,
        slot=slot.lower(),
        resource_hash=resource_hash,
        backing_hash=backing_hash,
        vs_hash=vs_hash,
        ps_hash=ps_hash,
        extension=extension,
        path=absolute_path,
        data_path=data_path,
    )


def _resolve_data_path(path: str, redirects: dict[str, str]) -> str:
    normalized_path = os.path.normcase(os.path.abspath(path))
    redirected = redirects.get(normalized_path)
    if redirected:
        return redirected
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    return ""


def _parse_draw_states(log_path: str) -> dict[int, DrawState]:
    current_vs = ""
    current_ps = ""
    current_rt_count = -1
    current_ia_index_format = -1
    current_ia_index_offset = -1
    current_ia_backing_hash = ""
    draw_states: dict[int, DrawState] = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for raw_line in file_handle:
            line = raw_line.rstrip("\n")
            match = _VS_SET_RE.match(line)
            if match:
                current_vs = match.group("hash").lower()
                continue
            match = _PS_SET_RE.match(line)
            if match:
                current_ps = match.group("hash").lower()
                continue
            match = _OM_SET_RE.match(line)
            if match:
                current_rt_count = int(match.group("count"))
                continue
            match = _IA_IB_RE.match(line)
            if match:
                current_ia_index_format = int(match.group("format"))
                current_ia_index_offset = int(match.group("offset"))
                current_ia_backing_hash = match.group("hash").lower()
                continue
            match = _DRAW_INDEXED_RE.match(line)
            if not match:
                continue
            draw_index = int(match.group("draw"))
            draw_states[draw_index] = DrawState(
                draw_index=draw_index,
                vs_hash=current_vs,
                ps_hash=current_ps,
                rt_count=current_rt_count,
                index_count=int(match.group("count")),
                start_index=int(match.group("start")),
                base_vertex=int(match.group("base")),
                instance_count=int(match.group("instances") or 1),
                ia_index_format=current_ia_index_format,
                ia_index_offset=current_ia_index_offset,
                ia_backing_hash=current_ia_backing_hash,
            )
    return draw_states


def _first_distinct_vs_pair(
    target_hits: list[DumpFile],
    draw_states: dict[int, DrawState],
    warnings: list[dict],
) -> list[str]:
    selected: list[str] = []
    for dump_file in target_hits:
        vs_hash = _state_vs_hash(draw_states, dump_file)
        if not vs_hash or vs_hash in selected:
            continue
        selected.append(vs_hash)
        if len(selected) >= 2:
            break
    if len(selected) == 1:
        warnings.append(
            {
                "severity": "warning",
                "code": "single_shadow_vs_hash",
                "message": "Only one shadow VS hash was inferred from target IB hits.",
                "draw_indices": [dump_file.draw_index for dump_file in target_hits[:3]],
            }
        )
    return selected


def _state_vs_hash(draw_states: dict[int, DrawState], dump_file: DumpFile) -> str:
    draw_state = draw_states.get(dump_file.draw_index)
    if draw_state and draw_state.vs_hash:
        return draw_state.vs_hash
    return dump_file.vs_hash


def _candidate_key_from_ib_dump(dump_file: DumpFile) -> tuple[str, int, int]:
    header = _parse_buffer_header(dump_file.path)
    first_index = header.first_index
    index_count = header.index_count
    return dump_file.resource_hash.lower(), int(first_index), int(index_count)


@lru_cache(maxsize=4096)
def _parse_buffer_header(path: str) -> BufferHeader:
    header = BufferHeader()
    current_element: HeaderElement | None = None
    if not path or not os.path.exists(path):
        return header
    with open(path, "r", encoding="utf-8", errors="replace") as file_handle:
        for raw_line in file_handle:
            line = raw_line.rstrip("\n")
            if line == "vertex-data:":
                break
            if line and line[0].isdigit() and ":" not in line:
                break
            match = _HEADER_INT_RE.match(line)
            if match:
                _assign_header_int(header, match.group("name"), int(match.group("value")))
                continue
            match = _HEADER_TEXT_RE.match(line)
            if match:
                _assign_header_text(header, match.group("name"), match.group("value"))
                continue
            match = _ELEMENT_RE.match(line)
            if match:
                current_element = HeaderElement()
                header.elements.append(current_element)
                continue
            if current_element is None:
                continue
            stripped = line.strip()
            if stripped.startswith("SemanticName:"):
                current_element.semantic_name = stripped.split(":", 1)[1].strip().upper()
            elif stripped.startswith("SemanticIndex:"):
                current_element.semantic_index = int(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Format:"):
                current_element.fmt = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("InputSlot:"):
                current_element.input_slot = int(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("AlignedByteOffset:"):
                current_element.aligned_byte_offset = int(stripped.split(":", 1)[1].strip())
    return header


def _assign_header_int(header: BufferHeader, name: str, value: int) -> None:
    if name == "byte offset":
        header.byte_offset = value
    elif name == "first index":
        header.first_index = value
    elif name == "index count":
        header.index_count = value
    elif name == "stride":
        header.stride = value
    elif name == "first vertex":
        header.first_vertex = value
    elif name == "vertex count":
        header.vertex_count = value


def _assign_header_text(header: BufferHeader, name: str, value: str) -> None:
    if name == "topology":
        header.topology = value.strip()
    elif name == "format":
        header.fmt = value.strip()


def _pass_role_for_vs(vs_hash: str, normal_vs_hash: str, transparent_vs_hash: str) -> str:
    if transparent_vs_hash and vs_hash == transparent_vs_hash:
        return "transparent_shadow"
    if normal_vs_hash and vs_hash == normal_vs_hash:
        return "normal_shadow"
    return "shadow_unknown"


def _build_draw_hit_payload(
    frameanalysis_dir: str,
    dump_file: DumpFile,
    draw_state: DrawState | None,
    draw_files: list[DumpFile],
    pass_role: str,
) -> dict:
    ib_header = _parse_buffer_header(dump_file.path)
    ib_binary_path = _binary_data_path_for_slot(draw_files, "ib")
    cb1_file = _first_slot_file(draw_files, "vs-cb1", "txt")
    vs_t0_file = _first_slot_file(draw_files, "vs-t0", "buf")
    cb1_xy = _parse_cb1_xy(cb1_file.path if cb1_file else "")
    return {
        "draw_index": dump_file.draw_index,
        "ib_hash": dump_file.resource_hash,
        "first_index": int(ib_header.first_index),
        "index_count": int(ib_header.index_count or (draw_state.index_count if draw_state else 0)),
        "base_vertex": int(draw_state.base_vertex if draw_state else 0),
        "instance_count": int(draw_state.instance_count if draw_state else 1),
        "vs_hash": (draw_state.vs_hash if draw_state and draw_state.vs_hash else dump_file.vs_hash),
        "ps_hash": (draw_state.ps_hash if draw_state and draw_state.ps_hash else dump_file.ps_hash),
        "rt_count": int(draw_state.rt_count if draw_state else -1),
        "ib_dump_path": _relpath(dump_file.path, frameanalysis_dir),
        "ib_buf_path": _relpath(ib_binary_path, frameanalysis_dir),
        "vb_dump_paths": _build_vb_paths_payload(frameanalysis_dir, draw_files),
        "vs_t0_hash": _resource_hash_or_data_hash(vs_t0_file),
        "vs_t0_path": _relpath(vs_t0_file.data_path if vs_t0_file else "", frameanalysis_dir),
        "cb1_path": _relpath(cb1_file.path if cb1_file else "", frameanalysis_dir),
        "cb1_palette_current": cb1_xy[0],
        "cb1_palette_previous": cb1_xy[1],
        "producer_dispatch_index": -1,
        "pass_role": pass_role,
    }


def _first_slot_file(draw_files: list[DumpFile], slot: str, extension: str = "") -> DumpFile | None:
    for dump_file in draw_files:
        if dump_file.slot != slot:
            continue
        if extension and dump_file.extension != extension:
            continue
        return dump_file
    return None


def _parse_cb1_xy(path: str) -> list[float]:
    if not path or not os.path.exists(path):
        return [0.0, 0.0]
    values: dict[str, float] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as file_handle:
        for raw_line in file_handle:
            match = _CB1_RE.match(raw_line.strip())
            if not match:
                continue
            if int(match.group("row")) != 5:
                continue
            component = match.group("component")
            if component in {"x", "y"}:
                values[component] = float(match.group("value"))
            if "x" in values and "y" in values:
                break
    return [float(values.get("x", 0.0)), float(values.get("y", 0.0))]


def _build_vb_paths_payload(frameanalysis_dir: str, draw_files: list[DumpFile]) -> dict:
    payload: dict[str, dict] = {}
    for dump_file in draw_files:
        if not dump_file.slot.startswith("vb") or dump_file.extension != "txt":
            continue
        header = _parse_buffer_header(dump_file.path)
        slot_index = _slot_index(dump_file.slot)
        binary_data_path = _binary_data_path_for_slot(draw_files, dump_file.slot)
        payload[dump_file.slot] = {
            "hash": dump_file.resource_hash,
            "backing_hash": dump_file.backing_hash,
            "txt_path": _relpath(dump_file.path, frameanalysis_dir),
            "buf_path": _relpath(binary_data_path, frameanalysis_dir),
            "offset": int(header.byte_offset),
            "stride": int(header.stride),
            "vertex_count": int(header.vertex_count),
            "fields": [
                element.to_dict()
                for element in header.elements
                if slot_index < 0 or element.input_slot == slot_index
            ],
        }
    return payload


def _build_import_paths_payload(files_by_draw: dict[int, list[DumpFile]], all_hits: list[DumpFile]) -> dict:
    if not all_hits:
        return {"ib": "", "vb": {}, "layout": ""}
    first_hit = all_hits[0]
    first_draw_files = files_by_draw.get(first_hit.draw_index, [])
    all_draw_files = [dump_file for hit in all_hits for dump_file in files_by_draw.get(hit.draw_index, [])]
    vb_slots = sorted({dump_file.slot for dump_file in all_draw_files if dump_file.slot.startswith("vb")})
    return {
        "ib": first_hit.path,
        "ib_buf": _binary_data_path_for_slot(first_draw_files, "ib"),
        "vb": {
            slot: {
                "txt": [
                    dump_file.path
                    for dump_file in all_draw_files
                    if dump_file.slot == slot and dump_file.extension == "txt"
                ],
                "buf": next(
                    (
                        dump_file.data_path
                        for dump_file in all_draw_files
                        if dump_file.slot == slot and dump_file.extension == "buf" and dump_file.data_path
                    ),
                    "",
                ),
            }
            for slot in vb_slots
        },
        "layout": "",
    }


def _infer_candidate_local_bone_count(
    files_by_draw: dict[int, list[DumpFile]],
    all_hits: list[DumpFile],
    warnings: list[dict],
) -> int:
    for hit in all_hits:
        draw_files = files_by_draw.get(hit.draw_index, [])
        for dump_file in draw_files:
            if not dump_file.slot.startswith("vb") or dump_file.extension != "txt":
                continue
            header = _parse_buffer_header(dump_file.path)
            slot_index = _slot_index(dump_file.slot)
            blend_index_element = next(
                (
                    element
                    for element in header.elements
                    if element.input_slot == slot_index and element.semantic_name == "BLENDINDICES"
                ),
                None,
            )
            if blend_index_element is None:
                continue
            binary_data_path = _binary_data_path_for_slot(draw_files, dump_file.slot)
            max_index = _read_max_blend_index(
                data_path=binary_data_path,
                byte_offset=header.byte_offset,
                stride=header.stride,
                vertex_count=header.vertex_count,
                blend_offset=blend_index_element.aligned_byte_offset,
            )
            if max_index >= 0:
                return int(max_index + 1)
    if all_hits:
        warnings.append(
            {
                "severity": "warning",
                "code": "missing_blendindices",
                "message": f"Could not infer local bone count for {all_hits[0].resource_hash}.",
                "draw_indices": [hit.draw_index for hit in all_hits[:5]],
            }
        )
    return 0


def _binary_data_path_for_slot(draw_files: list[DumpFile], slot: str) -> str:
    for dump_file in draw_files:
        if dump_file.slot == slot and dump_file.extension == "buf" and dump_file.data_path:
            return dump_file.data_path
    return ""


def _read_max_blend_index(
    data_path: str,
    byte_offset: int,
    stride: int,
    vertex_count: int,
    blend_offset: int,
) -> int:
    if not data_path or not os.path.exists(data_path) or stride <= 0 or vertex_count <= 0:
        return -1
    max_index = -1
    try:
        with open(data_path, "rb") as file_handle:
            file_handle.seek(byte_offset)
            raw_vertex_data = file_handle.read(vertex_count * stride)
    except OSError:
        return -1
    for vertex_offset in range(blend_offset, len(raw_vertex_data), stride):
        raw_indices = raw_vertex_data[vertex_offset : vertex_offset + 4]
        if len(raw_indices) < 4:
            break
        max_index = max(max_index, raw_indices[0], raw_indices[1], raw_indices[2], raw_indices[3])
    return max_index


def _build_texture_candidates(
    files_by_draw: dict[int, list[DumpFile]],
    all_hits_by_key: dict[tuple[str, int, int], list[DumpFile]],
) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    key_by_draw: dict[int, tuple[str, int, int]] = {}
    for key, hits in all_hits_by_key.items():
        for hit in hits:
            key_by_draw[hit.draw_index] = key

    for draw_index, key in sorted(key_by_draw.items()):
        region_key = f"{key[0]}-{key[2]}-{key[1]}"
        for dump_file in files_by_draw.get(draw_index, []):
            if not dump_file.slot.startswith("ps-t"):
                continue
            if dump_file.extension not in {"dds", "jpg", "png", "tga", "buf"}:
                continue
            texture_hash = dump_file.resource_hash or _data_hash_from_path(dump_file.data_path)
            if not texture_hash:
                continue
            unique_key = (region_key, dump_file.slot, texture_hash)
            if unique_key in seen:
                continue
            seen.add(unique_key)
            candidates.append(
                {
                    "region_key": region_key,
                    "draw_index": int(draw_index),
                    "ps_hash": dump_file.ps_hash,
                    "rt_count": -1,
                    "slot": dump_file.slot,
                    "hash": texture_hash,
                    "source_path": dump_file.path,
                    "semantic_hint": "",
                }
            )
    return candidates


def _build_bone_pool_order(candidates: list[dict]) -> list[dict]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -int(item.get("source_index_count", 0)),
            -int(item.get("local_bone_count", 0)),
            str(item.get("ib_hash", "")),
            int(item.get("match_first_index", 0)),
        ),
    )
    next_base = 0
    payload = []
    for candidate in ordered:
        local_bone_count = int(candidate.get("local_bone_count", 0))
        payload.append(
            {
                "ib_hash": str(candidate.get("ib_hash", "")),
                "match_first_index": int(candidate.get("match_first_index", 0)),
                "match_index_count": int(candidate.get("match_index_count", 0)),
                "producer_dispatch_index": -1,
                "global_bone_base": int(next_base),
                "local_bone_count": local_bone_count,
                "capture_store_base": int(next_base),
            }
        )
        next_base += max(0, local_bone_count)
    return payload


def _validate_payload(payload: dict) -> None:
    if not payload.get("candidate_ibs"):
        payload.setdefault("validation", []).append(
            {
                "severity": "error",
                "code": "no_candidate_ibs",
                "message": "Analyze did not find any candidate IBs.",
                "draw_indices": [],
            }
        )


def _resource_hash_or_data_hash(dump_file: DumpFile | None) -> str:
    if dump_file is None:
        return ""
    return dump_file.resource_hash or _data_hash_from_path(dump_file.data_path)


def _data_hash_from_path(path: str) -> str:
    if not path:
        return ""
    name = os.path.basename(path)
    match = re.match(r"^(?P<hash>[0-9a-fA-F]{8})", name)
    return match.group("hash").lower() if match else ""


def _slot_index(slot: str) -> int:
    match = re.search(r"(\d+)$", slot)
    return int(match.group(1)) if match else -1


def _relpath(path: str, root: str) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, root).replace("\\", "/")
    except ValueError:
        return path
