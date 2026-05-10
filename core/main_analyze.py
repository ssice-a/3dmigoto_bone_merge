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
from .data_types import annotate_vertex_layout
from .draw_arrays import read_index_array, require_numpy, used_skin_slots
from .io import write_json
from .numpy_buffers import (
    dxgi_format_size,
    max_interleaved_uint4,
    read_interleaved_fields_from_file,
)


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
_VERTEX_DATA_RE = re.compile(
    r"^vb(?P<slot>\d+)\[(?P<vertex>\d+)\]\+(?P<offset>\d+)\s+[^:]+:\s*(?P<values>.+)$"
)
ANALYZER_VERSION = 2


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
    target_ib_hashes: Iterable[str] | None = None,
    output_dir: str | None = None,
) -> tuple[dict, str]:
    """Analyze the main FrameAnalysis folder and write the redesigned manifest."""

    normalized_frameanalysis_dir = os.path.abspath(frameanalysis_dir)
    normalized_output_dir = os.path.abspath(output_dir or normalized_frameanalysis_dir)
    payload = analyze_main_frameanalysis(normalized_frameanalysis_dir, target_ib_hashes or [])
    manifest_path = write_json(os.path.join(normalized_output_dir, CAPTURE_MANIFEST_FILE_NAME), payload)
    return payload, manifest_path


def analyze_main_frameanalysis(frameanalysis_dir: str, target_ib_hashes: Iterable[str] | None = None) -> dict:
    normalized_frameanalysis_dir = os.path.abspath(frameanalysis_dir)
    target_hashes = _normalize_target_hashes(target_ib_hashes or [])
    log_path = os.path.join(normalized_frameanalysis_dir, "log.txt")
    if not os.path.exists(log_path):
        raise ValueError(f"log.txt not found in {normalized_frameanalysis_dir}")

    redirects = _parse_dump_redirects(log_path)
    files_by_draw = _scan_dump_files(normalized_frameanalysis_dir, redirects)
    draw_states = _parse_draw_states(log_path)
    warnings: list[dict] = []

    all_ib_dumps = _all_ib_txt_dumps(files_by_draw)
    if not all_ib_dumps:
        raise ValueError("No IB txt dumps found in FrameAnalysis")
    key_index = _group_ib_dumps_by_key(all_ib_dumps)

    visible_anchor = _discover_visible_anchor_keys(all_ib_dumps, draw_states, target_hashes, warnings)
    discovery = _discover_shadow_stage_from_visible_anchors(
        files_by_draw=files_by_draw,
        draw_states=draw_states,
        all_ib_dumps=all_ib_dumps,
        key_index=key_index,
        visible_anchor=visible_anchor,
        warnings=warnings,
    )
    role_vs_hashes = list(discovery["shadow_vs_hashes"])
    if not role_vs_hashes:
        raise ValueError("Could not infer shadow VS hash pair from FrameAnalysis")
    normal_vs_hash = role_vs_hashes[0]
    transparent_vs_hash = role_vs_hashes[1] if len(role_vs_hashes) > 1 else ""
    stage_draw_start = int(discovery.get("stage_draw_start", 0) or 0)
    stage_draw_end = int(discovery.get("stage_draw_end", 0) or 0)
    shadow_vs_hashes = _expand_shadow_capture_vs_hashes(
        files_by_draw=files_by_draw,
        draw_states=draw_states,
        all_ib_dumps=all_ib_dumps,
        key_index=key_index,
        seed_vs_hashes=role_vs_hashes,
        stage_draw_start=stage_draw_start,
        stage_draw_end=stage_draw_end,
    )

    shadow_hits = [
        dump_file
        for dump_file in all_ib_dumps
        if _state_vs_hash(draw_states, dump_file) in shadow_vs_hashes
        and _draw_in_stage_window(dump_file.draw_index, stage_draw_start, stage_draw_end)
    ]
    shadow_hits.sort(key=lambda item: item.draw_index)
    if not shadow_hits:
        raise ValueError("No IB draw hits matched the inferred shadow VS hash pair")

    shadow_hits_by_key = _group_ib_dumps_by_key(shadow_hits)
    candidate_entries = _build_candidate_entries(
        frameanalysis_dir=normalized_frameanalysis_dir,
        files_by_draw=files_by_draw,
        draw_states=draw_states,
        key_index=key_index,
        shadow_hits_by_key=shadow_hits_by_key,
        shadow_vs_hashes=shadow_vs_hashes,
        warnings=warnings,
    )
    if not candidate_entries:
        raise ValueError("No importable skinned candidate IBs found in FrameAnalysis")

    shadow_draw_indices = sorted({dump_file.draw_index for dump_file in shadow_hits})
    host_draw_index = max(shadow_draw_indices)
    host_dump = next(dump_file for dump_file in reversed(shadow_hits) if dump_file.draw_index == host_draw_index)
    host_key = _candidate_key_from_ib_dump(host_dump)

    draw_hits = []
    for entry in candidate_entries:
        for dump_file in entry["manifest_hits"]:
            vs_hash = _state_vs_hash(draw_states, dump_file)
            is_shadow_hit = dump_file.draw_index in set(entry["shadow_draw_indices"])
            pass_role = _pass_role_for_vs(vs_hash, normal_vs_hash, transparent_vs_hash)
            if not is_shadow_hit and pass_role == "shadow_unknown":
                pass_role = _pass_role_for_material(draw_states.get(dump_file.draw_index))
            payload = _build_draw_hit_payload(
                frameanalysis_dir=normalized_frameanalysis_dir,
                dump_file=dump_file,
                draw_state=draw_states.get(dump_file.draw_index),
                draw_files=files_by_draw.get(dump_file.draw_index, []),
                pass_role=pass_role,
            )
            payload["use_role"] = "both" if is_shadow_hit and dump_file.draw_index == entry["import_draw_index"] else (
                "capture" if is_shadow_hit else "import"
            )
            draw_hits.append(payload)
        entry.pop("manifest_hits", None)

    candidates = [
        entry
        for entry in sorted(
            candidate_entries,
            key=lambda item: (
                -int(item.get("match_index_count", 0)),
                str(item.get("ib_hash", "")),
                int(item.get("match_first_index", 0)),
            ),
        )
    ]
    for candidate in candidates:
        candidate["vertex_layout_key"] = _candidate_source_key(candidate)

    included_key_index = {
        (str(candidate["ib_hash"]), int(candidate["match_first_index"]), int(candidate["match_index_count"])): key_index[
            (str(candidate["ib_hash"]), int(candidate["match_first_index"]), int(candidate["match_index_count"]))
        ]
        for candidate in candidates
        if (str(candidate["ib_hash"]), int(candidate["match_first_index"]), int(candidate["match_index_count"])) in key_index
    }
    texture_candidates = _build_texture_candidates(files_by_draw, draw_states, included_key_index)
    vertex_layout_table = _build_vertex_layout_table(normalized_frameanalysis_dir, candidates)
    bone_pool_order = build_bone_pool_order(candidates)
    payload = {
        "schema_version": 1,
        "analyzer_version": ANALYZER_VERSION,
        "frameanalysis_dir": normalized_frameanalysis_dir,
        "target": {
            "visible_anchor_ibs": [f"{key[0]}-{key[2]}-{key[1]}" for key in visible_anchor.get("anchor_keys", [])],
            "source_ib_hashes": target_hashes,
            "selection_mode": discovery["selection_mode"],
            "auto_discovery": discovery,
        },
        "shadow_stage": {
            "shadow_vs_hashes": shadow_vs_hashes,
            "role_vs_hashes": role_vs_hashes,
            "stage_draw_start": int(stage_draw_start or min(shadow_draw_indices)),
            "stage_draw_end": int(stage_draw_end or max(shadow_draw_indices)),
            "normal_vs_hash": normal_vs_hash,
            "transparent_vs_hash": transparent_vs_hash,
            "host_draw_index": int(host_draw_index),
            "host_ib_hash": host_key[0],
            "host_match_first_index": int(host_key[1]),
            "host_match_index_count": int(host_key[2]),
        },
        "candidate_ibs": candidates,
        "vertex_layout_table": vertex_layout_table,
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


def _all_ib_txt_dumps(files_by_draw: dict[int, list[DumpFile]]) -> list[DumpFile]:
    dumps = [
        dump_file
        for dump_files in files_by_draw.values()
        for dump_file in dump_files
        if dump_file.slot == "ib" and dump_file.extension == "txt"
    ]
    dumps.sort(key=lambda item: item.draw_index)
    return dumps


def _group_ib_dumps_by_key(dump_files: Iterable[DumpFile]) -> dict[tuple[str, int, int], list[DumpFile]]:
    grouped: dict[tuple[str, int, int], list[DumpFile]] = {}
    for dump_file in dump_files:
        key = _candidate_key_from_ib_dump(dump_file)
        if not key[0] or key[2] <= 0:
            continue
        grouped.setdefault(key, []).append(dump_file)
    for hits in grouped.values():
        hits.sort(key=lambda item: item.draw_index)
    return grouped


def _draw_in_stage_window(draw_index: int, stage_draw_start: int, stage_draw_end: int) -> bool:
    if stage_draw_start <= 0 or stage_draw_end <= 0:
        return True
    return int(stage_draw_start) <= int(draw_index) <= int(stage_draw_end)


def _expand_shadow_capture_vs_hashes(
    *,
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
    all_ib_dumps: list[DumpFile],
    key_index: dict[tuple[str, int, int], list[DumpFile]],
    seed_vs_hashes: list[str],
    stage_draw_start: int,
    stage_draw_end: int,
) -> list[str]:
    """Return every VS that can capture bones inside the inferred early shadow window."""

    expanded: list[str] = []
    seen: set[str] = set()

    def add(vs_hash: str) -> None:
        normalized = str(vs_hash or "").strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            expanded.append(normalized)

    for vs_hash in seed_vs_hashes:
        add(vs_hash)

    candidate_local_bone_cache: dict[tuple[str, int, int], int] = {}
    for dump_file in sorted(all_ib_dumps, key=lambda item: item.draw_index):
        if not _draw_in_stage_window(dump_file.draw_index, stage_draw_start, stage_draw_end):
            continue
        draw_state = draw_states.get(dump_file.draw_index)
        if draw_state is not None and int(draw_state.rt_count) != 0:
            continue
        draw_files = files_by_draw.get(dump_file.draw_index, [])
        if not _draw_has_vs_t0(draw_files):
            continue
        key = _candidate_key_from_ib_dump(dump_file)
        if key[2] <= 0:
            continue
        local_bone_count = candidate_local_bone_cache.get(key)
        if local_bone_count is None:
            local_warnings: list[dict] = []
            local_bone_count = _infer_candidate_local_bone_count(
                files_by_draw,
                key_index.get(key, []),
                local_warnings,
            )
            candidate_local_bone_cache[key] = int(local_bone_count)
        if int(local_bone_count) <= 0:
            continue
        add(_state_vs_hash(draw_states, dump_file))
    return expanded


def _draw_has_vs_t0(draw_files: list[DumpFile]) -> bool:
    return any(dump_file.slot == "vs-t0" and dump_file.extension == "buf" for dump_file in draw_files)


def _discover_visible_anchor_keys(
    all_ib_dumps: list[DumpFile],
    draw_states: dict[int, DrawState],
    target_hashes: list[str],
    warnings: list[dict],
) -> dict:
    if target_hashes:
        manual_hits = [
            dump_file
            for dump_file in all_ib_dumps
            if dump_file.resource_hash.lower() in target_hashes
        ]
        if manual_hits:
            return {
                "selection_mode": "manual_hash_anchor",
                "anchor_keys": sorted(
                    {_candidate_key_from_ib_dump(dump_file) for dump_file in manual_hits},
                    key=lambda item: (-item[2], item[0], item[1]),
                ),
                "anchor_draw_indices": [dump_file.draw_index for dump_file in manual_hits[:32]],
                "max_rt_count": max((_state_rt_count(draw_states, dump_file) for dump_file in manual_hits), default=-1),
            }
        warnings.append(
            {
                "severity": "warning",
                "code": "manual_anchor_missing",
                "message": f"Manual anchor IB hash(es) were not found: {', '.join(target_hashes)}.",
                "draw_indices": [],
            }
        )

    rt_counts = [
        _state_rt_count(draw_states, dump_file)
        for dump_file in all_ib_dumps
        if _state_rt_count(draw_states, dump_file) > 0
    ]
    max_rt_count = max(rt_counts, default=-1)
    if max_rt_count > 0:
        anchor_hits = [
            dump_file
            for dump_file in all_ib_dumps
            if _state_rt_count(draw_states, dump_file) == max_rt_count
        ]
    else:
        anchor_hits = []
    if not anchor_hits:
        anchor_hits = [
            dump_file
            for dump_file in all_ib_dumps
            if _state_rt_count(draw_states, dump_file) > 0
        ]
    if not anchor_hits:
        anchor_hits = list(all_ib_dumps)
        warnings.append(
            {
                "severity": "warning",
                "code": "gbuffer_anchor_missing",
                "message": "No visible/g-buffer IB draws with RT output were found; using all dumped IBs as weak anchors.",
                "draw_indices": [],
            }
        )
    return {
        "selection_mode": "auto_gbuffer_anchor" if max_rt_count > 0 else "auto_any_ib_anchor",
        "anchor_keys": sorted(
            {_candidate_key_from_ib_dump(dump_file) for dump_file in anchor_hits},
            key=lambda item: (-item[2], item[0], item[1]),
        ),
        "anchor_draw_indices": [dump_file.draw_index for dump_file in anchor_hits[:64]],
        "max_rt_count": int(max_rt_count),
    }


def _discover_shadow_stage_from_visible_anchors(
    *,
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
    all_ib_dumps: list[DumpFile],
    key_index: dict[tuple[str, int, int], list[DumpFile]],
    visible_anchor: dict,
    warnings: list[dict],
) -> dict:
    anchor_keys = set(visible_anchor.get("anchor_keys", []) or [])
    first_anchor_draw_by_key: dict[tuple[str, int, int], int] = {}
    for key in anchor_keys:
        anchor_draws = [
            dump_file.draw_index
            for dump_file in key_index.get(key, [])
            if _state_rt_count(draw_states, dump_file) > 0
        ]
        if anchor_draws:
            first_anchor_draw_by_key[key] = min(anchor_draws)

    backtrack_hits = [
        dump_file
        for key, first_draw in first_anchor_draw_by_key.items()
        for dump_file in key_index.get(key, [])
        if dump_file.draw_index < first_draw
    ]
    selected_pair = _best_shadow_vs_pair_from_hits(backtrack_hits, files_by_draw, draw_states, key_index)
    if selected_pair:
        return {
            **selected_pair,
            "selection_mode": f"{visible_anchor.get('selection_mode', 'auto_anchor')}_backtrack_shadow_vs",
            "anchor_draw_indices": list(visible_anchor.get("anchor_draw_indices", []) or []),
            "anchor_keys": [f"{key[0]}-{key[2]}-{key[1]}" for key in sorted(anchor_keys)],
        }

    auto_candidates = _auto_shadow_vs_pair_candidates(files_by_draw, draw_states, warnings)
    if not auto_candidates:
        raise ValueError("No skinned shadow VS candidate pair found in FrameAnalysis")
    selected_candidate = auto_candidates[0]
    warnings.append(
        {
            "severity": "warning",
            "code": "shadow_vs_pair_fallback",
            "message": "Could not backtrack shadow VS from visible anchors; used structural shadow-pair fallback.",
            "draw_indices": [],
        }
    )
    return {
        "selection_mode": "auto_shadow_vs_pair_fallback",
        "shadow_vs_hashes": list(selected_candidate["shadow_vs_hashes"]),
        "stage_draw_start": int(selected_candidate.get("draw_start", 0)),
        "stage_draw_end": int(selected_candidate.get("draw_end", 0)),
        "anchor_draw_indices": list(visible_anchor.get("anchor_draw_indices", []) or []),
        "anchor_keys": [f"{key[0]}-{key[2]}-{key[1]}" for key in sorted(anchor_keys)],
        "auto_candidates": auto_candidates[:8],
    }


def _best_shadow_vs_pair_from_hits(
    hits: list[DumpFile],
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
    key_index: dict[tuple[str, int, int], list[DumpFile]],
) -> dict | None:
    hits_by_vs: dict[str, list[DumpFile]] = {}
    for dump_file in hits:
        vs_hash = _state_vs_hash(draw_states, dump_file)
        if not vs_hash:
            continue
        hits_by_vs.setdefault(vs_hash, []).append(dump_file)
    if not hits_by_vs:
        return None
    vs_infos = {
        vs_hash: _build_vs_group_info(vs_hash, vs_hits, files_by_draw, draw_states, key_index)
        for vs_hash, vs_hits in hits_by_vs.items()
    }
    pairs: list[dict] = []
    vs_hashes = sorted(vs_infos, key=lambda value: vs_infos[value]["draw_start"])
    for left_index, left_vs in enumerate(vs_hashes):
        for right_vs in vs_hashes[left_index + 1 :]:
            left_info = vs_infos[left_vs]
            right_info = vs_infos[right_vs]
            left_start = int(left_info["draw_start"])
            right_start = int(right_info["draw_start"])
            if abs(left_start - right_start) > 260:
                continue
            union_keys = left_info["candidate_keys"] | right_info["candidate_keys"]
            skinned_keys = left_info["skinned_keys"] | right_info["skinned_keys"]
            overlap_keys = left_info["candidate_keys"] & right_info["candidate_keys"]
            if not skinned_keys:
                continue
            score = (
                len(skinned_keys) * 100000
                + len(overlap_keys) * 50000
                + min(sum(key[2] for key in union_keys), 800000)
                + max(0, 260 - min(left_start, right_start)) * 2000
            )
            pairs.append(
                {
                    "shadow_vs_hashes": [left_vs, right_vs],
                    "score": int(score),
                    "stage_draw_start": min(left_info["draw_start"], right_info["draw_start"]),
                    "stage_draw_end": max(left_info["draw_end"], right_info["draw_end"]),
                    "candidate_count": int(len(union_keys)),
                    "skinned_candidate_count": int(len(skinned_keys)),
                    "overlap_count": int(len(overlap_keys)),
                }
            )
    if pairs:
        pairs.sort(
            key=lambda item: (
                int(item["score"]),
                int(item["skinned_candidate_count"]),
                int(item["overlap_count"]),
                -int(item["stage_draw_start"]),
            ),
            reverse=True,
        )
        return {**pairs[0], "auto_candidates": pairs[:8]}

    single_infos = sorted(
        vs_infos.values(),
        key=lambda item: (len(item["skinned_keys"]), int(item["index_count_sum"]), -int(item["draw_start"])),
        reverse=True,
    )
    if single_infos and single_infos[0]["skinned_keys"]:
        info = single_infos[0]
        return {
            "shadow_vs_hashes": [str(info["vs_hash"])],
            "score": int(info["index_count_sum"]),
            "stage_draw_start": int(info["draw_start"]),
            "stage_draw_end": int(info["draw_end"]),
            "candidate_count": int(len(info["candidate_keys"])),
            "skinned_candidate_count": int(len(info["skinned_keys"])),
            "overlap_count": 0,
            "auto_candidates": [],
        }
    return None


def _build_candidate_entries(
    *,
    frameanalysis_dir: str,
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
    key_index: dict[tuple[str, int, int], list[DumpFile]],
    shadow_hits_by_key: dict[tuple[str, int, int], list[DumpFile]],
    shadow_vs_hashes: list[str],
    warnings: list[dict],
) -> list[dict]:
    candidates: list[dict] = []
    skipped_no_bones = 0
    skipped_no_import = 0
    for key, all_hits in sorted(key_index.items(), key=lambda item: (-item[0][2], item[0][0], item[0][1])):
        import_hit = _select_import_hit(files_by_draw, draw_states, all_hits)
        if import_hit is None:
            skipped_no_import += 1
            continue
        ordered_hits = [import_hit] + [hit for hit in all_hits if hit.draw_index != import_hit.draw_index]
        local_warnings: list[dict] = []
        source_local_bone_count = _infer_candidate_local_bone_count(files_by_draw, ordered_hits, local_warnings)
        if source_local_bone_count <= 0:
            skipped_no_bones += 1
            continue
        ib_hash, first_index, index_count = key
        key_shadow_hits = [
            dump_file
            for dump_file in shadow_hits_by_key.get(key, [])
            if _state_vs_hash(draw_states, dump_file) in shadow_vs_hashes
        ]
        import_state = draw_states.get(import_hit.draw_index)
        position_stream = _build_position_stream_payload(files_by_draw, import_hit)
        lod_match_excluded = _is_dynamic_vb0_position_stream(position_stream)
        lod_match_excluded_reason = (
            "dynamic_vb0_backing_hash_mismatch" if lod_match_excluded else ""
        )
        manifest_hits = [import_hit] + [
            dump_file
            for dump_file in key_shadow_hits
            if dump_file.draw_index != import_hit.draw_index
        ]
        candidate = {
            "enabled": True,
            "ib_hash": ib_hash,
            "match_first_index": int(first_index),
            "match_index_count": int(index_count),
            "display_name": f"{ib_hash}-{index_count}-{first_index}",
            "draw_indices": [dump_file.draw_index for dump_file in all_hits],
            "import_draw_index": int(import_hit.draw_index),
            "import_vs_hash": str(import_state.vs_hash if import_state else import_hit.vs_hash),
            "import_ps_hash": str(import_state.ps_hash if import_state else import_hit.ps_hash),
            "shadow_capture_ready": bool(key_shadow_hits),
            "shadow_draw_indices": [dump_file.draw_index for dump_file in key_shadow_hits],
            "source_index_count": int(index_count),
            "source_local_bone_count": int(source_local_bone_count),
            "local_bone_count": int(source_local_bone_count),
            "texture_region_key": f"{ib_hash}-{index_count}-{first_index}",
            "status": _candidate_runtime_status(bool(key_shadow_hits), lod_match_excluded),
            "import_paths": _build_import_paths_payload(files_by_draw, ordered_hits),
            "skin_format": _build_skin_format_payload(files_by_draw, import_hit),
            "position_stream": position_stream,
            "dynamic_vb0": lod_match_excluded,
            "lod_match_excluded": lod_match_excluded,
            "lod_match_excluded_reason": lod_match_excluded_reason,
            "manifest_hits": manifest_hits,
        }
        used_local_bone_indices = _infer_candidate_used_local_bone_indices(
            files_by_draw,
            import_hit,
            candidate,
            warnings,
        )
        if not used_local_bone_indices:
            skipped_no_bones += 1
            continue
        candidate["used_local_bone_indices"] = used_local_bone_indices
        candidate["local_bone_count"] = len(used_local_bone_indices)
        used_source_local_bone_count = max(used_local_bone_indices) + 1
        candidate["source_local_bone_count"] = max(
            int(source_local_bone_count) if 0 < int(source_local_bone_count) <= 256 else 0,
            used_source_local_bone_count,
        )
        candidates.append(candidate)
    if skipped_no_import:
        warnings.append(
            {
                "severity": "warning",
                "code": "skipped_no_import_buffers",
                "message": f"Skipped {skipped_no_import} IB candidate(s) without enough dumped IB/VB data.",
                "draw_indices": [],
            }
        )
    if skipped_no_bones:
        warnings.append(
            {
                "severity": "info",
                "code": "skipped_no_blendindices",
                "message": f"Skipped {skipped_no_bones} non-skinned or unsupported IB candidate(s).",
                "draw_indices": [],
            }
        )
    return candidates


def _build_position_stream_payload(files_by_draw: dict[int, list[DumpFile]], import_hit: DumpFile) -> dict:
    draw_files = files_by_draw.get(import_hit.draw_index, [])
    vb0_file = _first_slot_file(draw_files, "vb0", "txt")
    return {
        "ib_hash": import_hit.resource_hash,
        "ib_backing_hash": import_hit.backing_hash,
        "vb0_hash": vb0_file.resource_hash if vb0_file else "",
        "vb0_backing_hash": vb0_file.backing_hash if vb0_file else "",
    }


def _is_dynamic_vb0_position_stream(position_stream: dict) -> bool:
    ib_backing_hash = str(position_stream.get("ib_backing_hash", "") or "").lower()
    vb0_backing_hash = str(position_stream.get("vb0_backing_hash", "") or "").lower()
    return bool(ib_backing_hash and vb0_backing_hash and ib_backing_hash != vb0_backing_hash)


def _candidate_runtime_status(shadow_capture_ready: bool, lod_match_excluded: bool) -> str:
    if lod_match_excluded:
        return "capture_ready_dynamic_vb0_lod_excluded" if shadow_capture_ready else "import_only_dynamic_vb0_lod_excluded"
    return "capture_ready" if shadow_capture_ready else "import_only_no_early_shadow"


def _select_import_hit(
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
    hits: list[DumpFile],
) -> DumpFile | None:
    importable_hits = [
        dump_file
        for dump_file in hits
        if _draw_has_minimum_import_buffers(files_by_draw.get(dump_file.draw_index, []))
    ]
    if not importable_hits:
        return None
    return max(importable_hits, key=lambda dump_file: _import_hit_score(files_by_draw, draw_states, dump_file))


def _draw_has_minimum_import_buffers(draw_files: list[DumpFile]) -> bool:
    if not _binary_data_path_for_slot(draw_files, "ib"):
        return False
    has_vb0_txt = any(dump_file.slot == "vb0" and dump_file.extension == "txt" for dump_file in draw_files)
    has_vb0_buf = bool(_binary_data_path_for_slot(draw_files, "vb0"))
    return has_vb0_txt and has_vb0_buf


def _import_hit_score(
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
    dump_file: DumpFile,
) -> tuple[int, int, int, int, int, int]:
    draw_files = files_by_draw.get(dump_file.draw_index, [])
    draw_state = draw_states.get(dump_file.draw_index)
    rt_count = int(draw_state.rt_count if draw_state else -1)
    vb_slot_count = len({item.slot for item in draw_files if item.slot.startswith("vb") and item.extension == "txt"})
    has_vb1 = any(item.slot == "vb1" and item.extension == "txt" for item in draw_files)
    has_vb2 = any(item.slot == "vb2" and item.extension == "txt" for item in draw_files)
    has_ps = 1 if draw_state and draw_state.ps_hash else 0
    return (
        1 if rt_count > 0 else 0,
        min(max(rt_count, 0), 8),
        1 if has_vb2 else 0,
        1 if has_vb1 else 0,
        vb_slot_count + has_ps,
        int(dump_file.draw_index),
    )


def _state_rt_count(draw_states: dict[int, DrawState], dump_file: DumpFile) -> int:
    draw_state = draw_states.get(dump_file.draw_index)
    return int(draw_state.rt_count if draw_state else -1)


def _pass_role_for_material(draw_state: DrawState | None) -> str:
    if draw_state and int(draw_state.rt_count) > 0:
        return "visible_material"
    return "unknown"


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


def _discover_shadow_vs_hashes(
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
    target_hashes: list[str],
    warnings: list[dict],
) -> dict:
    if target_hashes:
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
        return {
            "selection_mode": "manual_ib_hash_seed",
            "shadow_vs_hashes": shadow_vs_hashes,
            "seed_ib_hashes": target_hashes,
            "seed_draw_indices": [dump_file.draw_index for dump_file in target_hits[:16]],
            "auto_candidates": [],
        }

    auto_candidates = _auto_shadow_vs_pair_candidates(files_by_draw, draw_states, warnings)
    if not auto_candidates:
        raise ValueError("No skinned shadow VS candidate pair found in FrameAnalysis")
    selected_candidate = auto_candidates[0]
    return {
        "selection_mode": "auto_shadow_vs_pair",
        "shadow_vs_hashes": list(selected_candidate["shadow_vs_hashes"]),
        "seed_ib_hashes": [],
        "seed_draw_indices": [],
        "auto_candidates": auto_candidates[:8],
    }


def _auto_shadow_vs_pair_candidates(
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
    warnings: list[dict],
) -> list[dict]:
    hits_by_vs: dict[str, list[DumpFile]] = {}
    all_ib_dumps: list[DumpFile] = []
    for dump_files in files_by_draw.values():
        for dump_file in dump_files:
            if dump_file.slot != "ib" or dump_file.extension != "txt":
                continue
            key = _candidate_key_from_ib_dump(dump_file)
            if key[2] <= 0:
                continue
            all_ib_dumps.append(dump_file)
            vs_hash = _state_vs_hash(draw_states, dump_file)
            if not vs_hash:
                continue
            hits_by_vs.setdefault(vs_hash, []).append(dump_file)

    key_index: dict[tuple[str, int, int], list[DumpFile]] = {}
    for dump_file in all_ib_dumps:
        key_index.setdefault(_candidate_key_from_ib_dump(dump_file), []).append(dump_file)

    vs_infos = {
        vs_hash: _build_vs_group_info(vs_hash, hits, files_by_draw, draw_states, key_index)
        for vs_hash, hits in hits_by_vs.items()
    }
    pairs: list[dict] = []
    vs_hashes = sorted(vs_infos, key=lambda value: vs_infos[value]["draw_start"])
    for left_index, left_vs in enumerate(vs_hashes):
        for right_vs in vs_hashes[left_index + 1 :]:
            left_info = vs_infos[left_vs]
            right_info = vs_infos[right_vs]
            if not _could_be_shadow_pair(left_info, right_info):
                continue
            union_keys = left_info["candidate_keys"] | right_info["candidate_keys"]
            skinned_keys = left_info["skinned_keys"] | right_info["skinned_keys"]
            if len(skinned_keys) < 3:
                continue
            overlap_keys = left_info["candidate_keys"] & right_info["candidate_keys"]
            shared_index_count_sum = sum(key[2] for key in overlap_keys)
            index_count_sum = sum(key[2] for key in union_keys)
            draw_start = min(left_info["draw_start"], right_info["draw_start"])
            draw_end = max(left_info["draw_end"], right_info["draw_end"])
            rt0_count = int(left_info["rt_counts"].get(0, 0)) + int(right_info["rt_counts"].get(0, 0))
            score = (
                len(skinned_keys) * 100000
                + min(index_count_sum, 800000)
                + len(overlap_keys) * 10000
                + min(shared_index_count_sum, 300000)
                + rt0_count * 200000
                + max(0, 260 - draw_start) * 5000
                - draw_start * 1000
                - max(0, draw_end - draw_start - 260) * 5000
            )
            pairs.append(
                {
                    "shadow_vs_hashes": [left_vs, right_vs],
                    "score": int(score),
                    "candidate_count": int(len(union_keys)),
                    "skinned_candidate_count": int(len(skinned_keys)),
                    "overlap_count": int(len(overlap_keys)),
                    "index_count_sum": int(index_count_sum),
                    "shared_index_count_sum": int(shared_index_count_sum),
                    "rt0_draw_count": int(rt0_count),
                    "draw_start": int(draw_start),
                    "draw_end": int(draw_end),
                    "rt_counts": {
                        left_vs: dict(left_info["rt_counts"]),
                        right_vs: dict(right_info["rt_counts"]),
                    },
                }
            )
    pairs.sort(
        key=lambda item: (
            int(item["score"]),
            int(item["skinned_candidate_count"]),
            int(item["index_count_sum"]),
            -int(item["draw_start"]),
        ),
        reverse=True,
    )
    if len(pairs) > 1 and int(pairs[0]["score"]) - int(pairs[1]["score"]) < 50000:
        warnings.append(
            {
                "severity": "warning",
                "code": "ambiguous_auto_shadow_vs_pair",
                "message": "Auto Analyze found multiple plausible shadow VS pairs; using the highest score.",
                "top_pairs": pairs[:3],
            }
        )
    return pairs


def _build_vs_group_info(
    vs_hash: str,
    hits: list[DumpFile],
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
    key_index: dict[tuple[str, int, int], list[DumpFile]],
) -> dict:
    candidate_keys = {_candidate_key_from_ib_dump(dump_file) for dump_file in hits}
    skinned_keys: set[tuple[str, int, int]] = set()
    local_bone_counts: dict[str, int] = {}
    for key in candidate_keys:
        local_warnings: list[dict] = []
        local_bone_count = _infer_candidate_local_bone_count(files_by_draw, key_index.get(key, []), local_warnings)
        if local_bone_count > 0:
            skinned_keys.add(key)
        local_bone_counts[f"{key[0]}-{key[2]}-{key[1]}"] = int(local_bone_count)
    draw_indices = sorted({dump_file.draw_index for dump_file in hits})
    rt_counter: dict[int, int] = {}
    for dump_file in hits:
        draw_state = draw_states.get(dump_file.draw_index)
        rt_count = int(draw_state.rt_count if draw_state else -1)
        rt_counter[rt_count] = rt_counter.get(rt_count, 0) + 1
    return {
        "vs_hash": vs_hash,
        "candidate_keys": candidate_keys,
        "skinned_keys": skinned_keys,
        "local_bone_counts": local_bone_counts,
        "draw_start": min(draw_indices) if draw_indices else 0,
        "draw_end": max(draw_indices) if draw_indices else 0,
        "draw_count": len(draw_indices),
        "index_count_sum": sum(key[2] for key in candidate_keys),
        "rt_counts": rt_counter,
    }


def _could_be_shadow_pair(left_info: dict, right_info: dict) -> bool:
    left_start = int(left_info["draw_start"])
    right_start = int(right_info["draw_start"])
    if abs(left_start - right_start) > 260:
        return False
    if int(left_info["draw_count"]) < 3 or int(right_info["draw_count"]) < 3:
        return False
    if not (left_info["skinned_keys"] or right_info["skinned_keys"]):
        return False
    if not (left_info["candidate_keys"] & right_info["candidate_keys"]):
        return False
    return True


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


@lru_cache(maxsize=8192)
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
        header_path = dump_file.data_path if dump_file.data_path and dump_file.data_path.endswith(".txt") else dump_file.path
        header = _parse_buffer_header(header_path)
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
    vb_slots = sorted({dump_file.slot for dump_file in first_draw_files if dump_file.slot.startswith("vb")})
    return {
        "ib": first_hit.path,
        "ib_buf": _binary_data_path_for_slot(first_draw_files, "ib"),
        "ib_hash": first_hit.resource_hash,
        "ib_backing_hash": first_hit.backing_hash,
        "vb": {
            slot: {
                "hash": next(
                    (
                        dump_file.resource_hash
                        for dump_file in first_draw_files
                        if dump_file.slot == slot and dump_file.extension == "txt"
                    ),
                    "",
                ),
                "backing_hash": next(
                    (
                        dump_file.backing_hash
                        for dump_file in first_draw_files
                        if dump_file.slot == slot and dump_file.extension == "txt"
                    ),
                    "",
                ),
                "txt": [
                    dump_file.path
                    for dump_file in first_draw_files
                    if dump_file.slot == slot and dump_file.extension == "txt"
                ],
                "layout_txt": [
                    dump_file.data_path
                    for dump_file in first_draw_files
                    if dump_file.slot == slot
                    and dump_file.extension == "txt"
                    and dump_file.data_path
                    and dump_file.data_path != dump_file.path
                ],
                "buf": next(
                    (
                        dump_file.data_path
                        for dump_file in first_draw_files
                        if dump_file.slot == slot and dump_file.extension == "buf" and dump_file.data_path
                    ),
                    "",
                ),
            }
            for slot in vb_slots
        },
        "layout": "",
    }


def _candidate_source_key(candidate: dict) -> str:
    return (
        f"{str(candidate.get('ib_hash', '') or '').lower()}-"
        f"{int(candidate.get('match_index_count', 0) or 0)}-"
        f"{int(candidate.get('match_first_index', 0) or 0)}"
    )


def _build_vertex_layout_table(frameanalysis_dir: str, candidates: list[dict]) -> dict:
    table: dict[str, dict] = {}
    for candidate in candidates:
        key = str(candidate.get("vertex_layout_key", "") or _candidate_source_key(candidate))
        table[key] = _build_candidate_vertex_layout(frameanalysis_dir, candidate)
    return table


def _build_candidate_vertex_layout(frameanalysis_dir: str, candidate: dict) -> dict:
    import_paths = dict(candidate.get("import_paths", {}) or {})
    layout = {
        "source_key": str(candidate.get("vertex_layout_key", "") or _candidate_source_key(candidate)),
        "ib_hash": str(candidate.get("ib_hash", "") or "").lower(),
        "import_vs_hash": str(candidate.get("import_vs_hash", "") or "").lower(),
        "match_first_index": int(candidate.get("match_first_index", 0) or 0),
        "match_index_count": int(candidate.get("match_index_count", 0) or 0),
        "ib": _build_index_layout(frameanalysis_dir, import_paths),
        "vertex_buffers": {},
    }
    vb_payload = dict(import_paths.get("vb", {}) or {})
    for slot_name, slot_payload in sorted(vb_payload.items(), key=lambda item: _slot_index(str(item[0]))):
        slot_index = _slot_index(str(slot_name))
        if slot_index < 0:
            continue
        slot_layout = _build_vertex_slot_layout(frameanalysis_dir, str(slot_name), slot_index, dict(slot_payload or {}))
        if slot_layout:
            layout["vertex_buffers"][str(slot_name)] = slot_layout
    return annotate_vertex_layout(layout, str(candidate.get("import_vs_hash", "") or ""))


def _build_index_layout(frameanalysis_dir: str, import_paths: dict) -> dict:
    ib_path = _resolve_manifest_path(str(import_paths.get("ib", "") or ""), frameanalysis_dir)
    if not ib_path or not os.path.exists(ib_path):
        return {}
    header = _parse_buffer_header(ib_path)
    return {
        "format": str(header.fmt or ""),
        "byte_offset": int(header.byte_offset),
        "first_index": int(header.first_index),
        "index_count": int(header.index_count),
        "topology": str(header.topology or ""),
        "source_txt": _relpath(ib_path, frameanalysis_dir),
        "source_buf": _relpath(str(import_paths.get("ib_buf", "") or ""), frameanalysis_dir),
    }


def _build_vertex_slot_layout(frameanalysis_dir: str, slot_name: str, slot_index: int, slot_payload: dict) -> dict:
    header_paths = _layout_header_paths(frameanalysis_dir, slot_payload)
    if not header_paths:
        return {}
    header = _first_valid_layout_header(header_paths)
    if int(header.stride) <= 0 or int(header.vertex_count) <= 0:
        return {}
    fields = [
        element.to_dict()
        for element in header.elements
        if int(element.input_slot) == int(slot_index)
    ]
    return {
        "slot": slot_name,
        "slot_index": int(slot_index),
        "resource_hash": str(slot_payload.get("hash", "") or "").lower(),
        "backing_hash": str(slot_payload.get("backing_hash", "") or "").lower(),
        "stride": int(header.stride),
        "vertex_count": int(header.vertex_count),
        "byte_offset": int(header.byte_offset),
        "source_txt": [_relpath(path, frameanalysis_dir) for path in _resolve_manifest_paths(slot_payload.get("txt", []), frameanalysis_dir)],
        "source_layout_txt": [_relpath(path, frameanalysis_dir) for path in header_paths],
        "source_buf": _relpath(str(slot_payload.get("buf", "") or ""), frameanalysis_dir),
        "fields": fields,
    }


def _layout_header_paths(frameanalysis_dir: str, slot_payload: dict) -> list[str]:
    return [
        path
        for path in _resolve_manifest_paths(
            list(slot_payload.get("layout_txt", []) or []) + list(slot_payload.get("txt", []) or []),
            frameanalysis_dir,
        )
        if os.path.exists(path)
    ]


def _resolve_manifest_paths(paths, frameanalysis_dir: str) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_path in paths or []:
        path = _resolve_manifest_path(str(raw_path or ""), frameanalysis_dir)
        if not path or path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def _resolve_manifest_path(path: str, frameanalysis_dir: str) -> str:
    if not path:
        return ""
    return os.path.abspath(path) if os.path.isabs(path) else os.path.abspath(os.path.join(frameanalysis_dir, path))


def _first_valid_layout_header(paths: list[str]) -> BufferHeader:
    for path in paths:
        header = _parse_buffer_header(path)
        if int(header.stride) > 0 and int(header.vertex_count) > 0:
            return header
    return _parse_buffer_header(paths[0])


def _build_skin_format_payload(files_by_draw: dict[int, list[DumpFile]], import_hit: DumpFile) -> dict:
    for dump_file in files_by_draw.get(import_hit.draw_index, []):
        if not dump_file.slot.startswith("vb") or dump_file.extension != "txt":
            continue
        header_path = dump_file.data_path if dump_file.data_path and dump_file.data_path.endswith(".txt") else dump_file.path
        header = _parse_buffer_header(header_path)
        slot_index = _slot_index(dump_file.slot)
        blend_indices = next(
            (
                element
                for element in header.elements
                if element.input_slot == slot_index and element.semantic_name == "BLENDINDICES"
            ),
            None,
        )
        if blend_indices is None:
            continue
        blend_weights = next(
            (
                element
                for element in header.elements
                if element.input_slot == slot_index and element.semantic_name == "BLENDWEIGHTS"
            ),
            None,
        )
        return {
            "slot": dump_file.slot,
            "slot_index": int(slot_index),
            "stride": int(header.stride),
            "vertex_count": int(header.vertex_count),
            "blend_weights_format": str(blend_weights.fmt if blend_weights else ""),
            "blend_weights_offset": int(blend_weights.aligned_byte_offset if blend_weights else -1),
            "blend_indices_format": str(blend_indices.fmt),
            "blend_indices_offset": int(blend_indices.aligned_byte_offset),
            "source_txt": dump_file.path,
            "source_layout_txt": header_path,
        }
    return {}


def _infer_candidate_used_local_bone_indices(
    files_by_draw: dict[int, list[DumpFile]],
    import_hit: DumpFile,
    candidate: dict,
    warnings: list[dict],
) -> list[int]:
    try:
        used_indices = _read_used_local_bone_indices_from_import_draw(
            files_by_draw.get(import_hit.draw_index, []),
            import_hit,
            dict(candidate.get("skin_format", {}) or {}),
        )
    except Exception as exc:
        warnings.append(
            {
                "severity": "warning",
                "code": "compact_palette_scan_failed",
                "message": (
                    f"Could not compact local bone palette for "
                    f"{candidate.get('display_name') or candidate.get('ib_hash')}: {exc}"
                ),
                "draw_indices": [int(candidate.get("import_draw_index", -1) or -1)],
            }
        )
        return []

    if used_indices:
        return sorted({int(value) for value in used_indices if int(value) >= 0})
    return []


def _read_used_local_bone_indices_from_import_draw(
    draw_files: list[DumpFile],
    import_hit: DumpFile,
    skin_format: dict,
) -> list[int]:
    ib_buf_path = _binary_data_path_for_slot(draw_files, "ib")
    if not ib_buf_path:
        return []
    ib_header_path = import_hit.data_path if import_hit.data_path and import_hit.data_path.endswith(".txt") else import_hit.path
    ib_header = _parse_buffer_header(ib_header_path)
    indices = read_index_array(
        ib_buf_path,
        str(ib_header.fmt or ""),
        int(ib_header.index_count),
        byte_offset=int(ib_header.byte_offset),
        first_index=int(ib_header.first_index),
    )
    np = require_numpy()
    vertex_id_array = np.unique(indices[indices >= 0]) if indices.size else np.zeros((0,), dtype=np.int64)
    vertex_ids = [int(value) for value in vertex_id_array.tolist()]
    if not vertex_ids:
        return []

    slot_name = str(skin_format.get("slot", "") or "vb2").lower()
    slot_index = _slot_index(slot_name)
    skin_txt = next(
        (
            dump_file
            for dump_file in draw_files
            if dump_file.slot == slot_name and dump_file.extension == "txt"
        ),
        None,
    )
    if skin_txt is None:
        return []
    header_path = skin_txt.data_path if skin_txt.data_path and skin_txt.data_path.endswith(".txt") else skin_txt.path
    header = _parse_buffer_header(header_path)
    weight_element = next(
        (
            element
            for element in header.elements
            if element.input_slot == slot_index and element.semantic_name == "BLENDWEIGHTS"
        ),
        None,
    )
    index_element = next(
        (
            element
            for element in header.elements
            if element.input_slot == slot_index and element.semantic_name == "BLENDINDICES"
        ),
        None,
    )
    if weight_element is None or index_element is None:
        return []
    data_path = _binary_data_path_for_slot(draw_files, slot_name)
    if not data_path:
        return []
    base_offset = _infer_buffer_data_base_offset(
        data_path=data_path,
        header_path=header_path,
        header=header,
        slot=slot_name,
        slot_index=slot_index,
    )
    weight_size = _vertex_format_size(weight_element.fmt)
    index_size = _vertex_format_size(index_element.fmt)
    if weight_size <= 0 or index_size <= 0 or header.stride <= 0:
        return []
    numpy_fields = read_interleaved_fields_from_file(
        data_path,
        vertex_ids,
        byte_offset=base_offset,
        vertex_count=int(header.vertex_count),
        stride=int(header.stride),
        fields=[
            ("weights", int(weight_element.aligned_byte_offset), str(weight_element.fmt)),
            ("indices", int(index_element.aligned_byte_offset), str(index_element.fmt)),
        ],
        converted=True,
    )
    if numpy_fields is None:
        raise ValueError(f"{slot_name}: failed to read compact palette skin fields with numpy")
    return used_skin_slots(numpy_fields["indices"], numpy_fields["weights"], epsilon=0.0)


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
            header_path = dump_file.data_path if dump_file.data_path and dump_file.data_path.endswith(".txt") else dump_file.path
            header = _parse_buffer_header(header_path)
            slot_index = _slot_index(dump_file.slot)
            blend_index_element = next(
                (
                    element
                    for element in header.elements
                    if element.input_slot == slot_index and element.semantic_name == "BLENDINDICES"
                ),
                None,
            )
            if blend_index_element is None or not _is_supported_blend_index_format(blend_index_element.fmt):
                continue
            binary_data_path = _binary_data_path_for_slot(draw_files, dump_file.slot)
            base_offset = _infer_buffer_data_base_offset(
                data_path=binary_data_path,
                header_path=header_path,
                header=header,
                slot=dump_file.slot,
                slot_index=slot_index,
            )
            max_index = _read_max_blend_index(
                data_path=binary_data_path,
                byte_offset=base_offset,
                stride=header.stride,
                vertex_count=header.vertex_count,
                blend_offset=blend_index_element.aligned_byte_offset,
                blend_format=blend_index_element.fmt,
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


def _infer_buffer_data_base_offset(
    data_path: str,
    header_path: str,
    header: BufferHeader,
    slot: str,
    slot_index: int,
) -> int:
    header_offset = max(0, int(header.byte_offset))
    if not data_path or not os.path.exists(data_path):
        return header_offset
    required_size = header_offset + int(header.vertex_count) * int(header.stride)
    try:
        file_size = os.path.getsize(data_path)
    except OSError:
        return header_offset
    if required_size <= file_size:
        return header_offset
    slice_size = int(header.vertex_count) * int(header.stride)
    if slice_size <= file_size:
        return 0
    return header_offset


def _first_vertex_samples_from_path(header_path: str, slot: str, slot_index: int) -> dict[int, list[float]]:
    samples: dict[int, list[float]] = {}
    if not header_path or not os.path.exists(header_path) or not slot.startswith("vb"):
        return samples
    try:
        slot_number = int(slot[2:])
    except ValueError:
        return samples
    if slot_number != slot_index:
        return samples
    in_vertex_data = False
    with open(header_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if line == "vertex-data:":
                in_vertex_data = True
                continue
            if not in_vertex_data:
                continue
            match = _VERTEX_DATA_RE.match(line)
            if not match:
                continue
            if int(match.group("slot")) != slot_number:
                continue
            if int(match.group("vertex")) != 0:
                break
            samples[int(match.group("offset"))] = _parse_vertex_data_values(match.group("values"))
    return samples


def _parse_vertex_data_values(value_text: str) -> list[float]:
    values: list[float] = []
    for raw_value in value_text.split(","):
        value = raw_value.strip()
        if not value:
            continue
        try:
            values.append(float(value))
        except ValueError:
            pass
    return values


def _buffer_sample_matches(file_handle, byte_offset: int, element: HeaderElement, expected: list[float]) -> bool:
    if not expected:
        return False
    size = _vertex_format_size(element.fmt)
    if size <= 0:
        return False
    try:
        file_handle.seek(byte_offset)
        raw_data = file_handle.read(size)
    except OSError:
        return False
    if len(raw_data) < size:
        return False
    actual = _unpack_vertex_format(raw_data, element.fmt)
    if len(actual) < len(expected):
        return False
    return all(abs(float(actual[index]) - float(expected[index])) <= 1e-4 for index in range(len(expected)))


def _vertex_format_size(fmt: str) -> int:
    return dxgi_format_size(fmt)


def _header_format_is(fmt: str, name: str) -> bool:
    upper_fmt = str(fmt or "").upper()
    upper_name = str(name or "").upper()
    return upper_fmt == upper_name or upper_fmt == f"DXGI_FORMAT_{upper_name}"


def _is_supported_blend_index_format(fmt: str) -> bool:
    return _header_format_is(fmt, "R8G8B8A8_UINT") or _header_format_is(fmt, "R32G32B32A32_UINT")


def _unpack_vertex_format(raw_data: bytes, fmt: str) -> tuple[float, ...]:
    import struct

    upper_fmt = str(fmt or "").upper()
    if upper_fmt in {"R32_FLOAT", "DXGI_FORMAT_R32_FLOAT"}:
        return (float(struct.unpack_from("<f", raw_data, 0)[0]),)
    if upper_fmt in {"R32G32_FLOAT", "DXGI_FORMAT_R32G32_FLOAT"}:
        return tuple(float(value) for value in struct.unpack_from("<2f", raw_data, 0))
    if upper_fmt in {"R32G32B32_FLOAT", "DXGI_FORMAT_R32G32B32_FLOAT"}:
        return tuple(float(value) for value in struct.unpack_from("<3f", raw_data, 0))
    if upper_fmt in {"R32G32B32A32_FLOAT", "DXGI_FORMAT_R32G32B32A32_FLOAT"}:
        return tuple(float(value) for value in struct.unpack_from("<4f", raw_data, 0))
    if upper_fmt in {"R32G32B32A32_UINT", "DXGI_FORMAT_R32G32B32A32_UINT"}:
        return tuple(float(value) for value in struct.unpack_from("<4I", raw_data, 0))
    if upper_fmt in {"R16G16B16A16_UNORM", "DXGI_FORMAT_R16G16B16A16_UNORM"}:
        return tuple(float(value) / 65535.0 for value in struct.unpack_from("<4H", raw_data, 0))
    if upper_fmt in {"R8G8B8A8_UINT", "DXGI_FORMAT_R8G8B8A8_UINT"}:
        return tuple(float(value) for value in struct.unpack_from("<4B", raw_data, 0))
    if upper_fmt in {"R8G8B8A8_SNORM", "DXGI_FORMAT_R8G8B8A8_SNORM"}:
        return tuple(float(value - 256 if value >= 128 else value) / 127.0 for value in struct.unpack_from("<4B", raw_data, 0))
    return ()


def _binary_data_path_for_slot(draw_files: list[DumpFile], slot: str) -> str:
    for dump_file in draw_files:
        if dump_file.slot == slot and dump_file.extension == "buf" and dump_file.data_path:
            return dump_file.data_path
    return ""


@lru_cache(maxsize=2048)
def _read_max_blend_index(
    data_path: str,
    byte_offset: int,
    stride: int,
    vertex_count: int,
    blend_offset: int,
    blend_format: str,
) -> int:
    if not data_path or not os.path.exists(data_path) or stride <= 0 or vertex_count <= 0:
        return -1
    record_size = _vertex_format_size(blend_format)
    if record_size <= 0:
        return -1
    max_index = -1
    try:
        with open(data_path, "rb") as file_handle:
            file_handle.seek(byte_offset)
            raw_vertex_data = file_handle.read(vertex_count * stride)
    except OSError:
        return -1
    numpy_max = max_interleaved_uint4(
        raw_vertex_data,
        stride=int(stride),
        offset=int(blend_offset),
        fmt=str(blend_format),
        vertex_count=int(vertex_count),
    )
    if numpy_max >= 0:
        return int(numpy_max)
    for vertex_offset in range(blend_offset, len(raw_vertex_data), stride):
        raw_indices = raw_vertex_data[vertex_offset : vertex_offset + record_size]
        if len(raw_indices) < record_size:
            break
        values = _unpack_vertex_format(raw_indices, blend_format)
        if len(values) < 4:
            continue
        max_index = max(max_index, *(int(value) for value in values[:4]))
    return max_index


def _build_texture_candidates(
    files_by_draw: dict[int, list[DumpFile]],
    draw_states: dict[int, DrawState],
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
                    "rt_count": int(draw_states.get(draw_index).rt_count if draw_states.get(draw_index) else -1),
                    "slot": dump_file.slot,
                    "hash": texture_hash,
                    "source_path": dump_file.path,
                    "semantic_hint": "",
                }
            )
    return candidates


def build_bone_pool_order(candidates: list[dict]) -> list[dict]:
    ordered = sorted(
        [
            candidate
            for candidate in candidates
            if bool(candidate.get("enabled", True))
            and int(candidate.get("local_bone_count", 0)) > 0
        ],
        key=lambda item: (
            0 if bool(item.get("shadow_capture_ready", False)) else 1,
            -int(item.get("match_index_count", item.get("source_index_count", 0))),
            -int(item.get("local_bone_count", 0)),
            str(item.get("mesh_fingerprint", "")),
            str(item.get("ib_hash", "")),
            int(item.get("match_first_index", 0)),
        ),
    )
    next_base = 0
    payload = []
    for candidate in ordered:
        used_local_bone_indices = _candidate_used_local_bone_indices(candidate)
        local_bone_count = len(used_local_bone_indices)
        if local_bone_count <= 0:
            continue
        source_local_bone_count = max(
            int(candidate.get("source_local_bone_count", candidate.get("local_bone_count", 0)) or 0),
            max(used_local_bone_indices) + 1,
        )
        capture_available = bool(candidate.get("shadow_capture_ready", False))
        lod_match_excluded = bool(candidate.get("lod_match_excluded", False))
        lod_match_excluded_reason = str(candidate.get("lod_match_excluded_reason", "") or "")
        payload.append(
            {
                "ib_hash": str(candidate.get("ib_hash", "")),
                "match_first_index": int(candidate.get("match_first_index", 0)),
                "match_index_count": int(candidate.get("match_index_count", 0)),
                "producer_dispatch_index": -1,
                "global_bone_base": int(next_base),
                "local_bone_count": local_bone_count,
                "source_local_bone_count": int(source_local_bone_count),
                "used_local_bone_indices": used_local_bone_indices,
                "capture_store_base": int(next_base),
                "shadow_capture_ready": capture_available,
                "bone_capture_available": capture_available,
                "lod_match_excluded": lod_match_excluded,
                "lod_match_excluded_reason": lod_match_excluded_reason,
                "dynamic_vb0": bool(candidate.get("dynamic_vb0", lod_match_excluded)),
                "position_stream": dict(candidate.get("position_stream", {}) or {}),
                "status": _pool_record_status(capture_available, lod_match_excluded),
            }
        )
        next_base += max(0, local_bone_count)
    return payload


def _pool_record_status(capture_available: bool, lod_match_excluded: bool) -> str:
    if lod_match_excluded:
        return "capture_ready_dynamic_vb0_lod_excluded" if capture_available else "bone_mapping_only_dynamic_vb0_lod_excluded"
    return "capture_ready" if capture_available else "bone_mapping_only_no_shadow_capture"


def _candidate_used_local_bone_indices(candidate: dict) -> list[int]:
    raw_indices = candidate.get("used_local_bone_indices")
    if isinstance(raw_indices, (list, tuple)) and raw_indices:
        return sorted({int(value) for value in raw_indices if int(value) >= 0})
    local_bone_count = int(candidate.get("local_bone_count", 0) or 0)
    return list(range(max(0, local_bone_count)))


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
