"""FrameAnalysis parsing helpers."""

from __future__ import annotations

import glob
import os
import re
import struct
from collections import OrderedDict

from ..constants import GLOBAL_RESERVED_ROWS
from .models import BoneAlias, DrawRecord, LoggedDraw, PartRecord, ShadowHostRecord, TargetObjectSpec


_DRAW_VS_RE = re.compile(r"^(?P<draw>\d{6}) VSSetShader\(.* hash=(?P<hash>[0-9a-fA-F]+)\s*$")
_DRAW_VS_CB1_RE = re.compile(r"^(?P<draw>\d{6}) VSSetConstantBuffers1\(StartSlot:1,.*$")
_DRAW_CB_INFO_RE = re.compile(
    r"^\s+1: resource=.* hash=(?P<hash>[0-9a-fA-F]+) first_constant=(?P<first>\d+) num_constants=(?P<count>\d+)\s*$"
)
_DRAW_INDEXED_RE = re.compile(r"^(?P<draw>\d{6}) DrawIndexedInstanced\(IndexCountPerInstance:(?P<count>\d+),")
_OM_SET_RENDER_TARGETS_RE = re.compile(r"^(?P<draw>\d{6}) OMSetRenderTargets\(NumViews:(?P<num>\d+),")
_IB_DUMP_FILE_RE = re.compile(r"^\d{6}-ib=(?P<hash>[0-9a-fA-F]{8})")
_TEXTURE_OVERRIDE_IB_RE = re.compile(
    r"^(?P<draw>\d{6}) 3DMigoto\s+\[TextureOverride\\.*?_IB_(?P<hash>[0-9a-fA-F]{8})_merge\]"
)
_OBJECT_NAME_RE = re.compile(r"(?P<hash>[0-9A-Fa-f]{8})[-_](?P<count>\d+)")


def normalize_hash(raw_hash: str) -> str:
    normalized = str(raw_hash or "").strip().lower()
    if len(normalized) == 16:
        return normalized
    if len(normalized) == 8:
        return normalized
    raise ValueError(f"Invalid hash: {raw_hash}")


def resolve_output_dir(frameanalysis_dir: str, output_dir: str | None) -> str:
    return os.path.abspath(output_dir or frameanalysis_dir)


def parse_logged_draws(frameanalysis_dir: str) -> tuple[dict[int, LoggedDraw], tuple[str, ...]]:
    log_path = os.path.join(frameanalysis_dir, "log.txt")
    if not os.path.exists(log_path):
        raise ValueError(f"log.txt not found in {frameanalysis_dir}")

    current_vs_hash = ""
    current_vs_cb1_first = -1
    current_vs_cb1_count = -1
    expecting_vs_cb1_info = False
    draws_by_index: OrderedDict[int, LoggedDraw] = OrderedDict()

    with open(log_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for raw_line in file_handle:
            line = raw_line.rstrip("\n")

            match = _DRAW_VS_RE.match(line)
            if match:
                current_vs_hash = match.group("hash").lower()
                expecting_vs_cb1_info = False
                continue

            match = _DRAW_VS_CB1_RE.match(line)
            if match:
                expecting_vs_cb1_info = True
                continue

            if expecting_vs_cb1_info:
                info_match = _DRAW_CB_INFO_RE.match(line)
                if info_match:
                    current_vs_cb1_first = int(info_match.group("first"))
                    current_vs_cb1_count = int(info_match.group("count"))
                    expecting_vs_cb1_info = False
                    continue
                if line.startswith("000") or line.startswith(" "):
                    expecting_vs_cb1_info = False

            match = _DRAW_INDEXED_RE.match(line)
            if not match:
                continue

            draw_index = int(match.group("draw"))
            draws_by_index[draw_index] = LoggedDraw(
                draw_index=draw_index,
                vs_hash=current_vs_hash.lower(),
                match_index_count=int(match.group("count")),
                vs_cb1_first_constant=current_vs_cb1_first,
                vs_cb1_num_constants=current_vs_cb1_count,
            )

    return dict(draws_by_index), tuple()


def find_draw_records_for_targets(
    frameanalysis_dir: str,
    target_specs: list[TargetObjectSpec],
) -> tuple[list[DrawRecord], tuple[str, ...]]:
    logged_draws, warnings = parse_logged_draws(frameanalysis_dir)
    extra_warnings: list[str] = list(warnings)
    draw_records: list[DrawRecord] = []
    ib_dump_files = _index_ib_dump_files(frameanalysis_dir)
    override_draws_by_ib_hash = _index_textureoverride_ib_draws(frameanalysis_dir)

    for target_spec in target_specs:
        candidate_draw_indices = {
            _parse_draw_index_from_path(candidate_path)
            for candidate_path in ib_dump_files.get(target_spec.ib_hash.lower(), [])
        }
        candidate_draw_indices.update(override_draws_by_ib_hash.get(target_spec.ib_hash.lower(), ()))
        chosen_draw: DrawRecord | None = None
        candidate_errors: list[str] = []
        for draw_index in sorted(candidate_draw_indices):
            logged_draw = logged_draws.get(draw_index)
            if logged_draw is None:
                continue
            if logged_draw.match_index_count != target_spec.match_index_count:
                continue
            try:
                chosen_draw = _finalize_draw_record(
                    frameanalysis_dir=frameanalysis_dir,
                    target_spec=target_spec,
                    logged_draw=logged_draw,
                )
                break
            except ValueError as exc:
                candidate_errors.append(f"{target_spec.object_name}: draw {draw_index:06d}: {exc}")

        if chosen_draw is None:
            extra_warnings.extend(candidate_errors[:3])
            extra_warnings.append(
                f"{target_spec.object_name}: no draw found for {target_spec.ib_hash}-{target_spec.match_index_count}"
            )
            continue

        draw_records.append(chosen_draw)

    draw_records.sort(key=lambda item: item.draw_index)
    return draw_records, tuple(extra_warnings)


def _index_ib_dump_files(frameanalysis_dir: str) -> dict[str, list[str]]:
    files_by_ib_hash: dict[str, list[str]] = {}
    for candidate_path in glob.glob(os.path.join(frameanalysis_dir, "*-ib=*.txt")):
        file_name = os.path.basename(candidate_path)
        match = _IB_DUMP_FILE_RE.match(file_name)
        if not match:
            continue
        files_by_ib_hash.setdefault(match.group("hash").lower(), []).append(candidate_path)
    return files_by_ib_hash


def _index_textureoverride_ib_draws(frameanalysis_dir: str) -> dict[str, list[int]]:
    log_path = os.path.join(frameanalysis_dir, "log.txt")
    if not os.path.exists(log_path):
        return {}

    draws_by_ib_hash: dict[str, set[int]] = {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for raw_line in file_handle:
            match = _TEXTURE_OVERRIDE_IB_RE.match(raw_line.rstrip("\n"))
            if not match:
                continue
            draws_by_ib_hash.setdefault(match.group("hash").lower(), set()).add(int(match.group("draw")))
    return {ib_hash: sorted(draw_indices) for ib_hash, draw_indices in draws_by_ib_hash.items()}


def _finalize_draw_record(frameanalysis_dir: str, target_spec: TargetObjectSpec, logged_draw: LoggedDraw) -> DrawRecord:
    draw_prefix = f"{logged_draw.draw_index:06d}"
    vb2_matches = _glob_dump_variants(frameanalysis_dir, draw_prefix, "vb2", ".txt")
    vs_t0_matches = _glob_dump_variants(frameanalysis_dir, draw_prefix, "vs-t0", ".buf")
    vs_cb1_matches = _glob_dump_variants(frameanalysis_dir, draw_prefix, "vs-cb1", ".buf")

    if not vs_t0_matches:
        raise ValueError("missing vs-t0 dump")
    if not vs_cb1_matches:
        raise ValueError("missing vs-cb1 dump")
    if logged_draw.vs_cb1_first_constant < 0:
        raise ValueError("missing VS cb1 first_constant")
    if target_spec.local_bone_count <= 0:
        raise ValueError("invalid local_bone_count inferred from Blender object")

    return DrawRecord(
        draw_index=logged_draw.draw_index,
        object_name=target_spec.object_name,
        vs_hash=logged_draw.vs_hash,
        match_index_count=logged_draw.match_index_count,
        vs_cb1_first_constant=logged_draw.vs_cb1_first_constant,
        vs_cb1_num_constants=logged_draw.vs_cb1_num_constants,
        ib_hash=target_spec.ib_hash.lower(),
        local_bone_count=target_spec.local_bone_count,
        vb2_path=vb2_matches[0] if vb2_matches else "",
        vs_t0_path=vs_t0_matches[0],
        vs_cb1_path=vs_cb1_matches[0],
    )


def _parse_draw_index_from_path(path: str) -> int:
    file_name = os.path.basename(path)
    return int(file_name[:6])


def _glob_dump_variants(frameanalysis_dir: str, draw_prefix: str, slot_name: str, extension: str) -> list[str]:
    matches: list[str] = []
    seen_paths: set[str] = set()
    for pattern in (
        f"{draw_prefix}-{slot_name}=*{extension}",
        f"{draw_prefix}-{slot_name}-*{extension}",
    ):
        for candidate_path in glob.glob(os.path.join(frameanalysis_dir, pattern)):
            normalized_path = os.path.abspath(candidate_path)
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            matches.append(candidate_path)
    return sorted(matches)


def build_part_records(draw_records: list[DrawRecord]) -> list[PartRecord]:
    part_records: list[PartRecord] = []
    next_global_bone_base = 0
    for draw_record in sorted(draw_records, key=lambda item: item.draw_index):
        bone_count = int(draw_record.local_bone_count)
        if bone_count <= 0:
            raise ValueError(
                f"{draw_record.object_name}: invalid local bone count {bone_count}; expected Blender numeric groups"
            )
        part_records.append(
            PartRecord(
                draw_index=draw_record.draw_index,
                object_name=draw_record.object_name,
                vs_hash=draw_record.vs_hash,
                ib_hash=draw_record.ib_hash,
                match_index_count=draw_record.match_index_count,
                bone_count=bone_count,
                global_bone_base=next_global_bone_base,
                vb2_path=draw_record.vb2_path,
                vs_t0_path=draw_record.vs_t0_path,
                vs_cb1_path=draw_record.vs_cb1_path,
                vs_cb1_first_constant=draw_record.vs_cb1_first_constant,
                vs_cb1_num_constants=draw_record.vs_cb1_num_constants,
            )
        )
        next_global_bone_base += bone_count
    return part_records


def detect_last_shadow_host(frameanalysis_dir: str) -> ShadowHostRecord:
    log_path = os.path.join(frameanalysis_dir, "log.txt")
    if not os.path.exists(log_path):
        raise ValueError(f"log.txt not found in {frameanalysis_dir}")

    first_gbuffer_draw_index: int | None = None
    last_draw_before_gbuffer: LoggedDraw | None = None
    current_draws, _warnings = parse_logged_draws(frameanalysis_dir)

    with open(log_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for raw_line in file_handle:
            line = raw_line.rstrip("\n")

            om_match = _OM_SET_RENDER_TARGETS_RE.match(line)
            if om_match:
                num_views = int(om_match.group("num"))
                draw_index = int(om_match.group("draw"))
                if num_views >= 5:
                    first_gbuffer_draw_index = draw_index
                    break

    if first_gbuffer_draw_index is None:
        raise ValueError("Could not detect the first G-buffer draw from OMSetRenderTargets(NumViews:5)")

    for draw_index in sorted(current_draws):
        if draw_index >= first_gbuffer_draw_index:
            break
        last_draw_before_gbuffer = current_draws[draw_index]

    if last_draw_before_gbuffer is None:
        raise ValueError("Could not find a draw before the first G-buffer draw")

    ib_dump_files = _index_ib_dump_files(frameanalysis_dir)
    draw_prefix = f"{last_draw_before_gbuffer.draw_index:06d}"
    ib_hash = ""
    for paths in ib_dump_files.values():
        for candidate_path in paths:
            if os.path.basename(candidate_path).startswith(f"{draw_prefix}-ib="):
                file_name = os.path.basename(candidate_path)
                match = _IB_DUMP_FILE_RE.match(file_name)
                if match:
                    ib_hash = match.group("hash").lower()
                    break
        if ib_hash:
            break

    if not ib_hash:
        direct_matches = glob.glob(os.path.join(frameanalysis_dir, f"{draw_prefix}-ib=*.txt"))
        if direct_matches:
            file_name = os.path.basename(direct_matches[0])
            match = _IB_DUMP_FILE_RE.match(file_name)
            if match:
                ib_hash = match.group("hash").lower()

    if not ib_hash:
        raise ValueError(f"Could not resolve IB hash for draw {last_draw_before_gbuffer.draw_index:06d}")

    return ShadowHostRecord(
        draw_index=last_draw_before_gbuffer.draw_index,
        ib_hash=ib_hash,
        match_index_count=last_draw_before_gbuffer.match_index_count,
        vs_hash=last_draw_before_gbuffer.vs_hash,
    )


def build_duplicate_bone_aliases(part_records: list[PartRecord]) -> list[BoneAlias]:
    signature_to_canonical: dict[bytes, tuple[PartRecord, int]] = {}
    aliases: list[BoneAlias] = []

    for part_record in part_records:
        current_base, previous_base = _read_palette_bases_from_vs_cb1(part_record.vs_cb1_path)
        with open(part_record.vs_t0_path, "rb") as file_handle:
            vs_t0_blob = file_handle.read()
        total_rows = len(vs_t0_blob) // 16
        if total_rows <= 0:
            raise ValueError(f"vs-t0 buffer is empty: {part_record.vs_t0_path}")

        for local_bone in range(part_record.bone_count):
            signature = _build_bone_signature_from_blob(
                vs_t0_blob=vs_t0_blob,
                total_rows=total_rows,
                current_base=current_base,
                previous_base=previous_base,
                local_bone=local_bone,
            )
            canonical = signature_to_canonical.get(signature)
            if canonical is None:
                signature_to_canonical[signature] = (part_record, local_bone)
                continue

            canonical_part, canonical_local_bone = canonical
            if (
                canonical_part.draw_index == part_record.draw_index
                and canonical_part.ib_hash == part_record.ib_hash
            ):
                continue

            aliases.append(
                BoneAlias(
                    src_draw_index=part_record.draw_index,
                    src_object_name=part_record.object_name,
                    src_ib_hash=part_record.ib_hash,
                    src_local_bone=local_bone,
                    src_global_bone=part_record.global_bone_base + local_bone,
                    canonical_draw_index=canonical_part.draw_index,
                    canonical_object_name=canonical_part.object_name,
                    canonical_ib_hash=canonical_part.ib_hash,
                    canonical_local_bone=canonical_local_bone,
                    canonical_global_bone=canonical_part.global_bone_base + canonical_local_bone,
                    confidence="exact_current_previous",
                )
            )
    return aliases


def _read_palette_bases_from_vs_cb1(vs_cb1_path: str) -> tuple[int, int]:
    with open(vs_cb1_path, "rb") as file_handle:
        file_handle.seek(5 * 16)
        row_bytes = file_handle.read(16)
    if len(row_bytes) != 16:
        raise ValueError(f"vs-cb1 buffer too small: {vs_cb1_path}")
    x_value, y_value, _z_value, _w_value = struct.unpack("<4I", row_bytes)
    return x_value, y_value


def _build_bone_signature_from_blob(
    vs_t0_blob: bytes,
    total_rows: int,
    current_base: int,
    previous_base: int,
    local_bone: int,
) -> bytes:
    current_row = current_base + GLOBAL_RESERVED_ROWS + local_bone * 3
    previous_row = previous_base + GLOBAL_RESERVED_ROWS + local_bone * 3
    current_blob = _read_three_rows_from_blob(vs_t0_blob, current_row, total_rows)
    previous_blob = _read_three_rows_from_blob(vs_t0_blob, previous_row, total_rows)
    return current_blob + previous_blob


def _read_three_rows_from_blob(vs_t0_blob: bytes, row_index: int, total_rows: int) -> bytes:
    blobs: list[bytes] = []
    for row_offset in range(3):
        wrapped_row_index = (int(row_index) + row_offset) % int(total_rows)
        byte_offset = wrapped_row_index * 16
        row_blob = vs_t0_blob[byte_offset : byte_offset + 16]
        if len(row_blob) != 16:
            raise ValueError(f"vs-t0 buffer too small for row {wrapped_row_index}")
        blobs.append(row_blob)
    return b"".join(blobs)


def infer_mesh_identity_from_name(object_name: str) -> tuple[str, int] | None:
    match = _OBJECT_NAME_RE.search(str(object_name or ""))
    if not match:
        return None
    return match.group("hash").lower(), int(match.group("count"))
