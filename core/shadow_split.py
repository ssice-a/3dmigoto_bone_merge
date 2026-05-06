"""Generate main-INI shadow split sections that consume BoneStore palettes."""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass

from .frameanalysis import detect_last_shadow_host
from .ini_export import build_bonestore_namespace
from .io import read_json, write_json
from .models import ShadowHostRecord, ShadowSplitResult

_SECTION_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")
_HASH_RE = re.compile(r"^\s*hash\s*=\s*(?P<hash>[0-9A-Fa-f]{8,16})\s*(?:;.*)?$")
_MATCH_INDEX_COUNT_RE = re.compile(r"^\s*match_index_count\s*=\s*(?P<count>\d+)\s*(?:;.*)?$")
_NAMESPACE_RE = re.compile(r"^\s*namespace\s*=\s*(?P<value>.+?)\s*$", re.IGNORECASE)
_IF_RE = re.compile(r"^(?P<indent>\s*)if\s+(?P<condition>.+?)\s*$", re.IGNORECASE)
_ENDIF_RE = re.compile(r"^\s*endif\s*(?:;.*)?$", re.IGNORECASE)
_LOCAL_COMMANDLIST_RUN_RE = re.compile(
    r"^(?P<indent>\s*)run\s*=\s*(?P<name>commandlist_[^\\\s;]+)\s*(?:;.*)?$",
    re.IGNORECASE,
)

_HOST_BEGIN_MARKER = "; BMC_SHADOW_HOST_BEGIN"
_HOST_END_MARKER = "; BMC_SHADOW_HOST_END"
_WHITE_RESOURCE_BEGIN_MARKER = "; BMC_WHITE_RESOURCE_BEGIN"
_WHITE_RESOURCE_END_MARKER = "; BMC_WHITE_RESOURCE_END"
_DIRECT_PAYLOAD_BEGIN_MARKER = "; BMC_DIRECT_SHADOW_PAYLOAD_BEGIN"
_DIRECT_PAYLOAD_END_MARKER = "; BMC_DIRECT_SHADOW_PAYLOAD_END"
_WHITE_RESOURCE_NAME = "ResourceBMCWhite"
_WHITE_TEXTURE_RELATIVE_PATH = "Texture/BMC_White.dds"

_SOURCE_CONTROL_LINE_RE = re.compile(
    r"^\s*(hash|match_first_index|match_index_count|match_priority|handling)\s*=",
    re.IGNORECASE,
)
_SOURCE_VARIABLE_LINE_RE = re.compile(r"^\s*\$[A-Za-z_][A-Za-z0-9_]*\s*=")


@dataclass
class ParsedSection:
    header_line: str
    name: str
    body_lines: list[str]


@dataclass(frozen=True)
class ChunkSectionPayload:
    ib_hash: str
    match_index_count: int
    chunk_index: int
    resource_suffix: str
    source_section_name: str
    payload_kind: str
    payload_lines: tuple[str, ...]


def generate_shadow_split(
    frameanalysis_dir: str,
    export_manifest_path: str,
    bonestore_ini_path: str,
    source_ini_path: str,
    shadow_host_hash: str = "",
    shadow_host_match_index_count: int = -1,
) -> ShadowSplitResult:
    normalized_frameanalysis_dir = os.path.abspath(frameanalysis_dir) if frameanalysis_dir else ""
    normalized_export_manifest_path = os.path.abspath(export_manifest_path)
    normalized_bonestore_ini_path = os.path.abspath(bonestore_ini_path)
    normalized_source_ini_path = os.path.abspath(source_ini_path)

    if not os.path.exists(normalized_export_manifest_path):
        raise ValueError("Export manifest does not exist; run Prepare Export / Palette first")
    if not os.path.exists(normalized_bonestore_ini_path):
        raise ValueError("BoneStore.ini does not exist; run Scan / Prepare Export first")
    if not os.path.exists(normalized_source_ini_path):
        raise ValueError("Source main INI path does not exist")

    export_manifest = read_json(normalized_export_manifest_path)
    palette_records = [
        record
        for record in export_manifest.get("palettes", [])
        if int(record.get("local_bone_count", 0)) > 0
    ]
    if not palette_records:
        raise ValueError("Export manifest does not contain any prepared local palettes")

    bonestore_namespace = str(
        export_manifest.get("bonestore_namespace")
        or build_bonestore_namespace(os.path.dirname(normalized_bonestore_ini_path))
    )
    bonestore_namespace = _ensure_bonestore_namespace(normalized_bonestore_ini_path, bonestore_namespace)

    host_record = _resolve_shadow_host(
        frameanalysis_dir=normalized_frameanalysis_dir,
        shadow_host_hash=shadow_host_hash,
        shadow_host_match_index_count=shadow_host_match_index_count,
    )

    source_text, source_encoding, source_newline = _read_text_preserving_format(normalized_source_ini_path)
    source_lines = _strip_generated_blocks(_split_lines_preserving_empty(source_text))
    preamble_lines, sections = _parse_sections(source_lines)
    _ensure_white_texture_file(os.path.dirname(normalized_source_ini_path))
    has_white_resource = any(section.name.lower() == _WHITE_RESOURCE_NAME.lower() for section in sections)
    payloads: list[ChunkSectionPayload] = []
    rewritten_section_names: list[str] = []
    for palette_record in palette_records:
        section_payloads, rewritten_names = _rewrite_sections_for_chunk(sections, palette_record)
        payloads.extend(section_payloads)
        rewritten_section_names.extend(rewritten_names)

    if not payloads:
        raise ValueError("No matching source TextureOverride sections were found for the prepared export chunks")

    host_blocks: list[str] = _build_generated_host_block(
        host_record=host_record,
        bonestore_namespace=bonestore_namespace,
        payloads=payloads,
        name_suffix="main",
    )
    lod_shadow_variants = []
    for lod_variant in export_manifest.get("lod_variants", []) or []:
        lod_shadow_hash = str(lod_variant.get("shadow_host_hash", "") or "").strip().lower()
        lod_shadow_count = int(lod_variant.get("shadow_host_match_index_count", -1))
        if not lod_shadow_hash or lod_shadow_count <= 0:
            continue
        palette_override_records = [
            record
            for record in lod_variant.get("palette_overrides", []) or []
            if str(record.get("base_resource_suffix", "") or "").strip()
            and str(record.get("resource_suffix", "") or "").strip()
        ]
        resource_suffix_by_base = {
            str(record.get("base_resource_suffix", "")).strip(): str(record.get("resource_suffix", "")).strip()
            for record in palette_override_records
        }
        if not resource_suffix_by_base:
            continue
        variant_payloads: list[ChunkSectionPayload] = []
        for payload in payloads:
            mapped_resource_suffix = resource_suffix_by_base.get(str(payload.resource_suffix))
            if not mapped_resource_suffix:
                continue
            variant_payloads.append(
                ChunkSectionPayload(
                    ib_hash=str(payload.ib_hash),
                    match_index_count=int(payload.match_index_count),
                    chunk_index=int(payload.chunk_index),
                    resource_suffix=mapped_resource_suffix,
                    source_section_name=str(payload.source_section_name),
                    payload_kind=str(payload.payload_kind),
                    payload_lines=tuple(payload.payload_lines),
                )
            )
        if not variant_payloads:
            continue
        lod_host_record = ShadowHostRecord(
            draw_index=-1,
            ib_hash=lod_shadow_hash,
            match_index_count=lod_shadow_count,
            vs_hash=str(lod_variant.get("shadow_host_vs_hash", "") or ""),
        )
        host_blocks.extend(
            [
                "",
                *_build_generated_host_block(
                    host_record=lod_host_record,
                    bonestore_namespace=bonestore_namespace,
                    payloads=variant_payloads,
                    name_suffix=str(lod_variant.get("variant_id", "lod") or "lod"),
                ),
            ]
        )
        lod_shadow_variants.append(
            {
                "variant_id": str(lod_variant.get("variant_id", "lod") or "lod"),
                "shadow_host_hash": lod_shadow_hash,
                "shadow_host_match_index_count": lod_shadow_count,
                "shadow_host_vs_hash": str(lod_variant.get("shadow_host_vs_hash", "") or ""),
                "migrated_chunks": [
                    {
                        "base_resource_suffix": str(record.get("base_resource_suffix", "") or ""),
                        "resource_suffix": str(record.get("resource_suffix", "") or ""),
                    }
                    for record in palette_override_records
                ],
            }
        )
    generated_host_lines = [_HOST_BEGIN_MARKER, *host_blocks, _HOST_END_MARKER]

    final_lines: list[str] = list(preamble_lines)
    for section in sections:
        final_lines.append(section.header_line)
        final_lines.extend(section.body_lines)
    if final_lines and final_lines[-1] != "":
        final_lines.append("")
    final_lines.extend(generated_host_lines)
    if not has_white_resource:
        if final_lines and final_lines[-1] != "":
            final_lines.append("")
        final_lines.extend(_build_white_resource_block())
    final_text = source_newline.join(final_lines).rstrip() + source_newline

    with open(normalized_source_ini_path, "w", encoding=source_encoding, newline="") as file_handle:
        file_handle.write(final_text)

    export_manifest["shadow_migration"] = {
        "source_ini_path": normalized_source_ini_path,
        "bonestore_path": normalized_bonestore_ini_path,
        "bonestore_namespace": bonestore_namespace,
        "shadow_host_hash": host_record.ib_hash,
        "shadow_host_match_index_count": host_record.match_index_count,
        "shadow_host_vs_hash": host_record.vs_hash,
        "shadow_host_draw_index": host_record.draw_index,
        "white_texture_resource": _WHITE_RESOURCE_NAME,
        "white_texture_path": _WHITE_TEXTURE_RELATIVE_PATH,
        "migrated_chunks": [
            {
                "ib_hash": payload.ib_hash,
                "match_index_count": payload.match_index_count,
                "chunk_index": payload.chunk_index,
                "resource_suffix": payload.resource_suffix,
                "source_section_name": payload.source_section_name,
                "payload_kind": payload.payload_kind,
            }
            for payload in payloads
        ],
        "rewritten_source_sections": sorted(set(rewritten_section_names)),
        "lod_shadow_variants": lod_shadow_variants,
    }
    write_json(normalized_export_manifest_path, export_manifest)

    return ShadowSplitResult(
        source_ini_path=normalized_source_ini_path,
        bonestore_ini_path=normalized_bonestore_ini_path,
        export_manifest_path=normalized_export_manifest_path,
        shadow_host_hash=host_record.ib_hash,
        shadow_host_match_index_count=host_record.match_index_count,
        migrated_chunks=len({(payload.ib_hash, payload.match_index_count, payload.chunk_index) for payload in payloads}),
        rewritten_sections=len(set(rewritten_section_names)),
        shadow_host_vs_hash=host_record.vs_hash,
    )


def _resolve_shadow_host(
    frameanalysis_dir: str,
    shadow_host_hash: str,
    shadow_host_match_index_count: int,
) -> ShadowHostRecord:
    normalized_hash = str(shadow_host_hash or "").strip().lower()
    if normalized_hash and int(shadow_host_match_index_count) > 0:
        return ShadowHostRecord(
            draw_index=-1,
            ib_hash=normalized_hash,
            match_index_count=int(shadow_host_match_index_count),
            vs_hash="",
        )
    if not frameanalysis_dir:
        raise ValueError("FrameAnalysis Dir is required when Shadow Host Hash/Count are not set")
    return detect_last_shadow_host(frameanalysis_dir)


def _read_text_preserving_format(path: str) -> tuple[str, str, str]:
    raw_bytes = b""
    with open(path, "rb") as file_handle:
        raw_bytes = file_handle.read()

    encoding_candidates = ("utf-8-sig", "utf-8", "gb18030", "utf-16", "utf-16-le")
    chosen_encoding = "utf-8"
    for candidate in encoding_candidates:
        try:
            return raw_bytes.decode(candidate), candidate, _detect_newline(raw_bytes)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace"), chosen_encoding, _detect_newline(raw_bytes)


def _ensure_bonestore_namespace(path: str, fallback_namespace: str) -> str:
    text, encoding, newline = _read_text_preserving_format(path)
    lines = _split_lines_preserving_empty(text)
    for line in lines[:20]:
        match = _NAMESPACE_RE.match(line)
        if match:
            return match.group("value").strip()

    updated_lines = [f"namespace = {fallback_namespace}", ""]
    updated_lines.extend(lines)
    updated_text = newline.join(updated_lines).rstrip() + newline
    with open(path, "w", encoding=encoding, newline="") as file_handle:
        file_handle.write(updated_text)
    return fallback_namespace


def _detect_newline(raw_bytes: bytes) -> str:
    if b"\r\n" in raw_bytes:
        return "\r\n"
    return "\n"


def _split_lines_preserving_empty(text: str) -> list[str]:
    return text.splitlines()


def _strip_generated_blocks(lines: list[str]) -> list[str]:
    lines = _strip_generated_block(lines, _HOST_BEGIN_MARKER, _HOST_END_MARKER)
    lines = _strip_generated_block(lines, _WHITE_RESOURCE_BEGIN_MARKER, _WHITE_RESOURCE_END_MARKER)
    return lines


def _strip_generated_block(lines: list[str], begin_marker: str, end_marker: str) -> list[str]:
    start_index = -1
    end_index = -1
    for index, line in enumerate(lines):
        if line.strip() == begin_marker:
            start_index = index
            continue
        if start_index >= 0 and line.strip() == end_marker:
            end_index = index
            break
    if start_index >= 0 and end_index >= start_index:
        return lines[:start_index] + lines[end_index + 1 :]
    return lines


def _parse_sections(lines: list[str]) -> tuple[list[str], list[ParsedSection]]:
    preamble_lines: list[str] = []
    sections: list[ParsedSection] = []
    current_header: str | None = None
    current_name = ""
    current_body: list[str] = []

    for line in lines:
        section_match = _SECTION_RE.match(line)
        if section_match:
            if current_header is None:
                if preamble_lines and preamble_lines[-1] != "":
                    preamble_lines.append("")
            else:
                sections.append(
                    ParsedSection(header_line=current_header, name=current_name, body_lines=current_body)
                )
            current_header = line
            current_name = section_match.group("name").strip()
            current_body = []
            continue

        if current_header is None:
            preamble_lines.append(line)
            continue
        current_body.append(line)

    if current_header is not None:
        sections.append(ParsedSection(header_line=current_header, name=current_name, body_lines=current_body))
    return preamble_lines, sections


def _rewrite_sections_for_chunk(
    sections: list[ParsedSection],
    palette_record: dict,
) -> tuple[list[ChunkSectionPayload], list[str]]:
    ib_hash = str(palette_record.get("ib_hash", "")).lower()
    match_index_count = int(palette_record.get("match_index_count", 0))
    chunk_index = int(palette_record.get("chunk_index", 0))
    resource_suffix = str(palette_record.get("resource_suffix", f"{ib_hash}_{match_index_count}_{chunk_index}"))

    payloads: list[ChunkSectionPayload] = []
    rewritten_names: list[str] = []
    for section in sections:
        if not _section_matches_hash_and_count(section, ib_hash, match_index_count):
            continue

        direct_payload, direct_rewritten = _extract_direct_payload(section.body_lines)
        if direct_payload:
            payloads.append(
                ChunkSectionPayload(
                    ib_hash=ib_hash,
                    match_index_count=match_index_count,
                    chunk_index=chunk_index,
                    resource_suffix=resource_suffix,
                    source_section_name=section.name,
                    payload_kind="direct",
                    payload_lines=tuple(direct_payload),
                )
            )
            if direct_rewritten:
                rewritten_names.append(section.name)
            continue

        commandlist_names, commandlist_rewritten = _rewrite_commandlist_textureoverride(section)
        for commandlist_name in commandlist_names:
            payloads.append(
                ChunkSectionPayload(
                    ib_hash=ib_hash,
                    match_index_count=match_index_count,
                    chunk_index=chunk_index,
                    resource_suffix=resource_suffix,
                    source_section_name=section.name,
                    payload_kind="commandlist",
                    payload_lines=(f"run = {commandlist_name}",),
                )
            )
        if commandlist_names and commandlist_rewritten:
            rewritten_names.append(section.name)
    return payloads, rewritten_names


def _section_matches_hash_and_count(section: ParsedSection, ib_hash: str, match_index_count: int) -> bool:
    section_hash = ""
    section_match_count = -1
    for line in section.body_lines:
        hash_match = _HASH_RE.match(line)
        if hash_match:
            section_hash = hash_match.group("hash").lower()
            continue
        count_match = _MATCH_INDEX_COUNT_RE.match(line)
        if count_match:
            section_match_count = int(count_match.group("count"))
    return section_hash == ib_hash and section_match_count == match_index_count


def _extract_direct_payload(section_body_lines: list[str]) -> tuple[list[str], bool]:
    marked_payload = _extract_marked_direct_payload(section_body_lines)
    if marked_payload is not None:
        return marked_payload, False

    block_range = _find_condition_block(section_body_lines, "vs != 200")
    if block_range is not None:
        start_index, end_index, _indent = block_range
        return list(section_body_lines[start_index + 1 : end_index]), False

    block_range = _find_condition_block(section_body_lines, "vs == 200")
    if block_range is not None:
        start_index, end_index, _indent = block_range
        return list(section_body_lines[start_index + 1 : end_index]), False

    payload_range = _find_unconditional_payload_range(section_body_lines)
    if payload_range is None:
        return [], False

    start_index, end_index = payload_range
    payload_lines = list(section_body_lines[start_index : end_index + 1])
    replacement_lines = ["if vs != 200"]
    replacement_lines.extend(payload_lines)
    replacement_lines.append("endif")
    section_body_lines[start_index : end_index + 1] = replacement_lines
    return payload_lines, True


def _extract_marked_direct_payload(section_body_lines: list[str]) -> list[str] | None:
    start_index = -1
    end_index = -1
    for index, line in enumerate(section_body_lines):
        stripped = line.strip()
        if stripped == _DIRECT_PAYLOAD_BEGIN_MARKER:
            start_index = index
            continue
        if start_index >= 0 and stripped == _DIRECT_PAYLOAD_END_MARKER:
            end_index = index
            break
    if start_index < 0 or end_index < start_index:
        return None

    payload_lines: list[str] = []
    for line in section_body_lines[start_index + 1 : end_index]:
        stripped = line.lstrip()
        if not stripped.startswith(";"):
            payload_lines.append(line)
            continue
        uncommented = stripped[1:]
        if uncommented.startswith(" "):
            uncommented = uncommented[1:]
        payload_lines.append(uncommented)
    return payload_lines


def _find_condition_block(section_body_lines: list[str], target_condition: str) -> tuple[int, int, str] | None:
    block_stack: list[str] = []
    target_start_index = -1
    target_depth = -1
    target_indent = ""
    normalized_target = _normalize_condition(target_condition)

    for index, line in enumerate(section_body_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        if_match = _IF_RE.match(line)
        if if_match:
            condition = _normalize_condition(if_match.group("condition"))
            if condition == normalized_target and target_start_index < 0:
                target_start_index = index
                target_depth = len(block_stack)
                target_indent = if_match.group("indent")
            block_stack.append(condition)
            continue

        if _ENDIF_RE.match(line):
            if target_start_index >= 0 and len(block_stack) - 1 == target_depth:
                return target_start_index, index, target_indent
            if block_stack:
                block_stack.pop()
    return None


def _find_unconditional_payload_range(section_body_lines: list[str]) -> tuple[int, int] | None:
    first_payload_index = -1
    last_payload_index = -1

    for index, line in enumerate(section_body_lines):
        if not _is_executable_payload_line(line):
            continue
        first_payload_index = index
        break

    if first_payload_index < 0:
        return None

    for index in range(len(section_body_lines) - 1, first_payload_index - 1, -1):
        line = section_body_lines[index]
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(";"):
            continue
        last_payload_index = index
        break

    if last_payload_index < first_payload_index:
        return None
    return first_payload_index, last_payload_index


def _is_executable_payload_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(";"):
        return False
    if _SOURCE_CONTROL_LINE_RE.match(line):
        return False
    if _SOURCE_VARIABLE_LINE_RE.match(line):
        return False
    return True


def _normalize_condition(condition: str) -> str:
    lowered = condition.strip().lower()
    return " ".join(lowered.replace("==", " == ").replace("!=", " != ").split())


def _rewrite_commandlist_textureoverride(section: ParsedSection) -> tuple[list[str], bool]:
    rewritten_lines: list[str] = []
    commandlist_names: list[str] = []
    changed = False
    active_conditions: list[str] = []

    for line in section.body_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(";"):
            if_match = _IF_RE.match(line)
            if if_match:
                active_conditions.append(_normalize_condition(if_match.group("condition")))
                rewritten_lines.append(line)
                continue
            if _ENDIF_RE.match(line):
                if active_conditions:
                    active_conditions.pop()
                rewritten_lines.append(line)
                continue

        run_match = _LOCAL_COMMANDLIST_RUN_RE.match(line)
        if run_match and not stripped.startswith(";"):
            commandlist_name = run_match.group("name")
            commandlist_names.append(commandlist_name)
            if "vs != 200" in active_conditions:
                rewritten_lines.append(line)
                continue

            indent = run_match.group("indent")
            rewritten_lines.extend(
                [
                    f"{indent}if vs != 200",
                    f"{indent}    run = {commandlist_name}",
                    f"{indent}endif",
                ]
            )
            changed = True
            continue

        rewritten_lines.append(line)

    if changed:
        section.body_lines[:] = rewritten_lines
    unique_commandlists: list[str] = []
    seen_names: set[str] = set()
    for commandlist_name in commandlist_names:
        lowered_name = commandlist_name.lower()
        if lowered_name in seen_names:
            continue
        seen_names.add(lowered_name)
        unique_commandlists.append(commandlist_name)
    return unique_commandlists, changed


def _build_generated_host_block(
    host_record: ShadowHostRecord,
    bonestore_namespace: str,
    payloads: list[ChunkSectionPayload],
    name_suffix: str = "",
) -> list[str]:
    section_name = (
        f"TextureOverride_BMC_ShadowHost_{host_record.ib_hash.lower()}_{int(host_record.match_index_count)}"
        f"_{re.sub(r'[^0-9A-Za-z_]+', '_', str(name_suffix or 'host')).strip('_') or 'host'}"
    )
    lines = [
        f"[{section_name}]",
        f"hash = {host_record.ib_hash.lower()}",
        f"match_index_count = {int(host_record.match_index_count)}",
        "match_priority = -300",
        "if vs == 200",
        f"  ps-t0 = {_WHITE_RESOURCE_NAME}",
        f"  run = CustomShader\\{bonestore_namespace}\\_ExtractCB1",
        f"  vs-t0 = Resource\\{bonestore_namespace}\\LocalFakeT0_SRV",
        f"  run = CustomShader\\{bonestore_namespace}\\_RedirectCB1",
        f"  vs-cb1 = Resource\\{bonestore_namespace}\\FakeCB1",
    ]

    for payload in payloads:
        lines.append(
            f"  ; BMC chunk {payload.ib_hash}-{payload.match_index_count}-{payload.chunk_index} from [{payload.source_section_name}]"
        )
        lines.append(f"  cs-t2 = Resource\\{bonestore_namespace}\\LocalPalette_{payload.resource_suffix}")
        lines.append(f"  cs-t3 = Resource\\{bonestore_namespace}\\LocalPaletteMeta_{payload.resource_suffix}")
        lines.append(f"  run = CustomShader\\{bonestore_namespace}\\_GatherBones")
        for payload_line in payload.payload_lines:
            lines.append(f"  {payload_line}" if payload_line else "")

    lines.extend(
        [
            "endif",
        ]
    )
    return lines


def _build_white_resource_block() -> list[str]:
    return [
        _WHITE_RESOURCE_BEGIN_MARKER,
        f"[{_WHITE_RESOURCE_NAME}]",
        f"filename = {_WHITE_TEXTURE_RELATIVE_PATH}",
        _WHITE_RESOURCE_END_MARKER,
    ]


def _ensure_white_texture_file(source_ini_dir: str) -> str:
    texture_path = os.path.join(source_ini_dir, *_WHITE_TEXTURE_RELATIVE_PATH.split("/"))
    os.makedirs(os.path.dirname(texture_path), exist_ok=True)
    if not os.path.exists(texture_path):
        with open(texture_path, "wb") as file_handle:
            file_handle.write(_build_white_dds_1x1())
    return texture_path


def _build_white_dds_1x1() -> bytes:
    magic = b"DDS "
    header_size = 124
    flags = 0x0000100F  # CAPS | HEIGHT | WIDTH | PITCH | PIXELFORMAT
    height = 1
    width = 1
    pitch_or_linear_size = 4
    depth = 0
    mip_map_count = 0
    reserved1 = (0,) * 11
    pixel_format_size = 32
    pixel_format_flags = 0x00000041  # ALPHAPIXELS | RGB
    four_cc = 0
    rgb_bit_count = 32
    r_bit_mask = 0x00FF0000
    g_bit_mask = 0x0000FF00
    b_bit_mask = 0x000000FF
    a_bit_mask = 0xFF000000
    caps = 0x00001000
    caps2 = 0
    caps3 = 0
    caps4 = 0
    reserved2 = 0
    header = struct.pack(
        "<7I11I8I5I",
        header_size,
        flags,
        height,
        width,
        pitch_or_linear_size,
        depth,
        mip_map_count,
        *reserved1,
        pixel_format_size,
        pixel_format_flags,
        four_cc,
        rgb_bit_count,
        r_bit_mask,
        g_bit_mask,
        b_bit_mask,
        a_bit_mask,
        caps,
        caps2,
        caps3,
        caps4,
        reserved2,
    )
    return magic + header + b"\xff\xff\xff\xff"
