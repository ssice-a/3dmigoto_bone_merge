"""Vertex layout model and lossless raw-carrier helpers.

The input assembler describes semantic views, but export writes physical byte
ranges.  Keeping those concepts separate is important because multiple shader
semantics can legally alias the same bytes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class VertexLayoutError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DxgiFormatSpec:
    name: str
    dtype: str
    component_count: int
    conversion: str | None = None

    @property
    def byte_size(self) -> int:
        return _DTYPE_SIZES[self.dtype] * int(self.component_count)


_DTYPE_SIZES = {
    "u1": 1,
    "i1": 1,
    "<u2": 2,
    "<i2": 2,
    "<f2": 2,
    "<u4": 4,
    "<i4": 4,
    "<f4": 4,
}


DXGI_FORMATS: dict[str, DxgiFormatSpec] = {
    spec.name: spec
    for spec in (
        DxgiFormatSpec("R32_FLOAT", "<f4", 1),
        DxgiFormatSpec("R32_UINT", "<u4", 1),
        DxgiFormatSpec("R32_SINT", "<i4", 1),
        DxgiFormatSpec("R32G32_FLOAT", "<f4", 2),
        DxgiFormatSpec("R32G32_UINT", "<u4", 2),
        DxgiFormatSpec("R32G32_SINT", "<i4", 2),
        DxgiFormatSpec("R32G32B32_FLOAT", "<f4", 3),
        DxgiFormatSpec("R32G32B32_UINT", "<u4", 3),
        DxgiFormatSpec("R32G32B32_SINT", "<i4", 3),
        DxgiFormatSpec("R32G32B32A32_FLOAT", "<f4", 4),
        DxgiFormatSpec("R32G32B32A32_UINT", "<u4", 4),
        DxgiFormatSpec("R32G32B32A32_SINT", "<i4", 4),
        DxgiFormatSpec("R16G16_FLOAT", "<f2", 2),
        DxgiFormatSpec("R16G16_UNORM", "<u2", 2, "unorm16"),
        DxgiFormatSpec("R16G16_SNORM", "<i2", 2, "snorm16"),
        DxgiFormatSpec("R16G16_UINT", "<u2", 2),
        DxgiFormatSpec("R16G16_SINT", "<i2", 2),
        DxgiFormatSpec("R16G16B16A16_FLOAT", "<f2", 4),
        DxgiFormatSpec("R16G16B16A16_UNORM", "<u2", 4, "unorm16"),
        DxgiFormatSpec("R16G16B16A16_SNORM", "<i2", 4, "snorm16"),
        DxgiFormatSpec("R16G16B16A16_UINT", "<u2", 4),
        DxgiFormatSpec("R16G16B16A16_SINT", "<i2", 4),
        DxgiFormatSpec("R8G8B8A8_UNORM", "u1", 4, "unorm8"),
        DxgiFormatSpec("R8G8B8A8_SNORM", "i1", 4, "snorm8"),
        DxgiFormatSpec("R8G8B8A8_UINT", "u1", 4),
        DxgiFormatSpec("R8G8B8A8_SINT", "i1", 4),
    )
}


@dataclass(frozen=True, slots=True)
class VertexField:
    semantic_name: str
    semantic_index: int
    format: str
    offset: int
    size: int

    @property
    def semantic(self) -> str:
        return f"{self.semantic_name}{self.semantic_index}"

    @property
    def end(self) -> int:
        return int(self.offset) + int(self.size)


@dataclass(frozen=True, slots=True)
class PhysicalVertexField:
    offset: int
    size: int
    aliases: tuple[VertexField, ...]

    @property
    def end(self) -> int:
        return int(self.offset) + int(self.size)

    @property
    def primary(self) -> VertexField:
        return min(self.aliases, key=_semantic_priority)


@dataclass(frozen=True, slots=True)
class VertexSlotLayout:
    slot_name: str
    slot_index: int
    stride: int
    fields: tuple[VertexField, ...]
    physical_fields: tuple[PhysicalVertexField, ...]
    resource_hash: str = ""
    backing_hash: str = ""
    source_buf: str = ""
    byte_offset: int = 0
    vertex_count: int = 0

    @property
    def raw_word_count(self) -> int:
        return (int(self.stride) + 3) // 4

    @property
    def source_identity(self) -> tuple[str, str, str, int, int, int] | None:
        resource_hash = str(self.resource_hash or "").lower()
        source_buf = str(self.source_buf or "").replace("\\", "/").lower()
        if not resource_hash and not source_buf:
            return None
        return (
            resource_hash,
            str(self.backing_hash or "").lower(),
            source_buf,
            int(self.byte_offset),
            int(self.stride),
            int(self.vertex_count),
        )


def normalize_dxgi_format(fmt: str) -> str:
    normalized = str(fmt or "").strip().upper()
    if normalized.startswith("DXGI_FORMAT_"):
        normalized = normalized[len("DXGI_FORMAT_"):]
    return normalized


def dxgi_format_spec(fmt: str) -> DxgiFormatSpec | None:
    return DXGI_FORMATS.get(normalize_dxgi_format(fmt))


def dxgi_format_size(fmt: str) -> int:
    normalized = normalize_dxgi_format(fmt)
    spec = DXGI_FORMATS.get(normalized)
    if spec is not None:
        return int(spec.byte_size)
    packed_match = re.fullmatch(
        r"(?:[RGBA]\d+)+_(?:FLOAT|UINT|SINT|UNORM|SNORM|TYPELESS)",
        normalized,
    )
    if packed_match is None:
        return 0
    bit_count = sum(int(value) for value in re.findall(r"[RGBA](\d+)", normalized))
    return bit_count // 8 if bit_count > 0 and bit_count % 8 == 0 else 0


def build_slot_layout(slot_name: str, payload: dict) -> VertexSlotLayout:
    normalized_name = str(payload.get("slot", slot_name) or slot_name).lower()
    slot_index = _slot_index(normalized_name, payload)
    stride = int(payload.get("stride", 0) or 0)
    if stride <= 0:
        raise VertexLayoutError(f"{normalized_name}: invalid vertex stride {stride}")
    raw_fields = list(payload.get("elements", payload.get("fields", [])) or [])
    fields: list[VertexField] = []
    for raw_field in raw_fields:
        semantic_name = str(raw_field.get("semantic_name", "") or "").upper()
        semantic_index = int(raw_field.get("semantic_index", 0) or 0)
        if not semantic_name:
            semantic_name, semantic_index = split_semantic(str(raw_field.get("semantic", "") or ""))
        fmt = normalize_dxgi_format(str(raw_field.get("format", "") or ""))
        size = dxgi_format_size(fmt)
        if size <= 0:
            raise VertexLayoutError(
                f"{normalized_name}: {semantic_name}{semantic_index} has unknown format {fmt or '<empty>'}"
            )
        offset = int(raw_field.get("aligned_byte_offset", raw_field.get("offset", 0)) or 0)
        if offset < 0 or offset + size > stride:
            raise VertexLayoutError(
                f"{normalized_name}: {semantic_name}{semantic_index} byte range "
                f"[{offset}, {offset + size}) exceeds stride {stride}"
            )
        fields.append(VertexField(semantic_name, semantic_index, fmt, offset, size))
    physical_fields = _build_physical_fields(normalized_name, fields)
    return VertexSlotLayout(
        slot_name=normalized_name,
        slot_index=slot_index,
        stride=stride,
        fields=tuple(fields),
        physical_fields=physical_fields,
        resource_hash=str(payload.get("resource_hash", "") or ""),
        backing_hash=str(payload.get("backing_hash", "") or ""),
        source_buf=str(payload.get("source_buf", "") or ""),
        byte_offset=int(payload.get("byte_offset", 0) or 0),
        vertex_count=int(payload.get("vertex_count", 0) or 0),
    )


def build_vertex_layout(layout: dict) -> dict[str, VertexSlotLayout]:
    raw_buffers = dict(layout.get("buffers", layout.get("vertex_buffers", {})) or {})
    return {
        str(slot_name).lower(): build_slot_layout(str(slot_name), dict(payload or {}))
        for slot_name, payload in raw_buffers.items()
    }


def slots_share_source(left: VertexSlotLayout, right: VertexSlotLayout) -> bool:
    left_identity = left.source_identity
    right_identity = right.source_identity
    return left_identity is not None and left_identity == right_identity


def raw_word_attribute_name(slot_name: str, word_index: int) -> str:
    return f"bmc_raw_{str(slot_name).lower()}_u32_{int(word_index)}"


def semantic_component_attribute_name(
    slot_name: str,
    semantic_name: str,
    semantic_index: int,
    component_index: int,
) -> str:
    return (
        f"bmc_{str(slot_name).lower()}_{str(semantic_name).lower()}"
        f"{int(semantic_index)}_{int(component_index)}"
    )


def semantic_color_attribute_name(slot_name: str, semantic_name: str, semantic_index: int) -> str:
    return f"bmc_{str(slot_name).lower()}_{str(semantic_name).lower()}{int(semantic_index)}_color"


def split_semantic(semantic: str) -> tuple[str, int]:
    match = re.match(r"^([A-Z_]+?)(\d*)$", str(semantic or "").upper())
    if match is None:
        return str(semantic or "").upper(), 0
    return match.group(1), int(match.group(2) or 0)


def _build_physical_fields(slot_name: str, fields: list[VertexField]) -> tuple[PhysicalVertexField, ...]:
    grouped: dict[tuple[int, int], list[VertexField]] = {}
    for field in fields:
        grouped.setdefault((int(field.offset), int(field.size)), []).append(field)
    ranges = sorted(grouped)
    for range_index, (offset, size) in enumerate(ranges):
        end = offset + size
        for other_offset, other_size in ranges[range_index + 1:]:
            if other_offset >= end:
                break
            if other_offset != offset or other_size != size:
                raise VertexLayoutError(
                    f"{slot_name}: partially overlapping vertex fields "
                    f"[{offset}, {end}) and [{other_offset}, {other_offset + other_size})"
                )
    return tuple(
        PhysicalVertexField(
            offset=offset,
            size=size,
            aliases=tuple(sorted(aliases, key=_semantic_priority)),
        )
        for (offset, size), aliases in sorted(grouped.items())
    )


def _semantic_priority(field: VertexField) -> tuple[int, str, int, str]:
    semantic_name = str(field.semantic_name).upper()
    semantic_index = int(field.semantic_index)
    if semantic_name == "POSITION" and semantic_index == 0:
        rank = 0
    elif semantic_name == "NORMAL" and semantic_index == 0:
        rank = 1
    elif semantic_name == "TANGENT" and semantic_index == 0:
        rank = 2
    elif semantic_name == "TEXCOORD" and semantic_index in {0, 1}:
        rank = 3
    elif semantic_name == "COLOR":
        rank = 4
    elif semantic_name == "TEXCOORD" and semantic_index == 4:
        rank = 5
    elif semantic_name == "TEXCOORD":
        rank = 6
    elif semantic_name == "BLENDWEIGHTS":
        rank = 7
    elif semantic_name == "BLENDINDICES":
        rank = 8
    else:
        rank = 9
    return rank, semantic_name, semantic_index, str(field.format)


def _slot_index(slot_name: str, payload: dict) -> int:
    match = re.fullmatch(r"vb(\d+)", str(slot_name).lower())
    if match is not None:
        return int(match.group(1))
    return int(payload.get("slot_index", payload.get("input_slot", -1)) or -1)
