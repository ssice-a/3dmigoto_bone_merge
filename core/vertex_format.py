"""Vertex field packing helpers for BoneMerge export buffers."""

from __future__ import annotations

import math
import struct
from typing import Iterable


def pack_vertex_format(fmt: str, values: Iterable[float | int]) -> bytes:
    """Pack one vertex element according to a DXGI-like FrameAnalysis format."""

    normalized_fmt = _normalize_format(fmt)
    items = list(values)
    if normalized_fmt == "R32_FLOAT":
        return struct.pack("<f", float(_component(items, 0)))
    if normalized_fmt == "R32G32_FLOAT":
        return struct.pack("<2f", *_float_components(items, 2))
    if normalized_fmt == "R32G32B32_FLOAT":
        return struct.pack("<3f", *_float_components(items, 3))
    if normalized_fmt == "R32G32B32A32_FLOAT":
        return struct.pack("<4f", *_float_components(items, 4))
    if normalized_fmt == "R32G32B32A32_UINT":
        return struct.pack("<4I", *_uint_components(items, 4, 0, 0xFFFFFFFF))
    if normalized_fmt == "R16G16B16A16_UNORM":
        return struct.pack("<4H", *[_unorm_to_int(value, 65535) for value in _float_components(items, 4)])
    if normalized_fmt == "R8G8B8A8_UINT":
        return struct.pack("<4B", *_uint_components(items, 4, 0, 0xFF))
    if normalized_fmt == "R8G8B8A8_SNORM":
        return struct.pack("<4B", *[_signed_byte_to_storage(_snorm_to_int(value, 127)) for value in _float_components(items, 4)])
    if normalized_fmt == "R8G8B8A8_UNORM":
        return struct.pack("<4B", *[_unorm_to_int(value, 255) for value in _float_components(items, 4)])
    raise ValueError(f"Unsupported vertex format: {fmt}")


def pack_into_vertex_format(buffer: bytearray, offset: int, fmt: str, values: Iterable[float | int]) -> None:
    data = pack_vertex_format(fmt, values)
    end_offset = int(offset) + len(data)
    if int(offset) < 0 or end_offset > len(buffer):
        raise ValueError(f"Packed field exceeds target buffer: offset={offset}, size={len(data)}, buffer={len(buffer)}")
    buffer[int(offset):end_offset] = data


def format_size(fmt: str) -> int:
    normalized_fmt = _normalize_format(fmt)
    sizes = {
        "R32_FLOAT": 4,
        "R32G32_FLOAT": 8,
        "R32G32B32_FLOAT": 12,
        "R32G32B32A32_FLOAT": 16,
        "R32G32B32A32_UINT": 16,
        "R16G16B16A16_UNORM": 8,
        "R8G8B8A8_UINT": 4,
        "R8G8B8A8_SNORM": 4,
        "R8G8B8A8_UNORM": 4,
    }
    if normalized_fmt not in sizes:
        raise ValueError(f"Unsupported vertex format: {fmt}")
    return sizes[normalized_fmt]


def encode_game_packed_normal(normal: tuple[float, float, float]) -> int:
    """Encode a normal into this game's packed NORMAL0 R32_FLOAT storage.

    FrameAnalysis labels the field as R32_FLOAT, but the shader treats the bits
    as a packed octahedral-ish 10-bit x/y normal with bit 30 used as the valid
    marker. We return the raw uint32 bit pattern that must be written.
    """

    x_value, y_value, z_value = _normalize3(normal)
    inv_l1 = 1.0 / max(abs(x_value) + abs(y_value) + abs(z_value), 1e-12)
    oct_x = x_value * inv_l1
    oct_y = y_value * inv_l1
    if z_value < 0.0:
        folded_x = (1.0 - abs(oct_y)) * (1.0 if oct_x >= 0.0 else -1.0)
        folded_y = (1.0 - abs(oct_x)) * (1.0 if oct_y >= 0.0 else -1.0)
        oct_x, oct_y = folded_x, folded_y

    quant_x = _signed10_storage(_snorm_to_int(oct_x, 511))
    quant_y = _signed10_storage(_snorm_to_int(oct_y, 511))
    return 0x40000000 | quant_x | (quant_y << 10)


def _normalize_format(fmt: str) -> str:
    normalized = str(fmt or "").upper()
    if normalized.startswith("DXGI_FORMAT_"):
        normalized = normalized[len("DXGI_FORMAT_"):]
    return normalized


def _float_components(values: list[float | int], count: int) -> list[float]:
    return [float(_component(values, index)) for index in range(count)]


def _uint_components(values: list[float | int], count: int, min_value: int, max_value: int) -> list[int]:
    return [_clamp_int(round(float(_component(values, index))), min_value, max_value) for index in range(count)]


def _component(values: list[float | int], index: int) -> float | int:
    if index >= len(values):
        return 0
    return values[index]


def _unorm_to_int(value: float, max_value: int) -> int:
    return _clamp_int(round(_clamp_float(float(value), 0.0, 1.0) * int(max_value)), 0, int(max_value))


def _snorm_to_int(value: float, max_abs: int) -> int:
    return _clamp_int(round(_clamp_float(float(value), -1.0, 1.0) * int(max_abs)), -int(max_abs), int(max_abs))


def _signed_byte_to_storage(value: int) -> int:
    return int(value) & 0xFF


def _signed10_storage(value: int) -> int:
    return int(value) & 0x3FF


def _normalize3(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    x_value, y_value, z_value = (float(vector[0]), float(vector[1]), float(vector[2]))
    length = math.sqrt(x_value * x_value + y_value * y_value + z_value * z_value)
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (x_value / length, y_value / length, z_value / length)


def _clamp_float(value: float, min_value: float, max_value: float) -> float:
    return max(float(min_value), min(float(max_value), float(value)))


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(int(min_value), min(int(max_value), int(value)))
