"""Vertex field packing helpers for BoneMerge export buffers."""

from __future__ import annotations

import math
import struct
from typing import Iterable

from .vertex_layout_codec import dxgi_format_size, dxgi_format_spec


_STRUCT_CODES = {
    "u1": "B",
    "i1": "b",
    "<u2": "H",
    "<i2": "h",
    "<f2": "e",
    "<u4": "I",
    "<i4": "i",
    "<f4": "f",
}

_INTEGER_LIMITS = {
    "u1": (0, 0xFF),
    "i1": (-0x80, 0x7F),
    "<u2": (0, 0xFFFF),
    "<i2": (-0x8000, 0x7FFF),
    "<u4": (0, 0xFFFFFFFF),
    "<i4": (-0x80000000, 0x7FFFFFFF),
}


def pack_vertex_format(fmt: str, values: Iterable[float | int]) -> bytes:
    """Pack one vertex element according to a DXGI-like FrameAnalysis format."""

    spec = dxgi_format_spec(fmt)
    if spec is None:
        raise ValueError(f"Unsupported vertex format: {fmt}")
    items = list(values)
    component_count = int(spec.component_count)
    components = [_component(items, index) for index in range(component_count)]
    if spec.conversion == "unorm8":
        packed_values = [_unorm_to_int(value, 0xFF) for value in components]
    elif spec.conversion == "unorm16":
        packed_values = [_unorm_to_int(value, 0xFFFF) for value in components]
    elif spec.conversion == "snorm8":
        packed_values = [_snorm_to_int(value, 0x7F) for value in components]
    elif spec.conversion == "snorm16":
        packed_values = [_snorm_to_int(value, 0x7FFF) for value in components]
    elif spec.dtype in _INTEGER_LIMITS:
        minimum, maximum = _INTEGER_LIMITS[spec.dtype]
        packed_values = _uint_components(components, component_count, minimum, maximum)
    else:
        packed_values = [float(value) for value in components]
    return struct.pack(
        f"<{component_count}{_STRUCT_CODES[spec.dtype]}",
        *packed_values,
    )


def pack_into_vertex_format(buffer: bytearray, offset: int, fmt: str, values: Iterable[float | int]) -> None:
    data = pack_vertex_format(fmt, values)
    end_offset = int(offset) + len(data)
    if int(offset) < 0 or end_offset > len(buffer):
        raise ValueError(f"Packed field exceeds target buffer: offset={offset}, size={len(data)}, buffer={len(buffer)}")
    buffer[int(offset):end_offset] = data


def unpack_vertex_format(fmt: str, data: bytes | bytearray | memoryview, offset: int = 0) -> tuple[float, ...]:
    """Unpack one vertex element according to a DXGI-like FrameAnalysis format."""

    spec = dxgi_format_spec(fmt)
    if spec is None:
        raise ValueError(f"Unsupported vertex format: {fmt}")
    raw_values = struct.unpack_from(
        f"<{int(spec.component_count)}{_STRUCT_CODES[spec.dtype]}",
        data,
        offset,
    )
    if spec.conversion == "unorm8":
        return tuple(float(value) / 255.0 for value in raw_values)
    if spec.conversion == "unorm16":
        return tuple(float(value) / 65535.0 for value in raw_values)
    if spec.conversion == "snorm8":
        return tuple(max(float(value) / 127.0, -1.0) for value in raw_values)
    if spec.conversion == "snorm16":
        return tuple(max(float(value) / 32767.0, -1.0) for value in raw_values)
    return tuple(float(value) for value in raw_values)


def format_size(fmt: str) -> int:
    size = dxgi_format_size(fmt)
    if size <= 0:
        raise ValueError(f"Unsupported vertex format: {fmt}")
    return int(size)


def encode_game_packed_normal(normal: tuple[float, float, float]) -> int:
    """Encode only the normal portion of this game's packed NORMAL0 storage.

    This helper is kept for import/debug round trips.  Production export of an
    R32_FLOAT NORMAL0 field should use :func:`encode_game_packed_tangent_frame`
    so the shader receives both the normal and tangent frame it expects.
    """

    quant_x, quant_y = _encode_octahedral_signed10(normal)
    return 0x40000000 | quant_x | (quant_y << 10)


def encode_game_packed_tangent_frame(
    normal: tuple[float, float, float],
    tangent: tuple[float, float, float],
    handedness: float,
) -> int:
    """Encode the normal and tangent-frame roll used by b30-style shaders."""

    quant_x, quant_y = _encode_octahedral_signed10(normal)
    decoded_normal = decode_game_packed_normal(0x40000000 | quant_x | (quant_y << 10))
    basis_u, basis_v = _packed_tangent_basis(decoded_normal)
    tangent_value = _normalize3(tangent)
    tangent_value = _orthogonalize(tangent_value, decoded_normal)

    roll_cos = _dot3(tangent_value, basis_u)
    roll_sin = _dot3(tangent_value, basis_v)
    denom = abs(roll_cos) + abs(roll_sin)
    roll = 0.0 if denom <= 1e-12 else roll_sin / denom
    quant_roll = _signed10_storage(_snorm_to_int(roll, 511))
    sign_bit = 0x80000000 if float(handedness) >= 0.0 else 0
    return sign_bit | 0x40000000 | quant_x | (quant_y << 10) | (quant_roll << 20)


def decode_game_packed_normal(value: int) -> tuple[float, float, float]:
    packed = int(value) & 0xFFFFFFFF
    if not (packed & 0x40000000):
        return (0.0, 0.0, 1.0)
    x_value = _signed10_to_float(packed & 0x3FF)
    y_value = _signed10_to_float((packed >> 10) & 0x3FF)
    z_value = 1.0 - abs(x_value) - abs(y_value)
    normal_x = x_value
    normal_y = y_value
    if z_value < 0.0:
        normal_x = (1.0 - abs(y_value)) * (1.0 if x_value >= 0.0 else -1.0)
        normal_y = (1.0 - abs(x_value)) * (1.0 if y_value >= 0.0 else -1.0)
    return _normalize3((normal_x, normal_y, z_value))


def decode_game_packed_tangent_frame(value: int) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    packed = int(value) & 0xFFFFFFFF
    normal = decode_game_packed_normal(packed)
    if not (packed & 0x40000000):
        return normal, (1.0, 0.0, 0.0), 0.0
    roll = _signed10_to_float((packed >> 20) & 0x3FF)
    basis_u, basis_v = _packed_tangent_basis(normal)
    roll_sign = -1.0 if roll < 0.0 else 1.0
    roll_x = 1.0 - abs(roll)
    roll_y = roll_sign * (1.0 - abs(roll_x))
    length = math.sqrt(roll_x * roll_x + roll_y * roll_y)
    if length <= 1e-12:
        tangent = basis_u
    else:
        cosine = roll_x / length
        sine = roll_y / length
        tangent = _normalize3(
            (
                cosine * basis_u[0] + sine * basis_v[0],
                cosine * basis_u[1] + sine * basis_v[1],
                cosine * basis_u[2] + sine * basis_v[2],
            )
        )
    handedness = 1.0 if (packed & 0x80000000) else -1.0
    return normal, tangent, handedness


def _encode_octahedral_signed10(normal: tuple[float, float, float]) -> tuple[int, int]:
    x_value, y_value, z_value = _normalize3(normal)
    inv_l1 = 1.0 / max(abs(x_value) + abs(y_value) + abs(z_value), 1e-12)
    oct_x = x_value * inv_l1
    oct_y = y_value * inv_l1
    if z_value < 0.0:
        folded_x = (1.0 - abs(oct_y)) * (1.0 if oct_x >= 0.0 else -1.0)
        folded_y = (1.0 - abs(oct_x)) * (1.0 if oct_y >= 0.0 else -1.0)
        oct_x, oct_y = folded_x, folded_y
    return (
        _signed10_storage(_snorm_to_int(oct_x, 511)),
        _signed10_storage(_snorm_to_int(oct_y, 511)),
    )


def _packed_tangent_basis(normal: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    normal = _normalize3(normal)
    nx, ny, nz = normal
    basis_u = (ny - nz, nz - nx, nx - ny)
    projection = _dot3(basis_u, normal)
    basis_u = _normalize3((basis_u[0] - projection, basis_u[1] - projection, basis_u[2] - projection))
    if _length3(basis_u) <= 1e-6:
        basis_u = _fallback_tangent(normal)
    basis_v = _normalize3(_cross3(normal, basis_u))
    return basis_u, basis_v


def _orthogonalize(vector: tuple[float, float, float], normal: tuple[float, float, float]) -> tuple[float, float, float]:
    normal = _normalize3(normal)
    projected = _dot3(vector, normal)
    tangent = (
        vector[0] - projected * normal[0],
        vector[1] - projected * normal[1],
        vector[2] - projected * normal[2],
    )
    if _length3(tangent) <= 1e-6:
        return _fallback_tangent(normal)
    return _normalize3(tangent)


def _fallback_tangent(normal: tuple[float, float, float]) -> tuple[float, float, float]:
    axis = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0)
    tangent = _cross3(axis, normal)
    if _length3(tangent) <= 1e-6:
        tangent = _cross3((0.0, 0.0, 1.0), normal)
    return _normalize3(tangent)


def _signed10_to_float(value: int) -> float:
    signed = int(value) - 1024 if int(value) >= 512 else int(value)
    return float(signed) / 511.0


def _dot3(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return float(left[0]) * float(right[0]) + float(left[1]) * float(right[1]) + float(left[2]) * float(right[2])


def _cross3(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _length3(value: tuple[float, float, float]) -> float:
    return math.sqrt(float(value[0]) * float(value[0]) + float(value[1]) * float(value[1]) + float(value[2]) * float(value[2]))


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
