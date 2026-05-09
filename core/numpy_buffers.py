"""Shared NumPy helpers for binary buffer and point-cloud hot paths."""

from __future__ import annotations

import os

from .numpy_compat import optional_numpy


def normalize_dxgi_format(fmt: str) -> str:
    normalized = str(fmt or "").upper()
    if normalized.startswith("DXGI_FORMAT_"):
        normalized = normalized[len("DXGI_FORMAT_"):]
    return normalized


def dxgi_format_size(fmt: str) -> int:
    spec = dxgi_format_spec(fmt)
    if spec is None:
        return 0
    dtype_name, component_count, _conversion = spec
    np = optional_numpy()
    if np is not None:
        return int(np.dtype(dtype_name).itemsize) * int(component_count)
    fallback_sizes = {
        "u1": 1,
        "<u2": 2,
        "<u4": 4,
        "<f4": 4,
    }
    return int(fallback_sizes.get(dtype_name, 0)) * int(component_count)


def dxgi_format_spec(fmt: str):
    normalized = normalize_dxgi_format(fmt)
    return {
        "R32_FLOAT": ("<f4", 1, None),
        "R32_UINT": ("<u4", 1, None),
        "R32G32_FLOAT": ("<f4", 2, None),
        "R32G32B32_FLOAT": ("<f4", 3, None),
        "R32G32B32A32_FLOAT": ("<f4", 4, None),
        "R32G32B32A32_UINT": ("<u4", 4, None),
        "R16G16B16A16_UNORM": ("<u2", 4, "unorm16"),
        "R8G8B8A8_UINT": ("u1", 4, None),
        "R8G8B8A8_SNORM": ("u1", 4, "snorm8"),
        "R8G8B8A8_UNORM": ("u1", 4, "unorm8"),
    }.get(normalized)


def index_format_spec(fmt: str):
    normalized = normalize_dxgi_format(fmt)
    if "R16_UINT" in normalized:
        return "<u2", 2
    if "R32_UINT" in normalized:
        return "<u4", 4
    return None


def read_index_file(path: str, fmt: str, index_count: int, *, byte_offset: int = 0, first_index: int = 0) -> list[int]:
    spec = index_format_spec(fmt)
    if spec is None or not path or not os.path.exists(path) or int(index_count) <= 0:
        return []
    dtype_name, stride = spec
    read_size = int(index_count) * int(stride)
    with open(path, "rb") as file_handle:
        file_handle.seek(int(byte_offset) + int(first_index) * int(stride))
        data = file_handle.read(read_size)
    np = optional_numpy()
    if np is not None:
        try:
            count = min(int(index_count), len(data) // int(stride))
            return np.frombuffer(data, dtype=np.dtype(dtype_name), count=count).astype(np.int64, copy=False).tolist()
        except Exception:
            pass
    import struct

    unpack = "<H" if int(stride) == 2 else "<I"
    return [
        int(record[0])
        for record in struct.iter_unpack(unpack, data[: len(data) - (len(data) % int(stride))])
    ]


def read_interleaved_field(
    data: bytes,
    vertex_ids,
    *,
    stride: int,
    offset: int,
    fmt: str,
    vertex_count: int | None = None,
    converted: bool = True,
):
    np = optional_numpy()
    if np is None:
        return None
    spec = dxgi_format_spec(fmt)
    if spec is None or int(stride) <= 0 or int(offset) < 0:
        return None
    dtype_name, component_count, conversion = spec
    try:
        field_dtype = np.dtype(dtype_name)
        count = int(vertex_count) if vertex_count is not None else len(data) // int(stride)
        record_dtype = np.dtype(
            {
                "names": ["field"],
                "formats": [(field_dtype, (int(component_count),))],
                "offsets": [int(offset)],
                "itemsize": int(stride),
            }
        )
        records = np.frombuffer(data, dtype=record_dtype, count=count)
        indices = np.asarray(vertex_ids, dtype=np.intp)
        values = records["field"][indices]
        if not converted:
            return values
        if conversion == "unorm16":
            return values.astype(np.float64) / 65535.0
        if conversion == "unorm8":
            return values.astype(np.float64) / 255.0
        if conversion == "snorm8":
            signed = values.astype(np.int16)
            return np.where(signed >= 128, signed - 256, signed).astype(np.float64) / 127.0
        if values.dtype.kind in {"u", "i"}:
            return values.copy()
        with np.errstate(invalid="ignore", over="ignore"):
            return values.astype(np.float64)
    except Exception:
        return None


def read_interleaved_fields(
    data: bytes,
    vertex_ids,
    *,
    stride: int,
    fields: list[tuple[str, int, str, bool]],
    vertex_count: int | None = None,
) -> dict[str, object] | None:
    np = optional_numpy()
    if np is None or int(stride) <= 0:
        return None
    try:
        names = []
        formats = []
        offsets = []
        conversions = {}
        for name, offset, fmt, converted in fields:
            spec = dxgi_format_spec(fmt)
            if spec is None or int(offset) < 0:
                return None
            dtype_name, component_count, conversion = spec
            names.append(str(name))
            formats.append((np.dtype(dtype_name), (int(component_count),)))
            offsets.append(int(offset))
            conversions[str(name)] = conversion if bool(converted) else None
        count = int(vertex_count) if vertex_count is not None else len(data) // int(stride)
        record_dtype = np.dtype(
            {
                "names": names,
                "formats": formats,
                "offsets": offsets,
                "itemsize": int(stride),
            }
        )
        records = np.frombuffer(data, dtype=record_dtype, count=count)
        indices = np.asarray(vertex_ids, dtype=np.intp)
        return {
            name: _convert_interleaved_values(records[name][indices], conversions[name])
            for name in names
        }
    except Exception:
        return None


def _convert_interleaved_values(values, conversion):
    np = optional_numpy()
    if np is None:
        return values
    if conversion == "unorm16":
        return values.astype(np.float64) / 65535.0
    if conversion == "unorm8":
        return values.astype(np.float64) / 255.0
    if conversion == "snorm8":
        signed = values.astype(np.int16)
        return np.where(signed >= 128, signed - 256, signed).astype(np.float64) / 127.0
    if conversion is None:
        return values.copy()
    if values.dtype.kind in {"u", "i"}:
        return values.copy()
    with np.errstate(invalid="ignore", over="ignore"):
        return values.astype(np.float64)


def read_interleaved_fields_from_file(
    path: str,
    vertex_ids,
    *,
    byte_offset: int,
    vertex_count: int,
    stride: int,
    fields: list[tuple[str, int, str]],
    converted: bool = True,
) -> dict[str, object] | None:
    if not path or not os.path.exists(path) or int(vertex_count) <= 0 or int(stride) <= 0:
        return None
    try:
        with open(path, "rb") as file_handle:
            file_handle.seek(int(byte_offset))
            data = file_handle.read(int(vertex_count) * int(stride))
    except OSError:
        return None
    result = {}
    field_specs = [(str(name), int(offset), str(fmt), bool(converted)) for name, offset, fmt in fields]
    batch = read_interleaved_fields(
        data,
        vertex_ids,
        stride=int(stride),
        fields=field_specs,
        vertex_count=int(vertex_count),
    )
    if batch is not None:
        return batch
    for name, offset, fmt, item_converted in field_specs:
        values = read_interleaved_field(
            data,
            vertex_ids,
            stride=int(stride),
            offset=int(offset),
            fmt=str(fmt),
            vertex_count=int(vertex_count),
            converted=bool(item_converted),
        )
        if values is None:
            return None
        result[str(name)] = values
    return result


def assign_bytes(target, offset: int, values) -> None:
    np = optional_numpy()
    if np is None:
        raise ValueError("numpy is required for byte assignment")
    array = np.ascontiguousarray(values)
    if array.size == 0:
        return
    byte_view = array.view(np.uint8).reshape(int(array.shape[0]), -1)
    target[:, int(offset):int(offset) + byte_view.shape[1]] = byte_view


def max_interleaved_uint4(data: bytes, *, stride: int, offset: int, fmt: str, vertex_count: int) -> int:
    np = optional_numpy()
    if np is not None:
        values = read_interleaved_field(
            data,
            range(int(vertex_count)),
            stride=int(stride),
            offset=int(offset),
            fmt=str(fmt),
            vertex_count=int(vertex_count),
            converted=False,
        )
        if values is not None and values.size:
            return int(values.max())
    return -1


def positions_diag(positions) -> float:
    try:
        if len(positions) <= 0:
            return 0.0
    except TypeError:
        positions = list(positions)
        if not positions:
            return 0.0
    np = optional_numpy()
    if np is not None:
        try:
            values = np.asarray(positions, dtype=np.float64)
            if values.size == 0:
                return 0.0
            mins = values.min(axis=0)
            maxs = values.max(axis=0)
            delta = maxs - mins
            return float(np.sqrt(np.dot(delta, delta)))
        except Exception:
            pass
    import math

    min_x = min(position[0] for position in positions)
    min_y = min(position[1] for position in positions)
    min_z = min(position[2] for position in positions)
    max_x = max(position[0] for position in positions)
    max_y = max(position[1] for position in positions)
    max_z = max(position[2] for position in positions)
    return math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2)
