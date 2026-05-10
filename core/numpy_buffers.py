"""Shared NumPy helpers for binary buffer and point-cloud hot paths."""

from __future__ import annotations

import os

import numpy as np


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
    return int(np.dtype(dtype_name).itemsize) * int(component_count)


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
    count = min(int(index_count), len(data) // int(stride))
    return np.frombuffer(data, dtype=np.dtype(dtype_name), count=count).astype(np.int64, copy=False).tolist()


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
    spec = dxgi_format_spec(fmt)
    if spec is None or int(stride) <= 0 or int(offset) < 0:
        return None
    dtype_name, component_count, conversion = spec
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


def read_interleaved_fields(
    data: bytes,
    vertex_ids,
    *,
    stride: int,
    fields: list[tuple[str, int, str, bool]],
    vertex_count: int | None = None,
) -> dict[str, object] | None:
    if int(stride) <= 0:
        return None
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


def _convert_interleaved_values(values, conversion):
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
    field_specs = [(str(name), int(offset), str(fmt), bool(converted)) for name, offset, fmt in fields]
    return read_interleaved_fields(
        data,
        vertex_ids,
        stride=int(stride),
        fields=field_specs,
        vertex_count=int(vertex_count),
    )


def assign_bytes(target, offset: int, values) -> None:
    array = np.ascontiguousarray(values)
    if array.size == 0:
        return
    byte_view = array.view(np.uint8).reshape(int(array.shape[0]), -1)
    target[:, int(offset):int(offset) + byte_view.shape[1]] = byte_view


def foreach_get_array(collection, attribute_name: str, *, dtype="float64", shape: tuple[int, ...] | None = None):
    getter = getattr(collection, "foreach_get", None)
    if not callable(getter):
        raise ValueError(f"collection does not support foreach_get({attribute_name})")
    count = len(collection)
    flat_count = int(count)
    if shape:
        for dimension in shape:
            flat_count *= int(dimension)
    values = np.empty(flat_count, dtype=np.dtype(dtype))
    getter(attribute_name, values)
    if shape:
        return values.reshape((int(count), *tuple(int(dimension) for dimension in shape)))
    return values


def object_attribute_array(items, attribute_name: str, *, dtype="int64", start: int = 0, end: int | None = None):
    item_count = len(items)
    start_index = max(0, int(start))
    end_index = item_count if end is None else min(item_count, int(end))
    if end_index < start_index:
        end_index = start_index
    return np.fromiter(
        (getattr(items[index], attribute_name) for index in range(start_index, end_index)),
        dtype=np.dtype(dtype),
        count=end_index - start_index,
    )


def normalize_rows(values, *, columns: int = 3, dtype="float32"):
    vectors = np.asarray(values, dtype=np.dtype(dtype))
    if vectors.size == 0:
        return vectors.reshape((0, int(columns)))
    vectors = vectors[:, : int(columns)]
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths = np.where(lengths <= 1.0e-12, 1.0, lengths)
    return vectors / lengths


def point_bounds(points):
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return None
    values = values.reshape((-1, 3))
    return values.min(axis=0), values.max(axis=0)


def cell_key_array(points, tolerance: float):
    values = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    inverse = 1.0 / max(float(tolerance), 1.0e-6)
    return np.floor(values * inverse).astype(np.int64)


def cell_key_set_from_array(cell_keys) -> frozenset[tuple[int, int, int]]:
    keys = np.asarray(cell_keys, dtype=np.int64).reshape((-1, 3))
    if keys.size == 0:
        return frozenset()
    unique_keys = np.unique(keys, axis=0)
    return frozenset((int(x), int(y), int(z)) for x, y, z in unique_keys)


def expanded_cell_key_set(cell_keys) -> frozenset[tuple[int, int, int]]:
    expanded: set[tuple[int, int, int]] = set()
    for key in cell_keys or ():
        base_x, base_y, base_z = key
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for offset_z in (-1, 0, 1):
                    expanded.add((int(base_x) + offset_x, int(base_y) + offset_y, int(base_z) + offset_z))
    return frozenset(expanded)


def max_interleaved_uint4(data: bytes, *, stride: int, offset: int, fmt: str, vertex_count: int) -> int:
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
    values = np.asarray(positions, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    delta = maxs - mins
    return float(np.sqrt(np.dot(delta, delta)))
