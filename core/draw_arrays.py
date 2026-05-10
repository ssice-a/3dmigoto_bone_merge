"""Shared NumPy array helpers for draw-indexed geometry paths."""

from __future__ import annotations

import os

from .numpy_compat import optional_numpy
from .numpy_buffers import index_format_spec, positions_diag


_WEIGHT_EPSILON = 1.0e-5


def require_numpy():
    np = optional_numpy()
    if np is None:  # pragma: no cover - numpy is a hard dependency in normal builds
        raise RuntimeError("NumPy is required for Bone Merge buffer processing")
    return np


def read_index_array(path: str, fmt: str, index_count: int, *, byte_offset: int = 0, first_index: int = 0):
    spec = index_format_spec(fmt)
    if spec is None:
        raise ValueError(f"Unsupported IB format: {fmt}")
    if int(index_count) <= 0:
        return require_numpy().zeros((0,), dtype="int64")
    if not path or not os.path.exists(path):
        raise ValueError(f"IB buffer is missing: {path}")

    np = require_numpy()
    dtype_name, stride = spec
    read_size = int(index_count) * int(stride)
    start_offset = int(byte_offset) + int(first_index) * int(stride)
    expected_end = start_offset + read_size
    if os.path.getsize(path) < expected_end:
        raise ValueError(f"IB buffer is shorter than expected: {path}")
    with open(path, "rb") as file_handle:
        file_handle.seek(start_offset)
        data = file_handle.read(read_size)
    if len(data) < read_size:
        raise ValueError(f"IB buffer is shorter than expected: {path}")
    return np.frombuffer(data, dtype=np.dtype(dtype_name), count=int(index_count)).astype(np.int64, copy=False)


def build_topology_arrays(indices):
    np = require_numpy()
    values = np.asarray(indices, dtype=np.int64).reshape((-1,))
    triangle_count = int(values.size // 3)
    if triangle_count <= 0:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
        )
    source_triangles = values[: triangle_count * 3].reshape((triangle_count, 3))
    original_vertex_ids, inverse = np.unique(source_triangles.reshape((-1,)), return_inverse=True)
    triangles = inverse.reshape((triangle_count, 3)).astype(np.int64, copy=False)
    return triangles, original_vertex_ids.astype(np.int64, copy=False), source_triangles


def row_tuple_list(values, *, dtype=None):
    np = require_numpy()
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        return [array.item()]
    if array.ndim == 1:
        return [int(value) if array.dtype.kind in {"i", "u"} else float(value) for value in array.tolist()]
    return [tuple(_python_scalar(component) for component in row) for row in array.tolist()]


def used_skin_slots(blend_indices, blend_weights, *, epsilon: float = _WEIGHT_EPSILON) -> list[int]:
    np = require_numpy()
    indices = np.asarray(blend_indices, dtype=np.int64)
    weights = np.asarray(blend_weights, dtype=np.float64)
    if indices.size == 0 or weights.size == 0:
        return []
    indices = indices.reshape((-1, 4))
    weights = weights.reshape((-1, 4))
    valid = weights > float(epsilon)
    used = indices[valid]
    used = used[used >= 0]
    if used.size == 0:
        return []
    return [int(value) for value in np.unique(used)]


def skin_influence_histogram(blend_weights, *, epsilon: float = _WEIGHT_EPSILON) -> tuple[int, list[int]]:
    np = require_numpy()
    weights = np.asarray(blend_weights, dtype=np.float64)
    if weights.size == 0:
        return 0, [0, 0, 0, 0, 0]
    counts = (weights.reshape((-1, 4)) > float(epsilon)).sum(axis=1).astype(np.int64)
    histogram = np.bincount(np.minimum(counts, 4), minlength=5)
    return int(np.count_nonzero(counts)), [int(value) for value in histogram[:5]]


def skin_signature(positions, blend_indices, blend_weights, *, declared_used_slots=None, epsilon: float = _WEIGHT_EPSILON) -> dict:
    np = require_numpy()
    declared_slots = sorted({int(value) for value in (declared_used_slots or []) if int(value) >= 0})
    used_slots = used_skin_slots(blend_indices, blend_weights, epsilon=epsilon)
    if not used_slots:
        used_slots = declared_slots
    weighted_vertex_count, influence_hist = skin_influence_histogram(blend_weights, epsilon=epsilon)

    position_values = np.asarray(positions, dtype=np.float64).reshape((-1, 3)) if len(positions) else np.zeros((0, 3), dtype=np.float64)
    if position_values.size:
        center_values = position_values.mean(axis=0)
        center = [float(center_values[0]), float(center_values[1]), float(center_values[2])]
        diag = positions_diag(position_values)
    else:
        center = [0.0, 0.0, 0.0]
        diag = 0.0
    return {
        "slot_count": int(len(used_slots)),
        "used_slots": [int(value) for value in used_slots],
        "slot_min": int(min(used_slots)) if used_slots else -1,
        "slot_max": int(max(used_slots)) if used_slots else -1,
        "slot_span": int(max(used_slots) - min(used_slots) + 1) if used_slots else 0,
        "vertex_count": int(position_values.shape[0]),
        "weighted_vertex_count": int(weighted_vertex_count),
        "influence_hist": [int(value) for value in (influence_hist + [0, 0, 0, 0, 0])[:5]],
        "center": center,
        "diag": float(diag),
    }


def _python_scalar(value):
    try:
        return value.item()
    except AttributeError:
        return value
