"""Small reusable spatial-grid helpers for point-cloud matching."""

from __future__ import annotations

from math import floor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def cell_key(position: tuple[float, float, float], tolerance: float) -> tuple[int, int, int]:
    inverse = 1.0 / max(float(tolerance), 1.0e-6)
    return (
        floor(float(position[0]) * inverse),
        floor(float(position[1]) * inverse),
        floor(float(position[2]) * inverse),
    )


def neighbor_keys(base_key: tuple[int, int, int]):
    base_x, base_y, base_z = base_key
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                yield base_x + offset_x, base_y + offset_y, base_z + offset_z


def expanded_neighbor_cell_keys(cell_keys: Iterable[tuple[int, int, int]]) -> frozenset[tuple[int, int, int]]:
    expanded: set[tuple[int, int, int]] = set()
    for key in cell_keys:
        expanded.update(neighbor_keys(key))
    return frozenset(expanded)


def build_spatial_hash(
    items: Iterable[T],
    position_getter: Callable[[T], tuple[float, float, float]],
    tolerance: float,
) -> dict[tuple[int, int, int], list[T]]:
    spatial_hash: dict[tuple[int, int, int], list[T]] = {}
    for item in items:
        spatial_hash.setdefault(cell_key(position_getter(item), tolerance), []).append(item)
    return spatial_hash
