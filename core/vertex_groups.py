"""Shared helpers for numeric/global Blender vertex groups."""

from __future__ import annotations

import re

_NUMERIC_GROUP_RE = re.compile(r"^\d+$")


def parse_numeric_group_name(group_name: str) -> int | None:
    raw_name = str(group_name or "").strip()
    if not _NUMERIC_GROUP_RE.match(raw_name):
        return None
    return int(raw_name)


def build_group_index_to_global_map(mesh_obj) -> dict[int, int]:
    group_index_to_global: dict[int, int] = {}
    for vertex_group in getattr(mesh_obj, "vertex_groups", []) or []:
        numeric_group = parse_numeric_group_name(getattr(vertex_group, "name", ""))
        if numeric_group is None:
            continue
        group_index_to_global[int(vertex_group.index)] = numeric_group
    return group_index_to_global


def iter_weighted_global_assignments(mesh_obj):
    group_index_to_global = build_group_index_to_global_map(mesh_obj)
    vertices = getattr(getattr(mesh_obj, "data", None), "vertices", []) or []
    for vertex in vertices:
        vertex_index = int(getattr(vertex, "index", 0))
        for group_element in getattr(vertex, "groups", []) or []:
            global_group = group_index_to_global.get(int(getattr(group_element, "group", -1)))
            if global_group is None:
                continue
            weight = float(getattr(group_element, "weight", 0.0))
            if weight <= 0.0:
                continue
            yield global_group, vertex_index, weight


def collect_weighted_numeric_vertex_groups(mesh_obj) -> set[int]:
    return {
        int(global_group)
        for global_group, _vertex_index, _weight in iter_weighted_global_assignments(mesh_obj)
    }
