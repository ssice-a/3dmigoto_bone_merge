"""Independent seam-based vertex-group matcher."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .spatial_index import (
    build_spatial_hash as _generic_build_spatial_hash,
    cell_key as _generic_cell_key,
    neighbor_keys as _generic_neighbor_keys,
)
from .draw_arrays import require_numpy
from .numpy_buffers import expanded_cell_key_set, point_bounds

try:
    from mathutils import Vector as _MathutilsVector
except Exception:
    _MathutilsVector = None


_MATCH_TOLERANCE = 0.0015
_WEIGHT_TOLERANCE = 0.001
_MIN_VERTEX_PAIRS = 4
_MIN_MAPPING_VOTES = 3
_MAX_AVERAGE_DISTANCE = 0.001
_OBJECT_BBOX_GAP_TOLERANCE = 0.01
_GROUP_BBOX_GAP_TOLERANCE = 0.003
_WEIGHT_EPSILON = 1.0e-8
_WEIGHT_FLOOR = 1.0e-4


@dataclass(frozen=True)
class SeamAliasRecord:
    src_object_name: str
    src_group: int
    dst_object_name: str
    dst_group: int
    votes: int
    score: float
    average_distance: float
    average_weight_difference: float


@dataclass(frozen=True)
class SeamBuildResult:
    aliases: tuple[SeamAliasRecord, ...]
    pair_summaries: tuple[str, ...]
    matched_pairs: int
    skipped_pairs: int


@dataclass(frozen=True)
class SeamApplyResult:
    updated_objects: int
    renamed_groups: int
    skipped_messages: tuple[str, ...]


@dataclass(frozen=True)
class SeamMergeResult:
    aliases: tuple[SeamAliasRecord, ...]
    pair_summaries: tuple[str, ...]
    matched_pairs: int
    skipped_pairs: int
    updated_objects: int
    renamed_groups: int
    skipped_messages: tuple[str, ...]


def build_and_apply_seam_mapping(mesh_objects) -> SeamMergeResult:
    total_start = time.perf_counter()
    perf: dict[str, float | int | list] = {}
    build_start = time.perf_counter()
    build_result = build_seam_mapping(mesh_objects, perf=perf)
    perf["build_total"] = time.perf_counter() - build_start
    apply_start = time.perf_counter()
    alias_payload = seam_aliases_to_payload(build_result.aliases)
    apply_result = apply_seam_mapping(mesh_objects, alias_payload)
    perf["apply"] = time.perf_counter() - apply_start
    perf["total"] = time.perf_counter() - total_start
    _print_seam_performance_report(perf, build_result)
    return SeamMergeResult(
        aliases=build_result.aliases,
        pair_summaries=build_result.pair_summaries,
        matched_pairs=build_result.matched_pairs,
        skipped_pairs=build_result.skipped_pairs,
        updated_objects=apply_result.updated_objects,
        renamed_groups=apply_result.renamed_groups,
        skipped_messages=apply_result.skipped_messages,
    )


def seam_aliases_to_payload(aliases: tuple[SeamAliasRecord, ...] | list[SeamAliasRecord]) -> list[dict]:
    return [
        {
            "enabled": True,
            "src_object_name": alias.src_object_name,
            "src_group": int(alias.src_group),
            "dst_object_name": alias.dst_object_name,
            "dst_group": int(alias.dst_group),
            "votes": int(alias.votes),
            "score": float(alias.score),
            "average_distance": float(alias.average_distance),
            "average_weight_difference": float(alias.average_weight_difference),
        }
        for alias in aliases
    ]


def build_seam_mapping(mesh_objects, *, perf: dict | None = None) -> SeamBuildResult:
    stage_start = time.perf_counter()
    mesh_objects = _mesh_objects_with_bbox_neighbors(list(mesh_objects), _OBJECT_BBOX_GAP_TOLERANCE)
    if perf is not None:
        perf["bbox_filter"] = time.perf_counter() - stage_start
        perf["candidate_objects"] = len(mesh_objects)

    stage_start = time.perf_counter()
    caches = {}
    cache_details = []
    for mesh_obj in mesh_objects:
        cache_start = time.perf_counter()
        cache_detail = {"object": mesh_obj.name}
        cache = _build_seam_cache(mesh_obj, perf=cache_detail)
        caches[mesh_obj.name] = cache
        cache_seconds = time.perf_counter() - cache_start
        cache_detail["seconds"] = cache_seconds
        if cache is not None:
            cache_detail["seam_vertices"] = len(cache["seam_vertices"])
            cache_detail["groups"] = len(cache["group_clouds"])
            cache_details.append(cache_detail)
    if perf is not None:
        perf["cache_build"] = time.perf_counter() - stage_start
        perf["cache_details"] = cache_details

    object_names = [mesh_obj.name for mesh_obj in mesh_objects if caches.get(mesh_obj.name)]
    total_possible_pairs = max(0, len(object_names) * (len(object_names) - 1) // 2)
    candidate_edges: list[dict] = []
    tested_pairs = 0

    for source_name, source_cache, target_name, target_cache, allowed_group_pairs in _iter_bbox_candidate_pairs(
        caches,
        object_names,
        _OBJECT_BBOX_GAP_TOLERANCE,
        perf=perf,
    ):
        tested_pairs += 1
        stage_start = time.perf_counter()
        source_vertices, source_spatial_hash, target_vertices, target_spatial_hash = _pair_vertices_for_allowed_groups(
            source_cache,
            target_cache,
            allowed_group_pairs,
        )
        if perf is not None:
            perf["pair_prepare"] = perf.get("pair_prepare", 0.0) + (time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        matched_pairs = _build_vertex_pairs(
            source_vertices,
            source_spatial_hash,
            target_vertices,
            target_spatial_hash,
            _MATCH_TOLERANCE,
        )
        if perf is not None:
            perf["nearest_pairs"] = perf.get("nearest_pairs", 0.0) + (time.perf_counter() - stage_start)
        if len(matched_pairs) < _MIN_VERTEX_PAIRS:
            continue

        stage_start = time.perf_counter()
        candidates = _build_mapping_candidates_from_seams(
            source_cache["weight_items_by_vertex"],
            target_cache["weight_items_by_vertex"],
            matched_pairs,
            _WEIGHT_TOLERANCE,
            allowed_group_pairs,
        )
        if perf is not None:
            perf["mapping_candidates"] = perf.get("mapping_candidates", 0.0) + (time.perf_counter() - stage_start)
        for candidate in candidates:
            source_group = int(candidate["group_a"])
            target_group = int(candidate["group_b"])
            if source_group == target_group:
                continue
            candidate_edges.append(
                {
                    "source_name": source_name,
                    "source_group": source_group,
                    "target_name": target_name,
                    "target_group": target_group,
                    "votes": int(candidate["votes"]),
                    "score": float(candidate["score"]),
                    "average_distance": float(candidate["average_distance"]),
                    "average_weight_difference": float(candidate["average_weight_difference"]),
                }
            )

    stage_start = time.perf_counter()
    aliases = _build_aliases_from_edges(candidate_edges)
    summaries = _build_pair_summaries(aliases)
    if perf is not None:
        perf["alias_build"] = time.perf_counter() - stage_start
        perf["tested_pairs"] = tested_pairs
        perf["candidate_edges"] = len(candidate_edges)
    return SeamBuildResult(
        aliases=tuple(aliases),
        pair_summaries=tuple(summaries),
        matched_pairs=tested_pairs,
        skipped_pairs=total_possible_pairs - tested_pairs,
    )


def apply_seam_mapping(mesh_objects, aliases: list[dict]) -> SeamApplyResult:
    mesh_by_name = {mesh_obj.name: mesh_obj for mesh_obj in mesh_objects}
    aliases_by_object: dict[str, list[dict]] = {}
    for alias in aliases:
        if not alias.get("enabled", True):
            continue
        aliases_by_object.setdefault(str(alias["src_object_name"]), []).append(alias)

    updated_objects = 0
    renamed_groups = 0
    skipped_messages: list[str] = []

    for object_name, object_aliases in aliases_by_object.items():
        mesh_obj = mesh_by_name.get(object_name)
        if mesh_obj is None:
            skipped_messages.append(f"{object_name}: object not found")
            continue

        planned: list[tuple[object, str, str]] = []
        source_names: set[str] = set()
        target_names: set[str] = set()
        for alias in object_aliases:
            source_name = str(int(alias["src_group"]))
            target_name = str(int(alias["dst_group"]))
            if source_name == target_name:
                continue
            if source_name in source_names:
                raise ValueError(f"{object_name}: source group {source_name} is listed more than once")
            if target_name in target_names:
                raise ValueError(f"{object_name}: multiple groups would be renamed to {target_name}")
            source_group = mesh_obj.vertex_groups.get(source_name)
            if source_group is None:
                skipped_messages.append(f"{object_name}: source group {source_name} not found")
                continue
            if mesh_obj.vertex_groups.get(target_name) is not None:
                raise ValueError(
                    f"{object_name}: target group {target_name} already exists. "
                    "This fast path only renames groups and will not merge weights."
                )
            planned.append((source_group, source_name, target_name))
            source_names.add(source_name)
            target_names.add(target_name)

        if not planned:
            continue

        temp_names: list[tuple[str, str, str]] = []
        for source_group, source_name, target_name in planned:
            temp_name = f"__bmc_seam_tmp__{source_group.index}__{source_name}"
            source_group.name = temp_name
            temp_names.append((temp_name, source_name, target_name))

        for temp_name, source_name, target_name in temp_names:
            mesh_obj.vertex_groups[temp_name].name = target_name
            if mesh_obj.vertex_groups.get(source_name) is None:
                mesh_obj.vertex_groups.new(name=source_name)
            renamed_groups += 1

        updated_objects += 1

    return SeamApplyResult(
        updated_objects=updated_objects,
        renamed_groups=renamed_groups,
        skipped_messages=tuple(skipped_messages),
    )


def _build_seam_cache(mesh_obj, *, perf: dict | None = None) -> dict | None:
    stage_start = time.perf_counter()
    seam_vertices = _collect_seam_vertices(mesh_obj)
    if perf is not None:
        perf["collect_vertices"] = time.perf_counter() - stage_start
    if not seam_vertices:
        return None
    stage_start = time.perf_counter()
    group_index_to_number = _build_group_index_to_number_map(mesh_obj)
    if perf is not None:
        perf["group_map"] = time.perf_counter() - stage_start
    if not group_index_to_number:
        return None
    stage_start = time.perf_counter()
    weight_items_by_vertex = _build_sorted_vertex_weight_cache(
        mesh_obj,
        {vertex_index for vertex_index, _world_co in seam_vertices},
        group_index_to_number,
    )
    if perf is not None:
        perf["weights"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    weighted_seam_vertices = [
        (vertex_index, world_co)
        for vertex_index, world_co in seam_vertices
        if weight_items_by_vertex.get(vertex_index)
    ]
    if perf is not None:
        perf["filter_vertices"] = time.perf_counter() - stage_start
    if len(weighted_seam_vertices) < _MIN_VERTEX_PAIRS:
        return None
    stage_start = time.perf_counter()
    spatial_hash = _build_spatial_hash(weighted_seam_vertices, _MATCH_TOLERANCE)
    cell_keys = frozenset(spatial_hash)
    expanded_cell_keys = expanded_cell_key_set(cell_keys)
    if perf is not None:
        perf["spatial_hash"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    group_clouds = _build_group_clouds(weighted_seam_vertices, weight_items_by_vertex)
    if perf is not None:
        perf["group_clouds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    group_array_index = _build_group_array_index(group_clouds)
    if perf is not None:
        perf["group_array_index"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    bounds = point_bounds([world_co for _vertex_index, world_co in weighted_seam_vertices])
    bounds_min = tuple(float(value) for value in bounds[0])
    bounds_max = tuple(float(value) for value in bounds[1])
    if perf is not None:
        perf["bounds"] = time.perf_counter() - stage_start
    return {
        "seam_vertices": weighted_seam_vertices,
        "spatial_hash": spatial_hash,
        "cell_keys": cell_keys,
        "expanded_cell_keys": expanded_cell_keys,
        "group_clouds": group_clouds,
        "group_array_index": group_array_index,
        "group_numbers": frozenset(group_clouds),
        "group_vertex_cache": {},
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "weight_items_by_vertex": weight_items_by_vertex,
    }


def _build_group_clouds(weighted_seam_vertices, weight_items_by_vertex) -> dict[int, dict]:
    accumulators: dict[int, dict] = {}
    for vertex_index, world_co in weighted_seam_vertices:
        cell_key = _cell_key(world_co, _MATCH_TOLERANCE)
        for group_number, _weight in weight_items_by_vertex.get(vertex_index, ()):
            group_number = int(group_number)
            accumulator = accumulators.get(group_number)
            if accumulator is None:
                accumulator = {
                    "points": [],
                    "cell_keys": set(),
                    "min_x": float(world_co[0]),
                    "min_y": float(world_co[1]),
                    "min_z": float(world_co[2]),
                    "max_x": float(world_co[0]),
                    "max_y": float(world_co[1]),
                    "max_z": float(world_co[2]),
                }
                accumulators[group_number] = accumulator
            accumulator["points"].append((vertex_index, world_co))
            accumulator["cell_keys"].add(cell_key)
            accumulator["min_x"] = min(accumulator["min_x"], float(world_co[0]))
            accumulator["min_y"] = min(accumulator["min_y"], float(world_co[1]))
            accumulator["min_z"] = min(accumulator["min_z"], float(world_co[2]))
            accumulator["max_x"] = max(accumulator["max_x"], float(world_co[0]))
            accumulator["max_y"] = max(accumulator["max_y"], float(world_co[1]))
            accumulator["max_z"] = max(accumulator["max_z"], float(world_co[2]))

    clouds: dict[int, dict] = {}
    for group_number, accumulator in accumulators.items():
        points = accumulator["points"]
        if len(points) < _MIN_VERTEX_PAIRS:
            continue
        cell_keys = frozenset(accumulator["cell_keys"])
        expanded_cell_keys = expanded_cell_key_set(cell_keys)
        clouds[int(group_number)] = {
            "points": tuple(points),
            "bounds_min": (
                float(accumulator["min_x"]),
                float(accumulator["min_y"]),
                float(accumulator["min_z"]),
            ),
            "bounds_max": (
                float(accumulator["max_x"]),
                float(accumulator["max_y"]),
                float(accumulator["max_z"]),
            ),
            "cell_keys": cell_keys,
            "expanded_cell_keys": expanded_cell_keys,
            "cell_codes": _cell_key_codes(cell_keys),
            "expanded_cell_codes": _cell_key_codes(expanded_cell_keys),
        }
    return clouds


def _build_group_array_index(group_clouds: dict[int, dict]) -> dict:
    np = require_numpy()
    groups = np.asarray(sorted(group_clouds), dtype=np.int64)
    if groups.size == 0:
        empty_bounds = np.zeros((0, 3), dtype=np.float64)
        return {
            "groups": groups,
            "bounds_min": empty_bounds,
            "bounds_max": empty_bounds,
        }
    return {
        "groups": groups,
        "bounds_min": np.asarray([group_clouds[int(group)]["bounds_min"] for group in groups], dtype=np.float64),
        "bounds_max": np.asarray([group_clouds[int(group)]["bounds_max"] for group in groups], dtype=np.float64),
    }


def _build_group_index_to_number_map(mesh_obj) -> dict[int, int]:
    group_index_to_number = {}
    for vertex_group in mesh_obj.vertex_groups:
        try:
            group_number = int(str(vertex_group.name).strip())
        except ValueError:
            continue
        if group_number < 0:
            continue
        group_index_to_number[int(vertex_group.index)] = group_number
    return group_index_to_number


def _collect_seam_vertices(mesh_obj):
    return _collect_seam_vertices_numpy(mesh_obj)


def _collect_seam_vertices_numpy(mesh_obj):
    np = require_numpy()
    mesh = mesh_obj.data
    vertex_count = len(mesh.vertices)
    edge_count = len(mesh.edges)
    loop_count = len(mesh.loops)
    if vertex_count <= 0 or edge_count <= 0 or loop_count <= 0:
        return []

    loop_edge_indices = np.empty(loop_count, dtype=np.int64)
    mesh.loops.foreach_get("edge_index", loop_edge_indices)
    edge_face_counts = np.bincount(loop_edge_indices, minlength=edge_count)
    boundary_edge_mask = edge_face_counts[:edge_count] == 1

    loose_mask = np.zeros(edge_count, dtype=bool)
    mesh.edges.foreach_get("is_loose", loose_mask)
    boundary_edge_mask = np.logical_or(boundary_edge_mask, loose_mask)

    if not bool(boundary_edge_mask.any()):
        return []

    edge_vertices = np.empty(edge_count * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", edge_vertices)
    edge_vertices = edge_vertices.reshape((edge_count, 2))
    boundary_indices = np.unique(edge_vertices[boundary_edge_mask].reshape(-1))
    if boundary_indices.size == 0:
        return []

    coords = np.empty(vertex_count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape((vertex_count, 3))[boundary_indices]
    matrix = _matrix_world_numpy(mesh_obj.matrix_world, np)
    world_coords = coords @ matrix[:3, :3].T + matrix[:3, 3]
    order = np.lexsort((
        boundary_indices,
        world_coords[:, 2],
        world_coords[:, 1],
        world_coords[:, 0],
    ))
    boundary_indices = boundary_indices[order]
    world_coords = world_coords[order]
    return [
        (
            int(vertex_index),
            (
                float(world_co[0]),
                float(world_co[1]),
                float(world_co[2]),
            ),
        )
        for vertex_index, world_co in zip(boundary_indices, world_coords)
    ]


def _matrix_world_numpy(matrix_world, np):
    try:
        rows = [list(row) for row in matrix_world]
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix
    except Exception:
        pass
    return np.eye(4, dtype=np.float64)


def _world_coord_tuple(world_co) -> tuple[float, float, float]:
    return float(world_co[0]), float(world_co[1]), float(world_co[2])


def _build_sorted_vertex_weight_cache(mesh_obj, vertex_indices: set[int], group_index_to_number: dict[int, int]):
    cached_weight_items = {}
    for vertex_index in vertex_indices:
        cached_weight_items[int(vertex_index)] = _sorted_weight_items(
            _read_vertex_weights(mesh_obj, int(vertex_index), group_index_to_number)
        )
    return cached_weight_items


def _read_vertex_weights(mesh_obj, vertex_index: int, group_index_to_number: dict[int, int]):
    weight_map = {}
    vertex = mesh_obj.data.vertices[vertex_index]
    for assignment in vertex.groups:
        group_number = group_index_to_number.get(int(assignment.group))
        if group_number is None:
            continue
        weight = float(assignment.weight)
        if weight <= _WEIGHT_EPSILON:
            continue
        weight_map[group_number] = min(1.0, weight_map.get(group_number, 0.0) + weight)
    return weight_map


def _sorted_weight_items(weight_map: dict[int, float]):
    filtered_items = [(group, weight) for group, weight in weight_map.items() if weight > _WEIGHT_FLOOR]
    filtered_items.sort(key=lambda item: (-item[1], item[0]))
    return tuple(filtered_items)


def _build_mapping_candidates_from_seams(
    source_weight_items_by_vertex,
    target_weight_items_by_vertex,
    matched_pairs,
    weight_tolerance,
    allowed_group_pairs,
):
    allowed_group_pairs = _allowed_group_pairs_from_matched_seams(
        source_weight_items_by_vertex,
        target_weight_items_by_vertex,
        matched_pairs,
        allowed_group_pairs,
    )
    if not allowed_group_pairs:
        return []
    candidate_stats = {}
    for source_index, target_index, pair_distance in matched_pairs:
        source_items = source_weight_items_by_vertex.get(source_index, ())
        target_items = target_weight_items_by_vertex.get(target_index, ())
        if not source_items or not target_items:
            continue
        _accumulate_mapping_candidates(
            candidate_stats,
            source_items,
            target_items,
            weight_tolerance,
            pair_distance,
            allowed_group_pairs,
        )

    candidates = []
    for (group_a, group_b), stats in candidate_stats.items():
        if stats["votes"] < _MIN_MAPPING_VOTES:
            continue
        average_distance = stats["distance_sum"] / max(1, stats["votes"])
        average_weight_difference = stats["weight_difference_sum"] / max(1, stats["votes"])
        if average_distance > _MAX_AVERAGE_DISTANCE:
            continue
        candidates.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "score": stats["score"],
                "votes": stats["votes"],
                "average_distance": average_distance,
                "average_weight_difference": average_weight_difference,
            }
        )
    candidates.sort(key=lambda item: (item["votes"], item["score"]), reverse=True)

    used_group_a = set()
    used_group_b = set()
    mappings = []
    for candidate in candidates:
        if candidate["group_a"] in used_group_a or candidate["group_b"] in used_group_b:
            continue
        used_group_a.add(candidate["group_a"])
        used_group_b.add(candidate["group_b"])
        mappings.append(candidate)
    return mappings


def _allowed_group_pairs_from_matched_seams(
    source_weight_items_by_vertex,
    target_weight_items_by_vertex,
    matched_pairs,
    allowed_group_pairs,
):
    pair_counts: dict[tuple[int, int], int] = {}
    for source_index, target_index, _pair_distance in matched_pairs:
        for source_group, _source_weight in source_weight_items_by_vertex.get(source_index, ()):
            for target_group, _target_weight in target_weight_items_by_vertex.get(target_index, ()):
                pair = (int(source_group), int(target_group))
                if pair not in allowed_group_pairs:
                    continue
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return {
        pair
        for pair, count in pair_counts.items()
        if count >= _MIN_MAPPING_VOTES
    }


def _accumulate_mapping_candidates(candidate_stats, source_items, target_items, weight_tolerance, pair_distance, allowed_group_pairs):
    used_target_indices = set()
    for source_group, source_weight in source_items:
        best_match_index = None
        best_match_score = -1.0
        best_match_group = None
        for target_index, (target_group, target_weight) in enumerate(target_items):
            if target_index in used_target_indices:
                continue
            if (int(source_group), int(target_group)) not in allowed_group_pairs:
                continue
            difference = abs(source_weight - target_weight)
            if difference > weight_tolerance:
                continue
            similarity = 1.0 - (difference / max(weight_tolerance, _WEIGHT_EPSILON))
            match_score = min(source_weight, target_weight) * similarity
            if match_score > best_match_score:
                best_match_index = target_index
                best_match_score = match_score
                best_match_group = target_group
        if best_match_index is None or best_match_group is None:
            continue
        used_target_indices.add(best_match_index)
        key = (source_group, best_match_group)
        stats = candidate_stats.setdefault(
            key,
            {"score": 0.0, "votes": 0, "distance_sum": 0.0, "weight_difference_sum": 0.0},
        )
        stats["score"] += best_match_score
        stats["votes"] += 1
        stats["distance_sum"] += float(pair_distance)
        stats["weight_difference_sum"] += abs(float(source_weight) - float(target_items[best_match_index][1]))


def _build_aliases_from_edges(candidate_edges: list[dict]) -> list[SeamAliasRecord]:
    if not candidate_edges:
        return []

    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: tuple[str, int], right: tuple[str, int]) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    best_edge_by_node: dict[tuple[str, int], dict] = {}
    for edge in candidate_edges:
        source_node = (str(edge["source_name"]), int(edge["source_group"]))
        target_node = (str(edge["target_name"]), int(edge["target_group"]))
        union(source_node, target_node)
        for node in (source_node, target_node):
            existing = best_edge_by_node.get(node)
            if existing is None or float(existing["score"]) < float(edge["score"]):
                best_edge_by_node[node] = edge

    component_nodes: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in list(parent):
        component_nodes.setdefault(find(node), []).append(node)

    aliases: list[SeamAliasRecord] = []
    for nodes in component_nodes.values():
        if len(nodes) < 2:
            continue
        object_names = [object_name for object_name, _group in nodes]
        if len(object_names) != len(set(object_names)):
            repeated_objects = sorted({name for name in object_names if object_names.count(name) > 1})
            raise ValueError(
                "Multiple groups from one object were mapped into one seam component: "
                + ", ".join(repeated_objects[:4])
                + ". Clean the seam candidates before applying."
            )

        canonical_node = min(nodes, key=lambda item: (int(item[1]), str(item[0])))
        canonical_group = int(canonical_node[1])
        for source_node in sorted(nodes, key=lambda item: (int(item[1]), str(item[0]))):
            source_group = int(source_node[1])
            if source_node == canonical_node or source_group == canonical_group:
                continue
            edge = best_edge_by_node.get(source_node, {})
            aliases.append(
                SeamAliasRecord(
                    src_object_name=source_node[0],
                    src_group=source_group,
                    dst_object_name=canonical_node[0],
                    dst_group=canonical_group,
                    votes=int(edge.get("votes", 0)),
                    score=float(edge.get("score", 0.0)),
                    average_distance=float(edge.get("average_distance", 0.0)),
                    average_weight_difference=float(edge.get("average_weight_difference", 0.0)),
                )
            )
    aliases.sort(key=lambda alias: (alias.dst_group, alias.src_group, alias.src_object_name))
    return aliases


def _build_pair_summaries(aliases: list[SeamAliasRecord]) -> list[str]:
    counts: dict[tuple[str, str], int] = {}
    for alias in aliases:
        key = tuple(sorted((alias.src_object_name, alias.dst_object_name)))
        counts[key] = counts.get(key, 0) + 1
    return [
        f"{left} <-> {right}: {count} aliases"
        for (left, right), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _cell_key(world_co, tolerance):
    return _generic_cell_key(world_co, tolerance)


def _cell_key_codes(cell_keys):
    np = require_numpy()
    if not cell_keys:
        return np.zeros((0,), dtype=_cell_code_dtype(np))
    values = np.asarray(list(cell_keys), dtype=np.int64).reshape((-1, 3))
    values = np.ascontiguousarray(values)
    codes = values.view(_cell_code_dtype(np)).reshape((-1,))
    return np.unique(codes)


def _cell_code_dtype(np):
    return np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])


def _neighbor_keys(base_key):
    yield from _generic_neighbor_keys(base_key)


def _build_spatial_hash(vertices, tolerance):
    return _generic_build_spatial_hash(vertices, lambda item: item[1], tolerance)


def _mesh_objects_with_bbox_neighbors(mesh_objects: list, max_gap: float) -> list:
    if len(mesh_objects) < 2:
        return mesh_objects
    object_bounds = []
    for mesh_obj in mesh_objects:
        bounds = _object_bounds_world(mesh_obj)
        if bounds is None:
            continue
        object_bounds.append((mesh_obj, bounds[0], bounds[1]))
    if len(object_bounds) < 2:
        return []

    sorted_bounds = sorted(object_bounds, key=lambda item: (item[1][0], item[2][0], item[0].name))
    neighbor_names: set[str] = set()
    for source_index, (source_obj, source_min, source_max) in enumerate(sorted_bounds):
        source_max_x = float(source_max[0]) + float(max_gap)
        for target_index in range(source_index + 1, len(sorted_bounds)):
            target_obj, target_min, target_max = sorted_bounds[target_index]
            if float(target_min[0]) > source_max_x:
                break
            if not _bounds_overlap_with_gap(source_min, source_max, target_min, target_max, max_gap):
                continue
            neighbor_names.add(source_obj.name)
            neighbor_names.add(target_obj.name)
    return [mesh_obj for mesh_obj in mesh_objects if mesh_obj.name in neighbor_names]


def _object_bounds_world(mesh_obj):
    raw_points = getattr(mesh_obj, "bound_box", None)
    if raw_points:
        points = [_world_coord_tuple(_transform_point(mesh_obj.matrix_world, point)) for point in raw_points]
    else:
        vertices = getattr(getattr(mesh_obj, "data", None), "vertices", None)
        if not vertices:
            return None
        points = [_world_coord_tuple(_transform_point(mesh_obj.matrix_world, vertex.co)) for vertex in vertices]
    if not points:
        return None
    return _points_bounds_min(points), _points_bounds_max(points)


def _transform_point(matrix, point):
    if _MathutilsVector is not None and isinstance(point, (tuple, list)):
        try:
            return matrix @ _MathutilsVector(point)
        except Exception:
            return point
    try:
        return matrix @ point
    except TypeError:
        if _MathutilsVector is None:
            return point
        try:
            return matrix @ _MathutilsVector(point)
        except Exception:
            return point


def _iter_bbox_candidate_pairs(caches: dict, object_names: list[str], max_gap: float, *, perf: dict | None = None):
    sorted_items = sorted(
        ((name, caches[name]) for name in object_names),
        key=lambda item: (item[1]["bounds_min"][0], item[1]["bounds_max"][0], item[0]),
    )
    for source_index, (source_name, source_cache) in enumerate(sorted_items):
        source_max_x = float(source_cache["bounds_max"][0]) + float(max_gap)
        for target_index in range(source_index + 1, len(sorted_items)):
            target_name, target_cache = sorted_items[target_index]
            if float(target_cache["bounds_min"][0]) > source_max_x:
                break
            if not _bounds_overlap_with_gap(
                source_cache["bounds_min"],
                source_cache["bounds_max"],
                target_cache["bounds_min"],
                target_cache["bounds_max"],
                max_gap,
            ):
                continue
            if source_cache["expanded_cell_keys"].isdisjoint(target_cache["cell_keys"]):
                continue
            stage_start = time.perf_counter()
            allowed_group_pairs = _allowed_group_pairs_from_caches(source_cache, target_cache)
            if perf is not None:
                perf["allowed_groups"] = perf.get("allowed_groups", 0.0) + (time.perf_counter() - stage_start)
            if not allowed_group_pairs:
                continue
            yield source_name, source_cache, target_name, target_cache, allowed_group_pairs


def _allowed_group_pairs_from_caches(source_cache: dict, target_cache: dict) -> set[tuple[int, int]]:
    source_group_clouds = source_cache["group_clouds"]
    target_group_clouds = target_cache["group_clouds"]
    source_index = source_cache["group_array_index"]
    target_index = target_cache["group_array_index"]
    np = require_numpy()
    source_groups = source_index["groups"]
    target_groups = target_index["groups"]
    if source_groups.size == 0 or target_groups.size == 0:
        return set()
    source_mins = source_index["bounds_min"]
    source_maxs = source_index["bounds_max"]
    target_mins = target_index["bounds_min"]
    target_maxs = target_index["bounds_max"]
    gap = float(_GROUP_BBOX_GAP_TOLERANCE)
    overlaps = np.all(
        (source_mins[:, None, :] <= target_maxs[None, :, :] + gap)
        & (target_mins[None, :, :] <= source_maxs[:, None, :] + gap),
        axis=2,
    )
    source_positions, target_positions = np.nonzero(overlaps)
    if source_positions.size == 0:
        return set()
    allowed: set[tuple[int, int]] = set()
    for source_position, target_position in zip(source_positions.tolist(), target_positions.tolist()):
        source_group = int(source_groups[int(source_position)])
        target_group = int(target_groups[int(target_position)])
        source_cloud = source_group_clouds[source_group]
        target_cloud = target_group_clouds[target_group]
        if not source_cloud["cell_keys"].isdisjoint(target_cloud["expanded_cell_keys"]):
            allowed.add((int(source_group), int(target_group)))
    return allowed


def _pair_vertices_for_allowed_groups(source_cache: dict, target_cache: dict, allowed_group_pairs: set[tuple[int, int]]):
    source_groups = {int(source_group) for source_group, _target_group in allowed_group_pairs}
    target_groups = {int(target_group) for _source_group, target_group in allowed_group_pairs}
    source_vertices, source_spatial_hash = _cached_vertices_and_hash_for_groups(source_cache, source_groups)
    target_vertices, target_spatial_hash = _cached_vertices_and_hash_for_groups(target_cache, target_groups)
    return source_vertices, source_spatial_hash, target_vertices, target_spatial_hash


def _cached_vertices_and_hash_for_groups(cache: dict, groups: set[int]):
    normalized_groups = frozenset(int(group) for group in groups)
    if not normalized_groups:
        return (), {}
    if normalized_groups.issuperset(cache["group_numbers"]):
        return cache["seam_vertices"], cache["spatial_hash"]
    group_vertex_cache = cache.setdefault("group_vertex_cache", {})
    cached = group_vertex_cache.get(normalized_groups)
    if cached is not None:
        return cached
    vertices = _vertices_for_groups(cache["group_clouds"], set(normalized_groups))
    spatial_hash = _build_spatial_hash(vertices, _MATCH_TOLERANCE)
    cached = (vertices, spatial_hash)
    group_vertex_cache[normalized_groups] = cached
    return cached


def _vertices_for_groups(group_clouds: dict[int, dict], groups: set[int]):
    by_vertex: dict[int, tuple[int, tuple[float, float, float]]] = {}
    for group in groups:
        cloud = group_clouds.get(int(group))
        if not cloud:
            continue
        for vertex_index, world_co in cloud["points"]:
            by_vertex.setdefault(int(vertex_index), (int(vertex_index), world_co))
    return sorted(by_vertex.values(), key=lambda item: (item[1][0], item[1][1], item[1][2], item[0]))


def _print_seam_performance_report(perf: dict, build_result: SeamBuildResult) -> None:
    print("[BMC Seam] Performance report")
    print(
        "[BMC Seam] "
        f"total={float(perf.get('total', 0.0)):.3f}s "
        f"build={float(perf.get('build_total', 0.0)):.3f}s "
        f"bbox_filter={float(perf.get('bbox_filter', 0.0)):.3f}s "
        f"cache_build={float(perf.get('cache_build', 0.0)):.3f}s "
        f"allowed_groups={float(perf.get('allowed_groups', 0.0)):.3f}s "
        f"pair_prepare={float(perf.get('pair_prepare', 0.0)):.3f}s "
        f"nearest_pairs={float(perf.get('nearest_pairs', 0.0)):.3f}s "
        f"mapping_candidates={float(perf.get('mapping_candidates', 0.0)):.3f}s "
        f"alias_build={float(perf.get('alias_build', 0.0)):.3f}s "
        f"apply={float(perf.get('apply', 0.0)):.3f}s"
    )
    print(
        "[BMC Seam] "
        f"objects={int(perf.get('candidate_objects', 0) or 0)} "
        f"tested_pairs={build_result.matched_pairs} "
        f"skipped_pairs={build_result.skipped_pairs} "
        f"candidate_edges={int(perf.get('candidate_edges', 0) or 0)} "
        f"aliases={len(build_result.aliases)}"
    )
    cache_details = list(perf.get("cache_details", []) or [])
    if cache_details:
        print("[BMC Seam] Slowest seam caches:")
        for index, detail in enumerate(sorted(cache_details, key=lambda item: item["seconds"], reverse=True)[:5], start=1):
            print(
                "[BMC Seam]   "
                f"{index}. {detail['object']} "
                f"total={float(detail['seconds']):.3f}s "
                f"collect_vertices={float(detail.get('collect_vertices', 0.0)):.3f}s "
                f"weights={float(detail.get('weights', 0.0)):.3f}s "
                f"group_clouds={float(detail.get('group_clouds', 0.0)):.3f}s "
                f"vertices={int(detail['seam_vertices'])} "
                f"groups={int(detail['groups'])}"
            )


def _build_nearest_vertex_map_from_hash(source_spatial_hash, target_spatial_hash, tolerance):
    return _build_nearest_vertex_map_from_hash_numpy(source_spatial_hash, target_spatial_hash, tolerance)


def _build_nearest_vertex_map_from_hash_numpy(source_spatial_hash, target_spatial_hash, tolerance):
    np = require_numpy()
    tolerance_squared = float(tolerance) * float(tolerance)
    nearest_by_source = {}
    max_distance_values = 1_000_000
    for source_key, source_items in source_spatial_hash.items():
        if not source_items:
            continue
        target_items = [
            item
            for neighbor_key in _neighbor_keys(source_key)
            for item in target_spatial_hash.get(neighbor_key, ())
        ]
        if not target_items:
            continue

        source_indices = np.fromiter((int(item[0]) for item in source_items), dtype=np.int64, count=len(source_items))
        target_indices = np.fromiter((int(item[0]) for item in target_items), dtype=np.int64, count=len(target_items))
        source_coords = np.asarray([item[1] for item in source_items], dtype=np.float64)
        target_coords = np.asarray([item[1] for item in target_items], dtype=np.float64)
        if source_coords.size == 0 or target_coords.size == 0:
            continue

        chunk_size = max(1, int(max_distance_values // max(1, len(target_items))))
        for chunk_start in range(0, len(source_items), chunk_size):
            chunk_end = min(len(source_items), chunk_start + chunk_size)
            source_chunk = source_coords[chunk_start:chunk_end]
            deltas = source_chunk[:, None, :] - target_coords[None, :, :]
            distances_squared = np.einsum("ijk,ijk->ij", deltas, deltas)
            nearest_positions = np.argmin(distances_squared, axis=1)
            row_indices = np.arange(chunk_end - chunk_start)
            nearest_distances_squared = distances_squared[row_indices, nearest_positions]
            valid_rows = np.nonzero(nearest_distances_squared <= tolerance_squared)[0]
            for row in valid_rows:
                source_index = int(source_indices[chunk_start + int(row)])
                target_position = int(nearest_positions[int(row)])
                target_index = int(target_indices[target_position])
                nearest_by_source[source_index] = (
                    target_index,
                    float(nearest_distances_squared[int(row)] ** 0.5),
                )
    return nearest_by_source


def _build_vertex_pairs(source_vertices, source_spatial_hash, target_vertices, target_spatial_hash, tolerance):
    source_to_target = _build_nearest_vertex_map_from_hash(source_spatial_hash, target_spatial_hash, tolerance)
    if len(source_to_target) < _MIN_VERTEX_PAIRS:
        return []
    candidate_target_indices = {target_index for target_index, _distance in source_to_target.values()}
    target_candidates = [
        (target_index, target_world_co)
        for target_index, target_world_co in target_vertices
        if target_index in candidate_target_indices
    ]
    if len(target_candidates) < _MIN_VERTEX_PAIRS:
        return []
    target_candidates_spatial_hash = _build_spatial_hash(target_candidates, tolerance)
    target_to_source = _build_nearest_vertex_map_from_hash(target_candidates_spatial_hash, source_spatial_hash, tolerance)
    matched_pairs = []
    for source_index, (target_index, distance) in source_to_target.items():
        reverse = target_to_source.get(target_index)
        if reverse is None:
            continue
        reverse_source_index, _reverse_distance = reverse
        if reverse_source_index != source_index:
            continue
        matched_pairs.append((source_index, target_index, distance))
    return matched_pairs


def _points_bounds_min(points):
    first = points[0]
    min_x = float(first[0])
    min_y = float(first[1])
    min_z = float(first[2])
    for point in points[1:]:
        min_x = min(min_x, float(point[0]))
        min_y = min(min_y, float(point[1]))
        min_z = min(min_z, float(point[2]))
    return min_x, min_y, min_z


def _points_bounds_max(points):
    first = points[0]
    max_x = float(first[0])
    max_y = float(first[1])
    max_z = float(first[2])
    for point in points[1:]:
        max_x = max(max_x, float(point[0]))
        max_y = max(max_y, float(point[1]))
        max_z = max(max_z, float(point[2]))
    return max_x, max_y, max_z


def _axis_gap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    if max_a < min_b:
        return min_b - max_a
    if max_b < min_a:
        return min_a - max_b
    return 0.0


def _bounds_overlap_with_gap(bounds_min_a, bounds_max_a, bounds_min_b, bounds_max_b, max_gap: float) -> bool:
    gap_x = _axis_gap(bounds_min_a[0], bounds_max_a[0], bounds_min_b[0], bounds_max_b[0])
    if gap_x > max_gap:
        return False
    gap_y = _axis_gap(bounds_min_a[1], bounds_max_a[1], bounds_min_b[1], bounds_max_b[1])
    if gap_y > max_gap:
        return False
    gap_z = _axis_gap(bounds_min_a[2], bounds_max_a[2], bounds_min_b[2], bounds_max_b[2])
    return gap_z <= max_gap
