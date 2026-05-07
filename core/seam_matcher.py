"""Independent seam-based vertex-group matcher."""

from __future__ import annotations

from dataclasses import dataclass

from .spatial_index import (
    build_spatial_hash as _generic_build_spatial_hash,
    cell_key as _generic_cell_key,
    expanded_neighbor_cell_keys as _generic_expanded_neighbor_cell_keys,
    neighbor_keys as _generic_neighbor_keys,
)

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
    build_result = build_seam_mapping(mesh_objects)
    alias_payload = seam_aliases_to_payload(build_result.aliases)
    apply_result = apply_seam_mapping(mesh_objects, alias_payload)
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


def build_seam_mapping(mesh_objects) -> SeamBuildResult:
    mesh_objects = _mesh_objects_with_bbox_neighbors(list(mesh_objects), _OBJECT_BBOX_GAP_TOLERANCE)
    caches = {mesh_obj.name: _build_seam_cache(mesh_obj) for mesh_obj in mesh_objects}
    object_names = [mesh_obj.name for mesh_obj in mesh_objects if caches.get(mesh_obj.name)]
    total_possible_pairs = max(0, len(object_names) * (len(object_names) - 1) // 2)
    candidate_edges: list[dict] = []
    tested_pairs = 0

    for source_name, source_cache, target_name, target_cache, allowed_group_pairs in _iter_bbox_candidate_pairs(
        caches,
        object_names,
        _OBJECT_BBOX_GAP_TOLERANCE,
    ):
        tested_pairs += 1
        source_vertices, source_spatial_hash, target_vertices, target_spatial_hash = _pair_vertices_for_allowed_groups(
            source_cache,
            target_cache,
            allowed_group_pairs,
        )
        matched_pairs = _build_vertex_pairs(
            source_vertices,
            source_spatial_hash,
            target_vertices,
            target_spatial_hash,
            _MATCH_TOLERANCE,
        )
        if len(matched_pairs) < _MIN_VERTEX_PAIRS:
            continue

        candidates = _build_mapping_candidates_from_seams(
            source_cache["weight_items_by_vertex"],
            target_cache["weight_items_by_vertex"],
            matched_pairs,
            _WEIGHT_TOLERANCE,
            allowed_group_pairs,
        )
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

    aliases = _build_aliases_from_edges(candidate_edges)
    summaries = _build_pair_summaries(aliases)
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


def _build_seam_cache(mesh_obj) -> dict | None:
    seam_vertices = _collect_seam_vertices(mesh_obj)
    if not seam_vertices:
        return None
    group_index_to_number = _build_group_index_to_number_map(mesh_obj)
    if not group_index_to_number:
        return None
    weight_items_by_vertex = _build_sorted_vertex_weight_cache(
        mesh_obj,
        {vertex_index for vertex_index, _world_co in seam_vertices},
        group_index_to_number,
    )
    weighted_seam_vertices = [
        (vertex_index, world_co)
        for vertex_index, world_co in seam_vertices
        if weight_items_by_vertex.get(vertex_index)
    ]
    if len(weighted_seam_vertices) < _MIN_VERTEX_PAIRS:
        return None
    spatial_hash = _build_spatial_hash(weighted_seam_vertices, _MATCH_TOLERANCE)
    cell_keys = frozenset(spatial_hash)
    group_clouds = _build_group_clouds(weighted_seam_vertices, weight_items_by_vertex)
    return {
        "seam_vertices": weighted_seam_vertices,
        "spatial_hash": spatial_hash,
        "cell_keys": cell_keys,
        "neighbor_cell_keys": _expanded_neighbor_cell_keys(cell_keys),
        "group_clouds": group_clouds,
        "bounds_min": _bounds_min(weighted_seam_vertices),
        "bounds_max": _bounds_max(weighted_seam_vertices),
        "weight_items_by_vertex": weight_items_by_vertex,
    }


def _build_group_clouds(weighted_seam_vertices, weight_items_by_vertex) -> dict[int, dict]:
    points_by_group: dict[int, list[tuple[int, tuple[float, float, float]]]] = {}
    for vertex_index, world_co in weighted_seam_vertices:
        for group_number, _weight in weight_items_by_vertex.get(vertex_index, ()):
            points_by_group.setdefault(int(group_number), []).append((vertex_index, world_co))

    clouds: dict[int, dict] = {}
    for group_number, points in points_by_group.items():
        if len(points) < _MIN_VERTEX_PAIRS:
            continue
        cell_keys = frozenset(_cell_key(world_co, _MATCH_TOLERANCE) for _vertex_index, world_co in points)
        clouds[int(group_number)] = {
            "points": tuple(points),
            "bounds_min": _bounds_min(points),
            "bounds_max": _bounds_max(points),
            "cell_keys": cell_keys,
            "neighbor_cell_keys": _expanded_neighbor_cell_keys(cell_keys),
        }
    return clouds


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
    boundary_indices = _resolve_boundary_vertex_indices(mesh_obj)
    matrix_world = mesh_obj.matrix_world
    vertices = []
    for vertex_index in boundary_indices:
        vertex = mesh_obj.data.vertices[vertex_index]
        vertices.append((vertex.index, _world_coord_tuple(matrix_world @ vertex.co)))
    vertices.sort(key=lambda item: (item[1][0], item[1][1], item[1][2], item[0]))
    return vertices


def _world_coord_tuple(world_co) -> tuple[float, float, float]:
    return float(world_co[0]), float(world_co[1]), float(world_co[2])


def _resolve_boundary_vertex_indices(mesh_obj):
    mesh = mesh_obj.data
    edge_face_counts = {}
    for polygon in mesh.polygons:
        for edge_key in polygon.edge_keys:
            normalized_edge_key = _normalized_edge_key(edge_key)
            edge_face_counts[normalized_edge_key] = edge_face_counts.get(normalized_edge_key, 0) + 1

    boundary_indices = set()
    for edge in mesh.edges:
        edge_key = _normalized_edge_key(edge.vertices)
        if edge.is_loose or edge_face_counts.get(edge_key, 0) == 1:
            boundary_indices.update(edge.vertices)
    return boundary_indices


def _normalized_edge_key(edge_vertices) -> tuple[int, int]:
    left = int(edge_vertices[0])
    right = int(edge_vertices[1])
    return (left, right) if left <= right else (right, left)


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


def _neighbor_keys(base_key):
    yield from _generic_neighbor_keys(base_key)


def _build_spatial_hash(vertices, tolerance):
    return _generic_build_spatial_hash(vertices, lambda item: item[1], tolerance)


def _expanded_neighbor_cell_keys(cell_keys):
    return _generic_expanded_neighbor_cell_keys(cell_keys)


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


def _iter_bbox_candidate_pairs(caches: dict, object_names: list[str], max_gap: float):
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
            if source_cache["neighbor_cell_keys"].isdisjoint(target_cache["cell_keys"]):
                continue
            allowed_group_pairs = _allowed_group_pairs_from_group_clouds(
                source_cache["group_clouds"],
                target_cache["group_clouds"],
            )
            if not allowed_group_pairs:
                continue
            yield source_name, source_cache, target_name, target_cache, allowed_group_pairs


def _allowed_group_pairs_from_group_clouds(source_group_clouds: dict[int, dict], target_group_clouds: dict[int, dict]) -> set[tuple[int, int]]:
    if not source_group_clouds or not target_group_clouds:
        return set()
    allowed: set[tuple[int, int]] = set()
    target_items = sorted(
        target_group_clouds.items(),
        key=lambda item: (item[1]["bounds_min"][0], item[1]["bounds_max"][0], int(item[0])),
    )
    for source_group, source_cloud in sorted(source_group_clouds.items()):
        source_max_x = float(source_cloud["bounds_max"][0]) + _GROUP_BBOX_GAP_TOLERANCE
        for target_group, target_cloud in target_items:
            if float(target_cloud["bounds_min"][0]) > source_max_x:
                break
            if not _bounds_overlap_with_gap(
                source_cloud["bounds_min"],
                source_cloud["bounds_max"],
                target_cloud["bounds_min"],
                target_cloud["bounds_max"],
                _GROUP_BBOX_GAP_TOLERANCE,
            ):
                continue
            if source_cloud["neighbor_cell_keys"].isdisjoint(target_cloud["cell_keys"]):
                continue
            allowed.add((int(source_group), int(target_group)))
    return allowed


def _pair_vertices_for_allowed_groups(source_cache: dict, target_cache: dict, allowed_group_pairs: set[tuple[int, int]]):
    source_groups = {int(source_group) for source_group, _target_group in allowed_group_pairs}
    target_groups = {int(target_group) for _source_group, target_group in allowed_group_pairs}
    source_vertices = _vertices_for_groups(source_cache["group_clouds"], source_groups)
    target_vertices = _vertices_for_groups(target_cache["group_clouds"], target_groups)
    source_spatial_hash = _build_spatial_hash(source_vertices, _MATCH_TOLERANCE)
    target_spatial_hash = _build_spatial_hash(target_vertices, _MATCH_TOLERANCE)
    return source_vertices, source_spatial_hash, target_vertices, target_spatial_hash


def _vertices_for_groups(group_clouds: dict[int, dict], groups: set[int]):
    by_vertex: dict[int, tuple[int, tuple[float, float, float]]] = {}
    for group in groups:
        cloud = group_clouds.get(int(group))
        if not cloud:
            continue
        for vertex_index, world_co in cloud["points"]:
            by_vertex.setdefault(int(vertex_index), (int(vertex_index), world_co))
    return sorted(by_vertex.values(), key=lambda item: (item[1][0], item[1][1], item[1][2], item[0]))


def _build_nearest_vertex_map(source_vertices, target_spatial_hash, tolerance):
    tolerance_squared = tolerance * tolerance
    nearest_by_source = {}
    for source_index, source_world_co in source_vertices:
        nearest_match = None
        nearest_distance_squared = None
        for key in _neighbor_keys(_cell_key(source_world_co, tolerance)):
            for target_index, target_world_co in target_spatial_hash.get(key, ()):
                delta_x = source_world_co[0] - target_world_co[0]
                delta_y = source_world_co[1] - target_world_co[1]
                delta_z = source_world_co[2] - target_world_co[2]
                distance_squared = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
                if distance_squared > tolerance_squared:
                    continue
                if nearest_distance_squared is None or distance_squared < nearest_distance_squared:
                    nearest_match = target_index
                    nearest_distance_squared = distance_squared
        if nearest_match is not None:
            nearest_by_source[source_index] = (nearest_match, nearest_distance_squared ** 0.5)
    return nearest_by_source


def _build_vertex_pairs(source_vertices, source_spatial_hash, target_vertices, target_spatial_hash, tolerance):
    source_to_target = _build_nearest_vertex_map(source_vertices, target_spatial_hash, tolerance)
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
    target_to_source = _build_nearest_vertex_map(target_candidates, source_spatial_hash, tolerance)
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


def _bounds_min(vertices):
    first_world_co = vertices[0][1]
    min_x = float(first_world_co[0])
    min_y = float(first_world_co[1])
    min_z = float(first_world_co[2])
    for _vertex_index, world_co in vertices[1:]:
        min_x = min(min_x, float(world_co[0]))
        min_y = min(min_y, float(world_co[1]))
        min_z = min(min_z, float(world_co[2]))
    return min_x, min_y, min_z


def _bounds_max(vertices):
    first_world_co = vertices[0][1]
    max_x = float(first_world_co[0])
    max_y = float(first_world_co[1])
    max_z = float(first_world_co[2])
    for _vertex_index, world_co in vertices[1:]:
        max_x = max(max_x, float(world_co[0]))
        max_y = max(max_y, float(world_co[1]))
        max_z = max(max_z, float(world_co[2]))
    return max_x, max_y, max_z


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
