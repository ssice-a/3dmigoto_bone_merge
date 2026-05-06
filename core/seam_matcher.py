"""Independent seam-based vertex-group matcher."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


_MATCH_TOLERANCE = 0.0015
_WEIGHT_TOLERANCE = 0.001
_MIN_VERTEX_PAIRS = 4
_MIN_MAPPING_VOTES = 3
_MAX_AVERAGE_DISTANCE = 0.001
_OBJECT_BBOX_GAP_TOLERANCE = 0.01
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
    caches = {mesh_obj.name: _build_seam_cache(mesh_obj) for mesh_obj in mesh_objects}
    object_names = [mesh_obj.name for mesh_obj in mesh_objects if caches.get(mesh_obj.name)]
    candidate_edges: list[dict] = []
    skipped_pairs = 0
    tested_pairs = 0

    for source_index, source_name in enumerate(object_names):
        source_cache = caches[source_name]
        for target_name in object_names[source_index + 1 :]:
            target_cache = caches[target_name]
            tested_pairs += 1
            if not _bounds_overlap_with_gap(
                source_cache["bounds_min"],
                source_cache["bounds_max"],
                target_cache["bounds_min"],
                target_cache["bounds_max"],
                _OBJECT_BBOX_GAP_TOLERANCE,
            ):
                skipped_pairs += 1
                continue

            matched_pairs = _build_vertex_pairs(
                source_cache["seam_vertices"],
                source_cache["spatial_hash"],
                target_cache["seam_vertices"],
                target_cache["spatial_hash"],
                _MATCH_TOLERANCE,
            )
            if len(matched_pairs) < _MIN_VERTEX_PAIRS:
                continue

            candidates = _build_mapping_candidates_from_seams(
                source_cache["weight_items_by_vertex"],
                target_cache["weight_items_by_vertex"],
                matched_pairs,
                _WEIGHT_TOLERANCE,
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
        matched_pairs=tested_pairs - skipped_pairs,
        skipped_pairs=skipped_pairs,
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
    return {
        "seam_vertices": weighted_seam_vertices,
        "spatial_hash": _build_spatial_hash(weighted_seam_vertices, _MATCH_TOLERANCE),
        "bounds_min": _bounds_min(weighted_seam_vertices),
        "bounds_max": _bounds_max(weighted_seam_vertices),
        "weight_items_by_vertex": weight_items_by_vertex,
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
            edge_face_counts[tuple(sorted(edge_key))] = edge_face_counts.get(tuple(sorted(edge_key)), 0) + 1

    boundary_indices = set()
    for edge in mesh.edges:
        edge_key = tuple(sorted(edge.vertices))
        if edge.is_loose or edge_face_counts.get(edge_key, 0) == 1:
            boundary_indices.update(edge.vertices)
    return boundary_indices


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


def _build_mapping_candidates_from_seams(source_weight_items_by_vertex, target_weight_items_by_vertex, matched_pairs, weight_tolerance):
    candidate_stats = {}
    for source_index, target_index, pair_distance in matched_pairs:
        source_items = source_weight_items_by_vertex.get(source_index, ())
        target_items = target_weight_items_by_vertex.get(target_index, ())
        if not source_items or not target_items:
            continue
        _accumulate_mapping_candidates(candidate_stats, source_items, target_items, weight_tolerance, pair_distance)

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


def _accumulate_mapping_candidates(candidate_stats, source_items, target_items, weight_tolerance, pair_distance):
    used_target_indices = set()
    for source_group, source_weight in source_items:
        best_match_index = None
        best_match_score = -1.0
        best_match_group = None
        for target_index, (target_group, target_weight) in enumerate(target_items):
            if target_index in used_target_indices:
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
    inverse_tolerance = 1.0 / tolerance
    return (
        floor(world_co[0] * inverse_tolerance),
        floor(world_co[1] * inverse_tolerance),
        floor(world_co[2] * inverse_tolerance),
    )


def _neighbor_keys(base_key):
    base_x, base_y, base_z = base_key
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                yield (base_x + offset_x, base_y + offset_y, base_z + offset_z)


def _build_spatial_hash(vertices, tolerance):
    spatial_hash = {}
    for vertex_index, world_co in vertices:
        spatial_hash.setdefault(_cell_key(world_co, tolerance), []).append((vertex_index, world_co))
    return spatial_hash


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
    target_to_source = _build_nearest_vertex_map(target_vertices, source_spatial_hash, tolerance)
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
