from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

seam_matcher = importlib.import_module(f"{PACKAGE_DIR.name}.core.seam_matcher")


class FakeAssignment:
    def __init__(self, group: int, weight: float):
        self.group = group
        self.weight = weight


class FakeVertex:
    def __init__(self, index: int, co: tuple[float, float, float], groups: list[FakeAssignment]):
        self.index = index
        self.co = co
        self.groups = groups


class FakeEdge:
    def __init__(self, vertices: tuple[int, int]):
        self.vertices = vertices
        self.is_loose = True


class FakePolygon:
    edge_keys: tuple[tuple[int, int], ...] = ()


class FakeVertexGroup:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index


class FakeVertexGroups:
    def __init__(self, names: list[str]):
        self._groups = [FakeVertexGroup(name, index) for index, name in enumerate(names)]

    def __iter__(self):
        return iter(self._groups)

    def __getitem__(self, name: str):
        group = self.get(name)
        if group is None:
            raise KeyError(name)
        return group

    def get(self, name: str):
        for group in self._groups:
            if group.name == name:
                return group
        return None

    def new(self, name: str):
        group = FakeVertexGroup(name, len(self._groups))
        self._groups.append(group)
        return group


class IdentityMatrix:
    def __matmul__(self, vector):
        return vector


class FakeMeshData:
    def __init__(self, vertices: list[FakeVertex]):
        self.vertices = vertices
        self.edges = [FakeEdge((index, (index + 1) % len(vertices))) for index in range(len(vertices))]
        self.polygons = [FakePolygon()]


class FakeObject:
    def __init__(self, name: str, group_names: list[str], points: list[tuple[float, float, float]]):
        self.name = name
        self.matrix_world = IdentityMatrix()
        self.vertex_groups = FakeVertexGroups(group_names)
        self.data = FakeMeshData(
            [FakeVertex(index, point, [FakeAssignment(0, 1.0)]) for index, point in enumerate(points)]
        )


class SeamMatcherTests(unittest.TestCase):
    def test_build_and_apply_renames_to_canonical_group(self):
        points = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0), (0.0, 3.0, 0.0)]
        left = FakeObject("left", ["10"], points)
        right = FakeObject("right", ["20"], points)

        result = seam_matcher.build_and_apply_seam_mapping([left, right])

        self.assertEqual(len(result.aliases), 1)
        self.assertEqual(result.renamed_groups, 1)
        self.assertIsNotNone(right.vertex_groups.get("10"))
        self.assertIsNotNone(right.vertex_groups.get("20"))

    def test_matching_same_global_group_is_noop(self):
        points = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0), (0.0, 3.0, 0.0)]
        left = FakeObject("left", ["10"], points)
        right = FakeObject("right", ["10"], points)

        result = seam_matcher.build_and_apply_seam_mapping([left, right])

        self.assertEqual(result.aliases, ())
        self.assertEqual(result.renamed_groups, 0)


if __name__ == "__main__":
    unittest.main()
