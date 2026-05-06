from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

lod_fallback = importlib.import_module(f"{PACKAGE_DIR.name}.core.lod_fallback")


class FakeGroup:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index


class FakeGroupWeight:
    def __init__(self, group: int, weight: float):
        self.group = group
        self.weight = weight


class FakeVertex:
    def __init__(self, index: int, co, groups):
        self.index = index
        self.co = co
        self.groups = [FakeGroupWeight(group, weight) for group, weight in groups]


class FakeMesh:
    def __init__(self, vertices):
        self.vertices = vertices


class FakeObject(dict):
    type = "MESH"

    def __init__(self, name: str, group_names: list[str], vertices):
        super().__init__()
        self.name = name
        self.vertex_groups = [FakeGroup(group_name, index) for index, group_name in enumerate(group_names)]
        self.data = FakeMesh(vertices)


class FakeCollection:
    def __init__(self, objects=None, children=None):
        self.objects = list(objects or [])
        self.children = list(children or [])


class LodFallbackTests(unittest.TestCase):
    def test_preview_inherits_missing_group_from_shared_weight_donor(self):
        export_collection = FakeCollection(
            objects=[
                FakeObject(
                    "body",
                    ["10", "11"],
                    [
                        FakeVertex(0, (0.0, 0.0, 0.0), [(0, 0.8), (1, 0.2)]),
                        FakeVertex(1, (1.0, 0.0, 0.0), [(0, 0.6)]),
                    ],
                )
            ]
        )
        manifest = {
            "lod_mapping": [
                {
                    "canonical_global_bone": 10,
                    "lod_record_key": "",
                    "lod_local_bone": -1,
                    "status": "unmatched",
                },
                {
                    "canonical_global_bone": 11,
                    "lod_record_key": "lod_eye-3-0",
                    "lod_local_bone": 4,
                    "status": "matched",
                },
            ]
        }

        preview = lod_fallback.preview_lod_fallbacks_for_export(export_collection, manifest)

        self.assertEqual(preview["unmatched_used_global_bones"], [10])
        self.assertEqual(len(preview["fallbacks"]), 1)
        fallback = preview["fallbacks"][0]
        self.assertEqual(fallback["canonical_global_bone"], 10)
        self.assertEqual(fallback["donor_global_bone"], 11)
        self.assertEqual(fallback["lod_record_key"], "lod_eye-3-0")
        self.assertEqual(fallback["lod_local_bone"], 4)
        self.assertEqual(fallback["method"], "shared_vertex_weight")

    def test_unused_unmatched_group_does_not_block_export(self):
        export_collection = FakeCollection(
            objects=[
                FakeObject(
                    "body",
                    ["11"],
                    [FakeVertex(0, (0.0, 0.0, 0.0), [(0, 1.0)])],
                )
            ]
        )
        manifest = {
            "lod_mapping": [
                {"canonical_global_bone": 10, "status": "unmatched", "lod_local_bone": -1},
                {
                    "canonical_global_bone": 11,
                    "lod_record_key": "lod_body-3-0",
                    "lod_local_bone": 1,
                    "status": "matched",
                },
            ]
        }

        preview = lod_fallback.preview_lod_fallbacks_for_export(export_collection, manifest)

        self.assertEqual(preview["unmatched_used_global_bones"], [])
        self.assertEqual(preview["unused_unmatched_global_bones"], [10])

    def test_apply_fallback_updates_mapping_and_capture_pairs(self):
        manifest = {
            "lod_mapping": [
                {"canonical_global_bone": 10, "status": "unmatched", "lod_local_bone": -1},
                {
                    "canonical_global_bone": 11,
                    "lod_record_key": "lod_body-3-0",
                    "lod_local_bone": 1,
                    "status": "matched",
                },
            ],
            "lod_capture_records": [{"lod_record_key": "lod_body-3-0", "scatter_pairs": []}],
        }
        preview = {
            "fallbacks": [
                {
                    "canonical_global_bone": 10,
                    "donor_global_bone": 11,
                    "lod_record_key": "lod_body-3-0",
                    "lod_local_bone": 1,
                    "method": "shared_vertex_weight",
                    "confidence": 0.8,
                }
            ]
        }

        lod_fallback.apply_lod_fallbacks_to_manifest(manifest, preview)

        entry = manifest["lod_mapping"][0]
        self.assertEqual(entry["status"], "fallback_inherited")
        self.assertEqual(entry["donor_global_bone"], 11)
        self.assertEqual(
            manifest["lod_capture_records"][0]["scatter_pairs"][0]["canonical_global_bone"],
            10,
        )


if __name__ == "__main__":
    unittest.main()
