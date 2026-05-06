from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

lod_analyze = importlib.import_module(f"{PACKAGE_DIR.name}.core.lod_analyze")


class LodAnalyzeTests(unittest.TestCase):
    def test_lod_local_bone_can_scatter_to_multiple_canonical_globals(self):
        canonical_points = [
            lod_analyze.WeightedPoint((0.0, 0.0, 0.0), ((10, 1.0),)),
            lod_analyze.WeightedPoint((0.0, 0.1, 0.0), ((11, 1.0),)),
            lod_analyze.WeightedPoint((1.0, 0.0, 0.0), ((20, 1.0),)),
        ]
        lod_points = [
            lod_analyze.WeightedPoint((0.0, 0.0, 0.0), ((("lod_a-3-0", 4), 1.0),)),
            lod_analyze.WeightedPoint((0.0, 0.1, 0.0), ((("lod_a-3-0", 4), 1.0),)),
            lod_analyze.WeightedPoint((1.0, 0.0, 0.0), ((("lod_b-6-0", 2), 1.0),)),
        ]

        result = lod_analyze.build_lod_scatter_mapping(canonical_points, lod_points, 24)

        self.assertEqual(result["global_to_lod"][10]["lod_record_key"], "lod_a-3-0")
        self.assertEqual(result["global_to_lod"][10]["lod_local_bone"], 4)
        self.assertEqual(result["global_to_lod"][11]["lod_record_key"], "lod_a-3-0")
        self.assertEqual(result["global_to_lod"][11]["lod_local_bone"], 4)
        self.assertEqual(result["global_to_lod"][20]["lod_record_key"], "lod_b-6-0")
        self.assertEqual(result["global_to_lod"][20]["lod_local_bone"], 2)

    def test_capture_records_group_scatter_pairs_by_lod_source(self):
        lod_records = {
            "lod_a-3-0": {
                "lod_record_key": "lod_a-3-0",
                "lod_ib_hash": "lod_a",
                "lod_match_first_index": 0,
                "lod_match_index_count": 3,
                "lod_local_bone_count": 5,
            },
        }
        global_to_lod = {
            10: {"lod_record_key": "lod_a-3-0", "lod_local_bone": 4, "score": 1.0, "votes": 3},
            11: {"lod_record_key": "lod_a-3-0", "lod_local_bone": 4, "score": 0.9, "votes": 2},
        }

        records = lod_analyze._build_lod_capture_records(lod_records, global_to_lod)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["lod_record_key"], "lod_a-3-0")
        self.assertEqual(
            [(pair["lod_local_bone"], pair["canonical_global_bone"]) for pair in records[0]["scatter_pairs"]],
            [(4, 10), (4, 11)],
        )


if __name__ == "__main__":
    unittest.main()
