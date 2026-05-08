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
main_analyze = importlib.import_module(f"{PACKAGE_DIR.name}.core.main_analyze")

MAIN_FRAMEANALYSIS = Path(r"E:\XXMI\EFMI\FrameAnalysis-2026-05-05-222451")
LOD_FRAMEANALYSIS = Path(r"E:\XXMI\EFMI\FrameAnalysis-2026-05-05-225007")


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

    def test_bone_cloud_mapping_supports_lod_local_to_multiple_globals(self):
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

        result = lod_analyze.build_lod_bone_cloud_mapping(canonical_points, lod_points, 24)

        self.assertEqual(result["global_to_lod"][10]["lod_record_key"], "lod_a-3-0")
        self.assertEqual(result["global_to_lod"][10]["lod_local_bone"], 4)
        self.assertEqual(result["global_to_lod"][11]["lod_record_key"], "lod_a-3-0")
        self.assertEqual(result["global_to_lod"][11]["lod_local_bone"], 4)
        self.assertEqual(result["global_to_lod"][20]["lod_record_key"], "lod_b-6-0")
        self.assertEqual(result["global_to_lod"][20]["lod_local_bone"], 2)

    def test_bone_cloud_mapping_keeps_small_bone_samples(self):
        canonical_points = []
        lod_points = []
        for index in range(40):
            position = (index * 0.001, 0.0, 0.0)
            canonical_points.append(lod_analyze.WeightedPoint(position, ((254, 1.0),)))
            lod_points.append(lod_analyze.WeightedPoint(position, ((("lod_finger-8-0", 126), 1.0),)))
        for index in range(80):
            position = (1.0 + index * 0.001, 0.0, 0.0)
            canonical_points.append(lod_analyze.WeightedPoint(position, ((100, 1.0),)))
            lod_points.append(lod_analyze.WeightedPoint(position, ((("lod_hand-8-0", 5), 1.0),)))

        result = lod_analyze.build_lod_bone_cloud_mapping(canonical_points, lod_points, 300)

        self.assertEqual(result["global_to_lod"][254]["lod_record_key"], "lod_finger-8-0")
        self.assertEqual(result["global_to_lod"][254]["lod_local_bone"], 126)

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

    def test_lod_links_drop_sparse_noise_sources(self):
        main_records = [
            {
                "source_key": "main_body-100-0",
                "ib_hash": "main_body",
                "match_first_index": 0,
                "match_index_count": 100,
                "global_bone_base": 100,
                "local_bone_count": 33,
            },
            {
                "source_key": "main_hair-200-0",
                "ib_hash": "main_hair",
                "match_first_index": 0,
                "match_index_count": 200,
                "global_bone_base": 200,
                "local_bone_count": 8,
            },
        ]
        lod_records = {
            "lod_sparse-1-0": {
                "lod_record_key": "lod_sparse-1-0",
                "lod_ib_hash": "lod_sparse",
                "lod_match_first_index": 0,
                "lod_match_index_count": 1,
            },
            "lod_hair-2-0": {
                "lod_record_key": "lod_hair-2-0",
                "lod_ib_hash": "lod_hair",
                "lod_match_first_index": 0,
                "lod_match_index_count": 2,
            },
        }
        global_to_lod = {
            100: {"lod_record_key": "lod_sparse-1-0", "lod_local_bone": 0, "score": 1.0, "votes": 4},
            101: {"lod_record_key": "lod_sparse-1-0", "lod_local_bone": 1, "score": 1.0, "votes": 4},
            102: {"lod_record_key": "lod_sparse-1-0", "lod_local_bone": 2, "score": 1.0, "votes": 4},
            200: {"lod_record_key": "lod_hair-2-0", "lod_local_bone": 0, "score": 1.0, "votes": 4},
            201: {"lod_record_key": "lod_hair-2-0", "lod_local_bone": 1, "score": 1.0, "votes": 4},
            202: {"lod_record_key": "lod_hair-2-0", "lod_local_bone": 2, "score": 1.0, "votes": 4},
            203: {"lod_record_key": "lod_hair-2-0", "lod_local_bone": 3, "score": 1.0, "votes": 4},
        }

        links = lod_analyze._build_lod_links(main_records, lod_records, global_to_lod)

        self.assertEqual(links[0]["source_key"], "main_body-100-0")
        self.assertEqual(links[0]["status"], "unmatched")
        self.assertEqual(links[0]["lod_sources"], [])
        self.assertEqual(links[1]["source_key"], "main_hair-200-0")
        self.assertEqual(links[1]["status"], "matched")
        self.assertEqual(links[1]["lod_sources"][0]["lod_record_key"], "lod_hair-2-0")

    def test_review_blocks_when_global_pool_is_not_filled(self):
        manifest = {
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_index_count": 10,
                    "match_first_index": 0,
                    "global_bone_base": 0,
                    "local_bone_count": 3,
                    "bone_capture_available": True,
                }
            ]
        }
        capture_records = [
            {
                "lod_record_key": "lod_a-3-0",
                "scatter_pairs": [
                    {"lod_local_bone": 0, "canonical_global_bone": 0},
                    {"lod_local_bone": 1, "canonical_global_bone": 2},
                ],
            }
        ]

        review = lod_analyze.review_lod_global_pool_coverage(manifest, capture_records)

        self.assertFalse(review["runtime_safe"])
        self.assertEqual(review["filled_global_bone_count"], 2)
        self.assertEqual(review["missing_global_bone_count"], 1)
        self.assertEqual(review["missing_capture_ready_count"], 1)
        self.assertEqual(review["missing_by_record"][0]["missing_global_bones"], [1])
        self.assertEqual(review["validation"][0]["severity"], "error")

    def test_review_accepts_fully_filled_global_pool(self):
        manifest = {
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_index_count": 10,
                    "match_first_index": 0,
                    "global_bone_base": 4,
                    "local_bone_count": 2,
                    "bone_capture_available": True,
                }
            ]
        }
        capture_records = [
            {
                "lod_record_key": "lod_a-3-0",
                "scatter_pairs": [
                    {"lod_local_bone": 0, "canonical_global_bone": 4},
                    {"lod_local_bone": 0, "canonical_global_bone": 5},
                ],
            }
        ]

        review = lod_analyze.review_lod_global_pool_coverage(manifest, capture_records)

        self.assertTrue(review["runtime_safe"])
        self.assertEqual(review["missing_global_bone_count"], 0)
        self.assertEqual(review["validation"], [])

    def test_review_ignores_lod_excluded_dynamic_vb0_records(self):
        manifest = {
            "bone_pool_order": [
                {
                    "ib_hash": "dynamic1",
                    "match_index_count": 10,
                    "match_first_index": 0,
                    "global_bone_base": 0,
                    "local_bone_count": 3,
                    "bone_capture_available": True,
                    "lod_match_excluded": True,
                    "lod_match_excluded_reason": "dynamic_vb0_backing_hash_mismatch",
                },
                {
                    "ib_hash": "static01",
                    "match_index_count": 20,
                    "match_first_index": 0,
                    "global_bone_base": 3,
                    "local_bone_count": 1,
                    "bone_capture_available": True,
                },
            ]
        }
        capture_records = [
            {
                "lod_record_key": "lod_a-3-0",
                "scatter_pairs": [
                    {"lod_local_bone": 0, "canonical_global_bone": 3},
                ],
            }
        ]

        review = lod_analyze.review_lod_global_pool_coverage(manifest, capture_records)

        self.assertTrue(review["runtime_safe"])
        self.assertEqual(review["required_global_bone_count"], 1)
        self.assertEqual(review["ignored_lod_global_bone_count"], 3)
        self.assertEqual(review["missing_global_bone_count"], 0)
        self.assertEqual(review["ignored_by_record"][0]["source_key"], "dynamic1-10-0")


@unittest.skipUnless(
    MAIN_FRAMEANALYSIS.exists() and LOD_FRAMEANALYSIS.exists(),
    "main/LOD FrameAnalysis folders are not available",
)
class RealLodAnalyzeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_manifest = main_analyze.analyze_main_frameanalysis(str(MAIN_FRAMEANALYSIS))
        cls.lod_result = lod_analyze.analyze_lod_for_manifest(cls.main_manifest, str(LOD_FRAMEANALYSIS))

    def test_transparent_tail_lod_capture_maps_cb71_local19_to_38b8_local27(self):
        mapping = next(
            entry
            for entry in self.lod_result["lod_mapping"]
            if int(entry.get("canonical_global_bone", -1)) == 405
        )
        self.assertEqual(mapping["status"], "matched")
        self.assertEqual(mapping["lod_record_key"], "38b8d614-7560-0")
        self.assertEqual(mapping["lod_local_bone"], 27)

        capture_record = next(
            record
            for record in self.lod_result["lod_capture_records"]
            if record.get("lod_record_key") == "38b8d614-7560-0"
        )
        self.assertIn(39, capture_record["lod_capture_draw_indices"])
        self.assertTrue(
            any(
                int(pair.get("lod_local_bone", -1)) == 27
                and int(pair.get("canonical_global_bone", -1)) == 405
                for pair in capture_record["scatter_pairs"]
            )
        )


if __name__ == "__main__":
    unittest.main()
