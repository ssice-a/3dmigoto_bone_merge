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

    def test_bone_cloud_mapping_splits_overlapping_lod_local_conflicts(self):
        canonical_points = [
            lod_analyze.WeightedPoint((0.0, 0.0, 0.0), ((124, 0.5), (125, 0.5))),
            lod_analyze.WeightedPoint((0.01, 0.0, 0.0), ((124, 0.5), (125, 0.5))),
            lod_analyze.WeightedPoint((1.0, 0.0, 0.0), ((140, 1.0),)),
        ]
        lod_points = [
            lod_analyze.WeightedPoint(
                (0.0, 0.0, 0.0),
                ((("lod_leg-10-0", 35), 0.5), (("lod_leg-10-0", 36), 0.5)),
            ),
            lod_analyze.WeightedPoint(
                (0.01, 0.0, 0.0),
                ((("lod_leg-10-0", 35), 0.5), (("lod_leg-10-0", 36), 0.5)),
            ),
            lod_analyze.WeightedPoint((1.0, 0.0, 0.0), ((("lod_other-10-0", 1), 1.0),)),
        ]

        result = lod_analyze.build_lod_bone_cloud_mapping(canonical_points, lod_points, 150)

        selected_locals = {
            result["global_to_lod"][124]["lod_local_bone"],
            result["global_to_lod"][125]["lod_local_bone"],
        }
        self.assertEqual({35, 36}, selected_locals)

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

    def test_lod_capture_records_use_point_cloud_candidates_before_vb2_slot_order(self):
        lod_records = {
            "lod_hair-3-0": {
                "lod_record_key": "lod_hair-3-0",
                "lod_ib_hash": "lod_hair",
                "lod_match_first_index": 0,
                "lod_match_index_count": 3,
                "lod_local_bone_count": 3,
                "lod_used_local_bone_indices": [0, 1, 2],
                "vb2_signature": {"slot_count": 3, "used_slots": [0, 1, 2]},
            },
        }
        noisy_candidates = {
            234: [{"lod_record_key": "lod_hair-3-0", "lod_local_bone": 1, "score": 999.0, "votes": 99}],
            235: [{"lod_record_key": "lod_hair-3-0", "lod_local_bone": 2, "score": 999.0, "votes": 99}],
            236: [{"lod_record_key": "lod_hair-3-0", "lod_local_bone": 0, "score": 999.0, "votes": 99}],
        }
        lod_links = [
            {
                "source_key": "main_hair-3-0",
                "ib_hash": "main_hair",
                "match_first_index": 0,
                "match_index_count": 3,
                "global_bone_base": 234,
                "local_bone_count": 3,
                "used_local_bone_indices": [0, 1, 2],
                "lod_sources": [
                    {
                        "lod_record_key": "lod_hair-3-0",
                        "relation_method": "vb2_slot_signature",
                        "score": 100.0,
                        "votes": 3,
                    }
                ],
            },
            {
                "source_key": "main_hair-3-0",
                "ib_hash": "main_hair",
                "match_first_index": 0,
                "match_index_count": 3,
                "global_bone_base": 234,
                "local_bone_count": 3,
                "used_local_bone_indices": [0, 1, 2],
                "lod_sources": [
                    {
                        "lod_record_key": "lod_hair-3-0",
                        "relation_method": "vb2_slot_signature",
                        "score": 100.0,
                        "votes": 3,
                    }
                ],
            },
        ]

        records = lod_analyze._build_lod_capture_records(
            lod_records,
            {},
            lod_links=lod_links,
            global_candidates=noisy_candidates,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["lod_record_key"], "lod_hair-3-0")
        self.assertEqual(
            [(pair["lod_local_bone"], pair["canonical_global_bone"]) for pair in records[0]["scatter_pairs"]],
            [(0, 236), (1, 234), (2, 235)],
        )

    def test_lod_capture_inherits_missing_link_bones_from_same_lod_part(self):
        lod_records = {
            "lod_body-3-0": {
                "lod_record_key": "lod_body-3-0",
                "lod_ib_hash": "lod_body",
                "lod_match_first_index": 0,
                "lod_match_index_count": 3,
            },
        }
        lod_links = [
            {
                "source_key": "main_body-3-0",
                "ib_hash": "main_body",
                "match_first_index": 0,
                "match_index_count": 3,
                "global_bone_base": 10,
                "local_bone_count": 3,
                "used_local_bone_indices": [0, 1, 2],
                "lod_sources": [{"lod_record_key": "lod_body-3-0"}],
            }
        ]
        global_candidates = {
            10: [{"lod_record_key": "lod_body-3-0", "lod_local_bone": 4, "score": 5.0, "votes": 5, "average_distance": 0.01}],
            12: [{"lod_record_key": "lod_body-3-0", "lod_local_bone": 8, "score": 4.0, "votes": 4, "average_distance": 0.02}],
        }

        records = lod_analyze._build_lod_capture_records(
            lod_records,
            {},
            lod_links=lod_links,
            global_candidates=global_candidates,
        )

        self.assertEqual(len(records), 1)
        pairs = {int(pair["canonical_global_bone"]): pair for pair in records[0]["scatter_pairs"]}
        self.assertEqual(4, pairs[10]["lod_local_bone"])
        self.assertEqual(4, pairs[11]["lod_local_bone"])
        self.assertEqual(10, pairs[11]["donor_global_bone"])
        self.assertEqual("same_lod_part_donor", pairs[11]["status"])
        self.assertEqual(8, pairs[12]["lod_local_bone"])

        mapping = lod_analyze._capture_records_to_lod_mapping(
            13,
            records,
            ignored_global_bones=set(range(10)),
        )
        entries = {int(entry["canonical_global_bone"]): entry for entry in mapping["mapping_entries"]}
        self.assertEqual("same_lod_part_donor", entries[11]["status"])
        self.assertEqual("lod_body-3-0", entries[11]["lod_record_key"])
        self.assertEqual(4, entries[11]["lod_local_bone"])
        self.assertEqual(10, entries[11]["donor_global_bone"])
        self.assertNotIn(11, {int(entry["canonical_global_bone"]) for entry in mapping["mapping_entries"] if entry["status"] == "unmatched"})

    def test_lod_signature_fast_path_only_skips_point_cloud_for_same_ib(self):
        canonical_manifest = {
            "frameanalysis_dir": "C:/fake-main",
            "bone_pool_order": [
                {
                    "ib_hash": "main",
                    "match_first_index": 0,
                    "match_index_count": 3,
                    "global_bone_base": 10,
                    "local_bone_count": 3,
                    "used_local_bone_indices": [0, 1, 2],
                    "bone_capture_available": True,
                }
            ],
        }
        lod_manifest = {
            "frameanalysis_dir": "C:/fake-lod",
            "candidate_ibs": [
                {
                    "ib_hash": "main",
                    "match_first_index": 0,
                    "match_index_count": 3,
                    "shadow_capture_ready": True,
                    "shadow_draw_indices": [7],
                    "local_bone_count": 3,
                    "source_local_bone_count": 3,
                    "used_local_bone_indices": [0, 1, 2],
                }
            ],
            "shadow_stage": {},
        }

        result = lod_analyze._try_analyze_lod_by_signatures(
            canonical_manifest,
            lod_manifest,
            "C:/fake-lod",
            lod_level=1,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["lod_frameanalysis"][0]["match_method"], "same_ib_identity_fast_path")
        self.assertEqual(
            [(pair["lod_local_bone"], pair["canonical_global_bone"]) for pair in result["lod_capture_records"][0]["scatter_pairs"]],
            [(0, 10), (1, 11), (2, 12)],
        )

        lod_manifest["candidate_ibs"][0]["ib_hash"] = "lod"
        self.assertIsNone(
            lod_analyze._try_analyze_lod_by_signatures(
                canonical_manifest,
                lod_manifest,
                "C:/fake-lod",
                lod_level=1,
            )
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

    def test_lod_links_do_not_broaden_from_raw_bone_candidates(self):
        main_records = [
            {
                "source_key": "main_hair-200-0",
                "ib_hash": "main_hair",
                "match_first_index": 0,
                "match_index_count": 200,
                "global_bone_base": 200,
                "local_bone_count": 8,
            }
        ]
        lod_records = {
            "lod_noise-1-0": {
                "lod_record_key": "lod_noise-1-0",
                "lod_ib_hash": "lod_noise",
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
        global_candidates = {
            200: [
                {"lod_record_key": "lod_noise-1-0", "lod_local_bone": 9, "score": 9.0, "votes": 9, "average_distance": 0.001},
                {"lod_record_key": "lod_hair-2-0", "lod_local_bone": 0, "score": 3.0, "votes": 3, "average_distance": 0.01},
            ],
            201: [{"lod_record_key": "lod_hair-2-0", "lod_local_bone": 1, "score": 3.0, "votes": 3, "average_distance": 0.01}],
            202: [{"lod_record_key": "lod_hair-2-0", "lod_local_bone": 2, "score": 3.0, "votes": 3, "average_distance": 0.01}],
            203: [{"lod_record_key": "lod_hair-2-0", "lod_local_bone": 3, "score": 3.0, "votes": 3, "average_distance": 0.01}],
        }

        links = lod_analyze._build_lod_links(main_records, lod_records, global_candidates)

        self.assertEqual(links[0]["status"], "matched")
        self.assertEqual(links[0]["lod_sources"][0]["lod_record_key"], "lod_hair-2-0")

        records = lod_analyze._build_lod_capture_records(
            lod_records,
            {
                200: global_candidates[200][0],
                201: global_candidates[201][0],
                202: global_candidates[202][0],
                203: global_candidates[203][0],
            },
        )
        self.assertIn("lod_noise-1-0", {record["lod_record_key"] for record in records})
        self.assertIn("lod_hair-2-0", {record["lod_record_key"] for record in records})

    def test_lod_links_prefer_vb2_slot_signature_over_broad_bone_candidates(self):
        main_records = [
            {
                "source_key": "main_body-100-0",
                "ib_hash": "main_body",
                "match_first_index": 0,
                "match_index_count": 100,
                "global_bone_base": 80,
                "local_bone_count": 67,
                "vb2_signature": {"slot_count": 67, "center": [0.0, 0.0, 0.0], "diag": 1.0},
            },
            {
                "source_key": "main_hair-50-0",
                "ib_hash": "main_hair",
                "match_first_index": 0,
                "match_index_count": 50,
                "global_bone_base": 230,
                "local_bone_count": 23,
                "vb2_signature": {"slot_count": 23, "center": [0.0, 1.0, 0.0], "diag": 0.5},
            },
        ]
        lod_records = {
            "lod_body-70-0": {
                "lod_record_key": "lod_body-70-0",
                "lod_ib_hash": "lod_body",
                "lod_match_first_index": 0,
                "lod_match_index_count": 70,
                "lod_local_bone_count": 70,
                "vb2_signature": {"slot_count": 70, "center": [0.0, 0.0, 0.0], "diag": 1.1},
            }
        }
        global_candidates = {
            **{
                global_bone: [
                    {
                        "lod_record_key": "lod_body-70-0",
                        "lod_local_bone": global_bone - 80,
                        "score": 1.0,
                        "votes": 1,
                        "average_distance": 0.0,
                    }
                ]
                for global_bone in range(80, 147)
            },
            **{
                global_bone: [
                    {
                        "lod_record_key": "lod_body-70-0",
                        "lod_local_bone": global_bone - 230,
                        "score": 1.0,
                        "votes": 1,
                        "average_distance": 0.0,
                    }
                ]
                for global_bone in range(230, 253)
            },
        }

        links = lod_analyze._build_lod_links(main_records, lod_records, global_candidates)

        self.assertEqual(links[0]["status"], "matched")
        self.assertEqual(links[0]["lod_sources"][0]["lod_record_key"], "lod_body-70-0")
        self.assertEqual(links[0]["lod_sources"][0]["relation_method"], "vb2_slot_signature")
        self.assertEqual(links[1]["status"], "unmatched")
        self.assertEqual(links[1]["lod_sources"], [])

    def test_vb2_signature_links_do_not_claim_one_lod_for_many_main_records(self):
        main_records = [
            {
                "source_key": "main_a-100-0",
                "ib_hash": "main_a",
                "match_first_index": 0,
                "match_index_count": 100,
                "global_bone_base": 0,
                "local_bone_count": 32,
                "vb2_signature": {"slot_count": 32, "center": [0.0, 0.0, 0.0], "diag": 1.0},
            },
            {
                "source_key": "main_b-100-0",
                "ib_hash": "main_b",
                "match_first_index": 0,
                "match_index_count": 100,
                "global_bone_base": 64,
                "local_bone_count": 31,
                "vb2_signature": {"slot_count": 31, "center": [3.0, 0.0, 0.0], "diag": 1.0},
            },
        ]
        lod_records = {
            "lod_a-80-0": {
                "lod_record_key": "lod_a-80-0",
                "lod_ib_hash": "lod_a",
                "lod_match_first_index": 0,
                "lod_match_index_count": 80,
                "lod_local_bone_count": 32,
                "vb2_signature": {"slot_count": 32, "center": [0.0, 0.0, 0.0], "diag": 1.0},
            }
        }

        links = lod_analyze._build_lod_links_from_vb2_signatures(main_records, lod_records)

        self.assertEqual(links[0]["status"], "matched")
        self.assertEqual(links[0]["lod_sources"][0]["lod_record_key"], "lod_a-80-0")
        self.assertEqual(links[1]["status"], "unmatched")
        self.assertEqual(links[1]["lod_sources"], [])

    def test_lod_links_are_split_by_recognized_chains(self):
        main_records = [
            {
                "source_key": "main_body-100-0",
                "ib_hash": "main_body",
                "match_first_index": 0,
                "match_index_count": 100,
                "global_bone_base": 0,
                "local_bone_count": 32,
                "vb2_signature": {"slot_count": 32, "center": [0.0, 0.0, 0.0], "diag": 1.0},
            }
        ]
        lod_records = {
            "lod_body_a-80-0": {
                "lod_record_key": "lod_body_a-80-0",
                "lod_ib_hash": "lod_a",
                "lod_match_first_index": 0,
                "lod_match_index_count": 80,
                "lod_capture_draw_indices": [10, 11],
                "lod_local_bone_count": 32,
                "vb2_signature": {"slot_count": 32, "center": [0.0, 0.0, 0.0], "diag": 1.0},
            },
            "lod_body_b-82-0": {
                "lod_record_key": "lod_body_b-82-0",
                "lod_ib_hash": "lod_b",
                "lod_match_first_index": 0,
                "lod_match_index_count": 82,
                "lod_capture_draw_indices": [40],
                "lod_local_bone_count": 33,
                "vb2_signature": {"slot_count": 33, "center": [0.0, 0.0, 0.0], "diag": 1.0},
            },
        }

        chains = lod_analyze._build_lod_record_chains(lod_records)
        links = lod_analyze._build_lod_links(main_records, lod_records, {}, lod_chains=chains)

        self.assertEqual([0, 1], [int(chain["chain_index"]) for chain in chains])
        self.assertEqual(2, len(links))
        self.assertEqual(["matched", "matched"], [link["status"] for link in links])
        self.assertEqual(
            ["lod_body_a-80-0", "lod_body_b-82-0"],
            [link["lod_sources"][0]["lod_record_key"] for link in links],
        )
        self.assertEqual([0, 1], [link["lod_sources"][0]["lod_chain_index"] for link in links])
        self.assertTrue(all(len(link["lod_sources"]) == 1 for link in links))

    def test_lod_chains_split_interleaved_root_instance_signatures(self):
        lod_records = {
            "part_a-10-0": {
                "lod_record_key": "part_a-10-0",
                "lod_ib_hash": "part_a",
                "lod_match_first_index": 0,
                "lod_match_index_count": 10,
                "lod_capture_draw_indices": [10, 30],
                "lod_capture_instance_signatures": {"10": "instance_a", "30": "instance_a"},
            },
            "part_b-20-0": {
                "lod_record_key": "part_b-20-0",
                "lod_ib_hash": "part_b",
                "lod_match_first_index": 0,
                "lod_match_index_count": 20,
                "lod_capture_draw_indices": [11, 31],
                "lod_capture_instance_signatures": {"11": "instance_b", "31": "instance_b"},
            },
            "host_a-30-0": {
                "lod_record_key": "host_a-30-0",
                "lod_ib_hash": "host_a",
                "lod_match_first_index": 0,
                "lod_match_index_count": 30,
                "lod_capture_draw_indices": [12, 32],
                "lod_capture_instance_signatures": {"12": "instance_a", "32": "instance_a"},
            },
            "host_b-40-0": {
                "lod_record_key": "host_b-40-0",
                "lod_ib_hash": "host_b",
                "lod_match_first_index": 0,
                "lod_match_index_count": 40,
                "lod_capture_draw_indices": [13, 33],
                "lod_capture_instance_signatures": {"13": "instance_b", "33": "instance_b"},
            },
        }

        chains = lod_analyze._build_lod_record_chains(lod_records)

        self.assertEqual(4, len(chains))
        self.assertEqual(
            [
                ("instance_a", 10, 12, "host_a"),
                ("instance_b", 11, 13, "host_b"),
                ("instance_a", 30, 32, "host_a"),
                ("instance_b", 31, 33, "host_b"),
            ],
            [
                (
                    chain["instance_signature"],
                    chain["draw_start"],
                    chain["draw_end"],
                    chain["host_key"]["ib_hash"],
                )
                for chain in chains
            ],
        )

    def test_lod_identity_selection_excludes_other_characters_in_mixed_frame(self):
        main_records = [
            {
                "source_key": "main_anchor-100-0",
                "ib_hash": "main_anchor",
                "match_first_index": 0,
                "match_index_count": 100,
            },
            {
                "source_key": "main_body-200-0",
                "ib_hash": "main_body",
                "match_first_index": 0,
                "match_index_count": 200,
            },
        ]
        chains = [
            {
                "chain_index": 0,
                "instance_signature": "target_instance",
                "lod_record_keys": ["main_anchor-100-0", "target_lod-80-0"],
            },
            {
                "chain_index": 1,
                "instance_signature": "other_character_a",
                "lod_record_keys": ["other_a-80-0", "other_a_host-300-0"],
            },
            {
                "chain_index": 2,
                "instance_signature": "other_character_b",
                "lod_record_keys": ["other_b-82-0", "other_b_host-320-0"],
            },
            {
                "chain_index": 3,
                "instance_signature": "target_instance",
                "lod_record_keys": ["main_anchor-100-0", "target_lod_far-60-0"],
            },
        ]

        selected, identity = lod_analyze._select_target_lod_chains(main_records, chains)

        self.assertEqual([0, 3], [chain["chain_index"] for chain in selected])
        self.assertEqual(["target_instance"], identity["selected_instance_signatures"])
        self.assertEqual(
            ["other_character_a", "other_character_b"],
            identity["excluded_instance_signatures"],
        )
        self.assertEqual("exact_main_ib_anchor", identity["selection_method"])

    def test_lod_identity_selection_rejects_ambiguous_multi_character_frame(self):
        main_records = [
            {
                "source_key": "main_body-200-0",
                "ib_hash": "main_body",
                "match_first_index": 0,
                "match_index_count": 200,
            }
        ]
        chains = [
            {
                "chain_index": 0,
                "instance_signature": "unknown_a",
                "lod_record_keys": ["lod_a-80-0"],
            },
            {
                "chain_index": 1,
                "instance_signature": "unknown_b",
                "lod_record_keys": ["lod_b-82-0"],
            },
        ]

        with self.assertRaisesRegex(ValueError, "multiple character identities"):
            lod_analyze._select_target_lod_chains(main_records, chains)

    def test_bone_cloud_mapping_keeps_multiple_lod_candidates_for_one_global(self):
        canonical_points = [
            lod_analyze.WeightedPoint((0.0, 0.0, 0.0), ((200, 1.0),)),
            lod_analyze.WeightedPoint((0.0, 0.1, 0.0), ((200, 1.0),)),
        ]
        lod_points = [
            lod_analyze.WeightedPoint((0.0, 0.0, 0.0), ((("lod_noise-1-0", 9), 1.0),)),
            lod_analyze.WeightedPoint((0.0, 0.1, 0.0), ((("lod_hair-2-0", 0), 1.0),)),
        ]

        result = lod_analyze.build_lod_bone_cloud_mapping(canonical_points, lod_points, 256)
        candidate_keys = {
            candidate["lod_record_key"]
            for candidate in result["global_candidates"][200]
        }

        self.assertIn("lod_noise-1-0", candidate_keys)
        self.assertIn("lod_hair-2-0", candidate_keys)

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
