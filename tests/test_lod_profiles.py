from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

lod_profiles = importlib.import_module(f"{PACKAGE_DIR.name}.core.lod_profiles")
ini_export = importlib.import_module(f"{PACKAGE_DIR.name}.core.ini_export")


def _result(profile_hash: str, source_local: int, *, runtime_safe: bool = True) -> dict:
    record_key = f"{profile_hash}-30-0"
    return {
        "lod_frameanalysis": [{"lod_level": 1, "frameanalysis_dir": f"X:/{profile_hash}"}],
        "lod_links": [
            {
                "source_key": "main-30-0",
                "lod_sources": [{"lod_record_key": record_key}],
            }
        ],
        "lod_chains": [{"chain_index": 0, "lod_record_keys": [record_key]}],
        "lod_capture_records": [
            {
                "lod_record_key": record_key,
                "lod_ib_hash": profile_hash,
                "lod_match_first_index": 0,
                "lod_match_index_count": 30,
                "scatter_pairs": [
                    {"lod_local_bone": source_local, "canonical_global_bone": 4}
                ],
            }
        ],
        "lod_mapping": [
            {
                "canonical_global_bone": 4,
                "lod_record_key": record_key,
                "lod_local_bone": source_local,
                "status": "matched",
            }
        ],
        "lod_review": {
            "runtime_safe": runtime_safe,
            "missing_global_bone_count": 0 if runtime_safe else 1,
        },
        "validation": [],
        "lod_manifest_snapshot": {
            "frameanalysis_dir": f"X:/{profile_hash}",
            "candidate_ibs": [{"ib_hash": profile_hash}],
        },
    }


class LodProfileTests(unittest.TestCase):
    def test_profiles_aggregate_without_overwriting_results(self):
        manifest = {"global_pool_generation": "pool-a", "lod_profiles": []}
        lod_profiles.upsert_lod_profile(
            manifest,
            profile_id="lod1",
            label="LOD 1",
            lod_level=1,
            frameanalysis_dir="X:/lod1",
            enabled=True,
            result=_result("aaaaaaaa", 2),
        )
        lod_profiles.upsert_lod_profile(
            manifest,
            profile_id="lod2",
            label="LOD 2",
            lod_level=2,
            frameanalysis_dir="X:/lod2",
            enabled=True,
            result=_result("bbbbbbbb", 3),
        )

        self.assertEqual(2, len(manifest["lod_profiles"]))
        self.assertEqual(2, len(manifest["lod_capture_records"]))
        self.assertEqual({"lod1", "lod2"}, {item["lod_profile_id"] for item in manifest["lod_mapping"]})
        self.assertEqual(2, len(manifest["lod_manifest_snapshots"]))
        self.assertTrue(manifest["lod_review"]["runtime_safe"])

    def test_disabled_profile_is_not_in_runtime_aggregate(self):
        manifest = {"global_pool_generation": "pool-a", "lod_profiles": []}
        for profile_id, enabled in (("lod1", True), ("lod2", False)):
            lod_profiles.upsert_lod_profile(
                manifest,
                profile_id=profile_id,
                label=profile_id,
                lod_level=1,
                frameanalysis_dir=f"X:/{profile_id}",
                enabled=enabled,
                result=_result("aaaaaaaa" if enabled else "bbbbbbbb", 2),
            )

        self.assertEqual(["lod1"], [item["lod_profile_id"] for item in manifest["lod_capture_records"]])

    def test_pool_change_invalidates_profiles_and_blocks_export(self):
        manifest = {"global_pool_generation": "pool-a", "lod_profiles": []}
        lod_profiles.upsert_lod_profile(
            manifest,
            profile_id="lod1",
            label="LOD 1",
            lod_level=1,
            frameanalysis_dir="X:/lod1",
            enabled=True,
            result=_result("aaaaaaaa", 2),
        )

        manifest["global_pool_generation"] = "pool-b"
        lod_profiles.invalidate_lod_profiles(manifest, "global_bone_pool_changed")

        self.assertEqual([], manifest["lod_capture_records"])
        self.assertTrue(manifest["lod_profiles"][0]["stale"])
        with self.assertRaisesRegex(ValueError, "analysis is stale"):
            lod_profiles.assert_lod_profiles_exportable(manifest)

    def test_missing_profile_pool_generation_is_stale_when_pool_exists(self):
        profile = {
            "profile_id": "lod1",
            "enabled": True,
            "stale": False,
            "global_pool_generation": "",
            "result": _result("aaaaaaaa", 2),
        }

        self.assertTrue(lod_profiles.lod_profile_is_stale(profile, "pool-a"))

    def test_profile_issue_prefers_error_over_earlier_warning(self):
        result = {
            "validation": [
                {"severity": "warning", "message": "partial match"},
                {"severity": "error", "message": "pool is incomplete"},
            ]
        }

        self.assertEqual("pool is incomplete", lod_profiles.first_lod_profile_issue(result))

    def test_changing_profile_source_marks_analysis_stale(self):
        manifest = {"global_pool_generation": "pool-a", "lod_profiles": []}
        lod_profiles.upsert_lod_profile(
            manifest,
            profile_id="lod1",
            label="LOD 1",
            lod_level=1,
            frameanalysis_dir="X:/lod1",
            enabled=True,
            result=_result("aaaaaaaa", 2),
        )

        lod_profiles.sync_lod_profile_settings(
            manifest,
            [
                {
                    "profile_id": "lod1",
                    "label": "LOD 1",
                    "lod_level": 2,
                    "frameanalysis_dir": "X:/lod2",
                    "enabled": True,
                }
            ],
        )

        self.assertTrue(manifest["lod_profiles"][0]["stale"])
        self.assertEqual([], manifest["lod_capture_records"])

    def test_enabled_configured_profile_must_be_analyzed_before_export(self):
        manifest = {
            "lod_profiles": [
                {
                    "profile_id": "lod1",
                    "label": "LOD 1",
                    "lod_level": 1,
                    "frameanalysis_dir": "X:/lod1",
                    "enabled": True,
                    "result": {},
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "have not been analyzed"):
            lod_profiles.assert_lod_profiles_exportable(manifest)

    def test_same_override_key_with_conflicting_source_bones_is_blocked(self):
        manifest = {"global_pool_generation": "pool-a", "lod_profiles": []}
        lod_profiles.upsert_lod_profile(
            manifest,
            profile_id="lod1",
            label="LOD 1",
            lod_level=1,
            frameanalysis_dir="X:/lod1",
            enabled=True,
            result=_result("aaaaaaaa", 2),
        )
        lod_profiles.upsert_lod_profile(
            manifest,
            profile_id="lod2",
            label="LOD 2",
            lod_level=2,
            frameanalysis_dir="X:/lod2",
            enabled=True,
            result=_result("aaaaaaaa", 7),
        )

        self.assertEqual(1, len(manifest["lod_profile_conflicts"]))
        with self.assertRaisesRegex(ValueError, "cannot distinguish"):
            lod_profiles.assert_lod_profiles_exportable(manifest)

    def test_runtime_chains_remain_separate_when_profile_draw_indices_overlap(self):
        manifest = {"global_pool_generation": "pool-a", "lod_profiles": []}
        for profile_id, ib_hash in (("lod1", "aaaaaaaa"), ("lod2", "bbbbbbbb")):
            result = _result(ib_hash, 2)
            result["lod_capture_records"][0]["lod_capture_draw_indices"] = [10, 11]
            result["lod_chains"] = [
                {
                    "chain_index": 0,
                    "draw_start": 10,
                    "draw_end": 11,
                    "host_draw_index": 11,
                    "host_key": {
                        "ib_hash": ib_hash,
                        "match_first_index": 0,
                        "match_index_count": 30,
                    },
                    "lod_record_keys": [f"{ib_hash}-30-0"],
                }
            ]
            lod_profiles.upsert_lod_profile(
                manifest,
                profile_id=profile_id,
                label=profile_id,
                lod_level=1,
                frameanalysis_dir=f"X:/{profile_id}",
                enabled=True,
                result=result,
            )

        runtime_records = ini_export._build_lod_capture_records(manifest)
        chains = ini_export._build_lod_profile_chains(manifest, runtime_records)

        self.assertEqual(2, len(chains))
        self.assertEqual({"lod1", "lod2"}, {chain["lod_profile_id"] for chain in chains})
        self.assertEqual({"aaaaaaaa", "bbbbbbbb"}, {chain["host_key"]["ib_hash"] for chain in chains})

    def test_shadow_donor_index_does_not_cross_profile_with_same_draw_index(self):
        records = [
            {
                "lod_profile_id": "lod1",
                "capture_draw_indices": [10],
                "capture_pairs": [(2, 4)],
            },
            {
                "lod_profile_id": "lod2",
                "capture_draw_indices": [10],
                "capture_pairs": [(7, 9)],
            },
        ]

        donors = ini_export._lod_shadow_chain_donor_index(
            records,
            lod_profile_id="lod1",
            host_draw_index=11,
            stage_draw_start=10,
            stage_draw_end=11,
        )

        self.assertEqual({4}, set(donors))

    def test_legacy_single_lod_manifest_is_migrated(self):
        manifest = {
            "global_pool_generation": "pool-a",
            **_result("aaaaaaaa", 2),
        }
        manifest["lod_validation"] = manifest.pop("validation")

        profiles = lod_profiles.ensure_lod_profiles(manifest)
        lod_profiles.rebuild_lod_aggregate(manifest)

        self.assertEqual(1, len(profiles))
        self.assertEqual(1, len(manifest["lod_capture_records"]))
        self.assertTrue(manifest["lod_capture_records"][0]["lod_profile_id"])


if __name__ == "__main__":
    unittest.main()
