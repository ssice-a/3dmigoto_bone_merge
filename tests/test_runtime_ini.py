import struct
import tempfile
import unittest
import importlib
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

ini_export = importlib.import_module(f"{PACKAGE_DIR.name}.core.ini_export")
hlsl_assets = importlib.import_module(f"{PACKAGE_DIR.name}.core.hlsl_assets")
models = importlib.import_module(f"{PACKAGE_DIR.name}.core.models")
LocalPaletteRecord = models.LocalPaletteRecord


class RuntimeIniTests(unittest.TestCase):
    def test_current_game_uses_b1_for_capture_and_shadow_but_b2_for_visible_replay(self):
        manifest = {
            "shadow_stage": {
                "shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"],
                "normal_vs_hash": "aaaaaaaaaaaaaaaa",
                "host_ib_hash": "12345678",
                "host_match_first_index": 0,
                "host_match_index_count": 42,
                "host_draw_index": 10,
            },
            "bone_pool_order": [
                {
                    "ib_hash": "12345678",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "12345678",
                    "first_index": 0,
                    "index_count": 42,
                    "pass_role": "normal_shadow",
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="body",
            ib_hash="12345678",
            match_index_count=42,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="12345678-42-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="12345678_42_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("12345678", 42, 0, 3)],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)
            hlsl_dir = Path(hlsl_assets.export_required_hlsl(tmpdir))
            record_shader = (hlsl_dir / "record_bones_cs.hlsl").read_text(encoding="utf-8")
            redirect_shader = (hlsl_dir / "redirect_cb_cs.hlsl").read_text(encoding="utf-8")

        self.assertNotIn("CustomShader_Extract", ini_text)
        self.assertIn("if vs == 200", ini_text)
        self.assertIn("PoolBMCInstanceRegistry[$bmc_instance_uid] = copy vs-cb1", ini_text)
        self.assertIn("ResourceCapturedCB = ref PoolBMCInstanceRegistry[$bmc_instance_uid]", ini_text)
        self.assertIn("vs-cb1->HashRegion($bmc_hash_offset, 64)", ini_text)
        self.assertIn("; delayed normal shadow replay", ini_text)
        self.assertIn("vs-cb1 = ResourceFakeCB", ini_text)
        self.assertIn("(vs == 201 || vs == 202 || vs == 203)", ini_text)
        self.assertIn("ResourceCapturedCB = copy vs-cb2", ini_text)
        self.assertIn("vs-cb2->HashRegion($bmc_hash_offset, 64)", ini_text)
        self.assertIn("vs-cb2 = ResourceFakeCB", ini_text)
        self.assertIn("$bmc_instance_uid == $bmc_slot_uid_0", ini_text)
        self.assertIn("drawindexedinstanced = 3,1,0,0,$bmc_slot_native_0", ini_text)
        self.assertIn("drawindexedinstanced = 3,1,0,0,$bmc_slot_native_1", ini_text)
        visible_section = ini_text[ini_text.index("if (vs == 201 || vs == 202 || vs == 203)") :]
        self.assertNotIn("#PoolBMCInstanceRegistry", visible_section)
        self.assertNotIn("PoolBMCInstanceRegistry = null", ini_text)
        self.assertIn("if $bmc_mapping_valid == 1\n    handling = skip", visible_section)
        self.assertIn("StructuredBuffer<uint4> CapturedCB", record_shader)
        self.assertIn("StructuredBuffer<uint4> CapturedCB", redirect_shader)

    def test_current_game_shader_filters_are_owned_by_core_shaderregex(self):
        manifest = {"shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]}, "bone_pool_order": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [])
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertNotIn("[ShaderOverride", ini_text)
        self.assertNotIn("filter_index =", ini_text)
        self.assertIn("[PoolBMCInstanceRegistry]", ini_text)
        self.assertIn("pool_index_type = fifo", ini_text)

    def test_main_capture_uses_one_static_bone_map(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
            "bone_pool_order": [
                {
                    "ib_hash": "12345678",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 10,
                    "local_bone_count": 3,
                    "used_local_bone_indices": [2, 52, 248],
                    "bone_capture_available": True,
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="chunk",
            ib_hash="12345678",
            match_index_count=42,
            chunk_index=0,
            local_bone_count=3,
            palette_values=(10, 11, 12),
            file_name="12345678-42-0-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="12345678_42_0",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [palette])
            map_path = runtime["buffers"]["capture_bone_maps"][0]["file_path"]

            self.assertEqual(
                (3, 3, 0, 0, 2, 10, 52, 11, 248, 12),
                _read_uints(map_path),
            )

            ini_text = ini_export.build_bonestore_ini_content(runtime)
            part_map_path = Path(tmpdir) / "Buffer" / "12345678-42-0-PartLocalToGlobalBoneMap.buf"
            self.assertEqual((3, 10, 11, 12), _read_uints(str(part_map_path)))
            self.assertNotIn("namespace =", ini_text)
            self.assertNotIn("match_priority", ini_text)
            self.assertNotIn("x102", ini_text)
            self.assertNotIn("ResourceBoneMeta", ini_text)
            self.assertIn("ResourceCaptureBoneMap_12345678_42_0_MAIN", ini_text)
            self.assertNotIn("ResourceMainCaptureSourceLocalBones", ini_text)
            self.assertNotIn("x100 =", ini_text)
            self.assertIn("ResourcePartLocalToGlobalBoneMap_12345678_42_0", ini_text)
            self.assertNotIn("CommandList_BuildLocalBoneBuffer", ini_text)

    def test_lod_scatter_pairs_are_materialized(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
            "lod_manifest_snapshot": {"shadow_stage": {"shadow_vs_hashes": ["bbbbbbbbbbbbbbbb"]}},
            "bone_pool_order": [
                {
                    "ib_hash": "12345678",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 0,
                    "local_bone_count": 2,
                    "used_local_bone_indices": [0, 1],
                    "bone_capture_available": True,
                }
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-a",
                    "lod_ib_hash": "87654321",
                    "lod_match_first_index": 5,
                    "lod_match_index_count": 77,
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [
                        {"lod_local_bone": 0, "canonical_global_bone": 0},
                        {"lod_local_bone": 0, "canonical_global_bone": 1},
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [])
            lod_map_path = next(
                payload["file_path"]
                for payload in runtime["buffers"]["capture_bone_maps"]
                if payload["resource_name"] == "ResourceCaptureBoneMap_87654321_77_5_LOD"
            )
            self.assertEqual(
                (2, 1, 1, 0, 0, 0, 0, 1),
                _read_uints(lod_map_path),
            )

            ini_text = ini_export.build_bonestore_ini_content(runtime)
            self.assertIn("hash = 87654321", ini_text)
            self.assertIn("[TextureOverride_BMC_87654321_77_5_LOD]", ini_text)
            self.assertNotIn("match_first_index = 5", ini_text)
            self.assertIn("cs-t2 = ResourceCaptureBoneMap_87654321_77_5_LOD", ini_text)
            self.assertIn("run = CustomShader_RecordBones", ini_text)
            self.assertNotIn("CustomShader_RecordBonesScatter", ini_text)
            self.assertIn("ResourceCaptureBoneMap_87654321_77_5_LOD", ini_text)
            self.assertNotIn("hash = bbbbbbbbbbbbbbbb", ini_text)

    def test_same_override_key_records_all_capture_maps(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "lod_manifest_snapshot": {"shadow_stage": {"shadow_vs_hashes": ["2222222222222222"]}},
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-same-key",
                    "lod_ib_hash": "aaaaaaaa",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 10,
                    "lod_capture_draw_indices": [20, 21],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                }
            ],
            "lod_links": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-same-key",
                            "lod_ib_hash": "aaaaaaaa",
                            "lod_match_first_index": 0,
                            "lod_match_index_count": 10,
                            "mapped_global_count": 1,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="main",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 6, object_names=["main_mesh"])],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertFalse(runtime["uses_lod_profile_flag"])
        self.assertNotIn("$bmc_profile_lod", ini_text)
        self.assertIn("cs-t2 = ResourceCaptureBoneMap_aaaaaaaa_10_0_MAIN_LOD", ini_text)
        self.assertNotIn("ResourceMainCaptureBoneMap", ini_text)
        self.assertNotIn("ResourceLodCaptureBoneMap", ini_text)
        self.assertNotIn("x102", ini_text)

    def test_same_override_key_emits_all_shadow_replay_without_lod_profile(self):
        manifest = {
            "shadow_stage": {
                "shadow_vs_hashes": ["1111111111111111"],
                "host_ib_hash": "bbbbbbbb",
                "host_match_first_index": 0,
                "host_match_index_count": 20,
                "host_draw_index": 40,
            },
            "lod_manifest_snapshot": {"shadow_stage": {"shadow_vs_hashes": ["2222222222222222"]}},
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                },
                {
                    "draw_index": 40,
                    "ib_hash": "bbbbbbbb",
                    "first_index": 0,
                    "index_count": 20,
                    "pass_role": "normal_shadow",
                },
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-same-key",
                    "lod_ib_hash": "aaaaaaaa",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 10,
                    "lod_capture_draw_indices": [50],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                }
            ],
            "lod_links": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-same-key",
                            "lod_ib_hash": "aaaaaaaa",
                            "lod_match_first_index": 0,
                            "lod_match_index_count": 10,
                            "mapped_global_count": 1,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="main",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 6, object_names=["same_key_mesh"])],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertFalse(runtime["uses_lod_profile_flag"])
        section = ini_text[
            ini_text.index("[TextureOverride_BMC_aaaaaaaa_10_0_MAIN_LOD]") :
            ini_text.index("[TextureOverride_BMC_bbbbbbbb_20_0]")
        ]
        self.assertNotIn("$bmc_profile_lod", section)
        self.assertIn("if vs == 200", section)
        self.assertIn("  run = CustomShader_RecordBones", section)
        self.assertIn("  handling = skip", section)
        self.assertIn("  ps-t0 = ResourceBMCWhiteShadow", section)
        self.assertIn("  ; delayed normal shadow replay", section)

    def test_lod_shadow_host_uses_raw_chain_host_even_when_host_record_is_filtered(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                }
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-source",
                    "lod_ib_hash": "aaaaaaaa",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 10,
                    "lod_capture_draw_indices": [50],
                    "lod_local_bone_count": 2,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                },
                {
                    "lod_record_key": "lod-host",
                    "lod_ib_hash": "bbbbbbbb",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 20,
                    "lod_capture_draw_indices": [60],
                    "lod_local_bone_count": 2,
                    "scatter_pairs": [{"lod_local_bone": 1, "canonical_global_bone": 1}],
                },
            ],
            "lod_links": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-source",
                            "lod_ib_hash": "aaaaaaaa",
                            "lod_match_first_index": 0,
                            "lod_match_index_count": 10,
                            "mapped_global_count": 1,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="main",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 6, object_names=["main_mesh"])],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertFalse(runtime["uses_lod_profile_flag"])
        self.assertEqual(
            {"ib_hash": "bbbbbbbb", "match_first_index": 0, "match_index_count": 20},
            runtime["lod_profile_chains"][0]["host_key"],
        )
        self.assertEqual(["aaaaaaaa"], [record["ib_hash"] for record in runtime["lod_capture_records"]])
        host_section = ini_text[
            ini_text.index("[TextureOverride_BMC_bbbbbbbb_20_0_LOD]") :
            ini_text.index("[Present]")
        ]
        self.assertIn("  draw = from_caller", host_section)
        self.assertNotIn("$bmc_profile_lod", host_section)

    def test_main_only_capture_in_raw_lod_chain_records_without_profile_guard(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "lod_manifest_snapshot": {
                "shadow_stage": {"shadow_vs_hashes": ["2222222222222222"]},
                "candidate_ibs": [
                    {
                        "ib_hash": "cccccccc",
                        "match_first_index": 0,
                        "match_index_count": 30,
                        "shadow_draw_indices": [50],
                        "shadow_capture_ready": True,
                    },
                    {
                        "ib_hash": "eeeeeeee",
                        "match_first_index": 0,
                        "match_index_count": 40,
                        "shadow_draw_indices": [52],
                        "shadow_capture_ready": True,
                    },
                    {
                        "ib_hash": "bbbbbbbb",
                        "match_first_index": 0,
                        "match_index_count": 20,
                        "shadow_draw_indices": [55],
                        "shadow_capture_ready": True,
                    },
                    {
                        "ib_hash": "dddddddd",
                        "match_first_index": 0,
                        "match_index_count": 99,
                        "shadow_draw_indices": [60],
                        "shadow_capture_ready": True,
                    },
                ],
            },
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                },
                {
                    "ib_hash": "bbbbbbbb",
                    "match_first_index": 0,
                    "match_index_count": 20,
                    "global_bone_base": 1,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                },
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                }
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-body-a",
                    "lod_ib_hash": "cccccccc",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 30,
                    "lod_capture_draw_indices": [50],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                },
                {
                    "lod_record_key": "lod-body-b",
                    "lod_ib_hash": "eeeeeeee",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 40,
                    "lod_capture_draw_indices": [52],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 1}],
                },
            ],
            "lod_links": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-body-a",
                            "lod_ib_hash": "cccccccc",
                            "lod_match_first_index": 0,
                            "lod_match_index_count": 30,
                            "mapped_global_count": 1,
                            "lod_chain_index": 0,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="main",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=2,
            palette_values=(0, 1),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 6, object_names=["main_mesh"])],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertFalse(runtime["uses_lod_profile_flag"])
        self.assertEqual([], runtime["lod_profile_capture_guard_keys"])
        main_only_section = ini_text[
            ini_text.index("[TextureOverride_BMC_bbbbbbbb_20_0]") :
            ini_text.index("[TextureOverride_BMC_cccccccc_30_0_LOD]")
        ]
        self.assertNotIn("$bmc_profile_lod", main_only_section)
        self.assertIn("cs-t2 = ResourceCaptureBoneMap_bbbbbbbb_20_0_MAIN", main_only_section)
        self.assertNotIn("ResourceLodCaptureBoneMap", main_only_section)
        self.assertNotIn("$bmc_profile_lod", ini_text)
        self.assertIn("if first_instance + instance_count <= 8", ini_text)
        self.assertIn("draw = from_caller", ini_text)

    def test_lod_capture_records_only_exported_lod_replay_globals(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "lod_manifest_snapshot": {"shadow_stage": {"shadow_vs_hashes": ["2222222222222222"]}},
            "bone_pool_order": [
                {
                    "ib_hash": "main_a",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                },
                {
                    "ib_hash": "main_b",
                    "match_first_index": 0,
                    "match_index_count": 20,
                    "global_bone_base": 1,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                },
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-a",
                    "lod_ib_hash": "lod_a",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 30,
                    "lod_capture_draw_indices": [10],
                    "lod_local_bone_count": 2,
                    "scatter_pairs": [
                        {"lod_local_bone": 0, "canonical_global_bone": 0},
                        {"lod_local_bone": 1, "canonical_global_bone": 1},
                    ],
                }
            ],
            "lod_links": [
                {
                    "ib_hash": "main_a",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-a",
                            "lod_ib_hash": "lod_a",
                            "lod_match_first_index": 0,
                            "lod_match_index_count": 30,
                            "mapped_global_count": 1,
                        }
                    ],
                }
            ],
        }
        palettes = [
            LocalPaletteRecord(
                object_name="a",
                ib_hash="main_a",
                match_index_count=10,
                chunk_index=0,
                local_bone_count=1,
                palette_values=(0,),
                file_name="main_a-10-0_part00-PartLocalToGlobalBoneMap.buf",
                file_path="",
                resource_suffix="main_a_10_0_part00",
                match_first_index=0,
            ),
            LocalPaletteRecord(
                object_name="b",
                ib_hash="main_b",
                match_index_count=20,
                chunk_index=0,
                local_bone_count=1,
                palette_values=(1,),
                file_name="main_b-20-0_part00-PartLocalToGlobalBoneMap.buf",
                file_path="",
                resource_suffix="main_b_20_0_part00",
                match_first_index=0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                palettes,
                [
                    _geometry_record("main_a", 10, 0, 3, object_names=["a"]),
                    _geometry_record("main_b", 20, 0, 3, object_names=["b"]),
                ],
            )
            lod_capture_bone_map = _read_uints(
                next(
                    payload["file_path"]
                    for payload in runtime["buffers"]["capture_bone_maps"]
                    if payload["resource_name"] == "ResourceCaptureBoneMap_lod_a_30_0_LOD"
                )
            )

        self.assertEqual([0, 1], runtime["main_required_global_bones"])
        self.assertEqual([0], runtime["lod_required_global_bones"])
        self.assertEqual(
            [[0], [1]],
            [record["canonical_global_bones"] for record in runtime["capture_records"]],
        )
        self.assertEqual([[0]], [record["canonical_global_bones"] for record in runtime["lod_capture_records"]])
        self.assertEqual(
            (1, 2, 1, 0, 0, 0),
            lod_capture_bone_map,
        )

    def test_lod_capture_records_follow_replayed_collection_palette(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "lod_manifest_snapshot": {"shadow_stage": {"shadow_vs_hashes": ["2222222222222222"]}},
            "bone_pool_order": [
                {
                    "ib_hash": "047e538d",
                    "match_first_index": 0,
                    "match_index_count": 17337,
                    "global_bone_base": 232,
                    "local_bone_count": 2,
                    "used_local_bone_indices": [0, 1],
                    "bone_capture_available": True,
                },
                {
                    "ib_hash": "25937878",
                    "match_first_index": 0,
                    "match_index_count": 8718,
                    "global_bone_base": 276,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                },
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-259",
                    "lod_ib_hash": "25937878",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 8718,
                    "lod_capture_draw_indices": [45],
                    "lod_local_bone_count": 80,
                    "scatter_pairs": [
                        {"lod_local_bone": 10, "canonical_global_bone": 232},
                        {"lod_local_bone": 11, "canonical_global_bone": 233},
                        {"lod_local_bone": 44, "canonical_global_bone": 276},
                    ],
                }
            ],
            "lod_links": [
                {
                    "ib_hash": "25937878",
                    "match_first_index": 0,
                    "match_index_count": 8718,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-259",
                            "lod_ib_hash": "25937878",
                            "lod_match_first_index": 0,
                            "lod_match_index_count": 8718,
                            "mapped_global_count": 3,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="047-under-259",
            ib_hash="25937878",
            match_index_count=8718,
            chunk_index=0,
            local_bone_count=2,
            palette_values=(232, 233),
            file_name="25937878-8718-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="25937878_8718_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("25937878", 8718, 0, 17337, object_names=["047-under-259"])],
            )
            lod_capture_bone_map = _read_uints(
                next(
                    payload["file_path"]
                    for payload in runtime["buffers"]["capture_bone_maps"]
                    if payload["resource_name"] == "ResourceCaptureBoneMap_25937878_8718_0_LOD"
                )
            )

        self.assertEqual([232, 233], runtime["lod_required_global_bones"])
        self.assertEqual(
            [
                {
                    "key": {"ib_hash": "25937878", "match_first_index": 0, "match_index_count": 8718},
                    "canonical_global_bones": [232, 233],
                }
            ],
            runtime["lod_required_global_bones_by_key"],
        )
        self.assertEqual([[232, 233]], [record["canonical_global_bones"] for record in runtime["lod_capture_records"]])
        self.assertEqual(
            (2, 80, 1, 0, 10, 232, 11, 233),
            lod_capture_bone_map,
        )

    def test_lod_replay_uses_lod_hash_but_main_export_geometry(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "lod_manifest_snapshot": {"shadow_stage": {"shadow_vs_hashes": ["2222222222222222"]}},
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 2,
                    "used_local_bone_indices": [0, 1],
                    "bone_capture_available": True,
                }
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-body",
                    "lod_ib_hash": "bbbbbbbb",
                    "lod_match_first_index": 5,
                    "lod_match_index_count": 20,
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [
                        {"lod_local_bone": 0, "canonical_global_bone": 0},
                        {"lod_local_bone": 0, "canonical_global_bone": 1},
                    ],
                }
            ],
            "lod_links": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-body",
                            "lod_ib_hash": "bbbbbbbb",
                            "lod_match_first_index": 5,
                            "lod_match_index_count": 20,
                            "mapped_global_count": 2,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="main",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=2,
            palette_values=(0, 1),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 6, object_names=["body_mesh"])],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertEqual(
            [
                {
                    "lod_key": {"ib_hash": "bbbbbbbb", "match_first_index": 5, "match_index_count": 20},
                    "main_keys": [{"ib_hash": "aaaaaaaa", "match_first_index": 0, "match_index_count": 10}],
                    "geometry": [
                        {
                            "resource_suffix": "aaaaaaaa_10_0_part00",
                            "main_key": {"ib_hash": "aaaaaaaa", "match_first_index": 0, "match_index_count": 10},
                        }
                    ],
                    "geometry_suffixes": ["aaaaaaaa_10_0_part00"],
                }
            ],
            runtime["lod_replay_links"],
        )
        self.assertEqual(
            [
                {
                    "lod_key": {"ib_hash": "bbbbbbbb", "match_first_index": 5, "match_index_count": 20},
                    "main_keys": [{"ib_hash": "aaaaaaaa", "match_first_index": 0, "match_index_count": 10}],
                    "geometry_suffixes": ["aaaaaaaa_10_0_part00"],
                }
            ],
            runtime["lod_key_annotations"],
        )
        lod_section = ini_text[
            ini_text.index("[TextureOverride_BMC_bbbbbbbb_20_5_LOD]") :
            ini_text.index("[Present]")
        ]
        self.assertIn("hash = bbbbbbbb", lod_section)
        self.assertIn("; main:aaaaaaaa\nhash = bbbbbbbb", lod_section)
        self.assertNotIn("hash = bbbbbbbb ;", lod_section)
        self.assertNotIn("LOD maps to main", lod_section)
        self.assertNotIn("LOD replay exported", lod_section)
        self.assertIn("cs-t2 = ResourceCaptureBoneMap_bbbbbbbb_20_5_LOD", lod_section)
        self.assertIn("(vs == 201 || vs == 202 || vs == 203)", lod_section)
        self.assertIn("ResourceCapturedCB = copy vs-cb2", lod_section)
        self.assertIn("; replay aaaaaaaa_10_0_part00", lod_section)
        self.assertIn("ResourcePart_aaaaaaaa_10_0_part00_Index", lod_section)
        self.assertIn("ResourcePartLocalToGlobalBoneMap_aaaaaaaa_10_0_part00", lod_section)
        self.assertNotIn("ResourcePart_bbbbbbbb_20_5_part00_Index", lod_section)
        self.assertIn("; Blender objects: body_mesh\n    drawindexedinstanced", lod_section)

    def test_lod_shadow_replay_skips_lod_hash_and_draws_main_geometry(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "lod_manifest_snapshot": {
                "shadow_stage": {
                    "shadow_vs_hashes": ["2222222222222222"],
                    "host_ib_hash": "bbbbbbbb",
                    "host_match_first_index": 5,
                    "host_match_index_count": 20,
                    "host_draw_index": 50,
                }
            },
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                }
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-body",
                    "lod_ib_hash": "bbbbbbbb",
                    "lod_match_first_index": 5,
                    "lod_match_index_count": 20,
                    "lod_capture_draw_indices": [50],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                }
            ],
            "lod_links": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-body",
                            "lod_ib_hash": "bbbbbbbb",
                            "lod_match_first_index": 5,
                            "lod_match_index_count": 20,
                            "mapped_global_count": 1,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="main",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 6, object_names=["shadow_body"])],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)
            self.assertTrue((Path(tmpdir) / "Texture" / "white.dds").exists())

        self.assertTrue(runtime["lod_shadow_replay_plan"]["enabled"])
        self.assertEqual(
            [{"ib_hash": "bbbbbbbb", "match_first_index": 5, "match_index_count": 20}],
            runtime["lod_shadow_replay_plan"]["skip_keys"],
        )
        lod_section = ini_text[
            ini_text.index("[TextureOverride_BMC_bbbbbbbb_20_5_LOD]") :
            ini_text.index("[Present]")
        ]
        self.assertIn("if vs == 200", lod_section)
        self.assertIn("; main:aaaaaaaa\nhash = bbbbbbbb", lod_section)
        self.assertIn("  handling = skip", lod_section)
        self.assertIn("  ps-t0 = ResourceBMCWhiteShadow", lod_section)
        self.assertIn("; delayed normal shadow replay", lod_section)
        self.assertIn("; replay aaaaaaaa_10_0_part00", lod_section)
        self.assertIn("; Blender objects: shadow_body", lod_section)

    def test_lod_shadow_replay_inherits_missing_same_part_chain_bones(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "lod_manifest_snapshot": {
                "shadow_stage": {
                    "shadow_vs_hashes": ["2222222222222222"],
                    "host_ib_hash": "bbbbbbbb",
                    "host_match_first_index": 5,
                    "host_match_index_count": 20,
                    "host_draw_index": 50,
                }
            },
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 2,
                    "used_local_bone_indices": [0, 1],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                }
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-body",
                    "lod_ib_hash": "bbbbbbbb",
                    "lod_match_first_index": 5,
                    "lod_match_index_count": 20,
                    "lod_capture_draw_indices": [50],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                }
            ],
            "lod_links": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-body",
                            "lod_ib_hash": "bbbbbbbb",
                            "lod_match_first_index": 5,
                            "lod_match_index_count": 20,
                            "mapped_global_count": 1,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="main",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=2,
            palette_values=(0, 1),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 6, object_names=["shadow_body"])],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertTrue(runtime["lod_shadow_replay_plan"]["enabled"])
        self.assertEqual([], runtime["lod_shadow_replay_plan"]["missing_links"])
        self.assertEqual([[0, 1]], [record["canonical_global_bones"] for record in runtime["lod_capture_records"]])
        self.assertEqual(
            [
                {
                    "lod_local_bone": 0,
                    "canonical_global_bone": 1,
                    "donor_global_bone": 0,
                    "method": "same_part_shadow_chain_donor",
                    "geometry_suffix": "aaaaaaaa_10_0_part00",
                }
            ],
            runtime["lod_capture_records"][0]["auto_lod_donor_pairs"],
        )
        lod_section = ini_text[
            ini_text.index("[TextureOverride_BMC_bbbbbbbb_20_5_LOD]") :
            ini_text.index("[Present]")
        ]
        self.assertIn("if vs == 200", lod_section)
        self.assertIn("  run = CustomShader_RecordBones", lod_section)
        self.assertIn("  handling = skip", lod_section)
        self.assertIn("; delayed normal shadow replay", lod_section)

    def test_lod_shadow_replay_blocks_when_chain_has_no_same_part_donor(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "lod_manifest_snapshot": {
                "shadow_stage": {
                    "shadow_vs_hashes": ["2222222222222222"],
                    "host_ib_hash": "bbbbbbbb",
                    "host_match_first_index": 5,
                    "host_match_index_count": 20,
                    "host_draw_index": 50,
                }
            },
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 2,
                    "used_local_bone_indices": [0, 1],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                }
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-body",
                    "lod_ib_hash": "bbbbbbbb",
                    "lod_match_first_index": 5,
                    "lod_match_index_count": 20,
                    "lod_capture_draw_indices": [50],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                }
            ],
            "lod_links": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-body",
                            "lod_ib_hash": "bbbbbbbb",
                            "lod_match_first_index": 5,
                            "lod_match_index_count": 20,
                            "mapped_global_count": 1,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="main",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(1,),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "LOD shadow replay cannot capture"):
                ini_export.materialize_bonestore_runtime(
                    tmpdir,
                    manifest,
                    [palette],
                    [_geometry_record("aaaaaaaa", 10, 0, 6, object_names=["shadow_body"])],
                )

    def test_lod_shadow_replay_uses_shadow_stage_host(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "lod_manifest_snapshot": {
                "shadow_stage": {
                    "shadow_vs_hashes": ["2222222222222222"],
                    "stage_draw_start": 30,
                    "stage_draw_end": 39,
                    "host_ib_hash": "dddddddd",
                    "host_match_first_index": 0,
                    "host_match_index_count": 99,
                    "host_draw_index": 219,
                }
            },
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                }
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-exported",
                    "lod_ib_hash": "bbbbbbbb",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 20,
                    "lod_capture_draw_indices": [31],
                    "lod_capture_instance_signatures": {"31": ["instance_a"]},
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                },
                {
                    "lod_record_key": "lod-host",
                    "lod_ib_hash": "cccccccc",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 30,
                    "lod_capture_draw_indices": [39],
                    "lod_capture_instance_signatures": {"39": ["instance_a"]},
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                },
            ],
            "lod_links": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-exported",
                            "lod_ib_hash": "bbbbbbbb",
                            "lod_match_first_index": 0,
                            "lod_match_index_count": 20,
                            "mapped_global_count": 1,
                        }
                    ],
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="main",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 6, object_names=["shadow_body"])],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertTrue(runtime["lod_shadow_replay_plan"]["enabled"])
        self.assertEqual(
            {"ib_hash": "cccccccc", "match_first_index": 0, "match_index_count": 30},
            runtime["lod_shadow_replay_plan"]["host_key"],
        )
        self.assertEqual(39, runtime["lod_shadow_replay_plan"]["host_draw_index"])
        self.assertEqual("lod_shadow_chain_host", runtime["lod_shadow_replay_plan"]["host_source"])
        self.assertEqual(
            [
                {"ib_hash": "bbbbbbbb", "match_first_index": 0, "match_index_count": 20},
            ],
            runtime["lod_shadow_replay_plan"]["skip_keys"],
        )
        self.assertTrue(runtime["lod_shadow_replay_plan"]["preserve_host_draw"])
        host_section = ini_text[
            ini_text.index("[TextureOverride_BMC_cccccccc_30_0_LOD]") :
            ini_text.index("[Present]")
        ]
        exported_lod_section = ini_text[
            ini_text.index("[TextureOverride_BMC_bbbbbbbb_20_0_LOD]") :
            ini_text.index("[TextureOverride_BMC_cccccccc_30_0_LOD]")
        ]
        self.assertIn("hash = cccccccc", host_section)
        self.assertIn("cs-t2 = ResourceCaptureBoneMap_cccccccc_30_0_LOD", host_section)
        self.assertNotIn("x100 =", host_section)
        self.assertNotIn("  handling = skip", host_section)
        self.assertIn("  draw = from_caller", host_section)
        self.assertIn("; delayed normal shadow replay", host_section)
        self.assertIn("; replay aaaaaaaa_10_0_part00", host_section)
        self.assertIn("$bmc_instance_uid == $bmc_slot_uid_0", host_section)
        self.assertNotIn(
            "if $bmc_slot_uid_0 > 0 && $bmc_slot_native_0 >= 0 && $bmc_slot_native_0 < 8",
            host_section,
        )
        self.assertIn("  handling = skip", exported_lod_section)
        self.assertNotIn("; delayed normal shadow replay", exported_lod_section)

    def test_lod_shadow_coverage_is_scoped_to_root_instance_signature(self):
        records = [
            {
                "record_index": 0,
                "lod_profile_id": "lod1",
                "ib_hash": "aaaaaaaa",
                "match_first_index": 0,
                "match_index_count": 10,
                "capture_draw_indices": [10],
                "capture_instance_signatures": {"10": "instance_a"},
                "canonical_global_bones": [0],
            },
            {
                "record_index": 1,
                "lod_profile_id": "lod1",
                "ib_hash": "bbbbbbbb",
                "match_first_index": 0,
                "match_index_count": 20,
                "capture_draw_indices": [11],
                "capture_instance_signatures": {"11": "instance_b"},
                "canonical_global_bones": [1],
            },
        ]

        available, coverage = ini_export._lod_shadow_available_globals_for_chain(
            records,
            lod_profile_id="lod1",
            instance_signature="instance_a",
            host_draw_index=12,
            stage_draw_start=10,
            stage_draw_end=12,
        )

        self.assertEqual({0}, available)
        self.assertEqual([0], [record["record_index"] for record in coverage])

    def test_same_lod_host_with_different_instance_parts_falls_back_to_native_shadow(self):
        host = {"ib_hash": "aaaaaaaa", "match_first_index": 0, "match_index_count": 10}
        plans = [
            {
                "enabled": True,
                "host_key": host,
                "chain_index": 0,
                "instance_signature": "instance_a",
                "transparent_parts": [],
                "normal_parts": ["part_a"],
                "skip_keys": [host],
            },
            {
                "enabled": True,
                "host_key": host,
                "chain_index": 1,
                "instance_signature": "instance_b",
                "transparent_parts": [],
                "normal_parts": ["part_b"],
                "skip_keys": [host],
            },
        ]

        merged = ini_export._merge_lod_shadow_plans_by_host(plans)

        self.assertEqual(1, len(merged))
        self.assertFalse(merged[0]["enabled"])
        self.assertEqual("ambiguous_lod_host_instance_groups", merged[0]["reason"])
        self.assertEqual([], merged[0]["skip_keys"])

    def test_lod_shadow_replay_links_remain_chain_scoped_for_repeated_lod_key(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["1111111111111111"]},
            "bone_pool_order": [
                {
                    "ib_hash": "mainaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                },
                {
                    "ib_hash": "mainbbbb",
                    "match_first_index": 0,
                    "match_index_count": 20,
                    "global_bone_base": 1,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                },
            ],
            "draw_hits": [
                {
                    "draw_index": 1,
                    "ib_hash": "mainaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                },
                {
                    "draw_index": 2,
                    "ib_hash": "mainbbbb",
                    "first_index": 0,
                    "index_count": 20,
                    "pass_role": "normal_shadow",
                },
            ],
            "lod_capture_records": [
                {
                    "lod_record_key": "lod-a",
                    "lod_ib_hash": "lodsame1",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 30,
                    "lod_capture_draw_indices": [10],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                },
                {
                    "lod_record_key": "lod-host-a",
                    "lod_ib_hash": "hostaaaa",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 40,
                    "lod_capture_draw_indices": [11],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 0}],
                },
                {
                    "lod_record_key": "lod-b",
                    "lod_ib_hash": "lodsame1",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 30,
                    "lod_capture_draw_indices": [40],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 1}],
                },
                {
                    "lod_record_key": "lod-host-b",
                    "lod_ib_hash": "hostbbbb",
                    "lod_match_first_index": 0,
                    "lod_match_index_count": 50,
                    "lod_capture_draw_indices": [41],
                    "lod_local_bone_count": 1,
                    "scatter_pairs": [{"lod_local_bone": 0, "canonical_global_bone": 1}],
                },
            ],
            "lod_links": [
                {
                    "source_key": "main-a",
                    "ib_hash": "mainaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "lod_chain_index": 0,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-a",
                            "lod_ib_hash": "lodsame1",
                            "lod_match_first_index": 0,
                            "lod_match_index_count": 30,
                            "mapped_global_count": 1,
                            "lod_chain_index": 0,
                        }
                    ],
                },
                {
                    "source_key": "main-b",
                    "ib_hash": "mainbbbb",
                    "match_first_index": 0,
                    "match_index_count": 20,
                    "lod_chain_index": 1,
                    "lod_sources": [
                        {
                            "lod_record_key": "lod-b",
                            "lod_ib_hash": "lodsame1",
                            "lod_match_first_index": 0,
                            "lod_match_index_count": 30,
                            "mapped_global_count": 1,
                            "lod_chain_index": 1,
                        }
                    ],
                },
            ],
        }
        palettes = [
            LocalPaletteRecord(
                object_name="a",
                ib_hash="mainaaaa",
                match_index_count=10,
                chunk_index=0,
                local_bone_count=1,
                palette_values=(0,),
                file_name="mainaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
                file_path="",
                resource_suffix="mainaaaa_10_0_part00",
                match_first_index=0,
            ),
            LocalPaletteRecord(
                object_name="b",
                ib_hash="mainbbbb",
                match_index_count=20,
                chunk_index=0,
                local_bone_count=1,
                palette_values=(1,),
                file_name="mainbbbb-20-0_part00-PartLocalToGlobalBoneMap.buf",
                file_path="",
                resource_suffix="mainbbbb_20_0_part00",
                match_first_index=0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                palettes,
                [
                    _geometry_record("mainaaaa", 10, 0, 3, object_names=["a"]),
                    _geometry_record("mainbbbb", 20, 0, 3, object_names=["b"]),
                ],
            )

        enabled_plans = [plan for plan in runtime["lod_shadow_replay_plans"] if plan.get("enabled")]
        self.assertEqual(2, len(enabled_plans))
        self.assertEqual(
            ["hostaaaa", "hostbbbb"],
            [plan["host_key"]["ib_hash"] for plan in enabled_plans],
        )
        self.assertEqual(
            [["mainaaaa_10_0_part00"], ["mainbbbb_20_0_part00"]],
            [plan["normal_parts"] for plan in enabled_plans],
        )

    def test_generated_ini_uses_single_lifecycle_commandlist_only(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
            "bone_pool_order": [
                {
                    "ib_hash": "12345678",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="chunk",
            ib_hash="12345678",
            match_index_count=42,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="12345678-42-0-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="12345678_42_0",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [palette])
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertIn("[Present]", ini_text)
        self.assertIn("[CommandList_BMC_FrameEndReset]", ini_text)
        self.assertEqual(ini_text.count("[CommandList_"), 1)
        self.assertIn("ResourceGlobalBonePool_UAV", ini_text)
        self.assertIn("ResourceLocalBonePool_UAV", ini_text)
        self.assertIn("ResourceRuntimeState_UAV", ini_text)
        self.assertNotIn("ResourceFakeT0", ini_text)
        self.assertNotIn("ResourceLocalFakeT0", ini_text)
        self.assertNotIn("PoolBMCInstanceRegistry = null", ini_text)
        for resource_name in (
            "ResourceGlobalBonePool_UAV",
            "ResourceLocalBonePool_UAV",
            "ResourceInstanceMapping_UAV",
        ):
            section = ini_text[ini_text.index(f"[{resource_name}]") :]
            section = section[: section.index("\n\n")]
            self.assertIn("bind_flags = shader_resource unordered_access", section)

    def test_visible_replay_is_inlined_with_export_geometry(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
            "bone_pool_order": [
                {
                    "ib_hash": "12345678",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 0,
                    "local_bone_count": 2,
                    "used_local_bone_indices": [0, 1],
                    "bone_capture_available": True,
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="chunk",
            ib_hash="12345678",
            match_index_count=42,
            chunk_index=0,
            local_bone_count=2,
            palette_values=(0, 1),
            file_name="12345678-42-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="12345678_42_0_part00",
        )
        geometry = [
            {
                "ib_hash": "12345678",
                "match_first_index": 0,
                "match_index_count": 42,
                "part_index": 0,
                "index_buffer": {
                    "file_name": "12345678-42-0_part00-Index.buf",
                    "file_path": "Buffer/12345678-42-0_part00-Index.buf",
                    "index_count": 6,
                },
                "vertex_buffers": {
                    "vb0": {
                        "role": "Position",
                        "file_name": "12345678-42-0_part00-Position.buf",
                        "file_path": "Buffer/12345678-42-0_part00-Position.buf",
                        "stride": 16,
                    },
                    "vb1": {
                        "role": "Texcoord",
                        "file_name": "12345678-42-0_part00-Texcoord.buf",
                        "file_path": "Buffer/12345678-42-0_part00-Texcoord.buf",
                        "stride": 20,
                    },
                    "vb2": {
                        "role": "Blend",
                        "file_name": "12345678-42-0_part00-Blend.buf",
                        "file_path": "Buffer/12345678-42-0_part00-Blend.buf",
                        "stride": 12,
                    },
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [palette], geometry)
            ini_text = ini_export.build_bonestore_ini_content(runtime)
            part_map_path = Path(tmpdir) / "Buffer" / "12345678-42-0_part00-PartLocalToGlobalBoneMap.buf"
            part_map_values = _read_uints(str(part_map_path))

        self.assertIn("[ResourcePart_12345678_42_0_part00_Position]", ini_text)
        self.assertIn("[ResourcePart_12345678_42_0_part00_Index]", ini_text)
        self.assertIn("(vs == 201 || vs == 202 || vs == 203)", ini_text)
        self.assertIn("  handling = skip", ini_text)
        self.assertIn("if vs == 200", ini_text)
        self.assertIn("PoolBMCInstanceRegistry[$bmc_instance_uid] = copy vs-cb1", ini_text)
        self.assertIn("cs-t2 = ResourceCaptureBoneMap_12345678_42_0_MAIN", ini_text)
        self.assertIn("run = CustomShader_RecordBones", ini_text)
        self.assertNotIn("x100 =", ini_text)
        self.assertNotIn("visible fallback main bone capture", ini_text)
        self.assertNotIn("x101 =", ini_text)
        self.assertEqual((2, 0, 1), part_map_values)
        self.assertIn("  run = CustomShader_GatherLocalBones", ini_text)
        self.assertIn("  vs-t0 = ResourceLocalBonePool_SRV", ini_text)
        self.assertIn("  vb3 = ResourcePart_12345678_42_0_part00_Position", ini_text)
        self.assertIn("  drawindexedinstanced = 6,INSTANCE_COUNT,0,0,FIRST_INSTANCE", ini_text)

    def test_visible_capture_only_sources_refresh_pool_without_replay(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
            "bone_pool_order": [
                {
                    "ib_hash": "12345678",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 0,
                    "local_bone_count": 2,
                    "used_local_bone_indices": [0, 1],
                    "bone_capture_available": True,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [])
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        section = ini_text[ini_text.index("[TextureOverride_BMC_12345678_42_0]") :]
        self.assertIn("if vs == 200", section)
        self.assertIn("PoolBMCInstanceRegistry[$bmc_instance_uid] = copy vs-cb1", section)
        self.assertIn("cs-t2 = ResourceCaptureBoneMap_12345678_42_0_MAIN", section)
        self.assertIn("run = CustomShader_RecordBones", section)
        self.assertNotIn("x100 =", section)
        self.assertNotIn("(vs == 201 || vs == 202 || vs == 203)", section)
        self.assertNotIn("visible fallback main bone capture", section)
        self.assertNotIn("  handling = skip", section)
        self.assertNotIn("  drawindexedinstanced", section)

    def test_visible_replay_uses_fixed_stage_allowlist(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
            "bone_pool_order": [
                {
                    "ib_hash": "12345678",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 0,
                    "local_bone_count": 2,
                    "used_local_bone_indices": [0, 1],
                    "bone_capture_available": True,
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="part",
            ib_hash="12345678",
            match_index_count=42,
            chunk_index=0,
            local_bone_count=2,
            palette_values=(0, 1),
            file_name="12345678-42-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="Buffer/12345678-42-0_part00-PartLocalToGlobalBoneMap.buf",
            resource_suffix="12345678_42_0_part00",
            match_first_index=0,
        )
        geometry = [_geometry_record("12345678", 42, 0, 6)]

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [palette], geometry)
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertNotIn("ShaderOverrideBMCResidualVS", ini_text)
        self.assertNotIn("filter_index = 204", ini_text)
        self.assertIn("(vs == 201 || vs == 202 || vs == 203)", ini_text)

    def test_manifest_shader_filter_rules_do_not_reintroduce_per_mod_hash_lists(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
            "shader_filter_overrides": [
                {
                    "section_prefix": "ShaderOverrideAutoVS",
                    "hash": "bbbbbbbbbbbbbbbb",
                    "filter_index": 205,
                    "exclude_from_visible_replay": True,
                }
            ],
            "bone_pool_order": [
                {
                    "ib_hash": "12345678",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="part",
            ib_hash="12345678",
            match_index_count=42,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="12345678-42-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="Buffer/12345678-42-0_part00-PartLocalToGlobalBoneMap.buf",
            resource_suffix="12345678_42_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("12345678", 42, 0, 6)],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertNotIn("[ShaderOverrideAutoVS_bbbbbbbbbbbbbbbb]", ini_text)
        self.assertNotIn("hash = bbbbbbbbbbbbbbbb", ini_text)
        self.assertNotIn("filter_index = 205", ini_text)
        self.assertIn("(vs == 201 || vs == 202 || vs == 203)", ini_text)
        self.assertNotIn("vs != 205", ini_text)

    def test_shadow_replay_uses_late_host_and_white_resource_for_normal_parts(self):
        manifest = {
            "shadow_stage": {
                "shadow_vs_hashes": ["1111111111111111", "2222222222222222"],
                "normal_vs_hash": "1111111111111111",
                "transparent_vs_hash": "2222222222222222",
                "host_ib_hash": "bbbbbbbb",
                "host_match_first_index": 5,
                "host_match_index_count": 20,
                "host_draw_index": 99,
            },
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                },
                {
                    "ib_hash": "bbbbbbbb",
                    "match_first_index": 5,
                    "match_index_count": 20,
                    "global_bone_base": 1,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                },
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                },
                {
                    "draw_index": 11,
                    "ib_hash": "bbbbbbbb",
                    "first_index": 5,
                    "index_count": 20,
                    "pass_role": "transparent_shadow",
                },
            ],
        }
        palettes = [
            LocalPaletteRecord(
                object_name="normal",
                ib_hash="aaaaaaaa",
                match_index_count=10,
                chunk_index=0,
                local_bone_count=1,
                palette_values=(0,),
                file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
                file_path="",
                resource_suffix="aaaaaaaa_10_0_part00",
                match_first_index=0,
            ),
            LocalPaletteRecord(
                object_name="transparent",
                ib_hash="bbbbbbbb",
                match_index_count=20,
                chunk_index=0,
                local_bone_count=1,
                palette_values=(1,),
                file_name="bbbbbbbb-20-5_part00-PartLocalToGlobalBoneMap.buf",
                file_path="",
                resource_suffix="bbbbbbbb_20_5_part00",
                match_first_index=5,
            ),
        ]
        geometry = [
            _geometry_record("aaaaaaaa", 10, 0, 3),
            _geometry_record("bbbbbbbb", 20, 5, 6),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, palettes, geometry)
            ini_text = ini_export.build_bonestore_ini_content(runtime)
            self.assertTrue((Path(tmpdir) / "Texture" / "white.dds").exists())

        self.assertTrue(runtime["shadow_replay_plan"]["enabled"])
        self.assertIn("[ResourceBMCWhiteShadow]", ini_text)
        self.assertIn("[TextureOverride_BMC_bbbbbbbb_20_5]", ini_text)
        self.assertIn("if vs == 200", ini_text)
        self.assertIn("  handling = skip", ini_text)
        transparent_pos = ini_text.index("; delayed transparent shadow replay")
        white_pos = ini_text.index("ps-t0 = ResourceBMCWhiteShadow")
        normal_pos = ini_text.index("; delayed normal shadow replay")
        self.assertLess(transparent_pos, white_pos)
        self.assertLess(white_pos, normal_pos)
        self.assertIn("; replay aaaaaaaa_10_0_part00", ini_text)
        self.assertIn("; replay bbbbbbbb_20_5_part00", ini_text)

    def test_shadow_replay_preserves_non_replaced_host_draw(self):
        manifest = {
            "shadow_stage": {
                "shadow_vs_hashes": ["1111111111111111"],
                "normal_vs_hash": "1111111111111111",
                "host_ib_hash": "bbbbbbbb",
                "host_match_first_index": 5,
                "host_match_index_count": 20,
                "host_draw_index": 99,
            },
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                },
                {
                    "draw_index": 99,
                    "ib_hash": "bbbbbbbb",
                    "first_index": 5,
                    "index_count": 20,
                    "pass_role": "normal_shadow",
                },
            ],
        }
        palette = LocalPaletteRecord(
            object_name="normal",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 3)],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertTrue(runtime["shadow_replay_plan"]["preserve_host_draw"])
        self.assertEqual(
            [{"ib_hash": "aaaaaaaa", "match_first_index": 0, "match_index_count": 10}],
            runtime["shadow_replay_plan"]["skip_keys"],
        )
        host_section = ini_text[
            ini_text.index("[TextureOverride_BMC_bbbbbbbb_20_5]") :
            ini_text.index("[Present]")
        ]
        self.assertNotIn("  handling = skip", host_section)
        self.assertIn("  draw = from_caller", host_section)
        self.assertLess(
            host_section.index("  draw = from_caller"),
            host_section.index("; delayed normal shadow replay"),
        )

    def test_shadow_replay_keeps_both_roles_when_one_ib_has_two_shadow_passes(self):
        manifest = {
            "shadow_stage": {
                "shadow_vs_hashes": ["1111111111111111", "2222222222222222"],
                "host_ib_hash": "aaaaaaaa",
                "host_match_first_index": 0,
                "host_match_index_count": 10,
            },
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                },
                {
                    "draw_index": 11,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "transparent_shadow",
                },
            ],
        }
        palette = LocalPaletteRecord(
            object_name="part",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [_geometry_record("aaaaaaaa", 10, 0, 3)],
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertEqual(["aaaaaaaa_10_0_part00"], runtime["shadow_replay_plan"]["transparent_parts"])
        self.assertEqual(["aaaaaaaa_10_0_part00"], runtime["shadow_replay_plan"]["normal_parts"])
        expected_replays = 1 + 2 * int(runtime["instance_pool_size"])
        self.assertEqual(ini_text.count("; replay aaaaaaaa_10_0_part00"), expected_replays)

    def test_export_rejects_capture_unavailable_global_groups(self):
        manifest = {
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 2,
                    "used_local_bone_indices": [0, 1],
                    "bone_capture_available": False,
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="bad",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(1,),
            file_name="bad-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="bad",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "capture-unavailable"):
                ini_export.materialize_bonestore_runtime(tmpdir, manifest, [palette])

    def test_texture_marks_export_hash_style_overrides(self):
        manifest = {"shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]}, "bone_pool_order": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            source_texture = Path(tmpdir) / "source_abcd1234.dds"
            source_texture.write_bytes(b"DDS fake-test-texture")
            texture_payload = {
                "version": 1,
                "binding_mode": "texture_hash_override",
                "candidates": {
                    "12345678-42-0": {
                        "99": {
                            "ps-t7": {
                                "slot": "ps-t7",
                                "hash": "abcd1234",
                                "source_path": str(source_texture),
                                "draw_index": 99,
                                "ps_hash": "1111222233334444",
                                "rt_count": 4,
                            }
                        }
                    }
                },
                "marks": {
                    "12345678-42-0": {
                        "99": {
                            "ps-t7": {"semantic": "base_color", "semantic_index": 0}
                        }
                    }
                },
            }

            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [],
                texture_mark_payload=texture_payload,
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

            self.assertEqual(1, len(runtime["textures"]))
            self.assertTrue((Path(tmpdir) / "Texture" / "abcd1234_base_color.dds").is_file())
            self.assertIn("[ResourceBMCTexture_abcd1234]", ini_text)
            self.assertIn("filename = Texture/abcd1234_base_color.dds", ini_text)
            self.assertIn("[TextureOverride_BMCTexture_abcd1234_base_color_ps_t7]", ini_text)
            self.assertIn("hash = abcd1234", ini_text)
            self.assertIn("this = ResourceBMCTexture_abcd1234", ini_text)
            self.assertNotIn("ps-t7 = ResourceBMCTexture_abcd1234", ini_text)

    def test_replay_splits_one_buffer_by_object_draw_ranges(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="part",
            ib_hash="aaaaaaaa",
            match_index_count=42,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-42-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_42_0_part00",
            match_first_index=0,
        )
        geometry = _geometry_record(
            "aaaaaaaa",
            42,
            0,
            9,
            object_names=["body", "hair", "ribbon"],
            object_draws=[
                {"object_name": "body", "start_index": 0, "index_count": 3, "base_vertex": 0},
                {"object_name": "hair", "start_index": 3, "index_count": 3, "base_vertex": 0},
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [palette], [geometry])
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertIn("; Blender objects: body", ini_text)
        self.assertIn("  drawindexedinstanced = 3,INSTANCE_COUNT,0,0,FIRST_INSTANCE", ini_text)
        self.assertIn("; Blender objects: hair", ini_text)
        self.assertIn("  drawindexedinstanced = 3,INSTANCE_COUNT,3,0,FIRST_INSTANCE", ini_text)
        self.assertNotIn("  drawindexedinstanced = 6,INSTANCE_COUNT,0,0,FIRST_INSTANCE", ini_text)

    def test_toggle_draw_sets_emit_keys_and_guard_object_draws(self):
        manifest = {
            "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 42,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
        }
        palette = LocalPaletteRecord(
            object_name="part",
            ib_hash="aaaaaaaa",
            match_index_count=42,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-42-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_42_0_part00",
            match_first_index=0,
        )
        geometry = _geometry_record(
            "aaaaaaaa",
            42,
            0,
            9,
            object_names=["body", "hair", "ribbon"],
            object_draws=[
                {"object_name": "body", "start_index": 0, "index_count": 3, "base_vertex": 0},
                {
                    "object_name": "hair",
                    "start_index": 3,
                    "index_count": 3,
                    "base_vertex": 0,
                    "toggle_group_id": "hair",
                    "toggle_variable": "$bmc_toggle_hair",
                    "toggle_value": 1,
                    "toggle_condition": "$bmc_toggle_hair == 1",
                },
                {
                    "object_name": "ribbon",
                    "start_index": 6,
                    "index_count": 3,
                    "base_vertex": 0,
                    "toggle_group_id": "hair",
                    "toggle_variable": "$bmc_toggle_hair",
                    "toggle_value": 1,
                    "toggle_condition": "$bmc_toggle_hair == 1",
                },
            ],
        )
        toggle_draw_sets = [
            {
                "toggle_id": "hair",
                "label": "Hair",
                "key": "ctrl F6",
                "default_value": 0,
                "values": [
                    {"value": 0, "label": "Off", "objects": []},
                    {"value": 1, "label": "On", "objects": ["hair"]},
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(
                tmpdir,
                manifest,
                [palette],
                [geometry],
                toggle_draw_sets=toggle_draw_sets,
            )
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertIn("[Constants]", ini_text)
        self.assertIn("global persist $bmc_toggle_hair = 0", ini_text)
        self.assertIn("[Key_BMC_Toggle_hair]", ini_text)
        self.assertIn("key = ctrl F6", ini_text)
        self.assertIn("$bmc_toggle_hair = 0,1", ini_text)
        self.assertEqual(1, ini_text.count("if $bmc_toggle_hair == 1"))
        self.assertIn("    drawindexedinstanced = 3,INSTANCE_COUNT,3,0,FIRST_INSTANCE", ini_text)
        self.assertIn("    drawindexedinstanced = 3,INSTANCE_COUNT,6,0,FIRST_INSTANCE", ini_text)

    def test_toggle_shadow_replay_suppresses_raw_caller_shadow(self):
        manifest = {
            "shadow_stage": {
                "shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"],
                "host_ib_hash": "bbbbbbbb",
                "host_match_first_index": 5,
                "host_match_index_count": 20,
            },
            "bone_pool_order": [
                {
                    "ib_hash": "aaaaaaaa",
                    "match_first_index": 0,
                    "match_index_count": 10,
                    "global_bone_base": 0,
                    "local_bone_count": 1,
                    "used_local_bone_indices": [0],
                    "bone_capture_available": True,
                }
            ],
            "draw_hits": [
                {
                    "draw_index": 10,
                    "ib_hash": "aaaaaaaa",
                    "first_index": 0,
                    "index_count": 10,
                    "pass_role": "normal_shadow",
                },
                {
                    "draw_index": 99,
                    "ib_hash": "bbbbbbbb",
                    "first_index": 5,
                    "index_count": 20,
                    "pass_role": "normal_shadow",
                },
            ],
        }
        palette = LocalPaletteRecord(
            object_name="part",
            ib_hash="aaaaaaaa",
            match_index_count=10,
            chunk_index=0,
            local_bone_count=1,
            palette_values=(0,),
            file_name="aaaaaaaa-10-0_part00-PartLocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="aaaaaaaa_10_0_part00",
            match_first_index=0,
        )
        geometry = _geometry_record(
            "aaaaaaaa",
            10,
            0,
            3,
            object_names=["hair"],
            object_draws=[
                {
                    "object_name": "hair",
                    "start_index": 0,
                    "index_count": 3,
                    "base_vertex": 0,
                    "toggle_condition": "$bmc_toggle_hair == 1",
                }
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [palette], [geometry])
            ini_text = ini_export.build_bonestore_ini_content(runtime)

        self.assertTrue(runtime["shadow_replay_plan"]["preserve_host_draw"])
        host_section = ini_text[
            ini_text.index("[TextureOverride_BMC_bbbbbbbb_20_5]") :
            ini_text.index("[Present]")
        ]
        self.assertNotIn("  draw = from_caller", host_section)
        self.assertIn("if vs == 200", host_section)
        self.assertIn("if first_instance + instance_count <= 8\n    handling = skip", host_section)
        self.assertIn("if $bmc_toggle_hair == 1", host_section)


def _read_uints(path: str) -> tuple[int, ...]:
    with open(path, "rb") as handle:
        data = handle.read()
    return struct.unpack("<" + "I" * (len(data) // 4), data)


def _geometry_record(
    ib_hash: str,
    match_index_count: int,
    match_first_index: int,
    index_count: int,
    object_names: list[str] | None = None,
    object_draws: list[dict] | None = None,
) -> dict:
    key = f"{ib_hash}-{match_index_count}-{match_first_index}_part00"
    record = {
        "object_names": list(object_names or []),
        "ib_hash": ib_hash,
        "match_first_index": match_first_index,
        "match_index_count": match_index_count,
        "part_index": 0,
        "index_buffer": {
            "file_name": f"{key}-Index.buf",
            "file_path": f"Buffer/{key}-Index.buf",
            "index_count": index_count,
        },
        "vertex_buffers": {
            "vb0": {
                "role": "Position",
                "file_name": f"{key}-Position.buf",
                "file_path": f"Buffer/{key}-Position.buf",
                "stride": 16,
            },
            "vb1": {
                "role": "Texcoord",
                "file_name": f"{key}-Texcoord.buf",
                "file_path": f"Buffer/{key}-Texcoord.buf",
                "stride": 20,
            },
            "vb2": {
                "role": "Blend",
                "file_name": f"{key}-Blend.buf",
                "file_path": f"Buffer/{key}-Blend.buf",
                "stride": 12,
            },
        },
    }
    if object_draws is not None:
        record["object_draws"] = list(object_draws)
    return record


if __name__ == "__main__":
    unittest.main()
