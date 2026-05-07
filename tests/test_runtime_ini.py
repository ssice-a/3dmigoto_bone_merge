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
models = importlib.import_module(f"{PACKAGE_DIR.name}.core.models")
LocalPaletteRecord = models.LocalPaletteRecord


class RuntimeIniTests(unittest.TestCase):
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
            map_path = runtime["buffers"]["main_capture_bone_map"]["file_path"]

            self.assertEqual(
                (1, 8, 3, 0, 0, 3, 3, 0, 2, 10, 52, 11, 248, 12),
                _read_uints(map_path),
            )

            ini_text = ini_export.build_bonestore_ini_content(runtime)
            self.assertNotIn("namespace =", ini_text)
            self.assertNotIn("match_priority", ini_text)
            self.assertNotIn("x102", ini_text)
            self.assertNotIn("ResourceBoneMeta", ini_text)
            self.assertIn("ResourceMainCaptureBoneMap", ini_text)
            self.assertNotIn("ResourceMainCaptureSourceLocalBones", ini_text)
            self.assertIn("x100 = 0", ini_text)
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
            self.assertEqual(
                (1, 8, 2, 1, 0, 2, 1, 1, 0, 0, 0, 1),
                _read_uints(runtime["buffers"]["lod_capture_bone_map"]["file_path"]),
            )

            ini_text = ini_export.build_bonestore_ini_content(runtime)
            self.assertIn("hash = 87654321", ini_text)
            self.assertIn("[TextureOverride_BMC_87654321_77_5_LOD]", ini_text)
            self.assertNotIn("match_first_index = 5", ini_text)
            self.assertIn("cs-t2 = ResourceLodCaptureBoneMap", ini_text)
            self.assertIn("run = CustomShader_RecordBones", ini_text)
            self.assertNotIn("CustomShader_RecordBonesScatter", ini_text)
            self.assertIn("ResourceLodCaptureBoneMap", ini_text)
            self.assertIn("hash = bbbbbbbbbbbbbbbb", ini_text)

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
        self.assertIn("cs-t2 = ResourceLodCaptureBoneMap", lod_section)
        self.assertIn("if vs != 200", lod_section)
        self.assertIn("; replay aaaaaaaa_10_0_part00", lod_section)
        self.assertIn("ResourcePart_aaaaaaaa_10_0_part00_Index", lod_section)
        self.assertIn("ResourcePartLocalToGlobalBoneMap_aaaaaaaa_10_0_part00", lod_section)
        self.assertNotIn("ResourcePart_bbbbbbbb_20_5_part00_Index", lod_section)
        self.assertIn("; Blender objects: body_mesh\n  drawindexedinstanced", lod_section)

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

        self.assertIn("[ResourcePart_12345678_42_0_part00_Position]", ini_text)
        self.assertIn("[ResourcePart_12345678_42_0_part00_Index]", ini_text)
        self.assertIn("if vs != 200", ini_text)
        self.assertIn("  handling = skip", ini_text)
        self.assertIn("  x101 = 2", ini_text)
        self.assertIn("  run = CustomShader_GatherLocalBones", ini_text)
        self.assertIn("  vs-t0 = ResourceLocalBonePool_SRV", ini_text)
        self.assertIn("  vb3 = ResourcePart_12345678_42_0_part00_Position", ini_text)
        self.assertIn("  drawindexedinstanced = 6,INSTANCE_COUNT,0,0,FIRST_INSTANCE", ini_text)

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
        self.assertEqual(ini_text.count("; replay aaaaaaaa_10_0_part00"), 3)

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
) -> dict:
    key = f"{ib_hash}-{match_index_count}-{match_first_index}_part00"
    return {
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


if __name__ == "__main__":
    unittest.main()
