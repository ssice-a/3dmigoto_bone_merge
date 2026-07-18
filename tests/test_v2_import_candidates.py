from __future__ import annotations

import importlib
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

main_analyze = importlib.import_module(f"{PACKAGE_DIR.name}.core.main_analyze")
import_candidates = importlib.import_module(f"{PACKAGE_DIR.name}.core.import_candidates")
uv_transform = importlib.import_module(f"{PACKAGE_DIR.name}.core.uv_transform")


SAMPLE_FRAMEANALYSIS = Path(r"E:\XXMI\EFMI\FrameAnalysis-2026-05-05-225007")
MAIN_FRAMEANALYSIS = Path(r"E:\XXMI\EFMI\FrameAnalysis-2026-05-05-222451")
CPU_SKIN_FRAMEANALYSIS = Path(r"E:\XXMI\EFMI\FrameAnalysis-2026-07-17-141635")


class SyntheticCandidateImportTests(unittest.TestCase):
    def test_cpu_pre_skinned_mode_requires_all_independent_previous_position_evidence(self):
        payload = {
            "ib_backing_hash": "9a09f1f0",
            "vb0_backing_hash": "1d6a6186",
            "vb3_distinct": True,
            "texcoord4_input_slot": 3,
            "texcoord4_format": "R32G32B32_FLOAT",
            "cb2_skinning_flags": 0x10,
        }

        result = main_analyze._classify_skinning_mode(payload)

        self.assertEqual(result["kind"], "cpu_pre_skinned")
        self.assertEqual(result["confidence"], "high")
        self.assertTrue(result["uses_external_previous_position"])

    def test_dynamic_vertex_stream_without_external_previous_position_is_not_cpu_pre_skinned(self):
        payload = {
            "ib_backing_hash": "9a09f1f0",
            "vb0_backing_hash": "1d6a6186",
            "vb3_distinct": True,
            "texcoord4_input_slot": 3,
            "texcoord4_format": "R32G32B32_FLOAT",
            "cb2_skinning_flags": 0x31,
        }

        result = main_analyze._classify_skinning_mode(payload)

        self.assertEqual(result["kind"], "dynamic_vertex_stream")
        self.assertFalse(result["uses_external_previous_position"])

    def test_cpu_skinned_import_name_is_explicit_and_idempotent(self):
        base_name = "945c08a9-1698-0"
        marked = import_candidates._import_object_name(base_name, False)

        self.assertEqual(marked, "945c08a9-1698-0 [CPU_SKINNED_UNSUPPORTED]")
        self.assertEqual(import_candidates._import_object_name(marked, False), marked)
        self.assertEqual(import_candidates._import_object_name(base_name, True), base_name)

    def test_uv_v_transform_is_symmetric_between_import_and_export(self):
        game_uv = (0.125, 0.75)
        blender_uv = uv_transform.game_uv_to_blender(game_uv)
        self.assertEqual(blender_uv, (0.125, 0.25))
        self.assertEqual(uv_transform.blender_uv_to_game(blender_uv), game_uv)

    def test_uv_v_transform_can_be_disabled(self):
        uv = (0.125, 0.75)
        self.assertEqual(uv_transform.game_uv_to_blender(uv, flip_v=False), uv)
        self.assertEqual(uv_transform.blender_uv_to_game(uv, flip_v=False), uv)

    def test_bone_pool_order_places_capture_ready_candidates_before_mapping_only_candidates(self):
        candidates = [
            {
                "ib_hash": "bbbbbbbb",
                "match_first_index": 0,
                "match_index_count": 100,
                "local_bone_count": 5,
                "used_local_bone_indices": [2, 8],
                "enabled": True,
                "shadow_capture_ready": False,
            },
            {
                "ib_hash": "aaaaaaaa",
                "match_first_index": 0,
                "match_index_count": 200,
                "local_bone_count": 7,
                "source_local_bone_count": 20,
                "used_local_bone_indices": [0, 4, 19],
                "enabled": True,
                "shadow_capture_ready": True,
            },
            {
                "ib_hash": "cccccccc",
                "match_first_index": 0,
                "match_index_count": 300,
                "local_bone_count": 11,
                "enabled": False,
                "shadow_capture_ready": True,
            },
        ]

        order = main_analyze.build_bone_pool_order(candidates)

        self.assertEqual([entry["ib_hash"] for entry in order], ["aaaaaaaa", "bbbbbbbb"])
        self.assertEqual(order[0]["global_bone_base"], 0)
        self.assertEqual(order[0]["local_bone_count"], 3)
        self.assertEqual(order[0]["source_local_bone_count"], 20)
        self.assertEqual(order[0]["used_local_bone_indices"], [0, 4, 19])
        self.assertTrue(order[0]["bone_capture_available"])
        self.assertEqual(order[1]["global_bone_base"], 3)
        self.assertEqual(order[1]["local_bone_count"], 2)
        self.assertFalse(order[1]["bone_capture_available"])

    def test_bone_pool_order_keeps_dynamic_vb0_capture_available_but_excludes_lod(self):
        candidates = [
            {
                "ib_hash": "aaaaaaaa",
                "match_first_index": 0,
                "match_index_count": 100,
                "local_bone_count": 4,
                "used_local_bone_indices": [0, 1],
                "enabled": True,
                "shadow_capture_ready": True,
                "lod_match_excluded": True,
                "lod_match_excluded_reason": "dynamic_vb0_backing_hash_mismatch",
                "dynamic_vb0": True,
            }
        ]

        order = main_analyze.build_bone_pool_order(candidates)

        self.assertEqual(len(order), 1)
        self.assertTrue(order[0]["bone_capture_available"])
        self.assertTrue(order[0]["lod_match_excluded"])
        self.assertEqual(order[0]["lod_match_excluded_reason"], "dynamic_vb0_backing_hash_mismatch")
        self.assertEqual(order[0]["status"], "capture_ready_dynamic_vb0_lod_excluded")

    def test_bone_pool_order_excludes_cpu_pre_skinned_reference_geometry(self):
        order = main_analyze.build_bone_pool_order(
            [
                {
                    "ib_hash": "945c08a9",
                    "match_first_index": 0,
                    "match_index_count": 1698,
                    "local_bone_count": 12,
                    "used_local_bone_indices": [0, 1, 2],
                    "enabled": True,
                    "replacement_supported": False,
                    "skinning_mode": {"kind": "cpu_pre_skinned"},
                }
            ]
        )

        self.assertEqual([], order)

    def test_analyzer_reads_r32_uint_blend_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            buf_path = Path(temp_dir) / "blend.buf"
            with buf_path.open("wb") as file_handle:
                file_handle.write(struct.pack("<4I", 1, 2, 3, 4))
                file_handle.write(struct.pack("<4I", 0, 12, 7, 6))

            self.assertTrue(main_analyze._is_supported_blend_index_format("R32G32B32A32_UINT"))
            self.assertEqual(
                main_analyze._read_max_blend_index(
                    data_path=str(buf_path),
                    byte_offset=0,
                    stride=16,
                    vertex_count=2,
                    blend_offset=0,
                    blend_format="R32G32B32A32_UINT",
                ),
                12,
            )

    def test_blender_import_transform_mirrors_x_and_reverses_winding_by_default(self):
        positions = [(1.0, 2.0, 3.0), (-4.0, 5.0, 6.0)]
        normals = [(0.25, -0.5, 0.75)]
        triangles = [(0, 1, 2)]

        self.assertEqual(
            import_candidates._positions_for_blender(positions, mirror_flip=True),
            [(-1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        self.assertEqual(
            import_candidates._normals_for_blender(normals, mirror_flip=True),
            [(-0.25, -0.5, 0.75)],
        )
        self.assertEqual(
            import_candidates._triangles_for_blender(triangles, mirror_flip=True),
            [(0, 2, 1)],
        )

    def test_blender_import_transform_reverses_winding_without_mirror(self):
        self.assertEqual(
            import_candidates._triangles_for_blender([(0, 1, 2)], mirror_flip=False),
            [(0, 2, 1)],
        )

    def test_load_candidate_geometry_uses_first_index_and_corrects_vb_offsets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ib_txt, ib_buf = self._write_ib_fixture(root)
            vb0_txt, vb0_buf = self._write_vb0_fixture(root)
            vb1_txt, vb1_buf = self._write_vb1_fixture(root)
            vb2_txt, vb2_buf = self._write_vb2_fixture(root)

            candidate = {
                "display_name": "aaaaaaaa-6-3",
                "ib_hash": "aaaaaaaa",
                "match_first_index": 3,
                "match_index_count": 6,
                "import_paths": {
                    "ib": str(ib_txt),
                    "ib_buf": str(ib_buf),
                    "vb": {
                        "vb0": {"txt": [str(vb0_txt)], "layout_txt": [str(vb0_txt)], "buf": str(vb0_buf)},
                        "vb1": {"txt": [str(vb1_txt)], "layout_txt": [str(vb1_txt)], "buf": str(vb1_buf)},
                        "vb2": {"txt": [str(vb2_txt)], "layout_txt": [str(vb2_txt)], "buf": str(vb2_buf)},
                    },
                },
            }

            geometry = import_candidates.load_candidate_geometry(candidate)

            self.assertEqual(geometry.triangles, [(0, 1, 2), (0, 2, 3)])
            self.assertEqual(geometry.original_vertex_ids, [0, 1, 2, 3])
            self.assertEqual(geometry.positions[0], (1.0, 2.0, 3.0))
            self.assertAlmostEqual(geometry.uv1[0][0], 0.3, places=6)
            self.assertAlmostEqual(geometry.uv1[0][1], 0.4, places=6)
            self.assertEqual(geometry.texcoord4_raw[0], (-1, 2, -3, 4))
            self.assertEqual(geometry.blend_indices[0], (1, 2, 5, 0))
            self.assertAlmostEqual(geometry.blend_weights[0][1], 32768 / 65535.0, places=6)
            self.assertEqual(set(geometry.raw_vertex_streams), {"vb0", "vb1", "vb2"})
            self.assertEqual(
                bytes(geometry.raw_vertex_streams["vb1"][0]),
                struct.pack("<4f4B", 0.1, 0.2, 0.3, 0.4, 255, 2, 253, 4),
            )

    def test_decode_game_packed_normal_returns_unit_vector(self):
        normal = import_candidates.decode_game_packed_normal(0x40000000)
        self.assertAlmostEqual(math.sqrt(sum(component * component for component in normal)), 1.0, places=6)
        self.assertGreater(normal[2], 0.99)

    def test_byte_color_import_writes_srgb_channel_before_linear_fallback(self):
        class RecordingData:
            def __init__(self):
                self.calls = []

            def foreach_set(self, attribute_name, values):
                self.calls.append((attribute_name, tuple(values)))

        class Attribute:
            def __init__(self):
                self.data = RecordingData()

        class ColorAttributes:
            def __init__(self):
                self.attribute = None

            def get(self, _name):
                return self.attribute

            def new(self, **_kwargs):
                self.attribute = Attribute()
                return self.attribute

        class Mesh:
            def __init__(self):
                self.color_attributes = ColorAttributes()

        mesh = Mesh()
        import_candidates._store_snorm_byte_color_attribute(mesh, "raw", [(-1, 2, -3, 4)])

        call = mesh.color_attributes.attribute.data.calls[0]
        self.assertEqual(call[0], "color_srgb")
        self.assertEqual(
            [round(value * 255) for value in call[1]],
            [255, 2, 253, 4],
        )

    def test_color0_semantic_is_loaded_by_generic_vertex_adapter(self):
        element = main_analyze.HeaderElement(
            semantic_name="COLOR",
            semantic_index=0,
            fmt="R8G8B8A8_UNORM",
            input_slot=1,
            aligned_byte_offset=0,
        )
        slot = import_candidates._SlotSlice(
            slot_name="vb1",
            slot_index=1,
            buf_path="",
            header=main_analyze.BufferHeader(
                stride=4,
                vertex_count=2,
                elements=[element],
            ),
            elements={("COLOR", 0): element},
            base_offset=0,
            data=bytes([1, 2, 3, 4, 255, 128, 64, 0]),
        )

        records = import_candidates._read_vertex_semantics([slot], [0, 1])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["semantic_name"], "COLOR")
        self.assertEqual(records[0]["storage"], "uint8_raw")
        self.assertEqual(records[0]["values"].tolist(), [[1, 2, 3, 4], [255, 128, 64, 0]])

    def test_float_color0_is_not_misclassified_as_uv(self):
        element = main_analyze.HeaderElement(
            semantic_name="COLOR",
            semantic_index=0,
            fmt="R32G32_FLOAT",
            input_slot=1,
            aligned_byte_offset=0,
        )
        slot = import_candidates._SlotSlice(
            slot_name="vb1",
            slot_index=1,
            buf_path="",
            header=main_analyze.BufferHeader(
                stride=8,
                vertex_count=1,
                elements=[element],
            ),
            elements={("COLOR", 0): element},
            base_offset=0,
            data=struct.pack("<2f", 0.25, 0.75),
        )

        records = import_candidates._read_vertex_semantics([slot], [0])

        self.assertEqual(records[0]["semantic_name"], "COLOR")
        self.assertEqual(records[0]["storage"], "float")
        self.assertEqual(records[0]["values"].tolist(), [[0.25, 0.75]])

    def _write_ib_fixture(self, root: Path) -> tuple[Path, Path]:
        txt_path = root / "000001-ib=aaaaaaaa-vs=1-ps=2.txt"
        buf_path = root / "ib.buf"
        txt_path.write_text(
            "\n".join(
                [
                    "byte offset: 4",
                    "first index: 3",
                    "index count: 6",
                    "topology: trianglelist",
                    "format: DXGI_FORMAT_R16_UINT",
                    "",
                    "0 1 2",
                    "0 2 3",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        with buf_path.open("wb") as file_handle:
            file_handle.write(b"PAD!")
            for value in (99, 98, 97, 0, 1, 2, 0, 2, 3):
                file_handle.write(struct.pack("<H", value))
        return txt_path, buf_path

    def _write_vb0_fixture(self, root: Path) -> tuple[Path, Path]:
        txt_path = root / "000001-vb0=bbbbbbbb-vs=1-ps=2.txt"
        buf_path = root / "vb0.buf"
        txt_path.write_text(
            "\n".join(
                [
                    "byte offset: 0",
                    "stride: 16",
                    "first vertex: 0",
                    "vertex count: 4",
                    "topology: trianglelist",
                    "element[0]:",
                    "  SemanticName: POSITION",
                    "  SemanticIndex: 0",
                    "  Format: R32G32B32_FLOAT",
                    "  InputSlot: 0",
                    "  AlignedByteOffset: 0",
                    "element[1]:",
                    "  SemanticName: NORMAL",
                    "  SemanticIndex: 0",
                    "  Format: R32_FLOAT",
                    "  InputSlot: 0",
                    "  AlignedByteOffset: 12",
                    "",
                    "vertex-data:",
                    "",
                    "vb0[0]+000 POSITION: 1, 2, 3",
                    "vb0[0]+012 NORMAL: 2",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        records = [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, 9.0),
            (10.0, 11.0, 12.0),
        ]
        with buf_path.open("wb") as file_handle:
            for position in records:
                file_handle.write(struct.pack("<3fI", *position, 0x40000000))
        return txt_path, buf_path

    def _write_vb1_fixture(self, root: Path) -> tuple[Path, Path]:
        txt_path = root / "000001-vb1=cccccccc-vs=1-ps=2.txt"
        buf_path = root / "vb1.buf"
        texcoord4 = (-1, 2, -3, 4)
        txt_path.write_text(
            "\n".join(
                [
                    "byte offset: 20",
                    "stride: 20",
                    "first vertex: 0",
                    "vertex count: 4",
                    "topology: trianglelist",
                    "element[0]:",
                    "  SemanticName: TEXCOORD",
                    "  SemanticIndex: 0",
                    "  Format: R32G32_FLOAT",
                    "  InputSlot: 1",
                    "  AlignedByteOffset: 0",
                    "element[1]:",
                    "  SemanticName: TEXCOORD",
                    "  SemanticIndex: 1",
                    "  Format: R32G32_FLOAT",
                    "  InputSlot: 1",
                    "  AlignedByteOffset: 8",
                    "element[2]:",
                    "  SemanticName: TEXCOORD",
                    "  SemanticIndex: 4",
                    "  Format: R8G8B8A8_SNORM",
                    "  InputSlot: 1",
                    "  AlignedByteOffset: 16",
                    "",
                    "vertex-data:",
                    "",
                    "vb1[0]+000 TEXCOORD: 0.1, 0.2",
                    "vb1[0]+008 TEXCOORD1: 0.3, 0.4",
                    "vb1[0]+016 TEXCOORD4: "
                    + ", ".join(str(value / 127.0) for value in texcoord4),
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        with buf_path.open("wb") as file_handle:
            file_handle.write(b"X" * 20)
            for index in range(4):
                file_handle.write(
                    struct.pack(
                        "<4f4B",
                        0.1 + index,
                        0.2 + index,
                        0.3 + index,
                        0.4 + index,
                        *[(value + 256) % 256 for value in texcoord4],
                    )
                )
        return txt_path, buf_path

    def _write_vb2_fixture(self, root: Path) -> tuple[Path, Path]:
        txt_path = root / "000001-vb2=dddddddd-vs=1-ps=2.txt"
        buf_path = root / "vb2.buf"
        weights = (0, 32768, 65535, 16384)
        indices = (1, 2, 5, 0)
        txt_path.write_text(
            "\n".join(
                [
                    "byte offset: 8",
                    "stride: 12",
                    "first vertex: 0",
                    "vertex count: 4",
                    "topology: trianglelist",
                    "element[0]:",
                    "  SemanticName: BLENDWEIGHTS",
                    "  SemanticIndex: 0",
                    "  Format: R16G16B16A16_UNORM",
                    "  InputSlot: 2",
                    "  AlignedByteOffset: 0",
                    "element[1]:",
                    "  SemanticName: BLENDINDICES",
                    "  SemanticIndex: 0",
                    "  Format: R8G8B8A8_UINT",
                    "  InputSlot: 2",
                    "  AlignedByteOffset: 8",
                    "",
                    "vertex-data:",
                    "",
                    "vb2[0]+000 BLENDWEIGHTS: "
                    + ", ".join(str(value / 65535.0) for value in weights),
                    "vb2[0]+008 BLENDINDICES: 1, 2, 5, 0",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        with buf_path.open("wb") as file_handle:
            file_handle.write(b"Y" * 8)
            for vertex_index in range(4):
                file_handle.write(struct.pack("<4H4B", *weights, *(value + vertex_index for value in indices)))
        return txt_path, buf_path


@unittest.skipUnless(SAMPLE_FRAMEANALYSIS.exists(), "sample FrameAnalysis folder is not available")
class RealFrameAnalysisImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = main_analyze.analyze_main_frameanalysis(str(SAMPLE_FRAMEANALYSIS), ["640d1c0e"])

    def test_manifest_keeps_deduped_layout_paths(self):
        candidate = self._candidate("640d1c0e-46845-0")
        self.assertGreater(len(candidate["import_paths"]["vb"]["vb1"].get("layout_txt", [])), 0)

    def test_manifest_has_vertex_layout_table_for_export(self):
        table = self.manifest.get("vertex_layout_table", {})
        layout = table.get("640d1c0e-46845-0")
        self.assertIsInstance(layout, dict)
        self.assertEqual(layout["vertex_buffers"]["vb1"]["stride"], 20)
        fields = layout["vertex_buffers"]["vb1"]["fields"]
        self.assertIn(
            {
                "semantic": "TEXCOORD4",
                "semantic_name": "TEXCOORD",
                "semantic_index": 4,
                "format": "R8G8B8A8_SNORM",
                "input_slot": 1,
                "aligned_byte_offset": 16,
            },
            fields,
        )

    def test_640d1c0e_geometry_matches_known_frameanalysis_values(self):
        candidate = self._candidate("640d1c0e-46845-0")
        geometry = import_candidates.load_candidate_geometry(candidate, self.manifest["frameanalysis_dir"])
        self.assertEqual(len(geometry.positions), 13571)
        self.assertEqual(len(geometry.triangles), 15615)
        self.assertEqual(candidate["local_bone_count"], 145)
        self.assertEqual(candidate["skin_format"]["blend_weights_format"], "R16G16B16A16_UNORM")
        self.assertEqual(candidate["skin_format"]["blend_indices_format"], "R8G8B8A8_UINT")
        self.assertIsNotNone(geometry.uv0)
        self.assertIsNone(geometry.uv1)
        self.assertTrue(any(
            int(record["semantic_index"]) == 4
            and str(record["format"]).upper() == "R8G8B8A8_SNORM"
            for record in geometry.texcoord_semantics
        ))
        self.assertEqual(geometry.blend_indices[0], (2, 52, 55, 49))
        self.assertAlmostEqual(sum(geometry.blend_weights[0]), 1.0, places=4)

    def test_all_candidates_load_without_parser_failures(self):
        failures = []
        for candidate in self.manifest["candidate_ibs"]:
            try:
                import_candidates.load_candidate_geometry(candidate, self.manifest["frameanalysis_dir"])
            except Exception as exc:  # pragma: no cover - failure path reports all candidates at once
                failures.append((candidate.get("display_name"), str(exc)))
        self.assertEqual([], failures)

    def _candidate(self, display_name: str) -> dict:
        for candidate in self.manifest["candidate_ibs"]:
            if candidate.get("display_name") == display_name:
                return candidate
        self.fail(f"candidate not found: {display_name}")


@unittest.skipUnless(SAMPLE_FRAMEANALYSIS.exists(), "sample FrameAnalysis folder is not available")
class RealFrameAnalysisAutoShadowWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = main_analyze.analyze_main_frameanalysis(str(SAMPLE_FRAMEANALYSIS))

    def test_shadow_capture_vs_set_expands_inside_shadow_window(self):
        candidate = self._candidate("38b8d614-7560-0")
        self.assertTrue(candidate["shadow_capture_ready"])
        self.assertIn(39, candidate["shadow_draw_indices"])
        self.assertIn(103, candidate["shadow_draw_indices"])
        self.assertIn("6733250da4e23fd6", self.manifest["shadow_stage"]["shadow_vs_hashes"])

    def _candidate(self, display_name: str) -> dict:
        for candidate in self.manifest["candidate_ibs"]:
            if candidate.get("display_name") == display_name:
                return candidate
        self.fail(f"candidate not found: {display_name}")


@unittest.skipUnless(MAIN_FRAMEANALYSIS.exists(), "main FrameAnalysis folder is not available")
class MainFrameAnalysisAnalyzeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = main_analyze.analyze_main_frameanalysis(str(MAIN_FRAMEANALYSIS))

    def test_analyze_does_not_require_manual_target_hash(self):
        self.assertGreater(len(self.manifest["candidate_ibs"]), 0)
        self.assertEqual(
            "auto_gbuffer_anchor_backtrack_shadow_vs",
            self.manifest["target"]["selection_mode"],
        )
        self.assertGreaterEqual(len(self.manifest["shadow_stage"]["shadow_vs_hashes"]), 1)

    def test_main_shading_only_ib_enters_mapping_pool_after_capture_ready_entries(self):
        candidate = self._candidate("271dc0d1-402-0")
        self.assertFalse(candidate["shadow_capture_ready"])
        self.assertEqual(candidate["status"], "import_only_no_early_shadow")
        pool_names = [
            f"{entry['ib_hash']}-{entry['match_index_count']}-{entry['match_first_index']}"
            for entry in self.manifest["bone_pool_order"]
        ]
        self.assertIn("271dc0d1-402-0", pool_names)
        entry = next(
            entry
            for entry in self.manifest["bone_pool_order"]
            if f"{entry['ib_hash']}-{entry['match_index_count']}-{entry['match_first_index']}" == "271dc0d1-402-0"
        )
        self.assertFalse(entry["bone_capture_available"])
        self.assertLess(
            max(
                index
                for index, pool_name in enumerate(pool_names)
                if pool_name == "2009f0d6-1356-0"
            ),
            pool_names.index("271dc0d1-402-0"),
        )

    def test_main_r32_blendindices_ib_enters_candidate_pool(self):
        candidate = self._candidate("2009f0d6-1356-0")
        self.assertTrue(candidate["shadow_capture_ready"])
        self.assertGreater(candidate["local_bone_count"], 0)
        self.assertLessEqual(candidate["local_bone_count"], candidate["source_local_bone_count"])
        self.assertEqual(candidate["skin_format"]["blend_weights_format"], "R32G32B32A32_FLOAT")
        self.assertEqual(candidate["skin_format"]["blend_indices_format"], "R32G32B32A32_UINT")
        self.assertTrue(candidate["lod_match_excluded"])
        self.assertEqual(candidate["position_stream"]["ib_backing_hash"], "9a09f1f0")
        self.assertEqual(candidate["position_stream"]["vb0_backing_hash"], "1d6a6186")
        self.assertIn(12, candidate["shadow_draw_indices"])
        pool_names = {
            f"{entry['ib_hash']}-{entry['match_index_count']}-{entry['match_first_index']}"
            for entry in self.manifest["bone_pool_order"]
        }
        self.assertIn("2009f0d6-1356-0", pool_names)
        pool_entry = next(
            entry
            for entry in self.manifest["bone_pool_order"]
            if f"{entry['ib_hash']}-{entry['match_index_count']}-{entry['match_first_index']}" == "2009f0d6-1356-0"
        )
        self.assertTrue(pool_entry["bone_capture_available"])
        self.assertTrue(pool_entry["lod_match_excluded"])

    def test_main_static_vb0_ib_is_not_lod_excluded(self):
        candidate = self._candidate("58870754-96-0")
        self.assertFalse(candidate["lod_match_excluded"])
        self.assertEqual(candidate["position_stream"]["ib_backing_hash"], "9a09f1f0")
        self.assertEqual(candidate["position_stream"]["vb0_backing_hash"], "9a09f1f0")

    def _candidate(self, display_name: str) -> dict:
        for candidate in self.manifest["candidate_ibs"]:
            if candidate.get("display_name") == display_name:
                return candidate
        self.fail(f"candidate not found: {display_name}")


@unittest.skipUnless(CPU_SKIN_FRAMEANALYSIS.exists(), "CPU skin FrameAnalysis folder is not available")
class CpuSkinFrameAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = main_analyze.analyze_main_frameanalysis(str(CPU_SKIN_FRAMEANALYSIS))

    def test_draw_45_is_reference_only_cpu_pre_skinned_geometry(self):
        candidate = next(
            item for item in self.manifest["candidate_ibs"] if item.get("display_name") == "945c08a9-1698-0"
        )

        self.assertEqual(candidate["import_draw_index"], 45)
        self.assertEqual(candidate["position_stream"]["cb2_first_constant"], 14944)
        self.assertEqual(candidate["position_stream"]["cb2_skinning_flags"], 0x10)
        self.assertTrue(candidate["position_stream"]["vb3_distinct"])
        self.assertEqual(candidate["skinning_mode"]["kind"], "cpu_pre_skinned")
        self.assertFalse(candidate["replacement_supported"])
        self.assertEqual(candidate["status"], "cpu_pre_skinned_import_only")


if __name__ == "__main__":
    unittest.main()
