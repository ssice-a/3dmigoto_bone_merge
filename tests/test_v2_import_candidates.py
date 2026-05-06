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


SAMPLE_FRAMEANALYSIS = Path(r"E:\XXMI\EFMI\FrameAnalysis-2026-05-05-225007")
MAIN_FRAMEANALYSIS = Path(r"E:\XXMI\EFMI\FrameAnalysis-2026-05-05-222451")


class SyntheticCandidateImportTests(unittest.TestCase):
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

    def test_decode_game_packed_normal_returns_unit_vector(self):
        normal = import_candidates.decode_game_packed_normal(0x40000000)
        self.assertAlmostEqual(math.sqrt(sum(component * component for component in normal)), 1.0, places=6)
        self.assertGreater(normal[2], 0.99)

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

    def test_640d1c0e_geometry_matches_known_frameanalysis_values(self):
        candidate = self._candidate("640d1c0e-46845-0")
        geometry = import_candidates.load_candidate_geometry(candidate, self.manifest["frameanalysis_dir"])
        self.assertEqual(len(geometry.positions), 13571)
        self.assertEqual(len(geometry.triangles), 15615)
        self.assertEqual(candidate["local_bone_count"], 145)
        self.assertEqual(candidate["skin_format"]["blend_weights_format"], "R16G16B16A16_UNORM")
        self.assertEqual(candidate["skin_format"]["blend_indices_format"], "R8G8B8A8_UINT")
        self.assertAlmostEqual(geometry.uv1[0][0], geometry.uv0[0][0], places=6)
        self.assertAlmostEqual(geometry.uv1[0][1], geometry.uv0[0][1], places=6)
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
        self.assertIn(12, candidate["shadow_draw_indices"])
        pool_names = {
            f"{entry['ib_hash']}-{entry['match_index_count']}-{entry['match_first_index']}"
            for entry in self.manifest["bone_pool_order"]
        }
        self.assertIn("2009f0d6-1356-0", pool_names)

    def _candidate(self, display_name: str) -> dict:
        for candidate in self.manifest["candidate_ibs"]:
            if candidate.get("display_name") == display_name:
                return candidate
        self.fail(f"candidate not found: {display_name}")


if __name__ == "__main__":
    unittest.main()
