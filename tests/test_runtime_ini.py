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
    def test_sparse_capture_uses_static_local_index_table(self):
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
            file_name="12345678-42-0-LocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="12345678_42_0",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = ini_export.materialize_bonestore_runtime(tmpdir, manifest, [palette])
            indices_path = runtime["buffers"]["capture_local_indices"]["file_path"]
            meta_path = runtime["buffers"]["capture_meta"]["file_path"]

            self.assertEqual((2, 52, 248), _read_uints(indices_path))
            self.assertEqual((10, 3, 0, 0), _read_uints(meta_path))

            ini_text = ini_export.build_bonestore_ini_content(runtime)
            self.assertNotIn("ResourceBoneMeta", ini_text)
            self.assertIn("ResourceMainCaptureRecords", ini_text)
            self.assertIn("ResourceMainCaptureSourceLocalBones", ini_text)
            self.assertIn("x100 = 0", ini_text)
            self.assertIn("x101 = 3", ini_text)

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
            self.assertEqual((0, 2, 1, 0), _read_uints(runtime["buffers"]["lod_capture_meta"]["file_path"]))
            self.assertEqual((0, 0, 0, 1), _read_uints(runtime["buffers"]["lod_capture_pairs"]["file_path"]))

            ini_text = ini_export.build_bonestore_ini_content(runtime)
            self.assertIn("hash = 87654321", ini_text)
            self.assertIn("match_first_index = 5", ini_text)
            self.assertIn("run = CustomShader_RecordBonesScatter", ini_text)
            self.assertIn("hash = bbbbbbbbbbbbbbbb", ini_text)

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
            file_name="bad-LocalToGlobalBoneMap.buf",
            file_path="",
            resource_suffix="bad",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "capture-unavailable"):
                ini_export.materialize_bonestore_runtime(tmpdir, manifest, [palette])


def _read_uints(path: str) -> tuple[int, ...]:
    with open(path, "rb") as handle:
        data = handle.read()
    return struct.unpack("<" + "I" * (len(data) // 4), data)


if __name__ == "__main__":
    unittest.main()
