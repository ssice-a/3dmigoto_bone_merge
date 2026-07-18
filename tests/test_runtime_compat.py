import importlib
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

runtime_compat = importlib.import_module(f"{PACKAGE_DIR.name}.core.runtime_compat")


def _write_fake_runtime(path: Path, *tokens: str) -> None:
    path.write_bytes(b"MZ" + "\0".join(tokens).encode("utf-16-le"))


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_accepts_v094_runtime_with_hashregion_and_fifo_pool_support(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dll_path = root / "d3d11.dll"
            output_dir = root / "Mods" / "Character"
            _write_fake_runtime(dll_path, "->HashRegion", "pool_size", "pool_index_type")

            found = runtime_compat.assert_bone_merge_runtime_compatible(output_dir)

            self.assertEqual(found, dll_path)

    def test_rejects_v092_runtime_without_hashregion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dll_path = root / "d3d11.dll"
            output_dir = root / "Mods" / "Character"
            _write_fake_runtime(dll_path, "pool_size", "pool_index_type")

            with self.assertRaisesRegex(runtime_compat.RuntimeCompatibilityError, "HashRegion"):
                runtime_compat.assert_bone_merge_runtime_compatible(output_dir)

    def test_skips_check_when_export_is_not_inside_an_efmi_installation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "standalone-export"

            found = runtime_compat.assert_bone_merge_runtime_compatible(output_dir)

            self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
