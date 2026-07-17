from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

value_utils = importlib.import_module(f"{PACKAGE_DIR.name}.core.value_utils")


class ValueUtilsTests(unittest.TestCase):
    def test_int_or_default_preserves_zero(self):
        self.assertEqual(0, value_utils.int_or_default(0, -1))

    def test_int_or_default_handles_missing_values(self):
        self.assertEqual(-1, value_utils.int_or_default(None, -1))
        self.assertEqual(-1, value_utils.int_or_default("", -1))


if __name__ == "__main__":
    unittest.main()
