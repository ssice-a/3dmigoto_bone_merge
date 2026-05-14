import importlib
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

toggle_draw_sets = importlib.import_module(f"{PACKAGE_DIR.name}.core.toggle_draw_sets")


class ToggleDrawSetTests(unittest.TestCase):
    def test_apply_toggle_draw_sets_marks_matching_object_draws(self):
        geometry = [
            {
                "resource_suffix": "aaaaaaaa_42_0_part00",
                "object_draws": [
                    {"object_name": "body", "start_index": 0, "index_count": 3},
                    {"object_name": "hair", "start_index": 3, "index_count": 3},
                ],
            }
        ]
        groups = toggle_draw_sets.normalize_toggle_draw_sets(
            [
                {
                    "toggle_id": "style",
                    "label": "Style",
                    "key": "shift k",
                    "values": [
                        {"value": 0, "objects": ["hair"]},
                        {"value": 1, "objects": []},
                    ],
                }
            ]
        )

        updated, warnings = toggle_draw_sets.apply_toggle_draw_sets_to_geometry(geometry, groups)

        self.assertEqual([], warnings)
        body, hair = updated[0]["object_draws"]
        self.assertNotIn("toggle_condition", body)
        self.assertEqual("$bmc_toggle_style == 0", hair["toggle_condition"])

    def test_rejects_object_assigned_to_multiple_toggle_values(self):
        groups = toggle_draw_sets.normalize_toggle_draw_sets(
            [
                {
                    "toggle_id": "style",
                    "values": [
                        {"value": 0, "objects": ["hair"]},
                        {"value": 1, "objects": ["hair"]},
                    ],
                }
            ]
        )

        with self.assertRaises(ValueError):
            toggle_draw_sets.apply_toggle_draw_sets_to_geometry([], groups)


if __name__ == "__main__":
    unittest.main()
