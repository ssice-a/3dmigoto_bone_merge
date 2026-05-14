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

    def test_allows_object_assigned_to_multiple_toggle_values(self):
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
        geometry = [
            {
                "object_draws": [
                    {"object_name": "hair", "start_index": 0, "index_count": 3},
                ],
            }
        ]

        updated, warnings = toggle_draw_sets.apply_toggle_draw_sets_to_geometry(geometry, groups)

        self.assertEqual([], warnings)
        self.assertEqual("($bmc_toggle_style == 0 || $bmc_toggle_style == 1)", updated[0]["object_draws"][0]["toggle_condition"])

    def test_normalizes_blender_arrow_and_numpad_key_names(self):
        self.assertEqual("VK_UP", toggle_draw_sets.normalize_key_binding("UP_ARROW"))
        self.assertEqual("VK_NUMPAD8", toggle_draw_sets.normalize_key_binding("NUMPAD_8"))
        self.assertEqual("ctrl VK_UP", toggle_draw_sets.normalize_key_binding("CTRL+UP_ARROW"))


if __name__ == "__main__":
    unittest.main()
