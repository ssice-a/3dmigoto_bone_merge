from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

data_types = importlib.import_module(f"{PACKAGE_DIR.name}.core.data_types")


class DataTypeAnnotationTests(unittest.TestCase):
    def test_d5_contract_marks_texcoord2_and_texcoord4_as_vs_inputs(self):
        layout = {
            "import_vs_hash": "d5d284900d7d0543",
            "vertex_buffers": {
                "vb1": {
                    "slot": "vb1",
                    "stride": 12,
                    "fields": [
                        {
                            "semantic": "TEXCOORD0",
                            "semantic_name": "TEXCOORD",
                            "semantic_index": 0,
                            "format": "R32G32_FLOAT",
                            "aligned_byte_offset": 0,
                        },
                        {
                            "semantic": "TEXCOORD2",
                            "semantic_name": "TEXCOORD",
                            "semantic_index": 2,
                            "format": "R8G8B8A8_SNORM",
                            "aligned_byte_offset": 8,
                        },
                        {
                            "semantic": "TEXCOORD4",
                            "semantic_name": "TEXCOORD",
                            "semantic_index": 4,
                            "format": "R8G8B8A8_SNORM",
                            "aligned_byte_offset": 8,
                        },
                    ],
                }
            },
        }

        annotated = data_types.annotate_vertex_layout(layout)
        roles = annotated["vertex_buffers"]["vb1"]["vs_semantic_roles"]

        self.assertEqual(
            annotated["vertex_buffers"]["vb1"]["layout_profile"],
            "texcoord_uv0_aux_snorm_stride12",
        )
        self.assertTrue(roles["TEXCOORD2"]["used_by_import_vs"])
        self.assertEqual(roles["TEXCOORD2"]["vs_role"], "outline_expansion_tangent_xy")
        self.assertTrue(roles["TEXCOORD4"]["used_by_import_vs"])

    def test_9bac_contract_does_not_mark_texcoord2_as_used(self):
        layout = {
            "import_vs_hash": "9bac7486f7930a24",
            "vertex_buffers": {
                "vb1": {
                    "slot": "vb1",
                    "stride": 20,
                    "fields": [
                        {
                            "semantic": "TEXCOORD2",
                            "semantic_name": "TEXCOORD",
                            "semantic_index": 2,
                            "format": "R8G8B8A8_SNORM",
                            "aligned_byte_offset": 16,
                        },
                        {
                            "semantic": "TEXCOORD4",
                            "semantic_name": "TEXCOORD",
                            "semantic_index": 4,
                            "format": "R8G8B8A8_SNORM",
                            "aligned_byte_offset": 16,
                        },
                    ],
                }
            },
        }

        annotated = data_types.annotate_vertex_layout(layout)
        roles = annotated["vertex_buffers"]["vb1"]["vs_semantic_roles"]

        self.assertFalse(roles["TEXCOORD2"]["used_by_import_vs"])
        self.assertTrue(roles["TEXCOORD4"]["used_by_import_vs"])
        self.assertEqual(roles["TEXCOORD4"]["vs_role"], "packed_aux_vector")


if __name__ == "__main__":
    unittest.main()
