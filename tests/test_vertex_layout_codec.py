from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

codec = importlib.import_module(f"{PACKAGE_DIR.name}.core.vertex_layout_codec")


class VertexLayoutCodecTests(unittest.TestCase):
    def test_format_registry_covers_half_and_packed_sizes(self):
        self.assertEqual(codec.dxgi_format_size("DXGI_FORMAT_R16G16_FLOAT"), 4)
        self.assertEqual(codec.dxgi_format_size("R16G16B16A16_SNORM"), 8)
        self.assertEqual(codec.dxgi_format_size("R10G10B10A2_UNORM"), 4)

    def test_exact_semantic_aliases_share_one_physical_field(self):
        slot = codec.build_slot_layout(
            "vb1",
            {
                "stride": 12,
                "fields": [
                    {
                        "semantic_name": "TEXCOORD",
                        "semantic_index": 0,
                        "format": "R32G32_FLOAT",
                        "aligned_byte_offset": 0,
                    },
                    {
                        "semantic_name": "TEXCOORD",
                        "semantic_index": 2,
                        "format": "R8G8B8A8_SNORM",
                        "aligned_byte_offset": 8,
                    },
                    {
                        "semantic_name": "TEXCOORD",
                        "semantic_index": 4,
                        "format": "R8G8B8A8_SNORM",
                        "aligned_byte_offset": 8,
                    },
                ],
            },
        )

        self.assertEqual(len(slot.fields), 3)
        self.assertEqual(len(slot.physical_fields), 2)
        self.assertEqual(
            [alias.semantic for alias in slot.physical_fields[1].aliases],
            ["TEXCOORD4", "TEXCOORD2"],
        )
        self.assertEqual(slot.physical_fields[1].primary.semantic, "TEXCOORD4")

    def test_partial_overlap_is_rejected_instead_of_last_writer_winning(self):
        with self.assertRaisesRegex(codec.VertexLayoutError, "partially overlapping"):
            codec.build_slot_layout(
                "vb1",
                {
                    "stride": 12,
                    "fields": [
                        {
                            "semantic_name": "TEXCOORD",
                            "semantic_index": 0,
                            "format": "R32G32_FLOAT",
                            "aligned_byte_offset": 0,
                        },
                        {
                            "semantic_name": "COLOR",
                            "semantic_index": 0,
                            "format": "R8G8B8A8_UNORM",
                            "aligned_byte_offset": 4,
                        },
                    ],
                },
            )

    def test_stream_alias_requires_captured_resource_identity(self):
        fields = [
            {
                "semantic_name": "POSITION",
                "semantic_index": 0,
                "format": "R32G32B32_FLOAT",
                "aligned_byte_offset": 0,
            }
        ]
        vb0 = codec.build_slot_layout("vb0", {"stride": 12, "fields": fields})
        vb3 = codec.build_slot_layout("vb3", {"stride": 12, "fields": fields})
        self.assertFalse(codec.slots_share_source(vb0, vb3))

        source = {
            "stride": 12,
            "vertex_count": 3,
            "resource_hash": "abcdef01",
            "backing_hash": "11111111",
            "source_buf": "deduped/source.buf",
            "byte_offset": 64,
            "fields": fields,
        }
        vb0 = codec.build_slot_layout("vb0", source)
        vb3 = codec.build_slot_layout("vb3", source)
        self.assertTrue(codec.slots_share_source(vb0, vb3))


if __name__ == "__main__":
    unittest.main()
