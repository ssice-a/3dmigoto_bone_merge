from __future__ import annotations

import importlib
import math
import struct
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

vertex_format = importlib.import_module(f"{PACKAGE_DIR.name}.core.vertex_format")
import_candidates = importlib.import_module(f"{PACKAGE_DIR.name}.core.import_candidates")


class VertexFormatTests(unittest.TestCase):
    def test_pack_float_and_uint_formats(self):
        self.assertEqual(
            vertex_format.pack_vertex_format("R8G8B8A8_UINT", [1, 2, 300, -5]),
            bytes([1, 2, 255, 0]),
        )
        self.assertEqual(
            struct.unpack("<4I", vertex_format.pack_vertex_format("DXGI_FORMAT_R32G32B32A32_UINT", [1, 2, 3, 4])),
            (1, 2, 3, 4),
        )
        self.assertEqual(
            struct.unpack("<2f", vertex_format.pack_vertex_format("R32G32_FLOAT", [0.25, -0.5])),
            (0.25, -0.5),
        )
        self.assertEqual(
            vertex_format.unpack_vertex_format("R32G32_FLOAT", vertex_format.pack_vertex_format("R32G32_FLOAT", [0.25, -0.5])),
            (0.25, -0.5),
        )

    def test_pack_unorm_and_snorm_formats(self):
        self.assertEqual(
            struct.unpack("<4H", vertex_format.pack_vertex_format("R16G16B16A16_UNORM", [0.0, 0.5, 1.0, 2.0])),
            (0, 32768, 65535, 65535),
        )
        self.assertEqual(
            vertex_format.pack_vertex_format("R8G8B8A8_SNORM", [-1.0, 0.0, 1.0, -0.5]),
            bytes([129, 0, 127, 192]),
        )

    def test_pack_into_rejects_out_of_bounds_write(self):
        buffer = bytearray(4)
        with self.assertRaisesRegex(ValueError, "exceeds target buffer"):
            vertex_format.pack_into_vertex_format(buffer, 2, "R32_FLOAT", [1.0])

    def test_game_packed_normal_round_trips_through_import_decoder(self):
        source = (0.25, -0.5, 0.75)
        packed = vertex_format.encode_game_packed_normal(source)
        decoded = import_candidates.decode_game_packed_normal(packed)
        source_length = math.sqrt(sum(component * component for component in source))
        normalized_source = tuple(component / source_length for component in source)

        dot = sum(a * b for a, b in zip(normalized_source, decoded))
        self.assertGreater(dot, 0.999)

    def test_game_packed_tangent_frame_preserves_roll_and_sign(self):
        normal = (0.0, 0.0, 1.0)
        tangent = (-1.0, 0.0, 0.0)
        packed = vertex_format.encode_game_packed_tangent_frame(normal, tangent, 1.0)
        decoded_normal, decoded_tangent, decoded_sign = vertex_format.decode_game_packed_tangent_frame(packed)

        self.assertGreater(sum(a * b for a, b in zip(normal, decoded_normal)), 0.999)
        self.assertGreater(sum(a * b for a, b in zip(tangent, decoded_tangent)), 0.99)
        self.assertEqual(decoded_sign, 1.0)
        self.assertNotEqual((packed >> 20) & 0x3FF, 0)


if __name__ == "__main__":
    unittest.main()
