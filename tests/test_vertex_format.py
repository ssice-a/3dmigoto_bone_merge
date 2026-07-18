from __future__ import annotations

import importlib
import math
import struct
import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

vertex_format = importlib.import_module(f"{PACKAGE_DIR.name}.core.vertex_format")
vertex_layout_codec = importlib.import_module(f"{PACKAGE_DIR.name}.core.vertex_layout_codec")
import_candidates = importlib.import_module(f"{PACKAGE_DIR.name}.core.import_candidates")
export_buffers = importlib.import_module(f"{PACKAGE_DIR.name}.core.export_buffers")


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

    def test_every_registered_format_uses_the_shared_codec(self):
        for name, spec in vertex_layout_codec.DXGI_FORMATS.items():
            if spec.conversion and spec.conversion.startswith("unorm"):
                source = [0.0, 0.25, 0.5, 1.0]
            elif spec.conversion and spec.conversion.startswith("snorm"):
                source = [-1.0, -0.5, 0.0, 1.0]
            elif spec.dtype.startswith("<f"):
                source = [-1.0, -0.5, 0.25, 1.0]
            else:
                source = [1, 2, 3, 4]
            packed = vertex_format.pack_vertex_format(name, source)
            unpacked = vertex_format.unpack_vertex_format(name, packed)
            self.assertEqual(len(packed), spec.byte_size, name)
            self.assertEqual(len(unpacked), spec.component_count, name)

    def test_unpack_snorm_minimum_is_clamped(self):
        self.assertEqual(
            vertex_format.unpack_vertex_format("R8G8B8A8_SNORM", bytes([128, 129, 0, 127])),
            (-1.0, -1.0, 0.0, 1.0),
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

    def test_numpy_packed_tangent_frame_round_trips_roll(self):
        packed_values = np.asarray(
            [0xFB979E48, 0x54B77DB8, 0xF8FDB181, 0xF7AD89B4],
            dtype=np.uint32,
        )

        normals = np.asarray(import_candidates._decode_game_packed_normals(packed_values), dtype=np.float32)
        tangents, signs = import_candidates._decode_game_packed_tangent_frames(packed_values)
        x_values = normals[:, 0]
        y_values = normals[:, 1]
        z_values = normals[:, 2]
        inv_l1 = 1.0 / np.maximum(np.abs(x_values) + np.abs(y_values) + np.abs(z_values), 1e-12)
        oct_x = x_values * inv_l1
        oct_y = y_values * inv_l1
        fold_mask = z_values < 0.0
        if np.any(fold_mask):
            old_x = oct_x.copy()
            old_y = oct_y.copy()
            sign_x = np.where(old_x >= 0.0, 1.0, -1.0)
            sign_y = np.where(old_y >= 0.0, 1.0, -1.0)
            oct_x = np.where(fold_mask, (1.0 - np.abs(old_y)) * sign_x, old_x)
            oct_y = np.where(fold_mask, (1.0 - np.abs(old_x)) * sign_y, old_y)
        quant_x = np.rint(np.clip(oct_x, -1.0, 1.0) * 511.0).astype(np.int32)
        quant_y = np.rint(np.clip(oct_y, -1.0, 1.0) * 511.0).astype(np.int32)
        decoded_normals = export_buffers._decode_octahedral_quantized_numpy(quant_x, quant_y)
        packed_roll = export_buffers._encode_tangent_roll_numpy(decoded_normals, tangents)
        rebuilt = (
            np.uint32(0x40000000)
            | (quant_x.astype(np.uint32) & np.uint32(0x3FF))
            | ((quant_y.astype(np.uint32) & np.uint32(0x3FF)) << np.uint32(10))
            | ((packed_roll & np.uint32(0x3FF)) << np.uint32(20))
            | np.where(np.asarray(signs, dtype=np.float32) >= 0.0, np.uint32(0x80000000), np.uint32(0))
        ).astype(np.uint32)

        self.assertEqual([int(value) for value in rebuilt], [int(value) for value in packed_values])


if __name__ == "__main__":
    unittest.main()
