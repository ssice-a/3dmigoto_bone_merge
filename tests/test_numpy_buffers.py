import struct
import tempfile
import unittest
from pathlib import Path

from core import draw_arrays, numpy_buffers


class NumpyBuffersTests(unittest.TestCase):
    def test_read_index_file_uses_dxgi_offsets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ib.buf"
            path.write_bytes(b"pad!" + struct.pack("<5H", 9, 1, 2, 3, 8))
            self.assertEqual(
                numpy_buffers.read_index_file(
                    str(path),
                    "DXGI_FORMAT_R16_UINT",
                    3,
                    byte_offset=4,
                    first_index=1,
                ),
                [1, 2, 3],
            )

    def test_read_interleaved_field_converts_unorm(self):
        stride = 12
        data = (
            struct.pack("<4H4x", 0, 32768, 65535, 65535)
            + struct.pack("<4H4x", 65535, 0, 32768, 0)
        )
        values = numpy_buffers.read_interleaved_field(
            data,
            [1, 0],
            stride=stride,
            offset=0,
            fmt="R16G16B16A16_UNORM",
            vertex_count=2,
        )
        self.assertIsNotNone(values)
        self.assertAlmostEqual(float(values[0][0]), 1.0)
        self.assertAlmostEqual(float(values[1][2]), 1.0)

    def test_assign_bytes_handles_2d_vectors(self):
        np = draw_arrays.require_numpy()
        target = np.zeros((2, 16), dtype=np.uint8)
        values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        numpy_buffers.assign_bytes(target, 4, values)
        self.assertEqual(target[0, 4:12].tobytes(), struct.pack("<2f", 1.0, 2.0))

    def test_positions_diag(self):
        self.assertAlmostEqual(
            numpy_buffers.positions_diag([(0.0, 0.0, 0.0), (3.0, 4.0, 12.0)]),
            13.0,
        )
        np = draw_arrays.require_numpy()
        self.assertAlmostEqual(
            numpy_buffers.positions_diag(np.asarray([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)])),
            5.0,
        )

    def test_build_topology_arrays_remaps_sparse_vertices(self):
        np = draw_arrays.require_numpy()
        triangles, original_vertex_ids, source_triangles = draw_arrays.build_topology_arrays(
            np.asarray([9, 4, 7, 9, 7, 4], dtype=np.int64)
        )
        self.assertEqual(original_vertex_ids.tolist(), [4, 7, 9])
        self.assertEqual(triangles.tolist(), [[2, 0, 1], [2, 1, 0]])
        self.assertEqual(source_triangles.tolist(), [[9, 4, 7], [9, 7, 4]])

    def test_skin_signature_uses_weighted_slots(self):
        signature = draw_arrays.skin_signature(
            [(0.0, 0.0, 0.0), (0.0, 3.0, 4.0)],
            [(5, 2, 9, 0), (9, 0, 0, 0)],
            [(0.5, 0.0, 0.5, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )
        self.assertEqual(signature["used_slots"], [5, 9])
        self.assertEqual(signature["slot_count"], 2)
        self.assertEqual(signature["weighted_vertex_count"], 2)
        self.assertEqual(signature["diag"], 5.0)


if __name__ == "__main__":
    unittest.main()
