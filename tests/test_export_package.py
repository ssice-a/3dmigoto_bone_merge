from __future__ import annotations

import importlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

export_package = importlib.import_module(f"{PACKAGE_DIR.name}.core.export_package")
export_buffers = importlib.import_module(f"{PACKAGE_DIR.name}.core.export_buffers")
export_prepare = importlib.import_module(f"{PACKAGE_DIR.name}.core.export_prepare")


class FakeObject:
    type = "MESH"

    def __init__(self, name: str, groups, *, positions=None, triangles=None, layout=None):
        self.name = name
        self.groups = tuple(groups)
        self._props = {}
        if layout is not None:
            self._props["bmc_vertex_layout_json"] = json.dumps(layout, separators=(",", ":"))
            self._props["bmc_mirror_flip"] = False
        self.vertex_groups = [FakeVertexGroup(str(group), index) for index, group in enumerate(self.groups)]
        positions = list(positions or [(0.0, 0.0, 0.0)])
        triangles = list(triangles or [])
        self.data = FakeMeshData(positions, triangles, len(self.vertex_groups))

    def get(self, key, default=None):
        return self._props.get(key, default)

    def __setitem__(self, key, value):
        self._props[key] = value


class FakeMatrix:
    def __init__(self, offset):
        self.offset = tuple(offset)

    def __matmul__(self, value):
        return tuple(float(value[index]) + float(self.offset[index]) for index in range(3))


class FakeVertexGroup:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index


class FakeGroupElement:
    def __init__(self, group_index: int, weight: float):
        self.group = group_index
        self.weight = weight


class FakeVertex:
    def __init__(self, index: int, co, group_count: int):
        self.groups = [FakeGroupElement(group_index=group_index, weight=1.0) for group_index in range(group_count)]
        self.index = int(index)
        self.co = tuple(co)


class FakeLoop:
    def __init__(self, vertex_index: int):
        self.vertex_index = int(vertex_index)
        self.normal = (0.0, 0.0, 1.0)


class FakePolygon:
    def __init__(self, loop_indices):
        self.loop_indices = list(loop_indices)
        self.normal = (0.0, 0.0, 1.0)


class FakeMeshData:
    def __init__(self, positions, triangles, group_count: int):
        self.vertices = [
            FakeVertex(index, position, group_count)
            for index, position in enumerate(positions)
        ]
        self.loops = []
        self.polygons = []
        self.attributes = {}
        self.uv_layers = {}
        for triangle in triangles:
            loop_indices = []
            for vertex_index in triangle:
                loop_indices.append(len(self.loops))
                self.loops.append(FakeLoop(vertex_index))
            self.polygons.append(FakePolygon(loop_indices))


class FakeAttributeValue:
    def __init__(self, value):
        self.value = value


class FakeAttribute:
    def __init__(self, values):
        self.data = [FakeAttributeValue(value) for value in values]


class FakeUVValue:
    def __init__(self, uv):
        self.uv = tuple(uv)


class FakeUVLayer:
    def __init__(self, values):
        self.data = [FakeUVValue(value) for value in values]


class FakeUVLayers:
    def __init__(self, layers, active_name: str | None = None):
        self._layers = dict(layers)
        self.active = self._layers.get(active_name) if active_name else None

    def get(self, name):
        return self._layers.get(name)

    def __getitem__(self, index):
        return list(self._layers.values())[index]

    def __iter__(self):
        return iter(self._layers.values())


class FakeCollection:
    def __init__(self, name: str, objects=None, children=None):
        self.name = name
        self.objects = list(objects or [])
        self.children = list(children or [])


class FakeScene:
    def __init__(self, output_dir: str):
        self.bmc_output_dir = output_dir
        self.bmc_frameanalysis_dir = output_dir
        self.bmc_mirror_flip = True
        self.bmc_uv_flip_v = True


class FakeContext:
    def __init__(self, output_dir: str):
        self.scene = FakeScene(output_dir)


def collect_groups(mesh_obj):
    return mesh_obj.groups


class ExportPackageTests(unittest.TestCase):
    def test_implicit_region_part_builds_sorted_palette(self):
        mesh_a = FakeObject("body", [9, 0, 4])
        mesh_b = FakeObject("arm", [31, 4])
        root = FakeCollection(
            "ExportRoot",
            children=[FakeCollection("640d1c0e-46845-0", objects=[mesh_a, mesh_b])],
        )

        plan = export_package.build_export_plan(root, collect_groups)

        self.assertEqual(len(plan.parts), 1)
        part = plan.parts[0]
        self.assertEqual(part.region.ib_hash, "640d1c0e")
        self.assertEqual(part.region.match_index_count, 46845)
        self.assertEqual(part.region.match_first_index, 0)
        self.assertEqual(part.part_name, "part00")
        self.assertEqual(part.palette_values, (0, 4, 9, 31))
        self.assertEqual(part.palette_file_name, "640d1c0e-46845-0_part00-PartLocalToGlobalBoneMap.buf")

    def test_explicit_parts_keep_independent_palettes(self):
        root = FakeCollection(
            "ExportRoot",
            children=[
                FakeCollection(
                    "aaaaaaaa-10-5",
                    children=[
                        FakeCollection("part00", objects=[FakeObject("a", [5, 6])]),
                        FakeCollection("part01", objects=[FakeObject("b", [1, 6])]),
                    ],
                )
            ],
        )

        plan = export_package.build_export_plan(root, collect_groups)

        self.assertEqual([part.part_name for part in plan.parts], ["part00", "part01"])
        self.assertEqual(plan.parts[0].palette_values, (5, 6))
        self.assertEqual(plan.parts[1].palette_values, (1, 6))
        self.assertEqual(plan.parts[1].region.match_first_index, 5)

    def test_explicit_parts_accept_blender_unique_name_suffixes(self):
        root = FakeCollection(
            "ExportRoot",
            children=[
                FakeCollection(
                    "aaaaaaaa-10-5",
                    children=[
                        FakeCollection("part00.001", objects=[FakeObject("a", [5, 6])]),
                        FakeCollection("part01.001", objects=[FakeObject("b", [1, 6])]),
                    ],
                )
            ],
        )

        plan = export_package.build_export_plan(root, collect_groups)

        self.assertEqual([part.part_name for part in plan.parts], ["part00", "part01"])
        self.assertEqual(plan.parts[0].palette_values, (5, 6))
        self.assertEqual(plan.parts[1].palette_values, (1, 6))

    def test_direct_meshes_with_explicit_parts_are_planned_as_part00_with_warning(self):
        root = FakeCollection(
            "ExportRoot",
            children=[
                FakeCollection(
                    "bbbbbbbb-20-0",
                    objects=[FakeObject("direct", [3])],
                    children=[FakeCollection("part01", objects=[FakeObject("explicit", [4])])],
                )
            ],
        )

        plan = export_package.build_export_plan(root, collect_groups)

        self.assertEqual([part.part_name for part in plan.parts], ["part00", "part01"])
        self.assertIn("direct mesh object", plan.warnings[0])

    def test_object_level_split_when_implicit_part_exceeds_limit(self):
        root = FakeCollection(
            "ExportRoot",
            children=[
                FakeCollection(
                    "cccccccc-30-0",
                    objects=[
                        FakeObject("a", [0, 1, 2]),
                        FakeObject("b", [3, 4, 5]),
                        FakeObject("c", [2, 5]),
                    ],
                )
            ],
        )

        plan = export_package.build_export_plan(root, collect_groups, max_bones_per_part=4)

        self.assertEqual(len(plan.parts), 2)
        for part in plan.parts:
            self.assertLessEqual(len(part.palette_values), 4)
        self.assertTrue(any(part.generated for part in plan.parts))

    def test_single_object_over_limit_reports_triangle_split_requirement(self):
        root = FakeCollection(
            "ExportRoot",
            children=[
                FakeCollection(
                    "dddddddd-40-0",
                    objects=[FakeObject("oversized", [0, 1, 2, 3, 4])],
                )
            ],
        )

        with self.assertRaisesRegex(export_package.ExportPlanError, "triangle-level splitting"):
            export_package.build_export_plan(root, collect_groups, max_bones_per_part=4)

    def test_write_r32_index_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "part00-ib.buf")
            export_package.write_r32_index_buffer(path, [0, 65536, 3])
            with open(path, "rb") as handle:
                data = handle.read()

        self.assertEqual(struct.unpack("<3I", data), (0, 65536, 3))

    def test_prepare_export_buffer_only_writes_runtime_buffers_but_skips_ini_and_hlsl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = {
                "buffers": {
                    "vb0": {
                        "slot": "vb0",
                        "stride": 12,
                        "elements": [
                            {
                                "semantic_name": "POSITION",
                                "semantic_index": 0,
                                "format": "R32G32B32_FLOAT",
                                "aligned_byte_offset": 0,
                            }
                        ],
                    },
                    "vb2": {
                        "slot": "vb2",
                        "stride": 12,
                        "elements": [
                            {
                                "semantic_name": "BLENDWEIGHTS",
                                "semantic_index": 0,
                                "format": "R16G16B16A16_UNORM",
                                "aligned_byte_offset": 0,
                            },
                            {
                                "semantic_name": "BLENDINDICES",
                                "semantic_index": 0,
                                "format": "R8G8B8A8_UINT",
                                "aligned_byte_offset": 8,
                            },
                        ],
                    },
                }
            }
            root = FakeCollection(
                "ExportRoot",
                children=[
                    FakeCollection(
                        "640d1c0e-46845-0",
                        objects=[
                            FakeObject(
                                "body",
                                [0, 4],
                                positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                                triangles=[(0, 1, 2)],
                            )
                        ],
                    )
                ],
            )
            capture_manifest_path = Path(tmpdir) / "capture_manifest.json"
            capture_manifest_path.write_text(
                json.dumps(
                    {
                        "bone_pool_order": [
                            {
                                "ib_hash": "640d1c0e",
                                "match_first_index": 0,
                                "match_index_count": 46845,
                                "global_bone_base": 0,
                                "local_bone_count": 5,
                                "used_local_bone_indices": [0, 1, 2, 3, 4],
                                "bone_capture_available": True,
                            }
                        ],
                        "vertex_layout_table": {
                            "640d1c0e-46845-0": layout,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = export_prepare.prepare_export_collection(
                context=FakeContext(tmpdir),
                source_collection=root,
                output_dir=tmpdir,
                capture_manifest_path=str(capture_manifest_path),
                generate_ini=False,
            )

            manifest_path = Path(result["manifest_path"])
            self.assertTrue(manifest_path.exists())
            self.assertEqual(result["bonestore_ini_path"], "")
            self.assertEqual(result["hlsl_dir"], "")
            buffer_dir = Path(tmpdir) / "Buffer"
            self.assertTrue((buffer_dir / "640d1c0e-46845-0_part00-PartLocalToGlobalBoneMap.buf").exists())
            self.assertTrue((buffer_dir / "640d1c0e-46845-0_part00-Index.buf").exists())
            self.assertTrue((buffer_dir / "640d1c0e-46845-0_part00-Position.buf").exists())
            self.assertTrue((buffer_dir / "640d1c0e-46845-0_part00-Blend.buf").exists())
            self.assertTrue((buffer_dir / "MainCaptureBoneMap.buf").exists())
            self.assertFalse((buffer_dir / "MainCaptureRecords.buf").exists())
            self.assertFalse((buffer_dir / "MainCaptureSourceLocalBones.buf").exists())
            self.assertFalse((buffer_dir / "LodCaptureBoneMap.buf").exists())
            self.assertFalse((Path(tmpdir) / "BoneStore.ini").exists())
            self.assertFalse((Path(tmpdir) / "hlsl").exists())

            export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("runtime", export_manifest)
            self.assertIn("main_capture_bone_map", export_manifest["runtime"]["buffers"])
            self.assertEqual(
                struct.unpack("<3I", (buffer_dir / "640d1c0e-46845-0_part00-Index.buf").read_bytes()),
                (0, 2, 1),
            )
            self.assertEqual(
                export_manifest["geometry_buffers"][0]["vertex_buffers"]["vb2"]["stride"],
                12,
            )

    def test_prepare_export_names_ini_after_export_root_collection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = FakeCollection(
                "LXi Final Export",
                children=[
                    FakeCollection(
                        "640d1c0e-3-0",
                        objects=[
                            FakeObject(
                                "body",
                                [0],
                                positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                                triangles=[(0, 1, 2)],
                            )
                        ],
                    )
                ],
            )
            capture_manifest_path = Path(tmpdir) / "capture_manifest.json"
            capture_manifest_path.write_text(
                json.dumps(
                    {
                        "shadow_stage": {"shadow_vs_hashes": ["aaaaaaaaaaaaaaaa"]},
                        "bone_pool_order": [
                            {
                                "ib_hash": "640d1c0e",
                                "match_first_index": 0,
                                "match_index_count": 3,
                                "global_bone_base": 0,
                                "local_bone_count": 1,
                                "used_local_bone_indices": [0],
                                "bone_capture_available": True,
                            }
                        ],
                        "vertex_layout_table": {
                            "640d1c0e-3-0": {
                                "buffers": {
                                    "vb0": {"slot": "vb0", "stride": 16, "elements": []},
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = export_prepare.prepare_export_collection(
                context=FakeContext(tmpdir),
                source_collection=root,
                output_dir=tmpdir,
                capture_manifest_path=str(capture_manifest_path),
                generate_ini=True,
            )

            self.assertEqual(Path(result["bonestore_ini_path"]).name, "LXi Final Export.ini")
            self.assertTrue((Path(tmpdir) / "LXi Final Export.ini").exists())
            self.assertFalse((Path(tmpdir) / "BoneStore.ini").exists())
            export_manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(export_manifest["ini_file_name"], "LXi Final Export.ini")
            self.assertEqual(export_manifest["runtime"]["ini_file_name"], "LXi Final Export.ini")

    def test_texcoord4_missing_on_export_mesh_ignores_source_ib_object_and_uses_default(self):
        source = FakeObject(
            "640d1c0e-3-0-source",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[],
        )
        source["bmc_source_ib_hash"] = "640d1c0e"
        source["bmc_match_index_count"] = 3
        source["bmc_match_first_index"] = 0
        for component, values in enumerate(([1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12])):
            source.data.attributes[f"bmc_vb1_texcoord4_{component}"] = FakeAttribute(values)

        target = FakeObject(
            "Body.001",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        root = FakeCollection(
            "ExportRoot",
            objects=[source],
            children=[FakeCollection("640d1c0e-3-0", objects=[target])],
        )
        plan = export_package.build_export_plan(root, collect_groups)
        layout = {
            "640d1c0e-3-0": {
                "buffers": {
                    "vb1": {
                        "slot": "vb1",
                        "stride": 4,
                        "elements": [
                            {
                                "semantic_name": "TEXCOORD",
                                "semantic_index": 4,
                                "format": "R8G8B8A8_SNORM",
                                "aligned_byte_offset": 0,
                            }
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export_buffers.write_part_geometry_buffers(
                tmpdir,
                plan.parts,
                layout,
            )
            texcoord_path = Path(tmpdir) / "640d1c0e-3-0_part00-Texcoord.buf"
            self.assertEqual(texcoord_path.read_bytes(), bytes([0] * 12))

    def test_texcoord4_missing_without_source_uses_neutral_packed_default(self):
        target = FakeObject(
            "Body.001",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        root = FakeCollection(
            "ExportRoot",
            children=[FakeCollection("640d1c0e-3-0", objects=[target])],
        )
        plan = export_package.build_export_plan(root, collect_groups)
        layout = {
            "640d1c0e-3-0": {
                "buffers": {
                    "vb1": {
                        "slot": "vb1",
                        "stride": 4,
                        "elements": [
                            {
                                "semantic_name": "TEXCOORD",
                                "semantic_index": 4,
                                "format": "R8G8B8A8_SNORM",
                                "aligned_byte_offset": 0,
                            }
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export_buffers.write_part_geometry_buffers(
                tmpdir,
                plan.parts,
                layout,
            )
            texcoord_path = Path(tmpdir) / "640d1c0e-3-0_part00-Texcoord.buf"
            self.assertEqual(texcoord_path.read_bytes(), bytes([0] * 12))

    def test_external_mesh_without_mirror_property_uses_export_default(self):
        target = FakeObject(
            "ExternalBody",
            [0],
            positions=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)],
            triangles=[(0, 1, 2)],
        )
        root = FakeCollection(
            "ExportRoot",
            children=[FakeCollection("640d1c0e-3-0", objects=[target])],
        )
        plan = export_package.build_export_plan(root, collect_groups)
        layout = {
            "640d1c0e-3-0": {
                "buffers": {
                    "vb0": {
                        "slot": "vb0",
                        "stride": 12,
                        "elements": [
                            {
                                "semantic_name": "POSITION",
                                "semantic_index": 0,
                                "format": "R32G32B32_FLOAT",
                                "aligned_byte_offset": 0,
                            }
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export_buffers.write_part_geometry_buffers(
                tmpdir,
                plan.parts,
                layout,
                mirror_flip_default=True,
            )
            position_path = Path(tmpdir) / "640d1c0e-3-0_part00-Position.buf"
            values = struct.unpack("<9f", position_path.read_bytes())
            self.assertEqual(values[:3], (-1.0, 2.0, 3.0))

    def test_export_position_applies_object_matrix_before_game_mirror(self):
        target = FakeObject(
            "ExternalBody",
            [0],
            positions=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)],
            triangles=[(0, 1, 2)],
        )
        target.matrix_world = FakeMatrix((10.0, 20.0, 30.0))
        root = FakeCollection(
            "ExportRoot",
            children=[FakeCollection("640d1c0e-3-0", objects=[target])],
        )
        plan = export_package.build_export_plan(root, collect_groups)
        layout = {
            "640d1c0e-3-0": {
                "buffers": {
                    "vb0": {
                        "slot": "vb0",
                        "stride": 12,
                        "elements": [
                            {
                                "semantic_name": "POSITION",
                                "semantic_index": 0,
                                "format": "R32G32B32_FLOAT",
                                "aligned_byte_offset": 0,
                            }
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export_buffers.write_part_geometry_buffers(
                tmpdir,
                plan.parts,
                layout,
                mirror_flip_default=True,
            )
            position_path = Path(tmpdir) / "640d1c0e-3-0_part00-Position.buf"
            values = struct.unpack("<9f", position_path.read_bytes())
            self.assertEqual(values[:3], (-11.0, 22.0, 33.0))

    def test_external_mesh_texcoord0_uses_primary_uv_when_layer_is_not_named_uv0(self):
        target = FakeObject(
            "Body.001",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        target.data.uv_layers = FakeUVLayers(
            {
                "UVMap": FakeUVLayer([(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]),
            },
            active_name="UVMap",
        )
        root = FakeCollection(
            "ExportRoot",
            children=[FakeCollection("640d1c0e-3-0", objects=[target])],
        )
        plan = export_package.build_export_plan(root, collect_groups)
        layout = {
            "640d1c0e-3-0": {
                "buffers": {
                    "vb1": {
                        "slot": "vb1",
                        "stride": 8,
                        "elements": [
                            {
                                "semantic_name": "TEXCOORD",
                                "semantic_index": 0,
                                "format": "R32G32_FLOAT",
                                "aligned_byte_offset": 0,
                            }
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export_buffers.write_part_geometry_buffers(
                tmpdir,
                plan.parts,
                layout,
                mirror_flip_default=False,
            )
            texcoord_path = Path(tmpdir) / "640d1c0e-3-0_part00-Texcoord.buf"
            values = struct.unpack("<6f", texcoord_path.read_bytes())
            self.assertEqual(values, (0.25, 0.25, 0.5, 0.5, 0.75, 0.75))

    def test_external_mesh_texcoord0_can_disable_uv_v_flip(self):
        target = FakeObject(
            "Body.001",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        target.data.uv_layers = FakeUVLayers(
            {
                "UVMap": FakeUVLayer([(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]),
            },
            active_name="UVMap",
        )
        root = FakeCollection(
            "ExportRoot",
            children=[FakeCollection("640d1c0e-3-0", objects=[target])],
        )
        plan = export_package.build_export_plan(root, collect_groups)
        layout = {
            "640d1c0e-3-0": {
                "buffers": {
                    "vb1": {
                        "slot": "vb1",
                        "stride": 8,
                        "elements": [
                            {
                                "semantic_name": "TEXCOORD",
                                "semantic_index": 0,
                                "format": "R32G32_FLOAT",
                                "aligned_byte_offset": 0,
                            }
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export_buffers.write_part_geometry_buffers(
                tmpdir,
                plan.parts,
                layout,
                mirror_flip_default=False,
                uv_flip_v_default=False,
            )
            texcoord_path = Path(tmpdir) / "640d1c0e-3-0_part00-Texcoord.buf"
            values = struct.unpack("<6f", texcoord_path.read_bytes())
            self.assertEqual(values, (0.25, 0.75, 0.5, 0.5, 0.75, 0.25))


if __name__ == "__main__":
    unittest.main()
