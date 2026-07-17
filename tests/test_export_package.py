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
vertex_format = importlib.import_module(f"{PACKAGE_DIR.name}.core.vertex_format")


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

    def __iter__(self):
        yield (1.0, 0.0, 0.0, float(self.offset[0]))
        yield (0.0, 1.0, 0.0, float(self.offset[1]))
        yield (0.0, 0.0, 1.0, float(self.offset[2]))
        yield (0.0, 0.0, 0.0, 1.0)


class FakeVertexGroup:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index


class FakeDataList(list):
    def foreach_get(self, attribute_name, values):
        offset = 0
        for item in self:
            raw_value = getattr(item, attribute_name)
            if isinstance(raw_value, (tuple, list)):
                for component in raw_value:
                    values[offset] = component
                    offset += 1
            else:
                values[offset] = raw_value
                offset += 1


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


class FakeLoopTriangle:
    def __init__(self, loop_indices):
        self.loops = tuple(loop_indices)


class FakeMeshData:
    def __init__(self, positions, triangles, group_count: int):
        self.vertices = FakeDataList(
            FakeVertex(index, position, group_count)
            for index, position in enumerate(positions)
        )
        self.loops = FakeDataList()
        self.polygons = []
        self.loop_triangles = FakeDataList()
        self.attributes = {}
        self.uv_layers = {}
        for triangle in triangles:
            loop_indices = []
            for vertex_index in triangle:
                loop_indices.append(len(self.loops))
                self.loops.append(FakeLoop(vertex_index))
            self.polygons.append(FakePolygon(loop_indices))
            self.loop_triangles.append(FakeLoopTriangle(loop_indices))


class FakeAttributeValue:
    def __init__(self, value):
        self.value = value


class FakeAttribute:
    def __init__(self, values):
        self.data = FakeDataList(FakeAttributeValue(value) for value in values)


class FakeColorValue:
    def __init__(self, color):
        self.color = tuple(color)


class FakeColorAttribute:
    def __init__(self, values, *, domain="POINT"):
        self.data = FakeDataList(FakeColorValue(value) for value in values)
        self.domain = domain


class FakeUVValue:
    def __init__(self, uv):
        self.uv = tuple(uv)


class FakeUVLayer:
    def __init__(self, values):
        self.data = FakeDataList(FakeUVValue(value) for value in values)


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
    def test_cpu_pre_skinned_import_is_rejected_from_replacement_export(self):
        mesh = FakeObject("face [CPU_SKINNED_UNSUPPORTED]", [0, 1])
        mesh["bmc_replacement_supported"] = False
        mesh["bmc_replacement_unsupported_reason"] = "cpu_pre_skinned_vertex_stream"
        root = FakeCollection(
            "ExportRoot",
            children=[FakeCollection("945c08a9-1698-0", objects=[mesh])],
        )

        with self.assertRaisesRegex(export_package.ExportPlanError, "CPU pre-skinned draw cannot be exported"):
            export_package.build_export_plan(root, collect_groups)

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

    def test_direct_meshes_with_explicit_parts_are_rejected(self):
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

        with self.assertRaisesRegex(export_package.ExportPlanError, "Move them into an explicit part00"):
            export_package.build_export_plan(root, collect_groups)

    def test_region_child_mesh_collections_must_be_explicit_parts(self):
        root = FakeCollection(
            "ExportRoot",
            children=[
                FakeCollection(
                    "bbbbbbbb-20-0",
                    children=[FakeCollection("loose_child", objects=[FakeObject("loose", [3])])],
                )
            ],
        )

        with self.assertRaisesRegex(export_package.ExportPlanError, "must be named partNN"):
            export_package.build_export_plan(root, collect_groups)

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

    def test_one_part_records_per_object_draw_ranges(self):
        mesh_a = FakeObject(
            "body",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        mesh_b = FakeObject(
            "hair",
            [1],
            positions=[(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)],
            triangles=[(0, 1, 2)],
        )
        root = FakeCollection(
            "ExportRoot",
            children=[FakeCollection("640d1c0e-6-0", objects=[mesh_a, mesh_b])],
        )
        layout = {
            "640d1c0e-6-0": {
                "buffers": {
                    "vb0": {
                        "slot": "vb0",
                        "stride": 16,
                        "elements": [],
                    }
                }
            }
        }

        plan = export_package.build_export_plan(root, collect_groups)
        with tempfile.TemporaryDirectory() as tmpdir:
            records = export_buffers.write_part_geometry_buffers(tmpdir, plan.parts, layout)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["index_buffer"]["index_count"], 6)
        self.assertEqual(
            records[0]["object_draws"],
            [
                {
                    "object_name": "body",
                    "start_index": 0,
                    "index_count": 3,
                    "base_vertex": 0,
                    "start_vertex": 0,
                    "vertex_count": 3,
                },
                {
                    "object_name": "hair",
                    "start_index": 3,
                    "index_count": 3,
                    "base_vertex": 0,
                    "start_vertex": 3,
                    "vertex_count": 3,
                },
            ],
        )

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
            self.assertEqual(
                struct.unpack("<3I", (buffer_dir / "640d1c0e-46845-0_part00-PartLocalToGlobalBoneMap.buf").read_bytes()),
                (2, 0, 4),
            )
            self.assertTrue((buffer_dir / "640d1c0e-46845-0_part00-Index.buf").exists())
            self.assertTrue((buffer_dir / "640d1c0e-46845-0_part00-Position.buf").exists())
            self.assertTrue((buffer_dir / "640d1c0e-46845-0_part00-Blend.buf").exists())
            self.assertTrue((buffer_dir / "640d1c0e_46845_0_MAIN-CaptureBoneMap.buf").exists())
            self.assertFalse((buffer_dir / "MainCaptureBoneMap.buf").exists())
            self.assertFalse((buffer_dir / "MainCaptureRecords.buf").exists())
            self.assertFalse((buffer_dir / "MainCaptureSourceLocalBones.buf").exists())
            self.assertFalse((buffer_dir / "LodCaptureBoneMap.buf").exists())
            self.assertFalse((Path(tmpdir) / "BoneStore.ini").exists())
            self.assertFalse((Path(tmpdir) / "hlsl").exists())

            export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("runtime", export_manifest)
            self.assertNotIn("performance", export_manifest)
            self.assertIn("performance", result)
            self.assertNotIn("timings", export_manifest["geometry_buffers"][0])
            self.assertNotIn("stats", export_manifest["geometry_buffers"][0])
            self.assertIn("capture_bone_maps", export_manifest["runtime"]["buffers"])
            self.assertEqual(
                struct.unpack("<3I", (buffer_dir / "640d1c0e-46845-0_part00-Index.buf").read_bytes()),
                (0, 2, 1),
            )
            self.assertEqual(
                export_manifest["geometry_buffers"][0]["vertex_buffers"]["vb2"]["stride"],
                12,
            )

    def test_simple_override_keeps_source_local_blend_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = {
                "buffers": {
                    "vb0": {
                        "slot": "vb0",
                        "stride": 16,
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
                "Simple Export",
                children=[
                    FakeCollection(
                        "640d1c0e-3-0",
                        objects=[
                            FakeObject(
                                "cpu_skin",
                                [10, 11],
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
                        "vertex_layout_table": {"640d1c0e-3-0": layout},
                        "bone_pool_order": [
                            {
                                "ib_hash": "640d1c0e",
                                "match_first_index": 0,
                                "match_index_count": 3,
                                "global_bone_base": 10,
                                "local_bone_count": 2,
                                "source_local_bone_count": 5,
                                "used_local_bone_indices": [2, 4],
                                "bone_capture_available": True,
                            }
                        ],
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
                simple_override=True,
            )

            buffer_dir = Path(tmpdir) / "Buffer"
            blend = (buffer_dir / "640d1c0e-3-0_part00-Blend.buf").read_bytes()
            self.assertEqual(tuple(blend[8:12]), (2, 4, 0, 0))
            self.assertEqual(Path(result["bonestore_ini_path"]).name, "Simple Export.ini")
            ini_text = Path(result["bonestore_ini_path"]).read_text(encoding="utf-8")
            self.assertIn("[TextureOverride_BMCSimple_640d1c0e_3_0]", ini_text)
            self.assertIn("if vs == 201 || vs == 202 || vs == 203", ini_text)
            self.assertNotIn("[ShaderOverride", ini_text)
            self.assertNotIn("instance_count <= 8", ini_text)
            self.assertIn("vb2 = ResourcePart_640d1c0e_3_0_part00_Blend", ini_text)
            self.assertIn("drawindexedinstanced = 3,INSTANCE_COUNT,0,0,FIRST_INSTANCE", ini_text)
            self.assertNotIn("CustomShader_GatherLocalBones", ini_text)
            self.assertFalse((Path(tmpdir) / "hlsl").exists())
            self.assertFalse((buffer_dir / "640d1c0e_3_0_MAIN-CaptureBoneMap.buf").exists())
            export_manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(export_manifest["runtime"]["mode"], "simple_override")
            self.assertEqual(export_manifest["runtime"]["ini_file_name"], "Simple Export.ini")
            self.assertEqual(
                export_manifest["runtime"]["geometry"][0]["vertex_buffers"]["vb2"]["resource_name"],
                "ResourcePart_640d1c0e_3_0_part00_Blend",
            )
            self.assertEqual(export_manifest["runtime"]["textures"], [])

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
            record_bones_shader = (Path(tmpdir) / "hlsl" / "record_bones_cs.hlsl").read_text(encoding="utf-8")
            self.assertIn("capture_valid", record_bones_shader)
            self.assertIn("NativeT0.GetDimensions", record_bones_shader)
            self.assertIn("required_current_row < native_rows", record_bones_shader)
            self.assertNotIn("source_count", record_bones_shader)
            export_manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(export_manifest["ini_file_name"], "LXi Final Export.ini")
            self.assertEqual(export_manifest["runtime"]["ini_file_name"], "LXi Final Export.ini")

    def test_regenerate_runtime_files_refreshes_bundled_hlsl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hlsl_dir = Path(tmpdir) / "hlsl"
            hlsl_dir.mkdir()
            (hlsl_dir / "record_bones_cs.hlsl").write_text("// stale shader\n", encoding="utf-8")
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
                    }
                ),
                encoding="utf-8",
            )

            export_prepare.regenerate_bonestore_runtime_files(
                output_dir=tmpdir,
                capture_manifest_path=str(capture_manifest_path),
                write_ini=True,
            )

            record_bones_shader = (hlsl_dir / "record_bones_cs.hlsl").read_text(encoding="utf-8")
            self.assertNotIn("stale shader", record_bones_shader)
            self.assertIn("capture_valid", record_bones_shader)
            self.assertIn("NativeT0.GetDimensions", record_bones_shader)

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

    def test_texcoord4_raw_point_attributes_are_not_reused_without_color(self):
        target = FakeObject(
            "Body.001",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        for component, values in enumerate(([1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12])):
            target.data.attributes[f"bmc_vb1_texcoord4_{component}"] = FakeAttribute(values)
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
            export_buffers.write_part_geometry_buffers(tmpdir, plan.parts, layout)
            texcoord_path = Path(tmpdir) / "640d1c0e-3-0_part00-Texcoord.buf"
            self.assertEqual(texcoord_path.read_bytes(), bytes([0] * 12))

    def test_texcoord4_color_attribute_is_encoded(self):
        target = FakeObject(
            "Body.001",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        target.data.attributes["bmc_vb1_texcoord4_color"] = FakeColorAttribute(
            [
                (1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 4.0 / 255.0),
                (5.0 / 255.0, 6.0 / 255.0, 7.0 / 255.0, 8.0 / 255.0),
                (9.0 / 255.0, 10.0 / 255.0, 11.0 / 255.0, 12.0 / 255.0),
            ]
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
            export_buffers.write_part_geometry_buffers(tmpdir, plan.parts, layout)
            texcoord_path = Path(tmpdir) / "640d1c0e-3-0_part00-Texcoord.buf"
            self.assertEqual(texcoord_path.read_bytes(), bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

    def test_packed_normal_export_requires_tangent_frame(self):
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
                    "vb0": {
                        "slot": "vb0",
                        "stride": 16,
                        "elements": [
                            {
                                "semantic_name": "POSITION",
                                "semantic_index": 0,
                                "format": "R32G32B32_FLOAT",
                                "aligned_byte_offset": 0,
                            },
                            {
                                "semantic_name": "NORMAL",
                                "semantic_index": 0,
                                "format": "R32_FLOAT",
                                "aligned_byte_offset": 12,
                            },
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "packed NORMAL0 export requires"):
                export_buffers.write_part_geometry_buffers(tmpdir, plan.parts, layout)

    def test_packed_normal_export_uses_loop_tangent_frame(self):
        target = FakeObject(
            "Body.001",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        for loop in target.data.loops:
            loop.tangent = (-1.0, 0.0, 0.0)
            loop.bitangent_sign = 1.0
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
                        "stride": 16,
                        "elements": [
                            {
                                "semantic_name": "POSITION",
                                "semantic_index": 0,
                                "format": "R32G32B32_FLOAT",
                                "aligned_byte_offset": 0,
                            },
                            {
                                "semantic_name": "NORMAL",
                                "semantic_index": 0,
                                "format": "R32_FLOAT",
                                "aligned_byte_offset": 12,
                            },
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
            position_path = Path(tmpdir) / "640d1c0e-3-0_part00-Position.buf"
            packed = struct.unpack_from("<I", position_path.read_bytes(), 12)[0]
            self.assertTrue(packed & 0x40000000)
            self.assertTrue(packed & 0x80000000)
            self.assertNotEqual((packed >> 20) & 0x3FF, 0)

    def test_packed_normal_handedness_accounts_for_uv_and_mirror_flip(self):
        target = FakeObject(
            "Body.001",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        for loop in target.data.loops:
            loop.tangent = (-1.0, 0.0, 0.0)
            loop.bitangent_sign = 1.0
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
                        "stride": 16,
                        "elements": [
                            {
                                "semantic_name": "POSITION",
                                "semantic_index": 0,
                                "format": "R32G32B32_FLOAT",
                                "aligned_byte_offset": 0,
                            },
                            {
                                "semantic_name": "NORMAL",
                                "semantic_index": 0,
                                "format": "R32_FLOAT",
                                "aligned_byte_offset": 12,
                            },
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
                uv_flip_v_default=True,
            )
            position_path = Path(tmpdir) / "640d1c0e-3-0_part00-Position.buf"
            packed = struct.unpack_from("<I", position_path.read_bytes(), 12)[0]
            self.assertTrue(packed & 0x80000000)

    def test_packed_normal_export_prefers_imported_tangent_frame_attributes(self):
        target = FakeObject(
            "Body.001",
            [0],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
        )
        for loop in target.data.loops:
            loop.tangent = (1.0, 0.0, 0.0)
            loop.bitangent_sign = -1.0
        target.data.attributes["bmc_tangent_x"] = FakeAttribute([-1.0, -1.0, -1.0])
        target.data.attributes["bmc_tangent_y"] = FakeAttribute([0.0, 0.0, 0.0])
        target.data.attributes["bmc_tangent_z"] = FakeAttribute([0.0, 0.0, 0.0])
        target.data.attributes["bmc_bitangent_sign"] = FakeAttribute([1.0, 1.0, 1.0])
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
                        "stride": 16,
                        "elements": [
                            {
                                "semantic_name": "POSITION",
                                "semantic_index": 0,
                                "format": "R32G32B32_FLOAT",
                                "aligned_byte_offset": 0,
                            },
                            {
                                "semantic_name": "NORMAL",
                                "semantic_index": 0,
                                "format": "R32_FLOAT",
                                "aligned_byte_offset": 12,
                            },
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
            position_path = Path(tmpdir) / "640d1c0e-3-0_part00-Position.buf"
            packed = struct.unpack_from("<I", position_path.read_bytes(), 12)[0]
            expected = vertex_format.encode_game_packed_tangent_frame(
                (0.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
                1.0,
            )
            self.assertEqual(packed, expected)

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

    def test_texcoord_float_alias_does_not_break_uv_export(self):
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
                            },
                            {
                                "semantic_name": "TEXCOORD",
                                "semantic_index": 4,
                                "format": "R32G32_FLOAT",
                                "aligned_byte_offset": 0,
                            },
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export_buffers.write_part_geometry_buffers(tmpdir, plan.parts, layout, mirror_flip_default=False)
            texcoord_path = Path(tmpdir) / "640d1c0e-3-0_part00-Texcoord.buf"
            values = struct.unpack("<6f", texcoord_path.read_bytes())
            self.assertEqual(values, (0.25, 0.25, 0.5, 0.5, 0.75, 0.75))

    def test_texcoord_float_attributes_export_through_numpy_path(self):
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
        target.data.attributes["bmc_vb1_texcoord2_0"] = FakeAttribute([1.0, 2.0, 3.0])
        target.data.attributes["bmc_vb1_texcoord2_1"] = FakeAttribute([4.0, 5.0, 6.0])
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
                        "stride": 16,
                        "elements": [
                            {
                                "semantic_name": "TEXCOORD",
                                "semantic_index": 0,
                                "format": "R32G32_FLOAT",
                                "aligned_byte_offset": 0,
                            },
                            {
                                "semantic_name": "TEXCOORD",
                                "semantic_index": 2,
                                "format": "R32G32_FLOAT",
                                "aligned_byte_offset": 8,
                            },
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export_buffers.write_part_geometry_buffers(tmpdir, plan.parts, layout, mirror_flip_default=False)
            texcoord_path = Path(tmpdir) / "640d1c0e-3-0_part00-Texcoord.buf"
            values = struct.unpack("<12f", texcoord_path.read_bytes())
            self.assertEqual(values, (0.25, 0.25, 1.0, 4.0, 0.5, 0.5, 2.0, 5.0, 0.75, 0.75, 3.0, 6.0))


if __name__ == "__main__":
    unittest.main()
