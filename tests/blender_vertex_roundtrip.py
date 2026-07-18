"""Blender-side real-capture vertex round-trip audit.

Run with:
  blender --background --factory-startup --python tests/blender_vertex_roundtrip.py -- \
    --frameanalysis <FrameAnalysis directory>
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np


def main() -> int:
    args = _parse_args()
    repo_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_dir.parent))
    package_name = repo_dir.name
    addon = importlib.import_module(package_name)
    addon.register()
    export_buffers = importlib.import_module(f"{package_name}.core.export_buffers")
    export_package = importlib.import_module(f"{package_name}.core.export_package")
    import_candidates = importlib.import_module(f"{package_name}.core.import_candidates")
    layout_codec = importlib.import_module(f"{package_name}.core.vertex_layout_codec")
    main_analyze = importlib.import_module(f"{package_name}.core.main_analyze")
    vertex_groups = importlib.import_module(f"{package_name}.core.vertex_groups")
    _verify_all_byte_color_values(bpy, import_candidates, export_buffers)

    frameanalysis_dir = Path(args.frameanalysis).resolve()
    manifest_path = frameanalysis_dir / "capture_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else main_analyze.analyze_main_frameanalysis(str(frameanalysis_dir))
    )
    layout_table = dict(manifest.get("vertex_layout_table", {}) or {})
    reports = []
    for candidate in manifest.get("candidate_ibs", []) or []:
        report = _audit_candidate(
            bpy,
            candidate,
            manifest,
            layout_table,
            import_candidates,
            export_package,
            export_buffers,
            layout_codec,
            vertex_groups,
        )
        reports.append(report)
        has_mismatch = any(
            slot.get("byte_mismatch_count", 0) > 0
            for slot in report.get("slots", [])
        )
        has_mismatch = has_mismatch or bool(report.get("import_semantic_mismatch_count", 0))
        has_mismatch = has_mismatch or bool(report.get("blender_normal_warning_count", 0))
        if not args.only_mismatches or report["status"] == "error" or has_mismatch:
            print("BMC_ROUNDTRIP_ITEM=" + json.dumps(report, ensure_ascii=True, sort_keys=True))

    summary = {
        "frameanalysis": str(frameanalysis_dir),
        "byte_color_values_verified": 256,
        "candidate_count": len(reports),
        "tested_count": sum(report["status"] == "tested" for report in reports),
        "skipped_count": sum(report["status"] == "skipped" for report in reports),
        "error_count": sum(report["status"] == "error" for report in reports),
        "import_semantic_mismatch_count": sum(
            int(report.get("import_semantic_mismatch_count", 0))
            for report in reports
        ),
        "blender_normal_warning_count": sum(
            int(report.get("blender_normal_warning_count", 0))
            for report in reports
        ),
        "exact_slot_count": sum(
            slot["byte_mismatch_count"] == 0
            for report in reports
            for slot in report.get("slots", [])
            if slot.get("status") == "written"
        ),
        "mismatched_slot_count": sum(
            slot.get("semantic_mismatch_count", slot["byte_mismatch_count"]) > 0
            for report in reports
            for slot in report.get("slots", [])
            if slot.get("status") == "written"
        ),
        "palette_remap_slot_count": sum(
            bool(slot.get("palette_remap_only", False))
            for report in reports
            for slot in report.get("slots", [])
            if slot.get("status") == "written"
        ),
    }
    print("BMC_ROUNDTRIP_SUMMARY=" + json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 1 if (
        summary["error_count"]
        or summary["import_semantic_mismatch_count"]
        or summary["mismatched_slot_count"]
    ) else 0


def _verify_all_byte_color_values(bpy_module, import_candidates, export_buffers):
    mesh = bpy_module.data.meshes.new("__bmc_byte_color_probe__")
    mesh.from_pydata([(index, 0.0, 0.0) for index in range(256)], [], [])
    import_candidates._store_snorm_byte_color_attribute(
        mesh,
        "probe",
        [(value, value, value, value) for value in range(256)],
    )
    values = export_buffers._color_attribute_numpy_array(mesh.color_attributes["probe"])
    expected = np.arange(256, dtype=np.uint8)
    try:
        if not np.array_equal(values[:, 0], expected):
            mismatch = np.flatnonzero(values[:, 0] != expected).tolist()
            raise AssertionError(f"BYTE_COLOR round-trip mismatch: {mismatch}")
    finally:
        bpy_module.data.meshes.remove(mesh)


def _audit_candidate(
    bpy_module,
    candidate,
    manifest,
    layout_table,
    import_candidates,
    export_package,
    export_buffers,
    layout_codec,
    vertex_groups,
):
    display_name = str(candidate.get("display_name", "") or "")
    if not bool(candidate.get("replacement_supported", True)):
        return {
            "candidate": display_name,
            "status": "skipped",
            "reason": str(candidate.get("replacement_unsupported_reason", "") or "replacement unsupported"),
        }
    try:
        geometry = import_candidates.load_candidate_geometry(candidate, str(manifest.get("frameanalysis_dir", "") or ""))
        root = bpy_module.data.collections.new(f"RoundTripRoot-{display_name}")
        region = bpy_module.data.collections.new(display_name)
        bpy_module.context.scene.collection.children.link(root)
        root.children.link(region)
        mesh_obj = import_candidates.create_blender_object_from_geometry(
            bpy_module,
            geometry,
            region,
            draw_indices=list(candidate.get("draw_indices", []) or []),
            shadow_draw_indices=list(candidate.get("shadow_draw_indices", []) or []),
            mirror_flip=True,
            uv_flip_v=True,
        )
        import_semantics = _compare_imported_semantics(
            mesh_obj,
            geometry,
            export_buffers,
            layout_codec,
            mirror_flip=True,
            uv_flip_v=True,
        )
        plan = export_package.build_export_plan(
            root,
            vertex_groups.collect_weighted_numeric_vertex_groups,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            records = export_buffers.write_part_geometry_buffers(
                output_dir,
                plan.parts,
                layout_table,
                mirror_flip_default=True,
                uv_flip_v_default=True,
            )
            slots = _compare_slots(
                Path(output_dir),
                records[0],
                mesh_obj,
                geometry,
                layout_table[display_name],
                layout_codec,
                plan.parts[0].palette_values,
            )
        return {
            "candidate": display_name,
            "status": "tested",
            "vertex_count": len(geometry.positions),
            "triangle_count": len(geometry.triangles),
            "import_semantic_mismatch_count": sum(
                int(item["mismatch_count"])
                for item in import_semantics
                if not item.get("advisory", False)
            ),
            "blender_normal_warning_count": sum(
                int(item["mismatch_count"])
                for item in import_semantics
                if item.get("advisory", False)
            ),
            "import_semantics": import_semantics,
            "slots": slots,
        }
    except Exception as exc:
        return {
            "candidate": display_name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _compare_imported_semantics(
    mesh_obj,
    geometry,
    export_buffers,
    layout_codec,
    *,
    mirror_flip,
    uv_flip_v,
):
    mesh = mesh_obj.data
    point_count = len(mesh.vertices)
    loop_count = len(mesh.loops)
    loop_vertex_indices = np.empty(loop_count, dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_vertex_indices)
    reports = []

    expected_normals = np.asarray(geometry.normals, dtype=np.float32).copy()
    if mirror_flip and expected_normals.size:
        expected_normals[:, 0] *= -1.0
    normal_columns = [
        _float_attribute_values(mesh, f"bmc_normal_{axis}", point_count)
        for axis in ("x", "y", "z")
    ]
    if any(column is None for column in normal_columns):
        reports.append({"semantic": "NORMAL0 carrier", "mismatch_count": point_count})
    else:
        stored_normals = np.stack(normal_columns, axis=1)
        reports.append(
            {
                "semantic": "NORMAL0 carrier",
                "mismatch_count": _row_mismatch_count(stored_normals, expected_normals, atol=1.0e-6),
            }
        )
    loop_normals = np.empty((loop_count, 3), dtype=np.float32)
    mesh.loops.foreach_get("normal", loop_normals.reshape(-1))
    degenerate_loops = np.zeros(loop_count, dtype=bool)
    for polygon in mesh.polygons:
        if float(polygon.area) <= 1.0e-12:
            degenerate_loops[np.asarray(polygon.loop_indices, dtype=np.intp)] = True
    normal_report = _direction_mismatch_report(
        loop_normals,
        expected_normals[loop_vertex_indices],
        ignored=degenerate_loops,
    )
    if normal_report.get("samples"):
        loop_polygons = np.full(loop_count, -1, dtype=np.int64)
        for polygon in mesh.polygons:
            loop_polygons[np.asarray(polygon.loop_indices, dtype=np.intp)] = int(polygon.index)
        for sample in normal_report["samples"]:
            loop_index = int(sample["loop"])
            polygon_index = int(loop_polygons[loop_index])
            if polygon_index < 0:
                continue
            polygon = mesh.polygons[polygon_index]
            sample["polygon"] = polygon_index
            sample["polygon_area"] = float(polygon.area)
            sample["polygon_vertices"] = [int(value) for value in polygon.vertices]
            sample["polygon_positions"] = [
                list(mesh.vertices[int(value)].co)
                for value in polygon.vertices
            ]
    normal_report["semantic"] = "NORMAL0 Blender loops"
    normal_report["advisory"] = True
    reports.append(normal_report)

    tangent_columns = [
        _float_attribute_values(mesh, f"bmc_tangent_{axis}", point_count)
        for axis in ("x", "y", "z")
    ]
    has_packed_tangents = any(int(value) & 0x40000000 for value in geometry.normal_packed)
    if has_packed_tangents:
        expected_tangents = np.asarray(geometry.tangents, dtype=np.float32).copy()
        if mirror_flip and expected_tangents.size:
            expected_tangents[:, 0] *= -1.0
        tangent_mismatch = point_count if any(column is None for column in tangent_columns) else _row_mismatch_count(
            np.stack(tangent_columns, axis=1),
            expected_tangents,
            atol=1.0e-6,
        )
        reports.append({"semantic": "NORMAL0 tangent carrier", "mismatch_count": tangent_mismatch})
        actual_signs = _float_attribute_values(mesh, "bmc_bitangent_sign", point_count)
        expected_signs = np.asarray(geometry.bitangent_signs, dtype=np.float32).copy()
        if bool(mirror_flip) ^ bool(uv_flip_v):
            expected_signs *= -1.0
        reports.append(
            {
                "semantic": "NORMAL0 handedness carrier",
                "mismatch_count": point_count if actual_signs is None else _row_mismatch_count(
                    actual_signs,
                    expected_signs,
                    atol=0.0,
                ),
            }
        )

    for semantic in geometry.texcoord_semantics:
        storage = str(semantic.get("storage", "") or "")
        semantic_name = str(semantic.get("semantic_name", "") or "").upper()
        semantic_index = int(semantic.get("semantic_index", -1))
        slot_name = str(semantic.get("slot_name", "") or "")
        label = f"{slot_name} {semantic_name}{semantic_index}"
        if storage == "uv_layer":
            uv_indices = {
                int(alias.get("semantic_index", -1))
                for alias in semantic.get("aliases", [])
                if str(alias.get("semantic_name", "") or "").upper() == "TEXCOORD"
                and int(alias.get("semantic_index", -1)) in {0, 1}
            }
            uv_indices.add(semantic_index)
            for uv_index in sorted(uv_indices & {0, 1}):
                expected_uv = geometry.uv0 if uv_index == 0 else geometry.uv1
                reports.append(
                    _compare_uv_layer(
                        mesh,
                        f"UV{uv_index}",
                        expected_uv,
                        loop_vertex_indices,
                        uv_flip_v,
                        f"{slot_name} TEXCOORD{uv_index}",
                    )
                )
            continue
        expected = np.asarray(semantic.get("values", []))
        component_count = int(semantic.get("component_count", 0) or 0)
        if storage in {"sint8_raw", "uint8_raw"}:
            attribute_name = (
                f"bmc_{slot_name.lower()}_texcoord{semantic_index}_color"
                if semantic_name == "TEXCOORD"
                else layout_codec.semantic_color_attribute_name(
                    slot_name,
                    semantic_name,
                    semantic_index,
                )
            )
            attribute = mesh.color_attributes.get(attribute_name)
            if attribute is None:
                mismatch_count = point_count
            else:
                actual = export_buffers._color_attribute_numpy_array(attribute)
                expected_bytes = (expected.astype(np.int16) & 0xFF).astype(np.uint8)
                mismatch_count = _row_mismatch_count(actual, expected_bytes, atol=0.0)
            reports.append({"semantic": label, "mismatch_count": mismatch_count})
            continue
        columns = [
            _float_attribute_values(
                mesh,
                layout_codec.semantic_component_attribute_name(
                    slot_name,
                    semantic_name,
                    semantic_index,
                    component_index,
                ),
                point_count,
            )
            for component_index in range(component_count)
        ]
        mismatch_count = point_count if any(column is None for column in columns) else _row_mismatch_count(
            np.stack(columns, axis=1),
            expected,
            atol=1.0e-6,
        )
        reports.append({"semantic": label, "mismatch_count": mismatch_count})
    return reports


def _compare_uv_layer(mesh, layer_name, expected, loop_vertex_indices, uv_flip_v, label):
    layer = mesh.uv_layers.get(layer_name)
    if layer is None or expected is None:
        return {"semantic": label, "mismatch_count": len(loop_vertex_indices)}
    actual = np.empty((len(layer.data), 2), dtype=np.float32)
    layer.data.foreach_get("uv", actual.reshape(-1))
    expected_values = np.asarray(expected, dtype=np.float32)[loop_vertex_indices].copy()
    if uv_flip_v and expected_values.size:
        expected_values[:, 1] = 1.0 - expected_values[:, 1]
    return {
        "semantic": label,
        "mismatch_count": _row_mismatch_count(actual, expected_values, atol=1.0e-6),
    }


def _float_attribute_values(mesh, name, count):
    attribute = mesh.attributes.get(name)
    if attribute is None or len(attribute.data) < count:
        return None
    values = np.empty(count, dtype=np.float32)
    attribute.data.foreach_get("value", values)
    return values


def _row_mismatch_count(actual, expected, *, atol):
    actual_values = np.asarray(actual)
    expected_values = np.asarray(expected)
    if actual_values.shape != expected_values.shape:
        return max(len(actual_values), len(expected_values))
    equivalent = np.isclose(actual_values, expected_values, rtol=1.0e-6, atol=atol, equal_nan=True)
    if equivalent.ndim == 1:
        return int(np.count_nonzero(~equivalent))
    return int(np.count_nonzero(~np.all(equivalent, axis=1)))


def _direction_mismatch_report(actual, expected, *, ignored=None):
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    if actual_values.shape != expected_values.shape:
        return {
            "mismatch_count": max(len(actual_values), len(expected_values)),
            "minimum_dot": -1.0,
        }
    actual_lengths = np.linalg.norm(actual_values, axis=1)
    expected_lengths = np.linalg.norm(expected_values, axis=1)
    valid = (actual_lengths > 1.0e-12) & (expected_lengths > 1.0e-12)
    ignored_values = np.zeros(len(actual_values), dtype=bool) if ignored is None else np.asarray(ignored, dtype=bool)
    dots = np.zeros(len(actual_values), dtype=np.float64)
    dots[valid] = np.sum(actual_values[valid] * expected_values[valid], axis=1) / (
        actual_lengths[valid] * expected_lengths[valid]
    )
    mismatch = (~valid | (dots < 1.0 - 1.0e-4)) & ~ignored_values
    mismatch_indices = np.flatnonzero(mismatch)
    return {
        "mismatch_count": int(len(mismatch_indices)),
        "minimum_dot": float(np.min(dots)) if len(dots) else 1.0,
        "ignored_degenerate_loop_count": int(np.count_nonzero(ignored_values)),
        "samples": [
            {
                "loop": int(index),
                "dot": float(dots[index]),
                "actual": actual_values[index].tolist(),
                "expected": expected_values[index].tolist(),
            }
            for index in mismatch_indices[:3]
        ],
    }


def _compare_slots(
    output_dir,
    export_record,
    mesh_obj,
    geometry,
    raw_layout,
    layout_codec,
    palette_values,
):
    loop_vertex_indices = np.empty(len(mesh_obj.data.loops), dtype=np.int64)
    mesh_obj.data.loops.foreach_get("vertex_index", loop_vertex_indices)
    normalized = layout_codec.build_vertex_layout(raw_layout)
    written = dict(export_record.get("vertex_buffers", {}) or {})
    slots = []
    for slot_name, slot_layout in normalized.items():
        if slot_name not in written:
            alias = next(
                (
                    other_name
                    for other_name, other_layout in normalized.items()
                    if other_name in written and layout_codec.slots_share_source(slot_layout, other_layout)
                ),
                "",
            )
            slots.append(
                {
                    "slot": slot_name,
                    "status": "source_alias" if alias else "missing",
                    "alias_of": alias,
                }
            )
            continue
        source = np.asarray(geometry.raw_vertex_streams[slot_name], dtype=np.uint8)
        expected = source[loop_vertex_indices]
        output_path = output_dir / str(written[slot_name]["file_name"])
        actual = np.frombuffer(output_path.read_bytes(), dtype=np.uint8).reshape(expected.shape)
        byte_diff = actual != expected
        field_reports = []
        for physical in slot_layout.physical_fields:
            field_diff = byte_diff[:, physical.offset:physical.end]
            field_report = {
                "offset": int(physical.offset),
                "size": int(physical.size),
                "semantics": [alias.semantic for alias in physical.aliases],
                "byte_mismatch_count": int(np.count_nonzero(field_diff)),
                "vertex_mismatch_count": int(np.count_nonzero(np.any(field_diff, axis=1))),
            }
            if (
                physical.size == 4
                and any(alias.semantic == "NORMAL0" and alias.format == "R32_FLOAT" for alias in physical.aliases)
            ):
                source_packed = np.ascontiguousarray(
                    expected[:, physical.offset:physical.end]
                ).view("<u4").reshape(-1)
                actual_packed = np.ascontiguousarray(
                    actual[:, physical.offset:physical.end]
                ).view("<u4").reshape(-1)
                field_report["signed10_max_delta"] = [
                    int(np.max(np.abs(_signed10(source_packed >> shift) - _signed10(actual_packed >> shift))))
                    for shift in (0, 10, 20)
                ]
            field_reports.append(field_report)
        byte_mismatch_count = int(np.count_nonzero(byte_diff))
        vertex_mismatch_count = int(np.count_nonzero(np.any(byte_diff, axis=1)))
        skin_mismatch = _skin_semantic_mismatch_mask(
            expected,
            actual,
            slot_layout,
            palette_values,
            layout_codec,
        )
        exact_diff = byte_diff.copy()
        if skin_mismatch is not None:
            for physical in slot_layout.physical_fields:
                if physical.aliases and all(
                    alias.semantic_name in {"BLENDWEIGHTS", "BLENDINDICES"}
                    for alias in physical.aliases
                ):
                    exact_diff[:, physical.offset:physical.end] = False
        exact_mismatch = np.any(exact_diff, axis=1)
        semantic_mismatch = (
            exact_mismatch
            if skin_mismatch is None
            else exact_mismatch | skin_mismatch
        )
        semantic_mismatch_count = int(np.count_nonzero(semantic_mismatch))
        slots.append(
            {
                "slot": slot_name,
                "status": "written",
                "stride": int(slot_layout.stride),
                "record_count": int(len(expected)),
                "byte_mismatch_count": byte_mismatch_count,
                "vertex_mismatch_count": vertex_mismatch_count,
                "semantic_mismatch_count": semantic_mismatch_count,
                "palette_remap_only": bool(
                    byte_mismatch_count
                    and skin_mismatch is not None
                    and semantic_mismatch_count == 0
                ),
                "fields": field_reports,
            }
        )
    return slots


def _signed10(values):
    raw = np.asarray(values, dtype=np.uint32) & np.uint32(0x3FF)
    signed = raw.astype(np.int32)
    return np.where(signed >= 512, signed - 1024, signed)


def _skin_semantic_mismatch_mask(source, output, slot_layout, palette_values, layout_codec):
    weights_field = next(
        (
            field
            for field in slot_layout.fields
            if field.semantic_name == "BLENDWEIGHTS" and field.semantic_index == 0
        ),
        None,
    )
    indices_field = next(
        (
            field
            for field in slot_layout.fields
            if field.semantic_name == "BLENDINDICES" and field.semantic_index == 0
        ),
        None,
    )
    if weights_field is None or indices_field is None:
        return None
    source_weights = _decode_field(source, weights_field, layout_codec).astype(np.float64)
    output_weights = _decode_field(output, weights_field, layout_codec).astype(np.float64)
    source_groups = _decode_field(source, indices_field, layout_codec).astype(np.int64)
    output_local = _decode_field(output, indices_field, layout_codec).astype(np.int64)
    palette = np.asarray(palette_values, dtype=np.int64)
    output_groups = np.full(output_local.shape, -1, dtype=np.int64)
    valid = (output_local >= 0) & (output_local < len(palette))
    output_groups[valid] = palette[output_local[valid]]
    equivalent = np.all((output_weights <= 1.0e-8) | valid, axis=1)
    probe_groups = np.concatenate((source_groups, output_groups), axis=1)
    tolerance = 1.5 / 65535.0 if "UNORM" in weights_field.format else 1.0e-6
    for column in range(probe_groups.shape[1]):
        group = probe_groups[:, column:column + 1]
        source_sum = np.sum(np.where(source_groups == group, source_weights, 0.0), axis=1)
        output_sum = np.sum(np.where(output_groups == group, output_weights, 0.0), axis=1)
        equivalent &= np.isclose(source_sum, output_sum, rtol=1.0e-6, atol=tolerance)
    return ~equivalent


def _decode_field(records, field, layout_codec):
    spec = layout_codec.dxgi_format_spec(field.format)
    data = np.ascontiguousarray(records[:, field.offset:field.end])
    values = data.view(np.dtype(spec.dtype)).reshape(len(data), spec.component_count)
    if spec.conversion == "unorm16":
        return values.astype(np.float64) / 65535.0
    if spec.conversion == "snorm16":
        return np.maximum(values.astype(np.float64) / 32767.0, -1.0)
    if spec.conversion == "unorm8":
        return values.astype(np.float64) / 255.0
    if spec.conversion == "snorm8":
        return np.maximum(values.astype(np.float64) / 127.0, -1.0)
    return values


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frameanalysis", required=True)
    parser.add_argument("--only-mismatches", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
