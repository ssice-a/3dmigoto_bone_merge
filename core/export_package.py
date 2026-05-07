"""Region/part export planning for BoneMerge buffers.

This module is intentionally Blender-light: it only expects collection-like
objects with ``name``, ``objects`` and ``children`` attributes. The actual
evaluated-mesh VB writer lives above this layer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..constants import BI4_MAX_BONE_COUNT
from .io import write_uint32_buffer


_REGION_COLLECTION_RE = re.compile(
    r"(?P<hash>[0-9A-Fa-f]{8})[-_](?P<count>\d+)[-_](?P<first>\d+)"
)
_PART_COLLECTION_RE = re.compile(r"^part(?P<index>\d+)(?:\D.*)?$", re.IGNORECASE)


class ExportPlanError(ValueError):
    """Raised when the export collection tree cannot be represented safely."""


@dataclass(frozen=True)
class ExportRegionIdentity:
    ib_hash: str
    match_index_count: int
    match_first_index: int
    collection_name: str = ""

    @property
    def key(self) -> str:
        return f"{self.ib_hash}-{self.match_index_count}-{self.match_first_index}"


@dataclass(frozen=True)
class ExportObjectUsage:
    name: str
    object_ref: object
    used_global_groups: tuple[int, ...]


@dataclass(frozen=True)
class ExportPartPlan:
    region: ExportRegionIdentity
    part_index: int
    source_part_name: str
    mesh_objects: tuple[object, ...]
    object_usages: tuple[ExportObjectUsage, ...]
    palette_values: tuple[int, ...]
    generated: bool = False
    split_reason: str = ""

    @property
    def part_name(self) -> str:
        return f"part{self.part_index:02d}"

    @property
    def file_stem(self) -> str:
        return f"{self.region.key}_{self.part_name}"

    @property
    def resource_suffix(self) -> str:
        return f"{self.region.ib_hash}_{self.region.match_index_count}_{self.region.match_first_index}_{self.part_name}"

    @property
    def palette_file_name(self) -> str:
        return f"{self.file_stem}-PartLocalToGlobalBoneMap.buf"


@dataclass(frozen=True)
class ExportPlan:
    root_collection_name: str
    parts: tuple[ExportPartPlan, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def object_count(self) -> int:
        return len({usage.name for part in self.parts for usage in part.object_usages})

    @property
    def palette_count(self) -> int:
        return len(self.parts)


@dataclass(frozen=True)
class _PartSource:
    region: ExportRegionIdentity
    part_index: int
    source_part_name: str
    mesh_objects: tuple[object, ...]
    generated: bool = False
    split_reason: str = ""


def parse_region_collection_identity(collection_name: str) -> ExportRegionIdentity | None:
    """Parse ``<ib_hash>-<match_index_count>-<first_index>`` from a collection name."""

    match = _REGION_COLLECTION_RE.search(str(collection_name or ""))
    if not match:
        return None
    return ExportRegionIdentity(
        ib_hash=match.group("hash").lower(),
        match_index_count=int(match.group("count")),
        match_first_index=int(match.group("first")),
        collection_name=str(collection_name or ""),
    )


def parse_part_collection_index(collection_name: str) -> int | None:
    match = _PART_COLLECTION_RE.match(str(collection_name or "").strip())
    if not match:
        return None
    return int(match.group("index"))


def build_export_plan(
    root_collection,
    collect_used_groups: Callable[[object], Iterable[int]],
    *,
    max_bones_per_part: int = BI4_MAX_BONE_COUNT,
) -> ExportPlan:
    """Build a region/part export plan from the root export collection."""

    if root_collection is None:
        raise ExportPlanError("Export root collection is not set")
    if max_bones_per_part <= 0:
        raise ExportPlanError("max_bones_per_part must be positive")

    root_name = str(getattr(root_collection, "name", "") or "")
    warnings: list[str] = []
    part_sources: list[_PartSource] = []

    for region_collection in getattr(root_collection, "children", []) or []:
        identity = parse_region_collection_identity(getattr(region_collection, "name", ""))
        if identity is None:
            continue
        part_sources.extend(_collect_region_part_sources(region_collection, identity, warnings))

    if not part_sources:
        direct_meshes = [
            str(getattr(obj, "name", "") or "")
            for obj in getattr(root_collection, "objects", []) or []
            if _is_mesh_object(obj)
        ]
        hint = ""
        if direct_meshes:
            hint = " Direct mesh object(s) must be moved into an IB region collection: " + ", ".join(direct_meshes[:5])
        raise ExportPlanError(
            f"{root_name or 'ExportRoot'}: no IB region collections found. "
            "Create child collections named like 640d1c0e-46845-0."
            + hint
        )

    planned_parts: list[ExportPartPlan] = []
    used_part_indices_by_region: dict[str, set[int]] = {}
    for source in part_sources:
        used_part_indices_by_region.setdefault(source.region.key, set()).add(source.part_index)

    for source in part_sources:
        object_usages = _build_object_usages(source, collect_used_groups)
        source_palette = _union_usage_groups(object_usages)
        if len(source_palette) <= max_bones_per_part:
            planned_parts.append(_part_plan_from_source(source, object_usages, source.part_index))
            continue

        split_sources = _split_source_by_object(
            source,
            object_usages,
            used_part_indices_by_region.setdefault(source.region.key, set()),
            max_bones_per_part=max_bones_per_part,
        )
        planned_parts.extend(split_sources)

    _validate_single_part_membership(planned_parts)
    sorted_parts = tuple(
        sorted(
            planned_parts,
            key=lambda part: (
                part.region.ib_hash,
                part.region.match_index_count,
                part.region.match_first_index,
                part.part_index,
            ),
        )
    )
    return ExportPlan(root_collection_name=root_name, parts=sorted_parts, warnings=tuple(warnings))


def write_part_palette_files(buffer_dir: str, plan: ExportPlan) -> list[dict]:
    """Write all part palette files and return manifest-ready palette records."""

    os.makedirs(buffer_dir, exist_ok=True)
    records: list[dict] = []
    for part in plan.parts:
        file_path = os.path.join(buffer_dir, part.palette_file_name)
        write_uint32_buffer(file_path, part.palette_values)
        records.append(part_palette_manifest_record(part, file_path))
    return records


def part_palette_manifest_record(part: ExportPartPlan, file_path: str = "") -> dict:
    return {
        "region_collection": part.region.collection_name,
        "part_name": part.part_name,
        "source_part_name": part.source_part_name,
        "ib_hash": part.region.ib_hash,
        "match_first_index": int(part.region.match_first_index),
        "match_index_count": int(part.region.match_index_count),
        "part_index": int(part.part_index),
        "chunk_index": int(part.part_index),
        "local_bone_count": len(part.palette_values),
        "palette_values": list(part.palette_values),
        "file_name": part.palette_file_name,
        "file_path": file_path,
        "resource_suffix": part.resource_suffix,
        "generated": bool(part.generated),
        "split_reason": part.split_reason,
    }


def write_r32_index_buffer(path: str, indices: Iterable[int]) -> str:
    """Write an index buffer as little-endian DXGI_FORMAT_R32_UINT."""

    values = []
    for index in indices:
        value = int(index)
        if value < 0:
            raise ValueError(f"R32 index cannot be negative: {value}")
        values.append(value)
    return write_uint32_buffer(path, values)


def _collect_region_part_sources(region_collection, identity: ExportRegionIdentity, warnings: list[str]) -> list[_PartSource]:
    direct_meshes = tuple(obj for obj in getattr(region_collection, "objects", []) or [] if _is_mesh_object(obj))
    explicit_part_collections: dict[int, object] = {}
    duplicate_part_indices: set[int] = set()

    for child in getattr(region_collection, "children", []) or []:
        part_index = parse_part_collection_index(getattr(child, "name", ""))
        if part_index is None:
            continue
        if part_index in explicit_part_collections:
            duplicate_part_indices.add(part_index)
        explicit_part_collections[part_index] = child

    if duplicate_part_indices:
        duplicate = min(duplicate_part_indices)
        raise ExportPlanError(f"{identity.collection_name}: duplicate part{duplicate:02d} collection")

    sources: list[_PartSource] = []
    if explicit_part_collections:
        for part_index, part_collection in sorted(explicit_part_collections.items()):
            mesh_objects = tuple(_iter_mesh_objects_recursive(part_collection))
            if not mesh_objects:
                continue
            sources.append(
                _PartSource(
                    region=identity,
                    part_index=part_index,
                    source_part_name=str(getattr(part_collection, "name", "") or f"part{part_index:02d}"),
                    mesh_objects=mesh_objects,
                )
            )
        if direct_meshes:
            warnings.append(
                f"{identity.collection_name}: direct mesh object(s) were planned as implicit part00; "
                "Prepare Export may migrate them into part00."
            )
            existing = next((source for source in sources if source.part_index == 0), None)
            if existing is None:
                sources.append(
                    _PartSource(
                        region=identity,
                        part_index=0,
                        source_part_name="part00",
                        mesh_objects=direct_meshes,
                    )
                )
            else:
                sources.remove(existing)
                sources.append(
                    _PartSource(
                        region=existing.region,
                        part_index=existing.part_index,
                        source_part_name=existing.source_part_name,
                        mesh_objects=tuple(existing.mesh_objects) + direct_meshes,
                    )
                )
        return sorted(sources, key=lambda source: source.part_index)

    mesh_objects = tuple(_iter_mesh_objects_recursive(region_collection))
    if not mesh_objects:
        return []
    return [
        _PartSource(
            region=identity,
            part_index=0,
            source_part_name="part00",
            mesh_objects=mesh_objects,
        )
    ]


def _iter_mesh_objects_recursive(collection):
    seen_names: set[str] = set()

    def walk(current_collection):
        for obj in getattr(current_collection, "objects", []) or []:
            if not _is_mesh_object(obj):
                continue
            name = str(getattr(obj, "name", "") or "")
            if name in seen_names:
                continue
            seen_names.add(name)
            yield obj
        for child in getattr(current_collection, "children", []) or []:
            yield from walk(child)

    yield from walk(collection)


def _build_object_usages(source: _PartSource, collect_used_groups) -> tuple[ExportObjectUsage, ...]:
    usages: list[ExportObjectUsage] = []
    for mesh_obj in source.mesh_objects:
        used_groups = tuple(sorted({int(value) for value in collect_used_groups(mesh_obj) if int(value) >= 0}))
        if not used_groups:
            raise ExportPlanError(f"{source.region.collection_name}/{source.source_part_name}/{mesh_obj.name}: no weighted numeric vertex groups found")
        usages.append(
            ExportObjectUsage(
                name=str(getattr(mesh_obj, "name", "") or ""),
                object_ref=mesh_obj,
                used_global_groups=used_groups,
            )
        )
    if not usages:
        raise ExportPlanError(f"{source.region.collection_name}/{source.source_part_name}: no mesh objects found")
    return tuple(usages)


def _split_source_by_object(
    source: _PartSource,
    object_usages: tuple[ExportObjectUsage, ...],
    used_part_indices: set[int],
    *,
    max_bones_per_part: int,
) -> list[ExportPartPlan]:
    bins: list[list[ExportObjectUsage]] = []
    bin_groups: list[set[int]] = []

    for usage in object_usages:
        usage_groups = set(usage.used_global_groups)
        if len(usage_groups) > max_bones_per_part:
            raise ExportPlanError(
                f"{source.region.collection_name}/{source.source_part_name}/{usage.name}: "
                f"uses {len(usage_groups)} global bones; triangle-level splitting is required before export"
            )

        placed = False
        for bin_index, groups in enumerate(bin_groups):
            if len(groups.union(usage_groups)) <= max_bones_per_part:
                bins[bin_index].append(usage)
                groups.update(usage_groups)
                placed = True
                break
        if not placed:
            bins.append([usage])
            bin_groups.append(set(usage_groups))

    if len(bins) <= 1:
        return [_part_plan_from_source(source, object_usages, source.part_index)]

    split_parts: list[ExportPartPlan] = []
    first_part_index = source.part_index
    for bin_index, usages in enumerate(bins):
        if bin_index == 0:
            part_index = first_part_index
        else:
            part_index = _next_free_part_index(used_part_indices)
        used_part_indices.add(part_index)
        split_source = _PartSource(
            region=source.region,
            part_index=part_index,
            source_part_name=f"{source.source_part_name}_auto{bin_index:02d}",
            mesh_objects=tuple(usage.object_ref for usage in usages),
            generated=True,
            split_reason=f"object_split_over_{max_bones_per_part}_bones",
        )
        split_parts.append(_part_plan_from_source(split_source, tuple(usages), part_index))
    return split_parts


def _part_plan_from_source(source: _PartSource, object_usages: tuple[ExportObjectUsage, ...], part_index: int) -> ExportPartPlan:
    palette_values = _union_usage_groups(object_usages)
    return ExportPartPlan(
        region=source.region,
        part_index=part_index,
        source_part_name=source.source_part_name,
        mesh_objects=tuple(usage.object_ref for usage in object_usages),
        object_usages=object_usages,
        palette_values=palette_values,
        generated=source.generated,
        split_reason=source.split_reason,
    )


def _union_usage_groups(object_usages: Iterable[ExportObjectUsage]) -> tuple[int, ...]:
    groups: set[int] = set()
    for usage in object_usages:
        groups.update(usage.used_global_groups)
    return tuple(sorted(groups))


def _next_free_part_index(used_part_indices: set[int]) -> int:
    part_index = 0
    while part_index in used_part_indices:
        part_index += 1
    return part_index


def _validate_single_part_membership(parts: Iterable[ExportPartPlan]) -> None:
    memberships: dict[str, list[str]] = {}
    for part in parts:
        owner = f"{part.region.key}/{part.part_name}"
        for mesh_obj in part.mesh_objects:
            memberships.setdefault(str(getattr(mesh_obj, "name", "") or ""), []).append(owner)
    duplicated = {name: owners for name, owners in memberships.items() if len(owners) > 1}
    if duplicated:
        name, owners = next(iter(duplicated.items()))
        raise ExportPlanError(f"{name}: the same object is present in multiple export parts: {', '.join(owners)}")


def _is_mesh_object(obj) -> bool:
    return str(getattr(obj, "type", "") or "") == "MESH"
