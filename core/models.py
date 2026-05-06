"""Dataclasses shared across the Bone Merge Capture plugin."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetObjectSpec:
    object_name: str
    ib_hash: str
    match_index_count: int
    local_bone_count: int


@dataclass(frozen=True)
class LoggedDraw:
    draw_index: int
    vs_hash: str
    match_index_count: int
    vs_cb1_first_constant: int
    vs_cb1_num_constants: int


@dataclass(frozen=True)
class DrawRecord:
    draw_index: int
    object_name: str
    vs_hash: str
    match_index_count: int
    vs_cb1_first_constant: int
    vs_cb1_num_constants: int
    ib_hash: str
    local_bone_count: int
    vb2_path: str
    vs_t0_path: str
    vs_cb1_path: str


@dataclass(frozen=True)
class PartRecord:
    draw_index: int
    object_name: str
    vs_hash: str
    ib_hash: str
    match_index_count: int
    bone_count: int
    global_bone_base: int
    vb2_path: str
    vs_t0_path: str
    vs_cb1_path: str
    vs_cb1_first_constant: int
    vs_cb1_num_constants: int


@dataclass(frozen=True)
class LodPartRecord:
    draw_index: int
    object_name: str
    vs_hash: str
    ib_hash: str
    match_index_count: int
    bone_count: int
    lod_global_bone_base: int
    vb2_path: str
    vs_t0_path: str
    vs_cb1_path: str
    vs_cb1_first_constant: int
    vs_cb1_num_constants: int


@dataclass(frozen=True)
class BoneAlias:
    src_draw_index: int
    src_object_name: str
    src_ib_hash: str
    src_local_bone: int
    src_global_bone: int
    canonical_draw_index: int
    canonical_object_name: str
    canonical_ib_hash: str
    canonical_local_bone: int
    canonical_global_bone: int
    confidence: str


@dataclass(frozen=True)
class ObjectRemap:
    object_name: str
    ib_hash: str
    match_index_count: int
    local_group_to_global_group: dict[str, int]


@dataclass(frozen=True)
class LocalPaletteRecord:
    object_name: str
    ib_hash: str
    match_index_count: int
    chunk_index: int
    local_bone_count: int
    palette_values: tuple[int, ...]
    file_name: str
    file_path: str
    resource_suffix: str
    variant_id: str = ""


@dataclass(frozen=True)
class LodMappingRecord:
    canonical_global_bone: int
    mapped_lod_global_bone: int
    status: str
    score: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class LodRuntimePartRecord:
    variant_id: str
    draw_index: int
    vs_hash: str
    ib_hash: str
    match_index_count: int
    bone_count: int
    capture_store_base: int
    resource_suffix: str


@dataclass(frozen=True)
class ScanGenerateResult:
    frameanalysis_dir: str
    manifest_path: str
    ini_path: str
    scanned_parts: int
    total_global_bones: int
    shadow_host_hash: str = ""
    shadow_host_match_index_count: int = -1
    shadow_host_vs_hash: str = ""
    shadow_host_draw_index: int = -1
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RemapApplyResult:
    target_objects: int
    updated_objects: int
    renamed_groups: int
    skipped_objects: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DuplicateMergeResult:
    target_objects: int
    updated_objects: int
    merged_aliases: int
    skipped_objects: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ShadowHostRecord:
    draw_index: int
    ib_hash: str
    match_index_count: int
    vs_hash: str = ""


@dataclass(frozen=True)
class CapturePlan:
    part_records: tuple[PartRecord, ...]
    shadow_host: ShadowHostRecord | None
    warnings: tuple[str, ...] = field(default_factory=tuple)

