"""Dataclasses shared across the Bone Merge Capture plugin."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    match_first_index: int = 0
    object_usages: tuple[dict, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RemapApplyResult:
    target_objects: int
    updated_objects: int
    renamed_groups: int
    skipped_objects: tuple[str, ...] = field(default_factory=tuple)
