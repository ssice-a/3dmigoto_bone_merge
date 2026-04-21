"""Export local palette buffers for final merged/chunk draws."""

from __future__ import annotations

import os

from ..constants import BUFFER_EXPORT_DIR_NAME
from .io import ensure_directory, write_uint32_buffer
from .models import LocalPaletteRecord, ObjectRemap


def export_local_palettes(output_dir: str, object_remaps: list[ObjectRemap]) -> list[LocalPaletteRecord]:
    buffer_dir = ensure_directory(os.path.join(output_dir, BUFFER_EXPORT_DIR_NAME))
    palette_records: list[LocalPaletteRecord] = []

    for remap in object_remaps:
        palette_values = _build_palette_values(remap)
        chunk_index = 0
        file_name = f"{remap.ib_hash.lower()}-{remap.match_index_count}-{chunk_index}-Palette.buf"
        file_path = os.path.join(buffer_dir, file_name)
        write_uint32_buffer(file_path, palette_values)

        palette_records.append(
            LocalPaletteRecord(
                object_name=remap.object_name,
                ib_hash=remap.ib_hash.lower(),
                match_index_count=remap.match_index_count,
                chunk_index=chunk_index,
                local_bone_count=len(palette_values),
                palette_values=tuple(palette_values),
                file_name=file_name,
                file_path=file_path,
                resource_suffix=f"{remap.ib_hash.lower()}_{remap.match_index_count}_{chunk_index}",
            )
        )

    return palette_records


def _build_palette_values(remap: ObjectRemap) -> list[int]:
    local_to_global = {
        int(local_index): int(global_index)
        for local_index, global_index in remap.local_group_to_global_group.items()
    }
    if not local_to_global:
        raise ValueError(f"{remap.object_name}: no local->global remap data available for palette export")

    max_local_index = max(local_to_global.keys())
    palette_values: list[int] = []
    for local_index in range(max_local_index + 1):
        if local_index not in local_to_global:
            raise ValueError(
                f"{remap.object_name}: local palette is sparse, missing local bone {local_index}; "
                "current export expects dense local indices"
            )
        palette_values.append(local_to_global[local_index])
    return palette_values
