"""Export bundled HLSL assets used by BoneStore capture."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..constants import HLSL_EXPORT_DIR_NAME


_REQUIRED_HLSL_FILES = (
    "extract_cb1_vs.hlsl",
    "extract_cb1_ps.hlsl",
    "gather_bones_cs.hlsl",
    "record_bones_dynamic_cs.hlsl",
    "redirect_cb1_cs.hlsl",
)


def export_required_hlsl(output_directory: str) -> str:
    assets_dir = Path(__file__).resolve().parent.parent / "assets" / "hlsl"
    if not assets_dir.exists():
        raise ValueError(f"Bundled HLSL assets directory not found: {assets_dir}")

    hlsl_output_dir = Path(output_directory).resolve() / HLSL_EXPORT_DIR_NAME
    hlsl_output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in _REQUIRED_HLSL_FILES:
        source_path = assets_dir / file_name
        if not source_path.exists():
            raise ValueError(f"Missing bundled HLSL asset: {source_path}")
        shutil.copy2(source_path, hlsl_output_dir / file_name)

    return str(hlsl_output_dir)
