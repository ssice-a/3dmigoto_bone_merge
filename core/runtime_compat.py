"""Validate EFMI runtime features required by generated BoneStore INI files."""

from __future__ import annotations

from pathlib import Path


class RuntimeCompatibilityError(RuntimeError):
    """Raised when a nearby EFMI runtime cannot parse the generated INI."""


_REQUIRED_UTF16_TOKENS = {
    "HashRegion": "->hashregion",
    "Resource Pool": "pool_size",
    "FIFO Pool": "pool_index_type",
}


def find_runtime_dll(output_directory: str | Path) -> Path | None:
    """Find the nearest d3d11.dll above an export directory."""

    output_path = Path(output_directory).expanduser().resolve()
    for directory in (output_path, *output_path.parents):
        candidate = directory / "d3d11.dll"
        if candidate.is_file():
            return candidate
    return None


def missing_runtime_features(dll_path: str | Path) -> list[str]:
    """Return required INI features absent from a compiled EFMI DLL."""

    payload = Path(dll_path).read_bytes().lower()
    return [
        feature
        for feature, token in _REQUIRED_UTF16_TOKENS.items()
        if token.encode("utf-16-le") not in payload
    ]


def assert_bone_merge_runtime_compatible(output_directory: str | Path) -> Path | None:
    """Reject an installed EFMI runtime that cannot parse schema v3 output."""

    dll_path = find_runtime_dll(output_directory)
    if dll_path is None:
        return None

    missing = missing_runtime_features(dll_path)
    if missing:
        names = ", ".join(missing)
        raise RuntimeCompatibilityError(
            f"EFMI runtime {dll_path} does not support required Bone Merge INI features: {names}. "
            "Install the official XXMI-Libs v0.9.4 package or newer."
        )
    return dll_path
