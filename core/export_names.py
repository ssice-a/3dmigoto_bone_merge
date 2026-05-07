"""Shared naming helpers for generated export files."""

from __future__ import annotations

import re

from ..constants import BONESTORE_INI_FILE_NAME


_WINDOWS_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def ini_filename_from_collection_name(collection_name: str) -> str:
    """Return the generated INI file name for an export root collection."""

    clean_name = _WINDOWS_INVALID_FILENAME_RE.sub("_", str(collection_name or "")).strip()
    clean_name = clean_name.rstrip(" .")
    if not clean_name:
        return BONESTORE_INI_FILE_NAME
    if clean_name.lower().endswith(".ini"):
        return clean_name
    return f"{clean_name}.ini"
