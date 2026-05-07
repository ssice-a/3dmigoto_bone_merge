"""Identity parsing helpers shared by Blender operators."""

from __future__ import annotations

import re

_OBJECT_HASH_RE = re.compile(r"(?<![0-9A-Fa-f])(?P<hash>[0-9A-Fa-f]{8})(?![0-9A-Fa-f])")


def infer_ib_hash_from_name(object_name: str) -> str | None:
    match = _OBJECT_HASH_RE.search(str(object_name or ""))
    if not match:
        return None
    return match.group("hash").lower()


def infer_mesh_identity_from_name(object_name: str) -> tuple[str, int] | None:
    ib_hash = infer_ib_hash_from_name(object_name)
    if ib_hash is None:
        return None
    return ib_hash, 0
