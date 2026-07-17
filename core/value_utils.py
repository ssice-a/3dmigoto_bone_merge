"""Small coercion helpers for manifest values."""

from __future__ import annotations


def int_or_default(value, default: int = 0) -> int:
    """Convert a manifest value to int without treating zero as missing."""

    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
