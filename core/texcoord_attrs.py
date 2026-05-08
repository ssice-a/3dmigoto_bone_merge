"""Shared helpers for raw TEXCOORD attribute storage.

The game can bind packed auxiliary data as TEXCOORD semantics.  Those values
are not Blender UV layers, so we keep exact point attributes and optionally
mirror byte-packed TEXCOORD4 data into a color attribute for editing.
"""

from __future__ import annotations


def texcoord_component_attr_names(slot_name: str, semantic_index: int, component: int) -> tuple[str, ...]:
    slot_index = _slot_index(slot_name)
    names = [f"bmc_vb{slot_index}_texcoord{int(semantic_index)}_{int(component)}"]
    if int(semantic_index) == 4:
        names.append(f"bmc_texcoord4_raw_{int(component)}")
    return tuple(names)


def texcoord_color_attr_names(slot_name: str, semantic_index: int) -> tuple[str, ...]:
    slot_index = _slot_index(slot_name)
    semantic_index = int(semantic_index)
    names = [f"bmc_vb{slot_index}_texcoord{semantic_index}_color"]
    if semantic_index == 4:
        names.append("bmc_texcoord4_color")
    return tuple(names)


def snorm_byte_to_color_component(value: int) -> float:
    """Map a signed byte value to an exact editable 0..1 color component."""

    return float(int(value) & 0xFF) / 255.0


def color_component_to_raw_byte(value: float) -> int:
    """Map an editable color component back to the original byte payload."""

    clamped = max(0.0, min(1.0, float(value)))
    return int(round(clamped * 255.0)) & 0xFF


def _slot_index(slot_name: str) -> int:
    normalized = str(slot_name or "").lower()
    if normalized.startswith("vb") and normalized[2:].isdigit():
        return int(normalized[2:])
    try:
        return int(normalized)
    except ValueError:
        return -1
