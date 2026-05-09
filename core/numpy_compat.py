"""NumPy access for Blender fast paths."""

from __future__ import annotations

import numpy as np

def optional_numpy():
    """Return numpy."""

    return np


def numpy_status() -> str:
    return f"enabled {np.__version__}"
