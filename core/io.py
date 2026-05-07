"""JSON helpers."""

from __future__ import annotations

import sys
from array import array
import json
import os


def ensure_directory(path: str) -> str:
    normalized_path = os.path.abspath(path)
    os.makedirs(normalized_path, exist_ok=True)
    return normalized_path


def write_json(path: str, payload) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")
    return path


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def write_uint32_buffer(path: str, values) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    uint_values = [int(value) for value in values]
    for value in uint_values:
        if value < 0 or value > 0xFFFFFFFF:
            raise ValueError(f"uint32 value out of range: {value}")
    packed = array("I", uint_values)
    if sys.byteorder != "little":
        packed.byteswap()
    with open(path, "wb") as file_handle:
        file_handle.write(packed.tobytes())
    return path
