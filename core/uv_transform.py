"""Shared UV coordinate transforms between game buffers and Blender."""

from __future__ import annotations


DEFAULT_UV_FLIP_V = True


def flip_uv_v(uv: tuple[float, float]) -> tuple[float, float]:
    return (float(uv[0]), 1.0 - float(uv[1]))


def game_uv_to_blender(uv: tuple[float, float], *, flip_v: bool = DEFAULT_UV_FLIP_V) -> tuple[float, float]:
    if not flip_v:
        return (float(uv[0]), float(uv[1]))
    return flip_uv_v(uv)


def blender_uv_to_game(uv: tuple[float, float], *, flip_v: bool = DEFAULT_UV_FLIP_V) -> tuple[float, float]:
    if not flip_v:
        return (float(uv[0]), float(uv[1]))
    return flip_uv_v(uv)
