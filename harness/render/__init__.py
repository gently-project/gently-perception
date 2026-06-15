"""
Per-task volume → image rendering.

A RenderSpec.name resolves to a callable here. The loop renders once per
(frame, spec.name) and shares the result across tasks that ask for the same
spec. All renderers go through ``core.render.cached_render``-style disk
caching so re-runs don't re-render.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from harness.core import render as _r

Renderer = Callable[[Path, Mapping[str, Any]], str]
RENDERERS: dict[str, Renderer] = {}


def renderer(name: str) -> Callable[[Renderer], Renderer]:
    def _reg(fn: Renderer) -> Renderer:
        RENDERERS[name] = fn
        return fn

    return _reg


def render(spec_name: str, volume_path: Path, params: Mapping[str, Any] | None = None) -> str:
    try:
        fn = RENDERERS[spec_name]
    except KeyError as e:
        raise KeyError(f"unknown render spec {spec_name!r}; known: {sorted(RENDERERS)}") from e
    return fn(volume_path, params or {})


# --- built-in renderers -----------------------------------------------------


@renderer("three_view_mip")
def _three_view_mip(path: Path, params: Mapping[str, Any]) -> str:
    """The existing stage-classifier render: cropped 3-view MIP, percentile-normalized."""
    return _r.cached_render(path)


@renderer("single_view_fixed")
def _single_view_fixed(path: Path, params: Mapping[str, Any]) -> str:
    """Single XY max-projection with a FIXED intensity range.

    Onset detection needs absolute brightness to be comparable across
    timepoints, so this renderer does NOT percentile-stretch per image.
    The (lo, hi) range is a dataset constant passed via params; defaults are
    reasonable for 16-bit light-sheet data after dark subtraction.
    """
    lo = float(params.get("lo", 100.0))
    hi = float(params.get("hi", 4000.0))

    def _compute() -> str:
        vol = _r.load_volume(path)
        xy = vol.max(axis=0).astype(np.float32)
        xy = np.clip((xy - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        xy = (xy * 255).astype(np.uint8)
        return _r.array_to_b64(xy)

    return _r.cached_tool_result(path, "render.single_view_fixed", {"lo": lo, "hi": hi}, _compute)
