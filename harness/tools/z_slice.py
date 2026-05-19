"""z_slice / z_sweep — inspect individual depth planes instead of max-projection."""
from __future__ import annotations

from typing import Annotated

import numpy as np

from harness.core.render import array_to_b64, normalize
from harness.core.types import ImageResult
from harness.tools import Range, tool


@tool(returns="image")
def z_slice(volume: np.ndarray, *, index: Annotated[int, Range(0, "Z")]) -> ImageResult:
    """Return a single Z plane (no max-projection).

    Use when the max-projection collapses overlapping folds — individual
    slices can make fold count clearer.
    """
    plane = volume[index]
    return ImageResult(b64=array_to_b64(plane, target_long_edge=512), caption=f"z_slice {index}/{volume.shape[0]}")


@tool(returns="image")
def z_sweep(
    volume: np.ndarray,
    *,
    start: Annotated[int, Range(0, "Z")],
    end: Annotated[int, Range(0, "Z")],
    step: Annotated[int, Range(1, 20)] = 5,
) -> ImageResult:
    """Return a horizontal filmstrip of Z planes from `start` to `end` (inclusive).

    Use to count folds in depth — sweep through the middle of the volume.
    """
    lo, hi = (start, end) if start <= end else (end, start)
    indices = list(range(lo, min(hi + 1, volume.shape[0]), max(step, 1)))
    if not indices:
        indices = [lo]
    planes = [normalize(volume[i]) for i in indices]
    H = max(p.shape[0] for p in planes)
    W = sum(p.shape[1] for p in planes) + 2 * (len(planes) - 1)
    strip = np.zeros((H, W), dtype=np.uint8)
    cx = 0
    for p in planes:
        strip[: p.shape[0], cx : cx + p.shape[1]] = p
        cx += p.shape[1] + 2
    return ImageResult(
        b64=array_to_b64(strip, target_long_edge=min(1024, W)),
        caption=f"z_sweep {indices}",
    )
