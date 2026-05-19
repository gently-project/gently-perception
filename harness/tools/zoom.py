"""zoom — crop a region of one projection and re-render at full resolution."""
from __future__ import annotations

from typing import Annotated, Literal

import numpy as np

from harness.core.render import array_to_b64, project_axis
from harness.core.types import ImageResult
from harness.tools import Range, tool


@tool(returns="image")
def zoom(
    volume: np.ndarray,
    *,
    x: Annotated[int, Range(0, "X")],
    y: Annotated[int, Range(0, "Y")],
    w: Annotated[int, Range(1, "X")],
    h: Annotated[int, Range(1, "Y")],
    view: Literal["xy", "yz", "xz"] = "xy",
) -> ImageResult:
    """Crop a region of one max-intensity projection and re-render at full resolution.

    Use to inspect fold boundaries, eggshell edges, or ambiguous regions
    closely. Coordinates are in pixels on the chosen projection plane.
    """
    proj = project_axis(volume, view)
    H, W = proj.shape
    x1, y1 = min(x + w, W), min(y + h, H)
    crop = proj[y:y1, x:x1]
    if crop.size == 0:
        crop = proj  # graceful: return whole view rather than fail
    return ImageResult(b64=array_to_b64(crop, target_long_edge=512), caption=f"zoom {view} [{x}:{x1},{y}:{y1}]")
