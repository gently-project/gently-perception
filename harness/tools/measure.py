"""measure — quantitative morphology features via classical CV.

These operationalize the visual heuristics from the `scientific` prompt
("eggshell fill fraction", "count of body segments") as actual measurements
the model can request.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image

from harness.core.render import project_axis
from harness.core.types import NumericResult
from harness.tools import tool


def _otsu_mask(proj: np.ndarray) -> np.ndarray:
    """Otsu-threshold a projection. Keeps all foreground components."""
    proj = proj.astype(np.float32)
    if proj.max() <= proj.min():
        return np.zeros_like(proj, dtype=bool)
    return proj > threshold_otsu(proj)


def _fill_fraction(volume: np.ndarray) -> tuple[float, str]:
    """area(full foreground) / area(convex hull ≈ eggshell) on the XY projection.

    A concave embryo (folds with gaps) → hull is larger than mask → fraction < 1.
    A blob filling its hull → fraction ≈ 1.
    """
    proj = project_axis(volume, "xy")
    mask = _otsu_mask(proj)
    if not mask.any():
        return 0.0, "no signal detected"
    hull = convex_hull_image(mask)
    fill = float(mask.sum()) / max(float(hull.sum()), 1.0)
    return fill, "foreground_area / convex_hull_area on XY max-projection"


def _n_segments(volume: np.ndarray) -> tuple[float, str]:
    """Count of connected bright components in the XY projection — fold-count proxy."""
    proj = project_axis(volume, "xy")
    mask = _otsu_mask(proj)
    if not mask.any():
        return 0.0, "no signal detected"
    _, n = ndimage.label(mask)  # type: ignore[misc]
    return float(n), "connected foreground components on XY max-projection"


def _aspect_ratio(volume: np.ndarray) -> tuple[float, str]:
    proj = project_axis(volume, "xy")
    mask = _otsu_mask(proj)
    if not mask.any():
        return 0.0, "no signal detected"
    ys, xs = np.where(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return float(w) / max(float(h), 1.0), "bbox width / height on XY max-projection"


_FEATURES = {
    "fill_fraction": (_fill_fraction, ""),
    "n_segments": (_n_segments, "count"),
    "aspect_ratio": (_aspect_ratio, ""),
}


@tool(returns="numeric")
def measure(
    volume: np.ndarray, *, feature: Literal["fill_fraction", "n_segments", "aspect_ratio"]
) -> NumericResult:
    """Compute a quantitative morphology feature from the volume.

    - fill_fraction: area(embryo mass) / area(eggshell hull) on XY projection.
      Roughly ~0.5→1.5fold, ~0.7→2fold, ~0.9→pretzel.
    - n_segments: count of bright bands crossing the minor axis (fold-count proxy).
    - aspect_ratio: bounding-box width/height of the bright mass.
    """
    fn, unit = _FEATURES[feature]
    value, note = fn(volume)
    return NumericResult(value=round(value, 3), unit=unit, note=note)
