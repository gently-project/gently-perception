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


def _convexity(volume: np.ndarray) -> tuple[float, str]:
    """area(foreground) / area(its own convex hull) on the XY projection.

    DECREASES with development: an unfolded blob fills its hull (≈0.8); a folded
    body has gaps between folds so the hull outgrows the mass (pretzel ≈0.57,
    hatched worm ≈0.32). Empirical per-stage medians are in the tool docstring.
    """
    proj = project_axis(volume, "xy")
    mask = _otsu_mask(proj)
    if not mask.any():
        return 0.0, "no signal detected"
    hull = convex_hull_image(mask)
    fill = float(mask.sum()) / max(float(hull.sum()), 1.0)
    return fill, "foreground_area / own_convex_hull_area on XY max-projection (lower = more folded)"


def _n_segments(volume: np.ndarray) -> tuple[float, str]:
    """Count of connected bright components in the XY projection — fold-count proxy.

    Components smaller than 0.5% of the foreground are discarded as noise.
    """
    proj = project_axis(volume, "xy")
    mask = _otsu_mask(proj)
    if not mask.any():
        return 0.0, "no signal detected"
    labeled, n = ndimage.label(mask)  # type: ignore[misc]
    if n == 0:
        return 0.0, "no signal detected"
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    min_size = max(20, int(mask.sum() * 0.005))
    n_real = int(np.sum(sizes >= min_size))
    return float(n_real), f"connected components ≥{min_size}px on XY max-projection"


def _aspect_ratio(volume: np.ndarray) -> tuple[float, str]:
    proj = project_axis(volume, "xy")
    mask = _otsu_mask(proj)
    if not mask.any():
        return 0.0, "no signal detected"
    ys, xs = np.where(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return float(w) / max(float(h), 1.0), "bbox width / height on XY max-projection"


_FEATURES = {
    "convexity": (_convexity, ""),
    "n_segments": (_n_segments, "count"),
    "aspect_ratio": (_aspect_ratio, ""),
}


@tool(returns="numeric")
def measure(
    volume: np.ndarray, *, feature: Literal["convexity", "n_segments", "aspect_ratio"]
) -> NumericResult:
    """Compute a quantitative morphology feature from the XY max-projection.

    - convexity: foreground area / its own convex-hull area. DECREASES as the
      body folds (gaps between folds make the hull outgrow the mass).
      Empirical medians by stage on this dataset: early/bean/comma ≈0.80–0.82,
      1.5fold ≈0.76, 2fold ≈0.65, pretzel ≈0.57, hatched ≈0.32. The 2fold↔pretzel
      ranges overlap (2fold 0.61–0.70, pretzel 0.51–0.62) — treat as one signal
      among several, not a decision rule.
    - n_segments: count of separated bright components. early≈1, bean/comma≈2,
      1.5fold≈3, 2fold and pretzel both ≈5 (does NOT separate those two),
      hatched≈10.
    - aspect_ratio: bounding-box width/height. Not discriminative on this data
      (≈2.0 at every stage) — rarely worth a call.
    """
    fn, unit = _FEATURES[feature]
    value, note = fn(volume)
    return NumericResult(value=round(value, 3), unit=unit, note=note)
