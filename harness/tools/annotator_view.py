"""Raymarched volume renders at the annotator's poses.

Unlike the rotated MIPs (harness/tools/rotate.py — negative result twice),
these are true volume renders with depth occlusion and thresholding, produced
by the same raymarcher the human annotator used (gently_perception.render,
pixel-equivalent port of the annotator viewer). Tight pretzel coils that
merge in a max-projection stay visually distinct here.

Poses are frozen constants:
- "default": the annotator's startup pose — the view he spent most labeling
  time in.
- "tail": the pose from his embryo_5 T16 view note ("you can see the two
  lobes clearly. the left is the tail") — chosen because the prompt's
  transition-stage criteria are all about tail progress. A camera pose
  carries no label information, so freezing it is not ground-truth leakage.

Volumes normalize per-volume min-max to uint8 — that lands the imaging
background at ~30/255, which is exactly the annotator's default threshold,
i.e. the convention his pipeline uses.

One GL context and one cached Renderer per volume path are kept per process;
results are content-addressed via cached_tool_result so eval seeds beyond the
first hit the cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gently_perception.render import Renderer
from gently_perception.render.context import make_context
from gently_perception.types import CameraParams
from harness.core.render import array_to_b64, cached_tool_result, load_volume

POSES: dict[str, CameraParams] = {
    "default": CameraParams(),
    "tail": CameraParams(
        quaternion=(
            -0.5161865499649309,
            0.18639103472417876,
            0.014405218785850556,
            0.8358243341046486,
        ),
    ),
}

_ctx = None
_renderer: tuple[Path, Renderer] | None = None


def _render(volume_ref: Path, pose: str) -> np.ndarray:
    global _ctx, _renderer
    if _ctx is None:
        _ctx = make_context(allow_cpu=True)
    if _renderer is None or _renderer[0] != volume_ref:
        if _renderer is not None:
            _renderer[1].close()
        vol = load_volume(volume_ref).astype(np.float64)
        lo, hi = vol.min(), vol.max()
        vol_u8 = ((vol - lo) / max(hi - lo, 1.0) * 255).astype(np.uint8)
        _renderer = (volume_ref, Renderer(vol_u8, ctx=_ctx))
    rgba = _renderer[1].render(POSES[pose])
    # Premultiplied RGBA over the annotator's black canvas == the RGB channel.
    img = rgba[..., 0].astype(np.float32)  # grayscale volume → channels equal
    # Display stretch (annotator used a contrast slider for the same purpose):
    # late-stage volumes have hot spots that crush the body's brightness range.
    hi = float(np.percentile(img[img > 0], 99.5)) if (img > 0).any() else 1.0
    out = np.clip(img / max(hi, 1.0) * 255.0, 0, 255).astype(np.uint8)
    # Crop to content + 8% margin so the embryo fills the image the model sees.
    ys, xs = np.nonzero(out > 8)
    if len(ys):
        my, mx = int(out.shape[0] * 0.08), int(out.shape[1] * 0.08)
        out = out[
            max(ys.min() - my, 0) : ys.max() + my,
            max(xs.min() - mx, 0) : xs.max() + mx,
        ]
    return out


def annotator_view_b64(volume_ref: Path, pose: str) -> str:
    """Cached base64 JPEG of the volume raymarched at a named annotator pose."""
    assert pose in POSES, f"unknown pose {pose!r}"

    def compute() -> str:
        return array_to_b64(_render(Path(volume_ref), pose), target_long_edge=512)

    return cached_tool_result(
        Path(volume_ref), "annotator_view", {"pose": pose, "v": 3}, compute
    )
