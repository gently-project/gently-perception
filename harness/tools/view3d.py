"""view3d — raymarched render of the volume at a model-chosen camera angle.

Wraps gently_perception.render (the annotator's own viewer, pixel-equivalent)
as a model-callable tool: the model picks yaw/pitch relative to the
annotator's default working pose and optionally the intensity threshold,
and gets back a true volume render with depth occlusion — folds that overlap
in the flat projections stay separate here.

Determinism: same (volume, params) → same image. The GL context is
process-local plumbing (created once, reused); it does not affect output.
"""
from __future__ import annotations

import math
from typing import Annotated

import numpy as np

from gently_perception.render import Renderer
from gently_perception.render.context import make_context
from gently_perception.types import CameraParams, _DEFAULT_QUATERNION
from harness.core.render import array_to_b64
from harness.core.types import ImageResult
from harness.tools import Range, tool

_ctx = None


def _quat_mul(a, b):
    """Hamilton product of (x, y, z, w) quaternions: apply b, then a."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _pose(yaw_deg: float, pitch_deg: float):
    """Default annotator pose rotated by yaw (screen-vertical axis) then pitch."""
    half_yaw = math.radians(yaw_deg) / 2.0
    half_pitch = math.radians(pitch_deg) / 2.0
    q_yaw = (0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw))
    q_pitch = (math.sin(half_pitch), 0.0, 0.0, math.cos(half_pitch))
    return _quat_mul(q_pitch, _quat_mul(q_yaw, _DEFAULT_QUATERNION))


def _display(img_rgba: np.ndarray) -> np.ndarray:
    """Premultiplied RGBA over black → stretched, content-cropped grayscale."""
    img = img_rgba[..., 0].astype(np.float32)
    hi = float(np.percentile(img[img > 0], 99.5)) if (img > 0).any() else 1.0
    out = np.clip(img / max(hi, 1.0) * 255.0, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(out > 8)
    if len(ys):
        my, mx = int(out.shape[0] * 0.08), int(out.shape[1] * 0.08)
        out = out[max(ys.min() - my, 0) : ys.max() + my, max(xs.min() - mx, 0) : xs.max() + mx]
    return out


@tool(returns="image")
def view3d(
    volume: np.ndarray,
    *,
    yaw_deg: Annotated[int, Range(-180, 181)],
    pitch_deg: Annotated[int, Range(-90, 91)] = 0,
    threshold: Annotated[int, Range(5, 81)] = 30,
    zoom_pct: Annotated[int, Range(100, 401)] = 100,
    center_x_pct: Annotated[int, Range(0, 101)] = 50,
    center_y_pct: Annotated[int, Range(0, 101)] = 50,
) -> ImageResult:
    """Render the embryo volume in 3D — rotate, zoom, and center like the annotator's viewer.

    This is the same 3D viewer the human annotator used: a true volume render
    with depth — nearer structure occludes farther structure, so overlapping
    folds separate as you rotate. yaw_deg rotates the embryo about the
    screen-vertical axis (90 = side view, 180 = back), pitch_deg tilts it
    (positive = view more from above); both are relative to the annotator's
    default working pose. threshold sets the intensity cutoff (default 30 =
    the annotator's; raise it to peel away dim outer signal and expose
    internal fold structure). zoom_pct magnifies (200 = 2x, 400 = 4x) around
    the point (center_x_pct, center_y_pct), given as percentages of the
    current view's width/height — pick the point from a previous render of
    the SAME angle, e.g. zoom into where the tail tip or a suspected fold
    crossing is.
    """
    global _ctx
    if _ctx is None:
        _ctx = make_context(allow_cpu=True)
    vol = volume.astype(np.float32)
    lo, hi = float(vol.min()), float(vol.max())
    vol_u8 = ((vol - lo) / max(hi - lo, 1.0) * 255.0).astype(np.uint8)
    with Renderer(vol_u8, ctx=_ctx) as r:
        rgba = r.render(
            CameraParams(
                quaternion=_pose(yaw_deg, pitch_deg),
                threshold=float(threshold),
                image_size=(1024, 1024),  # render hi-res so zoom crops stay sharp
            )
        )
    img = _display(rgba)
    if zoom_pct > 100:
        h, w = img.shape
        win_h, win_w = int(h * 100 / zoom_pct), int(w * 100 / zoom_pct)
        cy = int(h * center_y_pct / 100)
        cx = int(w * center_x_pct / 100)
        y0 = min(max(cy - win_h // 2, 0), h - win_h)
        x0 = min(max(cx - win_w // 2, 0), w - win_w)
        img = img[y0 : y0 + win_h, x0 : x0 + win_w]
    caption = f"3D view yaw={yaw_deg}° pitch={pitch_deg}° threshold={threshold}"
    if zoom_pct > 100:
        caption += f" zoom={zoom_pct}% @({center_x_pct}%,{center_y_pct}%)"
    return ImageResult(b64=array_to_b64(img, target_long_edge=512), caption=caption)
