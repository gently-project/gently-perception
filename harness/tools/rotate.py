"""Rotated max-intensity projections of the 3D volume.

`rotated_mip_b64()` is the shared primitive: resample to isotropic voxels,
rotate about the anterior-posterior (X) axis, max-project, crop, encode.
Folds that overlap in the default XY projection separate under rotation,
which is the information the fold-count stages (1.5fold/2fold/pretzel) need.

Not yet registered as a model-callable tool — currently consumed directly by
the hybrid_3dviews solver (static angle set). Registering it for agentic use
would change tools_sha and fire baseline drift for every solver, so that step
is deliberately deferred to its own experiment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from harness.core.render import (
    VOXEL_SIZE,
    array_to_b64,
    cached_tool_result,
    crop_bounds,
    load_volume,
    normalize,
)

# XY downsample before the (expensive) isotropic resample + rotation. 2× keeps
# fold-scale structure (folds are tens of pixels wide) at ~8× less compute.
_XY_DOWNSAMPLE = 2
_TARGET_LONG_EDGE = 768


def rotated_mip(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    """Max-intensity projection after rotating `angle_deg` about the X axis.

    The volume is made isotropic first — Z spacing is ~6× the lateral spacing
    (VOXEL_SIZE), and rotating an anisotropic grid distorts body shape.
    Returns a uint8 image. angle_deg=0 reproduces the plain XY projection
    (downsampled), which the tests rely on.
    """
    y0, y1, x0, x1 = crop_bounds(volume)
    vol = volume[:, y0:y1:_XY_DOWNSAMPLE, x0:x1:_XY_DOWNSAMPLE].astype(np.float32)
    z_factor = VOXEL_SIZE[0] / (VOXEL_SIZE[1] * _XY_DOWNSAMPLE)
    vol = np.asarray(ndimage.zoom(vol, (z_factor, 1.0, 1.0), order=1))
    if angle_deg % 360 != 0:
        # axes=(0, 1) is the Z-Y plane: rotation about the X (body) axis.
        vol = np.asarray(ndimage.rotate(vol, angle_deg, axes=(0, 1), order=1, reshape=True))
    return normalize(vol.max(axis=0))


def rotated_mip_b64(volume_ref: Path, angle_deg: float) -> str:
    """Cached base64 JPEG of `rotated_mip` for a volume on disk."""

    def compute() -> str:
        return array_to_b64(
            rotated_mip(load_volume(volume_ref), angle_deg),
            target_long_edge=_TARGET_LONG_EDGE,
        )

    return cached_tool_result(
        Path(volume_ref),
        "rotated_mip",
        {"angle_deg": float(angle_deg), "xy_ds": _XY_DOWNSAMPLE, "edge": _TARGET_LONG_EDGE},
        compute,
    )
