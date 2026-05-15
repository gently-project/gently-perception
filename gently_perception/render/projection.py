"""Depth-coded max-intensity projection of a 3D volume.

Each slice along ``axis`` is tinted by a sample from a matplotlib colormap
at its normalized depth, optionally weighted by per-slice min-max-normalized
intensity, then max-projected. The result is an ``(H, W, 3)`` uint8 RGB
image where hue encodes the depth at which the brightest signal occurred.

This is the canonical pure-numpy implementation. The sibling ``Renderer``
in this package is the GPU raymarcher; this module is for the lightweight
projection use case (inline previews, benchmark figures, web thumbnails).
"""

from __future__ import annotations

import numpy as np


def depth_colored_projection(
    volume: np.ndarray,
    *,
    axis: int = 0,
    colormap: str = "turbo",
    voxel_size: tuple[float, float, float] | None = None,
    isometric: bool = False,
    slice_normalize: bool = True,
) -> np.ndarray:
    """Depth-coded max-intensity projection.

    Parameters
    ----------
    volume
        3D array, shape ``(D0, D1, D2)``.
    axis
        Projection axis (``0``, ``1``, or ``2``). Slices along this axis are
        tinted by depth.
    colormap
        Matplotlib colormap name. Default ``"turbo"``.
    voxel_size
        Physical voxel dimensions ``(d0, d1, d2)`` in any consistent unit.
        Required when ``isometric=True``; otherwise ignored.
    isometric
        If True, the output is resampled (PIL LANCZOS) so display pixels are
        physically square in the projection plane. Requires ``voxel_size``.
    slice_normalize
        If True (default), each slice is min-max-normalized to [0, 1] before
        coloring. This keeps depth coloring visible across dim and bright
        slices but discards absolute-intensity contrast. Set False for an
        "honest" MIP where the brightest voxels dominate.

    Returns
    -------
    ``(H, W, 3)`` uint8 RGB array.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    if isometric and voxel_size is None:
        raise ValueError("isometric=True requires voxel_size")

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    vol = np.moveaxis(volume, axis, 0).astype(np.float32)
    n, height, width = vol.shape

    if slice_normalize:
        colored = np.zeros((n, height, width, 3), dtype=np.float32)
        for i in range(n):
            color = np.asarray(cmap(i / max(1, n - 1))[:3], dtype=np.float32)
            s = vol[i]
            lo, hi = float(s.min()), float(s.max())
            sn = (s - lo) / max(1.0, hi - lo)
            colored[i] = sn[:, :, np.newaxis] * color
    else:
        lo, hi = float(vol.min()), float(vol.max())
        vol_norm = (vol - lo) / max(1.0, hi - lo)
        colored = np.zeros((n, height, width, 3), dtype=np.float32)
        for i in range(n):
            color = np.asarray(cmap(i / max(1, n - 1))[:3], dtype=np.float32)
            colored[i] = vol_norm[i, :, :, np.newaxis] * color

    rgb = (np.max(colored, axis=0) * 255).astype(np.uint8)

    if isometric:
        from PIL import Image
        dv, du = projection_pixel_size(voxel_size, axis)
        pix = min(dv, du)
        new_h = max(1, int(round(height * dv / pix)))
        new_w = max(1, int(round(width * du / pix)))
        if (new_h, new_w) != (height, width):
            rgb = np.array(
                Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
            )

    return rgb


def projection_pixel_size(
    voxel_size: tuple[float, float, float], axis: int
) -> tuple[float, float]:
    """Physical (row, col) pixel size of the MIP plane along ``axis``.

    Use to pass ``extent=(0, W*du, H*dv, 0)`` to ``matplotlib.pyplot.imshow``.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    remaining = [voxel_size[i] for i in (0, 1, 2) if i != axis]
    return remaining[0], remaining[1]
