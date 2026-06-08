"""rotated_mip golden tests with synthetic volumes.

The property under test: structure separated only in Z (invisible in the XY
max-projection) becomes separated in the image after rotation about X.
"""
import numpy as np
from scipy import ndimage

from harness.tools.rotate import rotated_mip


def _two_rods(z_a: int = 12, z_b: int = 38) -> np.ndarray:
    """Two bright rods along X at the same Y, different Z.

    Overlapping in the XY projection; distinct after a 90° rotation about X.
    """
    vol = np.zeros((50, 64, 160), dtype=np.float32)
    vol[z_a - 2 : z_a + 2, 30:34, 20:140] = 100.0
    vol[z_b - 2 : z_b + 2, 30:34, 20:140] = 100.0
    return vol


def _n_bands(img: np.ndarray) -> int:
    labeled, n = ndimage.label(img > 64)  # type: ignore[misc]
    sizes = ndimage.sum_labels(np.ones_like(img), labeled, index=list(range(1, int(n) + 1)))
    return int((np.asarray(sizes) > 50).sum())  # ignore interpolation speckle


def test_output_shape_and_dtype():
    img = rotated_mip(_two_rods(), 45)
    assert img.ndim == 2 and img.dtype == np.uint8


def test_zero_angle_collapses_z_overlap():
    assert _n_bands(rotated_mip(_two_rods(), 0)) == 1


def test_rotation_separates_z_structure():
    assert _n_bands(rotated_mip(_two_rods(), 90)) == 2


def test_full_turn_matches_zero():
    a = rotated_mip(_two_rods(), 0)
    b = rotated_mip(_two_rods(), 360)
    assert a.shape == b.shape
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) < 2
