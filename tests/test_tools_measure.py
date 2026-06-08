"""measure() golden tests with parametric synthetic volumes."""
import numpy as np

from harness.tools.measure import measure
from tests.conftest import make_volume


def test_convexity_range():
    """Single convex blob → fill ≈ 1; multi-blob with gaps → fill < 1."""
    convex = measure(make_volume(n_blobs=1, fill=0.5), feature="convexity").value
    gappy = measure(make_volume(shape=(20, 64, 160), n_blobs=3, fill=0.3), feature="convexity").value
    assert 0.85 <= convex <= 1.0, convex
    assert gappy < convex, (gappy, convex)


def test_n_segments_counts_separated_blobs():
    for n in (1, 2, 3):
        vol = make_volume(shape=(20, 64, 160), n_blobs=n, fill=0.3)
        r = measure(vol, feature="n_segments")
        assert r.value == n, f"n_blobs={n} → {r.value}"


def test_aspect_ratio_wide_vs_tall():
    wide = make_volume(shape=(20, 48, 160), n_blobs=3, fill=0.3)
    r = measure(wide, feature="aspect_ratio")
    assert r.value > 2.0, r.value


def test_empty_volume_safe():
    vol = np.zeros((10, 32, 32), dtype=np.float32)
    for feat in ("convexity", "n_segments", "aspect_ratio"):
        assert measure(vol, feature=feat).value == 0.0
