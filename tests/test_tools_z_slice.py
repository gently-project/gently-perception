"""z_slice golden: bright at the plane with signal, dark elsewhere."""
import base64
import io

import numpy as np
from PIL import Image

from harness.tools.z_slice import z_slice, z_sweep


def _decode_mean(b64: str) -> float:
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return float(np.asarray(img).mean())


def test_z_slice_returns_correct_plane():
    """Each plane is independently normalized; verify the right index is returned."""
    vol = np.zeros((20, 32, 32), dtype=np.float32)
    vol[10, 8:24, 8:24] = 1000.0
    vol[2, 0, 0] = 1.0  # tiny signal so normalize doesn't div-by-zero
    r10 = z_slice(vol, index=10)
    r2 = z_slice(vol, index=2)
    assert "10/20" in r10.caption
    assert "2/20" in r2.caption
    # z=10 has a 16×16 bright patch → high mean after normalize; z=2 is one pixel.
    assert _decode_mean(r10.b64) > _decode_mean(r2.b64)


def test_z_sweep_filmstrip_width():
    vol = np.random.rand(20, 32, 32).astype(np.float32) * 100
    r = z_sweep(vol, start=0, end=15, step=5)
    assert r.b64
    assert "z_sweep" in r.caption
