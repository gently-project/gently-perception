"""view3d golden tests with a synthetic volume.

Skipped wholesale when no GL context can be created (CI without EGL).
"""
import numpy as np
import pytest

pytest.importorskip("moderngl")

from gently_perception.render.context import make_context  # noqa: E402
from harness.tools.view3d import _pose, _quat_mul, view3d  # noqa: E402

try:
    make_context(allow_cpu=True).release()
except Exception:  # glcontext raises bare Exception when no backend works
    pytest.skip("no GL context available", allow_module_level=True)


def _asym_volume() -> np.ndarray:
    """A bright corner blob + axial rod: looks different from every side."""
    vol = np.zeros((40, 64, 96), dtype=np.float32)
    vol[5:15, 5:20, 5:25] = 900.0
    vol[18:22, 30:34, 10:90] = 600.0
    return vol


def test_identity_quaternion_math():
    q = _quat_mul((0, 0, 0, 1), (0.1, 0.2, 0.3, 0.9))
    assert np.allclose(q, (0.1, 0.2, 0.3, 0.9))


def test_zero_rotation_is_default_pose():
    from gently_perception.types import _DEFAULT_QUATERNION

    assert np.allclose(_pose(0, 0), _DEFAULT_QUATERNION)


def test_deterministic():
    vol = _asym_volume()
    a = view3d(vol, yaw_deg=45, pitch_deg=10, threshold=30)
    b = view3d(vol, yaw_deg=45, pitch_deg=10, threshold=30)
    assert a.b64 == b.b64


def test_rotation_changes_view():
    vol = _asym_volume()
    front = view3d(vol, yaw_deg=0).b64
    side = view3d(vol, yaw_deg=90).b64
    back = view3d(vol, yaw_deg=180).b64
    assert front != side and side != back and front != back


def test_caption_reports_params():
    out = view3d(_asym_volume(), yaw_deg=-30, pitch_deg=15, threshold=50)
    assert "yaw=-30" in out.caption and "pitch=15" in out.caption and "threshold=50" in out.caption
