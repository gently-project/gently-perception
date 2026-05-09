"""moderngl-based 3D volume raymarcher.

Pixel-equivalent port of the in-browser GLSL3 renderer in
``gently-annotator/static/js/viewer.js``. Same vertex/fragment shader,
same uniforms, same camera and box-sizing math — so the agent's
view matches what the human annotator sees.

Construction uploads the volume to the GPU once. Repeated ``render()``
calls reuse the upload — typical use is one Renderer per
(embryo, timepoint), then many ``render()`` calls at different poses
while the agent reasons.
"""

from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np

from ..types import CameraParams
from .context import make_context

_SHADERS_DIR = Path(__file__).parent / "shaders"


def _read_shader(name: str) -> str:
    return (_SHADERS_DIR / name).read_text(encoding="utf-8")


# ---- Math helpers (numpy float32, OpenGL conventions) ----------------------

def _quat_to_mat4(q):
    """(x, y, z, w) unit quaternion → 4×4 row-major rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),     0],
            [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),     0],
            [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy), 0],
            [0,                 0,                 0,                 1],
        ],
        dtype=np.float32,
    )


def _scale_mat4(sx, sy, sz):
    return np.diag([sx, sy, sz, 1]).astype(np.float32)


def _translate_mat4(tx, ty, tz):
    M = np.eye(4, dtype=np.float32)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


def _perspective_mat4(fov_deg, aspect, near, far):
    """Right-handed OpenGL perspective projection."""
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array(
        [
            [f / aspect, 0, 0,                              0],
            [0,          f, 0,                              0],
            [0,          0, (far + near) / (near - far),  (2 * far * near) / (near - far)],
            [0,          0, -1,                             0],
        ],
        dtype=np.float32,
    )


def _make_box_vertices(box_w, box_h, box_d):
    """36 positions for a centred box. CCW winding when viewed from outside."""
    hx, hy, hz = box_w / 2, box_h / 2, box_d / 2
    p = np.array(
        [
            [-hx, -hy, -hz],  # 0
            [ hx, -hy, -hz],  # 1
            [ hx,  hy, -hz],  # 2
            [-hx,  hy, -hz],  # 3
            [-hx, -hy,  hz],  # 4
            [ hx, -hy,  hz],  # 5
            [ hx,  hy,  hz],  # 6
            [-hx,  hy,  hz],  # 7
        ],
        dtype=np.float32,
    )
    idx = [
        0, 3, 2,    0, 2, 1,   # -Z face
        4, 5, 6,    4, 6, 7,   # +Z face
        0, 4, 7,    0, 7, 3,   # -X face
        1, 2, 6,    1, 6, 5,   # +X face
        0, 1, 5,    0, 5, 4,   # -Y face
        3, 7, 6,    3, 6, 2,   # +Y face
    ]
    return p[idx].copy()


# ---- Renderer --------------------------------------------------------------

class Renderer:
    """Pixel-equivalent port of the annotator's WebGL2 raymarcher.

    Usage as a context manager::

        with Renderer(volume_u8, voxel_size_um=(1.0, 0.1625, 0.1625)) as r:
            img = r.render(CameraParams(...))

    The returned ``img`` is an HxWx4 uint8 RGBA array (top-to-bottom,
    PNG/numpy convention). Alpha is the accumulated transparency from the
    raymarch — same as the annotator's canvas, which composites it over
    a black clear colour.
    """

    def __init__(
        self,
        volume_u8: np.ndarray,
        voxel_size_um: tuple[float, float, float] = (1.0, 0.1625, 0.1625),
        ctx: moderngl.Context | None = None,
    ):
        if volume_u8.dtype != np.uint8:
            raise ValueError(f"volume must be uint8, got {volume_u8.dtype}")
        if volume_u8.ndim != 3:
            raise ValueError(f"volume must be 3D (Z, Y, X), got shape {volume_u8.shape}")

        self.volume_shape = volume_u8.shape  # (zd, h, w)
        self.voxel_size_um = tuple(float(s) for s in voxel_size_um)
        self._owns_ctx = ctx is None
        self.ctx = ctx if ctx is not None else make_context()

        zd, h, w = self.volume_shape
        # Box extents normalised so the largest physical axis = 1.0 unit.
        # Matches the annotator's "fit inside a unit sphere" sizing.
        dz, dy, dx = self.voxel_size_um
        x_extent = w * dx
        y_extent = h * dy
        z_extent = zd * dz
        max_extent = max(x_extent, y_extent, z_extent)
        self.box_size = (
            x_extent / max_extent,
            y_extent / max_extent,
            z_extent / max_extent,
        )

        self.program = self.ctx.program(
            vertex_shader=_read_shader("raymarch.vert"),
            fragment_shader=_read_shader("raymarch.frag"),
        )

        # 3D texture: numpy (Z, Y, X) → OpenGL (W, H, D) with the same byte order.
        self.volume_tex = self.ctx.texture3d(
            size=(w, h, zd),
            components=1,
            data=volume_u8.tobytes(),
            dtype="f1",
        )
        self.volume_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.volume_tex.repeat_x = False
        self.volume_tex.repeat_y = False
        self.volume_tex.repeat_z = False

        verts = _make_box_vertices(*self.box_size)
        self.vbo = self.ctx.buffer(verts.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, "aPosition")

        self._fbo: moderngl.Framebuffer | None = None
        self._color_tex: moderngl.Texture | None = None
        self._fbo_size: tuple[int, int] = (0, 0)

    def _ensure_fbo(self, width: int, height: int) -> None:
        if self._fbo is not None and self._fbo_size == (width, height):
            return
        if self._fbo is not None:
            self._fbo.release()
            assert self._color_tex is not None
            self._color_tex.release()
        self._color_tex = self.ctx.texture((width, height), 4)  # RGBA8
        self._fbo = self.ctx.framebuffer(color_attachments=[self._color_tex])
        self._fbo_size = (width, height)

    def render(self, params: CameraParams) -> np.ndarray:
        """Render one view. Returns HxWx4 uint8 RGBA (top-to-bottom)."""
        W, H = params.image_size
        self._ensure_fbo(W, H)
        assert self._fbo is not None

        aspect = W / H

        R = _quat_to_mat4(params.quaternion)
        S = _scale_mat4(1, -1, 1)
        M = R @ S
        V = _translate_mat4(0, 0, -params.zoom)
        P = _perspective_mat4(params.fov_deg, aspect, params.near, params.far)

        # Camera in mesh-local frame. (R*S)^-1 = S^-1 * R^-1 = S * R^T.
        cam_world = np.array([0, 0, params.zoom, 1], dtype=np.float32)
        cam_local = (S @ R.T @ cam_world)[:3]

        # GLSL stores matrices column-major; numpy is row-major. Transpose.
        self.program["uModel"].write(np.ascontiguousarray(M.T).tobytes())
        self.program["uView"].write(np.ascontiguousarray(V.T).tobytes())
        self.program["uProjection"].write(np.ascontiguousarray(P.T).tobytes())
        self.program["uBoxSize"].value = self.box_size
        # Annotator stores threshold in 0..100 UI units; shader expects 0..1
        # against the normalised intensity (voxel/255). Match the annotator.
        self.program["uThreshold"].value = float(params.threshold) / 255.0
        self.program["uContrast"].value = float(params.contrast)
        self.program["uCameraObjectPos"].value = tuple(float(c) for c in cam_local)
        self.program["uMaxSteps"].value = int(params.max_steps)
        self.volume_tex.use(location=0)
        self.program["uVolume"].value = 0

        self._fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        # THREE.js BackSide ⇔ cull front faces in OpenGL — rays start
        # inside the cube, far walls render.
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.front_face = "ccw"
        self.ctx.cull_face = "front"

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.DEPTH_TEST)

        self.vao.render()

        # OpenGL origin is bottom-left; numpy/PNG is top-left.
        data = self._fbo.read(components=4, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 4)
        return np.flipud(img).copy()

    def close(self) -> None:
        if self._fbo is not None:
            self._fbo.release()
            self._fbo = None
        if self._color_tex is not None:
            self._color_tex.release()
            self._color_tex = None
        self.vao.release()
        self.vbo.release()
        self.volume_tex.release()
        self.program.release()
        if self._owns_ctx:
            self.ctx.release()

    def __enter__(self) -> Renderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
