"""3D volume raymarching primitive.

Pixel-equivalent port of the in-browser GLSL3 renderer used by
gently-annotator. Exposed as the ``Renderer`` class:

    from gently_perception.render import Renderer, CameraParams

    with Renderer(volume_u8, voxel_size_um=(1.0, 0.1625, 0.1625)) as r:
        img = r.render(CameraParams())     # default annotator startup pose
"""

from .projection import depth_colored_projection, projection_pixel_size

__all__ = [
    "Renderer",
    "CameraParams",
    "make_context",
    "depth_colored_projection",
    "projection_pixel_size",
]


def __getattr__(name):
    # Lazy-load the moderngl raymarcher so the lightweight projection helper
    # is importable in environments without moderngl.
    if name == "Renderer":
        from .renderer import Renderer
        return Renderer
    if name == "CameraParams":
        from ..types import CameraParams
        return CameraParams
    if name == "make_context":
        from .context import make_context
        return make_context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
