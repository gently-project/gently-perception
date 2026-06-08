"""3D volume raymarching primitive.

Pixel-equivalent port of the in-browser GLSL3 renderer used by
gently-annotator. Exposed as the ``Renderer`` class:

    from gently_perception.render import Renderer, CameraParams

    with Renderer(volume_u8, voxel_size_um=(1.0, 0.1625, 0.1625)) as r:
        img = r.render(CameraParams())     # default annotator startup pose
"""

from ..types import CameraParams
from .context import make_context
from .renderer import Renderer

__all__ = ["Renderer", "CameraParams", "make_context"]
