"""GL context creation with a GPU-only guard.

moderngl can silently fall back to a software rasterizer (Mesa llvmpipe,
SwiftShader, etc.) if no GPU-accelerated GL implementation is available.
Since the renderer is meant to run in an agent's perception loop where
per-call latency matters, silent CPU fallback is a real failure mode —
we'd rather refuse to start than render at 100x slowdown.

Set the env var ``GENTLY_PERCEPTION_ALLOW_CPU=1`` to bypass the guard
(useful for CI on machines without a GPU).
"""

from __future__ import annotations

import logging
import os

import moderngl

logger = logging.getLogger(__name__)

_ALLOW_CPU_ENV = "GENTLY_PERCEPTION_ALLOW_CPU"


def make_context(allow_cpu: bool = False) -> moderngl.Context:
    """Create a standalone moderngl context, refusing CPU fallback by default.

    Tries the platform default backend first (X11 on Linux desktops), then
    EGL, which is what works on headless machines (CI, remote boxes).
    """
    try:
        ctx = moderngl.create_context(standalone=True)
    except Exception as default_err:  # glcontext raises bare Exception for backend failures
        try:
            ctx = moderngl.create_context(standalone=True, backend="egl")
        except Exception as egl_err:
            raise RuntimeError(
                f"No usable GL backend: default backend failed ({default_err}); "
                f"EGL failed ({egl_err})."
            ) from egl_err
    info = ctx.info
    renderer = (info.get("GL_RENDERER") or "").lower()
    vendor = (info.get("GL_VENDOR") or "").lower()
    version = info.get("GL_VERSION", "?")

    is_software = any(
        marker in renderer
        for marker in ("llvmpipe", "softpipe", "swiftshader", "software")
    )

    logger.info(
        "GL context: vendor=%r renderer=%r version=%r software=%s",
        vendor, renderer, version, is_software,
    )

    if is_software and not allow_cpu and os.environ.get(_ALLOW_CPU_ENV) != "1":
        ctx.release()
        raise RuntimeError(
            f"Refusing to start with software OpenGL ({renderer!r}). "
            f"Install a vendor GL driver, or set {_ALLOW_CPU_ENV}=1 to override."
        )

    return ctx
