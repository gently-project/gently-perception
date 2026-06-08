"""Shared types for the perception harness."""

from dataclasses import dataclass, field


# Quaternion for the annotator's startup pose: THREE.Euler(-0.5, 0.5, 0, "XYZ").
# Computed from THREE's setFromEuler('XYZ') formula:
#   x = s1*c2*c3 + c1*s2*s3 = sin(-.25)*cos(.25) = -0.23971276955
#   y = c1*s2*c3 - s1*c2*s3 = cos(-.25)*sin(.25) =  0.23971276955
#   z = c1*c2*s3 + s1*s2*c3 = sin(-.25)*sin(.25) = -0.06120871779
#   w = c1*c2*c3 - s1*s2*s3 = cos(-.25)*cos(.25) =  0.93879116869
# Inlined so importing types doesn't pull in numpy at import time.
_DEFAULT_QUATERNION = (-0.23971276955, 0.23971276955, -0.06120871779, 0.93879116869)


@dataclass
class PerceptionOutput:
    """What every perceive function returns.

    confidence is optional — the paper shows VLM self-reported confidence
    is uncalibrated noise (0.867 correct vs 0.857 wrong). The harness
    derives reliability from session history (stability, temporal analysis)
    rather than this field. Experiments may still populate it for analysis.
    """

    stage: str
    reasoning: str
    confidence: float = 0.0  # VLM self-report; unreliable — see paper
    raw_response: str = ""


@dataclass
class CameraParams:
    """One render request for the volume raymarcher.

    Mirrors the annotator's ``captureViewParams`` shape — view-notes
    captured by humans in the annotator can be replayed verbatim by the
    renderer.

    quaternion: (x, y, z, w) — applied to the volumeGroup. Default is
        the annotator's startup pose (Euler -0.5, 0.5, 0 XYZ).
    zoom: camera distance along +Z. Default 0.9 = annotator default.
    threshold: 0..100 UI value (matches captureViewParams). The shader
        divides by 255 internally.
    contrast: 0.5..3.0
    image_size: (width, height) of the output image
    fov_deg, near, far: camera intrinsics. Defaults match the annotator.
    max_steps: raymarch step count. 192 = annotator default; agent calls
        can drop to 96 during exploration, raise to 256+ for final renders.
    """

    quaternion: tuple[float, float, float, float] = field(
        default_factory=lambda: _DEFAULT_QUATERNION
    )
    zoom: float = 0.9
    threshold: float = 30.0
    contrast: float = 1.0
    image_size: tuple[int, int] = (512, 512)
    fov_deg: float = 50.0
    near: float = 0.1
    far: float = 100.0
    max_steps: int = 384
