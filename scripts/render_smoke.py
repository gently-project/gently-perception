"""End-to-end smoke test for gently_perception.render.

Loads one timepoint from data/volumes/embryo_1/, applies the annotator's
preprocessing pipeline (X-half view crop + bg subtract + signal-percentile
contrast stretch + Z-blur + uint8 quantize), and renders at four canonical
poses. Writes PNGs under scripts/output/render_smoke/ for inspection.

This is intentionally a script (not a pytest test) because:
- it requires a real volume on disk and a working GL context
- the diff-against-reference assertion comes later, once we have a
  reference PNG captured from the annotator at the same pose

Run from the repo root::

    venv\\Scripts\\python.exe scripts\\render_smoke.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

from gently_perception.render import CameraParams, Renderer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

REPO = Path(__file__).resolve().parent.parent
TIF_PATH = REPO / "data" / "volumes" / "embryo_1" / "embryo_1_20251222_175656.tif"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "render_smoke"

VOXEL_SIZE_UM = (1.0, 0.1625, 0.1625)  # (dz, dy, dx)
BG_OFFSET = 100  # camera dark-current pedestal


def load_and_preprocess(tif_path: Path) -> np.ndarray:
    """Same pipeline as gently-annotator/annotator/volume_io.py."""
    vol = tifffile.imread(str(tif_path))
    if vol.ndim == 4:
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

    # X-half crop: each volume holds two views concatenated along X. The
    # left half is the strong-signal view in this dataset.
    x = vol.shape[2]
    if x % 2 != 0:
        raise ValueError(f"X dimension {x} odd; cannot split")
    vol = vol[:, :, : x // 2]

    # Background subtract with clip-to-zero.
    vol = vol.astype(np.int32, copy=False) - BG_OFFSET
    np.clip(vol, 0, None, out=vol)

    # Signal-percentile contrast stretch — only over non-zero voxels so the
    # background-clipped voxels don't drag p99 down.
    flat = vol.reshape(-1)
    positive = flat[flat > 0]
    if positive.size < 100:
        p1, p99 = 0.0, 1.0
    else:
        stride = max(1, positive.size // 100_000)
        p1, p99 = np.percentile(positive[::stride], [1.0, 99.0])

    vol_f = vol.astype(np.float32)
    vol_f = np.clip((vol_f - p1) / (p99 - p1 + 1e-8), 0, 1)
    vol_f = ndimage.gaussian_filter1d(vol_f, sigma=1.0, axis=0)

    return (vol_f * 255).astype(np.uint8)


def save_png(rgba: np.ndarray, path: Path) -> None:
    """Composite RGBA over black, write as RGB PNG."""
    Image.fromarray(rgba, mode="RGBA").convert("RGB").save(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TIF_PATH.exists():
        raise SystemExit(f"TIF not found: {TIF_PATH}")

    print(f"Loading {TIF_PATH.name} ...")
    vol_u8 = load_and_preprocess(TIF_PATH)
    print(f"  preprocessed shape={vol_u8.shape} dtype={vol_u8.dtype}")

    poses = {
        "default": CameraParams(),  # annotator's startup pose
        "front": CameraParams(quaternion=(0.0, 0.0, 0.0, 1.0)),  # identity
        "y_90":   CameraParams(quaternion=(0.0, 0.7071068, 0.0, 0.7071068)),
        "x_90":   CameraParams(quaternion=(0.7071068, 0.0, 0.0, 0.7071068)),
    }

    t0 = time.perf_counter()
    with Renderer(vol_u8, voxel_size_um=VOXEL_SIZE_UM) as r:
        ctor_ms = (time.perf_counter() - t0) * 1000
        info = r.ctx.info
        print(
            f"  GL_RENDERER={info.get('GL_RENDERER')!r} "
            f"GL_VENDOR={info.get('GL_VENDOR')!r}"
        )
        print(f"  Renderer construction (incl. volume upload): {ctor_ms:.1f} ms")

        for name, params in poses.items():
            print(f"Rendering pose '{name}' ...")
            img = r.render(params)
            out = OUTPUT_DIR / f"{name}.png"
            save_png(img, out)
            print(f"  -> {out}  alpha-mean={img[..., 3].mean():.1f}")

        # Steady-state latency: re-render the default pose N times and
        # report the distribution. First render in the loop above paid
        # for FBO creation, so this only times the hot path.
        N = 20
        params = next(iter(poses.values()))
        # Warmup
        r.render(params)
        timings_ms: list[float] = []
        for _ in range(N):
            t = time.perf_counter()
            r.render(params)
            timings_ms.append((time.perf_counter() - t) * 1000)
        timings = np.array(timings_ms)
        print(
            f"\nSteady-state render ({N} iters, {params.image_size[0]}x"
            f"{params.image_size[1]}, max_steps={params.max_steps}): "
            f"mean={timings.mean():.1f} ms  "
            f"min={timings.min():.1f} ms  "
            f"p50={np.median(timings):.1f} ms  "
            f"p95={np.percentile(timings, 95):.1f} ms  "
            f"max={timings.max():.1f} ms"
        )

    print(f"\nDone. Output PNGs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
