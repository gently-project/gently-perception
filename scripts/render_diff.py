"""Pixel diff: annotator canvas vs gently_perception.render at the same
volume bytes + camera params.

Reads the annotator canvas as base64 from
``scripts/output/render_smoke/annotator_canvas.b64`` (captured via Chrome
DevTools toDataURL), fetches the matching volume bytes from the
annotator's HTTP API, renders that volume at the annotator's documented
default pose, and writes:

  annotator_canvas.png     decoded annotator capture
  ours_at_annotator_pose.png   our renderer's output at the same pose
  diff_amplified_8x.png    abs-diff per pixel, 8x amplified for visibility
  side_by_side.png         annotator | ours | diff stacked horizontally

Run from repo root::

    venv\\Scripts\\python.exe scripts\\render_diff.py
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path
from urllib import request

import numpy as np
from PIL import Image

from gently_perception.render import CameraParams, Renderer

REPO = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent / "output" / "render_smoke"

ANNOTATOR_BASE = "http://127.0.0.1:8090"
DATASET = "Gently2"
SESSION = "0a288534"
EMBRYO = "embryo_3"
TIMEPOINT = 14


def fetch_volume() -> tuple[np.ndarray, tuple[float, float, float]]:
    """GET volume bytes from the annotator. Reads shape + voxel size from headers."""
    url = (
        f"{ANNOTATOR_BASE}/api/datasets/{DATASET}/sessions/{SESSION}"
        f"/embryos/{EMBRYO}/volumes/{TIMEPOINT}"
    )
    print(f"Fetching {url} ...")
    with request.urlopen(url) as r:
        headers = {k.lower(): v for k, v in r.headers.items()}
        body = r.read()
    print(f"  Volume headers: {[ (k, headers[k]) for k in headers if k.startswith('x-volume') ]}")

    # The annotator's volume route exposes shape + voxel via X-Volume-* headers.
    shape_hdr = headers.get("x-volume-shape")
    voxel_hdr = (
        headers.get("x-volume-voxel-size-um")
        or headers.get("x-volume-voxel-size")
        or headers.get("x-volume-voxel")
    )

    if shape_hdr and voxel_hdr:
        zd, h, w = (int(x) for x in shape_hdr.split(","))
        dz, dy, dx = (float(x) for x in voxel_hdr.split(","))
    else:
        # Fall back to the on-disk sidecar header (GAV1, see preview_cache.py)
        # format: <4sIIII3fI = magic(4) version(I) zd(I) h(I) w(I) dz dy dx(3f) reserved(I)
        magic, version, zd, h, w, dz, dy, dx, _ = struct.unpack("<4sIIII3fI", body[:36])
        if magic != b"GAV1":
            raise SystemExit(f"Unknown body header: shape_hdr={shape_hdr!r} magic={magic!r}")
        body = body[36:]
        print(f"  Parsed GAV1 header: shape=({zd},{h},{w}) voxel=({dz},{dy},{dx})")

    expected = zd * h * w
    if len(body) != expected:
        raise SystemExit(f"Body size {len(body)} != expected {expected}")
    vol = np.frombuffer(body, dtype=np.uint8).reshape(zd, h, w).copy()
    return vol, (dz, dy, dx)


def composite_over_black(rgba: np.ndarray) -> np.ndarray:
    """RGBA (top-to-bottom, premultiplied) → RGB displayed over black bg.

    Our renderer's framebuffer holds premultiplied RGBA (the shader does
    rgb += (1-a_dst) * color * alpha; alpha += (1-a_dst) * alpha). Composited
    over black, the displayed RGB is simply the premultiplied RGB channel.
    """
    return rgba[..., :3].copy()


def composite_png_over_black(path: Path) -> np.ndarray:
    """PNG (unpremultiplied RGBA per spec) → RGB displayed over black."""
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    out = Image.alpha_composite(bg, img).convert("RGB")
    return np.array(out)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # --- 1. Read the annotator canvas. Saved to disk by Chrome DevTools'
    #        take_screenshot (or by a base64 fallback for older flows).
    annot_png_path = OUTPUT / "annotator_canvas.png"
    if not annot_png_path.exists():
        b64_path = OUTPUT / "annotator_canvas.b64"
        if not b64_path.exists():
            raise SystemExit(
                f"Missing {annot_png_path} and {b64_path}; capture annotator first."
            )
        raw = b64_path.read_text(encoding="utf-8").strip()
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        annot_png_path.write_bytes(base64.b64decode(raw))
    annot_rgb_full = composite_png_over_black(annot_png_path)
    Hf, Wf, _ = annot_rgb_full.shape
    # Chrome's take_screenshot captures at devicePixelRatio. The canvas's
    # backing store (where GL actually renders) is the smaller of CSS-px
    # and backing-px — for our annotator both are 1376x356, but the screenshot
    # is upscaled. Downsample back so we compare native render resolutions.
    CANVAS_W, CANVAS_H = 1376, 356
    if (Wf, Hf) != (CANVAS_W, CANVAS_H):
        print(f"Annotator screenshot: {Wf}x{Hf} -> resampling to canvas backing {CANVAS_W}x{CANVAS_H}")
        annot_rgb = np.array(
            Image.fromarray(annot_rgb_full).resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        )
    else:
        annot_rgb = annot_rgb_full
    H, W, _ = annot_rgb.shape
    print(f"Annotator canvas (native): {W}x{H}, mean={annot_rgb.mean():.2f}, max={annot_rgb.max()}")

    # --- 2. Fetch the same volume bytes the annotator used ---
    vol, voxel_size = fetch_volume()
    print(f"Volume: shape={vol.shape}, voxel_size_um={voxel_size}")

    # --- 3. Render at the annotator's reset-view default pose ---
    params = CameraParams(image_size=(W, H))  # default = annotator startup pose
    with Renderer(vol, voxel_size_um=voxel_size) as r:
        rgba = r.render(params)
    ours_rgb = composite_over_black(rgba)
    ours_path = OUTPUT / "ours_at_annotator_pose.png"
    Image.fromarray(ours_rgb).save(ours_path)
    print(f"Our render: {ours_path}, mean={ours_rgb.mean():.2f}, max={ours_rgb.max()}")

    # --- 4. Diff ---
    diff = np.abs(annot_rgb.astype(np.int16) - ours_rgb.astype(np.int16)).astype(np.uint8)
    print("\nDiff (annotator vs ours, both composited over black):")
    print(f"  mean abs diff:   {diff.mean():.2f}  (out of 255)")
    print(f"  median abs diff: {float(np.median(diff)):.2f}")
    print(f"  p95 abs diff:    {float(np.percentile(diff, 95)):.2f}")
    print(f"  max abs diff:    {int(diff.max())}")
    print(f"  pixels with diff > 5:   {(diff.max(axis=-1) > 5).mean() * 100:.2f}%")
    print(f"  pixels with diff > 16:  {(diff.max(axis=-1) > 16).mean() * 100:.2f}%")

    diff_amp = np.clip(diff.astype(np.int16) * 8, 0, 255).astype(np.uint8)
    diff_path = OUTPUT / "diff_amplified_8x.png"
    Image.fromarray(diff_amp).save(diff_path)
    print(f"  diff (8x amplified): {diff_path}")

    side = np.concatenate([annot_rgb, ours_rgb, diff_amp], axis=1)
    side_path = OUTPUT / "side_by_side.png"
    Image.fromarray(side).save(side_path)
    print(f"  side-by-side (annotator | ours | diff8x): {side_path}")


if __name__ == "__main__":
    main()
