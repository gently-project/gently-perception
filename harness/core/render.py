"""
Volume → image rendering.

Lifted from benchmark/testset.py (_load_volume, _normalize_image,
_compute_crop_bounds, _projection_three_view) and made pure + cacheable.

The on-disk render cache makes full-loop evals tolerable on rerun: TIFF→JPEG is
the dominant cost, and the cache is content-addressed on (path, mtime, params).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# Bumping this invalidates every cached render — do so when projection logic changes.
RENDER_PARAMS_VERSION = 1

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / "data" / "render_cache"


@dataclass(frozen=True)
class RenderParams:
    isometric: bool = True
    crop: bool = True
    target_long_edge: int = 512
    jpeg_quality: int = 85


DEFAULT_PARAMS = RenderParams()


# --- Loading & normalization ------------------------------------------------


def load_volume(path: Path) -> np.ndarray:
    """Load a 3D volume from TIFF or NPZ. Returns (Z, Y, X) uint16/float array."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as f:
            return f[f.files[0]]
    return tifffile.imread(path)


def normalize(arr: np.ndarray, *, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    """Percentile-stretch to uint8."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def crop_bounds(volume: np.ndarray, *, margin: int = 16) -> tuple[slice, slice, slice]:
    """Center-of-mass auto-crop bounds for a (Z, Y, X) volume."""
    proj = volume.max(axis=0).astype(np.float32)
    thresh = proj.mean() + proj.std()
    mask = proj > thresh
    if not mask.any():
        return slice(None), slice(None), slice(None)
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - margin), min(proj.shape[0], ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(proj.shape[1], xs.max() + margin)
    return slice(None), slice(y0, y1), slice(x0, x1)


# --- Projections ------------------------------------------------------------


def project_axis(volume: np.ndarray, view: str) -> np.ndarray:
    """Max-intensity projection along one axis. view ∈ {xy, yz, xz}."""
    match view:
        case "xy":
            return volume.max(axis=0)
        case "yz":
            return volume.max(axis=2).T  # (Z, Y) → (Y, Z)
        case "xz":
            return volume.max(axis=1)  # (Z, X)
        case _:
            raise ValueError(f"unknown view {view!r}")


def three_view(volume: np.ndarray, *, isometric: bool = True) -> np.ndarray:
    """Compose XY (top-left), YZ (top-right), XZ (bottom-left) into one canvas.

    Isometric scaling stretches the Z axis so physical proportions are preserved
    (light-sheet Z spacing is typically coarser than XY).
    """
    xy = project_axis(volume, "xy")
    yz = project_axis(volume, "yz")
    xz = project_axis(volume, "xz")

    if isometric:
        z_scale = max(1, round(volume.shape[1] / max(volume.shape[0], 1) * 0.3))
        yz = np.repeat(yz, z_scale, axis=1)
        xz = np.repeat(xz, z_scale, axis=0)

    h_top = max(xy.shape[0], yz.shape[0])
    w_top = xy.shape[1] + yz.shape[1]
    h_bot = xz.shape[0]
    canvas = np.zeros((h_top + h_bot, w_top), dtype=volume.dtype)
    canvas[: xy.shape[0], : xy.shape[1]] = xy
    canvas[: yz.shape[0], xy.shape[1] : xy.shape[1] + yz.shape[1]] = yz
    canvas[h_top : h_top + xz.shape[0], : xz.shape[1]] = xz
    return canvas


# --- Encoding & cache -------------------------------------------------------


def array_to_b64(arr: np.ndarray, *, target_long_edge: int = 512, quality: int = 85) -> str:
    """Normalize, resize so the long edge is ~target, encode as base64 JPEG."""
    arr8 = normalize(arr)
    img = Image.fromarray(arr8)
    scale = target_long_edge / max(img.size)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render(volume: np.ndarray, params: RenderParams = DEFAULT_PARAMS) -> str:
    """Volume → base64 3-view JPEG. Pure; no caching."""
    if params.crop:
        volume = volume[crop_bounds(volume)]
    canvas = three_view(volume, isometric=params.isometric)
    return array_to_b64(canvas, target_long_edge=params.target_long_edge, quality=params.jpeg_quality)


def _cache_key(path: Path, params: RenderParams, extra: dict | None = None) -> str:
    payload = {
        "path": str(path),
        "mtime": path.stat().st_mtime if path.exists() else 0,
        "params": asdict(params),
        "version": RENDER_PARAMS_VERSION,
        "extra": extra or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cached_render(path: Path, params: RenderParams = DEFAULT_PARAMS) -> str:
    """Render with on-disk cache. Returns base64 JPEG."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(Path(path), params)
    cache_file = _CACHE_DIR / f"{key}.b64"
    if cache_file.exists():
        return cache_file.read_text()
    b64 = render(load_volume(path), params)
    cache_file.write_text(b64)
    return b64


def cached_tool_result(
    path: Path, tool_name: str, params: dict, compute, *, suffix: str = "b64"
) -> str:
    """Generic content-addressed cache for tool outputs keyed on (volume, tool, params)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(Path(path), DEFAULT_PARAMS, extra={"tool": tool_name, "args": params})
    cache_file = _CACHE_DIR / f"{key}.{suffix}"
    if cache_file.exists():
        return cache_file.read_text()
    result = compute()
    cache_file.write_text(result)
    return result
