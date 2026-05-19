"""
Shared fixtures: synthetic volumes and a stub model.generate.

No test in this suite makes a real API call — model.generate is monkeypatched
to return canned responses.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def arun(coro_or_agen):
    """Run a coroutine or fully drain an async generator. Bulletproof against
    monorepo PYTEST_DISABLE_PLUGIN_AUTOLOAD blocking pytest-asyncio."""
    import asyncio
    import inspect

    if inspect.isasyncgen(coro_or_agen):

        async def _drain():
            out = []
            async for x in coro_or_agen:
                out.append(x)
            return out

        return asyncio.run(_drain())
    return asyncio.run(coro_or_agen)


# --- Synthetic volumes ------------------------------------------------------


def make_volume(
    shape: tuple[int, int, int] = (20, 64, 64),
    *,
    n_blobs: int = 1,
    fill: float = 0.5,
) -> np.ndarray:
    """A (Z,Y,X) volume with `n_blobs` hard-edged bright spheres.

    Hard edges (binary masks) so Otsu thresholding produces predictable masks
    for the measure() golden tests.
    """
    Z, Y, X = shape
    vol = np.full(shape, 10.0, dtype=np.float32)  # low background, not zero
    blob_r = max(2, int(min(Y, X) * 0.12 * fill * 2))  # blob radius scales with fill
    spread = X * 0.6
    centers_x = np.linspace(X / 2 - spread / 2, X / 2 + spread / 2, n_blobs) if n_blobs > 1 else [X / 2]
    zz, yy, xx = np.mgrid[:Z, :Y, :X]
    for cx in centers_x:
        mask = ((zz - Z / 2) ** 2 + (yy - Y / 2) ** 2 + (xx - cx) ** 2) < blob_r**2
        vol[mask] = 1000.0
    return vol


@pytest.fixture
def volume() -> np.ndarray:
    return make_volume()


@pytest.fixture
def tiny_frame(tmp_path):
    """A FrameInput backed by a real on-disk synthetic volume (so tools work)."""
    from harness.core.types import FrameInput, Stage

    vol = make_volume()
    p = tmp_path / "embryo_1_T005.npz"
    np.savez(p, vol)
    return FrameInput(
        embryo_id="embryo_1",
        timepoint=5,
        image_b64="ZmFrZQ==",
        volume_ref=p,
        references={},
        history=(),
        history_text="",
        last_stage=None,
        prev_volume_refs=(),
    )


# --- Stub model -------------------------------------------------------------


def fake_response(blocks: list[dict[str, Any]], *, in_tok: int = 10, out_tok: int = 5) -> Any:
    """Build a fake anthropic.types.Message with given content blocks."""
    content = [SimpleNamespace(**b, model_dump=lambda b=b: b) for b in blocks]
    usage = SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok, cache_read_input_tokens=0)
    return SimpleNamespace(content=content, usage=usage, stop_reason="tool_use")


def classify_block(stage: str, reasoning: str = "test") -> dict[str, Any]:
    return {"type": "tool_use", "id": "tu_1", "name": "classify_stage", "input": {"stage": stage, "reasoning": reasoning}}


def tool_block(name: str, **input_kw) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"tu_{name}", "name": name, "input": input_kw}


@pytest.fixture
def stub_generate(monkeypatch):
    """Patch model.generate with a queue of canned responses."""
    from harness.core import model

    queue: list[Any] = []
    calls: list[dict] = []

    async def _fake(**kw):
        calls.append(kw)
        if not queue:
            raise AssertionError("stub_generate queue exhausted")
        return queue.pop(0)

    monkeypatch.setattr(model, "generate", _fake)
    return SimpleNamespace(queue=queue, calls=calls)
