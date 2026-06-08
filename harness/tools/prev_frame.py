"""prev_frame — render an earlier timepoint for direct visual comparison."""
from __future__ import annotations

from typing import Annotated

from harness.core.render import cached_render
from harness.core.types import ErrorResult, FrameInput, ImageResult, ToolResult
from harness.tools import Range, tool


@tool(returns="image")
def prev_frame(frame: FrameInput, *, offset: Annotated[int, Range(1, 10)] = 1) -> ToolResult:
    """Render the 3-view from `offset` timepoints ago for direct visual comparison.

    Morphological *change* indicates a stage transition; mere *movement* of the
    embryo within the eggshell does not — compare overall shape and fold count,
    not position.
    """
    refs = frame.prev_volume_refs
    if not refs or offset > len(refs):
        return ErrorResult(message=f"only {len(refs)} previous frame(s) available")
    path = refs[offset - 1]
    return ImageResult(b64=cached_render(path), caption=f"T{frame.timepoint - offset} (offset={offset})")
