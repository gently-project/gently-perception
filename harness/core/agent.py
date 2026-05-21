"""
The ReAct agent loop.

Modeled on inspect-ai's `basic_agent` and the internal `ToolUsingAgentInstance.act()`:

    generate → if classify_stage: stop
             → else: dispatch tool → append tool_result → repeat
    until classify_stage OR step budget exhausted (final step forces classify).

There is exactly one implementation of this loop. Both the eval runner and the
production Perceiver call `react()`; one-shot solvers (tools=(), max_steps=1)
degenerate to a single forced-classify call, so they go through the same path.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harness.core import model
from harness.core.render import load_volume
from harness.core.solver import Solver
from harness.core.types import (
    ErrorResult,
    FrameInput,
    ImageResult,
    ModelOutputError,
    NumericResult,
    Prediction,
    Stage,
    Step,
    ToolResult,
    Trajectory,
)
from harness.io.events import Event
from harness.tools import dispatch, get_tool_schemas
from harness.tools._classify import CLASSIFY_TOOL_SCHEMA

OnEvent = Callable[[Event], None]


def _noop(_: Event) -> None: ...


def reference_blocks(frame: FrameInput) -> list[dict[str, Any]]:
    """Cached reference-image prefix. Format matches perception/_base.py:build_reference_content."""
    from harness.core.types import STAGE_ORDER

    content: list[dict[str, Any]] = [model.text_block("REFERENCE EXAMPLES FOR EACH STAGE:")]
    for stage in STAGE_ORDER:
        imgs = frame.references.get(stage)
        if not imgs:
            continue
        content.append(model.text_block(f"\n{stage.value.upper()}"))
        for b64 in imgs:
            content.append(model.image_block(b64))
    if content:
        content[-1] = model.with_cache(content[-1])
    return content


def _default_user_blocks(frame: FrameInput) -> list[dict[str, Any]]:
    """Agentic default: image + history + tool-usage hint."""
    content: list[dict[str, Any]] = [model.text_block("Current frame:"), model.image_block(frame.image_b64)]
    if frame.history_text:
        content.append(model.text_block(frame.history_text))
    content.append(
        model.text_block(
            f"Embryo {frame.embryo_id}, timepoint {frame.timepoint}. "
            "Use tools to inspect if needed, then call classify_stage."
        )
    )
    return content


def _initial_user_block(frame: FrameInput, solver: Solver) -> list[dict[str, Any]]:
    blocks = solver.user_blocks or _default_user_blocks
    return reference_blocks(frame) + blocks(frame)


def _tool_result_block(tool_use_id: str, result: ToolResult) -> dict[str, Any]:
    match result:
        case ImageResult(b64=b64, caption=cap):
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [model.image_block(b64), model.text_block(cap or "")],
            }
        case NumericResult(value=v, unit=u, note=n):
            text = f"{v}{(' ' + u) if u else ''}" + (f"\n({n})" if n else "")
            return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
        case ErrorResult(message=m):
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Error: {m}",
                "is_error": True,
            }
    raise AssertionError(f"unhandled ToolResult: {result!r}")


def _first_tool_use(resp: Any) -> Any | None:
    for block in getattr(resp, "content", []):
        if getattr(block, "type", None) == "tool_use":
            return block
    return None


async def react(frame: FrameInput, solver: Solver, on_event: OnEvent = _noop) -> tuple[Prediction, Trajectory]:
    """Run the agent loop for one frame.

    Raises ModelOutputError if the model produces no tool_use block — never
    silently coerces to a fake stage.
    """
    volume = None  # lazily loaded on first volume-needing tool
    tool_schemas = get_tool_schemas(solver.tools) + [CLASSIFY_TOOL_SCHEMA]
    messages: list[dict[str, Any]] = [{"role": "user", "content": _initial_user_block(frame, solver)}]
    traj = Trajectory(messages=[m.copy() for m in messages])

    for step_i in range(solver.max_steps + 1):
        force_classify = step_i >= solver.max_steps or solver.is_one_shot
        choice = (
            {"type": "tool", "name": "classify_stage"} if force_classify else {"type": "any"}
        )
        resp = await model.generate(
            system=solver.system_for(frame),
            messages=messages,
            tools=tool_schemas,
            tool_choice=choice,
            model=solver.model,
            thinking={"type": solver.thinking} if solver.thinking else None,
            effort=solver.effort,
        )
        on_event(Event.model_turn(frame.embryo_id, frame.timepoint, step_i, resp.usage))
        traj.steps.append(Step.model(resp))

        tu = _first_tool_use(resp)
        if tu is None:
            raise ModelOutputError("model returned no tool_use block", resp)

        if tu.name == "classify_stage":
            traj.budget_exhausted = force_classify and step_i > 0 and not solver.is_one_shot
            traj.messages.append({"role": "assistant", "content": _content_dicts(resp)})
            return (
                Prediction(
                    stage=Stage(tu.input["stage"]),
                    reasoning=tu.input.get("reasoning", ""),
                    raw=dict(tu.input),
                ),
                traj,
            )

        # Perception tool — dispatch and continue.
        if volume is None and frame.volume_ref.exists():
            volume = load_volume(frame.volume_ref)
        result = dispatch(tu.name, dict(tu.input), volume=volume, frame=frame)
        on_event(
            Event.tool_call(
                frame.embryo_id, frame.timepoint, step_i, tu.name, dict(tu.input), result
            )
        )
        traj.steps.append(Step.tool(tu.name, dict(tu.input), result))

        assistant_msg = {"role": "assistant", "content": _content_dicts(resp)}
        tool_msg = {"role": "user", "content": [_tool_result_block(tu.id, result)]}
        messages += [assistant_msg, tool_msg]
        traj.messages += [assistant_msg, tool_msg]

    # Unreachable: the final iteration forces classify_stage via tool_choice.
    raise ModelOutputError("agent loop exhausted without classify_stage", None)


def _content_dicts(resp: Any) -> list[dict[str, Any]]:
    """Convert SDK content blocks to plain dicts for the message list."""
    out: list[dict[str, Any]] = []
    for b in getattr(resp, "content", []):
        if hasattr(b, "model_dump"):
            out.append(b.model_dump())
        elif hasattr(b, "__dict__"):
            out.append(dict(b.__dict__))
        else:
            out.append({"type": getattr(b, "type", "?")})
    return out
