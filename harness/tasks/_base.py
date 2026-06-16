"""
Task construction helpers.

``make_onset_task(spec)`` turns a declarative ``SignalSpec`` into a runnable
``Task`` with the two-call perceive→classify pipeline. Nothing here is
dat-1-specific.
"""
from __future__ import annotations

import json
from typing import Any

from harness.core.model import DEFAULT_MODEL, generate, image_block, text_block
from harness.core.types_v2 import (
    Intensity,
    IntensityObservation,
    ModelOutputError,
    SetCadence,
    SignalSpec,
    Step,
    Task,
    TaskInput,
    Trajectory,
    now,
)

_CLASSIFY_TOOL = {
    "name": "classify_intensity",
    "description": "Report the signal intensity level for this frame.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intensity_level": {"type": "string", "enum": [i.value for i in Intensity]},
            "reasoning": {"type": "string"},
        },
        "required": ["intensity_level", "reasoning"],
    },
}


def _rubric_text(spec: SignalSpec) -> str:
    lines = [f"- {lvl.name}: {spec.rubric[lvl]}" for lvl in Intensity if lvl in spec.rubric]
    return "\n".join(lines)


def _classifier_system(spec: SignalSpec) -> str:
    return (
        "You are reading a microscopist's free-form description of one frame "
        f"and classifying the {spec.name} signal level.\n\n"
        "Call classify_intensity with one of: "
        f"{', '.join(i.value for i in Intensity)}.\n\n"
        f"Rubric:\n{_rubric_text(spec)}\n\n"
        "When the description is ambiguous, choose the lower level."
    )


async def _perceive(spec: SignalSpec, inp: TaskInput, model: str) -> tuple[str, Any]:
    """Vision call → free-form prose. Skipped if the router supplied a description."""
    if inp.description is not None:
        return inp.description, None
    msg = await generate(
        system=spec.describe_prompt,
        messages=[{"role": "user", "content": [image_block(inp.image_b64)]}],
        tools=[],
        model=model,
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if not text.strip():
        raise ModelOutputError("perceiver returned no text", msg)
    return text, msg


async def _classify(spec: SignalSpec, description: str, model: str) -> tuple[dict[str, Any], Any]:
    """Text-only call → forced classify_intensity tool."""
    msg = await generate(
        system=_classifier_system(spec),
        messages=[{"role": "user", "content": [text_block(description)]}],
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_intensity"},
        model=model,
    )
    for b in msg.content:
        if getattr(b, "type", "") == "tool_use" and b.name == "classify_intensity":
            return dict(b.input), msg
    raise ModelOutputError("classifier produced no classify_intensity call", msg)


def make_onset_task(spec: SignalSpec, *, model: str = DEFAULT_MODEL) -> Task[IntensityObservation]:
    threshold, debounce = spec.threshold, spec.debounce

    async def run(inp: TaskInput) -> tuple[IntensityObservation, Trajectory]:
        traj = Trajectory()
        desc, p_msg = await _perceive(spec, inp, model)
        if p_msg is not None:
            traj.steps.append(Step.model(p_msg))
            traj.messages.append({"role": "perceiver", "text": desc})
        raw, c_msg = await _classify(spec, desc, model)
        traj.steps.append(Step.model(c_msg))
        traj.messages.append({"role": "classifier", "raw": raw})

        try:
            level = Intensity(str(raw["intensity_level"]).lower())
        except (KeyError, ValueError) as e:
            raise ModelOutputError(f"bad intensity_level: {raw!r}") from e

        prev_onset = any(
            isinstance(o, IntensityObservation) and o.onset for o in inp.own_history
        )
        streak = 1
        for o in reversed(inp.own_history):
            if isinstance(o, IntensityObservation) and o.value >= threshold:
                streak += 1
            else:
                break
        onset = (not prev_onset) and level >= threshold and streak >= debounce

        obs = IntensityObservation(
            task=spec.name,
            timepoint=inp.timepoint,
            timestamp=now(),
            reasoning=str(raw.get("reasoning", "")),
            raw={"description": desc, **raw},
            value=level,
            onset=onset,
        )
        return obs, traj

    def on_observe(obs: IntensityObservation):
        if obs.onset and spec.on_detect is not None:
            return (spec.on_detect,)
        return ()

    from harness.eval.score_onset import latency_scorer

    return Task(
        name=spec.name,
        render=spec.render,
        run=run,
        obs_type=IntensityObservation,
        scorer=latency_scorer(threshold),
        arming=spec.arming,
        on_observe=on_observe,
    )


def spec_sha(spec: SignalSpec) -> str:
    """Stable hash of the prompt-bearing fields, for baseline.lock."""
    import hashlib

    blob = json.dumps(
        {
            "name": spec.name,
            "describe": spec.describe_prompt,
            "rubric": {k.value: v for k, v in spec.rubric.items()},
            "threshold": spec.threshold.value,
            "debounce": spec.debounce,
            "render": spec.render.name,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
