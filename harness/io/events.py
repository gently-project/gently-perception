"""
Append-only event log.

Every run writes one events.jsonl per seed. Scoring, reporting, and the
trajectory viewer are pure readers of this file — a run can be re-scored or
inspected without re-calling the model.

Event payloads follow the ThreadPiece-style shape from agents_core: a flat
record with kind, timestamp, frame coordinates, optional step index, optional
tool name, and a free-form payload dict.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

EventKind = Literal[
    "run_start",
    "frame_start",
    "model_turn",
    "tool_call",
    "prediction",
    "verify_override",
    "error",
    "frame_end",
    "run_end",
]


@dataclass
class Event:
    kind: EventKind
    ts: float
    embryo_id: str | None = None
    timepoint: int | None = None
    step: int | None = None
    tool_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v is not None and v != {}}
        return json.dumps(d, default=str)

    @classmethod
    def from_json(cls, line: str) -> "Event":
        d = json.loads(line)
        return cls(
            kind=d["kind"],
            ts=d["ts"],
            embryo_id=d.get("embryo_id"),
            timepoint=d.get("timepoint"),
            step=d.get("step"),
            tool_name=d.get("tool_name"),
            payload=d.get("payload", {}),
        )

    # --- Constructors -------------------------------------------------------

    @classmethod
    def run_start(cls, config: dict[str, Any]) -> "Event":
        return cls(kind="run_start", ts=time.time(), payload=config)

    @classmethod
    def run_end(cls, summary: dict[str, Any]) -> "Event":
        return cls(kind="run_end", ts=time.time(), payload=summary)

    @classmethod
    def frame_start(cls, embryo_id: str, timepoint: int) -> "Event":
        return cls(kind="frame_start", ts=time.time(), embryo_id=embryo_id, timepoint=timepoint)

    @classmethod
    def frame_end(cls, embryo_id: str, timepoint: int, n_steps: int, tokens: int) -> "Event":
        return cls(
            kind="frame_end",
            ts=time.time(),
            embryo_id=embryo_id,
            timepoint=timepoint,
            payload={"n_steps": n_steps, "tokens": tokens},
        )

    @classmethod
    def model_turn(cls, embryo_id: str, timepoint: int, step: int, usage: Any) -> "Event":
        return cls(
            kind="model_turn",
            ts=time.time(),
            embryo_id=embryo_id,
            timepoint=timepoint,
            step=step,
            payload={
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read": getattr(usage, "cache_read_input_tokens", 0),
            },
        )

    @classmethod
    def tool_call(
        cls, embryo_id: str, timepoint: int, step: int, name: str, params: dict, result_kind: str
    ) -> "Event":
        return cls(
            kind="tool_call",
            ts=time.time(),
            embryo_id=embryo_id,
            timepoint=timepoint,
            step=step,
            tool_name=name,
            payload={"params": params, "result_kind": result_kind},
        )

    @classmethod
    def prediction(
        cls, embryo_id: str, timepoint: int, stage: str, reasoning: str, budget_exhausted: bool
    ) -> "Event":
        return cls(
            kind="prediction",
            ts=time.time(),
            embryo_id=embryo_id,
            timepoint=timepoint,
            payload={
                "stage": stage,
                "reasoning": reasoning,
                "budget_exhausted": budget_exhausted,
            },
        )

    @classmethod
    def verify_override(
        cls, embryo_id: str, timepoint: int, original: str, corrected: str, reason: str
    ) -> "Event":
        return cls(
            kind="verify_override",
            ts=time.time(),
            embryo_id=embryo_id,
            timepoint=timepoint,
            payload={"original": original, "corrected": corrected, "reason": reason},
        )

    @classmethod
    def error(cls, embryo_id: str, timepoint: int, message: str) -> "Event":
        return cls(
            kind="error",
            ts=time.time(),
            embryo_id=embryo_id,
            timepoint=timepoint,
            payload={"message": message},
        )


class EventWriter:
    """Append-only JSONL writer. Flushes on every event so a crash mid-run leaves a valid log."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def __call__(self, ev: Event) -> None:
        self._fh.write(ev.to_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_events(path: Path) -> Iterator[Event]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Event.from_json(line)
