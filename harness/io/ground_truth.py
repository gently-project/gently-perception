"""
Ground truth — stage transition timepoints per embryo.

Imported ONLY by harness/eval/. Never imported by core/, tools/, or solvers/.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from harness.core.types import STAGE_ORDER, Stage


@dataclass
class GroundTruth:
    """{embryo_id: {stage: start_timepoint}} — each stage runs until the next one starts."""

    transitions: dict[str, dict[Stage, int]]
    path: Path | None = None

    @classmethod
    def from_json(cls, path: Path) -> "GroundTruth":
        data = json.loads(Path(path).read_text())
        transitions: dict[str, dict[Stage, int]] = {}
        for embryo_id, stage_map in data["transitions"].items():
            transitions[embryo_id] = {Stage(s): t for s, t in stage_map.items()}
        return cls(transitions=transitions, path=Path(path))

    @property
    def embryo_ids(self) -> list[str]:
        return sorted(self.transitions)

    def get_stage_at(self, embryo_id: str, timepoint: int) -> Stage | None:
        stage_map = self.transitions.get(embryo_id)
        if not stage_map:
            return None
        current: Stage | None = None
        for stage in STAGE_ORDER:
            start = stage_map.get(stage)
            if start is not None and timepoint >= start:
                current = stage
        return current

    def sha(self) -> str:
        if self.path and self.path.exists():
            return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
        return hashlib.sha256(json.dumps(self.transitions, default=str, sort_keys=True).encode()).hexdigest()[:16]
