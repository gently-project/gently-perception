"""
Onset ground truth: per-embryo first-signal timepoint.

Kept separate from ``ground_truth.py`` (stage transitions) so the GT-import
invariant stays auditable: ``grep -r ground_truth harness/core harness/tools
harness/solvers harness/tasks`` must be empty.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OnsetGroundTruth:
    """Maps embryo_id → first timepoint at which the signal is present.

    ``None`` means the embryo is a negative control (no reporter) — every
    positive call on it is a false positive.
    """

    onset: dict[str, int | None]
    path: Path | None = None

    @classmethod
    def from_json(cls, path: Path) -> "OnsetGroundTruth":
        raw = json.loads(Path(path).read_text())
        onset = {k: (None if v is None else int(v)) for k, v in raw["onset"].items()}
        return cls(onset=onset, path=Path(path))

    @property
    def embryo_ids(self) -> list[str]:
        return sorted(self.onset)

    def is_positive(self, embryo_id: str, timepoint: int) -> bool | None:
        if embryo_id not in self.onset:
            return None
        t = self.onset[embryo_id]
        return False if t is None else timepoint >= t

    def sha(self) -> str:
        blob = json.dumps(self.onset, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
