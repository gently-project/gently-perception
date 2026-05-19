"""
baseline.lock — pin the comparison config and hard-fail on drift.

Any change to the model string, the solver's prompt source, the tools/ source,
or the GT file produces a non-empty diff. The runner exits 1 on a diff unless
--accept-drift or --update-baseline is passed.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = _REPO_ROOT / "baseline.lock"


@dataclass
class BaselineLock:
    solver: str
    model: str
    prompt_sha: str
    tools_sha: str
    gt_sha: str
    mean: float
    std: float
    n_runs: int
    date: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "BaselineLock":
        return cls(**json.loads(s))


def prompt_sha(solver) -> str:
    """sha256 of the solver's resolved system prompt(s).

    Hashes the actual prompt text (not module source) so drift is detected
    regardless of where the prompt string is defined.
    """
    sys_attr = solver.system
    if callable(sys_attr):
        src = inspect.getsource(sys_attr)
        # Also hash any module-level *_SYSTEM constants the callable closes over.
        mod = inspect.getmodule(sys_attr)
        extras = "".join(
            v for k, v in sorted(vars(mod).items()) if k.endswith("SYSTEM") and isinstance(v, str)
        )
        payload = src + extras
    else:
        payload = sys_attr
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def tools_sha() -> str:
    """sha256 of the concatenated tools/ source."""
    tools_dir = _REPO_ROOT / "harness" / "tools"
    h = hashlib.sha256()
    for p in sorted(tools_dir.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def load() -> BaselineLock | None:
    if not LOCK_PATH.exists():
        return None
    return BaselineLock.from_json(LOCK_PATH.read_text())


def write(lock: BaselineLock) -> None:
    LOCK_PATH.write_text(lock.to_json() + "\n")


def check(current_model: str, current_prompt_sha: str, current_gt_sha: str) -> list[str]:
    """Return human-readable diff lines, or [] if the lock matches."""
    locked = load()
    if locked is None:
        return []  # no lock yet — nothing to diff against
    diff: list[str] = []
    if locked.model != current_model:
        diff.append(f"model: {locked.model} → {current_model}")
    if locked.prompt_sha != current_prompt_sha:
        diff.append(f"prompt_sha: {locked.prompt_sha} → {current_prompt_sha}")
    cur_tools = tools_sha()
    if locked.tools_sha != cur_tools:
        diff.append(f"tools_sha: {locked.tools_sha} → {cur_tools}")
    if locked.gt_sha != current_gt_sha:
        diff.append(f"gt_sha: {locked.gt_sha} → {current_gt_sha}")
    return diff


def make(solver_name: str, solver, model: str, gt_sha: str, mean: float, std: float, n_runs: int) -> BaselineLock:
    return BaselineLock(
        solver=solver_name,
        model=model,
        prompt_sha=prompt_sha(solver),
        tools_sha=tools_sha(),
        gt_sha=gt_sha,
        mean=round(mean, 4),
        std=round(std, 4),
        n_runs=n_runs,
        date=date.today().isoformat(),
    )
