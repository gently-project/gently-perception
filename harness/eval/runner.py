"""
Research entry point — run a solver N times, score, aggregate, enforce baseline.
"""
from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path

from harness.core import loop, verify
from harness.core.solver import Solver
from harness.core.types import Stage
from harness.eval import baseline, report, score
from harness.io.events import Event, EventWriter
from harness.io.ground_truth import GroundTruth
from harness.io.volumes import OfflineSource

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO_ROOT / "data"
RUNS_DIR = _REPO_ROOT / "runs"


def _load_solver(name: str) -> tuple[Solver, object]:
    """Return (Solver, module) — module is needed for prompt_sha."""
    mod = importlib.import_module(f"harness.solvers.{name}")
    return mod.SOLVER, mod


def _load_refs() -> dict[Stage, tuple[str, ...]]:
    """Load reference images from data/examples/{stage}/*.jpg as base64."""
    import base64

    refs: dict[Stage, tuple[str, ...]] = {}
    examples = DATA_DIR / "examples"
    for stage in Stage:
        d = examples / stage.value
        if not d.exists():
            continue
        imgs: list[str] = []
        for p in sorted(d.glob("*.jpg"))[:2]:
            imgs.append(base64.b64encode(p.read_bytes()).decode("ascii"))
        if imgs:
            refs[stage] = tuple(imgs)
    return refs


async def run(
    solver_name: str,
    *,
    n_runs: int = 3,
    stages: set[Stage] | None = None,
    verifier_name: str = "monotonic",
    accept_drift: bool = False,
    update_baseline: bool = False,
    volumes_dir: Path | None = None,
    gt_path: Path | None = None,
    concurrency: int = 4,
    embryos: set[str] | None = None,
) -> dict:
    solver, _solver_mod = _load_solver(solver_name)
    gt = GroundTruth.from_json(gt_path or (DATA_DIR / "ground_truth" / "59799c78.json"))
    source: list[tuple[str, int, Path]] = list(OfflineSource(volumes_dir or (DATA_DIR / "volumes")))
    if embryos:
        source = [item for item in source if item[0] in embryos]
        assert source, f"no volumes found for embryos {sorted(embryos)}"
    refs = _load_refs()
    verifier = verify.VERIFIERS[verifier_name]

    # Baseline drift check (hard fail per user decision).
    p_sha = baseline.prompt_sha(solver)
    diff = baseline.check(solver.model, p_sha, gt.sha())
    if diff and not (accept_drift or update_baseline):
        print("BASELINE STALE:", file=sys.stderr)
        for line in diff:
            print(f"  {line}", file=sys.stderr)
        print("Pass --accept-drift or --update-baseline.", file=sys.stderr)
        sys.exit(1)

    if n_runs < 3:
        print(f"⚠ n_runs={n_runs} < 3 — single-run results are not defensible.", file=sys.stderr)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / solver_name / solver.model / ts
    seed_dirs: list[Path] = []

    for seed in range(n_runs):
        sd = run_dir / f"seed{seed}"
        seed_dirs.append(sd)
        with EventWriter(sd / "events.jsonl") as writer:
            writer(
                Event.run_start(
                    {
                        "solver": solver_name,
                        "model": solver.model,
                        "prompt_sha": p_sha,
                        "tools_sha": baseline.tools_sha(),
                        "gt_sha": gt.sha(),
                        "seed": seed,
                        "n_frames": len(source),
                        "tools": list(solver.tools),
                        "max_steps": solver.max_steps,
                    }
                )
            )
            n = 0
            async for frame, pred in loop.run_loop(
                source, solver, refs=refs, verifier=verifier, on_event=writer, concurrency=concurrency
            ):
                n += 1
                if n % 50 == 0:
                    print(f"[{solver_name}/seed{seed}] {n}/{len(source)} frames")
            writer(Event.run_end({"n_frames": n}))
            print(f"[{solver_name}/seed{seed}] done: {n} frames")

    # Score & aggregate.
    scores = [score.score_run(d / "events.jsonl", gt, stages=stages) for d in seed_dirs]
    agg = report.aggregate(scores)

    locked = baseline.load()
    report.print_table(agg, baseline_mean=locked.mean if locked else None, baseline_stale=bool(diff))

    if update_baseline:
        lock = baseline.make(solver_name, solver, solver.model, gt.sha(), agg["accuracy_mean"], agg["accuracy_std"], n_runs)
        baseline.write(lock)
        print(f"baseline.lock updated → {lock.mean:.1%} ± {lock.std:.1%}")

    # Static HTML report — pure reader of the run dir; thumbnails are render-cache hits.
    from harness.eval.html_report import generate as generate_report

    try:
        report_path = generate_report(run_dir, gt_path=gt_path, volumes_dir=volumes_dir)
        print(f"report: file://{report_path.resolve()}")
    except Exception as e:  # report failure must not fail the eval
        print(f"⚠ report generation failed: {e}", file=sys.stderr)

    error_rate = agg["n_errors"] / max(agg["n"] * n_runs, 1)
    if error_rate > 0.02:
        print(f"✗ error rate {error_rate:.1%} > 2% — run failed.", file=sys.stderr)
        sys.exit(2)

    return {"run_dir": str(run_dir), **agg}
