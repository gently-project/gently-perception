"""
CLI entry point.

    python -m harness eval --solver hybrid --n-runs 3 [--stages 1.5fold 2fold pretzel]
    python -m harness score runs/.../events.jsonl [--stages ...]
    python -m harness baseline
    python -m harness view runs/.../events.jsonl --frame embryo_2/T067
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from harness.core.types import Stage


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="run a solver N times and score")
    pe.add_argument("--solver", required=True)
    pe.add_argument("--n-runs", type=int, default=3)
    pe.add_argument("--stages", nargs="*", default=None)
    pe.add_argument("--verify", default="monotonic", choices=["none", "monotonic"])
    pe.add_argument("--accept-drift", action="store_true")
    pe.add_argument("--update-baseline", action="store_true")
    pe.add_argument("--concurrency", type=int, default=4, help="parallel embryo sessions")
    pe.add_argument("--embryos", nargs="*", default=None, help="restrict to these embryo ids")
    pe.add_argument("--volumes", type=Path, default=None, help="volumes directory override")
    pe.add_argument("--gt", type=Path, default=None, help="ground truth JSON override")

    ps = sub.add_parser("score", help="score an existing events.jsonl")
    ps.add_argument("events", type=Path)
    ps.add_argument("--stages", nargs="*", default=None)
    ps.add_argument("--gt", type=Path, default=None)

    sub.add_parser("baseline", help="show baseline.lock")

    pv = sub.add_parser("view", help="print one frame's trajectory")
    pv.add_argument("events", type=Path)
    pv.add_argument("--frame", required=True, help="embryo_id/Ttimepoint, e.g. embryo_2/T067")

    pr = sub.add_parser("report", help="generate a static HTML report for a run directory")
    pr.add_argument("run_dir", type=Path, help="e.g. runs/hybrid/claude-opus-4-6/20260520-201511")
    pr.add_argument("--seed", type=int, default=0, help="which seed's frame-level detail to show")
    pr.add_argument("--compare", type=Path, default=None, help="another run dir to diff against")
    pr.add_argument("--gt", type=Path, default=None)

    pt = sub.add_parser("eval-task", help="run a perception task (onset detection) and score")
    pt.add_argument("--task", required=True)
    pt.add_argument("--n-runs", type=int, default=1)
    pt.add_argument("--embryos", nargs="*", default=None)
    pt.add_argument("--limit", type=int, default=None, help="max frames per embryo")
    pt.add_argument("--concurrency", type=int, default=4)
    pt.add_argument("--volumes", type=Path, default=None)
    pt.add_argument("--gt", type=Path, default=None)

    args = p.parse_args(argv)

    if args.cmd == "eval":
        from harness.eval.runner import run

        stages = {Stage(s) for s in args.stages} if args.stages else None
        asyncio.run(
            run(
                args.solver,
                n_runs=args.n_runs,
                stages=stages,
                verifier_name=args.verify,
                accept_drift=args.accept_drift,
                update_baseline=args.update_baseline,
                concurrency=args.concurrency,
                embryos=set(args.embryos) if args.embryos else None,
                volumes_dir=args.volumes,
                gt_path=args.gt,
            )
        )
        return 0

    if args.cmd == "score":
        from harness.eval import report, score
        from harness.io.ground_truth import GroundTruth
        from harness.eval.runner import DATA_DIR

        gt = GroundTruth.from_json(args.gt or (DATA_DIR / "ground_truth" / "59799c78.json"))
        stages = {Stage(s) for s in args.stages} if args.stages else None
        s = score.score_run(args.events, gt, stages=stages)
        report.print_table(report.aggregate([s]))
        return 0

    if args.cmd == "baseline":
        from harness.eval import baseline

        lock = baseline.load()
        print(lock.to_json() if lock else "No baseline.lock yet.")
        return 0

    if args.cmd == "view":
        from harness.eval.report import view_trajectory

        eid, t = args.frame.split("/")
        view_trajectory(args.events, eid, int(t.lstrip("T")))
        return 0

    if args.cmd == "eval-task":
        from harness.eval.runner_task import run_task

        asyncio.run(
            run_task(
                args.task,
                n_runs=args.n_runs,
                embryos=set(args.embryos) if args.embryos else None,
                limit=args.limit,
                concurrency=args.concurrency,
                volumes_dir=args.volumes,
                gt_path=args.gt,
            )
        )
        return 0

    if args.cmd == "report":
        from harness.eval.html_report import generate

        out = generate(args.run_dir, seed=args.seed, compare_dir=args.compare, gt_path=args.gt)
        print(f"report: file://{out.resolve()}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
