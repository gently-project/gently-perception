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

    ps = sub.add_parser("score", help="score an existing events.jsonl")
    ps.add_argument("events", type=Path)
    ps.add_argument("--stages", nargs="*", default=None)
    ps.add_argument("--gt", type=Path, default=None)

    sub.add_parser("baseline", help="show baseline.lock")

    pv = sub.add_parser("view", help="print one frame's trajectory")
    pv.add_argument("events", type=Path)
    pv.add_argument("--frame", required=True, help="embryo_id/Ttimepoint, e.g. embryo_2/T067")

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

    return 1


if __name__ == "__main__":
    sys.exit(main())
