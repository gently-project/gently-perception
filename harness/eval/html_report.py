"""
Static HTML run report.

`generate(run_dir)` reads events.jsonl (+ media/), joins with ground truth,
renders thumbnails, and writes a self-contained report.html next to the seeds.
No server, no frameworks — vanilla HTML/CSS/JS in the style of make_filmstrip.py.

The report is a pure reader of the run directory: it never calls the model and
never mutates events.
"""
from __future__ import annotations

import base64
import importlib
import io
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image

from harness.core.render import cached_render, cached_tool_result, load_volume
from harness.core.types import STAGE_ORDER, ImageResult, Stage
from harness.eval import report as text_report
from harness.eval.score import score_run
from harness.io.events import read_events
from harness.io.ground_truth import GroundTruth
from harness.io.volumes import OfflineSource

_REPO_ROOT = Path(__file__).resolve().parents[2]
THUMB_LONG_EDGE = 260

_FS_UNSAFE = re.compile(r"[^A-Za-z0-9_\-]")


def _fs_name(value: Any) -> str:
    """Filesystem/URL-safe asset-name component. Embryo ids come from
    events.jsonl, which may not be trusted (shared/downloaded run dirs)."""
    return _FS_UNSAFE.sub("_", str(value))


# --- Data assembly ----------------------------------------------------------


def _frames_from_events(events_path: Path, gt: GroundTruth, seed_rel: str) -> list[dict[str, Any]]:
    """One record per frame: prediction, GT, trajectory steps, tokens."""
    frames: dict[tuple[str, int], dict[str, Any]] = {}

    def rec(e: str, t: int) -> dict[str, Any]:
        return frames.setdefault(
            (e, t),
            {"e": e, "t": t, "steps": [], "tokens": 0, "pred": None, "gt": None, "override": None, "err": None},
        )

    for ev in read_events(events_path):
        if ev.embryo_id is None or ev.timepoint is None:
            continue
        r = rec(ev.embryo_id, ev.timepoint)
        match ev.kind:
            case "model_turn":
                r["tokens"] += ev.payload.get("input_tokens", 0) + ev.payload.get("output_tokens", 0)
            case "tool_call":
                step: dict[str, Any] = {
                    "name": ev.tool_name,
                    "params": ev.payload.get("params", {}),
                    "kind": ev.payload.get("result_kind"),
                }
                if "image_path" in ev.payload:
                    step["img"] = f"{seed_rel}/{ev.payload['image_path']}"
                if "value" in ev.payload:
                    step["value"] = ev.payload["value"]
                    step["note"] = ev.payload.get("note", "")
                if "message" in ev.payload:
                    step["error"] = ev.payload["message"]
                r["steps"].append(step)
            case "verify_override":
                r["override"] = f"{ev.payload['original']} → {ev.payload['corrected']} ({ev.payload['reason']})"
            case "prediction":
                r["pred"] = ev.payload["stage"]
                r["reasoning"] = ev.payload.get("reasoning", "")
                r["budget_exhausted"] = ev.payload.get("budget_exhausted", False)
            case "error":
                r["err"] = ev.payload.get("message", "")

    out: list[dict[str, Any]] = []
    for (e, t), r in sorted(frames.items()):
        true = gt.get_stage_at(e, t)
        r["gt"] = true.value if true else None
        if r["pred"] is not None and true is not None:
            pred = Stage(r["pred"])
            r["ok"] = pred == true
            r["adj"] = pred.distance(true) <= 1
        else:
            r["ok"] = None
            r["adj"] = None
        out.append(r)
    return out


def _thumbnails(frames: list[dict[str, Any]], assets_dir: Path, volumes_dir: Path) -> None:
    """Write a small JPEG per frame into assets_dir. Skips frames whose volume is missing."""
    if not volumes_dir.exists():
        return
    paths = {(e, t): p for e, t, p in OfflineSource(volumes_dir)}
    assets_dir.mkdir(parents=True, exist_ok=True)
    for r in frames:
        key = (r["e"], r["t"])
        if key not in paths:
            continue
        thumb = assets_dir / f"{_fs_name(r['e'])}_T{int(r['t']):03d}.jpg"
        r["thumb"] = f"report_assets/{thumb.name}"
        if thumb.exists():
            continue
        full = Image.open(io.BytesIO(base64.b64decode(cached_render(paths[key]))))
        scale = THUMB_LONG_EDGE / max(full.size)
        if scale < 1.0:
            full = full.resize((int(full.width * scale), int(full.height * scale)), Image.Resampling.LANCZOS)
        full.save(thumb, format="JPEG", quality=80)


def _rotated_thumbnails(
    frames: list[dict[str, Any]], assets_dir: Path, volumes_dir: Path, solver_name: str
) -> None:
    """For solvers that fed rotated views to the model (an ANGLES attribute on
    the solver module), write those views for every FAILED frame so the report
    shows exactly what the model was looking at when it got it wrong."""
    try:
        angles = importlib.import_module(f"harness.solvers.{solver_name}").ANGLES
    except (ImportError, AttributeError):
        return
    if not volumes_dir.exists():
        return
    from harness.tools.rotate import rotated_mip_b64  # heavy scipy import, only needed here

    paths = {(e, t): p for e, t, p in OfflineSource(volumes_dir)}
    assets_dir.mkdir(parents=True, exist_ok=True)
    for r in frames:
        key = (r["e"], r["t"])
        if r.get("ok") is not False or key not in paths:  # only true failures
            continue
        rot: list[dict[str, Any]] = []
        for angle in angles:
            name = f"{_fs_name(r['e'])}_T{int(r['t']):03d}_rot{int(angle)}.jpg"
            out = assets_dir / name
            rot.append({"a": int(angle), "src": f"report_assets/{name}"})
            if out.exists():
                continue
            img = Image.open(io.BytesIO(base64.b64decode(rotated_mip_b64(paths[key], angle))))
            scale = THUMB_LONG_EDGE / max(img.size)
            if scale < 1.0:
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            img.save(out, format="JPEG", quality=80)
        r["rot"] = rot


def _view3d_step_assets(frames: list[dict[str, Any]], run_dir: Path, volumes_dir: Path) -> None:
    """Re-render every view3d step image from its recorded params.

    The harness keys media files by model-turn index, so parallel tool calls
    in one turn overwrite each other's image. view3d is deterministic, so the
    recorded params are the authoritative source; renders hit the run's own
    dispatch cache (same (volume, tool, params) key) and are cheap.
    """
    needs = [
        (r, j, s)
        for r in frames
        for j, s in enumerate(r.get("steps", []))
        if s.get("name") == "view3d" and s.get("params") is not None
    ]
    if not needs or not volumes_dir.exists():
        return
    from harness.tools import REGISTRY, _fill_defaults  # deferred: pulls in GL deps

    spec = REGISTRY.get("view3d")
    if spec is None:
        return
    paths = {(e, t): p for e, t, p in OfflineSource(volumes_dir)}
    assets = run_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    for r, j, s in needs:
        vol_ref = paths.get((r["e"], r["t"]))
        if vol_ref is None:
            continue
        params = s["params"]

        def compute(vol_ref: Path = vol_ref, params: dict[str, Any] = params) -> str:
            result = spec.fn(load_volume(vol_ref), **_fill_defaults(spec, params))
            assert isinstance(result, ImageResult)
            return result.b64

        name = f"{_fs_name(r['e'])}_T{int(r['t']):03d}_nav{j}.jpg"
        out = assets / name
        assert out.resolve().is_relative_to(assets.resolve())
        if not out.exists():
            b64 = cached_tool_result(Path(vol_ref), "view3d", params, compute)
            out.write_bytes(base64.b64decode(b64))
        s["img"] = f"report_assets/{name}"


def _summary(run_dir: Path, gt: GroundTruth) -> dict[str, Any]:
    seed_files = sorted(run_dir.glob("seed*/events.jsonl"))
    scores = [score_run(p, gt) for p in seed_files]
    agg = text_report.aggregate(scores)
    hard = [score_run(p, gt, stages={Stage.ONE_HALF_FOLD, Stage.TWO_FOLD, Stage.PRETZEL}) for p in seed_files]
    hard_agg = text_report.aggregate(hard) if hard else {}
    confusion: dict[str, int] = {}
    for s in scores[:1]:  # confusion from the detail seed only
        for (true, pred), n in s.confusion.items():
            confusion[f"{true}|{pred}"] = n
    return {**agg, "hard": hard_agg, "confusion": confusion, "n_seeds": len(seed_files)}


def _run_config(events_path: Path) -> dict[str, Any]:
    for ev in read_events(events_path):
        if ev.kind == "run_start":
            return dict(ev.payload)
        break
    return {}


def _compare(frames: list[dict[str, Any]], other_events: Path) -> dict[str, Any]:
    """Join on (embryo, t); bucket into wins / regressions / changed-still-wrong."""
    theirs: dict[tuple[str, int], str] = {}
    for ev in read_events(other_events):
        if ev.kind == "prediction" and ev.embryo_id is not None and ev.timepoint is not None:
            theirs[(ev.embryo_id, ev.timepoint)] = ev.payload["stage"]
    wins, regressions, changed = [], [], []
    for r in frames:
        key = (r["e"], r["t"])
        if key not in theirs or r["gt"] is None or r["pred"] is None:
            continue
        b_pred = theirs[key]
        b_ok = b_pred == r["gt"]
        entry = {"e": r["e"], "t": r["t"], "gt": r["gt"], "a": r["pred"], "b": b_pred, "thumb": r.get("thumb")}
        if r["ok"] and not b_ok:
            wins.append(entry)
        elif not r["ok"] and b_ok:
            regressions.append(entry)
        elif not r["ok"] and not b_ok and r["pred"] != b_pred:
            changed.append(entry)
    return {"wins": wins, "regressions": regressions, "changed": changed, "n_joined": len(theirs)}


def _failure_clusters(run_dir: Path, gt: GroundTruth, *, detail_seed: int) -> dict[str, Any] | None:
    """Group wrong predictions from all seeds into three behavioural clusters.

    late_arrival  — the model predicts the GT stage somewhere inside its window, just late
    window_missed — the GT stage's window passes without the model ever predicting it
    ahead         — the model predicts a stage later than the annotation

    Examples are taken from the detail seed so the report can link them to frame cards.
    """
    keys = ("late_arrival", "window_missed", "ahead")
    counts = dict.fromkeys(keys, 0)
    embryos: dict[str, dict[str, int]] = {k: {} for k in keys}
    pairs: dict[str, dict[str, int]] = {k: {} for k in keys}
    examples: dict[str, list[dict[str, Any]]] = {k: [] for k in keys}
    lags: list[int] = []
    windows_missed = windows_total = n_pred = n_fail = 0

    for events_path in sorted(run_dir.glob("seed*/events.jsonl")):
        preds: dict[tuple[str, int], Stage] = {}
        for ev in read_events(events_path):
            if ev.kind == "prediction" and ev.embryo_id is not None and ev.timepoint is not None:
                preds[(ev.embryo_id, ev.timepoint)] = Stage(ev.payload["stage"])
        if not preds:
            continue
        is_detail = events_path == run_dir / f"seed{detail_seed}" / "events.jsonl"

        # Which GT windows does this seed ever enter, and how late?
        reached: dict[tuple[str, Stage], bool] = {}
        for embryo, stage_map in gt.transitions.items():
            ts = sorted(t for (e, t) in preds if e == embryo)
            if not ts:
                continue
            ordered = sorted(stage_map.items(), key=lambda kv: kv[1])
            for i, (stage, start) in enumerate(ordered):
                end = ordered[i + 1][1] - 1 if i + 1 < len(ordered) else ts[-1]
                in_window = [t for t in ts if start <= t <= end]
                if not in_window:
                    continue
                windows_total += 1
                first_hit = next((t for t in in_window if preds[(embryo, t)] == stage), None)
                reached[(embryo, stage)] = first_hit is not None
                if first_hit is None:
                    windows_missed += 1
                elif first_hit > start:
                    lags.append(first_hit - start)

        for (embryo, t), pred in preds.items():
            true = gt.get_stage_at(embryo, t)
            if true is None:
                continue
            n_pred += 1
            if pred == true:
                continue
            n_fail += 1
            if pred.ordinal > true.ordinal:
                key = "ahead"
            elif reached.get((embryo, true)):
                key = "late_arrival"
            else:
                key = "window_missed"
            counts[key] += 1
            embryos[key][embryo] = embryos[key].get(embryo, 0) + 1
            pair = f"{true.value}→{pred.value}"
            pairs[key][pair] = pairs[key].get(pair, 0) + 1
            if is_detail:
                examples[key].append({"e": embryo, "t": t})

    if n_fail == 0:
        return None

    def spread(rows: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
        """Up to k examples, cycling embryos so one embryo doesn't take every slot."""
        by_embryo: dict[str, list[dict[str, Any]]] = {}
        for r in sorted(rows, key=lambda r: (r["e"], r["t"])):
            by_embryo.setdefault(r["e"], []).append(r)
        picked: list[dict[str, Any]] = []
        while len(picked) < k and any(by_embryo.values()):
            for e in sorted(by_embryo):
                if by_embryo[e] and len(picked) < k:
                    picked.append(by_embryo[e].pop(0))
        return picked

    lags.sort()
    return {
        "n_pred": n_pred,
        "n_fail": n_fail,
        "behind": counts["late_arrival"] + counts["window_missed"],
        "ahead": counts["ahead"],
        "windows_missed": windows_missed,
        "windows_total": windows_total,
        "lag_median": lags[len(lags) // 2] if lags else 0,
        "lag_max": lags[-1] if lags else 0,
        "clusters": [
            {
                "key": k,
                "count": counts[k],
                "embryos": embryos[k],
                "pairs": sorted(pairs[k].items(), key=lambda kv: -kv[1]),
                "examples": spread(examples[k]),
            }
            for k in keys
        ],
    }


def _embryo_span(ids: list[str]) -> str:
    """['embryo_5', ..., 'embryo_8'] → 'e5–8'; non-contiguous ids are listed."""
    nums = sorted(int(i.rsplit("_", 1)[-1]) for i in ids if i.rsplit("_", 1)[-1].isdigit())
    if not nums:
        return ""
    if nums == list(range(nums[0], nums[-1] + 1)) and len(nums) > 1:
        return f"e{nums[0]}–{nums[-1]}"
    return "e" + ",".join(str(n) for n in nums)


def _week_prefix(ts: str) -> str:
    """'20260609-...' → '[Week of 6/8]' (the Monday of that run's week)."""
    try:
        run_day = datetime.strptime(ts[:8], "%Y%m%d").date()
    except ValueError:
        return ""
    monday = run_day - timedelta(days=run_day.weekday())
    return f"[Week of {monday.month}/{monday.day}]"


def _run_label(d: Path, gt_embryos: dict[str, str]) -> str:
    """Human-readable dropdown label: week group, what the change was, dataset, date."""
    solver, _model, ts = d.parts[-3:]
    desc = solver
    try:
        mod = importlib.import_module(f"harness.solvers.{solver}")
        desc = (mod.__doc__ or solver).strip().splitlines()[0].rstrip(".")
    except Exception:
        pass  # deleted/renamed solver — fall back to its directory name
    span = ""
    try:
        start = json.loads((d / "seed0" / "events.jsonl").open().readline())
        span = gt_embryos.get(start["payload"]["gt_sha"], "")
    except Exception:
        pass
    date = f"{ts[4:6]}/{ts[6:8]}" if len(ts) >= 8 else ts
    body = " · ".join(x for x in (desc, span, date) if x)
    week = _week_prefix(ts)
    return f"{week} {body}" if week else body


def _experiment_description(solver_name: str) -> dict[str, Any]:
    """The solver docstring's full pre-RESULT content — what was tried and why.

    Returns {"paras": [...], "setup": "..."}. RESULT paragraphs are excluded;
    the report's own numbers speak for the outcome.
    """
    try:
        mod = importlib.import_module(f"harness.solvers.{solver_name}")
    except Exception:
        return {"paras": [], "setup": ""}
    paras: list[str] = []
    for block in (mod.__doc__ or "").strip().split("\n\n"):
        if block.strip().startswith("RESULT"):
            break
        paras.append(" ".join(line.strip() for line in block.splitlines()))
    setup = ""
    solver = getattr(mod, "SOLVER", None)
    if solver is not None:
        tools = ", ".join(solver.tools) if solver.tools else "none"
        mode = "one-shot" if solver.max_steps == 1 else f"agentic, {solver.max_steps} steps max"
        setup = f"solver {solver.name} · tools: {tools} · {mode}"
    return {"paras": paras, "setup": setup}


def _sibling_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Every run dir under runs/ with a report.html, for the run-switcher dropdown.

    Hrefs are relative to `run_dir` so the links work from file:// and any
    static server rooted at runs/. The current run is always included even
    though its report is still being written.
    """
    run_dir = run_dir.resolve()
    runs_root = _REPO_ROOT / "runs"
    gt_embryos = {
        GroundTruth.from_json(p).sha(): _embryo_span(sorted(json.load(p.open())["transitions"]))
        for p in (_REPO_ROOT / "data" / "ground_truth").glob("*.json")
    }
    dirs = {p.parent for p in runs_root.glob("*/*/*/report.html")} | {run_dir}
    out: list[dict[str, Any]] = []
    for d in sorted(dirs, key=lambda p: p.parts[-1], reverse=True):  # newest first → weeks cluster
        out.append(
            {
                "label": _run_label(d, gt_embryos),
                "path": "/".join(d.relative_to(runs_root).parts),
                "href": os.path.relpath(d / "report.html", run_dir),
                "current": d == run_dir,
            }
        )
    return out


def generate(
    run_dir: Path,
    *,
    seed: int = 0,
    compare_dir: Path | None = None,
    gt_path: Path | None = None,
    volumes_dir: Path | None = None,
) -> Path:
    """Build report.html + report_assets/ inside run_dir. Returns the report path."""
    run_dir = Path(run_dir)
    gt = GroundTruth.from_json(gt_path or (_REPO_ROOT / "data" / "ground_truth" / "59799c78.json"))
    events_path = run_dir / f"seed{seed}" / "events.jsonl"

    frames = _frames_from_events(events_path, gt, seed_rel=f"seed{seed}")
    config = _run_config(events_path)
    vols = volumes_dir or (_REPO_ROOT / "data" / "volumes")
    _thumbnails(frames, run_dir / "report_assets", vols)
    _rotated_thumbnails(frames, run_dir / "report_assets", vols, str(config.get("solver", "")))
    _view3d_step_assets(frames, run_dir, vols)

    data: dict[str, Any] = {
        "config": config,
        "run_dir": str(run_dir),
        "experiment": _experiment_description(str(config.get("solver", ""))),
        "runs": _sibling_runs(run_dir),
        "seed": seed,
        "summary": _summary(run_dir, gt),
        "clusters": _failure_clusters(run_dir, gt, detail_seed=seed),
        "frames": frames,
        "transitions": {e: {s.value: t for s, t in m.items()} for e, m in gt.transitions.items()},
        "stages": [s.value for s in STAGE_ORDER],
        "compare": None,
    }
    if compare_dir is not None:
        other = Path(compare_dir) / f"seed{seed}" / "events.jsonl"
        data["compare"] = {"label": str(compare_dir), **_compare(frames, other)}

    blob = json.dumps(data, default=str).replace("</", "<\\/")
    out = run_dir / "report.html"
    out.write_text(_TEMPLATE.replace("__DATA_JSON__", blob), encoding="utf-8")
    return out


# --- Template ---------------------------------------------------------------
# Deep-neutral dark (fluorescence images are bright-on-black; a light page would
# glare against 770 black tiles). Framed sections, mono numerals, muted
# desaturated stage accents, uppercase letter-spaced section labels.

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>run report</title>
<style>
:root{
  --bg:#15171c; --panel:#1c1f26; --panel2:#22262f; --line:#30353f;
  --txt:#c9cdd6; --dim:#7d8490; --bright:#eef0f4;
  --ok:#7fb685; --bad:#c97b7b; --adj:#c9a86a; --accent:#8aa3c4;
  --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
  --sans:-apple-system,'Segoe UI',system-ui,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;padding:28px 32px 80px;background:var(--bg);color:var(--txt);font:14px/1.55 var(--sans)}
h1{font:600 17px/1.3 var(--mono);color:var(--bright);margin:0 0 4px;letter-spacing:.02em}
.sub{color:var(--dim);font:12px var(--mono);margin-bottom:28px}
section{border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:20px 22px;margin-bottom:22px}
section>h2{margin:0 0 16px;font:600 11px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
.kpis{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}
.kpi{border:1px solid var(--line);border-radius:6px;background:var(--panel2);padding:12px 18px;min-width:130px}
.kpi b{display:block;font:600 22px var(--mono);color:var(--bright)}
.kpi span{font:11px var(--mono);color:var(--dim);letter-spacing:.06em;text-transform:uppercase}
.kpi.hard b{color:var(--accent)}
.bars{display:grid;grid-template-columns:90px 1fr 110px;gap:6px 12px;align-items:center;font:12px var(--mono)}
.bar{height:14px;background:var(--panel2);border:1px solid var(--line);border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent);opacity:.75}
.bars .n{color:var(--dim);text-align:right}
table.cm{border-collapse:collapse;font:11px var(--mono);margin-top:6px}
table.cm th,table.cm td{border:1px solid var(--line);padding:4px 9px;text-align:right;min-width:46px}
table.cm th{color:var(--dim);font-weight:500}
table.cm td.diag{color:var(--ok)}
table.cm td.off{color:var(--bad)}
table.cm td.zero{color:#3a3f4a}
.strip{margin-bottom:14px}
.strip .lbl{font:12px var(--mono);color:var(--dim);margin-bottom:5px;display:flex;justify-content:space-between}
.cells{display:flex;height:26px;border:1px solid var(--line);border-radius:4px;overflow:hidden}
.cell{flex:1;cursor:pointer;position:relative;min-width:1px}
.cell.ok{background:#2a4030}.cell.adj{background:#4a3d24}.cell.bad{background:#4d2b2b}.cell.na{background:#23262d}
.cell:hover{outline:2px solid var(--bright);outline-offset:-2px;z-index:2}
.cell.tr::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--bright);opacity:.55}
.legend{font:11px var(--mono);color:var(--dim);display:flex;gap:18px;margin-top:4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.filters{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;align-items:center}
.filters select{background:var(--panel2);color:var(--txt);border:1px solid var(--line);border-radius:5px;padding:5px 9px;font:12px var(--mono)}
.filters .count{margin-left:auto;font:12px var(--mono);color:var(--dim)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
.card{border:1px solid var(--line);border-radius:7px;background:var(--panel2);padding:13px 15px;display:flex;gap:14px;cursor:pointer}
.card:hover{border-color:var(--accent)}
.card img{width:108px;height:auto;align-self:flex-start;border-radius:4px;background:#000}
.card .meta{flex:1;min-width:0}
.card .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.card .id{font:12px var(--mono);color:var(--dim)}
.card .vs{font:12px var(--mono);margin:6px 0;display:flex;gap:8px;justify-content:flex-end}
.tag{display:inline-block;padding:2px 9px;border-radius:10px;font:11px var(--mono);border:1px solid var(--line)}
.tag.pred{color:var(--bad);border-color:#5a3a3a}
.tag.gt{color:var(--ok);border-color:#3a5a40}
.tag.win{color:var(--ok);border-color:#3a5a40}
.card .why{font:12px/1.5 var(--sans);color:var(--dim);display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.clNote{font:12px var(--mono);color:var(--dim);margin:2px 0 14px}
.clGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.clCard{border:1px solid var(--line);border-radius:7px;background:var(--panel2);padding:14px 16px;display:flex;flex-direction:column;gap:9px}
.clHead{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.clHead h3{margin:0;font:600 13px var(--sans);color:var(--bright)}
.clCount{font:600 12px var(--mono);color:var(--accent);white-space:nowrap}
.clCount b{font-size:15px}
.clChips{display:flex;flex-wrap:wrap;gap:6px}
.clChip{font:11px var(--mono);color:var(--txt);border:1px solid var(--line);border-radius:10px;padding:1px 8px}
.clPairs{font:11px var(--mono);color:var(--dim)}
.clPairs b{color:var(--txt)}
.clDef{font:12.5px/1.55 var(--sans);color:var(--txt);margin:0}
.clThumbs{display:flex;gap:10px;margin-top:auto}
.clThumb{margin:0;cursor:pointer;flex:1;min-width:0}
.clThumb img{width:100%;border-radius:4px;display:block;border:1px solid var(--line);background:#000}
.clThumb:hover img{border-color:var(--accent)}
.clThumb figcaption{font:10.5px var(--mono);color:var(--dim);margin-top:4px;text-align:center}
.modal{display:none;position:fixed;inset:0;background:rgba(10,11,14,.92);z-index:50;overflow-y:auto;padding:36px 5vw}
.modal.on{display:block}
.modal .box{max-width:880px;margin:0 auto;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:24px 28px}
.modal .head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px}
.modal .head .id{font:600 14px var(--mono);color:var(--bright)}
.modal .head .nav{font:12px var(--mono);color:var(--dim)}
.modal img.main{max-width:100%;border-radius:5px;background:#000;display:block;margin:0 auto 16px}
.rotRow{display:flex;gap:10px;margin:0 0 16px}
.rotRow figure{flex:1;margin:0}
.rotRow img{width:100%;border-radius:5px;background:#000;display:block}
.rotRow figcaption{font:10px var(--mono);color:var(--dim);letter-spacing:.08em;text-align:center;margin-top:5px;text-transform:uppercase}
.navHdr{font:11px var(--mono);color:var(--dim);letter-spacing:.14em;text-transform:uppercase;margin:18px 0 10px}
.navStrip{display:flex;gap:12px;overflow-x:auto;padding-bottom:8px;margin-bottom:6px}
.navStrip .step{flex:0 0 auto;width:300px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);padding:10px 12px;font:11px/1.5 var(--mono)}
.step .sn{display:inline-block;min-width:18px;height:18px;line-height:18px;text-align:center;background:var(--accent);color:#000;border-radius:9px;font-weight:700;margin-right:8px}
.step .cap{color:var(--bright)}
.step img{width:100%;display:block;margin-top:9px;border-radius:5px;background:#000}
.step .val{color:var(--bright);font-weight:600}
.step .err{color:var(--bad)}
.reason{font:13px/1.6 var(--sans);color:var(--txt);background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:13px 16px;white-space:pre-wrap}
.note{font:12px var(--mono);color:var(--adj);margin-top:10px}
.cmp h3{font:600 12px var(--mono);letter-spacing:.1em;text-transform:uppercase;margin:18px 0 10px}
.cmp h3.w{color:var(--ok)}.cmp h3.r{color:var(--bad)}.cmp h3.c{color:var(--adj)}
.close{position:fixed;top:18px;right:26px;font:300 30px var(--sans);color:var(--dim);cursor:pointer;z-index:51}
.close:hover{color:var(--bright)}
.hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;flex-wrap:wrap}
.runSel{display:flex;align-items:center;gap:10px;font:11px var(--mono);color:var(--dim);letter-spacing:.14em;text-transform:uppercase}
.runSel select{background:var(--panel2);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:7px 11px;font:12px var(--mono);max-width:380px;cursor:pointer}
.runSel select:hover{border-color:var(--accent)}
.expDesc{margin:18px 0 0;padding:16px 20px;background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;font:13px/1.7 var(--sans);color:var(--txt);max-width:980px}
.expDesc::before{content:"experiment";display:block;font:11px var(--mono);color:var(--dim);letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px}
.expDesc p{margin:0 0 10px}
.expDesc p:last-of-type{margin-bottom:0}
.expSetup{margin-top:12px;padding-top:10px;border-top:1px dashed var(--line);font:11px var(--mono);color:var(--dim);letter-spacing:.04em}
</style>
</head>
<body>
<div class="hdr">
  <div><h1 id="title"></h1><div class="sub" id="subtitle"></div></div>
  <label class="runSel" id="runSelWrap">run <select id="runSel"></select></label>
</div>
<div class="expDesc" id="expDesc" hidden></div>
<section><h2>Summary</h2><div class="kpis" id="kpis"></div>
  <div style="display:flex;gap:40px;flex-wrap:wrap"><div style="flex:1;min-width:300px"><div class="bars" id="bars"></div></div>
  <div><div style="font:11px var(--mono);color:var(--dim);letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">Confusion (true ↓ / predicted →)</div><div id="cm"></div></div></div>
</section>
<section id="clustersSec" style="display:none"><h2>Failure clusters</h2>
  <div class="kpis" id="clKpis"></div>
  <div class="clNote" id="clNote"></div>
  <div class="clGrid" id="clGrid"></div>
</section>
<section><h2>Filmstrip</h2><div id="strips"></div>
  <div class="legend"><span><i style="background:#2a4030"></i>correct</span><span><i style="background:#4a3d24"></i>adjacent</span><span><i style="background:#4d2b2b"></i>wrong</span><span><i style="background:#23262d"></i>unscored</span><span>│ GT transition</span></div>
</section>
<section id="cmpSection" style="display:none" class="cmp"><h2>Comparison</h2><div id="cmpBody"></div></section>
<section><h2>Errors</h2>
  <div class="filters">
    <select id="fStage"><option value="">all GT stages</option></select>
    <select id="fPred"><option value="">all predictions</option></select>
    <select id="fEmbryo"><option value="">all embryos</option></select>
    <span class="count" id="errCount"></span>
  </div>
  <div class="cards" id="errCards"></div>
</section>
<div class="close" id="closeBtn" style="display:none">×</div>
<div class="modal" id="modal"><div class="box" id="modalBox"></div></div>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const F=D.frames, byKey={}; F.forEach((f,i)=>byKey[f.e+'|'+f.t]=i);
const pct=x=>x==null?'–':(100*x).toFixed(1)+'%';
const esc=s=>(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');

// header
document.getElementById('title').textContent=(D.config.solver||'run')+' · '+(D.config.model||'');
document.getElementById('subtitle').textContent=D.run_dir+'  ·  seed'+D.seed+' detail  ·  '+D.summary.n_seeds+' seed(s) aggregated';

// experiment description
if(D.experiment&&D.experiment.paras&&D.experiment.paras.length){
  const ed=document.getElementById('expDesc');
  for(const p of D.experiment.paras){const el=document.createElement('p');el.textContent=p;ed.appendChild(el);}
  if(D.experiment.setup){const su=document.createElement('div');su.className='expSetup';su.textContent=D.experiment.setup;ed.appendChild(su);}
  ed.hidden=false;
}

// run switcher
const runSel=document.getElementById('runSel');
(D.runs||[]).forEach(r=>{const o=document.createElement('option');o.value=r.href;o.textContent=r.label;o.title=r.path||'';o.selected=!!r.current;runSel.appendChild(o)});
if((D.runs||[]).length<2)document.getElementById('runSelWrap').style.display='none';
runSel.addEventListener('change',()=>{location.href=runSel.value});

// KPIs
const S=D.summary, H=S.hard||{};
document.getElementById('kpis').innerHTML=
  kpi(pct(S.accuracy_mean)+(S.n_seeds>1?' ±'+(100*S.accuracy_std).toFixed(1):''),'exact · n='+S.n)+
  kpi(pct(S.adjacent_mean),'adjacent')+
  (H.n?kpi(pct(H.accuracy_mean),'hard stages · n='+H.n,'hard'):'')+
  kpi(F.filter(f=>f.ok===false).length,'errors (seed'+D.seed+')')+
  (Object.keys(S.tool_calls||{}).length?kpi(Object.values(S.tool_calls).reduce((a,b)=>a+b,0),'tool calls'):'')+
  (S.n_errors?kpi(S.n_errors,'frame failures'):'');
function kpi(v,l,c){return '<div class="kpi '+(c||'')+'"><b>'+v+'</b><span>'+l+'</span></div>'}

// per-stage bars
document.getElementById('bars').innerHTML=D.stages.filter(s=>S.per_stage[s]).map(s=>{
  const [m,sd,k]=S.per_stage[s];
  return '<div>'+s+'</div><div class="bar"><i style="width:'+(100*m)+'%"></i></div><div class="n">'+pct(m)+(S.n_seeds>1?' ±'+(100*sd).toFixed(0):'')+'</div>';
}).join('');

// confusion matrix
(()=>{const seen=new Set();Object.keys(S.confusion).forEach(k=>{const[a,b]=k.split('|');seen.add(a);seen.add(b)});
const st=D.stages.filter(s=>seen.has(s));if(!st.length)return;
let h='<table class="cm"><tr><th></th>'+st.map(s=>'<th>'+s.slice(0,7)+'</th>').join('')+'</tr>';
st.forEach(a=>{h+='<tr><th>'+a+'</th>'+st.map(b=>{const n=S.confusion[a+'|'+b]||0;
return '<td class="'+(n===0?'zero':a===b?'diag':'off')+'">'+(n||'·')+'</td>'}).join('')+'</tr>'});
document.getElementById('cm').innerHTML=h+'</table>'})();

// failure clusters
(()=>{const C=D.clusters;if(!C||!C.n_fail)return;
document.getElementById('clustersSec').style.display='';
document.getElementById('clKpis').innerHTML=
  kpi(C.n_fail+' / '+C.n_pred,'wrong predictions · all seeds')+
  kpi(C.behind+' · '+Math.round(100*C.behind/C.n_fail)+'%','earlier than gt')+
  kpi(C.ahead+' · '+Math.round(100*C.ahead/C.n_fail)+'%','later than gt')+
  kpi(C.windows_missed+' / '+C.windows_total,'gt windows never entered');
document.getElementById('clNote').textContent='grouped by behaviour across all seeds · examples from seed'+D.seed+' · click an example to open its trajectory';
const labels={late_arrival:'Reaches the stage late',window_missed:'Stage window missed outright',ahead:'Ahead of the annotation'};
const defs={late_arrival:'The model does predict the gt stage inside its window, just late — median lag '+C.lag_median+' frames (max '+C.lag_max+') after the gt boundary.',
  window_missed:'The gt stage window passes without the model ever predicting that stage.',
  ahead:'The model predicts a stage later than the annotation.'};
document.getElementById('clGrid').innerHTML=C.clusters.filter(c=>c.count).map(c=>{
  const chips=Object.entries(c.embryos).sort((a,b)=>b[1]-a[1]).map(([e,n])=>'<span class="clChip">'+esc(e.replace('embryo_','e'))+' · '+n+'</span>').join('');
  const pr=c.pairs.slice(0,3).map(p=>esc(p[0])+' <b>'+p[1]+'</b>').join(' &nbsp;·&nbsp; ');
  const th=(c.examples||[]).map(x=>{const i=byKey[x.e+'|'+x.t];if(i==null)return '';const f=F[i];
    return '<figure class="clThumb" data-k="'+f.e+'|'+f.t+'">'+(f.thumb?'<img loading="lazy" src="'+f.thumb+'">':'')+
      '<figcaption>'+esc(f.e.replace('embryo_','e'))+' T'+f.t+' · gt '+esc(f.gt)+' · pred '+esc(f.pred)+'</figcaption></figure>';}).join('');
  return '<div class="clCard"><div class="clHead"><h3>'+labels[c.key]+'</h3><span class="clCount"><b>'+c.count+'</b> · '+Math.round(100*c.count/C.n_fail)+'%</span></div>'+
    '<div class="clChips">'+chips+'</div><div class="clPairs">'+pr+'</div><p class="clDef">'+defs[c.key]+'</p>'+
    (th?'<div class="clThumbs">'+th+'</div>':'')+'</div>';
}).join('')})();

// filmstrip
(()=>{const emb={};F.forEach(f=>{(emb[f.e]=emb[f.e]||[]).push(f)});
document.getElementById('strips').innerHTML=Object.keys(emb).sort().map(e=>{
  const fs=emb[e],tr=new Set(Object.values(D.transitions[e]||{}));
  const nOk=fs.filter(f=>f.ok).length;
  return '<div class="strip"><div class="lbl"><span>'+e+'</span><span>'+nOk+'/'+fs.length+' correct</span></div><div class="cells">'+
    fs.map(f=>'<div class="cell '+(f.ok?'ok':f.ok===false?(f.adj?'adj':'bad'):'na')+(tr.has(f.t)?' tr':'')+'" data-k="'+f.e+'|'+f.t+'" title="T'+f.t+': '+f.pred+' (GT '+f.gt+')"></div>').join('')+'</div></div>';
}).join('')})();

// error cards
const sel={stage:'',pred:'',embryo:''};
function card(f,extra){return '<div class="card" data-k="'+f.e+'|'+f.t+'">'+
  (f.thumb?'<img loading="lazy" src="'+f.thumb+'">':'')+
  '<div class="meta"><div class="top"><span class="id">'+f.e+' · T'+f.t+'</span>'+(f.steps&&f.steps.length?'<span class="id">'+f.steps.length+' tool call'+(f.steps.length>1?'s':'')+'</span>':'')+'</div>'+
  (extra||'<div class="vs"><span class="tag pred">pred '+f.pred+'</span><span class="tag gt">gt '+f.gt+'</span></div>')+
  '<div class="why">'+esc(f.reasoning)+'</div></div></div>'}
function renderErrors(){
  const errs=F.filter(f=>f.ok===false&&(!sel.stage||f.gt===sel.stage)&&(!sel.pred||f.pred===sel.pred)&&(!sel.embryo||f.e===sel.embryo));
  document.getElementById('errCards').innerHTML=errs.map(f=>card(f)).join('');
  document.getElementById('errCount').textContent=errs.length+' shown';
}
(()=>{const errs=F.filter(f=>f.ok===false);
fill('fStage',[...new Set(errs.map(f=>f.gt))],v=>{sel.stage=v});
fill('fPred',[...new Set(errs.map(f=>f.pred))],v=>{sel.pred=v});
fill('fEmbryo',[...new Set(errs.map(f=>f.e))],v=>{sel.embryo=v});
function fill(id,vals,set){const el=document.getElementById(id);
D.stages.filter(s=>vals.includes(s)).concat(vals.filter(v=>!D.stages.includes(v))).forEach(v=>{const o=document.createElement('option');o.value=o.textContent=v;el.appendChild(o)});
el.onchange=()=>{set(el.value);renderErrors()}}
renderErrors()})();

// comparison
if(D.compare){const C=D.compare;document.getElementById('cmpSection').style.display='block';
const sec=(t,cls,rows,fmt)=>rows.length?'<h3 class="'+cls+'">'+t+' ('+rows.length+')</h3><div class="cards">'+rows.map(r=>{
  const f=F[byKey[r.e+'|'+r.t]]||r;
  return card(f,'<div class="vs"><span class="tag '+(fmt==='w'?'win':'pred')+'">this: '+r.a+'</span><span class="tag '+(fmt==='r'?'win':'pred')+'">other: '+r.b+'</span><span class="tag gt">gt '+r.gt+'</span></div>')
}).join('')+'</div>':'';
document.getElementById('cmpBody').innerHTML=
  '<div class="sub" style="margin-bottom:0">vs '+C.label+' · net '+(C.wins.length-C.regressions.length>=0?'+':'')+(C.wins.length-C.regressions.length)+' frames</div>'+
  sec('Wins — wrong there, right here','w',C.wins,'w')+sec('Regressions — right there, wrong here','r',C.regressions,'r')+sec('Changed, still wrong','c',C.changed,'c')}

// trajectory modal
let cur=-1;
function open(i){cur=i;const f=F[i];
let h='<div class="head"><span class="id">'+f.e+' · T'+f.t+'</span><span class="nav">'+(i+1)+' / '+F.length+' · ←→ navigate · esc close</span></div>';
if(f.thumb)h+='<img class="main" src="'+esc(f.thumb)+'">';
if(f.rot&&f.rot.length)h+='<div class="rotRow">'+f.rot.map(r=>'<figure><img loading="lazy" src="'+esc(r.src)+'"><figcaption>rotated '+r.a+'°</figcaption></figure>').join('')+'</div>';
h+='<div class="vs" style="justify-content:flex-end;margin-bottom:14px"><span class="tag '+(f.ok?'gt':'pred')+'">pred '+f.pred+'</span><span class="tag gt">gt '+f.gt+'</span><span class="tag">'+(f.tokens||0)+' tok</span></div>';
function stepCap(s){
  if(s.name==='view3d'){const p=s.params||{};let c='rotate → yaw '+(p.yaw_deg??0)+'° · pitch '+(p.pitch_deg??0)+'°';
    if(p.threshold!==undefined&&p.threshold!==30)c+=' · threshold '+p.threshold;
    if(p.zoom_pct&&p.zoom_pct>100)c+=' · zoom '+p.zoom_pct+'% @ ('+(p.center_x_pct??50)+'%, '+(p.center_y_pct??50)+'%)';
    return c;}
  return s.name+'('+Object.entries(s.params||{}).map(([k,v])=>k+'='+v).join(', ')+')';}
if((f.steps||[]).length){
  h+='<div class="navHdr">'+( (f.steps.some(s=>s.name==='view3d'))?'3D navigation — how the model explored this frame':'tool calls')+'</div><div class="navStrip">';
  f.steps.forEach((s,j)=>{h+='<div class="step"><span class="sn">'+(j+1)+'</span><span class="cap">'+esc(stepCap(s))+'</span>'+
    (s.error?'<div class="err">'+esc(s.error)+'</div>':'')+
    (s.value!==undefined?'<div>→ <span class="val">'+s.value+'</span> <span style="color:var(--dim)">'+esc(s.note)+'</span></div>':'')+
    (s.img?'<img loading="lazy" src="'+esc(s.img)+'">':'')+'</div>'});
  h+='</div>';}
h+='<div class="navHdr">model response</div><div class="reason">'+esc(f.reasoning||f.err||'(no reasoning)')+'</div>';
if(f.override)h+='<div class="note">⚠ verify override: '+esc(f.override)+'</div>';
if(f.budget_exhausted)h+='<div class="note">⚠ tool budget exhausted — classify was forced</div>';
document.getElementById('modalBox').innerHTML=h;
document.getElementById('modal').classList.add('on');document.getElementById('closeBtn').style.display='block'}
function close(){document.getElementById('modal').classList.remove('on');document.getElementById('closeBtn').style.display='none';cur=-1}
document.addEventListener('click',ev=>{const k=ev.target.closest('[data-k]');if(k){open(byKey[k.dataset.k]);return}
if(ev.target.id==='modal'||ev.target.id==='closeBtn')close()});
document.addEventListener('keydown',ev=>{if(cur<0)return;
if(ev.key==='Escape')close();else if(ev.key==='ArrowRight'&&cur<F.length-1)open(cur+1);else if(ev.key==='ArrowLeft'&&cur>0)open(cur-1)});
</script>
</body>
</html>
"""
