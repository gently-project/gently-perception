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
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from harness.core.render import cached_render
from harness.core.types import STAGE_ORDER, Stage
from harness.eval import report as text_report
from harness.eval.score import score_run
from harness.io.events import read_events
from harness.io.ground_truth import GroundTruth
from harness.io.volumes import OfflineSource

_REPO_ROOT = Path(__file__).resolve().parents[2]
THUMB_LONG_EDGE = 260


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
        thumb = assets_dir / f"{r['e']}_T{r['t']:03d}.jpg"
        r["thumb"] = f"report_assets/{thumb.name}"
        if thumb.exists():
            continue
        full = Image.open(io.BytesIO(base64.b64decode(cached_render(paths[key]))))
        scale = THUMB_LONG_EDGE / max(full.size)
        if scale < 1.0:
            full = full.resize((int(full.width * scale), int(full.height * scale)), Image.Resampling.LANCZOS)
        full.save(thumb, format="JPEG", quality=80)


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
    _thumbnails(frames, run_dir / "report_assets", volumes_dir or (_REPO_ROOT / "data" / "volumes"))

    data: dict[str, Any] = {
        "config": _run_config(events_path),
        "run_dir": str(run_dir),
        "seed": seed,
        "summary": _summary(run_dir, gt),
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
.modal{display:none;position:fixed;inset:0;background:rgba(10,11,14,.92);z-index:50;overflow-y:auto;padding:36px 5vw}
.modal.on{display:block}
.modal .box{max-width:880px;margin:0 auto;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:24px 28px}
.modal .head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px}
.modal .head .id{font:600 14px var(--mono);color:var(--bright)}
.modal .head .nav{font:12px var(--mono);color:var(--dim)}
.modal img.main{max-width:100%;border-radius:5px;background:#000;display:block;margin:0 auto 16px}
.step{border:1px solid var(--line);border-radius:6px;background:var(--panel2);padding:11px 14px;margin-bottom:10px;font:12px var(--mono)}
.step .sn{color:var(--accent);margin-right:8px}
.step img{max-width:340px;display:block;margin-top:9px;border-radius:4px;background:#000}
.step .val{color:var(--bright);font-weight:600}
.step .err{color:var(--bad)}
.reason{font:13px/1.6 var(--sans);color:var(--txt);background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:13px 16px;white-space:pre-wrap}
.note{font:12px var(--mono);color:var(--adj);margin-top:10px}
.cmp h3{font:600 12px var(--mono);letter-spacing:.1em;text-transform:uppercase;margin:18px 0 10px}
.cmp h3.w{color:var(--ok)}.cmp h3.r{color:var(--bad)}.cmp h3.c{color:var(--adj)}
.close{position:fixed;top:18px;right:26px;font:300 30px var(--sans);color:var(--dim);cursor:pointer;z-index:51}
.close:hover{color:var(--bright)}
</style>
</head>
<body>
<h1 id="title"></h1>
<div class="sub" id="subtitle"></div>
<section><h2>Summary</h2><div class="kpis" id="kpis"></div>
  <div style="display:flex;gap:40px;flex-wrap:wrap"><div style="flex:1;min-width:300px"><div class="bars" id="bars"></div></div>
  <div><div style="font:11px var(--mono);color:var(--dim);letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">Confusion (true ↓ / predicted →)</div><div id="cm"></div></div></div>
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
if(f.thumb)h+='<img class="main" src="'+f.thumb+'">';
h+='<div class="vs" style="justify-content:flex-end;margin-bottom:14px"><span class="tag '+(f.ok?'gt':'pred')+'">pred '+f.pred+'</span><span class="tag gt">gt '+f.gt+'</span><span class="tag">'+(f.tokens||0)+' tok</span></div>';
(f.steps||[]).forEach((s,j)=>{h+='<div class="step"><span class="sn">'+(j+1)+'</span>'+s.name+'('+esc(JSON.stringify(s.params))+')'+
  (s.error?' → <span class="err">'+esc(s.error)+'</span>':'')+
  (s.value!==undefined?' → <span class="val">'+s.value+'</span> <span style="color:var(--dim)">'+esc(s.note)+'</span>':'')+
  (s.img?'<img loading="lazy" src="'+s.img+'">':'')+'</div>'});
h+='<div class="reason">'+esc(f.reasoning||f.err||'(no reasoning)')+'</div>';
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
