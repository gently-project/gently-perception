# Embryo staging: experiment log & findings

Classifying C. elegans embryo developmental stages from light-sheet volumes,
evaluated on embryos 5–8 (session `2cfd8f4e`, 705 frames, ~3.7 min/frame),
3 seeds per experiment, claude-opus-4-6. Starting point: 62.2% exact. Current
champion: **hybrid_annot2 at 69.5% ± 0.2 exact / 87.8% adjacent** — all of the
gain from text rules, none of it from richer visual input.

## The ladder

| # | experiment | exact | adjacent | verdict |
|---|------------|-------|----------|---------|
| 0 | hybrid (baseline, May 28) | 62.2 ± 3.1 | 83.2 | starting point |
| 1 | hybrid_nodefer — remove defer-to-previous instructions | 64.2 ± 0.8 | 83.1 | **+2.0**, variance collapsed |
| 2 | hybrid_3dviews — static rotated MIPs (45/90/135°) | 61.9 ± 0.8 | 83.6 | −2.3: coils merge in projections, model undercounts folds |
| 3 | hybrid_3dviews2 — MIPs demoted to 1.5f/2f tie-breaker | 58.9 ± 3.8 | 82.7 | worse: views destabilize pretzel anyway |
| 4 | hybrid_annot — stage criteria rewritten from the annotator's notes (tail progress + 60-min timing rule) | 66.3 ± 0.2 | 87.0 | **+2.1**, gains exactly where aimed |
| 5 | hybrid_annot2 — late-pretzel vs hatching disambiguation, cadence-aware | **69.5 ± 0.2** | **87.8** | **+3.2**, killed the 99-frame hatching cascade; champion |
| 6 | hybrid_annotviews — static raymarched views at the annotator's poses | 66.5 ± 4.0 | 85.2 | −3.0: even faithful 3D renders dilute |
| 7 | hybrid_agentic3d — model rotates the 3D render itself (view3d tool) | 64.9 ± 5.3 | 82.0 | −4.6: renders re-litigated the settled hatching call |
| 8 | hybrid_explorer3d — full viewer autonomy (rotate + zoom + click) | 67.1 ± 1.1 | 85.9 | −2.4: best-behaved 3D variant, still loses |
| 9 | hybrid_contrast3d — explorer + contrastive boundary descriptions | 67.0 ± 1.6 | 86.4 | flat vs explorer; target stages unmoved |
| 10 | hybrid_pairwise — previous frame's image + comparison framing | 68.9 ± 1.3 | 87.0 | tie (paired Δ −0.5, CI [−1.6, +0.5]); lag unmoved |
| 11 | hybrid_prevonly — previous frame's image alone (ablation) | 63.7 ± 1.6 | 85.0 | **−5.7, CI [−7.2, −4.3]** — decisively harmful |

Reference points from embryos 1–4 (session `59799c78`): hybrid 86.2%,
agentic_baseline (all tools) 80.2% — embryos 5–8 are substantially harder,
and tools cost accuracy there too.

## Findings

### 1. Every win was a decision rule; every image-side change was flat or negative

The +7.3 points from baseline to champion decompose entirely into text:
removing the defer-to-previous anchor (+2.0), the annotator's tail-progress
criteria and his explicit 2fold→pretzel timing rule (+2.1), and the
late-pretzel vs hatching cadence rules (+3.2). Six experiments added visual
information — rotated MIPs, faithful raymarched views, model-driven 3D
navigation with zoom, the previous frame — and not one of them improved on
its text-only base. Agentic tool use carries a consistent ~2.4-point tax on
this task (explorer3d/contrast3d vs annot2), with tool selection itself
behaving exactly as prompted.

### 2. The model under-updates: transition lag is a decision behavior, not a perception gap

The dominant failure all along has been one-stage lag at boundaries (median
8–9 frames) plus brief stages missed outright (comma's window is ~5 frames;
it scores 0–5% everywhere). Three increasingly direct perception fixes failed
to move the lag at all: contrastive descriptions of each boundary's first
visible change (written from the e1–4 transition frames), full 3D viewing
autonomy, and finally showing the previous and current frames side by side —
lag median 8 vs 9, behind-failures 655 vs 641, identical clusters. The model
sees the change and does not act on it.

The prevonly ablation is the sharpest evidence: an **unguided** previous
frame made things much worse (−5.7), shifting every confusion laggier —
consecutive frames are ~95% identical pixels, and visual similarity reads as
"same stage" louder than any boundary cue. The comparison-first framing in
pairwise repaired ~5 of those 6 points. Inputs that support persistence
amplify the bias; framings that force the update question neutralize it.

### 3. The labels themselves are partly time-derived

Kesavan's notes state outright that he placed the 2fold→pretzel boundary
with a stereotypic-timing rule ("~60 minutes after 2fold onset") because
coils are genuinely hard to count in projections. Appearance-only
classification therefore fights the ground truth at that boundary. Telling
the model this honestly — and converting all time rules to run-time-measured
cadence (frame interval from acquisition timestamps, so the same solver works
on any series) — produced the single largest win.

### 4. What the failure map looks like now (annot2)

~30% of predictions still miss. Composition: late arrival (~390/seed at its
peak, now the lag tail), brief windows missed (33 of 81 transition windows),
ahead-errors eliminated (99 → ~0–5 after the hatching rules). Per stage:
early/hatched ~100%, pretzel ~70, bean ~33, 2fold ~26, 1.5fold ~14, comma ~5.

### 5. Infrastructure lessons

Transient API errors (grammar-compilation 400s, 529s) killed four multi-hour
runs at top level — `harness/core/model.py` has no retry; seeds were
recovered with a resume script. The tool dispatch cache is content-addressed
on params but not on tool code version, so changing a tool's implementation
silently serves stale renders. Tool media files key on turn index, so
parallel calls in one turn overwrite each other (the report now re-renders
steps from recorded params). `view3d` post-processing (brightness stretch +
content crop) broke pixel-equivalence with the annotator viewer — replays are
faithful to what the model saw, but what the model saw was not what the
annotator saw.

## Suggested next steps

1. **Generalize the timing-window recipe to every boundary.** The one
   boundary with an elapsed-time rule and an explicit exception to the
   prefer-earlier bias (2fold→pretzel) is the one transition that improved.
   Derive per-stage duration windows from the e1–4 session, and at each
   boundary have the prompt switch from "prefer earlier" to "actively hunt
   the next stage's first visible change" once the window passes. This is the
   only intervention class with a positive track record, aimed at the
   dominant remaining failure.
2. **Anchor-free second opinion at suspected transitions.** When the timing
   window has passed, reclassify the frame blind (no history, no previous
   stage) and reconcile with the monotonic verifier. The bias is
   persistence; removing the anchor only where lag is likely keeps stability
   elsewhere.
3. **Cross-session validation.** annot2's rules were developed against
   embryos 5–8. Run it on embryos 1–4 to check nothing regressed vs the 86.2%
   baseline and that the cadence machinery generalizes (different session,
   same ~4-min cadence).
4. **If 3D gets another attempt: annotator-native rendering.** Rebuild
   view3d with no stretch/crop, the annotator's contrast knob, true camera
   zoom, and a code-versioned cache — then rerun explorer3d. Five negatives
   say don't expect much, but the current tool never showed the model what
   the human actually saw.
5. **Data asks for the annotator.** More notes of the embryo_5/6/8 kind —
   they bought +4 points. Specifically: per-boundary "what changed in this
   frame" notes, typical stage durations from his experience, and ideally a
   third annotated session for a held-out test set.
6. **Harness hardening.** Exponential-backoff retry around the model call in
   core; tool-code version in the dispatch cache key; per-call media
   filenames.
