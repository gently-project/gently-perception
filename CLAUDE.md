# gently-perception

Autonomous perception experiment framework. You are a coding agent developing
VLM-based classification functions for C. elegans embryo stage classification.

## Your Task

Improve classification accuracy by modifying perception functions in `perception/`.
Read `program.md` for full context on what's been tried and what to explore.

## Iteration Loop

You operate autonomously in a think → code → test → analyze cycle:

1. **Think**: Read existing results in `data/results/`, identify the worst-performing stages, form a hypothesis for improvement
2. **Code**: Create or modify a perception function in `perception/`, register it in `perception/__init__.py`
3. **Test**: Run `python run.py --variant your_variant --stages pretzel 2fold 1.5fold --force` to evaluate on the hard stages
4. **Analyze**: Read the output, compare to baselines, understand failure patterns
5. **Repeat**: Refine based on what you learned

## Key Commands

```bash
# Evaluate on hard stages only (fast — ~3 min)
python run.py --variant my_variant --stages pretzel 2fold 1.5fold --force

# Quick full eval (30 timepoints, mostly early stage — less useful)
python run.py --variant my_variant --quick --force

# Full eval (all 769 timepoints — slow, ~1 hour)
python run.py --variant my_variant --force
```

## Model

Using `claude-opus-4-6` (set in `perception/_base.py`).

## Current Baselines (from Sonnet 4.5 — re-run with Opus for updated numbers)

| Variant | Exact | Pretzel (n=433) | 2fold (n=79) | 1.5fold (n=49) |
|---------|-------|-----------------|--------------|----------------|
| minimal | 48.5% | 29% | 82% | 12% |
| descriptive | 48.0% | 33% | 46% | 18% |

## Rules

- Only modify files in `perception/` — do not touch `benchmark/`, `run.py`, or `_base.py`
- Every perception function must have the standard signature (see `program.md`)
- Use `--force` to overwrite previous results
- Focus on the hard stages: pretzel, 1.5fold, comma
- Read `program.md` for failed experiments and promising directions

## Python Environment

Use the Python from the gently venv:
`C:\Users\christensenr\Documents\GitHub\gently\venv\Scripts\python.exe`
