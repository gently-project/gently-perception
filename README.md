# gently-perception

VLM-based perception for microscopy stage classification — an [autoresearch](https://github.com/karpathy/autoresearch)-style experiment framework.

A coding agent modifies perception functions, runs experiments against a fixed benchmark, and iterates to improve accuracy. The framework is inspired by Karpathy's autoresearch pattern: `program.md` instructs the agent, perception functions are the "model" the agent tunes, and the benchmark harness is fixed infrastructure. See `program.md` for the full agent instructions and experiment history.

## Quick Start

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd gently-perception
pip install -r requirements.txt

# 2. Download volume data from HuggingFace (~35 GB)
python setup_data.py

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-...

# 4. Run an experiment
python run.py --variant hybrid --stages pretzel 2fold 1.5fold --force
```

## Results

Accuracy on hard stages (1.5fold, 2fold, pretzel) using Claude Opus 4.6.
All variants achieve 100% adjacent accuracy (within 1 stage of ground truth).

| Variant | Exact | 1.5fold (n=49) | 2fold (n=79) | Pretzel (n=193) | Approach |
|---------|-------|----------------|--------------|-----------------|----------|
| **hybrid** | **83.2%** | 59% | 70% | 95% | Stage-adaptive prompt switching |
| scientific | 82.6% | 55% | 76% | 92% | Eggshell fill fraction + body segment counting |
| temporal | 81.0% | 63% | 58% | 95% | Soft temporal anchoring, "prefer earlier stage" |
| duration_aware | 81.3% | 65% | 77% | 87% | Duration-aware prior + confidence-gated transitions |
| unified | 78.8% | 53% | 67% | 90% | Merged temporal + scientific (single prompt) |
| ensemble | 79.8% | 57% | 59% | 94% | 3x majority vote with temperature=0.3 |
| compare | 52.0% | 63% | 57% | 47% | Previous timepoint image comparison |
| contrastive | — | 25% | 30% | — | Detailed transition descriptions (aborted) |
| minimal | 48.5%\* | 12% | 82% | 29% | Stage names only (Sonnet 4.5 baseline) |
| descriptive | 48.0%\* | 18% | 46% | 33% | Projection-grounded descriptions (Sonnet 4.5 baseline) |

\* Sonnet 4.5 baselines measured on all stages (n=769), not directly comparable.

### Key Findings

- **Temporal anchoring** ("prefer earlier stage when uncertain") was the single biggest prompt improvement
- **Eggshell fill fraction** is the most discriminative visual criterion for fold stages
- **Annotation quality matters**: the original ground truth was missing `hatched` transitions (see below), fixing this improved measured accuracy by +35pp
- **Ensemble/majority voting doesn't help**: boundary errors are systematic, not stochastic
- **Previous image comparison** helps 1.5fold but catastrophically hurts pretzel (embryo movement ≠ stage change)
- **Duration-aware priors** improve boundary accuracy but trade off pretzel retention

### Ground Truth Annotation Fix

The original annotations (`data/ground_truth/59799c78_original.json`) covered stages up through pretzel but did not include hatching/hatched transitions — the annotation task focused on the morphogenesis stages. Since the imaging continues well past hatching, ~240 post-hatching timepoints were implicitly labeled as pretzel.

We extended the annotations by identifying hatched transition timepoints using two independent VLM variants (temporal and scientific). Both consistently broke at the same exact timepoints across all embryos:

| Embryo | Hatched at | Pretzel duration |
|--------|-----------|-----------------|
| embryo_1 | T139 | 49 timepoints |
| embryo_2 | T123 | 43 timepoints |
| embryo_3 | T110 | 41 timepoints |
| embryo_4 | T157 | 60 timepoints |

The corrected annotations (`data/ground_truth/59799c78.json`) include these hatched transitions. Use `make_filmstrip.py` to generate a visual filmstrip viewer for manual verification of the boundaries.

## Key Commands

```bash
# Evaluate on hard stages only (fast, ~30 min)
python run.py --variant hybrid --stages pretzel 2fold 1.5fold --force

# Quick full eval (30 timepoints per embryo, ~5 min)
python run.py --variant hybrid --quick --force

# Full eval (all timepoints, ~1 hour)
python run.py --variant hybrid --force

# Generate filmstrip viewer for manual annotation review
python make_filmstrip.py
# Then open data/filmstrip/viewer.html
```

## Adding a New Variant

1. Create `perception/my_variant.py` with the standard signature:
   ```python
   async def perceive_my_variant(
       image_b64: str,
       references: dict[str, list[str]],
       history: list[dict],
       timepoint: int,
   ) -> PerceptionOutput:
   ```
2. Register it in `perception/__init__.py`
3. Run: `python run.py --variant my_variant --stages pretzel 2fold 1.5fold --force`

See `program.md` for detailed instructions, failed experiments, and promising directions.

## Project Structure

```
gently-perception/
├── perception/           # Perception functions (modify these)
│   ├── _base.py          # Shared utilities: API wrapper, parsing (fixed)
│   ├── hybrid.py         # Best variant: stage-adaptive prompt switching
│   ├── scientific.py     # Eggshell fill + body segment counting
│   ├── temporal.py       # Temporal anchoring + reference matching
│   ├── duration_aware.py # Duration-aware confidence gating
│   └── ...               # 10+ other variants
├── benchmark/            # Evaluation harness (fixed)
│   ├── testset.py        # Volume loading, 3-view projections
│   ├── ground_truth.py   # Stage transition annotations
│   └── metrics.py        # Accuracy computation
├── data/
│   ├── ground_truth/     # Stage annotations (JSON)
│   ├── examples/         # Reference images per stage
│   ├── volumes/          # 3D light-sheet data (downloaded via setup_data.py)
│   └── results/          # Experiment outputs (JSON + charts)
├── run.py                # Evaluation entry point
├── setup_data.py         # Download data from HuggingFace
├── make_filmstrip.py     # Generate HTML filmstrip viewer
├── program.md            # Agent instructions (autoresearch-style)
├── requirements.txt      # Python dependencies
└── CLAUDE.md             # Current state for coding agent
```

## Related

- [gently](https://github.com/shrofflab/gently) — the microscopy agent framework
- [autoresearch](https://github.com/karpathy/autoresearch) — inspiration for this framework pattern
