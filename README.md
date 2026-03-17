# gently-perception

VLM-based perception for microscopy stage classification — an autoresearch-style experiment framework.

A coding agent modifies perception functions, runs experiments against a fixed benchmark, and iterates to improve accuracy. AI developing AI perception.

## Quick Start

```bash
# 1. Symlink your volume data
ln -s /path/to/59799c78/volumes data/volumes

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-...

# 3. Run a quick experiment
python run.py --variant minimal --quick
```

## Structure

```
gently-perception/
├── program.md           # Instructions for the coding agent
├── run.py               # Evaluation entry point (fixed)
├── perception/          # Perception functions (agent modifies these)
│   ├── _base.py         # Shared utilities (fixed)
│   ├── minimal.py       # Baseline: stage names only
│   ├── descriptive.py   # Baseline: projection-grounded descriptions
│   └── ...              # New variants go here
├── benchmark/           # Evaluation harness (fixed)
│   ├── testset.py       # Volume loading + projection
│   ├── ground_truth.py  # Annotation parsing
│   └── metrics.py       # Accuracy, ECE, confusion matrices
├── data/
│   ├── ground_truth/    # Stage transition annotations
│   ├── examples/        # Reference stage images
│   ├── volumes/         # 3D microscopy data (symlink)
│   └── results/         # Experiment outputs
└── paper/               # Experiment write-ups
```

## How It Works

Like [autoresearch](https://github.com/karpathy/autoresearch), but for VLM perception:

| autoresearch | gently-perception |
|---|---|
| `train.py` (agent modifies) | `perception/*.py` |
| `prepare.py` (fixed) | `benchmark/`, `_base.py` |
| `program.md` (instructions) | `program.md` |
| val_bpb metric | exact accuracy |
| 5-min GPU training | `--quick` benchmark (~2 min) |

## Current Results

| Variant | Exact | Adjacent | N |
|---|---|---|---|
| minimal | 48.5% | 65.4% | 769 |
| descriptive | 48.0% | 65.1% | 769 |
| minimal_multishot | 12.7% | 34.7% | 769 |
| descriptive_multishot | 15.5% | 47.1% | 769 |

See `program.md` for what's been tried and what to explore next.

## Related

- [gently](https://github.com/shrofflab/gently) — the microscopy agent framework
- [autoresearch](https://github.com/karpathy/autoresearch) — inspiration for this framework
