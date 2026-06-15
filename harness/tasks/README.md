# harness/tasks/ — multi-task perception

A **Task** is one perception objective (stage classification, signal onset,
anomaly detection). v1 ships a single onset task (`dopaminergic`) and the
machinery to add more by writing a ~30-line `SignalSpec`.

## Running the dopaminergic eval

```bash
# 0. One-time environment
pip install -e . && pip install -r requirements.txt scikit-image pytest
export ANTHROPIC_API_KEY=...
hf auth login                    # SLS762 dataset is gated

# 1. Pull the dataset (writes data/sls762/)
python setup_data.py --dataset sls762

# 2. Fill in ground-truth onset timepoints from the dataset README
$EDITOR data/ground_truth/onset/sls762.json

# 3. Smoke run: 5 frames per embryo, embryos 3-6 only
python -m harness eval-task --task dopaminergic --limit 5

# 4. Full run, 3 seeds
python -m harness eval-task --task dopaminergic --n-runs 3

# 5. Inspect a frame's perceiver/classifier transcript
python -m harness view runs/task-dopaminergic/.../seed0/events.jsonl --frame embryo_3/T042
```

Output lands in `runs/task-dopaminergic/{model}/{ts}/seedN/events.jsonl` with a
`summary.json` reporting `miss_rate`, `mean_latency_frames`, `fp_rate`, and
`tokens` per seed.

## Adding a new onset detector

```python
# harness/tasks/onset/my_signal.py
from harness.core.types_v2 import ArmingPolicy, Intensity, RenderSpec, SetCadence, SignalSpec
from harness.tasks._base import make_onset_task

SPEC = SignalSpec(
    name="my_signal",
    describe_prompt="...",
    rubric={Intensity.NONE: "...", Intensity.WEAK: "...", ...},
    threshold=Intensity.MEDIUM,
    render=RenderSpec("single_view_fixed"),
    arming=ArmingPolicy(predicate=lambda s: ..., latch=True, sentinel_every=10),
    on_detect=SetCadence(interval_s=60, reason="my_signal onset"),
)
TASK = make_onset_task(SPEC)
DATASET = "my_dataset"   # key in harness/io/datasets.py
```

Then register it in `harness/tasks/__init__.py:get_tasks()` and add the dataset
in `harness/io/datasets.py`.

## Invariants

- `grep -r ground_truth harness/core harness/tools harness/solvers harness/tasks` → empty
- `TaskInput` is frozen; `own_history` is an immutable tuple of this task's own
  observations. No GT field.
- Perceiver/classifier parse failures raise `ModelOutputError` — never coerced
  to `Intensity.NONE`.

## Not yet implemented (v2)

- `harness/routers/` — `always_on`, `gated`, `shared_perceiver`
- `harness/tasks/stage.py` — stage classification wrapped as a `Task`
- `harness/core/policy.py` — observations → microscope actions
- Router scoring (`tokens_per_frame` × `miss_rate` Pareto)
