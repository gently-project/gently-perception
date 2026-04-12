# gently-perception

VLM-based perception harness for microscopy stage classification.

Two-layer architecture: a **harness** (pip-installable library) provides fixed infrastructure, and **experiments** use the harness to explore different perception strategies organized by the dimension of the optimization landscape they vary (prompt, representation, tools).

## Usage

### From gently (production)

```python
from gently_perception import Perceiver

perceiver = Perceiver()  # self-contained: loads its own examples and default strategy

result = await perceiver(embryo_id, timepoint, image_b64, timestamp)
result.stage      # "pretzel"
result.reasoning  # "dense coiled mass filling eggshell"
```

### Standalone (benchmarks)

```bash
pip install -e .
python setup_data.py           # download volumes from HuggingFace (~35 GB)
cd experiments
python run.py --variant hybrid --stages pretzel 2fold 1.5fold --force
```

## Results

The best prompt experiment (hybrid) achieves **83.2%** exact accuracy on hard stages. See [experiments/prompt/README.md](experiments/prompt/README.md) for the full results table, variant descriptions, and key findings.

## Architecture

```
gently-perception/
|
+-- gently_perception/          THE HARNESS (pip install, fixed infrastructure)
|   |-- perceiver.py            Perceiver + Session + Observation
|   |-- api.py                  call_claude, parsing, content builders
|   |-- temporal.py             analyze_temporal() pure function
|   |-- examples.py             load reference images from package data
|   |-- organism.py             OrganismConfig + C. elegans defaults
|   +-- types.py                PerceptionOutput(stage, reasoning)
|
+-- experiments/                AUTORESEARCH WORKSPACE (organized by dimension)
|   |-- CLAUDE.md               agent instructions
|   |-- program.md              experiment history & findings
|   |-- run.py                  benchmark runner
|   |
|   |-- prompt/                 dimension 1: prompt ablation (15 variants)
|   |   |-- hybrid.py           83.2% - stage-adaptive prompt switching
|   |   |-- temporal.py         81.0% - temporal anchoring
|   |   |-- scientific.py       82.6% - eggshell fill fraction
|   |   +-- ...
|   |
|   |-- representation/         dimension 2: volume-to-image (future)
|   +-- tools/                  dimension 5: agentic workflows (future)
|
+-- benchmark/                  FIXED evaluation infrastructure
|   |-- testset.py              volume loading, 3-view projections
|   |-- ground_truth.py         stage transition annotations
|   +-- metrics.py              accuracy, confusion matrix
|
+-- data/
    |-- examples/               reference images per stage (ships with package)
    |-- volumes/                3D light-sheet data (downloaded)
    |-- ground_truth/           stage annotations
    +-- results/                experiment outputs
```

## Adding a New Experiment

1. Create `experiments/prompt/my_experiment.py`:
   ```python
   from gently_perception.api import call_claude, build_reference_content, response_to_output

   MY_PROMPT = "..."

   async def perceive_my_experiment(image_b64, references, history, timepoint, **kw):
       content = build_reference_content(references)
       content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}})
       raw = await call_claude(system=MY_PROMPT, content=content)
       return response_to_output(raw)
   ```
2. Run: `cd experiments && python run.py --variant my_experiment --stages pretzel 2fold 1.5fold --force`

See `experiments/program.md` for detailed experiment history and promising directions.

## Related

- [gently](https://github.com/pskeshu/gently) — the microscopy agent framework
- [benchmark dataset](https://huggingface.co/datasets/pskeshu/gently-perception-benchmark) — volumes, ground truth, paper
- [autoresearch](https://github.com/karpathy/autoresearch) — inspiration for this framework pattern
