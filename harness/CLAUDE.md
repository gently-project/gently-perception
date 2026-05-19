# harness/ — agentic perception harness

One harness, two entry points. The agent loop, history, rendering, tools,
verification, and scoring are implemented exactly once in `core/`. Research
(`python -m harness eval`) and production (`harness.Perceiver`) are thin
wrappers on the same `core.loop.step()`.

## Iteration loop (autonomous coding agent)

1. **Read** `runs/` and `baseline.lock` to see what's been tried.
2. **Edit** a file in `harness/solvers/` (declarative `Solver` config) or add a
   tool in `harness/tools/`. Nothing else.
3. **Run**:
   ```bash
   # First run after editing a solver/tool — drift check WILL fire:
   python -m harness eval --solver my_solver --n-runs 3 --update-baseline

   # Subsequent runs comparing against the locked baseline:
   python -m harness eval --solver my_solver --n-runs 3
   ```
4. **Inspect** a single frame's full agent conversation:
   ```bash
   python -m harness view runs/my_solver/.../seed0/events.jsonl --frame embryo_2/T067
   ```
5. **Commit** with mean±std in the message. Keep iterating.

## Adding a solver

```python
# harness/solvers/my_solver.py
from harness.core.solver import Solver

SYSTEM = "..."
SOLVER = Solver(
    name="my_solver",
    system=SYSTEM,                  # or a callable FrameInput → str
    tools=("zoom", "measure"),      # subset of harness/tools/ registry
    max_steps=5,                    # 1 + tools=() ⇒ one-shot
    thinking="adaptive",            # None for parity with old baselines
    effort="high",
)
```
Then add one line to `harness/solvers/__init__.py:REGISTRY`.

## Adding a tool

```python
# harness/tools/my_tool.py
from harness.tools import tool, Range
from harness.core.types import NumericResult

@tool(returns="numeric")
def my_tool(volume, *, k: Annotated[int, Range(0, "Z")]) -> NumericResult:
    """Docstring becomes the model-facing tool description."""
    ...
```
Then `from harness.tools import my_tool as _` at the bottom of
`harness/tools/__init__.py`, and add `tests/test_tools_my_tool.py`.

## Invariants (CI-enforced)

- `grep -r ground_truth harness/core harness/tools harness/solvers` → empty
- `FrameInput` is frozen; `history` is an immutable tuple of predictions
- Parse failure raises `ModelOutputError` — never coerced to `stage="early"`
- Solvers may not import `harness.eval` or `harness.io.ground_truth`
- `baseline.lock` drift hard-fails without `--accept-drift`/`--update-baseline`

## Files you should NOT edit

`harness/core/`, `harness/eval/`, `harness/io/` — these are the harness.
If you think you need to, you're probably trying to do something a solver or
tool should do instead.
