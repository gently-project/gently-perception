"""Solvers may not import eval/, io/ground_truth, or hold mutable module state."""
import ast
from pathlib import Path

SOLVERS_DIR = Path(__file__).parent.parent / "harness" / "solvers"
FORBIDDEN_IMPORTS = {"harness.eval", "harness.io.ground_truth", "harness.io.events"}


def test_no_forbidden_imports():
    for py in SOLVERS_DIR.glob("*.py"):
        if py.name == "__init__.py":
            continue
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert not any(n.name.startswith(f) for f in FORBIDDEN_IMPORTS), f"{py.name}: {n.name}"
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(node.module.startswith(f) for f in FORBIDDEN_IMPORTS), f"{py.name}: {node.module}"


def test_no_mutable_module_state():
    """No module-level list/dict/set assignments (would cross-contaminate embryos)."""
    for py in SOLVERS_DIR.glob("*.py"):
        if py.name == "__init__.py":
            continue
        tree = ast.parse(py.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
                        raise AssertionError(f"{py.name}: mutable module-level {target.id}")


def test_gt_grep_invariant():
    """grep -r ground_truth harness/core harness/tools harness/solvers → empty."""
    import subprocess

    root = Path(__file__).parent.parent / "harness"
    for sub in ("core", "tools", "solvers"):
        out = subprocess.run(
            ["grep", "-rli", "ground_truth", str(root / sub)], capture_output=True, text=True
        )
        assert out.stdout.strip() == "", f"ground_truth referenced in {sub}/: {out.stdout}"
