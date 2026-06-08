"""baseline.lock drift detection."""
from harness.core.solver import Solver
from harness.eval import baseline


def test_lock_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "LOCK_PATH", tmp_path / "baseline.lock")
    s = Solver(name="t", system="hello world")
    lock = baseline.make("t", s, "claude-opus-4-6", "gtsha", 0.817, 0.024, 3)
    baseline.write(lock)
    loaded = baseline.load()
    assert loaded.mean == 0.817
    assert loaded.prompt_sha == lock.prompt_sha


def test_check_detects_prompt_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "LOCK_PATH", tmp_path / "baseline.lock")
    s1 = Solver(name="t", system="prompt v1")
    baseline.write(baseline.make("t", s1, "m", "g", 0.8, 0.0, 3))
    s2 = Solver(name="t", system="prompt v2")
    diff = baseline.check("m", baseline.prompt_sha(s2), "g")
    assert any("prompt_sha" in d for d in diff)


def test_check_detects_model_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "LOCK_PATH", tmp_path / "baseline.lock")
    s = Solver(name="t", system="x")
    baseline.write(baseline.make("t", s, "claude-opus-4-6", "g", 0.8, 0.0, 3))
    diff = baseline.check("claude-opus-4-7", baseline.prompt_sha(s), "g")
    assert any("model" in d for d in diff)


def test_check_clean_when_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "LOCK_PATH", tmp_path / "baseline.lock")
    s = Solver(name="t", system="x")
    lock = baseline.make("t", s, "m", "g", 0.8, 0.0, 3)
    baseline.write(lock)
    assert baseline.check("m", lock.prompt_sha, "g") == []


def test_no_lock_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "LOCK_PATH", tmp_path / "nope.lock")
    assert baseline.check("m", "p", "g") == []
