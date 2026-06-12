"""hybrid_annot + late-pretzel vs hatching disambiguation, cadence-aware.

Targets the one remaining 'ahead' failure cluster: 33 consecutive end-of-series
frames on embryo_5 (x3 seeds = 99 failed predictions) where a very late,
shell-filling, vigorously moving pretzel was called "hatching", and history
anchoring then locked the streak in ("it has been hatching for 2 frames").

Two-part fix, both prompts:
1. Cadence argument kills the trigger — hatching takes 2-3 minutes of real
   time, so how many frames it spans depends entirely on the acquisition
   interval (~4-minute imaging → at most 1 frame, may be missed outright;
   20-second imaging → ~6 frames). The interval is MEASURED at runtime from
   the volume filenames' acquisition timestamps and injected into the
   prompt, so the same solver works on any series. Late pretzel, by
   contrast, presses against and deforms the shell and wriggles WITHIN it
   for a long time, which mimics emergence.
2. Explicit cascade breaker — a "hatching" streak much longer than the
   computed span is self-contradictory; treat it as evidence the earlier
   calls were wrong rather than something to continue.

The 2fold→pretzel timing rule inherited from hybrid_annot ("~60 minutes
after 2fold onset") is converted from a hardcoded frame count to the same
measured-cadence arithmetic.

The annotator's embryo_5 notes back the hatching fix: "the worm moves a
lot ... within the egg shell" on exactly these frames.
"""
import math
import re
from datetime import datetime
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from harness.core import model
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage
from harness.solvers.hybrid_annot import (
    _SCIENTIFIC_ANALYSIS,
    _TEMPORAL_ANALYSIS,
    SCIENTIFIC_SYSTEM as _SCI_BASE,
    TEMPORAL_SYSTEM as _TMP_BASE,
)

_TS_RE = re.compile(r"(\d{8}_\d{6})")

# hybrid_annot's hardcoded frame counts, replaced with measured-cadence text.
_OLD_TIMING_FRAGMENT = "roughly 10-15 frames at this acquisition cadence"

_OLD_TMP_RULE = """\
5. **Late pretzel.** The pretzel stage is long-lasting. A compact bright mass \
that fills the eggshell is still pretzel even if it doesn't look "tangled" — \
it has not hatched unless you see the worm OUTSIDE the shell."""

_OLD_SCI_TEXT = """\
**HATCHING/HATCHED**: The worm is emerging or has left the eggshell — a thin \
elongated worm OUTSIDE the eggshell boundary, or an empty shell."""


@lru_cache(maxsize=16)
def frame_interval_minutes(volume_dir: str) -> float | None:
    """Median interval between acquisition timestamps in the volume filenames.

    Returns None when fewer than two frames carry parseable timestamps —
    callers fall back to cadence-free phrasing.
    """
    stamps = []
    for p in sorted(Path(volume_dir).glob("*.tif")):
        m = _TS_RE.search(p.name)
        if m:
            stamps.append(datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"))
    if len(stamps) < 2:
        return None
    diffs = sorted((b - a).total_seconds() / 60.0 for a, b in pairwise(stamps))
    return diffs[len(diffs) // 2]


def _cadence_facts(interval_min: float | None) -> tuple[str, str, str]:
    """(hatch_span, pretzel_frames, streak) clauses for the given interval."""
    if interval_min is None:
        return (
            "how many frames that spans depends on how often this series was "
            "imaged (unknown here) — frequent imaging captures several hatching "
            "frames, sparse imaging may skip the event entirely",
            "however many frames correspond to about an hour in this series",
            "a \"hatching\" streak lasting far longer than a 2-3 minute event "
            "plausibly could",
        )
    n_hatch = max(1, math.ceil(3.0 / interval_min))
    n_hour = max(2, round(60.0 / interval_min))
    return (
        f"frames in this series are about {interval_min:.1f} minutes apart, so "
        f"hatching appears in at most ~{n_hatch} frame(s) and may be skipped "
        f"entirely",
        f"roughly {n_hour} frames at this series' frame interval",
        f"a \"hatching\" streak much longer than ~{n_hatch} frame(s)",
    )


@lru_cache(maxsize=16)
def build_systems(interval_min: float | None) -> tuple[str, str]:
    """(temporal, scientific) prompts with cadence-derived numbers filled in."""
    hatch_span, pretzel_frames, streak = _cadence_facts(interval_min)

    new_tmp_rule = f"""\
5. **Late pretzel vs hatching.** The pretzel stage is long-lasting, and near \
its end the worm moves vigorously WITHIN the eggshell — pressing against it, \
deforming it, and changing pose between frames. This mimics emergence but is \
still pretzel. Hatching itself takes only 2-3 minutes of real time; \
{hatch_span} (pretzel in one frame, a thin worm clearly OUTSIDE the shell or \
an empty/partial shell in the next). If the bright mass is still shell-sized \
and shell-shaped, nothing has hatched. {streak} in the history is \
self-contradictory — it means those earlier calls were wrong; re-examine \
against the shell outline instead of continuing the streak."""

    new_sci_text = f"""\
**HATCHING/HATCHED**: The worm is emerging or has left the eggshell — a thin \
elongated worm OUTSIDE the eggshell boundary, or an empty shell. Hatching \
takes only 2-3 minutes of real time; {hatch_span}. Late pretzel moves \
vigorously WITHIN the shell, pressing against and deforming it — that mimics \
emergence but is still pretzel as long as the bright mass stays shell-sized \
and shell-shaped. {streak} in the history is self-contradictory; treat it as \
evidence the earlier calls were wrong rather than something to continue."""

    tmp = _TMP_BASE.replace(_OLD_TMP_RULE, new_tmp_rule)
    sci = _SCI_BASE.replace(_OLD_SCI_TEXT, new_sci_text)
    assert tmp != _TMP_BASE, "temporal late-pretzel rule did not match"
    assert sci != _SCI_BASE, "scientific hatching text did not match"

    assert _OLD_TIMING_FRAGMENT in tmp and _OLD_TIMING_FRAGMENT in sci
    tmp = tmp.replace(_OLD_TIMING_FRAGMENT, pretzel_frames)
    sci = sci.replace(_OLD_TIMING_FRAGMENT, pretzel_frames)
    return tmp, sci


def pretzel_window_text(volume_ref: Path) -> str:
    """User-turn phrasing for the 2fold→pretzel elapsed-time exception."""
    interval = frame_interval_minutes(str(Path(volume_ref).parent))
    _, pretzel_frames, _ = _cadence_facts(interval)
    return pretzel_frames


def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches harness/solvers/hybrid.py."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    interval = frame_interval_minutes(str(Path(frame.volume_ref).parent))
    tmp, sci = build_systems(interval)
    return sci if _is_scientific(frame) else tmp


def _user_blocks(frame: FrameInput) -> list[dict]:
    """hybrid_annot's structure with the cadence-derived pretzel window."""
    blocks: list[dict] = [model.text_block(f"\n=== CLASSIFY EMBRYO AT T{frame.timepoint} ===")]
    if frame.history_text:
        blocks.append(model.text_block(frame.history_text))
        last = frame.last_stage.value if frame.last_stage else "early"
        blocks.append(
            model.text_block(
                f"The most recent observation was '{last}'. "
                f"When uncertain, prefer the earlier stage — except at the "
                f"2fold→pretzel boundary: once the history shows 2fold for "
                f"{pretzel_window_text(frame.volume_ref)}, prolonged 2fold "
                f"appearance plus any twisting reads as pretzel, per the timing rule."
            )
        )
    blocks.append(model.image_block(frame.image_b64))
    blocks.append(model.text_block(_SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS))
    return blocks


SOLVER = Solver(name="hybrid_annot2", system=_system_for, tools=(), max_steps=1, user_blocks=_user_blocks)
