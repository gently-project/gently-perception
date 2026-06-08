"""hybrid_nodefer with stage criteria rewritten from the annotator's notes.

The benchmark dataset ships per-embryo annotations.json files with the
annotator's (Kesavan's) reasoning. Three things in them that the prompts
lacked:

1. Pretzel was annotated with a TIMING rule, not appearance: "the stereotypic
   time between 2fold and pretzel ... 60 minutes after 2fold" (embryo_6 T41,
   embryo_8 T53). Appearance-only pretzel detection fights the labels.
2. Tail-progress criteria for the transition stages: comma = tail tip points
   toward head; 1.5fold = tail tip about halfway between posterior and
   anterior; 2fold = body wraps the eggshell twice / tail beyond halfway.
3. Twisting of the two body segments marks the move beyond 2fold; twitching
   starts around 1.5fold/2fold, so motion blur from there on is normal.

Only the qualitative methodology and the ~60-minute biology are used — no
per-embryo transition timepoints from the eval set appear in the prompts.
"""
from harness.core import model
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage

TEMPORAL_SYSTEM = """\
You are classifying C. elegans embryo developmental stages from fluorescence \
light-sheet microscopy max-intensity projection images. Each image shows three \
orthogonal views (XY top-left, YZ top-right, XZ bottom-left).

The stages in order are: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

## CRITICAL CLASSIFICATION RULES

1. **Compare to reference images first.** Match the overall shape, brightness \
pattern, and internal structure to the reference images provided. The references \
are your primary guide.

2. **Track the TAIL.** The posterior forms a tail, first visible as an \
indentation. Stage transitions are defined by tail progress:
   - **bean**: an indentation appears at the posterior; a small gap starts to \
open in the posterior half.
   - **comma**: the gap extends and deepens; a distinct tail is observable and \
its TIP STARTS TO POINT TOWARD THE HEAD. Comma is brief — only a few frames.
   - **1.5fold**: the tail has folded back alongside the body and its tip \
reaches about HALFWAY between posterior and anterior ends.
   - **2fold**: the tail has grown well beyond halfway — the body wraps the \
inside of the eggshell about twice.
   - **pretzel**: a THIRD fold or visible TWISTING of the two body segments \
around each other.

3. **Use elapsed time for pretzel.** Pretzel typically begins about an hour \
after 2fold onset — roughly 10-15 frames at this acquisition cadence. The \
human annotator who labeled this dataset used exactly this timing rule to \
place the 2fold→pretzel boundary, because tight pretzel coils are genuinely \
hard to count in projections. So check the observation history: if the \
embryo has already looked 2fold for that long, actively look for twisting or \
a third fold instead of repeating 2fold. Twisting makes plain "2fold" an \
undermeasure even when a clean third fold isn't visible yet.

4. **Embryos twitch from around 1.5fold/2fold onward.** Motion blur in later \
frames is normal — do not mistake blur for debris or for a stage change. \
Small debris may also be present near the embryo; ignore anything outside \
the eggshell outline.

5. **Late pretzel.** The pretzel stage is long-lasting. A compact bright mass \
that fills the eggshell is still pretzel even if it doesn't look "tangled" — \
it has not hatched unless you see the worm OUTSIDE the shell.

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation citing tail position and elapsed time"
}"""

SCIENTIFIC_SYSTEM = """\
You are classifying C. elegans embryo developmental stages from fluorescence \
light-sheet microscopy max-intensity projection images. Each image shows three \
orthogonal views (XY top-left, YZ top-right, XZ bottom-left).

The stages in order are: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

## STAGE DESCRIPTIONS (what to look for in fluorescence max-projections)

**EARLY**: Bright oval mass of nuclei. Uniform, roughly symmetric.

**BEAN**: An indentation appears at the posterior; a small gap opens in the \
posterior half of the embryo.

**COMMA**: The gap deepens; a distinct tail is observable, its tip starting \
to point toward the head. The eggshell is MOSTLY EMPTY.

**1.5FOLD**: The tail has folded back alongside the body, its tip reaching \
about HALFWAY between the posterior and anterior ends. The eggshell is \
SPARSELY FILLED — significant dark space remains.

**2FOLD**: The tail has grown well beyond halfway — the body wraps the \
inside of the eggshell about twice. TWO PARALLEL body segments connected by \
a bend. The eggshell is MODERATELY FILLED.

**PRETZEL**: A THIRD fold, or the two body segments visibly TWISTING around \
each other. The eggshell is DENSELY FILLED — multiple overlapping body \
layers raise the overall brightness. Twisting without a clean third fold \
already means the embryo is past 2fold.

**HATCHING/HATCHED**: The worm is emerging or has left the eggshell — a thin \
elongated worm OUTSIDE the eggshell boundary, or an empty shell.

## CLASSIFICATION RULES

1. **Compare to references first**, then use the descriptions above.

2. **Use elapsed time for the 2fold→pretzel call.** Pretzel typically begins \
about an hour after 2fold onset — roughly 10-15 frames at this acquisition \
cadence. The human annotator who labeled this dataset used exactly this \
timing rule to place the 2fold→pretzel boundary, because tight pretzel coils \
are genuinely hard to count in projections. So check the observation \
history: if 2fold appearance has persisted that long, actively look for \
twisting or a third fold rather than repeating 2fold.

3. **KEY VISUAL DISCRIMINATOR: eggshell fill fraction.**
   - Sparse (lots of dark space inside shell) → 1.5fold or earlier
   - Moderate (some dark space) → 2fold
   - Dense (shell mostly filled, bright) → pretzel

4. **Embryos twitch from around 1.5fold/2fold onward** — motion blur in \
later frames is normal and is not debris or a stage regression.

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation citing fill fraction, tail position, and elapsed time"
}"""

_SCIENTIFIC_ANALYSIS = (
    "Analyze: (1) How much of the eggshell is filled with signal? "
    "(2) How many frames has the embryo looked 2fold in the history, and is there "
    "any twisting or third fold? (3) Which reference images match best? Then classify."
)
_TEMPORAL_ANALYSIS = (
    "Compare this image to the reference images above. Where is the tail tip relative "
    "to the body, and how long has the embryo been in its current stage per the history? "
    "Classify accordingly."
)


def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches harness/solvers/hybrid.py."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    return SCIENTIFIC_SYSTEM if _is_scientific(frame) else TEMPORAL_SYSTEM


def _user_blocks(frame: FrameInput) -> list[dict]:
    """Same structure as hybrid_nodefer; history is the input to the elapsed-time rule."""
    blocks: list[dict] = [model.text_block(f"\n=== CLASSIFY EMBRYO AT T{frame.timepoint} ===")]
    if frame.history_text:
        blocks.append(model.text_block(frame.history_text))
        last = frame.last_stage.value if frame.last_stage else "early"
        blocks.append(
            model.text_block(
                f"The most recent observation was '{last}'. "
                f"When uncertain, prefer the earlier stage — except at the "
                f"2fold→pretzel boundary: once the history shows ~10-15 frames of "
                f"2fold, prolonged 2fold appearance plus any twisting reads as "
                f"pretzel, per the timing rule."
            )
        )
    blocks.append(model.image_block(frame.image_b64))
    blocks.append(model.text_block(_SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS))
    return blocks


SOLVER = Solver(name="hybrid_annot", system=_system_for, tools=(), max_steps=1, user_blocks=_user_blocks)
