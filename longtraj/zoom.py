"""Second pass: re-read a handful of finalists in their own words.

The first localiser reads the whole trajectory, and it has to: the answer could
be anywhere. The price is that it reads most steps as a one-sentence paraphrase
and its answer has to be chosen from a hundred-odd of them at once. Two
measurements say that is where the answers are being lost, and they say it
differently on each corpus:

  * SWE-Bench-Pro — the restricted answer space already contains the reference
    step 78.6% of the time (mean 11 entries, 9% of the trajectory), yet the
    localiser picks it only 39.4% of the time it is there. The answer is inside
    an eleven-item list and the list is not being read closely.
  * TrajAudit — acc@3 is 0.7059 against acc@0 of 0.4804. A fifth of all cases
    land within three steps of the reference step and miss it. Nothing about
    those cases is a search failure; they are resolution failures.

Both are the same shape: the region is right and the step is wrong. So this
stage re-asks the question over a small finalist set, with every finalist shown
in its ORIGINAL text rather than a paraphrase. It is not the restore pass in
``compress.py`` -- putting every short step back verbatim was measured and lost
on both corpora, because it bought fidelity by adding bulk. Here the bulk is
bounded by construction: at most ``ZOOM_MAX`` steps are shown at all.

The first answer is kept in the record whatever happens, so a run can be
re-scored offline under a different acceptance rule -- "take the zoom only when
it is more confident", say -- without paying for the calls again.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from .llm import LLM
from .schema import FlowGraph, Prediction, Trajectory, clip_middle

logger = logging.getLogger(__name__)

# How many finalists are shown. Eleven is the mean size of the SWE-Bench-Pro
# candidate list and a +-5 window is eleven steps, so this covers both sources
# of finalists without truncating either in the common case.
ZOOM_MAX = 14

# Half-width of the window used when there is no candidate list to inherit.
# Measured on TrajAudit: +-3 encloses the reference step in 0.7059 of cases and
# +-5 in 0.7549, after which the curve flattens while the prompt keeps growing.
ZOOM_WINDOW = 5

# Per-step clip. The point of this stage is that the finalists arrive in their
# own words, so the clip has to be generous; 14 x 2400 is ~34k characters, a
# quarter of what the first localiser reads.
ZOOM_STEP_CLIP = 2400

ZOOM_BUDGET = 2000

ZOOM_SYSTEM = (
    "You perform root-cause localisation on failed agent trajectories. "
    "You are given a shortlist of candidate steps in full and must choose the "
    "earliest one that is actually wrong. Answer strictly in JSON."
)

ZOOM_PROMPT = """A coding agent FAILED this task. A first pass over the whole
trajectory narrowed the cause to the steps below and answered STEP {first}.

Your job is to check that answer against the FULL TEXT of the finalists. The
first pass read most of the trajectory in summary; you are reading these steps
as they were actually written. If the first answer is right, keep it. If a
DIFFERENT step below is the earliest one that is actually wrong, say so.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

WHY THE FIRST PASS CHOSE STEP {first}:
{first_reason}

------------------------------------------------------------
THE FINALISTS, IN FULL, IN ORDER
{finalists}
------------------------------------------------------------

HOW TO CHOOSE:
- The critical step is the EARLIEST step that is itself wrong. A step that only
  carries an earlier mistake forward is not it, however visible the damage is
  there.
- A step is "wrong" if it does something incorrect: a wrong edit, a wrong tool
  call, a wrong inference from what it had just read, a claim it had no
  evidence for, an abandoned requirement. A step that is merely unproductive --
  reading a file, searching, thinking aloud -- is not wrong.
- Judge each finalist on its own text. The first pass may have been reading a
  summary that lost the detail that matters.
- You may only answer with one of the finalist step numbers listed above.

Return ONLY JSON:
{{
  "critical_step": <int: one of the finalist step numbers>,
  "confidence": <float 0..1>,
  "changed": <bool: true if this differs from step {first}>,
  "error_type": "<plan|reason|act|obs|verify>.<short subtype>",
  "reason": "<2-4 sentences: what is wrong in THIS step's own text>"
}}"""


def finalists(traj: Trajectory, first: Optional[int],
              cands: Optional[Sequence[Dict[str, object]]] = None,
              window: int = ZOOM_WINDOW,
              cap: int = ZOOM_MAX) -> List[int]:
    """The small set of steps the second pass is allowed to choose from.

    Inherited from the candidate stage when there is one -- it is a measured
    shortlist and rebuilding it from a window would throw that away -- and cut
    from a window around the first answer when there is not. The first answer
    is always in the set: this stage exists to check it, and a set that cannot
    return it could only ever move the answer.

    When the set is over ``cap`` the entries nearest the first answer are kept,
    ties going to the earlier step. Dropping the far entries is the safe cut:
    a finalist twenty steps away was never going to be reached by a stage whose
    premise is that the region is already right.
    """
    n = len(traj.steps)
    pool: List[int] = []
    if cands:
        for c in cands:
            try:
                s = int(c["step"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= s < n:
                pool.append(s)
    elif first is not None:
        pool = [s for s in range(max(0, first - window),
                                min(n, first + window + 1))]
    if first is not None and 0 <= first < n:
        pool.append(first)

    keep = sorted({s for s in pool if traj.steps[s].attributable})
    if first is not None and first in keep and len(keep) > cap:
        keep = sorted(sorted(keep, key=lambda s: (abs(s - first), s))[:cap])
    elif len(keep) > cap:
        keep = keep[:cap]
    return keep


def _render(traj: Trajectory, steps: Sequence[int],
            clip: int = ZOOM_STEP_CLIP) -> str:
    out = []
    for s in steps:
        st = traj.steps[s]
        out.append("### STEP %d (%s)\n%s"
                   % (s, st.role, clip_middle(st.content or "", clip)))
    return "\n\n".join(out)


def zoom(traj: Trajectory, graph: FlowGraph, llm: LLM, pred: Prediction,
         cands: Optional[Sequence[Dict[str, object]]] = None,
         window: int = ZOOM_WINDOW, cap: int = ZOOM_MAX) -> Prediction:
    """Re-ask the question over the finalists and return the refined answer.

    The stage is a no-op -- no call, prediction returned unchanged -- when
    there is nothing to refine: no first answer, or fewer than two finalists.
    An answer outside the finalist set is refused rather than snapped: the
    whole premise is that the model was shown everything it was allowed to
    choose from, so a step outside the set means it did not do the task, not
    that it found something better.
    """
    first = pred.critical_step
    fin = finalists(traj, first, cands, window=window, cap=cap)
    if first is None or len(fin) < 2:
        return pred

    prompt = ZOOM_PROMPT.format(
        first=first,
        task=clip_middle(traj.task_description, 2000),
        failure_summary=graph.failure_summary or "(not summarised)",
        first_reason=clip_middle(pred.reason or "(no reason recorded)", 800),
        finalists=_render(traj, fin))

    obj = llm.generate_obj(prompt, max_tokens=ZOOM_BUDGET, system=ZOOM_SYSTEM)
    try:
        step = int(obj.get("critical_step"))
    except (TypeError, ValueError):
        step = None

    accepted = step is not None and step in fin
    record = {
        "first": first,
        "first_confidence": pred.confidence,
        "finalists": fin,
        "proposed": step,
        "accepted": bool(accepted and step != first),
        "in_set": bool(accepted),
        "confidence": None,
        "reason": "",
    }
    try:
        record["confidence"] = max(0.0, min(1.0, float(obj.get("confidence"))))
    except (TypeError, ValueError):
        record["confidence"] = 0.0
    record["reason"] = str(obj.get("reason") or "")[:800]

    if not accepted:
        if step is not None:
            logger.info("%s: zoom answered step %s outside its own finalists; "
                        "keeping step %d", traj.task_id, step, first)
        pred.zoom = record
        return pred

    if step != first:
        logger.info("%s: zoom moved the answer from step %d to step %d",
                    traj.task_id, first, step)
        pred.critical_step = step
        pred.confidence = record["confidence"]
        # The reason must describe the step now being returned, not the one the
        # first pass argued for, or the record would explain the wrong answer.
        pred.reason = ("%s\n[zoom: moved off step %d, which the first pass "
                       "chose]" % (record["reason"], first))[:1500]
        et = str(obj.get("error_type") or "")[:80]
        if et:
            pred.error_type = et
    pred.zoom = record
    return pred
