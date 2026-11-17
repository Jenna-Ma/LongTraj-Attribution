"""Group-stage elimination over the finalists.

Both corpora fail in the same place, and it is not search. Measured on the
anchors-off baselines, a finalist set built from the candidate list plus a
window around the first answer holds the reference step 0.7791 of the time on
SWE-Bench-Pro and 0.8039 on TrajAudit, at a mean size of 17 and 14 steps. The
runs score 0.3023 and 0.4804. So the reference step is already in a
fourteen-item shortlist and is picked out of it 39% and 60% of the time: every
remaining point is in the selection, none of it in the shortlist.

``zoom.py`` attacked the same gap by re-asking the same question over the same
shortlist, and it was worth exactly nothing -- on TrajAudit it fixed four cases
and broke four. That is what re-asking buys: the second pass is the same reader
with the same priors looking at the same list, so it reproduces the first
answer's bias. No acceptance rule rescued it either (``probe/zoom_rules.py``
scored seven of them offline; the best was two cases and flat on dev).

This stage changes the QUESTION rather than repeating it. The finalists are cut
into contiguous groups of four and each group is judged on its own, in
parallel, with every member shown in its original text. Contiguity is the point:
adjacent steps are the confusable ones, so a group call is forced to make the
fine distinction that a hundred-item prompt lets a reader skip -- "the region is
right and the step is wrong" is a within-region failure. The winners then meet
in one final call, which makes the between-region choice with four strong
arguments in front of it instead of twenty weak ones.

Two properties are deliberate:

  * No group sees the first answer, and groups are formed from the trajectory,
    not from it. The first pass's bias cannot propagate into the group stage;
    it re-enters only if the final call is shown it, which it is not.
  * The first answer survives in the record whatever happens, so acceptance
    rules can be re-scored offline without paying for the calls again. That is
    the same discipline ``zoom`` was built with, and it is what made judging
    ``zoom`` cost nothing.

Cost is ``ceil(|finalists| / 4) + 1`` calls, about six, and the group stage is
one round of latency because the groups are independent.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from .llm import LLM, map_concurrent
from .schema import FlowGraph, Prediction, Trajectory, clip_middle

logger = logging.getLogger(__name__)

# Finalists per group. Four is small enough that each member can be shown in
# full and still get real attention, and it keeps the group stage to five calls
# at the cap below.
GROUP_SIZE = 4

# Cap on the finalist set. Measured at 20 the set holds the reference step
# 0.7791 (SWE-Bench-Pro) and 0.8039 (TrajAudit); at 16 it is 0.7674 and 0.7941,
# and past 20 the curve flattens while every four entries cost another call.
BRACKET_MAX = 20

# Half-width of the window unioned with the candidate list.
BRACKET_WINDOW = 5

BRACKET_STEP_CLIP = 2600
GROUP_BUDGET = 1200
FINAL_BUDGET = 1600

GROUP_SYSTEM = (
    "You examine a few consecutive steps of a failed agent trajectory and "
    "decide which one of them, if any, is itself wrong. You judge only the "
    "steps you are shown. Answer strictly in JSON."
)

FINAL_SYSTEM = (
    "You perform root-cause localisation on failed agent trajectories. You are "
    "given the finalists from several regions of one trajectory, each with the "
    "case already made for it, and must choose one. Answer strictly in JSON."
)

GROUP_PROMPT = """A coding agent FAILED this task. Below are {n} steps taken
from one stretch of its trajectory, in full, exactly as it wrote them.

Decide which ONE of these steps is itself wrong -- or say that none of them is.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, which happened later; it is not the cause):
{failure_summary}

------------------------------------------------------------
THE STEPS
{steps}
------------------------------------------------------------

WHAT COUNTS AS WRONG:
- The step does something incorrect: a wrong edit, a wrong tool call, an
  inference its own evidence does not support, a claim it had no evidence for,
  a requirement of the task quietly dropped.
- A step that is merely unproductive is NOT wrong: reading a file, searching,
  thinking aloud, a command that fails loudly and is then fixed.
- A step that only carries an EARLIER mistake forward is NOT wrong either,
  however visible the damage is there.
- These steps are a fragment. The cause may well be outside them. Saying none
  of them is wrong is a real answer and is often the right one -- do not talk
  yourself into the least defensible step on the list.

Return ONLY JSON:
{{
  "best": <int: the step number of the one wrong step, or null if none is>,
  "confidence": <float 0..1: how sure you are that this step is wrong>,
  "why": "<2-3 sentences, citing what in THIS step's own text is wrong>"
}}"""

FINAL_PROMPT = """A coding agent FAILED this task. Several regions of its
trajectory were examined separately, and each put forward its own best
candidate for the step where things first went wrong. Choose between them.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, which happened later; it is not the cause):
{failure_summary}

HOW THE TASK WAS EXECUTED:
{outline}

------------------------------------------------------------
THE CANDIDATES, EARLIEST FIRST
{candidates}
------------------------------------------------------------

HOW TO CHOOSE:
- The critical step is the EARLIEST step that is itself wrong. If an earlier
  candidate is genuinely wrong, it wins over a later one even when the later one
  is where the damage became obvious.
- But earlier is not automatically better: a candidate that is not actually
  wrong loses to a later one that is, no matter how early it is.
- Each case above was made by a reader who saw only its own region. Weigh how
  well the cited text actually supports the claim, not how confident it sounds.
- You may only answer with one of the candidate step numbers listed above.

Return ONLY JSON:
{{
  "critical_step": <int: one of the candidate step numbers above>,
  "confidence": <float 0..1>,
  "error_type": "<plan|reason|act|obs|verify>.<short subtype>",
  "reason": "<2-4 sentences: why this step and not the others>"
}}"""


def finalists(traj: Trajectory, first: Optional[int],
              cands: Optional[Sequence[Dict[str, object]]] = None,
              window: int = BRACKET_WINDOW,
              cap: int = BRACKET_MAX) -> List[int]:
    """The steps the bracket is allowed to choose from.

    The union of the candidate list and a window around the first answer. Both
    are needed and each corpus needs a different one: the candidate list holds
    the reference step 0.7674 of the time on SWE-Bench-Pro but only 0.4314 on
    TrajAudit, where the window holds it 0.7353. Unioned, 0.7907 and 0.8137.

    A window of zero or less means no window at all -- the candidate list on its
    own -- which is how the window can be taken out of the finalist set without
    taking the stage out of the run.

    Over the cap, WINDOW entries are dropped and the candidate list is kept
    whole. Cutting by distance to the first answer instead was measured and is
    much worse on SWE-Bench-Pro -- 0.7907 down to 0.6512 at a cap of 16 --
    because there the reference step is usually nowhere near the first answer
    (a +-5 window alone encloses it in only 0.3953 of cases). Distance to an
    answer that is wrong 70% of the time is not evidence about where the
    reference step is; membership of the candidate list is.
    """
    n = len(traj.steps)
    keep: List[int] = []
    for c in (cands or []):
        try:
            s = int(c["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= s < n and traj.steps[s].attributable and s not in keep:
            keep.append(s)
    keep = keep[:cap]

    rest: List[int] = []
    if first is not None and 0 <= first < n and window > 0:
        for s in range(max(0, first - window), min(n, first + window + 1)):
            if s not in keep and traj.steps[s].attributable:
                rest.append(s)
        rest.sort(key=lambda s: (abs(s - first), s))
    return sorted(keep + rest[:max(0, cap - len(keep))])


def groups(fin: Sequence[int], size: int = GROUP_SIZE) -> List[List[int]]:
    """Contiguous chunks, balanced so no group is left with a single member.

    Contiguous rather than interleaved because the failure this stage targets is
    a within-region one: the steps that get confused for each other are the
    neighbouring ones, and putting neighbours in the same call is what forces
    the distinction to be made. A lone finalist in a group of one would be
    asked "is this step wrong" with nothing to lose to, which is the question
    this stage is trying to avoid.
    """
    fin = list(fin)
    if len(fin) <= size:
        return [fin] if fin else []
    k = max(1, int(round(len(fin) / float(size))))
    per, extra = divmod(len(fin), k)
    out, i = [], 0
    for g in range(k):
        take = per + (1 if g < extra else 0)
        out.append(fin[i:i + take])
        i += take
    out = [g for g in out if g]
    # Whatever the arithmetic, no group may be left alone: fold a singleton
    # into its neighbour rather than ask "is this one step wrong" in isolation.
    while len(out) > 1 and len(out[-1]) == 1:
        out[-2].extend(out.pop())
    return out


def _render(traj: Trajectory, steps: Sequence[int],
            clip: int = BRACKET_STEP_CLIP) -> str:
    return "\n\n".join(
        "### STEP %d (%s)\n%s"
        % (s, traj.steps[s].role, clip_middle(traj.steps[s].content or "", clip))
        for s in steps)


def _judge_group(traj: Trajectory, graph: FlowGraph, llm: LLM,
                 members: Sequence[int]) -> Optional[Dict[str, object]]:
    """One group's verdict, or None when it declined to name a step."""
    prompt = GROUP_PROMPT.format(
        n=len(members),
        task=clip_middle(traj.task_description, 1500),
        failure_summary=graph.failure_summary or "(not summarised)",
        steps=_render(traj, members))
    obj = llm.generate_obj(prompt, max_tokens=GROUP_BUDGET, system=GROUP_SYSTEM)
    if not isinstance(obj, dict):
        return None
    try:
        step = int(obj.get("best"))
    except (TypeError, ValueError):
        return None
    if step not in members:
        # An abstention is expressed as null; a step outside the group is the
        # model answering a question it was not asked, and is dropped rather
        # than snapped -- it saw only these steps, so it cannot have grounds for
        # any other.
        return None
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"step": step, "confidence": conf,
            "why": str(obj.get("why") or "")[:600],
            "group": list(members)}


def bracket(traj: Trajectory, graph: FlowGraph, llm: LLM, pred: Prediction,
            cands: Optional[Sequence[Dict[str, object]]] = None,
            outline=None, window: int = BRACKET_WINDOW,
            cap: int = BRACKET_MAX, size: int = GROUP_SIZE,
            workers: int = 8) -> Prediction:
    """Run the group stage and the final, and return the refined answer.

    No-op -- no calls, prediction untouched -- when there are fewer than two
    finalists. When every group abstains the first answer stands: an empty
    bracket is evidence that the shortlist does not contain a step anyone can
    argue is wrong, not evidence for some particular step.
    """
    first = pred.critical_step
    fin = finalists(traj, first, cands, window=window, cap=cap)
    if len(fin) < 2:
        return pred

    gs = groups(fin, size=size)
    verdicts = [v for v in map_concurrent(
        lambda m: _judge_group(traj, graph, llm, m), gs,
        workers=min(workers, max(1, len(gs)))) if v]

    record: Dict[str, object] = {
        "first": first,
        "first_confidence": pred.confidence,
        "finalists": fin,
        "n_groups": len(gs),
        "verdicts": [{"step": v["step"], "confidence": v["confidence"],
                      "group": v["group"]} for v in verdicts],
        "abstained": len(gs) - len(verdicts),
        "proposed": None,
        "accepted": False,
        "in_set": False,
        "confidence": None,
        "reason": "",
    }

    if not verdicts:
        logger.info("%s: every group abstained; keeping step %s",
                    traj.task_id, first)
        pred.bracket = record
        return pred

    if len(verdicts) == 1:
        # Nothing to arbitrate. Spending a call to confirm a single candidate
        # only gives the model a chance to disagree with itself.
        win = verdicts[0]
        step, conf, why = win["step"], win["confidence"], win["why"]
        record["final_call"] = False
    else:
        verdicts.sort(key=lambda v: v["step"])
        block = "\n\n".join(
            "### CANDIDATE: STEP %d (confidence %.2f)\nTHE CASE FOR IT: %s\n"
            "ITS OWN TEXT:\n%s"
            % (v["step"], v["confidence"], v["why"] or "(none given)",
               clip_middle(traj.steps[v["step"]].content or "",
                           BRACKET_STEP_CLIP))
            for v in verdicts)
        prompt = FINAL_PROMPT.format(
            task=clip_middle(traj.task_description, 1500),
            failure_summary=graph.failure_summary or "(not summarised)",
            outline=(clip_middle(outline.render(), 4000)
                     if outline is not None else "(not built)"),
            candidates=block)
        obj = llm.generate_obj(prompt, max_tokens=FINAL_BUDGET,
                               system=FINAL_SYSTEM)
        record["final_call"] = True
        allowed = [v["step"] for v in verdicts]
        try:
            step = int(obj.get("critical_step"))
        except (TypeError, ValueError):
            step = None
        if step not in allowed:
            # Refused, not snapped: it was shown everything it could choose
            # from, so an answer outside that set means it did not do the task.
            logger.info("%s: the final answered step %s outside its own "
                        "candidates; falling back to the earliest group winner",
                        traj.task_id, step)
            record["out_of_set"] = step
            step = allowed[0]
        conf = 0.0
        try:
            conf = max(0.0, min(1.0, float(obj.get("confidence"))))
        except (TypeError, ValueError):
            pass
        why = str(obj.get("reason") or "")[:800]
        et = str(obj.get("error_type") or "")[:80]
        if et:
            record["error_type"] = et

    record["proposed"] = step
    record["in_set"] = True
    record["accepted"] = bool(step != first)
    record["confidence"] = conf
    record["reason"] = why

    if step != first:
        logger.info("%s: the bracket moved the answer from step %s to step %d",
                    traj.task_id, first, step)
        pred.critical_step = step
        pred.confidence = conf
        pred.reason = ("%s\n[bracket: moved off step %s, which the first pass "
                       "chose]" % (why, first))[:1500]
        if record.get("error_type"):
            pred.error_type = str(record["error_type"])
    pred.bracket = record
    return pred
