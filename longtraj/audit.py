"""Final pass: audit the prediction's direct causal predecessors.

TrajAudit misses late. Measured on the shipped arm (``probe/ta_repro2.jsonl``,
``probe/bundle_ceiling.py ta``): of 49 misses, 35 answer a step LATER than the
reference one, and on 14 of those 49 the reference step is a DIRECT flow parent
of the answer. That is the whole opportunity -- 0.5000 -> 0.6429 if every one
of them were recovered and nothing already right were lost.

So this stage does not re-ask "which step is critical". Asking that again is
the move this repo has six negative results for. It asks a strictly narrower
question that only the graph can pose:

    the answer P used something. Did a PARENT of P already introduce the wrong
    object, leaving P to carry it out faithfully?

ONE HOP, AND THAT IS PRE-REGISTERED. The same measurement priced two hops:
the ancestor set doubles (2.0 -> 4.2 steps) and reaches two more rows (14 ->
16). Two hops buys 2 rows for 2x the answer space, and the second hop breaks
the question -- "did the parent introduce it" is answerable from an edge, "did
some ancestor introduce it" is the original search all over again.

THE INCUMBENT WINS TIES, as everywhere else here. Moving up requires all four
of: a named wrong object, an actual graph edge of the kind claimed carrying it,
P introducing no independent error of its own, and repairing the parent being
enough to break the chain. Anything less and the answer does not move. The
stage can only ever move EARLIER, which is the direction the misses lie in.

The pre-audit answer is kept in the record whatever happens, so an alternative
acceptance rule can be scored against a stored run without paying the calls
again.

The reference label never enters this module: it reads a ``Trajectory``, a
``FlowGraph`` and the incumbent ``Prediction``, none of which can carry one.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .llm import LLM
from .schema import (CONTROL_FLOW, DATA_FLOW, FlowGraph, Prediction,
                     Trajectory, clip_middle)

logger = logging.getLogger(__name__)

# Cap on the parents examined. The measured mean is 2.0 and the median 2, so
# this does not bite on a typical row; it exists so a hub step cannot turn one
# call into an unbounded prompt. Nearest-to-P kept when it does bite.
AUDIT_MAX = 6

# The two steps under comparison arrive in their own words -- the premise is
# that a summary is what lost the distinction between introducing an error and
# propagating one.
AUDIT_FOCAL_CLIP = 3000
AUDIT_PARENT_CLIP = 2200
AUDIT_EDGE_MAX = 4              # edges quoted per parent, of each kind

AUDIT_BUDGET = 1400

AUDIT_SYSTEM = (
    "You audit root-cause localisation on failed agent trajectories. You are "
    "given one proposed critical step and the steps it directly depends on, "
    "and you decide whether the proposal merely carried out an error that was "
    "already made. Answer strictly in JSON."
)

AUDIT_PROMPT = """A coding agent FAILED this task. A localiser has proposed
STEP {p} as the critical step -- the step that, had it gone differently, would
have prevented the failure.

You are NOT being asked to search the trajectory again. You are being asked one
question about STEP {p} and the steps it directly depends on:

    Had one of those parents ALREADY introduced the wrong information,
    assumption or decision -- so that STEP {p} only carried it out faithfully?

If so, the parent is the critical step and STEP {p} is downstream damage. If
not, STEP {p} stands.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

WHY THE LOCALISER CHOSE STEP {p}:
{p_reason}

------------------------------------------------------------
THE PROPOSED CRITICAL STEP, IN FULL

### STEP {p} ({p_role})
{p_text}
------------------------------------------------------------
ITS DIRECT PREDECESSORS, IN FULL, WITH THE DEPENDENCY THAT LINKS EACH TO
STEP {p}
{parents}
------------------------------------------------------------

YOU MAY MOVE THE ANSWER TO A PARENT ONLY IF ALL FOUR HOLD:

  1. You can NAME the specific wrong object the parent introduced -- a value, a
     file, a read of the requirements, a decision. "The parent was also part of
     the failure" is not naming anything.
  2. That object reaches STEP {p} along one of the dependencies listed above.
     You must say which kind of dependency carried it.
  3. STEP {p} introduces no independent error of its own. If STEP {p} is wrong
     in a way the parent does not account for, STEP {p} stays -- the earliest
     wrong step is what is wanted, but a step that adds a NEW error is itself
     wrong.
  4. Repairing the parent alone would be enough to break this failure chain.
     If STEP {p} would still have to be changed as well, the parent is not
     sufficient and STEP {p} stays.

If you cannot satisfy every one of the four, KEEP STEP {p}. Keeping it is the
correct answer whenever you are unsure: a parent is upstream of the answer by
construction, so "it is earlier, therefore it is more root" is always available
and is not evidence.

Return ONLY JSON:
{{
  "move": <bool: true only if all four conditions hold>,
  "to": <int: the parent step number, or null if keeping STEP {p}>,
  "object": "<the specific wrong thing the parent introduced, or \\"\\">",
  "edge_kind": "<data|control|null: which dependency carried it>",
  "p_adds_new_error": <bool: does STEP {p} add an error of its own?>,
  "parent_fix_suffices": <bool: would fixing the parent alone break the chain?>,
  "confidence": <float 0..1>,
  "reason": "<2-4 sentences>"
}}"""


def _parents(graph: FlowGraph, traj: Trajectory, p: int,
             cap: int = AUDIT_MAX) -> Tuple[List[int], Dict[int, List]]:
    """Direct flow parents of ``p`` that could themselves be an answer.

    Non-attributable steps are dropped: the corpus says only the agent's own
    steps can be the critical step, so offering a system or user step could
    only ever move the answer somewhere it is not allowed to be. When more than
    ``cap`` survive, the ones nearest ``p`` are kept -- a parent forty steps
    back is not the "P only carried it out" story this stage is testing.
    """
    n = len(traj.steps)
    edges: Dict[int, List] = {}
    for e in graph.edges:
        if e.dst != p or not (0 <= e.src < n) or e.src == p:
            continue
        if not traj.steps[e.src].attributable:
            continue
        edges.setdefault(e.src, []).append(e)
    keep = sorted(edges)
    if len(keep) > cap:
        keep = sorted(sorted(keep, key=lambda s: (abs(s - p), s))[:cap])
    return keep, edges


def _render(traj: Trajectory, steps: List[int], edges: Dict[int, List],
            p: int) -> str:
    out = []
    for s in steps:
        rel = []
        for kind, label in ((DATA_FLOW, "DATA"), (CONTROL_FLOW, "CONTROL")):
            hits = [e for e in edges.get(s, ()) if e.kind == kind]
            for e in hits[:AUDIT_EDGE_MAX]:
                rel.append("  - %s: %s%s"
                           % (label, e.entity or "(unnamed)",
                              (" -- %s" % e.rationale) if e.rationale else ""))
        st = traj.steps[s]
        out.append("### STEP %d (%s)  ->  STEP %d depends on it via:\n%s\n%s"
                   % (s, st.role, p,
                      "\n".join(rel) or "  - (dependency not described)",
                      clip_middle(st.content or "", AUDIT_PARENT_CLIP)))
    return "\n\n".join(out)


def audit(traj: Trajectory, graph: FlowGraph, llm: LLM, pred: Prediction,
          cap: int = AUDIT_MAX) -> Prediction:
    """Move the answer to a direct parent, but only on all four conditions.

    A no-op costing no call when there is no answer to audit or the answer has
    no admissible parent -- measured at 3 of 98 TrajAudit rows.
    """
    p = pred.critical_step
    if p is None or not (0 <= p < len(traj.steps)):
        return pred
    kept, edges = _parents(graph, traj, p, cap=cap)
    if not kept:
        return pred

    prompt = AUDIT_PROMPT.format(
        p=p,
        task=clip_middle(traj.task_description, 2000),
        failure_summary=graph.failure_summary or "(not summarised)",
        p_reason=clip_middle(pred.reason or "(no reason recorded)", 800),
        p_role=traj.steps[p].role,
        p_text=clip_middle(traj.steps[p].content or "", AUDIT_FOCAL_CLIP),
        parents=_render(traj, kept, edges, p))

    obj = llm.generate_obj(prompt, max_tokens=AUDIT_BUDGET, system=AUDIT_SYSTEM)
    try:
        to = int(obj.get("to"))
    except (TypeError, ValueError):
        to = None

    kind = str(obj.get("edge_kind") or "").strip().lower()
    obj_name = str(obj.get("object") or "").strip()
    # Condition 2 is checked against the graph, not taken on the model's word:
    # the claimed relation has to be one that actually links the two steps.
    kind_ok = to is not None and any(
        e.kind == kind for e in edges.get(to, ())) if kind in (
            DATA_FLOW, CONTROL_FLOW) else False

    record = {
        "first": p,
        "first_confidence": pred.confidence,
        "parents": kept,
        "proposed": to,
        "move": bool(obj.get("move")),
        "object": obj_name[:300],
        "edge_kind": kind or None,
        "edge_kind_ok": bool(kind_ok),
        "p_adds_new_error": bool(obj.get("p_adds_new_error")),
        "parent_fix_suffices": bool(obj.get("parent_fix_suffices")),
        "in_set": bool(to is not None and to in kept),
        "accepted": False,
        "confidence": 0.0,
        "reason": str(obj.get("reason") or "")[:800],
    }
    try:
        record["confidence"] = max(0.0, min(1.0, float(obj.get("confidence"))))
    except (TypeError, ValueError):
        record["confidence"] = 0.0

    accept = (record["move"]
              and record["in_set"]
              and to != p
              and bool(obj_name)                    # condition 1
              and kind_ok                           # condition 2
              and not record["p_adds_new_error"]    # condition 3
              and record["parent_fix_suffices"])    # condition 4
    record["accepted"] = bool(accept)

    if not accept:
        if record["move"] and to is not None and to != p:
            logger.info("%s: audit proposed step %s but the four conditions "
                        "did not hold; keeping step %d", traj.task_id, to, p)
        pred.audit = record
        return pred

    logger.info("%s: audit moved the answer from step %d up to its parent "
                "step %d (%s: %s)", traj.task_id, p, to, kind, obj_name[:80])
    pred.critical_step = to
    pred.confidence = record["confidence"]
    # The reason has to describe the step now being returned. Naming the step
    # it moved off keeps the record readable as a chain rather than as a
    # contradiction of the localiser's own account.
    pred.reason = ("%s\n[audit: moved off step %d, which only carried "
                   "forward %s introduced here]"
                   % (record["reason"], p, obj_name[:120]))[:1500]
    pred.audit = record
    return pred
