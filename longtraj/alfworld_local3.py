"""Adjacent-turn temporal attribution for ALFWorld.

The ordinary localiser remains the incumbent. This optional second look sees
only that turn and its immediate attributable neighbours, with exact local
observations, and may move only inside that closed three-turn set.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .alfworld_ledger import _neighbour_user, observation_of
from .llm import LLM
from .schema import Prediction, Trajectory, clip_middle


SYSTEM = (
    "You resolve which of three adjacent ALFWorld agent turns first introduced "
    "a causal fault. Use only the shown evidence and answer strictly in JSON."
)
MAX_TOKENS = 2600
TURN_CLIP = 5000
OBS_CLIP = 1200
ORIGINS = {"introduced", "inherited", "symptom", "unrelated"}


PROMPT = """An ALFWorld household agent failed this task:
{task}

The existing full-trajectory localiser chose step {incumbent}. Its answer is
the default. Decide only whether the causal boundary belongs to the immediately
previous or next AGENT turn instead.

This is a temporal attribution problem, not a search for the most visibly bad
action. For EVERY candidate classify its role:

- introduced: this turn first creates the failure-relevant false belief,
  misreading, committed plan, or wrong action;
- inherited: its incoming state was already wrong and this turn merely follows
  that state;
- symptom: it only exposes or reports a fault introduced elsewhere;
- unrelated: it did not cause this failure.

The simulator observation itself is not an attributable step. If an action's
result is misread, attribute the fault to the following agent turn that reads
and acts on that result. If a plan was already wrong before an action executed
it, attribute the plan turn rather than the later execution. Repetition alone
is not proof. Keep the incumbent unless a neighbour has concrete evidence of
being `introduced` while the incumbent is only `inherited` or `symptom`.

CLOSED CANDIDATES: {steps}

{blocks}

Return ONLY JSON:
{{"assessments": [
  {{"step": <candidate int>,
    "origin": "introduced|inherited|symptom|unrelated",
    "counterfactual_changes_outcome": <true|false>,
    "evidence": "<specific conflict between incoming observation, claim, plan or action>"}}
 ],
 "critical_step": <one integer from the closed candidates>,
 "move": <true only if a neighbour clearly introduced the fault and the incumbent did not>,
 "confidence": <float 0..1>,
 "reason": "<why this turn owns the boundary rather than its adjacent rivals>"}}
"""


def _clean(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return clip_middle(text, limit) if len(text) > limit else text


def _block(traj: Trajectory, index: int, incumbent: int) -> str:
    pos = next(i for i, step in enumerate(traj.steps) if step.index == index)
    before = observation_of(_neighbour_user(traj.steps, pos, -1))
    after = observation_of(_neighbour_user(traj.steps, pos, 1))
    step = traj.steps[pos]
    return (
        "========== STEP %d%s ==========\n"
        "INCOMING SIMULATOR OBSERVATION:\n%s\n\n"
        "AGENT TURN:\n%s\n\n"
        "RESULTING SIMULATOR OBSERVATION:\n%s"
        % (index, " | INCUMBENT" if index == incumbent else "",
           _clean(before, OBS_CLIP) or "(unparsed)",
           _clean(step.content, TURN_CLIP) or "(empty)",
           _clean(after, OBS_CLIP) or "(end/unparsed)"))


def _assessment_map(obj: Dict[str, Any], allowed: List[int]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    values = obj.get("assessments")
    for row in values if isinstance(values, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            step = int(row.get("step"))
        except (TypeError, ValueError):
            continue
        origin = str(row.get("origin") or "").strip().lower()
        if step not in allowed or origin not in ORIGINS or step in out:
            continue
        out[step] = {
            "step": step, "origin": origin,
            "counterfactual_changes_outcome": bool(
                row.get("counterfactual_changes_outcome")),
            "evidence": str(row.get("evidence") or "")[:1200],
        }
    return out


def refine(traj: Trajectory, llm: LLM, incumbent: Prediction) -> Prediction:
    attributable = traj.attributable_indices
    base = incumbent.critical_step
    if base not in attributable:
        return incumbent
    pos = attributable.index(base)
    allowed = attributable[max(0, pos - 1):min(len(attributable), pos + 2)]
    if len(allowed) < 2:
        return incumbent

    prompt = PROMPT.format(
        task=_clean(traj.task_description, 2000), incumbent=base,
        steps=", ".join(str(x) for x in allowed),
        blocks="\n\n".join(_block(traj, x, base) for x in allowed))
    raw = llm.generate_obj(prompt, max_tokens=MAX_TOKENS, system=SYSTEM)
    assessments = _assessment_map(raw, allowed)
    try:
        proposed = int(raw.get("critical_step"))
    except (TypeError, ValueError):
        proposed = None

    chosen = base
    accepted = False
    refused = ""
    if proposed is not None and proposed != base:
        new = assessments.get(proposed) or {}
        old = assessments.get(base) or {}
        conditions = {
            "closed": proposed in allowed,
            "move": raw.get("move") is True,
            "new_introduced": new.get("origin") == "introduced",
            "new_counterfactual": new.get("counterfactual_changes_outcome") is True,
            "incumbent_not_origin": old.get("origin") in {"inherited", "symptom"},
            "complete": len(assessments) == len(allowed),
        }
        accepted = all(conditions.values())
        if accepted:
            chosen = proposed
        else:
            refused = ",".join(k for k, value in conditions.items() if not value)

    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = incumbent.confidence
    reason = (str(raw.get("reason") or incumbent.reason)[:1800]
              if accepted else incumbent.reason)
    return Prediction(
        task_id=traj.task_id, critical_step=chosen,
        confidence=(confidence if accepted else incumbent.confidence),
        reason=reason, error_type=incumbent.error_type,
        span_prior=incumbent.span_prior, candidates=incumbent.candidates,
        experts={
            "kind": "alfworld_local3",
            "incumbent": {"step": base, "confidence": incumbent.confidence,
                          "error_type": incumbent.error_type,
                          "reason": incumbent.reason},
            "candidate_steps": allowed,
            "assessments": [assessments[k] for k in allowed if k in assessments],
            "raw_choice": proposed, "raw_move": raw.get("move"),
            "accepted": accepted, "refused": refused,
        },
    )

