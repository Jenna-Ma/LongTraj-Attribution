"""Cognitive-module audits for ALFWorld failure localisation.

Each auditor receives a different view of the same turns. This avoids asking a
single reader to compare memory omissions, reflection errors, search strategy,
and malformed actions with one undifferentiated criterion. The final call sees
only the incumbent and the auditors' validated nominations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .alfworld_ledger import (_neighbour_user, _normalise_action, _tag,
                              admissible_of, observation_of)
from .llm import LLM, map_concurrent
from .schema import Prediction, Trajectory, clip_middle

logger = logging.getLogger(__name__)

AUDIT_BUDGET = 1800
ARBITER_BUDGET = 2200
FIELD_CLIP = 1800
OBS_CLIP = 800

AUDIT_SYSTEM = (
    "You audit one cognitive module in an ALFWorld household-agent run. "
    "Use only the evidence shown for that module and answer strictly in JSON."
)
ARBITER_SYSTEM = (
    "You compare a small, closed set of proposed critical steps in a failed "
    "ALFWorld run. You may choose only one listed step. Answer strictly in JSON."
)


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    question: str


SPECS: Tuple[ModuleSpec, ...] = (
    ModuleSpec("memory", """Audit only <memory>. Compare each written memory
with the earlier action/observation timeline printed beside it. Look for the
EARLIEST turn where task-relevant established information was dropped,
fabricated, or contradicted and the plan then relied on that faulty memory.
Ordinary selective summarisation is not an error. Abstain if no memory fault
caused the final failure."""),
    ModuleSpec("reflection", """Audit only <reflection>. Compare it with the
PREVIOUS ACTION and BEFORE observation, which is that action's actual result.
Look for the EARLIEST turn that treats failure as success, success as failure,
or declares progress/completion inconsistent with the state. Do not blame the
simulator observation itself. Abstain if reflection was not causal."""),
    ModuleSpec("plan", """Audit only <plan>. Compare it with the task, current
observation, prior action/result timeline and emitted action. Look for the
EARLIEST commitment to an impossible or constraint-violating plan, a wrong
object/substitute, or a systematically inefficient search after a better
task-relevant option was available. Exploration alone is not enough; identify
where it became the committed strategy. Abstain if planning was not causal."""),
    ModuleSpec("action_system", """Audit the emitted <action> and simulator
result. `admissible=NO` is code-checked evidence of an ACTION fault. A correct,
admissible, correctly sequenced action followed by `Nothing happens.` may be a
SYSTEM fault, but refusal alone does not prove that. Find the EARLIEST causal
action/system event, or abstain. Do not diagnose memory, reflection, or plan."""),
)


@dataclass
class Turn:
    step: int
    before: str
    memory: str
    reflection: str
    plan: str
    action: str
    after: str
    admissible: str
    repeated_action: bool
    repeated_observation: bool


@dataclass
class AuditVerdict:
    module: str
    found: bool = False
    step: Optional[int] = None
    confidence: float = 0.0
    category: str = ""
    reason: str = ""
    evidence: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module, "found": self.found, "step": self.step,
            "confidence": round(self.confidence, 4),
            "category": self.category, "reason": self.reason,
            "evidence": self.evidence,
        }


def _clean(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return clip_middle(value, limit) if len(value) > limit else value


def turns(traj: Trajectory) -> List[Turn]:
    out: List[Turn] = []
    seen_actions = set()
    seen_observations = set()
    for pos, step in enumerate(traj.steps):
        if not step.attributable:
            continue
        before_text = _neighbour_user(traj.steps, pos, -1)
        after_text = _neighbour_user(traj.steps, pos, 1)
        action = _tag(step.content, "action", 10000)
        action_key = _normalise_action(action)
        allowed = admissible_of(before_text)
        allowed_keys = {_normalise_action(x) for x in allowed}
        if not action_key or not allowed:
            admissibility = "unknown"
        elif action_key in allowed_keys:
            admissibility = "yes"
        else:
            admissibility = "NO"
        before = observation_of(before_text)
        before_key = " ".join(before.split()).casefold()
        out.append(Turn(
            step=step.index,
            before=_clean(before, OBS_CLIP),
            memory=_clean(_tag(step.content, "memory", 10000), FIELD_CLIP),
            reflection=_clean(_tag(step.content, "reflection", 10000),
                              FIELD_CLIP),
            plan=_clean(_tag(step.content, "plan", 10000), FIELD_CLIP),
            action=_clean(action, 300),
            after=_clean(observation_of(after_text), OBS_CLIP),
            admissible=admissibility,
            repeated_action=bool(action_key and action_key in seen_actions),
            repeated_observation=bool(before_key and
                                      before_key in seen_observations),
        ))
        if action_key:
            seen_actions.add(action_key)
        if before_key:
            seen_observations.add(before_key)
    return out


def _render_view(items: Sequence[Turn], module: str) -> str:
    blocks: List[str] = []
    previous_action = "(none; first turn)"
    for t in items:
        common = (
            "[step %d]\nPREVIOUS ACTION: %s\nBEFORE: %s\nACTION: %s\n"
            "AFTER: %s\nCHECKS: admissible=%s; repeated_action=%s; "
            "repeated_observation=%s"
            % (t.step, previous_action, t.before or "(unparsed)",
               t.action or "(none)", t.after or "(end/unparsed)",
               t.admissible, str(t.repeated_action).lower(),
               str(t.repeated_observation).lower()))
        if module == "memory":
            body = common + "\nMEMORY: " + (t.memory or "(none)")
        elif module == "reflection":
            body = common + "\nREFLECTION: " + (t.reflection or "(none)")
        elif module == "plan":
            body = common + "\nPLAN: " + (t.plan or "(none)")
        else:
            body = common
        blocks.append(body)
        previous_action = t.action or "(none)"
    return "\n\n".join(blocks)


AUDIT_PROMPT = """An ALFWorld household agent failed this task:
{task}

You are the {module} auditor. The 30 original agent step indices are preserved.

YOUR TEST
{question}

MODULE-SPECIFIC TIMELINE
{timeline}

Return ONLY JSON:
{{"found": <true only if this module caused the final failure>,
  "critical_step": <one shown step or null>,
  "confidence": <float 0..1>,
  "category": "<short subtype>",
  "evidence": "<the concrete observation and claim/action that conflict>",
  "reason": "<2-3 sentences explaining the causal link; empty if abstaining>"}}
"""


def _parse(spec: ModuleSpec, obj: Any, allowed: Sequence[int]) -> AuditVerdict:
    raw = obj if isinstance(obj, dict) else {}
    try:
        step = int(raw.get("critical_step"))
    except (TypeError, ValueError):
        step = None
    found = bool(raw.get("found")) and step in set(allowed)
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return AuditVerdict(
        module=spec.key, found=found, step=(step if found else None),
        confidence=(confidence if found else 0.0),
        category=str(raw.get("category") or "")[:80],
        reason=str(raw.get("reason") or "")[:1200] if found else "",
        evidence=str(raw.get("evidence") or "")[:1200] if found else "",
        raw=raw,
    )


def audit_modules(traj: Trajectory, llm: LLM,
                  workers: int = 4) -> Tuple[List[Turn], List[AuditVerdict]]:
    items = turns(traj)
    allowed = [t.step for t in items]
    asks = [(spec, AUDIT_PROMPT.format(
        task=_clean(traj.task_description, 2000), module=spec.key,
        question=spec.question, timeline=_render_view(items, spec.key)))
            for spec in SPECS]
    replies = map_concurrent(
        lambda pair: llm.generate_obj(pair[1], max_tokens=AUDIT_BUDGET,
                                      system=AUDIT_SYSTEM),
        asks, workers=min(max(1, workers), len(asks)))
    return items, [_parse(asks[i][0], replies[i], allowed)
                   for i in range(len(asks))]


def _candidate_context(items: Sequence[Turn], step: int) -> str:
    pos = next((i for i, t in enumerate(items) if t.step == step), None)
    if pos is None:
        return "(missing)"
    blocks = []
    for t in items[max(0, pos - 1):min(len(items), pos + 2)]:
        blocks.append(
            "[step %d%s]\nBEFORE: %s\nMEMORY: %s\nREFLECTION: %s\n"
            "PLAN: %s\nACTION: %s\nAFTER: %s\nCHECKS: admissible=%s; "
            "repeated_action=%s; repeated_observation=%s"
            % (t.step, " | CANDIDATE" if t.step == step else " | context",
               t.before or "(unparsed)", t.memory or "(none)",
               t.reflection or "(none)", t.plan or "(none)",
               t.action or "(none)", t.after or "(end/unparsed)",
               t.admissible, str(t.repeated_action).lower(),
               str(t.repeated_observation).lower()))
    return "\n\n".join(blocks)


ARBITER_PROMPT = """An ALFWorld household agent failed this task:
{task}

Choose the critical step from this CLOSED candidate set: {steps}.
The incumbent is the existing full-trajectory localiser's answer. The module
auditors used narrower evidence and may be wrong or may abstain. Keep the
incumbent unless another listed step has more concrete evidence that it is the
EARLIEST causal fault. A refusal, repetition, or confident audit is not enough
by itself. You may not name an unlisted step.

PROPOSALS
{proposals}

EXACT LOCAL CONTEXT FOR EACH DISTINCT CANDIDATE
{contexts}

Return ONLY JSON:
{{"critical_step": <one integer from the closed set>,
  "confidence": <float 0..1>,
  "error_type": "<memory|reflection|plan|action|system>.<subtype>",
  "incumbent_rejected_because": "<empty if kept; concrete comparative reason if moved>",
  "reason": "<2-4 sentences explaining why this is earlier and more causal than the other proposals>"}}
"""


def refine(traj: Trajectory, llm: LLM, incumbent: Prediction,
           workers: int = 4) -> Prediction:
    items, verdicts = audit_modules(traj, llm, workers=workers)
    base = incumbent.critical_step
    valid = {t.step for t in items}
    candidate_steps = ([base] if base in valid else []) + [
        v.step for v in verdicts if v.found and v.step in valid]
    candidate_steps = sorted(set(int(s) for s in candidate_steps))
    if not candidate_steps:
        candidate_steps = [items[0].step] if items else [0]

    proposals = ["INCUMBENT: step %s | confidence %.2f | %s"
                 % (base, incumbent.confidence, incumbent.reason)]
    proposals += [
        "%s AUDIT: %s"
        % (v.module, ("step %d | confidence %.2f | evidence: %s | reason: %s"
                      % (v.step, v.confidence, v.evidence, v.reason))
                     if v.found else "ABSTAIN")
        for v in verdicts]

    arbiter_raw: Dict[str, Any] = {}
    if len(candidate_steps) == 1:
        chosen = candidate_steps[0]
        confidence = incumbent.confidence if chosen == base else 0.0
        reason = incumbent.reason if chosen == base else "single valid nomination"
        error_type = incumbent.error_type if chosen == base else ""
    else:
        prompt = ARBITER_PROMPT.format(
            task=_clean(traj.task_description, 2000),
            steps=", ".join(str(s) for s in candidate_steps),
            proposals="\n".join(proposals),
            contexts="\n\n==========\n\n".join(
                _candidate_context(items, s) for s in candidate_steps))
        obj = llm.generate_obj(prompt, max_tokens=ARBITER_BUDGET,
                               system=ARBITER_SYSTEM)
        arbiter_raw = obj if isinstance(obj, dict) else {}
        try:
            proposed = int(arbiter_raw.get("critical_step"))
        except (TypeError, ValueError):
            proposed = None
        chosen = proposed if proposed in candidate_steps else (
            base if base in candidate_steps else candidate_steps[0])
        try:
            confidence = max(0.0, min(1.0,
                float(arbiter_raw.get("confidence", incumbent.confidence))))
        except (TypeError, ValueError):
            confidence = incumbent.confidence
        reason = str(arbiter_raw.get("reason") or incumbent.reason)[:1800]
        error_type = str(arbiter_raw.get("error_type") or
                         incumbent.error_type)[:80]

    incumbent_record = {
        "step": base, "confidence": incumbent.confidence,
        "error_type": incumbent.error_type, "reason": incumbent.reason,
    }
    return Prediction(
        task_id=traj.task_id, critical_step=chosen, confidence=confidence,
        reason=reason, error_type=error_type,
        span_prior=incumbent.span_prior, candidates=incumbent.candidates,
        experts={
            "kind": "alfworld_modules", "incumbent": incumbent_record,
            "verdicts": [v.to_dict() for v in verdicts],
            "candidate_steps": candidate_steps,
            "arbiter": arbiter_raw,
            "moved": chosen != base,
        },
    )

