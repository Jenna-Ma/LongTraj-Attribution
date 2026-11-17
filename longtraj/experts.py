"""Multi-expert parallel failure attribution.

Replaces the single general-purpose localiser with four independent experts
that run concurrently, each responsible for one family of failures, followed by
a rule-based fusion of their verdicts.

    Compliance  task-constraint deviation, scope / role overreach
    Logic       hallucination, inconsistent reasoning, overconfidence
    Flow        redundant loops, premature termination, tool misuse
    Context     ambiguous communication, ignored context and history

WHY FOUR CALLS AND NOT ONE. The single localiser is not short of evidence: on
SWE-Bench-Pro half the reference steps are already on a painted path and it
still misses half of those (P(hit | gold_on_flow) = 0.50). Its problem is
deciding WHERE to look in a 117-step run. A general prompt has to hold every
failure mode at once and settles on whichever is most salient; a remit narrows
what counts as suspicious, so four narrow readers search four different ways
over the same evidence.

WHY THE EXPERTS READ THE SAME MATERIAL. They differ in remit, not in evidence.
Giving each one a different slice would confound two changes at once -- a
weaker expert could then be weak because its remit is wrong OR because it was
shown less -- and the fusion could not be read. Same input, different question.

WHY FUSION IS A RULE AND NOT A FIFTH CALL. A model that arbitrates four
verdicts is a general-purpose judge again, which is the thing being replaced.
A rule is also free to ablate: every verdict is recorded verbatim in the
prediction, so an alternative fusion can be scored offline against a stored run
without spending a single call.

An expert may decline. ``found: false`` costs it its vote rather than forcing a
guess inside a remit that does not apply -- three experts abstaining on a
straightforward tool-misuse failure is the correct behaviour, not a shortfall.

The ground truth never reaches this module: it is handed a ``Trajectory``, a
``FlowGraph`` and the compression artifacts, none of which can carry a label.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import candidates as candidates_mod
from .llm import LLM, map_concurrent
from .localize import (ANSWER_BUDGET, CANDIDATE_BLOCK, CANDIDATE_BLOCK_SOFT,
                       PRIOR_BLOCK, PRIOR_FIELDS, _fallback_step, _fit_budget,
                       _flow_block, _prior_record, _snap_to_attributable,
                       _with_rejection)
from .compress import render_compressed
from .schema import (CONTROL_FLOW, DATA_FLOW, FlowGraph, Outline, Prediction,
                     Span, SpanPrior, StepDigest, Trajectory, clip_middle)

logger = logging.getLogger(__name__)

# Each expert answers a narrower question than the single localiser did, so it
# needs less room than ANSWER_BUDGET -- but not much less: the reasoning still
# happens before the first character is emitted, and an answer truncated
# mid-JSON is an unmeasured case, not a wrong one. See the note on
# ANSWER_BUDGET in localize.py.
EXPERT_BUDGET = 3000

# Fusion. A vote is worth its confidence at the step it names, and half that at
# an immediately adjacent step.
#
# The neighbour credit is not a smoothing convenience. Two experts landing one
# step apart in a 117-step run have converged on the same region of the
# trajectory, and treating that as total disagreement would throw away the only
# signal multiple readers produce. The radius is 1 rather than a wider window
# because acc@1 and acc@0 are already equal on both corpora -- the localiser
# does not miss by one, it misses by tens -- so anything wider would be
# manufacturing agreement that the measurement says is not there.
VOTE_RADIUS = 1
NEIGHBOUR_WEIGHT = 0.5

EXPERT_SYSTEM = (
    "You are a specialist reviewer of failed agent trajectories. You examine "
    "one specific family of failures and report only what falls inside it. "
    "Answer strictly in JSON."
)


@dataclass(frozen=True)
class ExpertSpec:
    """One expert's identity and remit."""

    key: str                        # stable id, used in diagnostics
    title: str                      # human name shown in the prompt
    remit: str                      # what this expert is responsible for
    vocabulary: Sequence[str]       # the fault categories it may return

    def vocab_line(self) -> str:
        return " | ".join(self.vocabulary)


EXPERTS: List[ExpertSpec] = [
    ExpertSpec(
        key="compliance",
        title="COMPLIANCE",
        remit="""You detect DEVIATION FROM WHAT THE TASK ASKED FOR, and action
taken outside the agent's remit.

Look for:
  * A requirement stated in the task that the agent silently dropped, weakened,
    or satisfied only partially — it implemented four of the five things asked
    for, or handled the general case and not the stated edge case.
  * A constraint the task imposed that the agent violated — an interface it was
    told to keep, a file it was told not to touch, a behaviour it was told to
    preserve for existing callers.
  * Scope overreach: the agent changed things the task did not ask about.
    Refactoring unrelated code, "fixing" something it decided was also wrong,
    or rewriting a component when asked to extend it. A change that is an
    improvement in the abstract is still a deviation if it was not asked for
    and it is what broke the run.
  * Role overreach: the agent decided something that was not its to decide —
    it reinterpreted an ambiguous requirement its own way and proceeded as if
    the interpretation were given, rather than implementing what was written.

NOT yours: reasoning that is internally wrong but stays inside the task's
scope, and steps that carry out a flawed plan faithfully. Those belong to the
LOGIC expert. Judge what was asked against what was done.""",
        vocabulary=("constraint-violation", "requirement-dropped",
                    "scope-overreach", "role-overreach",
                    "spec-reinterpreted"),
    ),
    ExpertSpec(
        key="logic",
        title="LOGIC",
        remit="""You detect REASONING THAT DOES NOT HOLD, independently of
whether it matched the task.

Look for:
  * Hallucination: the agent asserted something about the codebase that is not
    supported by anything it read — a function, field, signature, or behaviour
    it never observed. Check its claims against what the observations actually
    returned. A claim that happens to be true but was never established is
    still a hallucination.
  * Inconsistency: a conclusion that contradicts an earlier observation or an
    earlier conclusion of its own, without the agent noticing the conflict or
    saying what changed its mind.
  * Invalid inference: the conclusion does not follow from the evidence given
    for it. It generalised from one case, confused a sufficient condition for a
    necessary one, or read a passing test as proof of something the test does
    not exercise.
  * Overconfidence: the agent committed to a course of action on evidence that
    does not support the commitment, where a cheap check was available and not
    taken. The tell is the absence of hedging where hedging was warranted —
    "this is clearly the cause", with nothing establishing it.

NOT yours: the agent reasoning correctly toward a goal that was not the one
asked for (COMPLIANCE), or reasoning that was sound when made and only
invalidated by information that arrived later (CONTEXT). Judge the inference
against the evidence that stood behind it at the time.""",
        vocabulary=("hallucination", "inconsistent-reasoning",
                    "invalid-inference", "overconfidence",
                    "unverified-assumption"),
    ),
    ExpertSpec(
        key="flow",
        title="FLOW",
        remit="""You detect PROCESS BREAKDOWN in how the run was executed.

Look for:
  * Premature termination: the agent stopped while its own evidence said the
    work was not done — it submitted with a test still failing, left a branch
    unimplemented it had identified, or declared success without running the
    thing that would have shown otherwise. The critical step is where it
    DECIDED to stop, not the final step that reports stopping.
  * Redundant loops: the agent repeated the same action, or cycled through the
    same few actions, without the state changing between attempts. Retrying a
    failing command with no modification, re-reading a file it had already
    read, alternating between two fixes and undoing each. The critical step is
    where the loop began to substitute for progress.
  * Tool misuse: a call made with wrong arguments, against the wrong target, in
    a form the tool does not accept, or a tool used for something it does not
    do. Also a tool whose error return was treated as a success.
  * Wrong ordering: two actions that are individually correct performed in an
    order that makes the second one useless or destructive — editing before
    reading, testing before building, patching a file it then overwrote.

NOT yours: a well-executed process pursuing the wrong goal (COMPLIANCE), or a
single action that is wrong because the reasoning behind it was wrong (LOGIC).
Judge the shape of the execution.""",
        vocabulary=("premature-termination", "redundant-loop", "tool-misuse",
                    "wrong-action-order", "error-ignored"),
    ),
    ExpertSpec(
        key="context",
        title="CONTEXT",
        remit="""You detect INFORMATION THAT WAS AVAILABLE AND NOT USED, and
communication that did not carry what it needed to.

Look for:
  * Ignored observation: a tool returned something that mattered — an error, a
    warning, a value that contradicts the plan, a file whose contents differ
    from what was assumed — and the next step proceeds as if it had not. This
    is the most common form: the evidence arrived and the agent did not read
    it.
  * Lost history: something established earlier in the run was forgotten and
    then contradicted — a constraint it had discovered, a decision it had made,
    a file it had already modified and then edited again from the original
    assumption.
  * Ambiguous communication: where the trajectory has more than one participant
    or the agent writes an instruction to itself, a message that is open to
    more than one reading, and the reading taken downstream is not the one
    meant. Also an under-specified handoff: a summary that drops the detail the
    next step needed.
  * Stale assumption: the agent carried forward a belief that was true when
    formed and was invalidated by something it later saw, without revisiting
    it.

NOT yours: information the agent never had (nothing was ignored), or a
conclusion wrong on its own terms with the observations read correctly
(LOGIC). Judge what was in front of it against what it used.""",
        vocabulary=("ignored-observation", "lost-history",
                    "ambiguous-communication", "stale-assumption",
                    "dropped-handoff"),
    ),
]


EXPERT_PROMPT = """A coding agent FAILED this task. You are one of four
specialists examining it in parallel. Each of you covers a different family of
failures; the others are handling theirs. Report only what falls inside YOURS.

============================================================
YOUR REMIT — {title}
============================================================
{remit}
============================================================

Find the EARLIEST step whose mistake, OF THE KIND DESCRIBED ABOVE, causally led
to the observed failure.

If nothing in your remit caused this failure, say so: answer with
"found": false. That is a real finding and the correct one when the failure is
of another kind. Do NOT nominate the least-innocent step you can see in order
to have an answer — a wrong nomination made at low confidence still pulls the
fused result toward it. Three of four experts abstaining on a failure that has
a single clear cause is the expected outcome, not a shortfall.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

SYMPTOM STEPS: {anchors}
These are where the failure became VISIBLE — the patch was submitted, the test
reported its error, the run ended. Being where a failure surfaces is not
evidence of having caused it. A symptom step is the end of the causal chain you
are looking for, so start there and walk backwards; do not stop there.

TRACED DEPENDENCIES
Two SEPARATE relations were traced back from the symptom steps. They are
independent: a step may appear in one, both, or neither. Read them apart — a
step is not connected to the failure just because a data edge and a control
edge happen to meet at it.

Each edge is annotated in parentheses with three facts about its position in
the traced graph. They are mechanical properties of the graph, not judgements
about blame, and the edges are NOT ordered by them:
  * "N hops from symptom" — how many edges lie between this edge's SOURCE and
    the nearest symptom step. The chain runs from cause to symptom, so a large
    N means further upstream, i.e. earlier in the causal story.
  * "names a failure artefact" — the concrete thing this data edge carries is
    one the failure summary is about. It says the edge is on-topic, not that
    the step mishandled it.
  * "K chains converge here" — K separate later steps depend on this edge's
    source. Something many things rest on is worth examining, though a widely
    used correct result also converges.
An edge marked "not on a chain to any symptom" was traced but never reaches a
symptom step; treat it as weak evidence.

--- DATA FLOW: where the information used at each step came from ---
An edge "step A -> step B [thing]" means the concrete thing named in brackets
was produced, transformed, or dropped at step A, and step B relied on it.
{data_block}

--- CONTROL FLOW: why the agent went from one step to the next ---
An edge "step A -> step B" means step A caused the agent to take step B: A set
the plan B carries out, or A failed and B is the reaction, or B retries,
reverses, or works around A. Plain sequence is NOT included.
{control_block}

============================================================
PART 1 — HOW THE AGENT EXECUTED THE TASK
A description of the run, phase by phase. It is descriptive only and makes no
claim about what went wrong.
============================================================
{outline}
{span_prior}
============================================================
PART 2 — COMPRESSED TRAJECTORY, STEP BY STEP
Every step of the run is here, under its original index — nothing was dropped.
Steps judged RELATED to the failure keep much more of their detail, while steps
judged largely unrelated are reduced to a summary of what the agent did and
what came back.

That judgement came from the dependency tracing above and it is a heuristic,
not a verdict. A step kept in full is not thereby guilty, and a summarised step
is not thereby innocent — if a summary reads like it hides something inside
your remit, say so rather than passing over it.
============================================================
{trajectory}

HOW TO CHOOSE:
1. Exploration is not an error. While the agent is still investigating
   alternatives, it has not gone wrong. The error begins when it COMMITS to a
   flawed direction — acts on it, or states it as the single path forward.
2. Apply the but-for test: if this step had been done correctly, would the
   observed failure still have happened? If yes, this is not the critical step.
3. Prefer the EARLIEST step that starts the causal chain, not a later step
   where the consequence became visible. If several steps repeat the same
   mistake, name the first one.
4. Ignore mistakes that did not contribute to THIS failure, however real.
5. Stay inside your remit. A failure you can see clearly but that belongs to
   another expert is not your answer — abstain and let them have it. Your
   confidence should reflect how well the failure fits YOUR family, not how
   sure you are that something went wrong somewhere.

Answering with a SYMPTOM STEP is a strong claim: it says the agent did
everything right until the very end and the fault was introduced there. Only
answer with one if you can say what it did wrong BEYOND revealing the failure.
{candidates}
Return ONLY JSON:
{{
  "found": <true if a failure inside YOUR remit caused this, false otherwise>,
  "critical_step": <int, or null if found is false>,
  "confidence": <float 0..1: how well this failure fits YOUR remit>,
  "fault_category": "<one of: {vocab}>",
  "error_type": "<plan|reason|act|obs|verify>.<short subtype>",
  "rejected_step": <int or null: the later step you considered and ruled out>,
  "rejected_because": "<why that step only propagated the fault>",{prior_fields}
  "reason": "<2-4 sentences: what went wrong here, why it falls in your remit,
              and how it caused the failure>"
}}"""


@dataclass
class ExpertVerdict:
    """One expert's answer, recorded before any fusion touches it."""

    key: str
    found: bool = False
    step: Optional[int] = None
    confidence: float = 0.0
    fault_category: str = ""
    error_type: str = ""
    reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"expert": self.key, "found": self.found, "step": self.step,
                "confidence": round(self.confidence, 4),
                "fault_category": self.fault_category,
                "error_type": self.error_type, "reason": self.reason}


def _expert_prompt(spec: ExpertSpec, ev: Dict[str, str]) -> str:
    """One expert's prompt: the shared evidence, plus its own remit."""
    return EXPERT_PROMPT.format(title=spec.title, remit=spec.remit,
                                vocab=spec.vocab_line(), **ev)


def _evidence(traj: Trajectory, spans: List[Span], graph: FlowGraph,
              digests: Sequence[StepDigest], outline: Optional[Outline],
              prior: Optional[SpanPrior], cands: Optional[List[Dict[str, object]]],
              enforce_cands: bool, max_prompt_chars: int) -> Dict[str, str]:
    """The material every expert reads. Built once and shared verbatim.

    Identical across the four calls by construction, so a difference between
    two experts is a difference in remit and cannot be a difference in what
    they were shown.
    """
    body = render_compressed(traj, spans, list(digests))
    if len(body) > max_prompt_chars:
        body = _fit_budget(traj, spans, digests, max_prompt_chars)
    outline_text = outline.render() if outline is not None else ""
    has_prior = bool(prior and prior.guesses)
    return {
        "task": clip_middle(traj.task_description, 3000),
        "failure_summary": graph.failure_summary or "(not summarised)",
        "anchors": (", ".join("step %d" % a for a in sorted(graph.anchors))
                    or "(none identified)"),
        "data_block": _flow_block(graph, DATA_FLOW),
        "control_block": _flow_block(graph, CONTROL_FLOW),
        "outline": clip_middle(outline_text, 12000) or "(no outline available)",
        "span_prior": (PRIOR_BLOCK.format(guesses=prior.render())
                       if has_prior else ""),
        "prior_fields": (PRIOR_FIELDS if has_prior else ""),
        "candidates": ((CANDIDATE_BLOCK if enforce_cands else CANDIDATE_BLOCK_SOFT)
                       .format(cands=candidates_mod.render(cands))
                       if cands else ""),
        "trajectory": body,
    }


def _parse_verdict(spec: ExpertSpec, obj: Dict[str, Any],
                   traj: Trajectory) -> ExpertVerdict:
    """Read one expert's JSON, keeping the raw object for the diagnostics.

    A malformed or empty answer becomes an abstention rather than a failure:
    one expert losing its vote degrades the fusion, it does not stop it.
    """
    v = ExpertVerdict(key=spec.key, raw=(obj if isinstance(obj, dict) else {}))
    if not isinstance(obj, dict) or not obj:
        return v

    step = obj.get("critical_step")
    try:
        step = int(step)
    except (TypeError, ValueError):
        step = None
    step = _snap_to_attributable(step, traj)

    found = obj.get("found")
    if not isinstance(found, bool):
        # Absent or non-boolean: infer it from whether a step was actually
        # named. A model that answered with a step meant to nominate it.
        found = step is not None

    conf = obj.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.0

    cat = str(obj.get("fault_category") or "")[:60]
    if cat and cat not in spec.vocabulary:
        # Off-vocabulary categories are kept, not dropped: the label is
        # descriptive and a model coining a better word for what it saw is
        # information, not an error. Marked so analysis can separate them.
        cat = "%s (off-vocab)" % cat

    v.found = bool(found and step is not None)
    v.step = step
    v.confidence = conf
    v.fault_category = cat
    v.error_type = str(obj.get("error_type") or "")[:80]
    v.reason = _with_rejection(obj)
    return v


def run_experts(traj: Trajectory, spans: List[Span], graph: FlowGraph, llm: LLM,
                digests: Optional[Sequence[StepDigest]] = None,
                outline: Optional[Outline] = None,
                prior: Optional[SpanPrior] = None,
                cands: Optional[List[Dict[str, object]]] = None,
                enforce_cands: bool = True,
                max_prompt_chars: int = 120000,
                workers: int = 4) -> List[ExpertVerdict]:
    """Run all four experts concurrently over one shared body of evidence."""
    ev = _evidence(traj, spans, graph, list(digests or []), outline, prior,
                   cands, enforce_cands, max_prompt_chars)

    def one(spec: ExpertSpec) -> ExpertVerdict:
        obj = llm.generate_obj(_expert_prompt(spec, ev),
                               max_tokens=EXPERT_BUDGET, system=EXPERT_SYSTEM)
        return _parse_verdict(spec, obj, traj)

    out = map_concurrent(one, EXPERTS, workers=max(1, min(workers, len(EXPERTS))))
    return [v if v is not None else ExpertVerdict(key=s.key)
            for v, s in zip(out, EXPERTS)]


def fuse(verdicts: Sequence[ExpertVerdict]) -> Dict[str, Any]:
    """Weighted vote: each verdict is worth its confidence at the step it names.

    A vote also counts at half weight one step either side, because two experts
    landing adjacent have converged on the same region — see VOTE_RADIUS.

    Ties break toward the EARLIER step. The whole method looks for the step that
    STARTED the causal chain, and two steps scoring equally means the evidence
    does not separate them; the earlier one is the one the method is after.

    Returns the winning step, an agreement-scaled confidence, and the full
    scoreboard. Nothing here calls a model, so an alternative rule can be
    scored against a stored run for free.
    """
    voting = [v for v in verdicts if v.found and v.step is not None]
    total_conf = sum(v.confidence for v in voting)
    scores: Dict[int, float] = {}
    for cand in {v.step for v in voting}:
        s = 0.0
        for v in voting:
            d = abs(v.step - cand)
            if d == 0:
                s += v.confidence
            elif d <= VOTE_RADIUS:
                s += v.confidence * NEIGHBOUR_WEIGHT
        scores[cand] = s

    if not scores:
        return {"step": None, "confidence": 0.0, "scores": {}, "backers": [],
                "n_voting": 0, "agreement": 0.0}

    best = max(scores, key=lambda s: (scores[s], -s))
    backers = [v.key for v in voting if abs(v.step - best) <= VOTE_RADIUS]
    # Confidence is the share of the total confidence expressed that ended up
    # behind the winner. One expert at 0.9 with three abstentions scores 1.0:
    # unanimity among those who spoke. Two experts split 0.5/0.5 scores 0.5.
    agreement = (scores[best] / total_conf) if total_conf > 0 else 0.0
    return {"step": best, "confidence": max(0.0, min(1.0, agreement)),
            "scores": {int(k): round(val, 4) for k, val in scores.items()},
            "backers": backers, "n_voting": len(voting),
            "agreement": round(agreement, 4)}


def localize_multi(traj: Trajectory, spans: List[Span], graph: FlowGraph,
                   llm: LLM, digests: Optional[Sequence[StepDigest]] = None,
                   outline: Optional[Outline] = None,
                   prior: Optional[SpanPrior] = None,
                   cands: Optional[List[Dict[str, object]]] = None,
                   enforce_cands: bool = True,
                   max_prompt_chars: int = 120000,
                   workers: int = 4) -> Prediction:
    """Four experts in parallel, then a rule-based fusion of their verdicts.

    Drop-in replacement for ``localize.localize``: same arguments, same
    ``Prediction``, and the same two guards on the answer (attributable steps
    only, and the candidate set applied exactly as the single localiser applies
    it). What changes is how the step was arrived at, which is recorded in
    ``pred.experts`` in full.
    """
    verdicts = run_experts(traj, spans, graph, llm, digests=digests,
                           outline=outline, prior=prior, cands=cands,
                           enforce_cands=enforce_cands,
                           max_prompt_chars=max_prompt_chars, workers=workers)
    fused = fuse(verdicts)
    step = fused["step"]
    conf = fused["confidence"]

    # Whose answer the final one is, for the error_type and the reason. The
    # winning step may be backed by several experts; the most confident of them
    # speaks for it.
    lead = None
    if step is not None:
        backing = [v for v in verdicts
                   if v.found and v.step is not None
                   and abs(v.step - step) <= VOTE_RADIUS]
        if backing:
            lead = max(backing, key=lambda v: (v.confidence, v.step == step))

    if step is None:
        # Every expert abstained or every answer was unusable. Fall back the
        # same way the single localiser does, and say so in the record rather
        # than presenting a structural default as a diagnosis.
        step = _fallback_step(traj, graph)
        conf = 0.0

    cand_record = None
    if cands:
        forced = candidates_mod.snap(cands, step)
        cand_record = {
            "n": len(cands),
            "steps": [int(c["step"]) for c in cands],
            "obeyed": forced == step,
            "enforced": bool(enforce_cands),
            "moved_from": (step if (enforce_cands and forced != step) else None),
        }
        if enforce_cands:
            step = forced

    logger.info("%s: experts %s -> step %s (agreement %.2f, backers %s)",
                traj.task_id,
                ",".join("%s=%s@%.2f" % (v.key, v.step, v.confidence)
                         for v in verdicts),
                step, fused["agreement"], fused["backers"])

    return Prediction(
        task_id=traj.task_id, critical_step=step, confidence=conf,
        reason=(lead.reason if lead is not None
                else "no expert reported a failure in its remit; "
                     "answered from the traced flow"),
        error_type=(lead.error_type if lead is not None else ""),
        span_prior=(_prior_record(lead.raw if lead is not None else {},
                                  prior, step) if prior is not None else None),
        candidates=cand_record,
        experts={
            "verdicts": [v.to_dict() for v in verdicts],
            "fusion": {k: fused[k] for k in
                       ("scores", "backers", "n_voting", "agreement")},
            "lead": (lead.key if lead is not None else None),
            "fault_category": (lead.fault_category if lead is not None else ""),
        },
    )
