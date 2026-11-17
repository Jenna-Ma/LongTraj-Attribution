"""Expected-behaviour specification, written BEFORE the step is read.

For each candidate step, two independent calls:

  1. SPEC.  A reader is given the original task, the description of the agent
     that produced the run, and everything that had reached the step -- and is
     NOT shown the step. It writes down what the step is REQUIRED to produce.
  2. CHECK. A second reader is given that list and the step as it actually
     happened, and marks each requirement met / violated / unclear.

The counts go to the localiser as evidence.

WHY THE SPEC CALL IS BLIND, AND WHY THAT IS THE WHOLE DESIGN. The obvious
implementation is one call that writes the specification and checks it in the
same breath. That call cannot work: the model has already read the step, so it
writes the requirements the step happens to violate, and the compliance count
becomes a restatement of a judgement it made first. This is the same trap as
telling the localiser to enumerate the traced steps and then answer -- measured
on TrajAudit, the enumeration was 0.979 complete and the accuracy still fell,
because the reader wrote the list to fit an answer it had already reached.
Splitting the stage in two is what makes the requirement independent of the
outcome, and it is the reason this costs two calls per candidate rather than
one.

NEITHER CALL IS TOLD HOW THE RUN ENDED. ``graph.failure_summary`` is available
here and is deliberately withheld from both prompts. Every localisation arm in
this repo so far has opened by naming the symptom, which is correct for a
reader whose job is to trace backwards from it -- and fatal for a reader whose
job is to say what SHOULD have happened, who would simply write a requirement
shaped like the failure. The check call is withheld from it for the same
reason: "does this step satisfy requirement 3" must not quietly become "is this
step the guilty one".

WHAT REACHES THE LOCALISER IS A COUNT, NOT A VERDICT. Of everything measured on
this corpus, the one axis that converted to acc@0 was ``--verify-outdeg``, and
what distinguished it from the six arms that lost was that it handed the merge
a MEASURED QUANTITY -- capped traced out-degree -- instead of another opinion
about which step was critical. "4 of 5 requirements met" is that shape. The
stage never asks any model whether a step is critical, which is also the one
concept the roll-call arm showed this model and the annotation disagree about.

PER CANDIDATE, NOT BATCHED. One call could check every candidate of a
trajectory at once and would cost a tenth as much. It would also put the
candidates in front of each other, which is the defect that sank E5: verdicts
that were produced by comparison are not commensurable as independent
measurements. Each candidate is specified and checked as though it were the
only step in the run.

DEGRADATION. No candidate set means no referent for "the candidate steps" --
the stage records ``skipped`` and returns, rather than inventing a pool. On
TrajAudit that is every row, and it is why this arm is a SWE-Bench-Pro claim.

WHAT THE COMPLIANCE SIGNAL ACTUALLY IS -- measured before the arm was run, on
2 trajectories / 12 candidates / 64 requirements, and worth stating because it
was not what the stage was designed to produce.

The spec reader writes requirements that PREDICT THE AGENT'S NEXT MOVE: "step
50 must edit wrap_var in unsafe_proxy.py". The agent then legitimately edits
the ``__all__`` list first, and every requirement on the form comes back
violated. That reads like a defect, so it was fixed: the reader was told the
next move is not its to guess and to write requirements that hold whatever the
step does ("if it edits <file>, the change must keep <property>"). The result
was that violations vanished -- 13 of 64 became 0, then 1, then 2 across three
wordings, with the violation rate among APPLICABLE requirements going 0.342 ->
0.000 / 0.033 / 0.057. Conditional requirements are nearly always satisfied,
because a step that does something else simply does not trigger them.

So the compliance signal on this corpus IS the next-move prediction. Removing
it does not clean the measurement up, it empties it, and a feature with no
variance cannot help localisation however well posed it is. The prescriptive
wording is therefore the one shipped, with its conflation stated rather than
hidden: "violated" here means "departed from what a reader who knew the task
and the context, but not the outcome, would have expected at this point". That
is not the same as "wrong", and the localiser is told so in as many words.
Whether the conflation is even a defect for THIS corpus is an open question --
SWE-Bench-Pro's reference step is very often one where the agent committed to
an edit a well-informed reader would not have made -- and it is what the run
tests. The three conditional wordings were eliminated without looking at a
single label: a flat feature is useless a priori.

The reference label never enters this module: it takes a ``Trajectory``, which
has no field able to hold one.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from .llm import LLM, map_concurrent
from .schema import FlowGraph, StepDigest, Trajectory, clip_middle

logger = logging.getLogger(__name__)

EXPECT_MAX = 32          # cost ceiling per trajectory; SP's largest set is 31
EXPECT_SPEC_BUDGET = 900
EXPECT_CHECK_BUDGET = 900
EXPECT_BEFORE = 24       # steps of run-up, one line each
EXPECT_LINE = 160        # one line per run-up step
EXPECT_INBOX_CLIP = 2500  # the last thing that came back before the step
EXPECT_PRIOR_ACT_CLIP = 1200   # the agent's own previous action
EXPECT_FOCAL_CLIP = 4000  # the step itself, at check time
EXPECT_RESULT_CLIP = 2000  # what came back immediately after it
EXPECT_SPECS_MIN = 3
EXPECT_SPECS_MAX = 6
EXPECT_MUST_CLIP = 200
EXPECT_EVIDENCE_CLIP = 160

# Fallback for corpora that do not ship an agent description. Stated as an
# absence rather than as an invented architecture: a guessed framework would
# put fictional requirements ("the agent must call submit") into the spec.
NO_FRAMEWORK = ("(not recorded for this corpus -- do not assume any particular "
                "tool set or protocol; require only what the task itself and "
                "the run so far imply)")

MET, VIOLATED, UNCLEAR = "met", "violated", "unclear"
_VERDICTS = (MET, VIOLATED, UNCLEAR)


SPEC_SYSTEM = (
    "You state in advance what one step of an agent run is required to "
    "produce. You have not been shown that step and must not guess at it. "
    "Answer strictly in JSON."
)

SPEC_PROMPT = """An agent is part-way through the task below. Everything that
had reached step {focal} is shown here. STEP {focal} ITSELF IS NOT SHOWN, and
neither is anything that happened after it.

Say what step {focal} is REQUIRED to do.

You are not predicting what the agent will do -- you are writing down the
conditions its next move has to satisfy to still be doing this task correctly.
Derive them from three things and nothing else: the task, the design of the
agent, and what has just come back to it.

THE AGENT THAT PRODUCED THIS RUN:
{framework}

ORIGINAL TASK:
{task}

WHAT THE AGENT HAS DONE SO FAR (one line each):
{before}

THE AGENT'S OWN PREVIOUS ACTION:
{prior_act}

WHAT CAME BACK FROM IT -- the immediate input to step {focal}:
{inbox}

Write between {lo} and {hi} requirements. Each one must be:

  * CHECKABLE against the text of step {focal} alone. "The edit must keep the
    existing call signature" can be checked; "the agent must make progress"
    cannot. Someone who reads only that step and your sentence has to be able
    to say met or violated without knowing anything else.
  * SPECIFIC to this moment. Name the file, the symbol, the value, the error
    that has just come back. A requirement that would fit any step of any run
    carries nothing.
  * INDEPENDENT of how things turn out. You do not know whether this run
    succeeds or fails, and you are not being asked to guess. Do not write a
    requirement whose point is that it is about to be broken.

Requirements the step can satisfy are as informative as ones it cannot. Write
what the task demands here, not what you suspect went wrong.

Return ONLY JSON:
{{
  "expects": [
    {{"id": 1,
      "must": "<the requirement, one sentence, checkable against step {focal}>",
      "why": "<at most 15 words: what it follows from>"}},
    ... between {lo} and {hi} entries
  ]
}}"""


CHECK_SYSTEM = (
    "You check one step of an agent run against requirements written before "
    "that step was seen. You judge compliance only. Answer strictly in JSON."
)

CHECK_PROMPT = """Below is ONE step of an agent run, and a list of requirements
written for that step BEFORE it was seen -- derived from the task, the agent's
design, and what had reached the step.

Go through the requirements one at a time and say whether the step satisfies
each one.

This is a compliance check and nothing more. You are NOT being asked whether
this step was a mistake, whether it caused anything, or how the run ended -- you
have not been told how it ended and should not infer it. A requirement can be
violated by a step that turned out to be harmless, and a step can satisfy every
requirement on this list and still be the wrong move. Answer only what is in
front of you.

The requirements were written without sight of the step, so some of them may
not apply, or may be unanswerable from the step's text. Say "unclear" in that
case. Do not stretch a requirement to fit, and do not mark something violated
because the step did something the list did not anticipate.

ORIGINAL TASK:
{task}

WHAT THE AGENT HAD DONE BEFORE STEP {focal} (one line each):
{before}

REQUIREMENTS WRITTEN FOR STEP {focal}:
{specs}

============================================================
STEP {focal}, AS IT ACTUALLY HAPPENED
============================================================
{focal_text}
============================================================

WHAT CAME BACK IMMEDIATELY AFTER IT:
{result}

Return ONLY JSON, one entry per requirement, in the order listed:
{{
  "checks": [
    {{"id": 1,
      "verdict": "met" or "violated" or "unclear",
      "evidence": "<at most 20 words, from the step itself>"}},
    ...
  ]
}}"""


def _context_lines(traj: Trajectory, dmap: Dict[int, StepDigest],
                   lo: int, hi: int) -> str:
    """One clipped line per step in [lo, hi). Digest where there is one."""
    lines = []
    for i in range(max(0, lo), min(len(traj.steps), hi)):
        d = dmap.get(i)
        text = (d.text if d is not None else (traj.steps[i].content or ""))
        one = " ".join((text or "").split())
        lines.append("  step %d: %s" % (i, clip_middle(one, EXPECT_LINE)
                                        or "(empty)"))
    return "\n".join(lines) or "  (nothing)"


def _inbox(traj: Trajectory, i: int) -> str:
    """The nearest thing that came BACK to the agent before step ``i``.

    Walks back to the last non-assistant step, which in both corpora is the
    tool result or the user turn the step is reacting to. Bounded by the run-up
    window so a step early in the run does not drag the whole prefix in.
    """
    for j in range(i - 1, max(-1, i - 1 - EXPECT_BEFORE), -1):
        st = traj.steps[j]
        if st.role != "assistant":
            return "[step %d | %s]\n%s" % (
                j, st.name or st.role,
                clip_middle(st.content or "", EXPECT_INBOX_CLIP) or "(empty)")
    return ("(nothing came back -- step %d follows directly on the agent's own "
            "previous action)" % i)


def _prior_act(traj: Trajectory, i: int) -> str:
    """The agent's own last action before ``i``. Never ``i`` itself."""
    for j in range(i - 1, max(-1, i - 1 - EXPECT_BEFORE), -1):
        st = traj.steps[j]
        if st.role == "assistant":
            return "[step %d]\n%s" % (
                j, clip_middle(st.content or "", EXPECT_PRIOR_ACT_CLIP)
                or "(empty)")
    return "(this is the agent's first action)"


def _result(traj: Trajectory, i: int) -> str:
    """What came back immediately after step ``i``, if anything did."""
    for j in range(i + 1, min(len(traj.steps), i + 4)):
        st = traj.steps[j]
        if st.role != "assistant":
            return "[step %d | %s]\n%s" % (
                j, st.name or st.role,
                clip_middle(st.content or "", EXPECT_RESULT_CLIP) or "(empty)")
        break
    return "(nothing came back before the agent acted again)"


def _parse_specs(obj) -> List[Dict[str, str]]:
    """Requirements out of one spec answer. Ids are renumbered from 1.

    The model's own ``id`` is not trusted for anything: it is re-issued here so
    that the check call's ids index this list positionally, and a duplicated or
    skipped id in the answer cannot silently misalign a verdict with a
    requirement it was not about.
    """
    rows = (obj or {}).get("expects") if isinstance(obj, dict) else None
    out: List[Dict[str, str]] = []
    for row in (rows or []):
        if not isinstance(row, dict):
            continue
        must = " ".join(str(row.get("must") or "").split())
        if not must:
            continue
        out.append({"id": len(out) + 1,
                    "must": must[:EXPECT_MUST_CLIP],
                    "why": " ".join(str(row.get("why") or "").split())[:120]})
        if len(out) >= EXPECT_SPECS_MAX:
            break
    return out


def _parse_checks(obj, specs: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Verdicts out of one check answer, aligned to ``specs`` by id.

    A requirement with no verdict is recorded as ``unclear`` rather than
    dropped: the denominator of "k of n met" has to be the number of
    requirements that were WRITTEN, not the number the checker got round to,
    or a checker that answered two of six would look maximally compliant.
    """
    rows = (obj or {}).get("checks") if isinstance(obj, dict) else None
    by_id: Dict[int, Dict[str, str]] = {}
    for row in (rows or []):
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        v = str(row.get("verdict") or "").strip().lower()
        if v not in _VERDICTS:
            v = UNCLEAR
        by_id.setdefault(i, {
            "id": i, "verdict": v,
            "evidence": " ".join(str(row.get("evidence") or "").split())
                        [:EXPECT_EVIDENCE_CLIP]})
    return [by_id.get(s["id"], {"id": s["id"], "verdict": UNCLEAR,
                                "evidence": "(no verdict returned)"})
            for s in specs]


def assess(traj: Trajectory, graph: FlowGraph, llm: LLM,
           cands: Optional[Sequence[Dict[str, object]]],
           digests: Optional[Sequence[StepDigest]] = None,
           workers: int = 8) -> Dict[str, object]:
    """Specify then check every candidate step. Two calls per candidate.

    ``graph`` is taken for the signature's sake and for nothing else: the
    symptom is deliberately not shown to either reader -- see the module
    docstring. Passing it keeps this stage's call site uniform with the others
    and makes the omission visible here rather than at the caller.

    Returns a record even when nothing ran, so that a row which skipped is
    distinguishable on disk from one where the stage was off.
    """
    steps: List[int] = []
    for c in (cands or []):
        try:
            i = int(c["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < len(traj.steps) and i not in steps:
            steps.append(i)
    if not steps:
        return {"asked": 0, "rows": [], "capped": False,
                "skipped": "no_candidates"}
    capped = len(steps) > EXPECT_MAX
    steps = steps[:EXPECT_MAX]

    dmap = {d.index: d for d in (digests or [])}
    framework = str((traj.meta or {}).get("framework") or "").strip()
    head = dict(
        task=clip_middle(traj.task_description, 3000) or "(not recorded)",
        framework=clip_middle(framework, 2500) or NO_FRAMEWORK,
    )

    def _spec_ask(i: int) -> str:
        return SPEC_PROMPT.format(
            focal=i, lo=EXPECT_SPECS_MIN, hi=EXPECT_SPECS_MAX,
            before=_context_lines(traj, dmap, i - EXPECT_BEFORE, i),
            prior_act=_prior_act(traj, i),
            inbox=_inbox(traj, i),
            **head)

    specs_by_step = {
        i: _parse_specs(o) for i, o in zip(steps, map_concurrent(
            lambda i: llm.generate_obj(_spec_ask(i),
                                       max_tokens=EXPECT_SPEC_BUDGET,
                                       system=SPEC_SYSTEM),
            steps, workers=max(1, int(workers))))}

    # Only steps that got a usable specification are checked. A step whose
    # spec call came back empty is recorded with n=0 rather than dropped: a
    # missing row and a row with nothing to check are different states and the
    # localiser must not read the first as the second.
    todo = [i for i in steps if specs_by_step.get(i)]

    def _check_ask(i: int) -> str:
        specs = specs_by_step[i]
        return CHECK_PROMPT.format(
            focal=i,
            task=head["task"],
            before=_context_lines(traj, dmap, i - EXPECT_BEFORE, i),
            specs="\n".join("  (%d) %s" % (s["id"], s["must"]) for s in specs),
            focal_text=(clip_middle(traj.steps[i].content or "",
                                    EXPECT_FOCAL_CLIP) or "(empty)"),
            result=_result(traj, i))

    checks_by_step = {
        i: _parse_checks(o, specs_by_step[i])
        for i, o in zip(todo, map_concurrent(
            lambda i: llm.generate_obj(_check_ask(i),
                                       max_tokens=EXPECT_CHECK_BUDGET,
                                       system=CHECK_SYSTEM),
            todo, workers=max(1, int(workers))))}

    rows: List[Dict[str, object]] = []
    for i in steps:
        specs = specs_by_step.get(i) or []
        checks = checks_by_step.get(i) or []
        tally = {v: sum(1 for c in checks if c["verdict"] == v)
                 for v in _VERDICTS}
        rows.append({
            "step": i,
            "n": len(specs),
            "met": tally[MET],
            "violated": tally[VIOLATED],
            "unclear": tally[UNCLEAR],
            "specs": specs,
            "checks": checks,
        })
    logger.info("%s: expected-behaviour check on %d candidate steps "
                "(%d requirements, %d violated)", traj.task_id, len(rows),
                sum(int(r["n"]) for r in rows),
                sum(int(r["violated"]) for r in rows))
    return {"asked": len(steps), "rows": rows, "capped": capped, "skipped": ""}
