"""Final stage: pick the critical step from the two compression artifacts.

The localiser reads three things: the traced flow graph, the high-level
outline of how the task was executed (artifact 1), and the step-level
compressed trajectory (artifact 2). The outline gives it the shape of the run;
the trajectory gives it the evidence, with every step index intact so any
claim can be pinned to a concrete step.

Two guards on the answer:
  * the returned index must be attributable — user/system turns state the task
    or the environment's input and cannot themselves be the agent's mistake;
  * if the model answers with a step that is not on any traced flow, that is
    allowed but recorded, since it means the tracer missed the dependency.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import alfworld_flownote
from . import alfworld_prompt
from . import candidates as candidates_mod
from .compress import render_compressed
from .llm import LLM, map_concurrent
from .schema import (COMPRESSED, CONTROL_FLOW, DATA_FLOW, FlowGraph, Outline,
                     Prediction, Span, SpanPrior, StepDigest, Trajectory,
                     clip_middle)

logger = logging.getLogger(__name__)

# Output budget for the localiser's JSON answer.
#
# This is not a "roughly big enough" guess. At 1600 the full-corpus run lost 20
# answers outright: the model was still writing when the ceiling hit, the
# doubling ladder in llm.py could not recover them within its retries, and each
# one collapsed to the deterministic fallback and scored zero. Those cases are
# not merely wrong, they are unmeasured — the localiser never got to answer.
#
# The answer is short prose but there are six fields, two of which ask for a
# justification, and a reasoning model spends budget before emitting the first
# character. Sizing this so the common case never touches the ceiling costs
# nothing when it is not needed: an unused budget is not billed.
ANSWER_BUDGET = 4000

LOCALIZE_SYSTEM = (
    "You perform root-cause localisation on failed agent trajectories. "
    "You identify the earliest step whose error causally led to the observed "
    "failure. Answer strictly in JSON."
)

LOCALIZE_PROMPT = """{head}

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}{symptom_evidence}

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
    N means further upstream, i.e. earlier in the causal story. You are looking
    for the earliest step that is still genuinely responsible, so a high N is
    where to look first — but a high N also means more inference stood between
    it and the failure, and a long chain can drift into steps that merely came
    before. Verify the link, do not assume it from the number.
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
was produced, transformed, or dropped at step A, and step B relied on it. This
is how a wrong value, a wrong file, a wrong field name, or a missing piece of
information travels forward. Follow it to find where the information FIRST
became wrong, as opposed to where it was merely passed along.
{data_block}

--- CONTROL FLOW: why the agent went from one step to the next ---
An edge "step A -> step B" means step A caused the agent to take step B: A set
the plan B carries out, or A failed and B is the reaction, or B retries,
reverses, or works around A. Plain sequence is NOT included — "A finished so B
ran next" is not an edge here. This is how a wrong DECISION propagates, even
when every individual action is well-formed. Follow it to find where the agent
committed to the wrong direction.
{control_block}

============================================================
PART 1 — HOW THE AGENT EXECUTED THE TASK
A description of the run, phase by phase. Use it to understand what the agent
was trying to do and when it changed direction. It is descriptive only and
makes no claim about what went wrong.
============================================================
{outline}
{span_prior}
============================================================
PART 2 — COMPRESSED TRAJECTORY, STEP BY STEP
Every step of the run is here, under its original index — nothing was dropped.
The steps have been compressed unevenly on purpose: steps judged RELATED to the
failure keep much more of their detail, while steps judged largely unrelated to
it are reduced to a summary of what the agent did and what came back.

That judgement came from the dependency tracing above and it is a heuristic,
not a verdict. It says where the evidence was preserved, not where the mistake
is. A step kept in full is not thereby guilty, and a summarised step is not
thereby innocent — if a summary reads like it hides something that matters,
say so rather than passing over it.
============================================================
{trajectory}

HOW TO CHOOSE:
1. Exploration is not an error. While the agent is still investigating
   alternatives, it has not gone wrong. The error begins when it COMMITS to a
   flawed direction — acts on it, or states it as the single path forward.
2. Apply the but-for test: if this step had been done correctly, would the
   observed failure still have happened? If yes, this is not the critical step.
3. Prefer the EARLIEST step that starts the causal chain, not a later step
   where the consequence became visible. The symptom steps above are endpoints,
   not causes. If several steps repeat the same mistake, name the first one.
4. Ignore mistakes that did not contribute to THIS failure.
5. A step can be critical because of flawed reasoning or a flawed plan, not
   only because of a wrong action. A step whose action is well-formed can still
   be critical if the decision behind it was wrong.
6. A step reduced to a summary lost detail. If your reasoning depends on one,
   say so in your answer rather than treating the summary as the whole story.
7. Before answering, walk the DATA flow back from the symptom and ask at each
   step: was the information already wrong when it arrived here? Keep going
   while the answer is yes. The step where the answer first becomes no — the
   step that introduced the fault rather than passing it on — is your answer.
   Then check the CONTROL flow for that step's ancestors: if an earlier step
   set the direction that made this fault inevitable, that earlier step is the
   better answer. Name the last step you rejected and why.

Answering with a SYMPTOM STEP is a strong claim: it says the agent did
everything right until the very end and the fault was introduced there. That is
sometimes true — a correct plan can be botched at the moment it is written out.
But usually something earlier is responsible and the symptom step merely
reports it. Only answer with a symptom step if you can say what it did wrong
BEYOND revealing the failure.
{candidates}{expect_block}{framing_note}{roll_block}{analyst_block}{judge_block}{rootcause_block}
Return ONLY JSON:
{{{roll_fields}
  "critical_step": <int>,
  "confidence": <float 0..1>,
  "error_type": "<plan|reason|act|obs|verify>.<short subtype>",
  "rejected_step": <int or null: the later step you considered and ruled out>,
  "rejected_because": "<why that step only propagated the fault>",{prior_fields}{rootcause_fields}
  "reason": "<2-4 sentences: what went wrong here and how it caused the failure>"
}}"""


# --- the same question, asked as different decision procedures -------------
# Five interventions have now tried to improve the CHOICE by improving what the
# reader is shown -- more original text (twice), a second reading, a tournament,
# an explicit comparison field. All five lost. What none of them changed is the
# QUESTION, which has been one sentence, unchanged, since the first run.
#
# These are alternative decision procedures over identical evidence and an
# identical answer space. That last property is the one E5 failed on: its groups
# each saw a different subset, so "this group answered and that one abstained"
# was not a comparison. Here every framing chooses from the same set and returns
# the same fields, so the answers are commensurable by construction and a
# majority means what it says.
#
# Entry 0 is the incumbent, reproduced to the byte -- its head is the literal
# text the template used to open with and its note is empty, spliced with no
# newline of its own. So a run with framings=1 sends the prompt it always sent,
# pays nothing for the option existing, and the A/B measures the added framings
# and nothing else.
INCUMBENT_HEAD = """A coding agent FAILED this task. Find the critical
step: the EARLIEST step whose mistake causally led to the observed failure."""

THINKING_HEAD = """A coding agent FAILED this task.

BEFORE answering, think through these questions step-by-step:

1. FAILURE ANALYSIS: What exactly went wrong? What was the agent trying to do,
   and what did it do instead?

2. TRACE BACKWARDS: Starting from the symptom, trace the data/control flow
   backwards. At each step ask: was the information already corrupted when it
   arrived here, or did THIS step introduce the fault?

3. CANDIDATE REVIEW: For each plausible candidate step, evaluate:
   - What decision/action did this step take?
   - Was there enough information at this point to make the right decision?
   - If this step had been done correctly, would the failure still occur?
   - Is this the EARLIEST step in the causal chain, or just where the
     consequence became visible?

4. ALTERNATIVE PATHS: Could the agent have recovered from this mistake in later
   steps? If yes, why didn't it?

After working through these questions, find the critical step: the EARLIEST
step whose mistake causally led to the observed failure."""

REPAIR_HEAD = """A coding agent FAILED this task. You may go back and redo
EXACTLY ONE step. Which step do you redo so that the task succeeds?"""

REPAIR_NOTE = """
THE QUESTION, RESTATED. You are not grading the run and not ranking its
mistakes by how bad they look. You get one intervention. Name the step at which
spending it makes the task succeed.

A step is the wrong place to spend it if redoing it correctly still leaves the
agent heading for the same failure -- that step was carrying the fault, not
introducing it, and a later step would simply reintroduce it. A step is also
the wrong place if the run was already unrecoverable before it. The right step
is the first one where a different choice was still available AND taking it
would have changed the outcome."""

COMMIT_HEAD = """A coding agent FAILED this task. Find the step at which the
run stopped being recoverable."""

COMMIT_NOTE = """
THE QUESTION, RESTATED. Read the run forwards, not backwards from the failure.
Early on the agent has many continuations available and most of them still
succeed; by the end it has none. Somewhere in between it made a commitment that
closed off every successful continuation -- after that step, no amount of
competent work that did not UNDO it could have finished the task.

Name that step. It is not necessarily where something looked wrong: a
commitment can be made in a confident, well-formed, entirely plausible-looking
step, and the consequences can stay invisible for a long time afterwards. It is
also not the last step where things still looked fine -- it is the first one
after which they could not be made fine again."""

# (name, head, note). Name is recorded in the artifact so a merge rule can be
# re-scored offline against a stored run.
FRAMINGS = (
    ("cause", INCUMBENT_HEAD, ""),
    ("repair", REPAIR_HEAD, REPAIR_NOTE),
    ("commit", COMMIT_HEAD, COMMIT_NOTE),
)


PRIOR_BLOCK = """
------------------------------------------------------------
A SEPARATE READER'S GUESS AT WHERE THE CAUSE LIES
Another model read ONLY the phase descriptions above — no step detail, no
dependency graph — and guessed which phases may contain the critical step:
{guesses}

This is a hint about where to look FIRST, not an answer, and it is a guess made
without the evidence you are about to read. It can be wrong in both directions:
a phase it did not name is not thereby cleared, and a phase it called "likely"
is not thereby guilty. If the evidence below points elsewhere, follow the
evidence and say why in "prior_disagreement".
------------------------------------------------------------"""

PRIOR_FIELDS = """
  "prior_span": <int or null: the start index of the guessed phase your answer
                 falls in, or null if your answer lies outside all of them>,
  "prior_disagreement": "<if null: why the evidence took you outside them>","""


# --- the same prior, with the ranking taken out --------------------------
# Measured on TrajAudit (98 rows, span_prior arm, zero-call decomposition of
# the stored artifacts). Split the rows by where the reference step sits
# relative to the prior's guesses, and by where the answer landed:
#
#   gold in the TOP guess, answer in the TOP guess    n=36   acc 0.806
#   gold in a LATER guess, answer in the TOP guess    n= 9   acc 0.000
#   gold in a LATER guess, answer in a LOWER guess    n= 9   acc 0.889
#   gold in NO guess,      answer in the TOP guess    n=13   acc 0.000
#   gold in NO guess,      answer OUTSIDE the prior   n=13   acc 0.538
#   no prior at all                                   n= 9   acc 0.778
#
# When the top guess is wrong the localiser scores 0/22. Rows just as hard
# where it escaped the prior score 0.538, and rows with no prior at all score
# 0.778 -- so the information needed to answer them is in the prompt and the
# localiser can use it, until a ranked hint tells it where to stop. Of the 93
# records carrying a prior, the reference step is inside SOME guess in 62
# (0.667) but inside the TOP one in only 42 (0.452): 20 rows sit in a region
# the prior correctly nominated and the localiser never considered, and a
# perfect re-rank of exactly those rows projects acc@0 0.580.
#
# No re-rank is available. Every zero-call ordering rule tried lost to the
# model's own order (likelihood-then-earliest 37/56, as-is 36/56, narrowest
# 32/56, earliest 25/56, widest 17/56, latest 16/56), so the order carries
# real signal and cannot be mechanically overridden. What CAN be changed is
# whether the order is shown at all. This block hands over the same regions in
# trajectory order with no likelihood label and no implied rank.
#
# IT WAS RUN AND IT LOST, and the reasoning above is refuted rather than merely
# unconfirmed. acc@0 0.500 -> 0.398 (dev -4, test -6, 3 wins / 13 losses), with
# 5c byte-identical (prior_recall, top_recall and coverage all moved +0.0000).
# The mechanism is the opposite of the one hypothesised: with the rank hidden
# the localiser landed on the model's own top region MORE often, not less
# (0.731 -> 0.803 of the answers that fell inside a multi-guess prior), and it
# did not substitute a positional anchor (earliest-region 0.388 -> 0.408). So
# the 0/22 is not the word "likely" pinning the reader down -- delete the word
# and it goes to the same region anyway. Having a shortlist at all is what
# does it. The cost landed somewhere the hypothesis did not predict: on the 28
# rows whose gold was in no guess (6 -> 1) and the 12 where the baseline had
# escaped the prior entirely (6 -> 1), i.e. "this whole list is wrong, look
# elsewhere" is the behaviour that asking for an even-handed weighing removed.
# acc@0_given_prior_miss 0.179 -> 0.036 is that in one number.
#
# One confound not separated: the unranked block is 178 chars longer and more
# emphatic. An equal-length control was not run, because the full ablation
# below makes the axis moot.
#
# For what it is worth in the other direction, --span-prior removed ENTIRELY
# costs 13 rows (0.500 -> 0.367). The prior is a local maximum boxed in on
# every side tried: sharpening it lost, broadening it lost, hiding its order
# lost, removing it lost.
#
# That last claim is narrower than it reads, and ``hide_prior`` exists to test
# the gap. By the time --prior-compress existed the prior had TWO consumers:
# it allocates the 90000-char compression budget at 3b (``prior-likely`` keeps
# 12000 chars verbatim, ``prior-possible`` clips to 2500/1200), and it is shown
# to the reader here. ``ta_noprior`` turned off the flag that feeds BOTH, on
# claude-opus-4-8, before 3b was written -- so its -13 rows cannot say how much
# was the lost hint and how much was the lost verbatim protection of the very
# region the hint named. ``hide_prior`` suppresses this block and PRIOR_FIELDS
# while 3b still shapes the plan, which is the one cut that separates them.
#
# It does NOT touch ``_fallback_step``, which reads ``prior`` directly rather
# than through ``has_prior``: the fallback is not a prompt, and leaving it
# alone keeps the arm a one-variable change.
PRIOR_BLOCK_UNRANKED = """
------------------------------------------------------------
A SEPARATE READER'S SHORTLIST OF PHASES TO EXAMINE
Another model read ONLY the phase descriptions above — no step detail, no
dependency graph — and shortlisted these phases as worth examining closely:
{guesses}

They are listed in trajectory order and are NOT ranked: that reader did not say
which is likelier, and you should not assume the first one is. Treat them as an
unordered set of equally open possibilities and let the evidence below decide
between them — weigh each one against the others before you answer. The
shortlist can also be wrong in both directions: a phase it did not name is not
thereby cleared, and being on the list is not evidence of guilt. If the
evidence points outside all of them, follow the evidence and say why in
"prior_disagreement".
------------------------------------------------------------"""


def render_prior_unranked(prior: SpanPrior) -> str:
    """The prior's regions in trajectory order, stripped of rank and label.

    Two things are removed relative to ``SpanPrior.render()``: the likelihood
    word, which on this corpus is an exclusive label (82 of 93 priors name
    exactly one "likely" guess), and the model's own ordering, which the reader
    can otherwise recover from the line order. ``why`` is kept -- it is the
    reason to look, not a claim about rank, and dropping it would confound this
    arm with an ablation of the prior's content.
    """
    lines = []
    for g in sorted(prior.guesses, key=lambda x: (x.start, x.end)):
        head = "steps %d-%d" % (g.start, g.end)
        lines.append("%s — %s" % (head, g.why.strip()) if g.why else head)
    return "\n".join(lines)

# --- calling the roll ------------------------------------------------------
# Every earlier attempt at this gap assumed the reader CHOOSES badly and tried
# to help it choose: more original text, a second reading, a tournament, an
# explicit comparison, four specialists, a pointwise poll. All of them lost.
# The conditional accuracy says the premise was wrong. Given the reference step
# is on the traced flow -- verbatim in the prompt, nothing left to find -- it is
# still picked only 0.569 of the time on TrajAudit and 0.571 on SWE-Bench-Pro,
# and in 0.830 / 0.696 of those misses the localiser's own reason never mentions
# it at all. It is not rejected. It is not seen.
#
# Two further measurements say the same thing from the other side. Split the
# TrajAudit rows where the reference step IS on the flow by how much flow there
# is to read: 0.6481 when the traced set is 20 steps or fewer (n=54), 0.3333
# when it is larger (n=18). And the long steps, the ones compression had to
# clip, are the EASY ones (0.8571, n=21) -- size is not what defeats the
# reader, quantity of things to pass over is.
#
# So this block does not add evidence and does not restate the question. It
# makes the enumeration a required output: one line per traced step, written
# before the answer exists. That is also what separates it from the pointwise
# poll (E9/E10/E12), which enumerated far more thoroughly -- one whole call per
# candidate -- and lost on TrajAudit (0.4510 < 0.4804). Isolating a candidate
# in its own call removes the rivals that make it judgeable; its failure was
# resolution, not coverage (reference in the top group 40/66, uniquely top only
# 12). Here every step is instantiated in ONE context, so they stay comparable.
ROLL_MAX = 40                   # above this the roll is skipped, not truncated
ROLL_MIN = 2                    # below this there is nothing to enumerate
ROLL_BUDGET = 8000              # answer budget when the roll is being written

ROLL_BLOCK = """
------------------------------------------------------------
BEFORE YOU ANSWER: CALL THE ROLL
Tracing put these {n} steps on a dependency chain to the failure:
{steps}

Go through them IN THE ORDER LISTED and put one line in "roll" for each — every
one, including the steps that look routine. Say what the step did, from its own
text in PART 2 above, and whether this is the step the run turned on.

Write the roll BEFORE you decide. It is how you look, not a summary of what you
concluded, so do not fix on an answer and then fill the roll in around it. A
step you never wrote a line for is a step you never considered — and the step
that turns out to be responsible is usually one that reads as unremarkable on
the way past, while the step that looks obviously wrong is usually where the
damage became visible rather than where it was done.

Then answer. "critical_step" must be one of the steps you marked "turning":
true. If none of them deserved it, mark none true, answer from the trajectory
as a whole, and say in "reason" what the traced chain failed to reach.
------------------------------------------------------------"""

ROLL_FIELDS = """
  "roll": [{"step": <int>, "did": "<at most 12 words: what this step did>",
            "turning": <true or false>}, ... one entry per step listed, in
            that order],"""

# --- expected-behaviour compliance ----------------------------------------
# Produced by ``expect.assess``; see that module for why the specification is
# written blind and why neither reader is told how the run ended. What arrives
# here is a COUNT per candidate step, which is the shape that has converted on
# this corpus: ``--verify-outdeg`` won by handing the merge a measured quantity
# rather than another opinion, and the arms that lost (E9, E10, E12, E14,
# ta_roll) all handed the reader a second judgement about which step was
# critical. Nothing in this block says a step is to blame, and no model in this
# stage was ever asked.
#
# Steps are listed in TRAJECTORY ORDER and never ranked by violation count.
# Sorting them by the number would be a ranking, i.e. the judgement this stage
# exists to avoid making; the localiser is free to derive one and has the
# numbers to do it with.
EXPECT_BLOCK = """
------------------------------------------------------------
WHAT EACH CANDIDATE STEP WAS SUPPOSED TO DO, AND WHETHER IT DID
Before any of these steps was read, a separate reader was shown the task, the
design of the agent that produced this run, and everything that had reached
that step — but NOT the step itself, and nothing after it. It wrote down what
the step was required to produce. A second reader then checked the step against
that list. Neither was told what went wrong in this run, so these are not
guesses at the answer working backwards from the failure.

  {rows}

Read this as a measurement, not a verdict:
  * "Violated" means the step did something other than what a reader who knew
    the task and the context — but not how the run ended — expected at that
    point. That is not the same as wrong. It fires on an agent that did the
    right things in a different order, or took a detour that paid off, as
    readily as on a mistake, and a requirement written in advance can simply
    be a bad requirement.
  * Meeting every requirement is not innocence. The step that commits to the
    wrong direction usually executes that direction impeccably, and a reader
    who never saw the failure will not have thought to forbid it.
  * "unclear" means the check could not be answered from the step's text. It
    is not a soft violation.
The steps are in trajectory order. They are NOT ranked, and the count is not a
score — a step with more requirements simply had more written about it.
------------------------------------------------------------"""

EXPECT_SHOW_VIOLATED = 3        # violated requirements quoted per step


def render_expect(rec: Optional[Dict[str, object]]) -> str:
    """The compliance block, or "" when there is nothing to report.

    Empty for a row the stage skipped, and empty when not one candidate got a
    usable requirement list — in both cases the localiser's prompt falls back
    to byte-identical with a run that never had this stage, which is what makes
    the A/B measure the stage and not the degradation.

    Rows that produced no requirements are shown as such rather than omitted.
    A step silently missing from a list the block calls "each candidate step"
    would read as a step with nothing against it, which is the one reading of a
    gap that must not be available.
    """
    rows = (rec or {}).get("rows") or []
    if not any(int(r.get("n") or 0) for r in rows):
        return ""
    lines = []
    for r in sorted(rows, key=lambda r: int(r["step"])):
        n = int(r.get("n") or 0)
        if not n:
            lines.append("step %d — no requirements could be written for this "
                         "step" % int(r["step"]))
            continue
        head = "step %d — %d of %d met" % (int(r["step"]),
                                           int(r.get("met") or 0), n)
        extra = []
        if int(r.get("violated") or 0):
            extra.append("%d violated" % int(r["violated"]))
        if int(r.get("unclear") or 0):
            extra.append("%d unclear" % int(r["unclear"]))
        lines.append(head + (", " + ", ".join(extra) if extra else ""))
        by_id = {int(s["id"]): s for s in (r.get("specs") or [])}
        shown = 0
        for c in (r.get("checks") or []):
            if c.get("verdict") != "violated":
                continue
            if shown >= EXPECT_SHOW_VIOLATED:
                lines.append("      (%d more violated)"
                             % (int(r["violated"]) - shown))
                break
            spec = by_id.get(int(c["id"]))
            if spec is None:
                continue
            ev = str(c.get("evidence") or "").strip()
            lines.append("      violated: %s%s"
                         % (spec["must"], (" — %s" % ev) if ev else ""))
            shown += 1
    return EXPECT_BLOCK.format(rows="\n  ".join(lines))

# Two stances on the same candidate list, because the measurement said they are
# not the same intervention. On SWE-Bench-Pro enforcement never fired -- the
# localiser obeyed on its own and acc@0 before and after snapping were equal --
# while on TrajAudit it moved three already-correct answers off the reference
# step. So the list and the enforcement are separate knobs, and the wording has
# to match which one is in force: a block that threatens to move the answer
# while nothing will move it is simply false, and a prompt that lies about the
# mechanism is not a prompt whose effect can be attributed.
#
# The STRICT text is frozen. It is what produced the run on disk, and the LLM
# cache is keyed on the prompt, so a re-run that reports zero uncached calls is
# proof that it did not drift. Edit it and that proof is gone.
_CANDIDATE_BLOCK = """
------------------------------------------------------------
YOUR ANSWER {must} BE ONE OF THE STEPS LISTED BELOW
These are the steps that CHANGED something — wrote a file, applied a patch, ran
a command with side effects. Steps that only read, searched, ran a test, or
reasoned are not listed.

Where this conflicts with point 5 above, this wins. A flawed plan or a flawed
piece of reasoning is still a perfectly good REASON to choose an answer; but the
step you NAME must be the one that acted on it. The but-for test needs a change
to undo — until something was written, there is nothing a correct alternative
would have altered, and the commitment is not yet made. If the decisive thinking
happened earlier, name the FIRST listed step that carried it out, and say in
"reason" which earlier step the thinking came from.

{{cands}}

{closing}
------------------------------------------------------------"""

CANDIDATE_BLOCK = _CANDIDATE_BLOCK.format(
    must="MUST",
    closing='If nothing here fits, still answer with the closest entry and '
            'explain why in\n"reason". An answer outside this list will be '
            'moved to the nearest entry.')

CANDIDATE_BLOCK_SOFT = _CANDIDATE_BLOCK.format(
    must="SHOULD",
    closing='If nothing here fits, answer with the step you believe is right '
            'anyway, and\nsay in "reason" why none of these is it. Your answer '
            'will not be changed for\nyou — this list is where to look, not a '
            'gate on what you may say.')


# The same block for a MIXED answer space -- one where --shortlist has merged a
# reader's proposals into the parsed set. Two of the paragraphs above are then
# false about their own list, which is why this is a separate text rather than
# an edit to the frozen one:
#
#   * "These are the steps that CHANGED something ... Steps that only read,
#     searched, ran a test, or reasoned are not listed." A proposed step is
#     usually a prose planning step -- that is the whole reason the stage
#     exists -- so under C3 the list contains exactly what this sentence says
#     it cannot.
#   * "the step you NAME must be the one that acted on it ... If the decisive
#     thinking happened earlier, name the FIRST listed step that carried it
#     out." Under C3 the decisive thinking step may itself be listed, and that
#     instruction would send the reader away from it.
#
# A prompt that misdescribes its own answer space is not one whose effect can be
# attributed, so the mixed arm gets a block that describes the list it is
# actually given. Everything else is held: same header, same position, same
# closing sentence about answers outside the list, and the entries are rendered
# by the same ``candidates.render`` in one flat list with no provenance marked.
# The reader is not told which entries came from where -- it was told that in
# the first version of this arm, in a second block, and the measurement said
# the two-block form made things worse rather than better.
_CANDIDATE_BLOCK_MIXED = """
------------------------------------------------------------
YOUR ANSWER {must} BE ONE OF THE STEPS LISTED BELOW
These are the steps that are candidates for being the critical one. Most of them
CHANGED something — wrote a file, applied a patch, ran a command with side
effects. Some of them changed nothing and only decided, planned, or reasoned:
those are listed because a wrong decision is as answerable as a wrong edit, and
a step that committed the run to the wrong course is a legitimate answer even
though it wrote nothing.

Where this conflicts with point 5 above, this wins. Choose the listed step where
the fault was INTRODUCED. If that was a decision, name the decision step; if it
was the writing out of a decision that was itself sound, name the step that
wrote it. The but-for test is what settles it: the step a correct alternative
would have had to change FIRST. Do not move your answer later than that step
just because the later one touched a file, and do not move it earlier than that
step just because something before it was vague or incomplete.

{{cands}}

{closing}
------------------------------------------------------------"""

CANDIDATE_BLOCK_MIXED = _CANDIDATE_BLOCK_MIXED.format(
    must="MUST",
    closing='If nothing here fits, still answer with the closest entry and '
            'explain why in\n"reason". An answer outside this list will be '
            'moved to the nearest entry.')

CANDIDATE_BLOCK_MIXED_SOFT = _CANDIDATE_BLOCK_MIXED.format(
    must="SHOULD",
    closing='If nothing here fits, answer with the step you believe is right '
            'anyway, and\nsay in "reason" why none of these is it. Your answer '
            'will not be changed for\nyou — this list is where to look, not a '
            'gate on what you may say.')


# --- the upstream-link test ------------------------------------------------
# Asked for directly: nudge the localiser to ask whether an edit step it is
# about to name is wrong only because an earlier plan/reason step was already
# wrong, and walk to the real root cause.
#
# Sized before it was written (probe/rootcause_headroom.py, zero calls, from
# the stored arms ta_qarm_fx / sp_qbase_fx). Split every row by whether the
# answer the pipeline gave was edit-ish, and by which side of gold it fell on:
#
#                               TA (102 rows)        SP (86 rows)
#   WINNABLE  pred LATER than gold    24 (15 edit-ish)    29 (24 edit-ish)
#   AT RISK   pred already EARLIER    28 (15 edit-ish)    29 (18 edit-ish)
#   SAFE      pred == gold            46 (25 edit-ish)    28 (21 edit-ish)
#
# So the intervention is two-sided, and by a thin margin: a hint that moves the
# answer earlier whenever the answer is an edit has 15 (TA) / 24 (SP) rows to
# win and 25 / 21 currently-correct rows to break, plus 28 / 29 rows that are
# already too early and would only get worse. An unconditional "walk back to
# the plan" therefore loses on TA by construction, and is a coin toss on SP.
# The walk-back distances are also long -- median 10 steps on TA, 16 on SP --
# so the hint cannot rely on the answer being a step or two away.
#
# That is why this block is written as a TEST with a default, not as a nudge.
# The procedure it adds on top of HOW TO CHOOSE 5 and 7 -- which already state
# that a flawed plan can be the critical step, and already ask for a backwards
# walk -- is the part those two rules leave out: what happens when the walk
# finds an upstream step and the link does not hold up. Here the default is
# stated (keep the acting step), the discharge condition is stated (the earlier
# step has to make the fault INEVITABLE, not merely precede it), and both the
# step considered and the outcome of the test have to be written down, so a run
# can be read back for whether the test was actually performed or the model
# just moved its answer earlier.
#
# Nothing here is evidence. Four arms have now lost by handing the localiser
# more pointed evidence (memo-scan, prior-unranked, localize-symptom,
# op-analysts; acc@0_given_prior_miss 0.179 -> 0.032), so this block adds no
# facts about the trajectory and names no step -- it changes the procedure
# only, and every step it talks about is one the model picked itself.
_ROOTCAUSE_BLOCK = """
------------------------------------------------------------
ONE CHECK BEFORE YOU COMMIT: IS YOUR STEP THE CAUSE OR THE EXECUTION?
Do this only if the step you are about to name ACTED — it edited or created a
file, applied a patch, wrote out a change. If the step you are naming is a
piece of reasoning or a plan, skip this check and answer; it does not apply.

An acting step can be wrong in two different ways, and they have different
answers:

  (a) The step itself got it wrong. The intent behind it was right and the
      execution was not — wrong file, wrong location, a change that does not
      say what it was meant to say, a correct fix applied to the wrong code.
      THE ACTING STEP IS THE ANSWER.

  (b) The step executed, correctly, a decision that was already wrong. An
      earlier step diagnosed the problem incorrectly, picked the wrong file or
      the wrong mechanism, or concluded something false from what it had read;
      this step then faithfully wrote that down. THE EARLIER STEP IS THE
      ANSWER, and the edit is just where the decision became visible.

To tell them apart, find the step where the content of this edit was DECIDED —
the diagnosis, the plan, the conclusion it is carrying out — and ask one
question about it: was the edit already doomed once that step had been taken?
Not "did it come earlier", not "was it related", not "could it have been
better": would ANY correct version of this edit have been impossible to write
once that earlier step had been accepted? Recovery is the test. If a competent
agent reading only that earlier step could still have produced a working edit,
then that step did not doom anything and it is NOT the answer.

The default is (a). Earlier is not automatically deeper: every edit has a chain
of thinking behind it, and that chain can be traced back indefinitely until it
reaches steps that merely preceded the failure. Case (b) has to be EARNED by
the recovery test above. If you cannot say what specifically the earlier step
got wrong — not that it was vague, or incomplete, or could have checked more,
but the actual false belief or wrong choice it introduced — then the answer is
the acting step, and you should say so.

Write the outcome down either way, in the fields below. "Not applicable" and
"checked and rejected" are both real results here, and they are useful ones.
{cand_note}------------------------------------------------------------"""

# Two reconciliations, because the two corpora run different arms. TA's arm has
# no candidate list, so the upstream step is nameable and the block above is the
# whole instruction. SP's arm runs --candidates --candidates-strict, and that
# block -- whose STRICT text is frozen, see the comment above _CANDIDATE_BLOCK
# -- already says "the step you NAME must be the one that acted on it", and
# names itself the winner over HOW TO CHOOSE point 5. It is also enforced in
# code: on sp_qbase_fx 82 of 84 rows obeyed and 2 were snapped to the nearest
# listed step, so an answer naming a plan step would be MOVED, not kept.
#
# So on SP the test cannot change which KIND of step is named, only which one.
# That is still the thing asked for -- "find the real root cause" there means
# the earliest acting step that carried the already-wrong decision, not the last
# one that touched the file -- and it is measurable: 16 of 84 rows have gold as
# an earlier LISTED step, against 21 currently-right rows that have an earlier
# listed step to be lured onto and 27 that are already too early. A block that
# said "name the plan step" on SP would produce answers the enforcement then
# discards, which is worse than useless: the prompt would be lying about the
# mechanism, and the arm's effect would stop being attributable.
_CAND_NOTE = """
One adjustment, because your answer is restricted to the list above: when the
test comes out "doomed", the deciding step may not be on that list, and the
list still wins. Do not name an unlisted step. Name instead the EARLIEST listed
step that carried the wrong decision out — the first place it was committed to,
not the last place it showed — and record the deciding step in the fields below.
"""

ROOTCAUSE_BLOCK = _ROOTCAUSE_BLOCK.format(cand_note="")
ROOTCAUSE_BLOCK_CAND = _ROOTCAUSE_BLOCK.format(cand_note=_CAND_NOTE)

ROOTCAUSE_FIELDS = """
  "upstream_step": <int or null: the step where the content of your answer was
                    DECIDED, if your answer is an acting step. null if your
                    answer is not an acting step, or if nothing earlier decided
                    it>,
  "upstream_verdict": "<one of: not-applicable | execution-fault | doomed |
                        rejected>  not-applicable = your answer is not an
                        acting step; execution-fault = case (a), the intent was
                        right; doomed = case (b), and your answer IS that
                        earlier step; rejected = you examined the earlier step
                        and it failed the recovery test>",
  "upstream_because": "<if doomed: the specific false belief or wrong choice
                        that earlier step introduced, and why no correct
                        version of the edit was possible after it. If rejected:
                        how a correct edit was still reachable from it>","""


def _cand_block(enforce: bool, mixed: bool) -> str:
    """Which of the four candidate-block texts this call gets.

    Two independent bits, so four texts rather than a conditional sentence
    inside one: ``enforce`` is whether an outside answer is moved (the closing
    paragraph), ``mixed`` is whether the list contains steps that wrote nothing
    (the two middle paragraphs). Keeping them as whole frozen strings is what
    lets the originals be pinned by hash -- a conditional would make the shipped
    text a function of the flags and there would be nothing left to pin.
    """
    if mixed:
        return CANDIDATE_BLOCK_MIXED if enforce else CANDIDATE_BLOCK_MIXED_SOFT
    return CANDIDATE_BLOCK if enforce else CANDIDATE_BLOCK_SOFT


def _hops_from_symptom(graph: FlowGraph, kind: str) -> Dict[int, int]:
    """Hops from the nearest symptom step, walking up one relation's forest.

    Anchors sit at 0. Everything reachable from them by parent edges gets the
    length of the shortest such walk. Steps with no path to a symptom are
    absent: they are in the trajectory but not on a chain to the failure.
    """
    parents: Dict[int, List[int]] = {}
    for e in graph.edges:
        if e.kind == kind:
            parents.setdefault(e.dst, []).append(e.src)
    depth = {a: 0 for a in graph.anchors}
    frontier, d = list(depth), 0
    while frontier:
        d += 1
        nxt = []
        for node in frontier:
            for p in parents.get(node, ()):
                if p not in depth:
                    depth[p] = d
                    nxt.append(p)
        frontier = nxt
    return depth


CAND_EVIDENCE_CLIP = 900        # per-candidate text in the comparison field
CAND_EVIDENCE_LEAD = """Each entry below is set out with what it did and where
it sits, so that you can compare them against one another directly rather than
hunt for them in PART 2. The facts on the "WHERE" line are mechanical -- they
say what was traced, not what is to blame -- and a step can be the cause with
nothing traced to it at all.
"""


def render_candidate_evidence(traj: Trajectory, spans: Sequence[Span],
                              graph: FlowGraph,
                              digests: Sequence[StepDigest],
                              cands: Sequence[Dict[str, object]],
                              clip: int = CAND_EVIDENCE_CLIP) -> str:
    """The answer space as an explicit comparison field, one row per candidate.

    The localiser is asked for a superlative over a hundred steps, and the
    measurements say that is where it loses: on both corpora the answer is
    already inside an eleven-item shortlist, and it is picked out of that
    shortlist only 39% (SWE-Bench-Pro) and 60% (TrajAudit) of the time. The
    shortlist has been in the prompt all along, but only as a list of bare
    indices -- each entry's actual text sits far away in PART 2, scattered
    among every other step -- so nothing in the prompt ever puts the entries
    side by side.

    This collects each entry's own text and the structural facts already
    computed for it (which phase it falls in, whether tracing reached it, how
    far it is from the symptom) into one block, and changes nothing else.

    The contrast with the group stage (``bracket.py``, measured and rejected)
    is the point. That one also tried to turn the superlative into a small
    comparison, but it did so by SPLITTING the question across independent
    calls, and the verdicts then could not be compared: whether a group spoke
    at all carried no signal on SWE-Bench-Pro (0.537 for the group holding the
    reference step against 0.509 for the rest) and self-reported confidence
    carried none on either corpus. Here the comparison stays inside one call
    with one reader and one context, so the entries remain commensurable.
    """
    if not cands:
        return ""
    dmap = {d.index: d for d in digests}
    hops = {k: _hops_from_symptom(graph, k) for k in (DATA_FLOW, CONTROL_FLOW)}
    on_flow = graph.on_flow()
    anchors = set(graph.anchors)
    rows = []
    for c in cands:
        try:
            i = int(c["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= i < len(traj.steps)):
            continue
        where = []
        for sp in spans:
            if sp.start <= i <= sp.end:
                where.append('phase "%s" (steps %d-%d)'
                             % (sp.label or "unlabelled", sp.start, sp.end))
                break
        if i in anchors:
            where.append("this is a step where the failure is VISIBLE")
        marks = []
        for kind, name in ((DATA_FLOW, "data"), (CONTROL_FLOW, "control")):
            h = hops[kind].get(i)
            if h is not None:
                marks.append("%s %d hop%s from the symptom"
                             % (name, h, "" if h == 1 else "s"))
        if marks:
            where.append("traced: " + ", ".join(marks))
        elif i not in on_flow:
            where.append("tracing reached no chain from the symptom to it")
        d = dmap.get(i)
        text = (d.text if d is not None else (traj.steps[i].content or ""))
        verbatim = "" if (d is None or d.tier == COMPRESSED) else " (its own words)"
        rows.append(
            "  --- STEP %d — %s ---\n"
            "  WHERE: %s\n"
            "  WHAT IT DID%s:\n%s"
            % (i, c.get("label") or "(edit)",
               "; ".join(where) or "(nothing recorded)", verbatim,
               _indent(clip_middle(text, clip), "    ")))
    if not rows:
        return ""
    return CAND_EVIDENCE_LEAD + "\n" + "\n\n".join(rows)


# --- pointwise verification: judging each candidate on its own ------------
# Six interventions (E1, E3, E5, E6, E7, E8) have now failed. Five changed what
# the reader was SHOWN; the sixth changed what it was ASKED. All six kept the
# same DECISION PROCEDURE: one reader, one call, "pick the one out of these
# eleven". The measurements say that procedure is where the loss is -- on
# SWE-Bench-Pro the reference step is inside the offered list 0.767 of the time
# and gets picked out of it only 39%.
#
# This replaces the superlative with a poll of independent absolute judgements:
# one call per candidate, each asked about ITS OWN step and shown no rivals.
#
# Why the verdicts are comparable, which is the question bracket.py (E5) failed
# to answer and was rejected for. There, each group saw a DIFFERENT subset of
# the answer space, so "this group's winner" meant something different in each
# call and the scores could not be pooled -- measured: the group holding the
# reference step spoke no more confidently than the rest (0.537 vs 0.509).
# Here every call is the same template over the same evidence with the same
# four questions and the same answer format; the ONLY thing that differs
# between two calls is which step index sits in the focal slot. The verdicts
# are therefore commensurable by construction rather than by assumption, which
# is the specific defect that sank E5.
#
# The four questions are binary on purpose. A free 0..1 confidence was measured
# to carry no signal on either corpus, and binary answers to a fixed rubric are
# far more stable across independent calls than a self-reported scalar.
VERIFY_SYSTEM = (
    "You examine one step of a failed agent trajectory and judge that step "
    "alone, against a fixed set of questions. Answer strictly in JSON."
)

VERIFY_BUDGET = 900
VERIFY_FOCAL_CLIP = 4000        # the step under examination, in its own words
VERIFY_CONTEXT_LINE = 160       # one line per surrounding step
VERIFY_BEFORE = 30              # how many steps of run-up to summarise
VERIFY_AFTER = 12               # how many steps of aftermath to summarise
VERIFY_MAX = 32                 # cost ceiling per trajectory; SP max is 31

VERIFY_PROMPT = """A coding agent FAILED this task. Below is ONE step of that
run. Judge THAT STEP ALONE.

You are not choosing between steps and you will not be shown the alternatives.
Each of the other steps is being judged separately on this same form, and the
completed forms are compared afterwards. So answer as though this step were the
only one you had been asked about, and answer every question even where you are
unsure -- a blank is read as "no", which is a claim in itself.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

SYMPTOM STEPS: {anchors}
These are where the failure became VISIBLE. Being where a failure surfaces is
not evidence of having caused it.

HOW THE RUN WENT, PHASE BY PHASE:
{outline}

WHAT THE AGENT DID BEFORE THE STEP UNDER EXAMINATION (one line each):
{before}

============================================================
THE STEP UNDER EXAMINATION -- STEP {focal}
WHERE IT SITS: {where}
============================================================
{focal_text}
============================================================

WHAT THE AGENT DID AFTER IT (one line each):
{after}

Answer these four questions about STEP {focal} and nothing else:

  "depends" -- but for what step {focal} did, would the observed failure still
      have happened? True means the failure DEPENDS on this step: had this step
      been done correctly, that failure would not have occurred.

  "averts" -- suppose you could go back and redo step {focal} correctly,
      changing nothing else about the run. Does the task then SUCCEED? This is
      stricter than "depends": a step can sit on the causal chain while redoing
      it alone still leaves the agent heading for the same failure.

  "earlier_cause" -- was the run ALREADY doomed to this failure before step
      {focal} began? True means some earlier step had already made the failure
      inevitable, which would make step {focal} not the EARLIEST responsible
      step even if it is genuinely part of the chain.

  "only_symptom" -- does step {focal} merely reveal, report, or pass along a
      fault that was introduced somewhere else, without introducing one itself?

Exploration is not an error: while the agent is still investigating options it
has not yet gone wrong. A step whose action is well-formed can still be at
fault if the DECISION behind it was wrong.

Return ONLY JSON:
{{
  "depends": <true or false>,
  "averts": <true or false>,
  "earlier_cause": <true or false>,
  "only_symptom": <true or false>,
  "confidence": <float 0..1: how sure you are of the four answers above>,
  "why": "<1-2 sentences naming what this step did, not what the run did>"
}}"""


# The binary rubric above is informative but blunt. Measured on SWE-Bench-Pro:
# all four questions separate the reference step in the right direction
# (depends true on 27.3% of reference steps against 8.4% of all candidates,
# only_symptom 24.2% against 45.5%), yet "averts" comes back true on 0.7% of
# candidates -- the question that defines the task is phrased as an absolute
# the model will not clear, so the double-weighted term is almost always zero.
# Four binaries give six score levels for eleven candidates, so 55% of polls
# end in a tie of mean size 3.21 and the shipped answer is decided by the
# tie-break rather than by the rubric. Grading the two positive questions 0-3
# is the smallest change that buys resolution where the signal already is.
#
# Self-reported confidence is deliberately NOT asked for here. On the binary
# run it was measured actively harmful as a tie-break -- 5 of 27 contested ties
# against 14 for keeping the listwise answer, worse even than a blind "earliest
# candidate" (9 of 27) -- which is the E5 finding again, now inside calls that
# are commensurable by construction.
VERIFY_GRADED_PROMPT = """A coding agent FAILED this task. Below is ONE step of
that run. Judge THAT STEP ALONE.

You are not choosing between steps and you will not be shown the alternatives.
Each of the other steps is being judged separately on this same form, and the
completed forms are compared afterwards. So answer as though this step were the
only one you had been asked about, and answer every question even where you are
unsure -- a blank is read as the lowest grade, which is a claim in itself.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

SYMPTOM STEPS: {anchors}
These are where the failure became VISIBLE. Being where a failure surfaces is
not evidence of having caused it.

HOW THE RUN WENT, PHASE BY PHASE:
{outline}

WHAT THE AGENT DID BEFORE THE STEP UNDER EXAMINATION (one line each):
{before}

============================================================
THE STEP UNDER EXAMINATION -- STEP {focal}
WHERE IT SITS: {where}
============================================================
{focal_text}
============================================================

WHAT THE AGENT DID AFTER IT (one line each):
{after}

Answer these four questions about STEP {focal} and nothing else. The first two
are GRADED: use the whole range, and do not reserve the top grade for
certainty. A grade of 2 means "probably", and most steps in a failed run
deserve a 0.

  "repair" -- you may go back and redo EXACTLY ONE step of this run correctly
      and then let the run continue from there. How good a choice is step
      {focal}?
        0 = redoing it changes nothing about how this run failed
        1 = redoing it helps, but the run still fails for the same reason
        2 = redoing it probably saves the run
        3 = redoing it saves the run, and nothing before it needed fixing

  "depends" -- how much does the observed failure DEPEND on what this step did?
        0 = the failure would have happened exactly the same way without it
        1 = it touched something involved in the failure, but only in passing
        2 = the failure runs through this step
        3 = the failure is this step's doing

  "earlier_cause" -- true if the run was ALREADY doomed to this failure before
      step {focal} began, i.e. some earlier step had already made it
      inevitable. This is what would make step {focal} not the EARLIEST
      responsible step even when it is genuinely part of the chain.

  "only_symptom" -- true if step {focal} merely reveals, reports, or passes
      along a fault introduced somewhere else, without introducing one itself.

Exploration is not an error: while the agent is still investigating options it
has not yet gone wrong. A step whose action is well-formed can still be at
fault if the DECISION behind it was wrong.

Return ONLY JSON:
{{
  "repair": <0, 1, 2 or 3>,
  "depends": <0, 1, 2 or 3>,
  "earlier_cause": <true or false>,
  "only_symptom": <true or false>,
  "why": "<1-2 sentences naming what this step did, not what the run did>"
}}"""


# The pointwise rubric is length-robust where the listwise reader is not: on
# SWE-Bench-Pro the listwise answer is right on 0.419 of 80-119-step runs but
# 0.250 of 120+-step ones, while the rubric puts the reference step on top at
# 0.548 and 0.571 respectively. A judgement made about one step at a time does
# not decay as the list of rivals grows -- but what it is given about that step
# does not grow with the run either. Its whole view of consequence is two hop
# counts in WHERE IT SITS.
#
# So give the focal step the one thing that is genuinely per-step and already
# computed: the traced edges OUT of it. That is not "more verbatim text" -- the
# family E1/E3/E5/E6/E7 closed after five defeats -- it is the structural fact
# the two positive questions are actually about. "Does the failure depend on
# this step" and "does it merely pass along someone else's fault" are questions
# about what later steps took from it, and until now the verifier had to guess
# that from a hop count.
VERIFY_CHAIN_MAX = 8            # data edges shown out of the focal step
VERIFY_CHAIN_CTRL = 4           # control edges shown out of the focal step


def _verify_consequences(traj: Trajectory, graph: FlowGraph,
                         dmap: Dict[int, StepDigest],
                         out_edges: Dict[int, List], hops: Dict[str, Dict[int, int]],
                         lo: int, hi: int, i: int) -> str:
    """What the traced graph says later steps took from step ``i``.

    Mechanical, like ``_verify_where``: it reports what was traced, never who
    is to blame. A step with nothing traced out of it says so plainly rather
    than being left blank, because "nothing later used this" is itself the
    answer to one of the questions being asked.
    """
    outs = out_edges.get(i) or []
    data = [e for e in outs if e.kind == DATA_FLOW]
    ctrl = [e for e in outs if e.kind == CONTROL_FLOW]
    if not data and not ctrl:
        return ("  Nothing was traced out of this step: no later step was found "
                "to use what it produced, and no later step was found to follow "
                "from it. Tracing is incomplete, so this is weak evidence.")

    ents = [e.strip().lower() for e in (graph.failure_entities or []) if e.strip()]

    def _names_failure(txt):
        t = (txt or "").strip().lower()
        return bool(t) and any(x in t or t in x for x in ents)

    def _line(d):
        # Consumers inside the run-up/aftermath already appear there; only the
        # far-away ones need a reminder of what they are.
        if lo <= d < hi:
            return ""
        dg = dmap.get(d)
        txt = (dg.text if dg is not None else (traj.steps[d].content or ""
                                               if 0 <= d < len(traj.steps) else ""))
        one = " ".join((txt or "").split())
        return ("\n        (step %d is: %s)"
                % (d, clip_middle(one, VERIFY_CONTEXT_LINE) or "empty"))

    lines = []
    for e in data[:VERIFY_CHAIN_MAX]:
        ent = clip_middle((e.entity or "").strip(), 90) or "unnamed"
        mark = ("  <- the failure summary is about this"
                if _names_failure(e.entity) else "")
        lines.append("    step %d used what this step produced [%s]%s%s"
                     % (e.dst, ent, mark, _line(e.dst)))
    if len(data) > VERIFY_CHAIN_MAX:
        lines.append("    ... and %d more later steps used its output"
                     % (len(data) - VERIFY_CHAIN_MAX))
    for e in ctrl[:VERIFY_CHAIN_CTRL]:
        why = clip_middle(" ".join((e.rationale or "").split()), 120)
        lines.append("    step %d happened BECAUSE of this step%s%s"
                     % (e.dst, (": " + why) if why else "", _line(e.dst)))
    if len(ctrl) > VERIFY_CHAIN_CTRL:
        lines.append("    ... and %d more later steps followed from it"
                     % (len(ctrl) - VERIFY_CHAIN_CTRL))

    # Which consumer, if any, the step's own traced path to the failure runs
    # through. A bare count of "consumers that are themselves on a chain" reads
    # as evidence of irrelevance whenever the chain leaves by a different edge,
    # which is most of the time; naming the next hop says the same thing
    # without the false implication.
    nxt = []
    for kind in (DATA_FLOW, CONTROL_FLOW):
        h = hops[kind].get(i)
        if h is None:
            continue
        for e in outs:
            if (e.kind == kind and hops[kind].get(e.dst) == h - 1
                    and e.dst not in nxt):
                nxt.append(e.dst)
    if nxt:
        lines.append("    (the traced chain from this step to where the failure "
                     "became visible continues through %s)"
                     % ", ".join("step %d" % d for d in sorted(nxt)))
    else:
        lines.append("    (no traced chain runs from this step through any of "
                     "them to where the failure became visible)")
    return "\n".join(lines)


def _with_chain(tpl: str) -> str:
    """The same form, with the consequence block inserted under WHERE IT SITS.

    Built by substitution from the shipped template rather than written out
    again, so the two cannot drift apart and the question being asked is
    provably unchanged -- only the evidence differs.
    """
    old = "WHERE IT SITS: {where}\n"
    new = (old + "WHAT LATER STEPS TOOK FROM IT, as traced (not a judgement):\n"
                 "{downstream}\n")
    assert tpl.count(old) == 1
    return tpl.replace(old, new, 1)


VERIFY_PROMPT_CHAIN = _with_chain(VERIFY_PROMPT)
VERIFY_GRADED_PROMPT_CHAIN = _with_chain(VERIFY_GRADED_PROMPT)


def _verify_where(traj: Trajectory, spans: Sequence[Span], graph: FlowGraph,
                  hops: Dict[str, Dict[int, int]], on_flow: set,
                  anchors: set, i: int) -> str:
    """The mechanical facts about where a step sits.

    Says what was traced, not what is to blame -- a step can be the cause with
    nothing traced to it at all.
    """
    where = []
    for sp in spans:
        if sp.start <= i <= sp.end:
            where.append('phase "%s" (steps %d-%d)'
                         % (sp.label or "unlabelled", sp.start, sp.end))
            break
    if i in anchors:
        where.append("this is a step where the failure is VISIBLE")
    marks = []
    for kind, name in ((DATA_FLOW, "data"), (CONTROL_FLOW, "control")):
        h = hops[kind].get(i)
        if h is not None:
            marks.append("%s %d hop%s from the symptom"
                         % (name, h, "" if h == 1 else "s"))
    if marks:
        where.append("traced: " + ", ".join(marks))
    elif i not in on_flow:
        where.append("tracing reached no chain from the symptom to it")
    return "; ".join(where) or "(nothing recorded)"


def _verify_context(traj: Trajectory, dmap: Dict[int, StepDigest],
                    lo: int, hi: int) -> str:
    """One clipped line per step in [lo, hi), for run-up and aftermath."""
    lines = []
    for i in range(max(0, lo), min(len(traj.steps), hi)):
        d = dmap.get(i)
        text = (d.text if d is not None else (traj.steps[i].content or ""))
        first = " ".join((text or "").split())
        lines.append("  step %d: %s"
                     % (i, clip_middle(first, VERIFY_CONTEXT_LINE) or "(empty)"))
    return "\n".join(lines) or "  (nothing)"


def verify_candidates(traj: Trajectory, spans: Sequence[Span],
                      graph: FlowGraph, llm: LLM,
                      digests: Sequence[StepDigest],
                      cands: Sequence[Dict[str, object]],
                      outline: Optional[Outline] = None,
                      workers: int = 8,
                      graded: bool = False,
                      chain: bool = False) -> List[Dict[str, object]]:
    """One independent absolute judgement per candidate step.

    Returns one record per candidate in the order given. Every prompt is the
    same template over the same evidence; only the focal step differs.
    Candidates beyond ``VERIFY_MAX`` are not judged -- the cap bounds cost and
    is recorded when it binds, rather than silently shrinking the answer space.

    ``graded`` swaps the four binaries for the 0-3 rubric. ``chain`` adds the
    traced edges out of the focal step. Both off, the prompt is byte-identical
    to the shipped one, so its cached answers stay free and every variant is
    measured against the same corpus.
    """
    steps = []
    for c in cands or []:
        try:
            i = int(c["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < len(traj.steps):
            steps.append(i)
    if not steps:
        return []
    steps = steps[:VERIFY_MAX]

    dmap = {d.index: d for d in (digests or [])}
    hops = {k: _hops_from_symptom(graph, k) for k in (DATA_FLOW, CONTROL_FLOW)}
    out_edges = {}
    if chain:
        for e in graph.edges:
            out_edges.setdefault(e.src, []).append(e)
    on_flow = graph.on_flow()
    anchors = set(graph.anchors)
    head = dict(
        task=clip_middle(traj.task_description, 3000),
        failure_summary=graph.failure_summary or "(not summarised)",
        anchors=(", ".join("step %d" % a for a in sorted(graph.anchors))
                 or "(none identified)"),
        outline=(clip_middle(outline.render(), 6000) if outline is not None
                 else "(no outline available)"),
    )

    def _ask(i):
        st = traj.steps[i]
        if chain:
            tpl = VERIFY_GRADED_PROMPT_CHAIN if graded else VERIFY_PROMPT_CHAIN
        else:
            tpl = VERIFY_GRADED_PROMPT if graded else VERIFY_PROMPT
        return tpl.format(
            focal=i,
            where=_verify_where(traj, spans, graph, hops, on_flow, anchors, i),
            downstream=(_verify_consequences(traj, graph, dmap, out_edges, hops,
                                             i - VERIFY_BEFORE,
                                             i + 1 + VERIFY_AFTER, i)
                        if chain else ""),
            focal_text=(clip_middle(st.content or "", VERIFY_FOCAL_CLIP)
                        or "(empty)"),
            before=_verify_context(traj, dmap, i - VERIFY_BEFORE, i),
            after=_verify_context(traj, dmap, i + 1, i + 1 + VERIFY_AFTER),
            **head)

    objs = map_concurrent(
        lambda i: llm.generate_obj(_ask(i), max_tokens=VERIFY_BUDGET,
                                   system=VERIFY_SYSTEM),
        steps, workers=max(1, int(workers)))

    out = []
    for i, obj in zip(steps, objs):
        obj = obj if isinstance(obj, dict) else {}

        def _b(key, _o=obj):
            v = _o.get(key)
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "1")
            if isinstance(v, (int, float)):
                return bool(v)
            return False

        try:
            conf = max(0.0, min(1.0, float(obj.get("confidence"))))
        except (TypeError, ValueError):
            conf = 0.0

        def _g(key, _o=obj):
            """A 0-3 grade. Anything unreadable is 0 -- the lowest grade is
            also what the prompt tells the model a blank will be read as, so a
            malformed answer and a refusal land in the same place."""
            v = _o.get(key)
            if isinstance(v, bool):
                return 3 if v else 0
            if isinstance(v, (int, float)):
                g = int(v)
            elif isinstance(v, str):
                m = re.search(r"-?\d+", v)
                if not m:
                    return 0
                g = int(m.group(0))
            else:
                return 0
            return max(0, min(3, g))

        if graded:
            rec = {"step": i, "graded": True,
                   "repair": _g("repair"), "depends": _g("depends"),
                   "earlier_cause": _b("earlier_cause"),
                   "only_symptom": _b("only_symptom"), "confidence": conf,
                   "why": str(obj.get("why") or "")[:300]}
            # The binary run measured all four questions pointing the right way
            # but "averts" firing on 0.7% of candidates, so six score levels had
            # to separate eleven candidates and 55% of polls tied. Grading the
            # two positive questions widens the range to -4..9 while keeping the
            # same shape: the defining question counts double, each negative
            # cancels a full graded point of it.
            rec["score"] = (2 * rec["repair"] + rec["depends"]
                            - 2 * rec["earlier_cause"] - 2 * rec["only_symptom"])
        else:
            rec = {"step": i, "depends": _b("depends"), "averts": _b("averts"),
                   "earlier_cause": _b("earlier_cause"),
                   "only_symptom": _b("only_symptom"), "confidence": conf,
                   "why": str(obj.get("why") or "")[:300]}
            # Weights fixed in advance, not fitted: "averts" is the question that
            # actually defines the task, so it counts double, and the two negative
            # questions each cancel one point of it. Range -2..+3. The whole
            # per-candidate record goes into the artifact, so any other weighting
            # can be re-scored offline at zero call cost.
            rec["score"] = (2 * rec["averts"] + rec["depends"]
                            - rec["earlier_cause"] - rec["only_symptom"])
        out.append(rec)
    return out


# The rubric cannot resolve its own ties (E10, E12, E14 each failed to), so
# bring in a discriminator that is not a judgement at all: how many traced edges
# leave the step. Measured per candidate on the SWE-Bench-Pro polls, the
# reference step has mean out-degree 4.47 against the rivals' 2.06 -- a larger
# separation than anything the model was asked for, and it costs nothing because
# the graph is already built. Read plainly it says a step matters when many later
# steps consumed what it produced, which is what "the failure turns on it" means
# structurally.
#
# It is a TERM, not a replacement: argmax out-degree alone scores 0.3372 and the
# rubric alone 0.3023, but summed they reach 0.3605 -- the two disagree often
# enough to be worth adding rather than choosing between. The cap exists because
# out-degree is unbounded and a single hub step would otherwise outvote the
# rubric everywhere; w and cap were swept on dev only (72 cells, mean test-dev
# gap +0.0031, i.e. no divergence).
#
# THIS IS A SWE-BENCH-PRO EFFECT AND MUST NOT BE CLAIMED ON TRAJAUDIT. The same
# family tops out at TA dev 0.4583 against its own shipped 0.5000, with a mean
# test-dev gap of -0.0394. Hence a knob, off by default, and not in run_arm's TA
# line.
VERIFY_OUTD_W = 1.0             # weight on the capped out-degree
VERIFY_OUTD_CAP = 8             # above this, more consumers say nothing more


def _outdegrees(graph, steps):
    """Traced out-degree per step -- structural, no model, no labels."""
    deg = {}
    for e in getattr(graph, "edges", ()) or ():
        deg[e.src] = deg.get(e.src, 0) + 1
    return {i: deg.get(i, 0) for i in steps}


def _merge_verify(verdicts, incumbent, outd=None):
    """Pick the best-scoring candidate, but never move on a tie.

    The downside is bounded the way E8's was: the listwise answer is displaced
    only by a candidate scoring STRICTLY higher, so a verifier that is merely
    noisy leaves the incumbent standing.

    Ties are NOT broken by position, but the reason has changed and the old one
    is no longer true. It used to be that no positional signal existed: gold sat
    at mean list position 0.367 and our answers at 0.374. ``outd`` broke that
    match -- out-degree rewards steps with many traced descendants, which sit
    early, so with the term on our answers move to 0.255 and the residual error
    acquires a direction: gold is LATER than the answer in 25 of 36 misses
    (p=0.014, against 21/40 p=0.44 without the term).

    The direction is real and still unusable. Misses sit a median of 3
    candidates away and only 7 of 36 are adjacent, so every monotone late-ward
    correction measured -- exact-tie-to-latest, a top-band widened to 0.5/1/2,
    and a lateness term at u=0.5..4 -- is flat or negative on the full corpus,
    with the dev-half gains reversing on test. A signal can have a significant
    direction and no exploitable step size; see probe/RUNS.md, "残差偏移".
    """
    if not verdicts:
        return {"winner": incumbent, "record": None}
    if outd:
        val = {v["step"]: v["score"] + VERIFY_OUTD_W * min(outd.get(v["step"], 0),
                                                           VERIFY_OUTD_CAP)
               for v in verdicts}
    else:
        val = {v["step"]: v["score"] for v in verdicts}
    best = max(val[v["step"]] for v in verdicts)
    top = [v for v in verdicts if val[v["step"]] == best]
    inc = next((v for v in verdicts if v["step"] == incumbent), None)
    # An unpolled incumbent is DISPLACED by the best-scoring polled step, which
    # is the shipped rule and stays the shipped rule. It was changed once, on
    # the theory that a poll which never asked about the answer has produced no
    # evidence against it, and the theory was wrong where it counts: on the rows
    # where it fired, the listwise answer it protected was the reference step
    # 3/68 of the time against 21/68 for the polled step it was displacing, and
    # acc@0 fell from 0.3256 to 0.1163. Keeping an unpolled incumbent is only
    # conservative if the incumbent is as good as the poll; measured, it is seven
    # times worse. The branch is gone and so is the pool-subsetting
    # (``verify_pool_c1``) that created unpolled incumbents in the first place:
    # the poll now always covers the answer space, so ``inc is None`` is back to
    # meaning only "VERIFY_MAX truncated the list".
    if inc is not None and val[inc["step"]] >= best:
        winner = incumbent
    elif len(top) == 1:
        winner = top[0]["step"]
    else:
        # Several candidates tie above the incumbent. Self-reported confidence
        # is the only discriminator left that is not positional; it was
        # measured to carry no signal BETWEEN calls in E5, so this is a weak
        # rule over an already-rare case, recorded so it can be re-scored.
        winner = max(top, key=lambda v: (v["confidence"], -v["step"]))["step"]
    return {"winner": winner,
            "record": {"asked": len(verdicts), "verdicts": verdicts,
                       "best_score": best, "n_tied": len(top),
                       "incumbent": incumbent,
                       "incumbent_score": (val[inc["step"]] if inc is not None
                                           else None),
                       # True when the answer under test was not in the
                       # poll at all. With the pool equal to the answer space
                       # that can now only be a VERIFY_MAX truncation, so this
                       # is a cost-cap counter, not a mechanism. Kept because a
                       # row settled by a poll that never saw the incumbent is
                       # a different result from one where the poll agreed.
                       "incumbent_unpolled": inc is None,
                       "outd": (dict(outd) if outd else None),
                       "winner_step": winner,
                       "changed": winner != incumbent}}


# The pointwise poll's failure mode on SWE-Bench-Pro is RESOLUTION, not
# direction. The reference step lands in the top-scoring group on 40 of the 66
# polls that contain it, but UNIQUELY on top only 12 times; the other 28 are
# settled by a tie-break that wins 14 of them, which is a coin flip. Grading the
# rubric (E10) and handing it the traced consequences (E12) each left that
# "uniquely top" count at exactly 12, and every re-weighting in
# probe/verify_solo.py scores the identical 0.3023 because they all tie the same
# steps. Fourteen recoverable rows is the largest measured headroom left on this
# corpus and no rearrangement of the existing answers can reach it.
#
# So stop asking for a finer score and ask the question the tie actually poses:
# OF THESE TWO, WHICH ONE. That is commensurable the same way the pointwise call
# is -- one template, one pair, the same evidence -- and it is asked only where
# the poll already failed to decide, so the cost lands on the contested polls
# alone.
#
# POSITION BIAS IS THE WAY THIS SILENTLY GOES WRONG. A model that always prefers
# the first-named step implements "earliest candidate", which was measured to
# lose (9 of 27 contested against the incumbent's 14) -- and it would look like
# a working method from the accuracy alone. Two defences, both structural:
# the two steps are ALWAYS printed in trajectory order, so the evidence layout
# is byte-identical between the two orderings and only the labels A/B move; and
# every pair is asked twice with the labels swapped, a win counting only when
# both askings name the same step. Disagreements are recorded as draws and
# reported, which turns the bias into a measurement instead of a load-bearing
# assumption.
VERIFY_DUEL_MAX = 4             # tied candidates admitted to the playoff
VERIFY_DUEL_CLIP = 2500         # each of the two steps, in its own words
VERIFY_DUEL_BETWEEN = 24        # lines of what happened between the two

VERIFY_DUEL_PROMPT = """A coding agent FAILED this task. Two steps of that run
are shown below. Exactly one question is being asked: WHICH OF THE TWO the
failure turns on.

Both steps have already been judged on their own and came out level, so
"they both look bad" is not an answer -- they do. Say which one the failure
actually depends on. Answer "neither" only if neither step is on the causal
chain at all.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

SYMPTOM STEPS: {anchors}
These are where the failure became VISIBLE. Being where a failure surfaces is
not evidence of having caused it.

HOW THE RUN WENT, PHASE BY PHASE:
{outline}

WHAT THE AGENT DID BEFORE EITHER OF THEM (one line each):
{before}

============================================================
OPTION {lo_label} -- STEP {lo}
WHERE IT SITS: {lo_where}
============================================================
{lo_text}
============================================================

WHAT HAPPENED BETWEEN THE TWO (one line each):
{between}

============================================================
OPTION {hi_label} -- STEP {hi}
WHERE IT SITS: {hi_where}
============================================================
{hi_text}
============================================================

WHAT THE AGENT DID AFTER BOTH (one line each):
{after}

Decide between OPTION A (step {a}) and OPTION B (step {b}):

  Which one does the observed failure DEPEND on? But for that step the failure
  would not have happened; but for the other one it still would.

  If both are on the chain, answer with the EARLIER-ACTING cause -- the step
  that made the failure inevitable, not a later step that carried it along.

  A step that merely reveals, reports, or passes along a fault introduced
  somewhere else loses to the step that introduced it.

Exploration is not an error: while the agent is still investigating options it
has not yet gone wrong. A step whose action is well-formed can still be at
fault if the DECISION behind it was wrong.

Return ONLY JSON:
{{
  "answer": "<A or B or neither>",
  "why": "<1-2 sentences naming what decided it>"
}}"""


def _duel_between(traj: Trajectory, dmap: Dict[int, StepDigest],
                  lo: int, hi: int) -> str:
    """What happened between the two steps, elided in the middle when long.

    Two tied candidates can sit eighty steps apart, and the run between them is
    the evidence for "the earlier one made it inevitable". It is also the one
    block here whose length is unbounded, so it is clipped by dropping the
    MIDDLE rather than the tail: the steps just after the earlier candidate and
    just before the later one are the ones that carry the dependency.
    """
    span = list(range(lo + 1, hi))
    if not span:
        return "  (they are adjacent)"
    if len(span) <= VERIFY_DUEL_BETWEEN:
        return _verify_context(traj, dmap, lo + 1, hi)
    half = VERIFY_DUEL_BETWEEN // 2
    head = _verify_context(traj, dmap, lo + 1, lo + 1 + half)
    tail = _verify_context(traj, dmap, hi - half, hi)
    return "%s\n  ... %d further steps ...\n%s" % (head, len(span) - 2 * half,
                                                   tail)


def verify_duels(traj: Trajectory, spans: Sequence[Span], graph: FlowGraph,
                 llm: LLM, digests: Sequence[StepDigest],
                 steps: Sequence[int], outline: Optional[Outline] = None,
                 workers: int = 8) -> Dict[str, object]:
    """A double round-robin over the tied steps: every pair, both labellings.

    Returns the per-pair record and a Copeland tally. A pair contributes a win
    only when the two labellings agree on the same step; when they disagree the
    pair is a draw and is counted as such, so the disagreement rate is visible
    in the artifact rather than being absorbed into the result.
    """
    pool = sorted({int(s) for s in steps if 0 <= int(s) < len(traj.steps)})
    if len(pool) < 2:
        return {"pool": pool, "pairs": [], "wins": {}, "disagreed": 0,
                "neither": 0}

    dmap = {d.index: d for d in (digests or [])}
    hops = {k: _hops_from_symptom(graph, k) for k in (DATA_FLOW, CONTROL_FLOW)}
    on_flow = graph.on_flow()
    anchors = set(graph.anchors)
    head = dict(
        task=clip_middle(traj.task_description, 3000),
        failure_summary=graph.failure_summary or "(not summarised)",
        anchors=(", ".join("step %d" % a for a in sorted(graph.anchors))
                 or "(none identified)"),
        outline=(clip_middle(outline.render(), 6000) if outline is not None
                 else "(no outline available)"),
    )

    def _text(i):
        return (clip_middle(traj.steps[i].content or "", VERIFY_DUEL_CLIP)
                or "(empty)")

    def _where(i):
        return _verify_where(traj, spans, graph, hops, on_flow, anchors, i)

    def _ask(job):
        lo, hi, a = job                     # ``a`` is whichever is labelled A
        return VERIFY_DUEL_PROMPT.format(
            lo=lo, hi=hi, a=a, b=(hi if a == lo else lo),
            lo_label=("A" if a == lo else "B"),
            hi_label=("B" if a == lo else "A"),
            lo_where=_where(lo), hi_where=_where(hi),
            lo_text=_text(lo), hi_text=_text(hi),
            before=_verify_context(traj, dmap, lo - VERIFY_BEFORE, lo),
            between=_duel_between(traj, dmap, lo, hi),
            after=_verify_context(traj, dmap, hi + 1, hi + 1 + VERIFY_AFTER),
            **head)

    jobs = []
    for x in range(len(pool)):
        for y in range(x + 1, len(pool)):
            lo, hi = pool[x], pool[y]
            jobs.append((lo, hi, lo))       # earlier step labelled A
            jobs.append((lo, hi, hi))       # and the same pair, labels swapped
    objs = map_concurrent(
        lambda j: llm.generate_obj(_ask(j), max_tokens=VERIFY_BUDGET,
                                   system=VERIFY_SYSTEM),
        jobs, workers=max(1, int(workers)))

    picks: Dict[Tuple[int, int, int], Optional[int]] = {}
    for job, obj in zip(jobs, objs):
        lo, hi, a = job
        ans = ""
        if isinstance(obj, dict):
            ans = str(obj.get("answer") or "").strip().upper()[:1]
        picks[job] = (a if ans == "A" else
                      ((hi if a == lo else lo) if ans == "B" else None))

    wins = {s: 0 for s in pool}
    pairs, disagreed, neither = [], 0, 0
    for x in range(len(pool)):
        for y in range(x + 1, len(pool)):
            lo, hi = pool[x], pool[y]
            first, second = picks[(lo, hi, lo)], picks[(lo, hi, hi)]
            if first is not None and first == second:
                wins[first] += 1
                won = first
            else:
                won = None
                if first is None and second is None:
                    neither += 1
                else:
                    disagreed += 1
            pairs.append({"lo": lo, "hi": hi, "a_first": first,
                          "b_first": second, "won": won})
    return {"pool": pool, "pairs": pairs, "wins": wins,
            "disagreed": disagreed, "neither": neither}


def _apply_duel(vres, verdicts, incumbent, run):
    """Let a playoff settle the tie the rubric could not.

    Runs only where the poll left several candidates level at the top and the
    incumbent does not already outscore them -- everywhere else the shipped
    merge stands and no call is made. A playoff that ends level itself falls
    back to that same shipped answer, so the knob can only act where there was
    nothing to preserve.
    """
    rec = vres["record"]
    if rec is None:
        return vres
    best = rec["best_score"]
    top = sorted(v["step"] for v in verdicts if v["score"] == best)
    inc_score = rec["incumbent_score"]
    if len(top) < 2 or (inc_score is not None and inc_score > best):
        return vres
    pool = top[:VERIFY_DUEL_MAX]
    duel = run(pool)
    rec["duel"] = duel
    rec["duel_capped"] = len(top) > VERIFY_DUEL_MAX
    wins = duel.get("wins") or {}
    champs = [s for s in pool if wins.get(s, 0) == max(wins.values())] if wins \
        else []
    if len(champs) == 1:
        winner = champs[0]
    elif incumbent in champs:
        winner = incumbent
    else:
        winner = vres["winner"]
    rec["winner_step"] = winner
    rec["changed"] = winner != incumbent
    return {"winner": winner, "record": rec}


# Where the write parse finds nothing, the poll does not happen at all -- and on
# TrajAudit that is 43 of 102 rows. The obvious reading, "the parse missed the
# framework's idiom", is wrong and was measured: 32 of those rows contain no
# write-looking step whatever, and across all 43 the reference step is one of
# the parsed writes exactly ONCE. Those trajectories are not under-parsed, they
# are decided somewhere a write-based answer space cannot reach, so widening the
# parse is the wrong repair.
#
# The repair is to give the poll a different answer space on exactly those rows.
# The traced graph already supplies one: the assistant steps on the dependency
# path back from the symptom. Measured on the 43, that pool holds a mean of 13.8
# steps -- the same size as the write sets on the rows that do get polled (13.8)
# -- and contains the reference step 72.1% of the time, against the write
# parse's 2.3%. It costs 592 calls and it is worth paying only because the
# rubric is sharp on this corpus: "depends" fires on 68.9% of reference steps
# against 13.3% of all candidates, and the reference step is top-scoring on
# 33 of 45 polls that contain it.
#
# The incumbent joins the pool when it is not already in it, so the merge keeps
# its meaning: the listwise answer is displaced only by a step that outscored it
# ON THE SAME FORM, never merely by being the only thing judged.
VERIFY_POOL_MAX = 20            # cap on the flow pool; ties broken by degree


def _flow_pool(traj: Trajectory, graph: FlowGraph,
               incumbent: Optional[int]) -> List[Dict[str, object]]:
    """The on-path assistant steps, as a fallback answer space for the poll.

    Ordered by traced degree when the cap binds, because degree is what the
    anchor rules already use and it keeps the pool centred on the steps the
    graph has most to say about; measured on TrajAudit, capping at 20 by degree
    loses nothing (0.721 with and without the cap).
    """
    on = set(graph.on_flow())
    pool = [s.index for s in traj.steps
            if s.role == "assistant" and s.index in on]
    if len(pool) > VERIFY_POOL_MAX:
        deg: Dict[int, int] = {}
        for e in graph.edges:
            deg[e.src] = deg.get(e.src, 0) + 1
            deg[e.dst] = deg.get(e.dst, 0) + 1
        pool = sorted(sorted(pool, key=lambda i: (-deg.get(i, 0), i))
                      [:VERIFY_POOL_MAX])
    if (incumbent is not None and 0 <= incumbent < len(traj.steps)
            and incumbent not in pool
            and traj.steps[incumbent].role == "assistant"):
        pool = sorted(pool + [incumbent])
    return [{"step": i, "label": "on the traced path from the symptom"}
            for i in pool]


# ============================================================================
# REFLECTION: listwise arbiter over candidate pool
# ============================================================================

REFLECT_BUDGET = 2600
REFLECT_TOTAL_CHARS = 16000
REFLECT_MIN_CHARS = 300
REFLECT_MAX_CHARS = 2000
REFLECT_HOPS_CAP = 6

REFLECT_SYSTEM = (
    "You examine multiple candidate steps from a failed agent trajectory and "
    "judge each one against a fixed rubric. Answer strictly in JSON."
)

REFLECT_PROMPT = """A coding agent FAILED this task. Below are CANDIDATE STEPS
from that run. For each candidate, judge whether it is the CRITICAL STEP.

Exactly ONE step in this run is the critical step. The critical step is the
EARLIEST step that COMMITTED the run to failing -- the step that introduced the
mistake, not a step that merely carried it forward or revealed it. If two steps
both had to go wrong, the critical step is the earlier one. A step that only
exposed a mistake made earlier is NOT the critical step.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

SYMPTOM STEPS: {anchors}
These are where the failure became VISIBLE. Being where a failure surfaces is
not evidence of having caused it.

HOW THE RUN WENT, PHASE BY PHASE:
{outline}

STRUCTURE, as traced (tracing is incomplete; treat it as weak evidence)
  step  later steps used its output   steps it consumed   hops to symptom
  ----  ----------------------------   -----------------   ---------------
{structure}

A step many later steps drew from is more likely to be where the run turned.
But a HIGH count can also mean the step was merely on the main path, and a LOW
count can mean tracing missed its effect. Compare the steps' TEXT against this
table; do not rank by the table alone.

CANDIDATE STEPS (judge each one):
{candidates}

For EACH candidate step, answer these four questions. The first two are GRADED:
use the whole range, and do not reserve the top grade for certainty. A grade of
2 means "probably", and most steps in a failed run deserve a 0.

  "repair" -- you may go back and redo EXACTLY ONE step of this run correctly
      and then let the run continue from there. How good a choice is this step?
        0 = redoing it changes nothing about how this run failed
        1 = redoing it helps, but the run still fails for the same reason
        2 = redoing it probably saves the run
        3 = redoing it saves the run, and nothing before it needed fixing

  "depends" -- but for what this step did, would the observed failure still
      have happened? This is the weakest claim: the step is on the causal chain.
        0 = no: the failure would have occurred anyway
        1 = the step contributed but was not decisive
        2 = the failure depends on this step
        3 = the failure depends entirely on this step

  "earlier_cause" -- was the run ALREADY doomed to this failure before this
      step began? This asks whether something earlier had ALREADY introduced
      the fault that this step then carried forward or revealed.
        0 = no: nothing before this committed the error
        1 = possibly: an earlier step may have started the problem
        2 = probably: an earlier step likely set the course
        3 = certainly: an earlier step had already made the failure inevitable

  "only_symptom" -- does this step merely reveal, report, or pass along a fault
      that was introduced somewhere else, without introducing one itself?
        0 = no: this step introduced new error
        1 = it partly reveals and partly introduces
        2 = it mostly reveals an earlier fault
        3 = it only reveals; it introduced nothing

Exploration is not an error: while the agent is still investigating options it
has not yet gone wrong. A step whose action is well-formed can still be at
fault if the DECISION behind it was wrong.

Return ONLY JSON with this structure:
{{
  "steps": [
    {{"step": <int>, "repair": <0-3>, "depends": <0-3>,
      "earlier_cause": <0-3>, "only_symptom": <0-3>,
      "why": "<1 sentence: what this step did and why it matters>"}},
    ...
  ],
  "pick": <int: which step number is the critical step>
}}

The "pick" field is your overall judgment of which ONE step is critical."""


def reflect_candidates(traj: Trajectory, spans: Sequence[Span],
                      graph: FlowGraph, llm: LLM,
                      digests: Sequence[StepDigest],
                      cands: Sequence[Dict[str, object]],
                      outline: Optional[Outline] = None,
                      outd: Optional[Dict[int, int]] = None
                      ) -> Optional[Dict[str, Any]]:
    """One listwise reflection call over all candidates, with structural table.

    Returns a record containing the model's per-step judgments and overall pick.
    Returns None if there are no candidates to judge.
    """
    steps = []
    for c in cands or []:
        try:
            i = int(c["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < len(traj.steps):
            steps.append(i)
    if not steps:
        return None
    steps = steps[:VERIFY_MAX]

    dmap = {d.index: d for d in (digests or [])}
    hops = {k: _hops_from_symptom(graph, k) for k in (DATA_FLOW, CONTROL_FLOW)}

    # Compute in-degrees
    indeg: Dict[int, int] = {}
    for e in graph.edges:
        indeg[e.dst] = indeg.get(e.dst, 0) + 1

    # Build shared head (identical to verify_candidates)
    head = dict(
        task=clip_middle(traj.task_description, 3000),
        failure_summary=graph.failure_summary or "(not summarised)",
        anchors=(", ".join("step %d" % a for a in sorted(graph.anchors))
                 if graph.anchors else "(none)"),
        outline=(outline.render_sections() if outline is not None
                 else "(not outlined)")
    )

    # Build structure table
    structure_lines = []
    for i in steps:
        od = min(outd.get(i, 0), VERIFY_OUTD_CAP) if outd else 0
        id_ = indeg.get(i, 0)
        h_data = hops[DATA_FLOW].get(i)
        h_ctrl = hops[CONTROL_FLOW].get(i)
        h = min(h_data if h_data is not None else 999,
                h_ctrl if h_ctrl is not None else 999)
        if h == 999:
            h_str = "not traced"
        else:
            h_str = str(min(h, REFLECT_HOPS_CAP))
        structure_lines.append("  %4d  %28d   %17d   %15s"
                              % (i, od, id_, h_str))

    # Build candidate text blocks with per-candidate budget
    n = len(steps)
    per_step_chars = max(REFLECT_MIN_CHARS,
                         min(REFLECT_MAX_CHARS, REFLECT_TOTAL_CHARS // n))

    # Compute on_flow set for _verify_where
    on_flow = set()
    for e in graph.edges:
        on_flow.add(e.src)
        on_flow.add(e.dst)

    cand_blocks = []
    for i in steps:
        d = dmap.get(i)
        text = (d.text if d is not None else (traj.steps[i].content or ""))
        where = _verify_where(traj, spans, graph, hops, on_flow,
                             set(graph.anchors) if graph.anchors else set(), i)
        clipped = clip_middle(text, per_step_chars)
        cand_blocks.append(
            "------------------------------------------------------------\n"
            "STEP %d\n"
            "WHERE IT SITS: %s\n"
            "------------------------------------------------------------\n"
            "%s\n"
            "------------------------------------------------------------"
            % (i, where, clipped or "(empty)")
        )

    prompt = REFLECT_PROMPT.format(
        task=head["task"],
        failure_summary=head["failure_summary"],
        anchors=head["anchors"],
        outline=head["outline"],
        structure="\n".join(structure_lines),
        candidates="\n\n".join(cand_blocks)
    )

    try:
        raw = llm.generate_obj(prompt, max_tokens=REFLECT_BUDGET,
                              system=REFLECT_SYSTEM)
    except Exception:
        return {"abstained": True, "raw": None, "steps": [], "pick": None,
                "n_missing": len(steps)}

    # Parse response
    if not isinstance(raw, dict):
        return {"abstained": True, "raw": raw, "steps": [], "pick": None,
                "n_missing": len(steps)}

    step_judgments = raw.get("steps")
    if not isinstance(step_judgments, list) or not step_judgments:
        return {"abstained": True, "raw": raw, "steps": [], "pick": None,
                "n_missing": len(steps)}

    # Grade parser (same as _g in verify_candidates)
    def _g(x):
        try:
            v = int(x)
            return max(0, min(3, v))
        except (TypeError, ValueError):
            return 0

    parsed = []
    covered = set()
    for entry in step_judgments:
        if not isinstance(entry, dict):
            continue
        try:
            step_num = int(entry.get("step"))
        except (TypeError, ValueError):
            continue
        if step_num not in steps:
            continue
        covered.add(step_num)
        parsed.append({
            "step": step_num,
            "repair": _g(entry.get("repair")),
            "depends": _g(entry.get("depends")),
            "earlier_cause": _g(entry.get("earlier_cause")),
            "only_symptom": _g(entry.get("only_symptom")),
            "why": (entry.get("why") or "")[:500]
        })

    n_missing = len([s for s in steps if s not in covered])
    pick = None
    try:
        pick_raw = raw.get("pick")
        if pick_raw is not None:
            pick = int(pick_raw)
            if pick not in steps:
                pick = None
    except (TypeError, ValueError):
        pass

    return {
        "abstained": False,
        "raw": raw,
        "steps": parsed,
        "pick": pick,
        "n_missing": n_missing
    }


def _merge_reflect(rrec: Optional[Dict[str, Any]], incumbent: Optional[int],
                  outd: Optional[Dict[int, int]],
                  use_outd: bool = True) -> Dict[str, Any]:
    """Merge reflection judgments into a single winner, with incumbent protection.

    Scoring formula matches the graded rubric:
      score = 2*repair + depends - 2*earlier_cause - 2*only_symptom
    Range is -4 to 9, same as VERIFY_GRADED_PROMPT.

    Tie-break and incumbent protection logic is copied verbatim from _merge_verify.
    """
    if rrec is None or rrec.get("abstained"):
        # Abstention: fallback depends on arm
        # Arm A (use_outd=True): fall back to outdeg-only (score=0 for all)
        # Arm B (use_outd=False): fall back to incumbent
        return {
            "winner": incumbent,
            "record": {"abstained": True, "n_missing": 0 if rrec is None else rrec.get("n_missing", 0)}
        }

    judgments = rrec.get("steps") or []
    if not judgments:
        return {
            "winner": incumbent,
            "record": {"abstained": True, "n_missing": rrec.get("n_missing", 0)}
        }

    # Compute scores: 2*repair + depends - 2*earlier_cause - 2*only_symptom
    scores = {}
    for j in judgments:
        step = j["step"]
        sc = (2 * j["repair"] + j["depends"]
              - 2 * j["earlier_cause"] - 2 * j["only_symptom"])
        scores[step] = sc

    # Add outdegree term if enabled
    if use_outd and outd:
        val = {s: scores[s] + VERIFY_OUTD_W * min(outd.get(s, 0), VERIFY_OUTD_CAP)
               for s in scores}
    else:
        val = scores

    best = max(val[s] for s in val)
    top = [j for j in judgments if val[j["step"]] == best]

    # Incumbent protection (verbatim from _merge_verify)
    inc = next((j for j in judgments if j["step"] == incumbent), None)
    if inc is not None and val[inc["step"]] >= best:
        winner = incumbent
        changed = False
    elif len(top) == 1:
        winner = top[0]["step"]
        changed = (winner != incumbent)
    else:
        # Tie: break by confidence (we don't have it, so just use -step)
        winner = min(top, key=lambda j: j["step"])["step"]
        changed = (winner != incumbent)

    return {
        "winner": winner,
        "record": {
            "asked": len(judgments),
            "judgments": judgments,
            "best_score": best,
            "n_tied": len(top),
            "incumbent": incumbent,
            "incumbent_score": (val[inc["step"]] if inc is not None else None),
            "changed": changed,
            "pick": rrec.get("pick"),
            "n_missing": rrec.get("n_missing", 0),
            "abstained": False
        }
    }


def _indent(text: str, pad: str) -> str:
    return "\n".join(pad + ln for ln in (text or "").splitlines()) or (pad + "(empty)")


def _names_failure_artefact(entity: str, failure_entities: Sequence[str]) -> bool:
    """Whether the thing an edge carries is one the failure is about.

    Matched by containment either way, because the tracer names an artefact as
    it appears in the step ("tests/test_cfg.py::test_load") while the failure
    summary may name it more loosely ("test_cfg"). Deliberately generous: this
    marks an edge as worth reading first, it does not accuse it.
    """
    ent = (entity or "").strip().lower()
    if len(ent) < 3:
        return False
    for f in failure_entities:
        f = (f or "").strip().lower()
        if len(f) >= 3 and (f in ent or ent in f):
            return True
    return False


def _flow_block(graph: FlowGraph, kind: str, max_edges: int = 60) -> str:
    """Render one relation's edges, newest-first, annotated.

    The two relations are rendered into separate blocks rather than one list.
    They are separate forests — painting walks them independently — so showing
    them together invites reading a path that alternates between the two, which
    is a path that does not exist.

    Each edge also carries three mechanical properties of the graph: how far up
    the chain it sits, whether the thing it carries is one the failure is about,
    and how many later steps depend on its source. All three were already in the
    graph and none costs a call; before this they were only implicit in the
    edge list, which meant the reader had to reconstruct the chain by hand from
    sixty unordered pairs. They are not scores and they are not ranked — the
    edge order is unchanged — because a weight fitted on the corpus would be
    tuned to it, and dropping low-scoring edges would lose the answer outright
    in the cases where the heuristic is wrong.
    """
    edges = [e for e in graph.edges if e.kind == kind]
    if not edges:
        return "  (none traced)"
    depth = _hops_from_symptom(graph, kind)
    fanout: Dict[int, set] = {}
    for e in edges:
        fanout.setdefault(e.src, set()).add(e.dst)
    lines = []
    for e in sorted(edges, key=lambda x: -x.dst)[:max_edges]:
        ent = (" [%s]" % e.entity) if (kind == DATA_FLOW and e.entity) else ""
        marks = []
        if e.src in depth:
            marks.append("%d hops from symptom" % depth[e.src])
        else:
            marks.append("not on a chain to any symptom")
        if _names_failure_artefact(e.entity, graph.failure_entities):
            marks.append("names a failure artefact")
        n_conv = len(fanout.get(e.src, ()))
        if n_conv >= 2:
            marks.append("%d chains converge here" % n_conv)
        why = (" — %s" % e.rationale) if e.rationale else ""
        lines.append("  step %d -> step %d%s (%s)%s"
                     % (e.src, e.dst, ent, "; ".join(marks), why))
    if len(edges) > max_edges:
        lines.append("  ... %d more %s edges omitted (oldest first)"
                     % (len(edges) - max_edges, kind))
    return "\n".join(lines)


def _vote(traj: Trajectory, objs: Sequence[Dict]) -> Dict[str, object]:
    """Majority vote over independent samples of the same prompt.

    Each sample is read the way a single answer would be -- parsed, then moved
    onto an attributable step -- before it votes, so a malformed or unusable
    draw drops out of the tally instead of poisoning it. Ties go to the EARLIEST
    step, the same rule the prompt itself states: a cause precedes its symptom,
    so when the evidence splits evenly the earlier reading is the one that has
    not yet assumed the fault was introduced late.

    ``agreement`` -- the winner's share of the votes cast -- is kept as the
    prediction's confidence. It is a measured quantity: across k draws it says
    how stable the answer actually was. The model's self-reported confidence is
    kept beside it but is not used to decide, because it is a claim about an
    answer rather than evidence about it.
    """
    votes: Dict[int, int] = {}
    first: Dict[int, Dict] = {}
    for obj in objs:
        try:
            step = int(obj.get("critical_step"))
        except (TypeError, ValueError):
            continue
        step = _snap_to_attributable(step, traj)
        if step is None:
            continue
        votes[step] = votes.get(step, 0) + 1
        first.setdefault(step, obj)
    if not votes:
        return {"step": None, "obj": {}, "votes": {}, "agreement": 0.0, "cast": 0}
    cast = sum(votes.values())
    best = max(votes, key=lambda s: (votes[s], -s))
    return {"step": best, "obj": first[best],
            "votes": {str(k): v for k, v in sorted(votes.items())},
            "agreement": round(votes[best] / float(cast), 4), "cast": cast}


def _merge_framings(traj: Trajectory, verdicts) -> Dict[str, object]:
    """Merge one answer per framing. Ties go to the INCUMBENT, not the earliest.

    ``verdicts`` is [(name, obj), ...] with the incumbent first. Each is read the
    way a single answer would be -- parsed, then moved onto an attributable step
    -- before it counts, so a malformed draw drops out instead of voting.

    The tie-break is the whole safety property of this stage. With three
    framings a three-way split carries no information, and the repo-wide
    "earliest wins" rule would resolve it by a positional prior that was
    measured NOT to hold here: inside the offered list the reference step sits
    at mean position 0.367 and our answers at 0.374, and a blind "always the
    first candidate" scores 0.167 against the model's 0.394. Falling back to the
    earliest of three model answers would therefore be trading a measured 0.394
    for a prior that is worse.

    Resolving to the incumbent instead means the answer can only MOVE when two
    framings agree against it. The arm's downside is bounded by construction
    rather than by hope, which is what the last five interventions lacked.
    """
    tally: Dict[int, int] = {}
    per: Dict[str, object] = {}
    objs: Dict[int, Dict] = {}
    incumbent = None
    for pos, (name, obj) in enumerate(verdicts):
        step = None
        if isinstance(obj, dict):
            try:
                step = _snap_to_attributable(int(obj.get("critical_step")), traj)
            except (TypeError, ValueError):
                step = None
        per[name] = step
        if pos == 0:
            incumbent = step
        if step is None:
            continue
        tally[step] = tally.get(step, 0) + 1
        objs.setdefault(step, obj)

    winner, agreed = incumbent, tally.get(incumbent, 0) if incumbent is not None else 0
    if tally:
        top = max(tally.values())
        if top > 1:
            # A real majority. Among steps tied at the top, the incumbent's own
            # answer keeps precedence; otherwise the earliest of them.
            best = [st for st, v in tally.items() if v == top]
            winner = incumbent if incumbent in best else min(best)
            agreed = top
    obj = objs.get(winner) or (verdicts[0][1] if verdicts else {})
    return {"winner": winner, "obj": obj if isinstance(obj, dict) else {},
            "record": {"asked": [n for n, _ in verdicts],
                       "answers": {k: v for k, v in per.items()},
                       "agreed": agreed, "winner": winner,
                       "incumbent": incumbent,
                       "changed": bool(winner != incumbent)}}


def _raw_symptom_evidence(traj: Trajectory) -> str:
    """Render harness evidence when a corpus supplies it.

    The block is observational input, not a location hint: it contains no
    reference step, annotation rationale, or reference patch.  It is kept out
    of the shipped prompt unless explicitly requested so existing cache keys
    and baselines remain byte-identical.
    """
    code = str(traj.meta.get("failed_test_code") or "").strip()
    error = str(traj.meta.get("failed_test_error") or "").strip()
    if not code and not error:
        return ""
    return """

RAW HARNESS EVIDENCE (observable outcome, not an attribution):
<failed_test_code>
%s
</failed_test_code>
<failed_test_error>
%s
</failed_test_error>
Use the assertion and error values to identify the exact violated requirement.
Then find the first trajectory step that committed to a state or plan
incompatible with that requirement. Do not infer a step number from this block.
""" % (clip_middle(code, 4000), clip_middle(error, 4000))


def localize(traj: Trajectory, spans: List[Span], graph: FlowGraph, llm: LLM,
             digests: Optional[Sequence[StepDigest]] = None,
             outline: Optional[Outline] = None,
             prior: Optional[SpanPrior] = None,
             prior_unranked: bool = False,
             hide_prior: bool = False,
             cands: Optional[List[Dict[str, object]]] = None,
             enforce_cands: bool = True,
             max_prompt_chars: int = 120000,
             samples: int = 1,
             cand_evidence: bool = False,
             framings: int = 1,
             verify: bool = False,
             verify_graded: bool = False,
             verify_chain: bool = False,
             verify_reflect: bool = False,
             verify_duel: bool = False,
             verify_pool: bool = False,
             verify_pool_union: bool = False,
             verify_outdeg: bool = False,
             roll: bool = False,
             expect: Optional[Dict[str, object]] = None,
             symptom_evidence: bool = False,
             analyst_findings: Optional[str] = None,
             judge_block: Optional[str] = None,
             rootcause: bool = False,
             cand_verbatim: Optional[Sequence[int]] = None,
             thinking: bool = False,
             cands_mixed: bool = False,
             alf_prompt: str = "",
             alf_flownote: str = "",
             obligation_block: Optional[str] = None,
             trajectory_override: Optional[str] = None,
             ) -> Prediction:
    """Localise over the outline (artifact 1) and step digests (artifact 2).

    ``prior`` is an optional region hint from ``span_prior``. ``cands`` is an
    optional restricted answer space from ``candidates``; ``enforce_cands``
    decides whether an answer outside it is moved into it or merely recorded as
    having left. When both stages are absent or empty the prompt is
    byte-identical to a run without either, so an A/B against a stored run
    measures the added stage and nothing else.

    ``samples`` > 1 asks the same prompt k times and takes the majority answer.
    The prompt does not change, so sample 0 is the stored single-sample answer
    and ``samples=1`` reduces to the original behaviour exactly -- which is what
    makes the extra draws, and not some other edit, the thing being measured.

    ``framings`` > 1 asks the same question as several different DECISION
    PROCEDURES -- see ``FRAMINGS`` -- and merges them. It is not ``samples``
    with a different temperature: the draws there are the same question asked
    twice, and on this corpus five of them changed nothing (0.3023 either way),
    which is what a stable-but-wrong answer looks like. Diversity has to come
    from the question, not from sampling noise.

    The two are mutually exclusive in effect: when ``framings`` > 1 the merge
    has already chosen an answer and ``samples`` is not consulted.

    ``verify`` replaces the listwise choice with a poll of pointwise absolute
    judgements -- one call per candidate, each shown no rivals -- and lets the
    result displace the listwise answer only when some candidate scores
    STRICTLY higher. It runs AFTER the candidate correction above, so its
    incumbent is the answer the run would otherwise have reported and its own
    winner is inside the answer space by construction.

    That last clause is the whole reason ``verify_pool_union`` exists. The poll
    draws from ``cands`` whenever a candidate list was passed, so the answer
    space stays restricted even with ``enforce_cands`` off: releasing the
    correction changes nothing, because the poll re-imposes it one stage later.
    Measured -- sp_csoft, 84 fresh calls, 0 of 86 rows moved. ``verify_pool``
    only substitutes the traced path when there is NO list at all, which is a
    different arm (and one that cost 5 rows on SP, because losing the list hurt
    the 66 rows it was getting right more than it helped the 18 it had locked
    out). ``verify_pool_union`` is the third option: keep the list intact and
    ADD the traced steps, so nothing that was working loses a signal.

    ``expect`` is the record from ``expect.assess`` -- per candidate, how many
    of the requirements written for it BEFORE it was read are satisfied. It
    renders to nothing when absent, and also when the stage ran but produced no
    requirements, so a run without the stage and a run where it had nothing to
    say both send the shipped prompt.
    """
    digests = list(digests or [])
    body = (trajectory_override if trajectory_override is not None
            else render_compressed(traj, spans, digests))
    if trajectory_override is None and len(body) > max_prompt_chars:
        # ``cand_verbatim`` names the steps that 4c restored to their own
        # words. Without this the trimmer would clip them first -- they are
        # off-path on most rows -- and the arm would report a restoration it
        # did not send. It is passed explicitly rather than read off ``cands``
        # because the two are not the same set: ``cands`` is dropped when the
        # list is not meant to restrict the answer, and the restoration still
        # happened.
        body = _fit_budget(traj, spans, digests, max_prompt_chars,
                           keep=(cand_verbatim or ()))

    outline_text = outline.render() if outline is not None else ""
    has_prior = bool(prior and prior.guesses) and not hide_prior
    has_cands = bool(cands)

    # The roll, if asked for. Decided here so that everything below -- the
    # prompt, the answer budget, the record -- reads one list, and so that a
    # row with nothing to enumerate degrades to the ordinary prompt rather than
    # to an empty instruction. Truncating a long list is deliberately NOT an
    # option: a partial roll is the behaviour this stage exists to prevent, and
    # a silently shortened one would look like compliance in the record.
    roll_steps: List[int] = []
    roll_skip = ""
    if roll:
        roll_steps = [i for i in sorted(graph.on_flow())
                      if 0 <= i < len(traj) and traj.steps[i].attributable]
        if len(roll_steps) < ROLL_MIN:
            roll_skip, roll_steps = "too_few", []
        elif len(roll_steps) > ROLL_MAX:
            roll_skip, roll_steps = "too_many", []
    roll_on = bool(roll_steps)

    # Rendered once: the same block goes to every framing and every sample, so
    # that what differs between two draws is still only what the draw is for.
    expect_block = render_expect(expect)

    # The three flow-evidence slots, hoisted out of ``_prompt`` so the ALFWorld
    # suppression arm can substitute them without touching the format call.
    _anchors_line = (", ".join("step %d" % a for a in sorted(graph.anchors))
                     or "(none identified)")
    _data_block = _flow_block(graph, DATA_FLOW)
    _control_block = _flow_block(graph, CONTROL_FLOW)

    # ALFWorld only, and only when asked for. ``blocks`` returns None for a
    # falsy mode, so every TA/SP run keeps the three values above verbatim and
    # sends a byte-identical prompt to the same cache key. This is the opposite
    # intervention to ``--alf-anchors``: that one paints the flow wider, this
    # one removes it, because the same measurement supports both readings and
    # only intervening in both directions can separate them. See
    # alfworld_flownote and probe/prereg_alf_flownote.md.
    if alf_flownote:
        _fn = alfworld_flownote.blocks(alf_flownote, _data_block,
                                       _control_block, _anchors_line)
        if _fn is not None:
            _data_block = _fn["data_block"]
            _control_block = _fn["control_block"]
            _anchors_line = _fn["anchors"]

    def _prompt(head: str, note: str) -> str:
        formatted = LOCALIZE_PROMPT.format(
        head=head, framing_note=note,
        task=clip_middle(traj.task_description, 3000),
        failure_summary=graph.failure_summary or "(not summarised)",
        symptom_evidence=(_raw_symptom_evidence(traj)
                          if symptom_evidence else ""),
        anchors=_anchors_line,
        data_block=_data_block,
        control_block=_control_block,
        roll_block=(ROLL_BLOCK.format(
            n=len(roll_steps),
            steps=", ".join("step %d" % i for i in roll_steps))
            if roll_on else ""),
        roll_fields=(ROLL_FIELDS if roll_on else ""),
        expect_block=expect_block,
        outline=clip_middle(outline_text, 12000) or "(no outline available)",
        span_prior=((PRIOR_BLOCK_UNRANKED.format(
                        guesses=render_prior_unranked(prior))
                     if prior_unranked
                     else PRIOR_BLOCK.format(guesses=prior.render()))
                    if has_prior else ""),
        prior_fields=(PRIOR_FIELDS if has_prior else ""),
        # One list, whichever arm is running. ``cands_mixed`` only swaps the
        # PROSE for a version that describes a list containing non-write steps;
        # the entries themselves go through the same renderer either way, in
        # trajectory order, with nothing marking where an entry came from.
        candidates=((_cand_block(enforce_cands, cands_mixed))
                    .format(cands=(render_candidate_evidence(
                        traj, spans, graph, digests, cands)
                        if cand_evidence
                        else candidates_mod.render(cands)))
                    if has_cands else ""),
        analyst_block=(analyst_findings or ""),
        # ``span_judge.render_verdicts`` already returns "" when no phase earned
        # a band, so a run where every phase was cleared sends the same prompt
        # as a run without the stage -- the block appears only where there is
        # something in it to read.
        judge_block=(judge_block or ""),
        # Both slots are "" when the flag is off, so a run without it sends a
        # byte-identical prompt and hits the same cache entries. With a
        # candidate list in force the block defers to it rather than
        # contradicting it -- the list's text is frozen and its enforcement is
        # in code, so an answer naming an unlisted plan step would be snapped
        # away and the block would be describing a mechanism that does not
        # exist.
        rootcause_block=(("" if not rootcause else
                          ROOTCAUSE_BLOCK_CAND if has_cands
                          else ROOTCAUSE_BLOCK)),
        rootcause_fields=(ROOTCAUSE_FIELDS if rootcause else ""),
        trajectory=body)
        if obligation_block:
            # Splice only on the opt-in path. Keeping the frozen template free
            # of a new format field preserves all older callers and makes the
            # off-path prompt byte-identical, including direct test callers.
            marker = "\n\nHOW TO CHOOSE:"
            formatted = formatted.replace(
                marker, "\n\n" + obligation_block + marker, 1)
        return formatted

    # Select the prompt head based on thinking flag
    if thinking:
        prompt_head = THINKING_HEAD
        prompt_note = ""
    else:
        prompt_head = FRAMINGS[0][1]
        prompt_note = FRAMINGS[0][2]

    # ALFWorld only, and only when asked for. ``head_note`` returns None for a
    # falsy variant, so every TA/SP run keeps the two lines above verbatim and
    # sends a byte-identical prompt to the same cache key. Applied after the
    # thinking branch so that the two framings cannot silently both apply.
    if alf_prompt:
        _alf = alfworld_prompt.head_note(alf_prompt,
                                         (prompt_head, prompt_note))
        if _alf is not None:
            prompt_head, prompt_note = _alf

    prompt = _prompt(prompt_head, prompt_note)
    # A roll of up to ROLL_MAX lines does not fit beside the reason in the
    # ordinary budget, and a roll truncated by max_tokens would arrive as
    # unparseable JSON -- i.e. as a silent fallback to no roll at all. Raised
    # only when the roll is actually being written, so every other arm keeps
    # its cache key (model, system, temperature, max_tokens, prompt) intact.
    budget = ROLL_BUDGET if roll_on else ANSWER_BUDGET

    nf = max(1, min(len(FRAMINGS), int(framings or 1)))
    frame_rec = None
    if nf > 1:
        # The incumbent's prompt is byte-identical to a framings=1 run, so that
        # draw is served from cache and only the added framings are paid for.
        asks = [(n, _prompt(h, note)) for n, h, note in FRAMINGS[:nf]]
        draws = map_concurrent(
            lambda a: llm.generate_obj(a[1], max_tokens=budget,
                                       system=LOCALIZE_SYSTEM),
            asks, workers=nf)
        merged = _merge_framings(
            traj, [(asks[i][0], draws[i] if isinstance(draws[i], dict) else {})
                   for i in range(len(asks))])
        frame_rec = merged["record"]
        obj = merged["obj"]
        vote = None

    k = max(1, int(samples or 1))
    if frame_rec is not None:
        # Already answered. ``obj`` is the winning framing's own object, so the
        # parse, the snap and the candidate correction below all run on it
        # exactly as they would on a single answer -- the merge chose which
        # answer to read, it did not become one.
        pass
    elif k == 1:
        obj = llm.generate_obj(prompt, max_tokens=budget,
                               system=LOCALIZE_SYSTEM)
        vote = None
    else:
        draws = map_concurrent(
            lambda v: llm.generate_obj(prompt, max_tokens=budget,
                                       system=LOCALIZE_SYSTEM, variant=v),
            list(range(k)), workers=k)
        vote = _vote(traj, [d for d in draws if isinstance(d, dict)])
        obj = vote["obj"]

    step = obj.get("critical_step")
    try:
        step = int(step)
    except (TypeError, ValueError):
        step = None

    step = _snap_to_attributable(step, traj)
    if step is None:
        step = _fallback_step(traj, graph, prior)

    # What the model did on its own is recorded before anything moves it,
    # because "right after being snapped" and "right unaided" are different
    # results. Under ``enforce_cands`` the answer is corrected into the set;
    # otherwise the departure is recorded and left standing — on TrajAudit the
    # correction cost three answers that were already on the reference step, so
    # enforcement has to be separable from the list that suggested it.
    cand_record = None
    if has_cands:
        forced = candidates_mod.snap(cands, step)
        cand_record = {
            "n": len(cands),
            "steps": [int(c["step"]) for c in cands],
            "obeyed": forced == step,
            "enforced": bool(enforce_cands),
            "moved_from": (step if (enforce_cands and forced != step) else None),
        }
        if cands_mixed:
            # Which of the two wordings this row got. ``n``/``steps`` already
            # record the answer space itself; this says whether the prose
            # describing it was the one that admits non-write entries, which is
            # the only thing that differs between the arms and so the only thing
            # an A/B needs to separate them by.
            cand_record["mixed"] = True
        if enforce_cands:
            step = forced

    # Pointwise verification, if asked for. The incumbent is whatever the
    # listwise path just settled on, so with the knob off this whole stage does
    # not exist and the answer is unchanged.
    verify_rec = None
    # The pool is the answer space, always. Holding it back to the parsed subset
    # was tried for one arm and produced the worst reading on this axis: it made
    # answering with a merged-in step self-defeating, since an unpolled answer
    # lost to whatever scored best among the polled ones (68 rows, overridden
    # 68 times). Whatever may be answered with must also be allowed to vote.
    vpool = list(cands or []) if has_cands else []
    pool_kind = "candidates" if has_cands else ""
    if verify and not has_cands and verify_pool:
        vpool = _flow_pool(traj, graph, step)
        pool_kind = "flow" if vpool else ""
    elif verify and has_cands and verify_pool_union:
        # Measured gap this repairs: on SP, 18 of the 84 rows with a usable
        # candidate set have gold OUTSIDE it, so those rows cannot be answered
        # at all -- the list caps accuracy at 0.786. But dropping the list
        # outright (sp_cnocand) woke 2 of those 18 and cost 7 of the 66 rows
        # where gold IS in the list, for a net -5. The union keeps the list
        # whole and only ADDS the traced steps, so the rows the list was
        # getting right keep every signal they had.
        #
        # This is also the only way to loosen the answer space at all: merely
        # turning off ``enforce_cands`` does nothing, because the poll below
        # draws from ``cands`` and re-imposes the restriction one stage later.
        # sp_csoft measured exactly that -- 84 fresh calls, 0 rows moved.
        seen_steps = {int(c["step"]) for c in vpool}
        vpool = vpool + [c for c in _flow_pool(traj, graph, step)
                         if int(c["step"]) not in seen_steps]
        pool_kind = "candidates+flow"
    if verify and vpool:
        if verify_reflect:
            rrec = reflect_candidates(traj, spans, graph, llm, digests, vpool,
                                     outline,
                                     outd=_outdegrees(graph, [int(c["step"])
                                                              for c in vpool]))
            vres = _merge_reflect(rrec, step,
                                 _outdegrees(graph, [int(c["step"])
                                                     for c in vpool])
                                 if verify_outdeg else None,
                                 use_outd=verify_outdeg)
            verify_rec = vres["record"]
            if verify_rec is not None:
                verify_rec["pool"] = pool_kind
                verify_rec["reflect"] = rrec
            if vres["winner"] is not None:
                step = vres["winner"]
        else:
            verdicts = verify_candidates(traj, spans, graph, llm, digests, vpool,
                                         outline, graded=verify_graded,
                                         chain=verify_chain)
            vres = _merge_verify(
                verdicts, step,
                _outdegrees(graph, [v["step"] for v in verdicts])
                if verify_outdeg else None)
            if verify_duel:
                vres = _apply_duel(
                    vres, verdicts, step,
                    lambda tied: verify_duels(traj, spans, graph, llm, digests,
                                              tied, outline))
            verify_rec = vres["record"]
            if verify_rec is not None:
                verify_rec["pool"] = pool_kind
            if vres["winner"] is not None:
                step = vres["winner"]

    conf = obj.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.0

    return Prediction(
        task_id=traj.task_id, critical_step=step, confidence=conf,
        reason=_with_rejection(obj),
        error_type=str(obj.get("error_type") or "")[:80],
        span_prior=(_prior_record(obj, prior, step) if prior is not None
                    else None),
        candidates=cand_record,
        samples=({"cast": vote["cast"], "votes": vote["votes"],
                  "agreement": vote["agreement"], "asked": k}
                 if vote is not None else None),
        framings=frame_rec,
        verify=verify_rec,
        roll=(_roll_record(obj, roll_steps, roll_skip, step)
              if roll else None),
        # Stored whole -- every requirement and every verdict, not just the
        # per-step counts the prompt showed. The counts are one rule for
        # reading this evidence and almost certainly not the best one; with
        # the requirements on disk, a merge that weights violations, or ranks
        # by violation rate, or ignores "unclear", can be scored against this
        # run offline for nothing. Re-running to ask that costs ~1,800 calls.
        expect=expect,
    )


def _roll_record(obj: Dict, asked: Sequence[int], skipped: str,
                 step: Optional[int]) -> Dict[str, object]:
    """What the roll call actually produced, including what it left out.

    ``missed`` is the point of this record. The stage exists because the reader
    passes over the responsible step without ever naming it, so a run has to be
    readable for whether the enumeration HAPPENED, separately from whether the
    accuracy moved. A roll that silently covers half the list and still answers
    would otherwise be indistinguishable in the numbers from one that did not
    run at all -- and it is the first thing to check if the arm loses, because
    "the instruction was ignored" and "the instruction did not help" are
    different findings with different next moves.

    Nothing here changes the answer. The model was told to pick from the steps
    it marked ``turning``, but that is not enforced: an enforcement would make
    this a restricted answer space, which is a separate intervention that has
    already been measured on its own, and the two must not arrive together.
    """
    seen: List[int] = []
    turning: List[int] = []
    for row in (obj.get("roll") or []):
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("step"))
        except (TypeError, ValueError):
            continue
        if i not in seen:
            seen.append(i)
        if row.get("turning") is True and i not in turning:
            turning.append(i)
    asked = list(asked)
    missed = [i for i in asked if i not in set(seen)]
    return {
        "asked": len(asked),
        "steps": asked,
        "returned": seen,
        "missed": missed,
        "extra": [i for i in seen if i not in set(asked)],
        "coverage": (round(1.0 - len(missed) / float(len(asked)), 4)
                     if asked else None),
        "turning": turning,
        "answer_in_turning": (step in turning) if turning else None,
        "skipped": skipped,
    }


def _prior_record(obj: Dict, prior: SpanPrior,
                  step: Optional[int]) -> Dict[str, object]:
    """The prior, plus whether the localiser actually followed it.

    ``chosen`` is recomputed from the answer rather than taken from the model's
    ``prior_span`` field: it will sometimes name a phase that does not contain
    the step it just gave. Trusting the self-report would put a number in the
    diagnostics that measures the model's bookkeeping instead of its behaviour.
    The claim is kept beside it as ``claimed`` — a run where the two disagree
    often is a run where the prior block was not really read.
    """
    hit = prior.covers(step) if step is not None else None
    try:
        claimed = int(obj.get("prior_span"))
    except (TypeError, ValueError):
        claimed = None
    rec = dict(prior.to_dict())
    rec["chosen"] = hit.start if hit is not None else None
    rec["claimed"] = claimed
    rec["disagreement"] = str(obj.get("prior_disagreement") or "")[:300]
    return rec


def _with_rejection(obj: Dict) -> str:
    """Keep the ruled-out step in the reason, where analysis can read it back.

    The localiser is asked to name the later step it considered and rejected.
    That is the whole point of the walk-backwards instruction, so it has to be
    recorded: if the field is empty across a run, the model skipped the walk
    rather than doing it and concluding the symptom was right.

    The same applies to the upstream-link test (``ROOTCAUSE_BLOCK``), and more
    sharply: that block only earns its place if the test was performed, and the
    only way to know is to read the verdict back. ``prediction.reason`` is the
    one free-text field the artifact keeps, so both go here. When the flag is
    off the model is never asked for those fields, the keys are absent, and
    this returns exactly what it returned before.
    """
    text = str(obj.get("reason") or "")[:1500]
    try:
        rej = int(obj.get("rejected_step"))
    except (TypeError, ValueError):
        pass
    else:
        why = str(obj.get("rejected_because") or "")[:300]
        text = "%s\n[ruled out step %d%s]" % (text, rej,
                                              (": %s" % why) if why else "")
    verdict = str(obj.get("upstream_verdict") or "").strip()[:40]
    if verdict:
        try:
            up = int(obj.get("upstream_step"))
        except (TypeError, ValueError):
            up = None
        because = str(obj.get("upstream_because") or "")[:300]
        text = "%s\n[upstream %s%s%s]" % (
            text, verdict,
            ("" if up is None else " at step %d" % up),
            (": %s" % because) if because else "")
    return text


def _fit_budget(traj: Trajectory, spans: List[Span],
                digests: Sequence[StepDigest], budget: int,
                keep: Sequence[int] = ()) -> str:
    """Shrink off-path steps until the trajectory fits.

    Off-path steps give way before on-path ones: the localiser can tolerate a
    thinner narrative, but not missing evidence. Clipping the rendered blob
    instead would drop whole steps from the tail, which is where the failure
    surfaces.

    ``keep`` names steps that may not be shrunk at all. It exists for
    ``--candidates-verbatim``: a candidate step is off-path on most rows, so
    the first pass below would clip the restored text to 400 characters and
    undo the intervention -- silently, and worst on the longest rows, i.e.
    exactly where it was biggest. Measured on SP: 2 of 86 rows cross the
    120000-char cap once the answer space is restored, and both would have
    reported "restored" in the artifact while sending a 400-char summary.
    Empty by default, so every arm that does not ask for this takes the path
    that ran before the argument existed.
    """
    import copy

    held = {int(i) for i in (keep or ())}
    work = [copy.copy(d) for d in digests]
    for limit in (400, 220, 120):
        body = render_compressed(traj, spans, work)
        if len(body) <= budget:
            return body
        for d in work:
            if d.index in held:
                continue
            if not d.on_flow and len(d.text) > limit:
                d.text = clip_middle(d.text, limit)
    body = render_compressed(traj, spans, work)
    if len(body) <= budget:
        return body
    # still too big: clip the longest on-path steps, largest first
    order = sorted((d for d in work if d.on_flow and d.index not in held),
                   key=lambda d: len(d.text), reverse=True)
    for d in order:
        if len(render_compressed(traj, spans, work)) <= budget:
            break
        d.text = clip_middle(d.text, max(600, len(d.text) // 2))
    body = render_compressed(traj, spans, work)
    logger.info("compressed trajectory trimmed to %d chars", len(body))
    return clip_middle(body, budget)


def _snap_to_attributable(step: Optional[int], traj: Trajectory) -> Optional[int]:
    """Move an out-of-range or non-attributable answer to the nearest valid step."""
    if step is None:
        return None
    n = len(traj)
    if n == 0:
        return None
    step = max(0, min(step, n - 1))
    if traj.steps[step].attributable:
        return step
    idxs = traj.attributable_indices
    if not idxs:
        return step
    # prefer the closest earlier agent step: the cause precedes the symptom
    earlier = [i for i in idxs if i <= step]
    if earlier:
        return earlier[-1]
    return idxs[0]


def _fallback_step(traj: Trajectory, graph: FlowGraph,
                   prior: Optional[SpanPrior] = None) -> Optional[int]:
    """If the model gave nothing usable, take the earliest traced flow step --
    but inside the prior's top region when there is one.

    The unqualified "earliest on-flow step" rule answers step 0 on most
    TrajAudit trajectories, and step 0 is a position the reference step
    essentially never occupies: measured on ``ta_qarm_fx`` and ``ta_qbase_fx``,
    4 rows each landed here and NONE of them had gold == 0, so the old rule
    converted a non-answer into a certain miss. The prior's top guess is the
    pipeline's own best region estimate and is already paid for, so preferring
    it costs nothing and cannot be worse than a position known to be wrong.

    Deliberately NOT calibrated against where reference steps actually sit --
    that distribution is a label and reading it here would be selection on the
    gold, the same rule that forbids tuning thresholds after a run. Only
    pipeline-internal quantities are admissible, and the within-region rule is
    the old rule unchanged.
    """
    on_flow = sorted(i for i in graph.on_flow()
                     if 0 <= i < len(traj) and traj.steps[i].attributable)
    guesses = list(getattr(prior, "guesses", None) or [])
    if guesses:
        top = guesses[0]
        inside = [i for i in range(int(top.start), int(top.end) + 1)
                  if 0 <= i < len(traj) and traj.steps[i].attributable]
        if inside:
            flowed = [i for i in inside if i in set(on_flow)]
            return (flowed or inside)[0]
    if on_flow:
        return on_flow[0]
    idxs = traj.attributable_indices
    return idxs[0] if idxs else None
