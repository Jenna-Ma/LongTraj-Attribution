# -*- coding: utf-8 -*-
"""ALFWorld-specific localiser framing. Additive: nothing here is reachable
unless ``Config.alf_prompt`` is set, and the TA/SP prompts are untouched.

WHY THIS FILE EXISTS. Every head in ``localize.py`` opens with "A coding agent
FAILED this task", and the body explains the symptom in terms of patches,
tests, files and field names. ALFWorld is an embodied household agent in a text
simulator: there is no patch, no test and no file. The question being asked is
the same; the vocabulary it is asked in is from another domain.

WHAT THE RUBRIC IS, AND WHAT IT IS NOT. The ALFWorld corpus annotates each
failed run with one COGNITIVE MODULE (memory / planning / reflection / action /
system) and a failure type within it. That taxonomy is a property of the
annotation guideline, not of any single record -- it is the rubric a human
annotator was handed. Restating it to the localiser is the same kind of act as
telling a grader what the marking scheme is.

It is NOT the label. No per-record ground truth is read, derived or embedded
here: the text below is a fixed constant, identical for every record, and
contains no step index. ``tests/test_no_leak.py`` §56 pins that mechanically
(no ``step <n>``, none of the gold field names, and a byte-identical prompt
when the flag is off).

The module/type distribution that motivated the wording was counted on the DEV
half only (``probe/prereg_alf_prompt.md`` §1.2), so the test half is untouched
by the design of this file.

TWO PLACES THE HOUSE PROMPT CONTRADICTS THE GUIDELINE, which is the whole
reason a rubric is worth sending:

  * ``HOW TO CHOOSE`` rule 1 says "Exploration is not an error." But
    ``inefficient_plan`` is the single largest failure type on dev -- the
    annotators DO blame unsystematic search. The house rule tells the model to
    skip exactly the steps the guideline marks.
  * The ``system`` module blames the ENVIRONMENT: the agent issued a correct,
    admissible action and the simulator refused it. The house prompt has no
    slot for "this step was right and the world was wrong", so that verdict is
    unreachable no matter how the model reasons.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from .schema import NON_ATTRIBUTABLE_ROLES

logger = logging.getLogger(__name__)

# The head replaces INCUMBENT_HEAD. Same question, ALFWorld's nouns.
ALF_HEAD = """A household robot agent FAILED this task in ALFWorld, a
text-based simulated home. Find the critical step: the EARLIEST step whose
mistake causally led to the observed failure."""

# --- P1: domain only -------------------------------------------------------
# Corrects the vocabulary and nothing else, so that P2 minus P1 isolates what
# the rubric buys as opposed to what merely saying "household, not code" buys.
ALF_DOMAIN_NOTE = """
WHAT THE DOMAIN ACTUALLY IS (read this before the generic guidance above; where
the two disagree, this section wins).

This run is NOT a coding task. There is no patch, no test suite, no source
file. The agent moves around a simulated home and manipulates objects. Each
`assistant` turn is one natural-language ACTION (`go to drawer`, `take apple`,
`open fridge`, `clean plate with sinkbasin`, `heat mug with microwave`). Each
following `user` turn is the simulator's reply to that action, plus the list of
actions it will currently accept.

Translate the generic guidance accordingly:
  * "the patch was submitted / the test reported its error" means: the run hit
    its step limit, or ended still holding the wrong object, or never reached
    the goal state. The failure is usually NOT announced -- the run simply
    stops. Absence of a loud error is not evidence that nothing went wrong.
  * "a wrong value / a wrong file / a wrong field name" means: a wrong BELIEF
    about the world. Which room-fixture the agent thinks it stands at, whether
    it thinks it already searched somewhere, whether it thinks it is holding
    the object, whether it thinks an action succeeded.
  * There is no compilation and no type error. Almost every wrong action in
    this domain is well-formed English that the simulator was willing to parse.
    Judging a step by whether it LOOKS malformed will find nothing.

OBSERVATION TIERS. The room-level view lists only large furniture; small
objects are invisible until the agent goes to the furniture holding them. So
"the object was not in the room description" is NOT evidence the agent should
have known the object was absent, and blaming a step for failing to notice it
there is a misreading of the environment."""

# --- P2: the annotation guideline ------------------------------------------
# Fixed constant, no record-specific content. Ordered by dev frequency so that
# the categories the guideline actually uses come first.
ALF_RUBRIC_NOTE = """
HOW THIS CORPUS DEFINES "THE CRITICAL STEP" (the marking scheme; it overrides
the generic guidance wherever the two disagree).

The step is attributed to the COGNITIVE MODULE that failed, not to the moment
the task became unwinnable. Choose the step where one of the following first
happened. They are listed in descending order of how often they are the answer
in this corpus:

1. PLANNING -- the agent commits to a plan that cannot work or need not work.
   * inefficient plan: it searches without system -- revisits places it already
     searched, wanders between fixtures with no ordering, or keeps probing
     where the object cannot be. THIS IS THE MOST COMMON ANSWER IN THIS CORPUS.
     It directly overrides the generic rule "Exploration is not an error":
     in this domain, exploration that is redundant or unsystematic IS the
     error, and the step that commits to it is the step to name.
   * constraint ignorance: it ignores a precondition the environment requires
     (trying to clean an object while holding it when the object must be put
     down first; trying to use a receptacle it has not opened).
   * impossible action: it plans something the environment never offers --
     the intended action is absent from the admissible list and no variation
     of it exists.

2. REFLECTION -- the agent misreads what just happened.
   * outcome misinterpretation: it treats a failed action as having succeeded,
     or a successful one as having failed, and proceeds on that belief.
   * progress misjudge: it believes it is closer to the goal than it is --
     thinks it holds the object, thinks a place is exhausted, thinks a
     subgoal is done.
   Mark the step that carries the misreading forward, i.e. the agent turn that
   reacts to the observation, not the observation itself.

3. MEMORY -- the agent loses or corrupts what it already knew.
   * over-simplification: it drops the record of where it has already been,
     and so goes back there later.
   * retrieval failure: the information is in its history and it fails to use
     it.
   * hallucination: it asserts having seen or done something it never did.
   Name the step where the knowledge was LOST or FABRICATED, which is normally
   earlier -- often much earlier -- than the step where the loss shows up as a
   wasted move.

4. ACTION -- the plan is sound and the emitted action is not: malformed, aimed
   at the wrong object, or outside the admissible list when a listed
   alternative would have done the same job.

5. SYSTEM -- the environment is at fault and the agent is not. The agent issued
   a sound, admissible, correctly-sequenced action and the simulator still
   refused it or returned nothing; or the run was cut off by the step limit
   while the agent was making real progress. THIS VERDICT IS AVAILABLE AND IS
   SOMETIMES CORRECT. When it is, name the step where the environment first
   misbehaved, even though that step is not the agent's mistake. Do not reach
   for it merely because no agent error is obvious -- it is the rarest of the
   five -- but do not rule it out on the grounds that a step the agent got
   right cannot be the answer.

PRECEDENCE WHEN SEVERAL FIT. Prefer the module whose failure came FIRST in
time. A memory loss that later produced a bad plan is a memory answer, not a
planning one. A misread observation that later produced a wasted search is a
reflection answer, not a planning one. Only call it planning when the plan was
already wrong on the information the agent actually had at that moment."""

# --- P3: the one structural signal the corpus carries ----------------------
# Soft by construction. On dev the failed-action pool holds the answer 44.7% of
# the time against a 31.7% base rate: a 1.41x lift, which is worth mentioning
# and nowhere near worth filtering on. Stating the strength keeps the model
# from treating it as a rule.
ALF_SIGNALS_NOTE = """
READING THE SIMULATOR'S REPLIES.

`Nothing happens.` is the simulator's refusal: the action parsed but changed
no state. It is the clearest no-progress signal in this domain, and it is the
SYMPTOM, not automatically the cause. Three different answers are consistent
with seeing one:
  * the action was wrong (ACTION), or its precondition was unmet (PLANNING);
  * the action was right and the agent then misread the refusal, or missed it
    entirely and carried on (REFLECTION) -- in which case the step to name is
    the agent's next turn, not the refused action;
  * the action was right, admissible and correctly sequenced, and the refusal
    was the environment's fault (SYSTEM).

Calibration, measured on this corpus: a refused action is somewhat more likely
than an average step to be the answer, but only somewhat -- well under half of
the answers are refused actions, and most refused actions are not the answer.
Treat it as a place to look, never as the choice itself. In particular an
agent turn that drew no refusal can still be the answer, and frequently is:
the planning and memory failures above usually draw no refusal at all, because
every individual action in them succeeds.

A re-observation identical to an earlier one with no state change between them
is the same kind of signal. But note that `examine <furniture>` right after
`go to <furniture>` returns the same inventory by design -- that repetition is
a redundant probe, not evidence the environment refused anything."""


def _compose(*parts: str) -> str:
    return "\n".join(p.rstrip("\n") for p in parts if p) + "\n"


# variant -> (head, note). The note lands in LOCALIZE_PROMPT's {framing_note}
# slot, after HOW TO CHOOSE, so that where it contradicts the house rules it is
# the text the model read last.
VARIANTS: Dict[str, Tuple[str, str]] = {
    "domain": (ALF_HEAD, _compose(ALF_DOMAIN_NOTE)),
    "rubric": (ALF_HEAD, _compose(ALF_DOMAIN_NOTE, ALF_RUBRIC_NOTE)),
    "full": (ALF_HEAD, _compose(ALF_DOMAIN_NOTE, ALF_RUBRIC_NOTE,
                                ALF_SIGNALS_NOTE)),
}

# The noise control. It must be the INCUMBENT head, not ALF_HEAD -- a control
# that already swapped the head would be measuring the head swap, which is
# arm P1's job. So the incumbent is passed in rather than imported (localize
# imports this module) or copied (a copy would drift the day the head is
# edited). The trailing space is the entire change: same words, different
# cache key, so the gap against the baseline is one re-draw of the localiser
# and nothing else. Without this arm, "how much of a gap is noise" would have
# to be assumed; last round showed that assuming it gets it wrong.
NULL_VARIANT = "null"


def head_note(variant: Optional[str],
              incumbent: Tuple[str, str]) -> Optional[Tuple[str, str]]:
    """Return ``(head, note)`` for an ALFWorld variant, or None when off.

    ``incumbent`` is the ``(head, note)`` the caller would otherwise have
    used; only the ``null`` control consults it.

    An unknown variant raises rather than silently falling back: a typo in a
    flag that quietly ran the incumbent would be recorded in the artifact as
    the variant having run, which is the one failure mode that corrupts an
    arm comparison after the fact.
    """
    if not variant:
        return None
    if variant == NULL_VARIANT:
        return (incumbent[0] + " ", incumbent[1])
    try:
        return VARIANTS[variant]
    except KeyError:
        raise ValueError("unknown alf_prompt variant %r; known: %s"
                         % (variant, ", ".join(sorted(VARIANTS) +
                                               [NULL_VARIANT])))



# ---------------------------------------------------------------------------
# Action-first rendering. Additive and default-off: ``Config.alf_actions`` is
# False everywhere but an ALFWorld arm that asks for it, and the pipeline does
# not even call this function when it is off, so every TA/SP digest is
# byte-identical and every upstream call still hits its cache.
# ---------------------------------------------------------------------------

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S)
ACTION_LEAD = "ACTION TAKEN AT THIS STEP: "

# How much of the action line to keep. ALFWorld actions are short commands
# ("go to cabinet 1", "take soapbar 1 from countertop 1"); 300 chars is far
# above the longest one measured on this corpus, so the clip is a guard
# against a malformed block, not a compression decision.
ACTION_CLIP = 300


def action_of(text):
    """The agent's actual command at this step, or "" when there is none.

    Every assistant step in this corpus is a four-block document --
    ``<memory>`` / ``<reflection>`` / ``<plan>`` / ``<action>`` -- and the
    tag is present on 3000/3000 assistant steps. Only the last block is the
    thing the simulator executed.
    """
    m = ACTION_RE.search(str(text or ""))
    if m is None:
        return ""
    return " ".join(m.group(1).split())[:ACTION_CLIP]


def lead_with_action(digests, traj=None):
    """Hoist each step's own action to the FRONT of its digest.

    WHY. Measured over all 3000 assistant steps of the corpus
    (``probe/alf_errors.py``):

      * ``<action>`` -- the only text the simulator actually ran -- is
        **1.0%** of the assistant text, and its median start position is
        **98%** of the way through the step.
      * ``<memory>`` / ``<reflection>`` / ``<plan>`` are 54.6% between them,
        and they are retrospective prose ABOUT OTHER STEPS.
      * **86%** of assistant steps cite a different step by number
        ("Step 3:", "Observation 8"), median 2 citations per step.

    So a localiser reading step 11 sees mostly sentences about steps 3 and 8,
    and the 1% that identifies step 11 is a tail. On the 0.3158 baseline, 5 of
    the 26 dev misses land on the IMMEDIATELY ADJACENT action, and for those
    rows the predicted and the reference step open with byte-identical text.
    ``acc@2`` is 0.4474 against ``acc@0`` 0.3158, i.e. a quarter of the gap is
    adjacency, not wrong-region error.

    This is NOT a budget problem and raising the budget cannot fix it: at the
    baseline's ``--verbatim-limit 2500``, ``clip_middle`` keeps the action for
    798 of the 799 clipped steps that have one. The action is DILUTED, not
    truncated, so the presentation is what has to change.

    WHAT THIS IS NOT. It adds no information the digest did not already carry
    -- the action text is copied out of the step's own content -- and it reads
    no metadata, no annotation and no gold. It only moves text that was
    already in the prompt to the position the prompt is read from first.
    Steps whose action is already the first thing in the digest are left
    alone, so the transform is idempotent.

    Returns how many digests were changed.
    """
    steps = {}
    if traj is not None:
        steps = {s.index: (s.content or "") for s in traj.steps}
    n = 0
    for d in digests:
        # ASSISTANT STEPS ONLY, and this is not a stylistic choice. Each user
        # turn replays the whole history and then closes with the simulator's
        # fixed instruction block, which contains the words "choose ONE
        # admissible action" -- so ``action_of`` matches on a user turn and
        # returns the INSTRUCTION, not an action anyone took. Leading a user
        # digest with that would be inventing a fact about a step that, by
        # NON_ATTRIBUTABLE_ROLES, can never be the answer anyway.
        if d.role in NON_ATTRIBUTABLE_ROLES:
            continue
        src = steps.get(d.index) or d.text or ""
        act = action_of(src)
        if not act:
            continue
        body = d.text or ""
        if body.startswith(ACTION_LEAD):
            continue
        d.text = ACTION_LEAD + act + "\n" + body
        n += 1
    if n:
        logger.info("alfworld: led %d digest(s) with their own action", n)
    return n
