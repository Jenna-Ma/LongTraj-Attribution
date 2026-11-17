# -*- coding: utf-8 -*-
"""ALFWorld answer space. Additive: unreachable unless ``Config.alf_space`` is
set, and ``candidates.py`` is not touched.

WHY A SEPARATE MODULE. ``candidates.py`` narrows the answer space to the steps
that MODIFIED something, parsed from coding idioms (``str_replace_editor({...})``,
``sed -i``, ``>`` redirection, ``git apply``, ``write_patch``). On SWE-Bench-Pro
that pool holds the reference step 78.57% of the time in 9 of ~57 slots, and
``acc@0_given_candidate_miss`` is 0.0000 -- i.e. the method is only ever right
when that pool caught the answer. In ALFWorld nothing is ever written, so the
parse correctly degrades to ``[]`` and the localiser runs unrestricted in all 30
slots. Reproducing SP's single largest lever therefore needs a different parse.

THE OBVIOUS PORT DOES NOT WORK, AND WAS KILLED WITH ZERO CALLS. The natural
analogue -- "actions that changed the world state": take / put / open / close /
heat / cool / clean / toggle, excluding go-to / look / inventory / examine --
carries no signal at all. Measured on dev 38 (``probe/prereg_alf_040.md`` §1.1),
P(gold | first word of the action) against a base rate of 1/30 = 0.0333:

    go 0.0318   examine 0.0235   open 0.0244   look 0.0345
    take 0.0370   move 0.0769    clean 0.0667  <blank> 0.0694

``go`` alone is 41.4% of the steps and 39.5% of the labels. The labels track
action frequency almost exactly, so a state-change pool would have
recall == coverage, a lift of 1.0, and would not be a restriction.

WHAT DOES CARRY SIGNAL IS POSITION, and one content feature.

  * ``early`` -- the first ``window`` steps. ALFWorld labels sit early: the dev
    median is step 13 of 60 (relative 0.217, against 0.619 on SP). Steps 1..21
    are 11 of the 30 attributable slots and hold 73.68% of the dev labels, a
    lift of 2.01x.
  * ``blank`` -- an assistant turn that committed to no action at all (no
    ``<action>`` block, or an empty one). It fires on 6.32% of steps and holds
    13.16% of the labels (lift 2.08x), the only content feature on this corpus
    with a lift above 1. Pool size is a median of ZERO, so it widens nothing on
    most records; it exists to reach the handful of late no-action labels the
    window would otherwise kill.

  Everything else was measured and rejected: ``repeat`` (the action already
  occurred) 0.69x, ``selffail`` (the reflection block admits the last action
  failed) 0.68x, ``inadm`` (the action is absent from the admissible list)
  0.85x, ``after_nothing`` 0.72x, ``after_nh`` (the previous novel observation
  said ``Nothing happens.``) 1.10x.

  The ``repeat`` reading of 0.69x MATTERS BEYOND THIS FILE: the label is LESS
  likely than chance to sit on a repeated action, which falsifies the sentence
  "inefficient plan: it revisits places it already searched. THIS IS THE MOST
  COMMON ANSWER IN THIS CORPUS" written into ``alfworld_prompt.ALF_RUBRIC_NOTE``
  last round. That sentence is why the rubric arm over-predicted ``plan`` 21
  times against a gold count of 12.

THIS IS FITTING AN ANNOTATION CONVENTION, STATED PLAINLY. The window was
measured FROM THE DEV LABEL DISTRIBUTION, not derived from the task semantics.
``candidates.py`` makes the same admission about its own write-based pool. The
defensible reading: the localiser is already asked for the EARLIEST step whose
mistake caused the failure, and "earliest" produces a front-loaded answer by
itself, so a positional prior formalises a criterion the prompt already claims.
Where that reading is wrong -- a run whose first real mistake is at step 51, of
which dev has four -- the restriction kills the case outright. 0.7632 is a
CEILING on the method, to be reported next to any score, not a bug to fix.

The window is frozen at 21 in ``probe/prereg_alf_040.md`` §4.6 and must not be
retuned against a dev reading.

NO LABEL ENTERS HERE. This module reads ``traj.steps[].role``, ``.index`` and
``.content`` and nothing else: no ``metadata``, no file handles, no ``json``.
``tests/test_no_leak.py`` §57 pins that mechanically.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# The agent's chosen action for the turn. The ALFWorld harness asks for it in
# its own block, and every record in the corpus uses the tag.
_ACTION = re.compile(r"<action>(.*?)</action>", re.S)

# Frozen default, see module docstring and prereg §4.6.
DEFAULT_WINDOW = 21

SPACES = ("early", "early+blank", "blank")


def action_of(text: str) -> Optional[str]:
    """The action an assistant turn committed to, or None when the tag is absent.

    None and ``""`` are different readings and callers must not conflate them:
    None means the turn never emitted an ``<action>`` block, ``""`` means it
    emitted an empty one. MEASURED on this corpus: 183 of 3000 assistant turns
    are None and ZERO are ``""``. So "the turn committed to no action" is the
    None case, and an earlier draft of this file that tested for ``""`` found an
    empty pool -- see ``blank_steps``.
    """
    m = _ACTION.search(text or "")
    if m is None:
        return None
    return m.group(1).strip().strip("'\" \n\t")


def _attributable(traj) -> List[int]:
    return [s.index for s in traj.steps if s.role == "assistant"]


def blank_steps(traj) -> List[int]:
    """Assistant steps that committed to no action.

    That is ``action_of() in (None, "")``: either no ``<action>`` block at all
    (183 of 3000 turns) or an empty one (0 of 3000). The dev reading in
    probe/prereg_alf_040.md §1.2 -- 6.32% of steps, 13.16% of labels, lift
    2.08x -- was taken with a regex that collapsed both cases, and the None
    case is the one that carries it. Testing only for ``""`` yields an empty
    pool, which is how the first draft of this module was caught.
    """
    out = []
    for s in traj.steps:
        if s.role != "assistant":
            continue
        act = action_of(s.content)
        if act is None or act == "":
            out.append(s.index)
    return out


def early_steps(traj, window: int = DEFAULT_WINDOW) -> List[int]:
    """The attributable steps at or before ``window``, in trajectory order."""
    return [i for i in _attributable(traj) if i <= window]


def answer_space(traj, space: str,
                 window: int = DEFAULT_WINDOW) -> List[Dict[str, object]]:
    """An answer space in ``candidates.candidates()``'s shape.

    Returns ``[{"step": int, "label": str}]`` so that every downstream knob --
    ``candidate_strict``, ``verify``, ``protect_candidates``, ``snap`` --
    consumes it with no change. An empty list means "do not restrict", exactly
    as in ``candidates.py``; it is never "no valid answers".

    ``space`` is validated rather than defaulted: a mistyped flag that quietly
    ran unrestricted would be recorded in the artifact as the restricted arm
    having run, which is the one failure mode that corrupts an arm comparison
    after the fact.
    """
    if space not in SPACES:
        raise ValueError("unknown alf_space %r; known: %s"
                         % (space, ", ".join(SPACES)))
    blank = set(blank_steps(traj))
    if space == "blank":
        keep = blank
    else:
        keep = set(early_steps(traj, window))
        if space == "early+blank":
            keep |= blank
    out: List[Dict[str, object]] = []
    for i in sorted(keep):
        out.append({"step": i, "label": _label(traj, i, window, blank)})
    return out


def _label(traj, index: int, window: int, blank) -> str:
    """One line of provenance per offered step.

    The label is what the localiser is shown next to the step number, so it
    must say WHY the step is on the list and nothing about whether it is the
    answer. ``candidates.render`` prints it verbatim.
    """
    act = None
    for s in traj.steps:
        if s.index == index:
            act = action_of(s.content)
            break
    if act is None or act == "":
        return "no action emitted"
    return act[:70]
