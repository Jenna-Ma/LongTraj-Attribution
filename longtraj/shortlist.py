"""Stage 5b-2: a reader that has read the whole compressed run names a few MORE
candidate steps, which are merged into the answer space the localiser receives.

``candidates.py`` builds the answer space (C1) by pattern-matching four families
of WRITE: ``str_replace_editor`` create/str_replace/insert, ``^Action:`` shell
writes (``sed -i``, redirection, ``patch``, ``rm``), ``_PY_WRITE`` and
``_FUNC_WRITE``. It has no understanding of the trajectory at all, and that is
measurable rather than arguable (``probe/cand_audit.py``, zero calls, SP, n=86):

  * C1 contains the reference step on 66 rows and misses it on 20 -- a ceiling
    of 0.7674, and under ``--candidates-strict`` the 20 are unanswerable by
    construction (acc@0 on them is 0.0000 against 0.4545 on the 66).
  * On FIFTEEN of those 20 rows the reference step is a pure prose
    planning/reasoning step. ``schema.Step.attributable`` only excludes
    non-assistant roles, so a prose step IS a legal answer -- the regexes simply
    cannot see one, ever, no matter how the run went.
  * Snapping cannot recover them: the distance from the reference step to the
    nearest C1 entry on the misses is median 5, p25 4, p75 18, max 48, with only
    3 rows at <= 2. The step has to actually enter the set.

So: ONE call per row, shown the same compressed body the localiser is about to
read, shown C1 and told not to repeat it, and asked for at most ``SHORTLIST_K``
steps C1 could not contain. The result (C2) is merged into C3 = C1 U C2.

Four design points, each with a reason that can be checked rather than argued:

**A separate pipeline stage, not a second turn inside ``localize()``.** The LLM
cache is keyed on the prompt. A two-turn localiser would make the second turn
depend on the first turn's output, and the repo's one cheap proof of
non-drift -- re-run an arm with the flag off and get ``calls: 0`` plus a
byte-identical output file -- would be gone. As a stage, every level caches
independently: arm A costs ~170 new calls and arm B ~350.

**ONE list, with no provenance marked.** The localiser is shown C3 as a single
flat list in trajectory order, rendered by ``candidates.render`` exactly like C1
ever was; nothing in the prompt says which entries the regexes found and which
this stage proposed. The first version of this arm did the opposite -- a second
``SHORTLIST_BLOCK`` headed "ALSO ELIGIBLE -- proposed by a reader who read the
whole run" -- and it was wrong twice over. It listed the proposed steps a second
time (they were already in the merged list the first block renders), and it
marked them as new, which is a thumb on the scale: the localiser answered with a
proposed step on 68 of the 78 rows that had one, while those answers were the
reference step 3/68 of the time against 21/68 for the parsed steps they
displaced.

The frozen ``CANDIDATE_BLOCK`` is still not touched, because under C3 its prose
is false -- it says in so many words that steps which "only read, searched, ran
a test, or reasoned are not listed", and a reasoned step is exactly what this
stage adds, and it then forbids naming one. ``localize.CANDIDATE_BLOCK_MIXED``
is a separate text for the mixed space, describing the list it actually gets.

**Abstention is free and leaves the run byte-identical.** Unparseable JSON, an
empty list, a missing field, every step already in C1 -- all land on
``steps=[]``, so C3 == C1, ``cands_mixed`` is never passed, and every downstream
prompt is byte-for-byte what it would have been without this stage. That also
gives a free per-row attribution floor: on rows where nothing new was proposed,
this stage provably did not move the answer.

**It is shown the symptom.** The opposite of ``expect.py``, which reads blind
because a reader that knows the symptom writes the requirement the symptom
violates. Here the task IS to reason backwards from the symptom, and
``span_prior`` and ``verify_candidates`` are both shown it too.

What this stage does NOT claim: enlarging the answer space has been measured
twice in this repo and lost both times -- ``sp_cnocand`` (block removed, -5) and
``sp_union`` (verify pool = C1 U flow, -6) each lost the SAME 7 in-set rows. This
stage widens the space too, and so inherits that risk; what it does differently
is that the entries it adds are NAMED by a reader rather than admitted wholesale
by a rule, and that the ceiling it lifts is a measured 20 rows rather than a
hoped-for one. The poll is not held back from them -- that was tried, once, as a
way of buying the hint without the wider vote, and it only made answering with a
proposed step self-defeating.

Like every other stage, the ground truth is unreachable from here: this module
sees ``Trajectory``, ``Span`` and ``FlowGraph``, none of which can hold a label.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .compress import render_compressed
from .llm import LLM
from .schema import FlowGraph, Span, Trajectory, clip_middle

logger = logging.getLogger(__name__)

# How many NEW steps one call may add. Fixed here and named in the
# pre-registration BEFORE the arm runs; re-tuning is a new arm swept on the dev
# half, not a re-read of this one.
#
# Why 5: |C1| is median 9 on SP, and acc@0 falls as the set grows (2-5 steps:
# 0.7143; 6-8: 0.4667; 9-12: 0.2308), so the names are expensive and have to be
# spent on the increment. K=8 pushes 3 rows past VERIFY_MAX=32 and lands the set
# at the size of the already-lost ``sp_union`` pool; K=3 cannot cover the 15
# prose-step misses this stage exists for.
SHORTLIST_K = 5

# Five entries and a sentence each. Same reasoning as span_judge.JUDGE_BUDGET:
# hitting the ceiling arrives as unparseable JSON, i.e. as a silent abstention,
# which is the behaviour this stage wants anyway.
SHORTLIST_BUDGET = 700

SHORTLIST_SYSTEM = (
    "You read the whole run of a failed coding agent and name a FEW steps that "
    "a pattern matcher for file writes could not have found. You are not "
    "choosing the answer. Answer strictly in JSON."
)

SHORTLIST_PROMPT = """A coding agent FAILED this task. Another reader is about to
be asked which SINGLE step caused the failure, and it will be made to answer from
a fixed list of steps. That list was built mechanically, by pattern-matching for
steps that WROTE something — applied a patch, edited a file, ran a command with
side effects.

Your job is to add to that list, not to choose from it.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

THE FAILURE BECAME VISIBLE AT: {anchors}
That is where the causal chain ENDS, and it is almost never where it starts.

============================================================
ALREADY ON THE LIST — do NOT name any of these
============================================================
{cands}
============================================================

============================================================
THE RUN
============================================================
{body}
============================================================

WHAT TO DO
Name at most {k} steps that are NOT in the list above and that could be the step
that caused the failure. Fewer is better than more. An empty list is a perfectly
good answer and costs you nothing.

1. **A step that only decided, planned, or reasoned is a legal answer, and it is
   the main thing you are here for.** The list above cannot contain one: it was
   built by matching writes, so a step whose entire act was to choose a wrong
   approach, misread an error, or commit to a wrong diagnosis is invisible to it.
   If the run went wrong at such a step, say so — that is the one thing the other
   reader cannot learn from its list.
2. Exploration is not an error. The error begins where the agent COMMITS to a
   direction — acts on it, or states it as the only way forward.
3. Apply the but-for test: if that one step had been done correctly, would the
   symptom above still have happened? If yes, it is not the step.
4. Do not name a step just because it is near the failure, touches the files the
   failure is about, or looks busy. The agent had to touch them to attempt the
   task at all.
5. Do not re-name a step from the list above, and do not name a step that is
   merely adjacent to one. If the decisive WRITE is already listed, the useful
   thing you can add is the earlier step that decided on it — not the write.
6. Only steps spoken by the agent itself can be named. Tool output and user
   turns are not steps you may name.

Return ONLY JSON:
{{
  "steps": [
    {{"step": <int>, "why": "<1 sentence: what this step decided or changed, and how it leads to the symptom>"}}
  ]
}}"""


@dataclass
class Shortlist:
    """One reader's proposed additions to the answer space.

    ``steps`` is already filtered, deduplicated, sorted and truncated to K, so a
    caller cannot accidentally widen the answer space past what was
    pre-registered. ``raw`` keeps what the model actually said, including the
    entries that were dropped, because "it named three steps and two were
    already in C1" and "it named one step" are different results and only the
    record can tell them apart.
    """

    steps: List[int] = field(default_factory=list)
    why: Dict[int, str] = field(default_factory=dict)
    called: bool = False
    skipped: str = ""                    # why no call was made, if none was
    n_proposed: int = 0                  # entries the model returned
    n_dropped_known: int = 0             # already in C1
    n_dropped_bad: int = 0               # out of range / not attributable
    n_truncated: int = 0                 # lost to the K cap
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        """No usable new step. C3 == C1 and every downstream prompt is unchanged."""
        return not self.steps

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": list(self.steps),
                "why": {str(k): v for k, v in self.why.items()},
                "called": self.called, "skipped": self.skipped,
                "n_proposed": self.n_proposed,
                "n_dropped_known": self.n_dropped_known,
                "n_dropped_bad": self.n_dropped_bad,
                "n_truncated": self.n_truncated}


def _parse(traj: Trajectory, obj: Any, known: Sequence[int],
           k: int) -> Shortlist:
    """Read the reader's JSON. Anything malformed becomes an abstention.

    An abstention is not a failure. ``steps=[]`` makes C3 == C1, which makes
    every prompt after this stage byte-identical to a run in which the stage
    never existed -- so a reader that cannot answer costs one call and changes
    nothing else.
    """
    sl = Shortlist(called=True, raw=(obj if isinstance(obj, dict) else {}))
    rows = obj.get("steps") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return sl

    seen = set(int(s) for s in known)
    picked: List[Tuple[int, str]] = []
    for row in rows:
        if isinstance(row, dict):
            raw_step, why = row.get("step"), row.get("why")
        elif isinstance(row, int):
            raw_step, why = row, ""
        else:
            continue
        sl.n_proposed += 1
        try:
            i = int(raw_step)
        except (TypeError, ValueError):
            sl.n_dropped_bad += 1
            continue
        if i in seen:
            # Either already in C1 or named twice. Both are "nothing new".
            sl.n_dropped_known += 1
            continue
        if not (0 <= i < len(traj.steps)) or not traj.steps[i].attributable:
            # A step number the run does not have, or a tool/user turn that
            # could never be the answer. Dropping beats passing it on: the
            # localiser would be offered an answer the metric cannot score.
            sl.n_dropped_bad += 1
            continue
        seen.add(i)
        picked.append((i, str(why or "")[:300]))

    # Truncate in the order the model gave them -- its own ranking is the only
    # ranking there is -- then sort for the prompt, so the block reads in
    # trajectory order like the one above it.
    if len(picked) > k:
        sl.n_truncated = len(picked) - k
        picked = picked[:k]
    picked.sort(key=lambda p: p[0])
    sl.steps = [i for i, _ in picked]
    sl.why = {i: w for i, w in picked}
    return sl


def propose(traj: Trajectory, spans: List[Span], graph: FlowGraph, llm: LLM,
            digests: Optional[Sequence[Any]] = None,
            cands: Optional[List[Dict[str, object]]] = None,
            k: int = SHORTLIST_K,
            max_prompt_chars: int = 120000) -> Shortlist:
    """One call: steps the write-matching parse could not have found.

    The body is built with ``compress.render_compressed`` and trimmed with
    ``localize._fit_budget``, i.e. the same two functions in the same order the
    localiser uses, so this reader sees the run in exactly the words the
    localiser will. Rendering it a second way would mean a step it nominates
    might not be legible to the reader it nominates it to.
    """
    if not cands:
        # No answer space means no answer space to add to: with C1 empty the
        # localiser is answering freely already, and ``merge`` would turn an
        # unrestricted run into one restricted to whatever this call said.
        return Shortlist(skipped="no candidates")

    # Imported here, not at module scope: ``localize`` is a large module and
    # this is the only thing needed from it. ``localize`` does not import this
    # one, so there is no cycle either way.
    from .localize import _fit_budget

    body = render_compressed(traj, spans, digests or [])
    if len(body) > max_prompt_chars:
        body = _fit_budget(traj, spans, digests or [], max_prompt_chars)

    known = [int(c["step"]) for c in cands]
    listed = "\n".join("  step %d — %s" % (c["step"], c["label"] or "(edit)")
                       for c in cands)
    prompt = SHORTLIST_PROMPT.format(
        task=clip_middle(traj.task_description, 2000),
        failure_summary=(graph.failure_summary or "(not summarised)"),
        anchors=(", ".join("step %d" % a for a in sorted(graph.anchors))
                 or "(none identified)"),
        cands=listed, body=body, k=k)
    obj = llm.generate_obj(prompt, max_tokens=SHORTLIST_BUDGET,
                           system=SHORTLIST_SYSTEM)
    return _parse(traj, obj, known, k)


LABEL_CHARS = 70                # == candidates.classify's own label width


def _clip_label(why: Optional[str]) -> str:
    """A proposed step's entry, cut to the width C1's entries already have.

    First clause or first sentence, whichever comes first, then a hard cut. No
    ellipsis: C1's labels do not have one, and a trailing "..." would be exactly
    the provenance mark this is removing.
    """
    s = " ".join((why or "").split())
    if not s:
        return "no write recorded"
    for sep in (". ", "; ", ", which ", " and then "):
        if sep in s[:LABEL_CHARS + 20]:
            s = s.split(sep, 1)[0]
            break
    return s[:LABEL_CHARS].rstrip(" ,.;:-")


def merge(c1: List[Dict[str, object]],
          shortlist: Optional[Shortlist]) -> Tuple[List[Dict[str, object]],
                                                   List[int]]:
    """C3 = C1 U C2, in trajectory order. Returns ``(C3, C2-only step numbers)``.

    Only ever grows: every entry of C1 survives unchanged, including its label,
    so an arm with this stage on is C1's arm plus some entries and never C1's
    arm minus one. On abstention the returned list is C1 itself, which is what
    keeps the downstream prompts byte-identical.
    """
    c1 = list(c1 or [])
    if shortlist is None or not shortlist.steps:
        return c1, []
    known = {int(c["step"]) for c in c1}
    new = [i for i in shortlist.steps if i not in known]
    if not new:
        return c1, []
    # The label is clipped to C1's own width. C1's labels are short mechanical
    # strings ("str_replace /x/y.py", 70 chars by construction in
    # ``candidates.classify``); the reader's reasons are sentences. Dropped into
    # one list that marks no provenance, the LENGTH would mark it instead --
    # every long entry would read as "this is one of the new ones", which is the
    # thumb on the scale this change exists to remove.
    added = [{"step": i, "label": _clip_label(shortlist.why.get(i))}
             for i in new]
    c3 = sorted(c1 + added, key=lambda c: int(c["step"]))
    return c3, new


def stats(shortlist: Optional[Shortlist],
          c1: Optional[Sequence[Dict[str, object]]] = None,
          c3: Optional[Sequence[Dict[str, object]]] = None,
          new_steps: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Diagnostics for the arm's record. Read by no prompt.

    ``n_new`` is the number to watch first, together with the abstention count:
    this stage's whole claim is that it adds steps the regexes could not see, and
    a row with ``n_new == 0`` is a row on which it provably changed nothing --
    the prompt is byte-identical and the answer came back from the cache. If
    ``n_dropped_known`` dominates, the reader is re-deriving C1 at full price
    instead of extending it, which is a result and only readable from here.
    """
    if shortlist is None:
        return {}
    out = shortlist.to_dict()
    out["n_c1"] = len(c1 or [])
    out["n_c3"] = len(c3 or [])
    out["n_new"] = len(new_steps or [])
    out["abstained"] = shortlist.abstained
    return out
