"""Stage 5c variant: widen the phase guess WITHOUT touching its strongest pick.

WHY THIS EXISTS, measured on SWE-Bench-Pro (86 cases, the
``--span-prior --prior-signals --candidates-strict`` arm):

    prior recall 0.535   top-guess recall 0.279   coverage 0.185
    acc@0 given the prior HIT   0.5217
    acc@0 given the prior MISSED 0.0500

That 0.52/0.05 split is the largest conditional gap anywhere in the metrics.
Where the prior lands on the right region the method works; where it misses,
the localiser is at chance and the case is lost before stage 6 runs. Recall is
therefore the ceiling on everything downstream, and it is 0.535 while the guess
nominates only 18.5% of the run. The question this module asks is narrow: is
the remaining 46.5% reachable by nominating MORE region, or is the prior's
miss a judgement failure that more coverage cannot fix?

WHAT LIMITS COVERAGE TODAY is not the prompt and not the 0.5 ``max_coverage``
cap — at 0.185 that cap never binds. It is the budget arithmetic in
``predict_spans``: ``min(max_guesses, max(1, len(spans) // 3))``, so a run
segmented into 4-5 phases is allowed exactly ONE guess. Widening is thus a
question about the budget, not about how the phases are described.

THE DESIGN, and the one property that makes the A/B mean anything:

  * **Tier 1 is the incumbent, called verbatim.** ``predict_spans`` runs with
    the same flags and the same prompt, so its call is a cache HIT and its
    guesses are bit-identical to the control arm's. The enhanced prior is a
    strict superset of the baseline prior, and the first guess — the one
    ``prior_top_recall`` scores — is unchanged by construction.
  * **Tier 2 is one extra call**, and it is shown tier 1's picks and told to
    name DIFFERENT phases. It answers a genuinely different question ("what
    else could it be"), which is the only reason to spend a call rather than
    re-sample tier 1 at a higher budget.
  * **Tier 3 costs nothing**: phases adjacent to a nominated one, added
    nearest-first. Segmentation boundaries are not causal boundaries, and a
    cause landing one phase before its commit is the failure mode adjacency
    covers.

Everything after tier 1 is labelled POSSIBLE, never LIKELY. The ordering
contract in ``SpanPrior`` is that the first guess is the strongest, and tier 1
keeps that slot.

THE HONEST PRIOR EXPECTATION IS THAT THIS LOSES. Measured this session, the
``--prior-unranked --candidates`` arm scored 0.267 against the 0.302 control
while ``prior_followed`` ROSE 0.709 -> 0.861. ``span_prior.py``'s own docstring
records why: "the localiser already treats this guess as a filter, not a hint".
A filter that admits 30% of the run instead of 18.5% earns recall arithmetically
and may hand the localiser a larger haystack in exchange. So the reading below
is pre-committed: recall and coverage rising is NOT the result. acc@0 is.

The ground truth never enters here — this module sees ``Trajectory``, ``Span``
and ``FlowGraph``, none of which can hold a label, and its only extra prompt is
built from phase summaries the outline stage already wrote.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from . import span_prior as span_prior_mod
from .llm import LLM
from .schema import (LIKELY, POSSIBLE, FlowGraph, Span, SpanGuess, SpanPrior,
                     Trajectory, clip_middle)

logger = logging.getLogger(__name__)

SECONDARY_SYSTEM = (
    "You widen a shortlist of suspect phases in an agent trajectory. "
    "You name phases, never individual steps. Answer strictly in JSON."
)

SECONDARY_PROMPT = """A coding agent FAILED this task. Another reader has
already named the phase(s) it finds most suspicious. Your job is DIFFERENT:
name phases that reader did NOT name, which could still contain the cause.

You are NOT choosing the step, and you are NOT re-ranking the shortlist below.
A later reader will look at the first reader's phases regardless. You are
buying it a second place to look if the first one is wrong.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

{symptom_block}THE FAILURE BECAME VISIBLE AT: {anchors}
That is where the causal chain ENDS, not where it started.

ALREADY NOMINATED (do not repeat these):
{already}

{signals_note}ALL PHASES ({n} phases over {steps} steps):
{phases}

HOW TO CHOOSE, given someone else already took the obvious ones:
1. Assume the shortlist above is WRONG. If the cause is not in those phases,
   where is it? That question, not "what else looks bad", is your job.
2. Prefer phases EARLIER than the shortlist. A cause that precedes the
   nominated commit is the common way the first reader misses.
3. A phase whose summary reads uneventful is not cleared. Summaries are thin,
   and a wrong decision taken quietly leaves no trace in one.
4. Exploration is not an error. Look for where a direction was COMMITTED to.
5. Do not pad. If nothing outside the shortlist is plausible, return an empty
   list — that is a useful answer and costs the next reader nothing. Naming a
   phase you cannot justify makes the hint point at most of the run, which
   tells the next reader nothing at all.

Name AT MOST {budget} phase(s), most plausible first. Use the exact step ranges
given above. Return ONLY JSON:
{{
  "spans": [
    {{"start": <int>, "end": <int>,
      "why": "<one sentence: what in this phase could have caused the failure>"}}
  ]
}}"""


def enhanced_predict_spans(traj: Trajectory, spans: List[Span],
                           graph: FlowGraph, llm: LLM,
                           signals: bool = False,
                           symptom: bool = False,
                           cands: Optional[List[Dict[str, Any]]] = None,
                           coverage_target: float = 0.30,
                           max_coverage: float = 0.5,
                           max_guesses: int = 3) -> SpanPrior:
    """``predict_spans`` plus a second tier, up to ``coverage_target``.

    Tier 1 is ``predict_spans`` called with the caller's own flags, so on a
    repeat run its call is served from cache and its guesses match the control
    arm exactly. Tiers 2 and 3 only ever APPEND, and only while coverage is
    below target; a prior that is already wide enough is returned untouched and
    the extra call is never made.

    ``max_coverage`` is the hard ceiling from ``predict_spans`` and is enforced
    again here: a guess nominating more than half the run cannot be read as
    narrowing anything, whatever tier produced it.
    """
    base = span_prior_mod.predict_spans(
        traj, spans, graph, llm, max_guesses=max_guesses,
        max_coverage=max_coverage, signals=signals, cands=cands,
        symptom=symptom)

    n_steps = len(traj)
    if not base.guesses or not n_steps:
        # Nothing to widen. An empty prior stays empty: tier 1 declining to
        # guess means the phases were too few or the answer unparseable, and
        # neither is repaired by asking a second time.
        return base

    target = int(n_steps * coverage_target)
    ceiling = int(n_steps * max_coverage)
    covered = _covered_steps(base.guesses)
    if len(covered) >= target:
        logger.info("%s: prior already at %.3f coverage, tier 2 skipped",
                    traj.task_id, len(covered) / n_steps)
        return base

    guesses = list(base.guesses)
    taken = {(g.start, g.end) for g in guesses}

    # Tier 2: one call, shown tier 1's picks, asked for different phases.
    budget = max(1, min(max_guesses, len(spans) // 3))
    for sp in _ask_secondary(traj, spans, graph, llm, guesses, budget,
                             signals=signals, symptom=symptom, cands=cands):
        if (sp.start, sp.end) in taken:
            continue
        steps = set(range(sp.start, sp.end + 1))
        if len(covered | steps) > ceiling:
            continue
        taken.add((sp.start, sp.end))
        covered |= steps
        guesses.append(SpanGuess(start=sp.start, end=sp.end,
                                 likelihood=POSSIBLE, why=sp.why))
        if len(covered) >= target:
            break

    # Tier 3: free. Phases touching an already-nominated one, nearest first.
    if len(covered) < target:
        for sp in _adjacent_spans(spans, taken):
            steps = set(range(sp.start, sp.end + 1))
            if len(covered | steps) > ceiling:
                continue
            taken.add((sp.start, sp.end))
            covered |= steps
            guesses.append(SpanGuess(
                start=sp.start, end=sp.end, likelihood=POSSIBLE,
                why="adjacent to a nominated phase; segmentation boundaries "
                    "are not causal boundaries"))
            if len(covered) >= target:
                break

    logger.info("%s: prior widened %d -> %d guesses, coverage %.3f -> %.3f",
                traj.task_id, len(base.guesses), len(guesses),
                _covered_n(base.guesses) / n_steps, len(covered) / n_steps)
    return SpanPrior(guesses=guesses)


def _covered_steps(guesses: List[SpanGuess]) -> Set[int]:
    out: Set[int] = set()
    for g in guesses:
        out.update(range(g.start, g.end + 1))
    return out


def _covered_n(guesses: List[SpanGuess]) -> int:
    return len(_covered_steps(guesses))


def _ask_secondary(traj: Trajectory, spans: List[Span], graph: FlowGraph,
                   llm: LLM, already: List[SpanGuess], budget: int,
                   signals: bool, symptom: bool,
                   cands: Optional[List[Dict[str, Any]]]) -> List[SpanGuess]:
    """The one extra call. Returns snapped guesses, or [] on any failure.

    Built from the SAME phase lines tier 1 read, including the signal brackets
    when they are on: a second reader given a thinner view than the first is
    not a second opinion, it is a worse one.
    """
    rows = (span_prior_mod._signal_rows(traj, spans, graph, cands) if signals
            else [""] * len(spans))
    phases = []
    for sp, sig in zip(spans, rows):
        phases.append("steps %d-%d — %s%s\n    %s"
                      % (sp.start, sp.end, sp.label or "(unlabelled)", sig,
                         (sp.summary or "(no summary)").strip()))

    prompt = SECONDARY_PROMPT.format(
        task=clip_middle(traj.task_description, 2000),
        failure_summary=(graph.failure_summary or "(not summarised)"),
        anchors=(", ".join("step %d" % a for a in sorted(graph.anchors))
                 or "(none identified)"),
        already="\n".join("  steps %d-%d — %s" % (g.start, g.end,
                                                  g.why.strip() or "(no reason)")
                          for g in already),
        n=len(spans), steps=len(traj), budget=budget,
        symptom_block=(span_prior_mod._symptom_block(traj) if symptom else ""),
        signals_note=(span_prior_mod.SIGNALS_NOTE.format(
            changes=(span_prior_mod.CHANGES_BULLET if cands is not None else ""))
            if signals else ""),
        phases=clip_middle("\n".join(phases), 40000))

    try:
        obj = llm.generate_obj(prompt, max_tokens=1000, system=SECONDARY_SYSTEM)
    except Exception as exc:                       # noqa: BLE001
        logger.warning("%s: secondary prior call failed: %s", traj.task_id, exc)
        return []

    raw = obj.get("spans") if isinstance(obj, dict) else None
    out: List[SpanGuess] = []
    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        try:
            lo, hi = int(item["start"]), int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if hi < lo:
            lo, hi = hi, lo
        # Snap onto a real phase, exactly as tier 1 does: a guess must always
        # denote a phase that exists or the coverage diagnostic stops meaning
        # anything.
        sp = span_prior_mod._best_overlap(lo, hi, spans)
        if sp is None:
            continue
        out.append(SpanGuess(start=sp.start, end=sp.end, likelihood=POSSIBLE,
                             why=str(item.get("why") or "")[:300]))
        if len(out) >= budget:
            break
    return out


def _adjacent_spans(spans: List[Span],
                    taken: Set[tuple]) -> List[Span]:
    """Un-nominated phases touching a nominated one, nearest-first.

    Distance is measured in phase positions, so the phase immediately before a
    nominated one comes before the phase immediately after any other. Earlier
    wins ties: a cause sits upstream of the commit it caused, not downstream.
    """
    idx = {(sp.start, sp.end): i for i, sp in enumerate(spans)}
    hot = sorted(idx[k] for k in taken if k in idx)
    if not hot:
        return []
    scored = []
    for i, sp in enumerate(spans):
        if (sp.start, sp.end) in taken:
            continue
        d = min(abs(i - h) for h in hot)
        before = 0 if any(i < h for h in hot) else 1
        scored.append((d, before, i, sp))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [sp for _, _, _, sp in scored]
