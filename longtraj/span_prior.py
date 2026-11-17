"""Stage 5b: a second reader's guess at WHICH PHASES contain the critical step.

Everything tried before this attacked the evidence: top-up painted more of the
trajectory (flow_recall 0.70 -> 0.80), edge annotation exposed each edge's
position in the graph. Neither moved acc@0, and ``P(hit | gold on flow)`` sat
at 0.50 throughout — the reference step was reaching the localiser and the
localiser still passed over it half the time. The bottleneck is not how much
evidence arrives, it is which REGION of a 117-step run gets read closely.

So this stage narrows the region. One extra call reads only the phase summaries
and the visible symptom, and names at most three phases where the cause
plausibly sits. The guess is handed to the localiser as a hint it is explicitly
allowed to overrule.

Two deliberate restrictions:

  * **It only picks regions, never a step.** Choosing the step is the
    localiser's job, over evidence this stage never sees.
  * **Blind to the flow by default**, and never shown the edge list at all. The
    original argument was that the flow is not the missing information -- 0.80
    flow_recall against 0.44 acc@0 -- so a prior that reads it returns an echo
    of the localiser's own view at the cost of a call. ``prior_signals`` softens
    that to the aggregate: per-phase COUNTS of on-flow steps, distance to the
    symptom, and how many steps in the phase changed anything. The edges, the
    entities and the WEAK/STRONG tiers stay out either way, so the stage remains
    a summary-level reader and cannot simply restate the graph.

``prior_symptom`` is the third, and it loosens a different thing: the symptom.
By default this stage reads ``failure_summary``, two sentences written by the
anchoring model, so a phase-level judgement inherits whatever that model
dropped. The flag adds the harness's own failing test and traceback where the
corpus ships them (TrajAudit's ``failed_tests``; TrajDebug has no such field and
the block renders empty). That is harness output, not annotation — the label is
``failure_id`` and the reference fix is ``gold_patch``, and the loader reads
neither into ``meta``. The same text already goes to the anchoring stage, so it
is not new information entering the method, only information reaching this stage
unfiltered.

WHY THE SIGNALS EXIST, measured on SWE-Bench-Pro (86 cases, sp_05):

  * The localiser already treats this guess as a filter, not a hint. Where the
    reference step is a candidate INSIDE a guessed phase it is found 0.559 of
    the time; where it is a candidate outside every guessed phase, 0.094 -- and
    0.094 is chance among ~11 candidates. So the prior is not one voice among
    several: its recall (0.500) is the ceiling on the whole method, and every
    downstream improvement is capped by it.
  * Half of what it spends is unspendable. 127 of 258 guessed phases contain no
    step that changed anything, and the localiser is restricted to steps that
    did -- so the answer provably cannot be in them. In 17 of 86 cases ALL the
    guesses are like that, and those cases score 0.118 against 0.290 for the
    rest. The stage was choosing regions without knowing where an answer was
    allowed to live, because nothing ever told it.

``outline.py`` forbids diagnosis in its own stage, and rightly: the outline is
read as a neutral description, so a diagnostic aside there would bias the
answer invisibly. This stage crosses that line on purpose, which is why it
lives in its own file and labels its output as a guess everywhere it appears.
The outline itself is unchanged and the prior never feeds back into it.

The ground truth never enters here: this module sees ``Trajectory``, ``Span``
and ``FlowGraph``, none of which can hold a label.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .llm import LLM, map_concurrent
from .localize import _hops_from_symptom, _names_failure_artefact
from .schema import (CONTROL_FLOW, DATA_FLOW, LIKELY, POSSIBLE, FlowGraph,
                     Span, SpanGuess, SpanPrior, Trajectory, clip_middle)

logger = logging.getLogger(__name__)

PRIOR_SYSTEM = (
    "You narrow down where a failure's cause lies in an agent trajectory. "
    "You name phases, never individual steps. Answer strictly in JSON."
)

PRIOR_PROMPT = """A coding agent FAILED this task. Its run has been divided
into phases, and each phase has been summarised. Say which phases plausibly
CONTAIN the step that caused the failure.

You are NOT choosing the step. A later reader will do that, over the full
step-by-step evidence, which you cannot see. Your job is to narrow down where
that reader should look FIRST.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

{symptom_block}THE FAILURE BECAME VISIBLE AT: {anchors}
That is where the causal chain ENDS. The phase containing those steps is rarely
the phase that caused the failure — usually it is only where the consequence
surfaced.

{signals_note}PHASES ({n} phases over {steps} steps):
{phases}

HOW TO CHOOSE PHASES:
1. Exploration is not an error. While the agent is still investigating
   alternatives it has not gone wrong. Look for the phase where it COMMITTED to
   a direction — acted on it, or settled on it as the single way forward.
2. Prefer the EARLIEST phase that could plausibly start the causal chain, not a
   later phase where the consequence showed up.
3. Apply the but-for test at phase level: if everything in this phase had been
   done correctly, would the failure described above still have happened?
4. You are reading summaries, not evidence. Where a summary is too thin to
   judge, say "possible", not "likely".

Name AT MOST {budget} phase(s), most likely first. Naming more than you can
justify is worse than naming one: a hint that points at most of the run tells
the next reader nothing. Use the exact step ranges given above. Return ONLY
JSON:
{{
  "spans": [
    {{"start": <int>, "end": <int>, "likelihood": "likely|possible",
      "why": "<one sentence: what in this phase could have caused the failure>"}}
  ]
}}"""


# The bracketed facts appended to each phase line when ``signals`` is on. Each
# bullet carries its own failure mode, because a count handed over without one
# is read as a score -- and none of these is a score. Nothing here names an
# edge, an entity or a tier: the stage stays a summary-level reader.
SIGNALS_NOTE = """WHAT THE BRACKETS AFTER EACH PHASE MEAN
Mechanical facts about the phase, computed from the trace. They are not
judgements and the phases are not ordered by them.
  * "data=N control=N" — how many of the phase's steps lie on a traced
    dependency path back to the symptom. Zero does not clear a phase: tracing is
    incomplete and gets thinner the further back it reaches.
  * "N hops from symptom" — fewest traced edges between any step in the phase
    and a step where the failure surfaced. Larger means further upstream, i.e.
    earlier in the causal story, which is where a fault gets introduced rather
    than passed along. But a long chain also had more room to drift into steps
    that merely came earlier, so treat it as where to look, not as evidence.
  * "names a failure artefact" — the phase handles a thing the failure summary
    is about. That makes it on-topic, not guilty.{changes}

"""

CHANGES_BULLET = """
  * "changes N" — how many steps in the phase CHANGED something: wrote a file,
    applied a patch, ran a command with side effects. The reader after you must
    name one of those steps and cannot answer with anything else, so a phase
    with "changes 0" cannot contain the answer however suspicious its summary
    reads — naming one spends a guess that cannot be cashed. This is the only
    hard constraint here; the rest are hints."""


# The harness's own report of the failure, when the corpus carries one. This is
# the SAME symptom the summary above describes, only unfiltered: the summary is
# an earlier model's two sentences about it, written while anchoring, and a
# phase-level judgement made from that summary inherits whatever that model
# dropped. TrajAudit ships it (``failed_tests``); TrajDebug does not, and the
# block then renders empty, which is also what keeps the SWE-Bench-Pro arm of
# any A/B byte-identical.
#
# It is harness output, not annotation: which test the instance is graded on and
# the traceback it produced. The human label lives in ``failure_id`` and the
# reference fix in ``gold_patch``, and the loader reads neither into ``meta``.
SYMPTOM_NOTE = """WHAT THE HARNESS REPORTED (verbatim; the summary above is one
reader's account of this, this is the thing itself)
  * The test states what the task REQUIRED. A phase that committed to a
    direction incompatible with it is a candidate however confident its summary
    reads — and a phase whose summary claims success the test contradicts is a
    phase that measured the wrong thing.
  * It names files, symbols and tests in the CODEBASE, not phases in this run.
    A phase that touches them is on-topic, not guilty: the agent had to touch
    them to attempt the task at all.{grade}

FAILING TEST:
{code}

ERROR:
{error}

"""

# Only stated when the corpus says which kind it is, because the two readings
# are opposite and guessing between them is worse than saying nothing.
GRADE_BULLETS = {
    "pass_to_pass": """
  * This test PASSED before the agent started, so the run BROKE it. The cause is
    a change the agent made, not something it failed to do.""",
    "fail_to_pass": """
  * This test was failing before the agent started: making it pass WAS the task.
    So the cause may equally be a change that was never made.""",
}


def _symptom_block(traj: Trajectory) -> str:
    """The harness's failing test and error, or "" when the corpus has none."""
    code = str(traj.meta.get("failed_test_code") or "").strip()
    err = str(traj.meta.get("failed_test_error") or "").strip()
    if not (code or err):
        return ""
    grade = GRADE_BULLETS.get(
        str(traj.meta.get("failed_test_expected") or "").strip().lower(), "")
    return SYMPTOM_NOTE.format(
        grade=grade,
        code=clip_middle(code, 1500) or "(not reported)",
        error=clip_middle(err, 1500) or "(not reported)")


def _signal_rows(traj: Trajectory, spans: List[Span], graph: FlowGraph,
                 cands: Optional[List[Dict[str, Any]]]) -> List[str]:
    """One bracketed fact string per phase, in phase order.

    Counts only. The prior gets to know WHERE the traced flow is dense and where
    an answer is allowed to live, not what the edges say -- that distinction is
    what keeps it from being a paraphrase of the localiser's own evidence.
    """
    data_hops = _hops_from_symptom(graph, DATA_FLOW)
    ctrl_hops = _hops_from_symptom(graph, CONTROL_FLOW)
    data_steps = graph.data_flow_steps()
    ctrl_steps = graph.control_flow_steps()
    cand_steps = {int(c["step"]) for c in (cands or [])}
    out = []
    for sp in spans:
        idxs = set(sp.indices)
        hops = [h[i] for h in (data_hops, ctrl_hops) for i in idxs if i in h]
        bits = ["data=%d control=%d" % (len(idxs & data_steps),
                                        len(idxs & ctrl_steps))]
        bits.append(("%d hop%s from symptom"
                     % (min(hops), "" if min(hops) == 1 else "s")) if hops
                    else "no traced path to the symptom")
        if cands is not None:
            bits.append("changes %d" % len(idxs & cand_steps))
        if any(_names_failure_artefact(e.entity, graph.failure_entities)
               for e in graph.edges if e.src in idxs or e.dst in idxs):
            bits.append("names a failure artefact")
        out.append("  [%s]" % ", ".join(bits))
    return out


def predict_spans(traj: Trajectory, spans: List[Span], graph: FlowGraph,
                  llm: LLM, max_guesses: int = 3,
                  max_coverage: float = 0.5, signals: bool = False,
                  cands: Optional[List[Dict[str, Any]]] = None,
                  symptom: bool = False, samples: int = 1) -> SpanPrior:
    """Guess which phases contain the critical step, from summaries alone.

    ``max_guesses`` is an upper bound, not a target: the effective cap scales
    with how many phases there are. Naming three of four phases is not a
    narrowing, and the smoke run showed exactly that — on short trajectories
    ``segment`` produces its floor of four spans and a three-phase guess
    nominated 80% of the run, which makes the recall it earns meaningless.

    ``signals`` appends per-phase counts to each phase line and explains them;
    ``cands`` adds the count of steps in each phase that changed something, so a
    guess can avoid phases the restricted answer space cannot reach.

    ``symptom`` adds the harness's own failing test and traceback, where the
    corpus has one. Independent of ``signals``: that one tells the prior where
    an answer may LIVE, this one tells it what the task was actually graded on.
    On a corpus without the field it renders to nothing.

    ``samples`` > 1 draws the SAME prompt k times and UNIONS the phases named,
    ranked by how many draws named each. It exists because of a measured arity
    deficit, not because the draws disagree: ``budget`` is 3 on every SP row and
    2-3 on 71 TA rows, yet qwen3-235b names exactly ONE phase on 78/86 SP and
    85/98 TA rows, where claude-opus named three. The parse was cleared first --
    neither the coverage rule nor the budget cut a second guess on any of those
    rows -- so the model is simply not emitting what it is asked for.

    This is a union, not a vote, and the distinction is the whole design: the
    falsified shape on this repo is ``--samples 5`` on the LOCALISER (SP 0.3023,
    TA 0.4216), which picked ONE answer out of k when the model was already
    self-consistent. Here k draws each contribute a region and ``_parse``'s
    unchanged ``budget`` and ``max_coverage`` rules then cap the result, so the
    union can never nominate more than the design already allowed for the model
    that did comply. At ``samples=1`` the call, its cache key and the parse are
    byte-identical to a run without the option.

    With every flag off the prompt is byte-identical to a run without any.
    """
    if len(spans or []) < 4:
        # Nothing to narrow: any guess here covers a large share of the run,
        # and it would still cost a call.
        return SpanPrior()
    # one guess per three phases: 4-5 phases -> 1, 6-8 -> 2, 9+ -> 3
    budget = min(max_guesses, max(1, len(spans) // 3))

    rows = (_signal_rows(traj, spans, graph, cands) if signals
            else [""] * len(spans))
    phases = []
    for sp, sig in zip(spans, rows):
        # ``summary`` is written by the compressor, so reading it is what pins
        # this stage AFTER compression. ``digest`` is the rule-derived
        # description ``spans.segment_units`` attaches, which exists before any
        # compression call -- so when the prior runs early to condition the
        # compressor, the phase lines still say what happened in each phase.
        # Order matters: ``summary`` wins where both exist, which is every arm
        # that runs the prior in its historical position, so those prompts stay
        # byte-identical.
        body = (sp.summary or sp.digest or "(no summary)").strip()
        phases.append("steps %d-%d — %s%s\n    %s"
                      % (sp.start, sp.end, sp.label or "(unlabelled)", sig,
                         body))

    prompt = PRIOR_PROMPT.format(
        task=clip_middle(traj.task_description, 2000),
        failure_summary=(graph.failure_summary or "(not summarised)"),
        anchors=(", ".join("step %d" % a for a in sorted(graph.anchors))
                 or "(none identified)"),
        n=len(spans), steps=len(traj), budget=budget,
        symptom_block=(_symptom_block(traj) if symptom else ""),
        signals_note=(SIGNALS_NOTE.format(
            changes=(CHANGES_BULLET if cands is not None else ""))
            if signals else ""),
        phases=clip_middle("\n".join(phases), 40000))

    k = max(1, int(samples or 1))
    if k == 1:
        obj = llm.generate_obj(prompt, max_tokens=1200, system=PRIOR_SYSTEM)
    else:
        # Variant 0 is the configured temperature and carries the key a stored
        # samples=1 run already wrote, so an A/B against one pays only for the
        # k-1 extra draws.
        draws = map_concurrent(
            lambda v: llm.generate_obj(prompt, max_tokens=1200,
                                       system=PRIOR_SYSTEM, variant=v),
            list(range(k)), workers=k)
        obj = _union_draws([d for d in draws if isinstance(d, dict)], spans)
    prior = _parse(obj, spans, budget, len(traj), max_coverage)
    if not prior:
        logger.info("%s: span prior empty, localising without it", traj.task_id)
    return prior


def _union_draws(draws: List[Dict[str, Any]],
                 spans: List[Span]) -> Dict[str, Any]:
    """Merge k draws of the prior prompt into one ``{"spans": [...]}`` object.

    Each draw's ranges are snapped onto real phases FIRST, so two draws that
    described the same phase with different off-by-one ends count as the same
    vote rather than as two guesses. The output is ordered by vote count, then
    by mean rank within the draws that named it -- a phase two draws put first
    outranks one that three draws put last only if the votes say so. Ordering
    matters because ``_parse`` keeps the model's own order and the first guess
    is reported separately as the top guess.

    Returned in the raw wire shape rather than as guesses, so ``_parse`` stays
    the single place that applies ``budget`` and ``max_coverage``. Nothing here
    reads the ground truth; votes and ranks are properties of the draws.
    """
    votes: Dict[Any, Dict[str, Any]] = {}
    for d in draws:
        seen_here = set()
        for rank, item in enumerate(d.get("spans") or []):
            if not isinstance(item, dict):
                continue
            try:
                lo, hi = int(item["start"]), int(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if hi < lo:
                lo, hi = hi, lo
            sp = _best_overlap(lo, hi, spans)
            if sp is None:
                continue
            key = (sp.start, sp.end)
            # One draw naming a phase twice is one vote for it.
            if key in seen_here:
                continue
            seen_here.add(key)
            rec = votes.setdefault(key, {"n": 0, "ranks": [], "likely": 0,
                                         "why": ""})
            rec["n"] += 1
            rec["ranks"].append(rank)
            if str(item.get("likelihood") or "").strip().lower() == LIKELY:
                rec["likely"] += 1
            if not rec["why"]:
                rec["why"] = str(item.get("why") or "")
    order = sorted(votes.items(),
                   key=lambda kv: (-kv[1]["n"],
                                   sum(kv[1]["ranks"]) / len(kv[1]["ranks"]),
                                   kv[0][0]))
    out = []
    for (start, end), rec in order:
        out.append({
            "start": start, "end": end,
            # A phase is "likely" when most of the draws that named it said so.
            "likelihood": (LIKELY if rec["likely"] * 2 > rec["n"]
                           else POSSIBLE),
            "why": rec["why"],
        })
    return {"spans": out}


def _parse(obj: Dict[str, Any], spans: List[Span], max_guesses: int,
           n_steps: int, max_coverage: float) -> SpanPrior:
    """Snap the model's ranges onto real phases; drop what does not land.

    The model is given the exact ranges but still invents them — an off-by-one
    end, or a range spanning two phases. Snapping keeps the artifact and the
    coverage diagnostic honest: a guess always denotes a phase that exists, so
    ``prior_coverage`` measures the region actually nominated rather than
    whatever interval the model happened to type. Order is the model's own,
    because the first guess is reported separately as the top guess.

    Guesses stop once they would nominate more than ``max_coverage`` of the
    run. This is a bound on what the output can mean, not a tuned filter: a
    hint pointing at most of the trajectory earns recall for free and cannot be
    read as narrowing anything. The FIRST guess is always kept — if the model's
    strongest phase is itself huge, that is a fact about the segmentation and
    the coverage number should say so rather than the prior vanishing.
    """
    raw = obj.get("spans") if isinstance(obj, dict) else None
    guesses: List[SpanGuess] = []
    seen = set()
    covered: set = set()
    limit = int(n_steps * max_coverage) if n_steps else 0
    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        try:
            lo, hi = int(item["start"]), int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if hi < lo:
            lo, hi = hi, lo
        sp = _best_overlap(lo, hi, spans)
        if sp is None or (sp.start, sp.end) in seen:
            continue
        steps = set(range(sp.start, sp.end + 1))
        if guesses and len(covered | steps) > limit:
            break
        seen.add((sp.start, sp.end))
        covered |= steps
        lk = str(item.get("likelihood") or "").strip().lower()
        guesses.append(SpanGuess(
            start=sp.start, end=sp.end,
            likelihood=(LIKELY if lk == LIKELY else POSSIBLE),
            why=str(item.get("why") or "")[:300]))
        if len(guesses) >= max_guesses:
            break
    return SpanPrior(guesses=guesses)


def _best_overlap(lo: int, hi: int, spans: List[Span]) -> Optional[Span]:
    """The phase sharing the most steps with ``[lo, hi]``; None if none do."""
    best, best_n = None, 0
    for sp in spans:
        n = min(hi, sp.end) - max(lo, sp.start) + 1
        if n > best_n:
            best, best_n = sp, n
    return best


def prior_stats(prior: Optional[SpanPrior], n_steps: int) -> Dict[str, Any]:
    covered = prior.n_steps_covered() if prior else 0
    return {
        "guesses": len(prior.guesses) if prior else 0,
        "likely": sum(1 for g in prior.guesses if g.likelihood == LIKELY)
                  if prior else 0,
        "steps_covered": covered,
        "coverage": round(covered / n_steps, 4) if n_steps else 0.0,
    }
