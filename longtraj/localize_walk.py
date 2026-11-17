"""Ranked map-reduce localiser: rank spans -> walk them -> adjudicate.

The single-shot :func:`longtraj.localize.localize` renders the whole compressed
trajectory into one prompt and answers in one call. It tends to a *late bias*:
the model anchors on the tail, near where the failure surfaced, instead of
walking back to the step that introduced the fault.

This module splits the job into three stages so no single call has to hold the
whole run at once:

  * **rank**   — one call orders the spans, most- to least-likely to contain the
                 root cause, and names a cutoff. Ranking is grounded in a
                 mechanical per-span signal table (flow counts, distance to the
                 symptom, failure-artefact hits), not prose alone.
  * **walk**   — the localiser agent visits spans in ranked order, up to a
                 budget. Per span it sees ONLY that span's compressed steps plus
                 pre-computed flow evidence (parents, distance, cross-span entity
                 hits) and the candidates gathered so far, and nominates at most
                 one suspect step or skips. The client has no function-calling,
                 so the "tools" are evidence computed here and injected, not live
                 calls.
  * **reduce** — one call adjudicates the candidate set into the final step,
                 preferring the earliest genuine cause.

Ground truth never enters this module: it only ever receives ``Trajectory``,
``Span``, ``FlowGraph``, ``StepDigest`` and ``Outline`` — none can hold a label.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .compress import render_compressed
from .llm import LLM
from .localize import (ANSWER_BUDGET, _fallback_step, _fit_budget, _flow_block,
                       _hops_from_symptom, _names_failure_artefact,
                       _snap_to_attributable, _with_rejection)
from .schema import (CONTROL_FLOW, DATA_FLOW, FlowGraph, Outline, Prediction,
                     Span, StepDigest, Trajectory, clip_middle)

logger = logging.getLogger(__name__)

RANK_BUDGET = 2000
PROPOSE_BUDGET = 1600

RANK_SYSTEM = (
    "You triage segments of a failed agent trajectory, ordering them by how "
    "likely each is to CONTAIN the step that introduced the fault. Answer "
    "strictly in JSON."
)

PROPOSE_SYSTEM = (
    "You examine ONE segment of a failed agent trajectory and decide whether it "
    "contains a step whose correction would have prevented the failure. If it "
    "does, you nominate the earliest such step; otherwise you skip. You judge "
    "the segment on its own merits — a later stage compares all nominations and "
    "picks the single root cause. Answer strictly in JSON."
)

ADJUDICATE_SYSTEM = (
    "You perform root-cause localisation on a failed agent trajectory. A prior "
    "segment-by-segment walk has nominated suspect steps; using those suspects "
    "together with the traced dependencies, the outline and the full compressed "
    "trajectory, you identify the earliest step whose error causally led to the "
    "observed failure. Answer strictly in JSON."
)


# ---------------------------------------------------------------------------
# Mechanical signals (no LLM) shared by ranking and the per-span evidence.
# ---------------------------------------------------------------------------
def _min_hops(i: int, data_hops: Dict[int, int],
              ctrl_hops: Dict[int, int]) -> Optional[int]:
    """Distance to the nearest symptom over either relation, or None."""
    vals = [h[i] for h in (data_hops, ctrl_hops) if i in h]
    return min(vals) if vals else None


def _entity_index(graph: FlowGraph) -> Dict[str, set]:
    """Map each data-flow entity (lower-cased) to the steps that carry it."""
    out: Dict[str, set] = {}
    for e in graph.edges:
        if e.kind == DATA_FLOW and e.entity:
            out.setdefault(e.entity.strip().lower(), set()).update({e.src, e.dst})
    return out


def _span_signals(traj: Trajectory, spans: Sequence[Span],
                  graph: FlowGraph) -> List[dict]:
    """One mechanical signal row per span, in span order."""
    data_hops = _hops_from_symptom(graph, DATA_FLOW)
    ctrl_hops = _hops_from_symptom(graph, CONTROL_FLOW)
    data_steps = graph.data_flow_steps()
    ctrl_steps = graph.control_flow_steps()
    rows = []
    for j, sp in enumerate(spans):
        idxs = set(sp.indices)
        hops = [h for h in (_min_hops(i, data_hops, ctrl_hops) for i in idxs)
                if h is not None]
        artefact = any(
            _names_failure_artefact(e.entity, graph.failure_entities)
            for e in graph.edges if e.src in idxs or e.dst in idxs)
        rows.append({
            "pos": j,
            "start": sp.start,
            "end": sp.end,
            "label": sp.label or "(unlabelled)",
            "tier": sp.tier,
            "ndata": len(idxs & data_steps),
            "nctrl": len(idxs & ctrl_steps),
            "hops": min(hops) if hops else None,
            "artefact": artefact,
        })
    return rows


def _signal_table(rows: Sequence[dict]) -> str:
    lines = []
    for r in rows:
        hops = r["hops"] if r["hops"] is not None else "-"
        flags = " *names failure artefact*" if r["artefact"] else ""
        lines.append(
            "#%d  steps %d-%d  [%s]  data=%d control=%d  nearest_hops=%s%s\n"
            "     %s"
            % (r["pos"], r["start"], r["end"], r["tier"], r["ndata"],
               r["nctrl"], hops, flags, r["label"]))
    return "\n".join(lines)


def _fallback_rank(rows: Sequence[dict]) -> Tuple[List[int], int]:
    """Deterministic order when the model gives nothing usable.

    On-flow (weak) spans first, furthest upstream first — a high hop count is
    further from the symptom, i.e. earlier in the causal story and a better
    place to look for the step that introduced the fault. Off-flow spans follow
    in trajectory order. The cutoff is the number of on-flow spans (>=1).
    """
    weak = [r for r in rows if r["tier"] == "weak"]
    strong = [r for r in rows if r["tier"] != "weak"]
    weak.sort(key=lambda r: (-(r["hops"] if r["hops"] is not None else -1),
                             r["start"]))
    strong.sort(key=lambda r: r["start"])
    order = [r["pos"] for r in weak] + [r["pos"] for r in strong]
    return order, max(1, len(weak))


# ---------------------------------------------------------------------------
# Stage A — rank the spans.
# ---------------------------------------------------------------------------
RANK_PROMPT = """A coding agent FAILED this task. Its run was segmented into the
numbered segments below. Order them by how likely each is to CONTAIN the step
that INTRODUCED the fault — not the step where the failure became visible.

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

SYMPTOM STEPS: {anchors}
These are where the failure surfaced. The cause is upstream of them, so a
segment further from the symptom (a higher "nearest_hops") is often where the
fault was introduced, while a segment sitting on the symptom usually only
reports it.

SEGMENT SIGNALS (mechanical facts about each segment's place in the traced
dependency graph — not judgements):
  * data / control = how many of the segment's steps lie on the data / control
    dependency path back to the symptom.
  * nearest_hops = fewest edges between any step in the segment and a symptom
    step ("-" means no traced path to the symptom).
  * "names a failure artefact" = the segment handles a thing the failure is about.

{table}

HOW THE AGENT EXECUTED THE TASK (descriptive only):
{outline}

Rank EVERY segment. Put the segments most likely to contain the fault-
introducing step first. Then give a cutoff: the number of top-ranked segments
worth examining closely (the rest are unlikely to repay the effort).

Return ONLY JSON:
{{"order": [<segment number>, ...],
  "cutoff": <int: how many of the top segments to examine>,
  "why": "<1-2 sentences on what drove the top of the ranking>"}}"""


def rank_spans(traj: Trajectory, spans: Sequence[Span], graph: FlowGraph,
               outline: Optional[Outline], llm: LLM) -> Tuple[List[int], int]:
    """Order span positions most- to least-likely to hold the root; + a cutoff."""
    rows = _span_signals(traj, spans, graph)
    outline_text = outline.render() if outline is not None else ""
    prompt = RANK_PROMPT.format(
        failure_summary=graph.failure_summary or "(not summarised)",
        anchors=(", ".join("step %d" % a for a in sorted(graph.anchors))
                 or "(none identified)"),
        table=_signal_table(rows),
        outline=clip_middle(outline_text, 8000) or "(no outline available)")

    obj = llm.generate_obj(prompt, max_tokens=RANK_BUDGET, system=RANK_SYSTEM)

    n = len(spans)
    order: List[int] = []
    seen = set()
    for v in (obj.get("order") or []):
        try:
            p = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= p < n and p not in seen:
            seen.add(p)
            order.append(p)
    # any span the model omitted still gets a place, in natural order
    for p in range(n):
        if p not in seen:
            order.append(p)

    try:
        cutoff = int(obj.get("cutoff"))
    except (TypeError, ValueError):
        cutoff = 0
    cutoff = max(1, min(cutoff, n))

    if not obj:
        order, cutoff = _fallback_rank(rows)
    logger.info("%s: span rank order=%s cutoff=%d", traj.task_id, order, cutoff)
    return order, cutoff


# ---------------------------------------------------------------------------
# Stage B — walk one span, nominate at most one step.
# ---------------------------------------------------------------------------
def _span_flow_evidence(sp: Span, graph: FlowGraph, on_flow: set,
                        ent_index: Dict[str, set], data_hops: Dict[int, int],
                        ctrl_hops: Dict[int, int], max_steps: int = 12,
                        max_refs: int = 6) -> str:
    """Pre-computed read-only 'tools' for the agent, scoped to this span.

    For each on-flow step in the span: its distance to the symptom, the steps it
    depends on (parents, with the thing each edge carries), and where the
    entities it touches reappear ELSEWHERE in the run (cross-span retrieval).
    """
    idxs = set(sp.indices)
    on = [i for i in sp.indices if i in on_flow]
    if not on:
        return "(no traced data or control flow runs through this segment)"
    lines = []
    for i in on[:max_steps]:
        hop = _min_hops(i, data_hops, ctrl_hops)
        head = "step %d — %s hops from symptom" % (
            i, hop if hop is not None else "no traced path")
        lines.append(head)
        parents = graph.predecessors(i)
        for e in parents[:max_refs]:
            ent = (" [%s]" % e.entity) if e.entity else ""
            art = (" — names a failure artefact"
                   if _names_failure_artefact(e.entity, graph.failure_entities)
                   else "")
            lines.append("    depends on step %d%s (%s)%s"
                         % (e.src, ent, e.kind, art))
        # cross-span: entities this step carries, seen outside the span
        carried = {e.entity.strip().lower()
                   for e in graph.edges
                   if e.kind == DATA_FLOW and e.entity and (e.src == i or e.dst == i)}
        for ent in list(carried)[:max_refs]:
            elsewhere = sorted(s for s in ent_index.get(ent, ()) if s not in idxs)
            if elsewhere:
                lines.append("    entity '%s' also at steps %s"
                             % (ent, ", ".join(str(s) for s in elsewhere[:max_refs])))
    if len(on) > max_steps:
        lines.append("    ... %d more on-flow steps in this segment" % (len(on) - max_steps))
    return "\n".join(lines)


def _candidates_block(candidates: Sequence[dict]) -> str:
    if not candidates:
        return "(none yet — this is the first segment examined)"
    lines = []
    for c in candidates:
        lines.append("  step %d (from segment #%d, confidence %.2f): %s"
                     % (c["step"], c["rank_pos"], c["confidence"],
                        (c["why"] or "")[:200]))
    return "\n".join(lines)


PROPOSE_PROMPT = """A coding agent FAILED this task. You are examining ONE
segment of its run. Decide whether THIS segment contains a step whose correction
would have prevented the failure, and if so name it.

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

SYMPTOM STEPS: {anchors}
These are where the failure surfaced, at the END of the run. A fix-point is
usually EARLIER — where a mistake was first made, not only where it showed up.

HOW THE AGENT EXECUTED THE TASK (descriptive only, for orientation):
{outline}

FLOW EVIDENCE FOR THIS SEGMENT
Pre-computed facts about how this segment's steps connect to the failure: how
far each is from the symptom, which earlier steps it depends on, and where the
things it touches reappear elsewhere. Use it to judge whether a step here is
causally implicated in the failure.
{evidence}

NOMINATIONS FROM OTHER SEGMENTS SO FAR (context only — do NOT defer to them):
These are suspects flagged elsewhere. They do NOT mean the cause is already
found. A later stage compares every nomination and picks the earliest genuine
cause, so you should still nominate this segment's own fix-point if it has one,
even if an earlier or later segment also looks suspect.
{candidates}

THIS SEGMENT — steps {start}-{end}: {label}
{body}

Decide, judging THIS segment on its own merits:
  * Apply the but-for test to the steps here: is there a step in this segment
    such that, had it been done correctly, the failure would NOT have happened?
    If so, nominate exactly ONE step — the earliest such step in this segment.
  * A step still qualifies even if the same fault also passes through later
    steps, or might also exist upstream. You are NOT deciding whether this is THE
    single root cause of the whole run — only surfacing this segment's best
    fix-point. The comparison across segments happens later.
  * Exploration and reading are not faults. A well-formed action can still be the
    fault if the DECISION behind it was wrong.
  * SKIP only if no step here is causally implicated — i.e. correcting any step
    in this segment would leave the failure unchanged. Do NOT skip merely because
    the mistake might also originate upstream.
  * Do not nominate a step just because the failure is visible near it.

Return ONLY JSON:
{{"has_candidate": <true|false>,
  "step": <int step index in this segment, or null>,
  "confidence": <float 0..1>,
  "why": "<1-3 sentences: what this step did wrong and how it fed the failure>"}}"""


def propose_for_span(traj: Trajectory, spans: Sequence[Span], span_pos: int,
                     graph: FlowGraph, digests: Sequence[StepDigest],
                     outline: Optional[Outline], candidates: Sequence[dict],
                     llm: LLM, on_flow: set, ent_index: Dict[str, set],
                     data_hops: Dict[int, int], ctrl_hops: Dict[int, int]
                     ) -> Optional[dict]:
    """Ask whether this span holds the fault-introducing step; nominate one or skip."""
    sp = spans[span_pos]
    idxs = set(sp.indices)
    span_digests = [d for d in digests if d.index in idxs]
    body = render_compressed(traj, [sp], span_digests) or "(segment is empty)"
    evidence = _span_flow_evidence(sp, graph, on_flow, ent_index,
                                   data_hops, ctrl_hops)
    outline_text = outline.render() if outline is not None else ""

    prompt = PROPOSE_PROMPT.format(
        failure_summary=graph.failure_summary or "(not summarised)",
        anchors=(", ".join("step %d" % a for a in sorted(graph.anchors))
                 or "(none identified)"),
        outline=clip_middle(outline_text, 4000) or "(no outline available)",
        evidence=evidence,
        candidates=_candidates_block(candidates),
        start=sp.start, end=sp.end, label=sp.label or "(unlabelled)",
        body=clip_middle(body, 40000))

    obj = llm.generate_obj(prompt, max_tokens=PROPOSE_BUDGET,
                           system=PROPOSE_SYSTEM)
    # Distinguish an endpoint failure (empty dict — the vveai gateway returns
    # empty HTTP-200s under load) from a deliberate skip ("has_candidate": false,
    # a non-empty object). Both yield no candidate, but only the former is a lost
    # nomination worth flagging when reading the walk's candidate counts.
    if not obj:
        logger.warning("%s: segment #%d produced an empty response (endpoint "
                       "failure, not a skip); no nomination from this segment",
                       traj.task_id, span_pos)
        return None
    if not obj.get("has_candidate"):
        return None
    try:
        step = int(obj.get("step"))
    except (TypeError, ValueError):
        return None
    step = _snap_to_attributable(step, traj)
    if step is None:
        return None
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"step": step, "rank_pos": span_pos,
            "span": [sp.start, sp.end], "confidence": conf,
            "why": str(obj.get("why") or "")[:600]}


# ---------------------------------------------------------------------------
# Stage C — final localisation over the candidate set + full evidence.
#
# The walk's map stage surfaces suspects one segment at a time, each judged in
# isolation. This stage does the reduce as a GROUNDED final localisation: it puts
# the nominated suspects in front of the model together with the whole run — the
# traced data/control flow, the outline, and the full compressed trajectory
# (strong where the flow says a step matters, weak elsewhere) — and asks for the
# single earliest cause. The suspects anchor attention against the late bias of a
# cold single-shot read, while the full trajectory lets the model confirm a
# suspect, choose the earliest of several, or walk one step further back to the
# true origin. The prompt mirrors localize.LOCALIZE_PROMPT.
# ---------------------------------------------------------------------------
ADJUDICATE_PROMPT = """A coding agent FAILED this task. Your job is to make
the final call: from all the evidence below, name the critical step — the
EARLIEST step whose mistake causally led to the observed failure.

The evidence is laid out in five sections:
  SECTION 1  the failure — the symptom you are explaining
  SECTION 2  suspect steps — the walk's nominations, your primary candidates
  SECTION 3  traced dependencies — data and control flow back from the symptom
  SECTION 4  trajectory outline — how the run unfolded, phase by phase
  SECTION 5  compressed trajectory — every step, in full or in summary
Read SECTION 1 to fix the target, start from the suspects in SECTION 2, then use
SECTIONS 3-5 to confirm the earliest genuine cause.

============================================================
SECTION 1 — THE FAILURE (the symptom, not the cause)
============================================================
{failure_summary}

SYMPTOM STEPS: {anchors}
These are where the failure became VISIBLE — the patch was submitted, the test
reported its error, the run ended. Being where a failure surfaces is not evidence
of having caused it. A symptom step is the END of the causal chain you are looking
for: start there and walk backwards, do not stop there.

============================================================
SECTION 2 — SUSPECT STEPS (nominated by the segment walk)
============================================================
Each suspect was flagged by examining ONE segment in isolation as that segment's
best fix-point, and is shown with the step it depends on and its distance from the
symptom. Treat them as your PRIMARY candidates — the answer is most likely among
them — but remember each was chosen WITHOUT seeing the whole run. Use SECTIONS 3-5
to (a) pick the earliest suspect that genuinely STARTED the causal chain, and
(b) check whether an even earlier step on the SAME chain is the true origin.
{candidates}

============================================================
SECTION 3 — TRACED DEPENDENCIES
============================================================
Two SEPARATE relations were traced back from the symptom steps. They are
independent: a step may appear in one, both, or neither. Each edge is annotated
with mechanical facts about its position in the graph (hops from the symptom,
whether it names a failure artefact, how many chains converge on it); these are
not judgements, and the edges are not ordered by them.

--- DATA FLOW: where the information used at each step came from ---
An edge "step A -> step B [thing]" means the thing named in brackets was produced,
transformed, or dropped at step A, and step B relied on it. Follow it to find
where the information FIRST became wrong, not where it was merely passed along.
{data_block}

--- CONTROL FLOW: why the agent went from one step to the next ---
An edge "step A -> step B" means step A caused the agent to take step B: A set the
plan B carries out, or A failed and B is the reaction. Plain sequence is NOT an
edge. Follow it to find where the agent committed to the wrong direction.
{control_block}

============================================================
SECTION 4 — TRAJECTORY OUTLINE
============================================================
A description of the run, phase by phase. Descriptive only; it makes no claim
about what went wrong.
{outline}

============================================================
SECTION 5 — COMPRESSED TRAJECTORY, STEP BY STEP
============================================================
Every step of the run is here under its original index — nothing was dropped.
Steps judged RELATED to the failure keep much more of their detail; steps judged
largely unrelated are reduced to a summary. That judgement came from the
dependency tracing and is a heuristic, not a verdict: a step kept in full is not
thereby guilty, and a summarised step is not thereby innocent.
{trajectory}

============================================================
HOW TO CHOOSE
============================================================
1. Exploration is not an error. The error begins when the agent COMMITS to a
   flawed direction — acts on it, or states it as the single path forward.
2. Apply the but-for test: if this step had been done correctly, would the
   failure still have happened? If yes, it is not the critical step.
3. Prefer the EARLIEST step that starts the causal chain, not a later step where
   the consequence became visible. If several steps repeat the same mistake, name
   the first one.
4. Ignore mistakes that did not contribute to THIS failure.
5. A step can be critical for flawed reasoning or a flawed plan, not only for a
   wrong action: a well-formed action can still be critical if the decision behind
   it was wrong.
6. A step reduced to a summary lost detail; if your reasoning depends on one, say
   so rather than treating the summary as the whole story.
7. Start from the suspects in SECTION 2 and prefer the earliest one whose
   correction would have prevented the failure. Choose a NON-suspect step only if
   SECTIONS 3-5 show it is an earlier link on the SAME chain that the walk missed —
   never a step that merely sits nearer the symptom.

Answering with a SYMPTOM STEP is a strong claim: only do so if you can say what it
did wrong BEYOND revealing the failure.

Return ONLY JSON:
{{
  "critical_step": <int>,
  "confidence": <float 0..1>,
  "error_type": "<plan|reason|act|obs|verify>.<short subtype>",
  "rejected_step": <int or null: a later step you considered and ruled out>,
  "rejected_because": "<why that step only propagated the fault>",
  "reason": "<2-4 sentences: what went wrong at the chosen step and how it caused the failure>"
}}"""


def _candidate_evidence(candidates: Sequence[dict], graph: FlowGraph,
                        data_hops: Dict[int, int],
                        ctrl_hops: Dict[int, int]) -> str:
    lines = []
    for c in sorted(candidates, key=lambda c: c["step"]):
        i = c["step"]
        hop = _min_hops(i, data_hops, ctrl_hops)
        parents = graph.predecessors(i)
        pref = "; ".join("step %d%s" % (e.src, (" [%s]" % e.entity) if e.entity else "")
                         for e in parents[:5]) or "none traced"
        lines.append(
            "step %d (segment #%d, confidence %.2f, %s hops from symptom)\n"
            "    depends on: %s\n"
            "    nominated because: %s"
            % (i, c["rank_pos"], c["confidence"],
               hop if hop is not None else "no traced path", pref,
               (c["why"] or "")[:400]))
    return "\n".join(lines)


def adjudicate(traj: Trajectory, spans: Sequence[Span], graph: FlowGraph,
               digests: Sequence[StepDigest], outline: Optional[Outline],
               candidates: List[dict], llm: LLM,
               max_prompt_chars: int = 120000) -> Prediction:
    """Localise the final step from the candidate set + full-trajectory evidence."""
    if not candidates:
        step = _fallback_step(traj, graph)
        return Prediction(
            task_id=traj.task_id, critical_step=step, confidence=0.0,
            reason="no segment nominated a candidate; fell back to the earliest "
                   "on-flow attributable step",
            candidates=[])

    data_hops = _hops_from_symptom(graph, DATA_FLOW)
    ctrl_hops = _hops_from_symptom(graph, CONTROL_FLOW)

    body = render_compressed(traj, list(spans), list(digests))
    if len(body) > max_prompt_chars:
        body = _fit_budget(traj, list(spans), list(digests), max_prompt_chars)
    outline_text = outline.render() if outline is not None else ""

    prompt = ADJUDICATE_PROMPT.format(
        failure_summary=graph.failure_summary or "(not summarised)",
        anchors=(", ".join("step %d" % a for a in sorted(graph.anchors))
                 or "(none identified)"),
        candidates=_candidate_evidence(candidates, graph, data_hops, ctrl_hops),
        data_block=_flow_block(graph, DATA_FLOW),
        control_block=_flow_block(graph, CONTROL_FLOW),
        outline=clip_middle(outline_text, 12000) or "(no outline available)",
        trajectory=body)

    obj = llm.generate_obj(prompt, max_tokens=ANSWER_BUDGET,
                           system=ADJUDICATE_SYSTEM)
    try:
        step = int(obj.get("critical_step"))
    except (TypeError, ValueError):
        step = None
    step = _snap_to_attributable(step, traj)
    if step is None:
        # Model gave nothing usable: keep the walk's work rather than a generic
        # default — take its highest-confidence nomination, then the on-flow floor.
        best = max(candidates, key=lambda c: c["confidence"])
        step = _snap_to_attributable(best["step"], traj) or _fallback_step(traj, graph)

    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0

    return Prediction(
        task_id=traj.task_id, critical_step=step, confidence=conf,
        reason=_with_rejection(obj),
        error_type=str(obj.get("error_type") or "")[:80],
        candidates=list(candidates))


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------
def localize_walk(traj: Trajectory, spans: List[Span], graph: FlowGraph,
                  llm: LLM, digests: Optional[Sequence[StepDigest]] = None,
                  outline: Optional[Outline] = None,
                  max_spans: int = 8, **_ignored) -> Prediction:
    """Rank spans, walk them under a budget, then adjudicate the candidates."""
    digests = list(digests or [])
    if not spans:
        step = _fallback_step(traj, graph)
        return Prediction(task_id=traj.task_id, critical_step=step,
                          reason="no spans to walk", candidates=[])

    order, cutoff = rank_spans(traj, spans, graph, outline, llm)
    # The ranker's cutoff may stop the walk early, but never below a floor of
    # ~3/4 of the budget: an over-eager cutoff otherwise truncates the walk a
    # slot short of the gold span (observed on django-15022, where the gold
    # span ranked 4th but cutoff=3 cut it off).
    floor = math.ceil(max_spans * 0.75)
    budget = min(max_spans, len(order), max(cutoff, floor))

    on_flow = graph.on_flow()
    ent_index = _entity_index(graph)
    data_hops = _hops_from_symptom(graph, DATA_FLOW)
    ctrl_hops = _hops_from_symptom(graph, CONTROL_FLOW)

    candidates: List[dict] = []
    for span_pos in order[:budget]:
        prop = propose_for_span(traj, spans, span_pos, graph, digests, outline,
                                candidates, llm, on_flow, ent_index,
                                data_hops, ctrl_hops)
        if prop is not None:
            candidates.append(prop)
            logger.info("%s: segment #%d nominated step %d (conf %.2f)",
                        traj.task_id, span_pos, prop["step"], prop["confidence"])

    logger.info("%s: walked %d/%d spans, %d candidates",
                traj.task_id, budget, len(spans), len(candidates))
    return adjudicate(traj, spans, graph, digests, outline, candidates, llm)
