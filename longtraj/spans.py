"""Segment a trajectory into semantic spans.

A span is a contiguous run of steps that together form one sub-goal — for
example "locate the failing module", "edit the parser", "run the test suite
and read the result". Spans are the unit of compression: the tier is decided
per span from how much of it lies on a traced flow.

The LLM proposes cut points from the trajectory digest; a deterministic
fallback segments on verb-shape changes so the pipeline still runs without a
working endpoint. Both paths guarantee a full, non-overlapping cover of
0..n-1, which the compressor relies on.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from . import units as units_mod
from .llm import LLM
from .schema import DATA_FLOW, STRONG, WEAK, FlowGraph, Span, Trajectory, clip_middle
from .tracing import build_index

logger = logging.getLogger(__name__)

SPAN_SYSTEM = ("You segment agent trajectories into coherent sub-goal phases. "
               "Answer strictly in JSON.")

SPAN_PROMPT = """Segment this agent trajectory into semantic spans.

A span is a run of CONSECUTIVE steps pursuing one sub-goal, for example:
exploring the repository, locating the relevant code, making an edit,
running tests and reading the outcome, reacting to a failure, finalising.

TASK:
{task}

TRAJECTORY DIGEST ({n} steps, indices 0..{last}):
{index}

Rules:
  - Spans must be contiguous, non-overlapping, and cover every step 0..{last}.
  - Start a new span when the agent's sub-goal changes: it switches target,
    reacts to a result, changes strategy, or moves from exploring to acting.
  - Aim for spans of {lo}-{hi} steps. Produce roughly {target} spans.
  - Give each span a short label naming its sub-goal.

Return ONLY JSON:
{{"spans": [{{"start": <int>, "end": <int>, "label": "<short phrase>"}}]}}"""


def segment(traj: Trajectory, llm: LLM, graph: Optional[FlowGraph] = None,
            target_spans: Optional[int] = None,
            index_chars: int = 50000) -> List[Span]:
    """Segment ``traj`` into semantic spans, LLM-first with a rule fallback."""
    n = len(traj)
    if n <= 2:
        return [Span(start=0, end=n - 1, label="whole trajectory")]

    target = target_spans or max(4, min(30, round(n / 6)))
    lo, hi = max(2, n // (target * 2)), max(4, n // max(1, target // 2))

    index = build_index(traj, per_step=150)
    if len(index) > index_chars:
        head = index[: index_chars // 2]
        tail = index[-(index_chars // 2):]
        index = head + "\n...[middle of digest omitted]...\n" + tail

    obj = llm.generate_json(
        SPAN_PROMPT.format(task=clip_middle(traj.task_description, 1500),
                           n=n, last=n - 1, index=index,
                           lo=lo, hi=hi, target=target),
        max_tokens=3000, system=SPAN_SYSTEM, default={})

    # A bare array means the model dropped the {"spans": ...} wrapper and
    # returned the list itself — that is the payload, so use it as-is rather
    # than discarding a perfectly good segmentation over its envelope.
    raw = obj if isinstance(obj, list) else (obj or {}).get("spans")
    spans = _coerce_spans(raw, n)
    if not spans:
        logger.info("span segmentation fell back to heuristic for %s", traj.task_id)
        spans = heuristic_spans(traj, target)
    return spans


# ---------------------------------------------------------------------------
def _coerce_spans(raw, n: int) -> List[Span]:
    """Validate LLM spans and repair them into a full non-overlapping cover."""
    if not isinstance(raw, list) or not raw:
        return []
    cuts = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            s, e = int(item.get("start")), int(item.get("end"))
        except (TypeError, ValueError):
            continue
        if s > e:
            s, e = e, s
        s, e = max(0, min(s, n - 1)), max(0, min(e, n - 1))
        cuts.append((s, e, str(item.get("label") or "")[:120]))
    if not cuts:
        return []
    cuts.sort(key=lambda t: (t[0], t[1]))

    spans: List[Span] = []
    cursor = 0
    for s, e, label in cuts:
        if e < cursor:
            continue                      # fully covered already
        start = max(s, cursor)
        if start > cursor:                # gap: attach it to the previous span
            if spans:
                spans[-1].end = start - 1
            else:
                spans.append(Span(start=cursor, end=start - 1, label="preamble"))
        if e < start:
            continue
        spans.append(Span(start=start, end=e, label=label))
        cursor = e + 1
    if cursor < n:
        if spans:
            spans[-1].end = n - 1
        else:
            spans.append(Span(start=0, end=n - 1, label="whole trajectory"))
    return [sp for sp in spans if sp.end >= sp.start]


_ACTION_HINTS = (
    ("edit", re.compile(r"str_replace|create|insert|write_patch|apply_patch|sed -i", re.I)),
    ("test", re.compile(r"pytest|python -m|npm test|go test|make test|run_tests", re.I)),
    ("search", re.compile(r"\b(grep|find|search_|rg|ls )\b", re.I)),
    ("read", re.compile(r"\bview\b|\bcat\b|\bopen\b|str_replace_editor view", re.I)),
    ("submit", re.compile(r"\bsubmit\b|\bfinish\b|final answer", re.I)),
)


def _shape(step) -> str:
    text = step.content or ""
    for name, rx in _ACTION_HINTS:
        if rx.search(text):
            return name
    return "other"


def heuristic_spans(traj: Trajectory, target: int) -> List[Span]:
    """Deterministic fallback: cut where the action shape changes."""
    n = len(traj)
    max_len = max(2, round(n / max(1, target)) * 2)
    spans: List[Span] = []
    start = 0
    prev = _shape(traj.steps[0])
    for i in range(1, n):
        cur = _shape(traj.steps[i])
        if cur != prev or (i - start) >= max_len:
            spans.append(Span(start=start, end=i - 1, label=prev))
            start, prev = i, cur
    spans.append(Span(start=start, end=n - 1, label=prev))
    return spans


# ---------------------------------------------------------------------------
# Rule + model segmentation
# ---------------------------------------------------------------------------
#
# ``segment`` above hands a model a one-line-per-step digest of a 117-step
# trajectory and asks it to invent cut points. That is a global problem answered
# off local evidence, and it is harder than it needs to be, because most of the
# cuts are already legible in the text: an agent that stops editing and starts
# running tests has changed sub-goal, and that transition is a regex away.
#
# So the work is split. ``units.unit_chain`` does the mechanical part with zero
# calls -- maximal runs of the same operation type, each labelled with what it
# operated on -- producing a chain like
#
#     [0-3] observation README.md -> [4] plan -> [5-9] localization grep parser
#     -> [10-11] edit src/parse.py -> [12] test pytest tests/
#
# and the model does only the part that needs judgement: which consecutive
# units belong to one sub-goal. That is a grouping problem over ~20 typed items
# instead of a cutting problem over 117 lines, and the answer space is small
# enough that a malformed reply is cheap to repair.
#
# The second reason this exists is ordering, and it is the one that unblocks
# prior-conditioned compression. The chain is available BEFORE any compression
# call, so ``Span.digest`` gives a pre-compression reader something to read.
# ``span_prior`` used to read ``Span.summary``, which only exists once
# ``compress`` has written it -- that is what made "let the prior tell the
# compressor what to keep" circular.
UNIT_SPAN_PROMPT = """Below is a FAILED agent trajectory, already broken into \
OPERATION UNITS by a rule. Each unit is a run of consecutive steps doing the \
same kind of operation, and is shown as its step range, its operation type, \
and what it operated on.

TASK:
{task}

OPERATION CHAIN ({u} units over steps 0..{last}):
{chain}

Group CONSECUTIVE units into semantic spans. A span is one sub-goal: for
example "locate the failing module", "edit the parser and re-run the test",
"react to the failure". A sub-goal usually spans several units, because
pursuing one goal means observing, planning, editing and testing in turn.

Rules:
  - Group by unit number (0..{lastu}), not by step number.
  - Groups must be contiguous, non-overlapping, and cover every unit.
  - Start a new group where the agent's GOAL changes, not merely where the
    operation type changes -- the rule already cut on operation type, and most
    of those cuts are inside one sub-goal.
  - Aim for about {target} groups.
  - Label each group with a short phrase naming its sub-goal.

Return ONLY JSON:
{{"spans": [{{"from_unit": <int>, "to_unit": <int>, "label": "<short phrase>"}}]}}"""


def segment_units(traj: Trajectory, llm: LLM, graph: Optional[FlowGraph] = None,
                  target_spans: Optional[int] = None,
                  max_units: int = 60, chain_chars: int = 40000) -> List[Span]:
    """Segment via a rule-built operation chain that a model then groups.

    Same contract as ``segment``: a full, non-overlapping cover of 0..n-1. The
    spans additionally carry ``role`` and ``digest``, which nothing requires but
    the prior-conditioned compressor reads.
    """
    n = len(traj)
    if n <= 2:
        return [Span(start=0, end=n - 1, label="whole trajectory",
                     role="observation", digest="")]

    chain = units_mod.unit_chain(traj, max_units=max_units)
    target = target_spans or max(4, min(30, round(n / 6)))
    if len(chain) <= 1:
        return [_span_of(chain, 0, 0, "whole trajectory")]

    # Fewer units than groups asked for: the grouping question is already
    # answered and a call would only add a chance to get it wrong.
    if len(chain) <= target:
        return [_span_of(chain, i, i, "") for i in range(len(chain))]

    rendered = units_mod.render_chain(chain)
    if len(rendered) > chain_chars:
        rendered = clip_middle(rendered, chain_chars)

    obj = llm.generate_json(
        UNIT_SPAN_PROMPT.format(task=clip_middle(traj.task_description, 1500),
                                u=len(chain), last=n - 1, lastu=len(chain) - 1,
                                chain=rendered, target=target),
        max_tokens=3000, system=SPAN_SYSTEM, default={})
    raw = obj if isinstance(obj, list) else (obj or {}).get("spans")
    spans = _coerce_unit_groups(raw, chain)
    if not spans:
        logger.info("unit grouping fell back to rule merge for %s", traj.task_id)
        merged = units_mod._merge_shortest(chain, target)
        spans = []
        for m in merged:
            lo = next(j for j, u in enumerate(chain) if u.end >= m.start)
            hi = max(j for j, u in enumerate(chain) if u.start <= m.end)
            spans.append(_span_of(chain, lo, hi, ""))
    return spans


def _dominant_role(chain, lo: int, hi: int) -> str:
    """The role covering the most STEPS in units lo..hi.

    Steps rather than units, because a span of one 40-step reading unit and
    three 1-step planning units is a reading span. Ties go to the earlier unit,
    so the answer does not depend on dict ordering.
    """
    tally = {}
    for u in chain[lo:hi + 1]:
        tally[u.role] = tally.get(u.role, 0) + len(u)
    if not tally:
        return ""
    best = max(tally.values())
    for u in chain[lo:hi + 1]:
        if tally.get(u.role) == best:
            return u.role
    return ""


def _span_of(chain, lo: int, hi: int, label: str) -> Span:
    lo = max(0, min(lo, len(chain) - 1))
    hi = max(lo, min(hi, len(chain) - 1))
    start, end = chain[lo].start, chain[hi].end
    role = _dominant_role(chain, lo, hi)
    digest = units_mod.chain_digest(chain, start, end)
    if not label:
        # No model label: the rule digest already names the operations and
        # their targets, so a truncation of it beats a bare "span 3".
        label = digest[:120]
    return Span(start=start, end=end, label=label, role=role, digest=digest)


def _coerce_unit_groups(raw, chain) -> List[Span]:
    """Validate unit groups and repair them into a full cover of the chain.

    The repair is the same shape as ``_coerce_spans`` -- sort, drop what is
    already covered, extend the previous group over any gap -- but it operates
    on unit numbers, where an off-by-one costs a whole unit rather than a step,
    so gaps are attached rather than dropped.
    """
    if not isinstance(raw, list) or not raw or not chain:
        return []
    m = len(chain)
    groups = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        a = item.get("from_unit", item.get("start"))
        b = item.get("to_unit", item.get("end"))
        try:
            s, e = int(a), int(b)
        except (TypeError, ValueError):
            continue
        if s > e:
            s, e = e, s
        s, e = max(0, min(s, m - 1)), max(0, min(e, m - 1))
        groups.append((s, e, str(item.get("label") or "")[:120]))
    if not groups:
        return []
    groups.sort(key=lambda t: (t[0], t[1]))

    spans: List[Span] = []
    cursor = 0
    for s, e, label in groups:
        if e < cursor:
            continue
        start = max(s, cursor)
        if start > cursor:
            if spans:                       # absorb the gap leftwards
                spans[-1] = _span_of(chain, _unit_of(chain, spans[-1].start),
                                     start - 1, spans[-1].label)
            else:
                spans.append(_span_of(chain, cursor, start - 1, "preamble"))
        spans.append(_span_of(chain, start, e, label))
        cursor = e + 1
    if cursor < m:
        if spans:
            spans[-1] = _span_of(chain, _unit_of(chain, spans[-1].start),
                                 m - 1, spans[-1].label)
        else:
            spans.append(_span_of(chain, 0, m - 1, "whole trajectory"))
    return [sp for sp in spans if sp.end >= sp.start]


def _unit_of(chain, step: int) -> int:
    for j, u in enumerate(chain):
        if u.start <= step <= u.end:
            return j
    return 0


# ---------------------------------------------------------------------------
def assign_tiers(spans: List[Span], graph: FlowGraph,
                 weak_ratio: float = 0.0) -> List[Span]:
    """Mark each span WEAK if it contains any step on a traced flow.

    Steps on a data or control flow carry the failure; their spans keep most
    of their content. Spans on neither flow are condensed hard. ``weak_ratio``
    can require more than one flow step before a span counts as weak, useful
    for very long spans.
    """
    on_flow = graph.on_flow()
    for sp in spans:
        idxs = set(sp.indices)
        hits = len(idxs & on_flow)
        if weak_ratio <= 0:
            sp.tier = WEAK if hits > 0 else STRONG
        else:
            sp.tier = WEAK if hits >= max(1, int(len(sp) * weak_ratio)) else STRONG
    return spans


def flow_step_sets(graph: FlowGraph):
    """Return (data_steps, control_steps, anchors) for prompt annotation."""
    return graph.data_flow_steps(), graph.control_flow_steps(), set(graph.anchors)
