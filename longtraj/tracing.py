"""Flow construction: anchor on the observable failure, then relate the steps.

Three stages, the first two LLM-driven:

1. ``locate_anchors`` — find where the failure becomes *visible*. For coding
   agents that is typically the submitted patch or the step whose output
   shows the validation error. This is a symptom, not a cause: the anchor is
   almost never the critical step.

2. ``build_forward`` — sweep the trajectory head to tail and, for every step,
   ask which earlier steps it depends on, along two complementary relations:

     data flow    : how information reaching this step was built by earlier
                    steps and requests, and where it was introduced,
                    dropped, or transformed on the way.
     control flow : why the agent moved from the earlier step to this one.

   The two are complementary because a failure can come from information
   propagating wrongly across steps, or from the transition logic between
   them — and they are kept as *separate* forests, because a step reached by
   alternating between the relations sits on neither path.

   Nothing about a step's parents depends on what comes after it, so the
   sweep splits into windows that all run at once: one wave of calls instead
   of the chain of rounds a backward walk needs.

3. Painting (``FlowGraph.paint``, no LLM) — walk up from each anchor through
   each forest independently. Everything on a root→anchor path is on the
   flow; branches that never reach an anchor belong to the trajectory but not
   to the failure, and are compressed hard.

To keep prompts bounded on 700-step trajectories, every call reads a compact
one-line-per-step index plus full text for its own window only.
"""
from __future__ import annotations

import heapq
import logging
import re
from typing import Any, Dict, List, Optional, Set

from .llm import LLM, map_concurrent
from .schema import (CONTROL_FLOW, DATA_FLOW, FlowEdge, FlowGraph, Step,
                     Trajectory, clip_middle)

logger = logging.getLogger(__name__)

# A single step naming more parents than this in one relation is the model
# losing the thread, not a real dependency fan-in: the observed maximum is 4.
# Truncating keeps one bad answer from painting the whole trajectory, and it
# is logged, so it is never a silent cap.
MAX_PARENTS_PER_STEP = 12

# Tokens a reasoning model burns before emitting the first character of JSON.
# It scales with the prompt, not with how many steps were asked about, so it
# is a flat term in the window budget rather than a per-step one.
REASONING_HEADROOM = 4000

TRACE_SYSTEM = (
    "You are analysing a FAILED agent trajectory to find how the failure "
    "propagated. You reason about dependencies between steps, never about "
    "which step is 'the mistake'. Answer strictly in JSON."
)

# ---------------------------------------------------------------------------
# Trajectory index: one compact line per step
# ---------------------------------------------------------------------------
def build_index(traj: Trajectory, per_step: int = 220) -> str:
    """A skimmable digest of the whole trajectory, one line per step."""
    return "\n".join(index_lines(traj, per_step))


def index_lines(traj: Trajectory, per_step: int = 220) -> List[str]:
    """One digest line per step, so callers can take a prefix of them."""
    lines = []
    for s in traj.steps:
        who = s.name or s.role
        body = " ".join((s.content or "").split())
        if len(body) > per_step:
            body = body[:per_step] + "…"
        lines.append("[%d|%s] %s" % (s.index, who, body))
    return lines


def _fit(text: str, limit: int) -> str:
    """Clip a digest to ``limit`` chars, keeping the head and the tail."""
    if len(text) <= limit:
        return text
    return (text[: limit // 3] + "\n...[middle of digest omitted]...\n"
            + text[-(2 * limit // 3):])


def render_steps(traj: Trajectory, idxs, budget: int = 3000) -> str:
    """Full-ish text for a handful of specific steps."""
    out = []
    for i in sorted(set(idxs)):
        if 0 <= i < len(traj.steps):
            out.append(traj.steps[i].render(budget))
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Stage 1 — anchor on the visible failure
# ---------------------------------------------------------------------------
ANCHOR_PROMPT = """Below is a FAILED agent trajectory, shown as a one-line digest per step.

TASK THE AGENT WAS GIVEN:
{task}

{symptom_block}TRAJECTORY DIGEST ({n} steps, indices 0..{last}):
{index}

Identify where the failure becomes OBSERVABLE — the symptom, not the cause.
Prefer, in this order:
  1. the step that submits or finalises the answer/patch,
  2. the step whose output shows the validation error, failing test, or
     rejected result,
  3. the last step that reports concrete progress before the trajectory ends.

Pick 1 to {k} steps. These are endpoints of the failure, so late steps are
expected. Do NOT try to identify the underlying mistake.

Also list the concrete artefacts the failure is about — file paths, symbol or
function names, test identifiers, field or parameter names, entity names.
Copy them verbatim from the trajectory.

Return ONLY JSON:
{{
  "anchors": [<step indices>],
  "failure_summary": "<2-3 sentences: what visibly went wrong>",
  "failure_entities": ["<verbatim artefact>", ...]
}}"""


def locate_anchors(traj: Trajectory, llm: LLM, max_anchors: int = 3,
                   index_chars: int = 60000) -> FlowGraph:
    """Stage 1: find the steps where the failure is visible."""
    index = build_index(traj)
    if len(index) > index_chars:
        # keep the head (task setup) and the tail (where failures surface)
        index = _fit(index, index_chars)

    symptom = ""
    fc = traj.meta.get("failed_test_code")
    fe = traj.meta.get("failed_test_error")
    if fc or fe:
        symptom = ("OBSERVED FAILURE REPORTED BY THE HARNESS:\n"
                   "failing test:\n%s\nerror:\n%s\n\n"
                   % (clip_middle(str(fc or ""), 1500),
                      clip_middle(str(fe or ""), 1500)))

    prompt = ANCHOR_PROMPT.format(
        task=clip_middle(traj.task_description, 3000),
        symptom_block=symptom, n=len(traj), last=len(traj) - 1,
        index=index, k=max_anchors)

    obj = llm.generate_obj(prompt, max_tokens=1500, system=TRACE_SYSTEM)
    anchors = _clean_indices(obj.get("anchors"), len(traj))
    if not anchors:
        # fall back to the last attributable steps: the failure surfaces late
        anchors = [s.index for s in traj.steps if s.attributable][-max_anchors:]

    ents = [str(x) for x in (obj.get("failure_entities") or [])
            if isinstance(x, (str, int, float)) and str(x).strip()][:40]
    return FlowGraph(anchors=anchors, anchors_located=list(anchors),
                     failure_summary=str(obj.get("failure_summary") or "")[:2000],
                     failure_entities=ents)


# ---------------------------------------------------------------------------
# Stage 1a' — the same anchors, by keyword, at zero calls
# ---------------------------------------------------------------------------
#
# ``ANCHOR_PROMPT`` asks a model to apply a three-line preference order:
#
#   1. the step that submits or finalises the answer,
#   2. the step whose output shows the error / failing test / rejection,
#   3. the last step reporting concrete progress.
#
# Every clause of that order is a surface pattern. (1) is a submit call, (2) is
# a traceback or a pytest FAILED line, (3) is "the last one". A model is not
# needed to apply it, and on the stored flows it does not appear to be adding
# judgement: the located anchors sit at relative position **median 0.982**, i.e.
# the model reads a 60k-char digest and answers "the end" nearly every time.
#
# So this is the rule version. Two design points that are not obvious:
#
# RECENCY IS A TERM, NOT A FILTER. A traceback is only a symptom if it is the
# one the run died on. Agents routinely reproduce the bug on purpose at step 4,
# and that step is a cause-adjacent step, not the anchor. Weighting position
# instead of windowing keeps an early step reachable when it is the ONLY thing
# in the trajectory that looks like a failure, which is what a fallback is for.
#
# PRECISION OVER RECALL IN THE PATTERNS. Every pattern here is distinctive of
# an error being *reported* rather than merely mentioned: ``Traceback (most
# recent call last)``, pytest's ``E   `` continuation prefix, a line that STARTS
# with ``error:``. A bare ``\berror\b`` would fire on every step that views a
# file containing the word, which on these corpora is most of them.
#
# This replaces the anchor INDICES only. ``failure_summary`` and
# ``failure_entities`` still come from the call, because they feed six
# downstream prompts and swapping them too would make the A/B multi-variable.
RULE_ANCHOR_RECENCY = 4.0    # score contributed by being at the very end
RULE_ANCHOR_SYMPTOM = 1.5    # per distinct failure pattern, capped
RULE_ANCHOR_FINAL = 3.0      # for a submit / finalise call
RULE_ANCHOR_MAX_HITS = 3     # cap on the symptom term: 9 tracebacks != 9x worse
RULE_ANCHOR_TAIL = 2000      # chars of each step to scan (evidence lands last)

_RX_SYMPTOM = (
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("pytest_e", re.compile(r"^E   \S", re.M)),
    ("pytest_failed", re.compile(r"^FAILED\s+\S|=+ (?:FAILURES|ERRORS) =+"
                                 r"|\b\d+ failed\b", re.M)),
    ("exception", re.compile(
        r"\b(?:AssertionError|SyntaxError|IndentationError|ImportError"
        r"|ModuleNotFoundError|NameError|TypeError|AttributeError|KeyError"
        r"|ValueError|RuntimeError|OSError|UnboundLocalError)\b")),
    ("exit_code", re.compile(r"\bexit (?:code|status)[:= ]+[1-9]"
                             r"|\bnon-?zero exit\b")),
    ("shell_error", re.compile(r"command not found|No such file or directory"
                               r"|^fatal: |Permission denied", re.M)),
    ("error_line", re.compile(r"^\s*(?:error|ERROR)[: ]", re.M)),
)

_RX_FINAL = re.compile(
    r"<function=submit|\bsubmit(?:_answer|_patch)?\b|\bfinish\b"
    r"|final answer|task (?:is )?complete|I(?:'ve| have) completed", re.I)


def anchor_evidence(text: str) -> Dict[str, Any]:
    """Which failure patterns fire on one step, and does it finalise?"""
    tail = (text or "")[-RULE_ANCHOR_TAIL:]
    hits = [name for name, rx in _RX_SYMPTOM if rx.search(tail)]
    return {"hits": hits, "final": bool(_RX_FINAL.search(tail))}


def keyword_anchors(traj: Trajectory, k: int = 3,
                    detail: bool = False):
    """Pick anchors by failure keyword + recency. No model call.

    Returns the anchor indices, ascending. With ``detail=True`` also returns the
    per-step scores, so a probe can ask how often the rule and the model agree
    without re-deriving the scoring.
    """
    steps = [s for s in traj.steps if s.attributable]
    if not steps:
        steps = list(traj.steps)
    if not steps:
        return ([], {}) if detail else []
    last = max(s.index for s in steps)
    span = float(last) or 1.0

    scored = []
    for s in steps:
        ev = anchor_evidence(s.content or "")
        score = RULE_ANCHOR_RECENCY * (s.index / span)
        score += RULE_ANCHOR_SYMPTOM * min(len(ev["hits"]), RULE_ANCHOR_MAX_HITS)
        if ev["final"]:
            score += RULE_ANCHOR_FINAL
        scored.append({"step": s.index, "score": round(score, 4),
                       "hits": ev["hits"], "final": ev["final"]})

    # Highest score first; later steps break ties, since a symptom that recurs
    # is anchored at its last appearance -- that is the one the run died on.
    order = sorted(scored, key=lambda r: (-r["score"], -r["step"]))
    picked: List[int] = []
    for row in order:
        if len(picked) >= max(1, k):
            break
        if any(abs(row["step"] - p) < ANCHOR_GAP for p in picked):
            continue
        picked.append(row["step"])
    picked.sort()
    if detail:
        return picked, {"scores": scored[-40:], "picked": list(picked)}
    return picked


# ---------------------------------------------------------------------------
# Stage 1b — which of the late steps actually went wrong
# ---------------------------------------------------------------------------
#
# ``locate_anchors`` picks 1-3 steps off a ONE-LINE-PER-STEP digest. Two things
# are wrong with that as a source of paint roots, and this stage fixes both:
# it reads the steps' RAW content, and it asks for an independent verdict per
# step rather than a global pick of three.
#
# WHY THE WINDOW IS 20 AND NOT 10. Measured on the stored flows, anchoring on
# every eligible step in the last W, recall / coverage / enrichment:
#
#     current anchors  0.4884 / 0.1596 / 3.06x
#     W=10, all        0.6628 / 0.2360 / 2.81x   <- below the current enrichment
#     W=10, capped     0.5233 / 0.1595 / 3.28x
#     W=20, capped     0.5814 / 0.1695 / 3.43x
#     no window        0.5814 / 0.1688 / 3.44x
#
# At W>=20 the window stops binding, so 20 buys the full effect while keeping
# the prompt to 20 steps of raw text. Going NARROWER is what costs: the located
# anchors already sit at position median 0.982, so a 10-step window re-commits
# to exactly the bunching this stage exists to break, and the reference step is
# almost never there anyway (6 of 86 cases on SWE-Bench-Pro; the median one is
# 42 steps upstream of the nearest anchor). An anchor earns its place by being
# a causal DESCENDANT of the mistake, not by being late.
ANCHOR_WINDOW = 20      # judge only the last this-many steps
ANCHOR_CAP_RATIO = 1.25  # cap = this x what the located anchors paint
ANCHOR_GAP = 3          # no two anchors closer than this
ANCHOR_STEP_CLIP = 3000  # per-step clip inside the judging prompt
ANCHOR_PROMPT_CLIP = 40000   # whole-prompt cap on the step bodies

ERROR_STEP_PROMPT = """Below are the LAST steps of a FAILED agent trajectory, \
shown in full.

TASK THE AGENT WAS GIVEN:
{task}

WHAT VISIBLY WENT WRONG:
{summary}

ARTEFACTS THE FAILURE IS ABOUT:
{entities}

{symptom_block}STEPS (each is verbatim; indices are the real step numbers):
{steps}

Judge EVERY step listed above, one at a time and independently. For each:

  "error"   - does THIS step itself show something going wrong? A command that
              failed, an exception or traceback, a test that did not pass, an
              assertion or validation rejecting the result, output that
              contradicts what the step was trying to do, or a decision the
              step states it is making on visibly wrong information.
              Routine successful work is NOT an error.
  "related" - does this step touch any of the artefacts listed above, or the
              failure described above?

Judge only what is visible in the step. Do not guess at the underlying mistake
and do not reason about which step CAUSED the failure -- these are endpoints,
and something upstream is expected to be the real cause.

Return ONLY JSON, one entry per step you were shown:
{{"steps": [{{"index": <int>, "error": true|false, "related": true|false,
             "why": "<short phrase>"}}]}}"""


def _judging_block(traj: Trajectory, cands: List[int]) -> str:
    """The raw step bodies, in trajectory order, clipped per step then whole."""
    parts = []
    for i in sorted(cands):
        st = traj.steps[i]
        parts.append("step %d [%s]:\n%s"
                     % (i, st.name or st.role,
                        clip_middle(st.content or "", ANCHOR_STEP_CLIP)))
    return clip_middle("\n\n".join(parts), ANCHOR_PROMPT_CLIP)


def judge_error_steps(traj: Trajectory, graph: FlowGraph, llm: LLM,
                      window: int = ANCHOR_WINDOW,
                      k: int = 3, gap: int = ANCHOR_GAP,
                      ratio: float = ANCHOR_CAP_RATIO) -> List[int]:
    """Re-anchor on the late steps a model judges to have actually gone wrong.

    One LLM call. Returns [] when there is nothing better than what
    ``locate_anchors`` already chose -- the caller keeps ``anchors_located`` in
    that case, so this stage can only replace anchors, never delete them.

    The verdicts nominate; ``_cap_pick`` spends the budget. Steps judged
    ``error`` are preferred; if none are, steps judged ``related`` are used
    instead, so a model that finds no outright failure in the tail still
    contributes something. Candidates are ordered by in-degree, so when the cap
    binds it drops the judged steps that reach least.

    GROUND TRUTH NEVER REACHES THIS. The signature takes (traj, graph, llm) and
    nothing else; the prompt is built from the task description, the graph's own
    ``failure_summary`` / ``failure_entities`` (which this pipeline produced),
    the harness's failing-test text where the corpus has it, and raw step
    bodies. On SWE-Bench-Pro there is no harness failing-test text at all --
    ``sp_07_prsym_noop`` proved that by being a byte-identical no-op -- so on
    that corpus the symptom block is simply absent.
    """
    n = max(1, len(traj))
    by = _parent_index(graph)
    located = [a for a in (graph.anchors_located or graph.anchors) if a >= 0]
    if not located:
        return []

    cands = _anchor_candidates(traj, by, window=window)
    if not cands:
        return []

    symptom = ""
    fc = traj.meta.get("failed_test_code")
    fe = traj.meta.get("failed_test_error")
    if fc or fe:
        symptom = ("OBSERVED FAILURE REPORTED BY THE HARNESS:\n"
                   "failing test:\n%s\nerror:\n%s\n\n"
                   % (clip_middle(str(fc or ""), 1500),
                      clip_middle(str(fe or ""), 1500)))

    prompt = ERROR_STEP_PROMPT.format(
        task=clip_middle(traj.task_description, 3000),
        summary=(graph.failure_summary or "(not available)"),
        entities=", ".join(graph.failure_entities[:40]) or "(none identified)",
        symptom_block=symptom,
        steps=_judging_block(traj, cands))

    try:
        obj = llm.generate_obj(prompt, max_tokens=2000, system=TRACE_SYSTEM)
    except Exception:                                       # noqa: BLE001
        logger.exception("%s: error-step judging failed", traj.task_id)
        return []

    eligible = set(cands)
    errored, related = [], []
    for row in (obj.get("steps") or []):
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if i not in eligible:
            continue          # out of range, or a step never shown
        if row.get("error") is True:
            errored.append(i)
        elif row.get("related") is True:
            related.append(i)

    picked = errored or related
    if not picked:
        logger.info("%s: no step in the last %d judged erroneous, keeping the "
                    "located anchors", traj.task_id, window)
        return []

    order = [c for c in cands if c in set(picked)]   # in-degree order, kept
    budget = ratio * (len(_paint_from(by, located)) / float(n))
    return _cap_pick(by, n, order, budget, k, gap)


def _parent_index(graph: FlowGraph) -> Dict[str, Dict[int, List[int]]]:
    """dst -> parents, one map per relation. Same shape painting walks."""
    by: Dict[str, Dict[int, List[int]]] = {DATA_FLOW: {}, CONTROL_FLOW: {}}
    for e in graph.edges:
        if e.kind in by:
            by[e.kind].setdefault(e.dst, []).append(e.src)
    return by


def _paint_from(by: Dict[str, Dict[int, List[int]]], roots) -> Set[int]:
    """What ``roots`` would paint under the SEPARATE reading.

    Deliberately mirrors ``FlowGraph.on_flow``'s default branch: each relation's
    forest walked on its own, then unioned. Re-selecting anchors must be scored
    with the same reading the run will use, or the budget below is measuring
    something the pipeline never does.
    """
    out: Set[int] = set()
    for kind in (DATA_FLOW, CONTROL_FLOW):
        par = by[kind]
        seen = {r for r in roots if r >= 0}
        stack = list(seen)
        while stack:
            for p in par.get(stack.pop(), ()):
                if p >= 0 and p not in seen:
                    seen.add(p)
                    stack.append(p)
        out |= seen
    return out


def failure_anchors(traj: Trajectory, graph: FlowGraph, k: int = 3,
                    gap: int = ANCHOR_GAP,
                    ratio: float = ANCHOR_CAP_RATIO) -> List[int]:
    """The failure-anchor set: the located anchors PLUS the confluence points.

    ``locate_anchors`` names the steps where the failure is VISIBLE; those are
    symptoms, and a symptom's ancestor subtree is small. ``reselect_anchors``
    names the steps where the most dependencies MEET; those are confluences, and
    a confluence's subtree is large. Measured on the stored flows, neither set
    contains the other: taking both is worth more than either alone.

        SP (n=86)   located 0.4884 / 0.1596 / 3.06x
                    indeg   0.5814 / 0.1688 / 3.44x
                    union   0.6047 / 0.2182 / 2.77x
        TA (n=98)   located 0.7041 / 0.4125 / 1.71x
                    indeg   0.7143 / 0.4296 / 1.66x
                    union   0.7959 / 0.4843 / 1.64x

    READ THOSE NUMBERS HONESTLY. The union's painted set is a SUPERSET of the
    located anchors', so recall cannot go down and "wins N, loses 0" on both
    corpora is arithmetic, not evidence. The real trade is the one the
    enrichment column shows: recall +0.116 on SP for +37% coverage, which is a
    LOSS by the standing enrichment rule. This set ships because recall and the
    compression ratio were the stated targets for this round, not because it
    beat 3.06x. ``reselect_anchors`` alone remains the enrichment-optimal rule.

    Costs no LLM calls. See ``judge_error_steps`` for the third rule tried here
    -- asking a model to read the tail and name the erroneous steps -- which
    measured BELOW baseline recall (0.4651) and is kept only as an ablation.
    """
    located = [a for a in (graph.anchors_located or graph.anchors) if a >= 0]
    extra = reselect_anchors(traj, graph, ratio=ratio, k=k, gap=gap)
    return sorted(set(located) | set(extra))


def reselect_anchors(traj: Trajectory, graph: FlowGraph, ratio: float,
                     k: int = 3, gap: int = 3) -> List[int]:
    """Re-anchor on the steps with the most parent edges, under a coverage cap.

    Costs no LLM calls, and needs no re-tracing: ``FORWARD_PROMPT`` never sees
    the anchors, so the parent forests are anchor-independent and this is a pure
    reading change over a stored flow.

    WHY ANCHORS AND NOT THE GRAPH. Measured on the stored SWE-Bench-Pro flows,
    37 of the 44 missed reference steps are reachable from SOME step in the
    graph the anchors simply do not sit on; only 8 need an edge that was never
    traced. The anchor-placement ceiling is recall 0.9070 at coverage 0.063,
    against 0.4884 at 0.1596 today. The cause is that the located anchors are
    three in name and one in effect: their normalised spread has median 0.056
    and 29 of 86 cases place every anchor within 2 steps of the others, all
    bunched at the very end (median position 0.982).

    THE RULE. Candidates are attributable steps that have at least one parent,
    scored by in-degree (ties to the later step), taken greedily, skipping any
    within ``gap`` of one already taken. Of eight label-free families screened
    in probe/anchor_rules.py this is the only one that moves recall on BOTH
    corpora; the entity families looked better on SWE-Bench-Pro (0.5465) and
    then came in BELOW baseline on TrajAudit (0.6531 vs 0.7041), which is what
    the holdout is for.

    THE CAP IS THE WHOLE POINT. Uncapped, in-degree buys recall with coverage at
    roughly one-for-one -- enrichment 3.06x -> 3.13x -- and by the standing rule
    ("recall bought by covering more of the run is not recall") that is not a
    win. ``ratio`` caps each case's painted fraction at a multiple of what its
    OWN located anchors paint, which is what makes the exchange favourable
    rather than proportional. Per-case self-calibration matters: an absolute cap
    at matched coverage wins 17 and loses 9, this wins 10 and loses 2.

    Measured on SWE-Bench-Pro (n=86, k=3, gap=3), recall / coverage /
    enrichment: 0.4884 / 0.1596 / 3.06x located; 0.5581 / 0.1593 / 3.50x at
    ratio 1.15; 0.5814 / 0.1688 / 3.44x at 1.25 (McNemar p=0.0386); 0.6279 /
    0.1962 / 3.20x at 2.00. Enrichment peaks near 1.15 and falls after, so this
    axis has an interior optimum -- unlike the paint axis, where all eight
    variants in probe/paint_frontier.py lose enrichment monotonically.

    ON TRAJAUDIT THIS IS A NULL, stated here so nobody reads the numbers above
    as general: ratio 1.25 gives 0.7143 / 0.4296 / 1.66x, wins 9 loses 8,
    p=1.0, and enrichment goes DOWN from 1.71x. TrajAudit only moves at ratio
    1.5-2.0, where enrichment falls further.
    """
    located = [a for a in (graph.anchors_located or graph.anchors) if a >= 0]
    if not located:
        return list(located)

    n = max(1, len(traj))
    by = _parent_index(graph)
    budget = ratio * (len(_paint_from(by, located)) / float(n))

    cands = _anchor_candidates(traj, by)
    if not cands:
        return list(located)
    return _cap_pick(by, n, cands, budget, k, gap)


def _in_degree(by: Dict[str, Dict[int, List[int]]], s: int) -> int:
    """How many parents a step names across both relations."""
    return len(by[DATA_FLOW].get(s, ())) + len(by[CONTROL_FLOW].get(s, ()))


def _anchor_candidates(traj: Trajectory,
                       by: Dict[str, Dict[int, List[int]]],
                       window: int = 0) -> List[int]:
    """Steps worth anchoring on, best first.

    A step with no parent paints only itself, so it spends coverage budget and
    buys no reach -- those are dropped before anything else looks at them.
    ``window`` > 0 restricts to the last ``window`` steps of the trajectory.

    Ordered by in-degree descending, ties to the later step. That order is what
    the cap consumes: when the budget binds it is the LOW in-degree steps that
    fall off the end, which is the behaviour probe/anchor_rules.py measured.
    """
    lo = max(0, len(traj) - window) if window > 0 else 0
    cands = [s.index for s in traj.steps if s.attributable and s.index >= lo]
    cands = [s for s in cands
             if by[DATA_FLOW].get(s) or by[CONTROL_FLOW].get(s)]
    cands.sort(key=lambda s: (-_in_degree(by, s), -s))
    return cands


def _cap_pick(by: Dict[str, Dict[int, List[int]]], n: int, cands: List[int],
              budget: float, k: int, gap: int) -> List[int]:
    """Greedily take anchors from ``cands`` under a painted-fraction cap.

    ``cands`` must already be in preference order. Takes at most ``k``, never
    two within ``gap`` steps of each other, and never lets the union paint more
    than ``budget`` of the trajectory.

    THE CAP IS WHY THIS AXIS WINS. Uncapped, better anchors buy recall with
    coverage at roughly one-for-one (enrichment 3.06x -> 3.13x on SWE-Bench-Pro)
    and by the standing rule -- recall bought by covering more of the run is not
    recall -- that is not a win at all.

    Shared by ``reselect_anchors`` (candidates = every eligible step) and
    ``judge_error_steps`` (candidates = the steps a model judged erroneous), so
    the two rules differ only in what they nominate, never in how the budget is
    spent.
    """
    chosen: List[int] = []
    painted: Set[int] = set()
    for c in cands:
        if any(abs(c - o) < gap for o in chosen):
            continue
        nxt = painted | _paint_from(by, [c])
        # The FIRST anchor is capped too. Exempting it (as the probe's
        # non-strict variant did) is what let coverage drift up despite a
        # per-case budget, so the strict form is the one that ships.
        if len(nxt) / float(n) > budget:
            continue
        chosen.append(c)
        painted = nxt
        if len(chosen) >= k:
            break

    if not chosen:
        # Every candidate overshoots on its own. Take the one that paints least
        # -- ties to the highest in-degree, since ``cands`` is already in that
        # order -- rather than returning nothing and leaving painting rootless.
        chosen = [min(cands, key=lambda s: len(_paint_from(by, [s])))]
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Stage 2 — forward construction of the two dependency forests
# ---------------------------------------------------------------------------
FORWARD_PROMPT = """You are reconstructing how information and decisions flowed
through a FAILED agent trajectory.

TASK:
{task}

TRAJECTORY DIGEST (indices 0..{last}, one line per step):
{index}

STEPS TO ANALYSE NOW (full text, indices {lo}..{hi}):
{window_text}

For EACH step shown in full above, name the earlier steps it depends on. Two
SEPARATE relations — keep them apart, do not merge them:

DATA FLOW — where the information this step uses came from.
  Report a parent when an earlier step INTRODUCED information this step relies
  on, or TRANSFORMED it, or DROPPED information this step needed.
  In "entity", name the concrete thing carried: a file path, a value, a
  function name, a field name, a test id, an observation.

CONTROL FLOW — why the agent arrived at this step, when something OTHER than
mere sequence explains it. Report a parent only when an earlier step:
    - set the plan or goal that this step is carrying out,
    - failed, errored, or returned an unexpected result that this step is
      reacting to,
    - is what this step retries, reverses, abandons, or works around,
    - made a choice between alternatives that determined this step happens.

  CRITICAL: "the previous step finished, so this one ran next" is NOT control
  flow. Neither is an assistant step followed by the output of the command it
  just issued. Those are plain sequence, and reporting them turns the whole
  trajectory into one chain in which every step is an ancestor of every later
  step, which is useless. Most steps have NO control parent. A control parent
  is usually several steps back — the decision or the failure being answered,
  not the line above.

Rules:
  - src MUST be strictly less than dst. Parents may be ANY earlier step,
    including steps before index {lo} — use the digest to refer to them.
  - Report every parent you can justify from the text, and no more. Most steps
    genuinely have one or two data parents; do not pad the list, and do not
    force a parent onto a step that has none.
  - A step has NO parent in a relation when it starts something new: its
    information entered from outside the trajectory (the task statement, the
    first read of a file, the raw output of a command, the environment), or it
    states an initial plan that no earlier step triggered. Omit it, or give it
    an empty list. These are the roots of the forest.
  - Do NOT judge which step is the mistake. Only report dependencies.

Return ONLY JSON:
{{
  "edges": [
    {{"src": <int>, "dst": <int>, "kind": "data"|"control",
      "entity": "<what is carried, verbatim if possible>",
      "rationale": "<one short clause>"}}
  ]
}}"""


# The density variant. Measured motivation: with the prompt above, each
# ``(dst, kind)`` group comes back with a mean of 1.21 parents (7594 of 8948
# groups hold exactly one) and 35% of edges span a single step. Out-degree is
# conserved, so a step only reaches out-degree 6 if six later steps each pick
# it as their sole parent -- winner-take-all, and the winner is usually the
# most recent mention rather than the step that introduced the value. Since
# the shipped SP arm decides its answer by out-degree, that undercount is the
# mechanism behind the 40 cases whose reference step sits at out-degree <=2.
#
# Only the DATA FLOW half is touched. CONTROL FLOW keeps its wording to the
# byte: that relation already collapsed into the step sequence once (86% of
# its edges were dst-src==1 before the current phrasing), and inviting more
# control parents is the known way to bring that back.
FORWARD_PROMPT_ORIGIN = """You are reconstructing how information and decisions
flowed through a FAILED agent trajectory.

TASK:
{task}

TRAJECTORY DIGEST (indices 0..{last}, one line per step):
{index}

STEPS TO ANALYSE NOW (full text, indices {lo}..{hi}):
{window_text}

For EACH step shown in full above, name the earlier steps it depends on. Two
SEPARATE relations — keep them apart, do not merge them:

DATA FLOW — where the information this step uses came from.
  Report a parent when an earlier step INTRODUCED information this step relies
  on, or TRANSFORMED it, or DROPPED information this step needed.
  In "entity", name the concrete thing carried: a file path, a value, a
  function name, a field name, a test id, an observation.

  TRACE EACH PIECE BACK TO WHERE IT ENTERED. Information usually reaches a
  step through intermediaries: step 12 reads a file, step 40 quotes part of
  it, step 41 acts on the quote. The parent of step 41 is NOT only step 40.
  Report BOTH — step 40, which relayed it, AND step 12, which introduced it —
  as separate data edges with the same entity. Naming only the nearest mention
  hides the step that actually put the value into play, which is the one that
  matters here. Whenever you are about to cite the immediately preceding step,
  stop and ask where that step got it, and cite that step too.

  ONE PARENT PER PIECE OF INFORMATION. A step that combines a file path, an
  error message and a chosen value is using three pieces, so it has at least
  three data parents. Four or five is normal for a step that acts on an
  accumulated context; one is right only for a step that uses a single thing.

CONTROL FLOW — why the agent arrived at this step, when something OTHER than
mere sequence explains it. Report a parent only when an earlier step:
    - set the plan or goal that this step is carrying out,
    - failed, errored, or returned an unexpected result that this step is
      reacting to,
    - is what this step retries, reverses, abandons, or works around,
    - made a choice between alternatives that determined this step happens.

  CRITICAL: "the previous step finished, so this one ran next" is NOT control
  flow. Neither is an assistant step followed by the output of the command it
  just issued. Those are plain sequence, and reporting them turns the whole
  trajectory into one chain in which every step is an ancestor of every later
  step, which is useless. Most steps have NO control parent. A control parent
  is usually several steps back — the decision or the failure being answered,
  not the line above.

Rules:
  - src MUST be strictly less than dst. Parents may be ANY earlier step,
    including steps before index {lo} — use the digest to refer to them.
  - Report every parent you can justify from the text. Do not invent a link
    you cannot point at, and do not force a parent onto a step that has none.
  - A step has NO parent in a relation when it starts something new: its
    information entered from outside the trajectory (the task statement, the
    first read of a file, the raw output of a command, the environment), or it
    states an initial plan that no earlier step triggered. Omit it, or give it
    an empty list. These are the roots of the forest.
  - Do NOT judge which step is the mistake. Only report dependencies.

Return ONLY JSON:
{{
  "edges": [
    {{"src": <int>, "dst": <int>, "kind": "data"|"control",
      "entity": "<what is carried, verbatim if possible>",
      "rationale": "<one short clause>"}}
  ]
}}"""


def _window_edges(traj: Trajectory, llm: LLM, index: str, lo: int, hi: int,
                  step_budget: int, origin_edges: bool = False) -> List[Dict]:
    """Ask for the parents of every step in ``[lo, hi]``. One call."""
    idxs = list(range(lo, hi + 1))
    # Two costs share this budget. The JSON itself runs ~400 tokens per step
    # (a parent per relation, each with an entity and a rationale), and a
    # reasoning model spends a few thousand more thinking before it writes a
    # character. A 20-step window measured at ~4.4k thinking + ~2.5k JSON;
    # the old 800 + 220/step gave 5200 and came back finish="length" with an
    # EMPTY body, which is indistinguishable from an answer holding no edges.
    # Whole windows were dropping out of the sweep that way.
    budget = min(REASONING_HEADROOM + 400 * len(idxs), 16000)
    if origin_edges:
        # Asking for every piece of information separately multiplies the JSON,
        # not the thinking. Under-budgeting here is not a partial answer: the
        # call comes back finish="length" with an EMPTY body, indistinguishable
        # from a window that found no edges, so the allowance grows with the
        # density the prompt asks for.
        budget = min(REASONING_HEADROOM + 1000 * len(idxs), 28000)
    obj = llm.generate_obj(
        (FORWARD_PROMPT_ORIGIN if origin_edges else FORWARD_PROMPT).format(
            task=clip_middle(traj.task_description, 2000),
            last=len(traj) - 1, index=index, lo=lo, hi=hi,
            window_text=render_steps(traj, idxs, step_budget)),
        max_tokens=budget, system=TRACE_SYSTEM)
    return [e for e in (obj.get("edges") or []) if isinstance(e, dict)]


def build_forward(traj: Trajectory, llm: LLM, graph: FlowGraph,
                  window: int = 20, index_chars: int = 40000,
                  step_budget: int = 2000, workers: int = 4,
                  origin_edges: bool = False,
                  index_per_step: int = 180) -> FlowGraph:
    """Stage 2: build the data and control forests in one forward sweep.

    Every step is asked about exactly once, so no part of the trajectory is
    left unexplored — the head, where the earliest decisions live, is covered
    as thoroughly as the tail.

    A step's parents lie strictly before it and are read off the text, so no
    window depends on another window's answer: all of them go out at once.
    That is the whole speed argument for building forwards. A backward walk
    has to finish one round to learn whom to ask about next, which serialises
    it into roughly one round-trip per four steps traced.

    The returned graph is a parent forest per relation, not yet a flow: call
    ``graph.paint()`` (or ``on_flow()``) to keep only what reaches an anchor.
    """
    # A parent is always an earlier step, so a window ending at ``hi`` has no
    # use for the digest of anything after ``hi``. Cutting it there focuses the
    # model on what it may legitimately cite and takes the prefill of the early
    # windows — the ones covering the head of a 700-step trajectory — from the
    # whole digest down to a few hundred lines.
    # How much of each step the citing windows get to see. A step is credited
    # with an out-edge only when some LATER window names it as a parent, and a
    # window names parents off these digest lines -- so whatever falls past
    # ``index_per_step`` is, for the purpose of being cited, not there.
    # Measured on SWE-Bench-Pro at the shipped 180: in-degree is flat across
    # every length band (1.12 -> 1.01), since a step reads its OWN parents from
    # full text, while out-degree rises to 1.52 at 1500-4000 raw chars and then
    # turns over to 1.24 past 4000 -- the one band where more content buys
    # fewer citations. The 26 reference steps this arm never gets right average
    # 4382 chars and carry out-degree <= 1 on normal in-degree.
    #
    # Raising this alone would backfire: the late windows already approach
    # ``index_chars`` and ``_fit`` drops the MIDDLE, so a wider per-step digest
    # would buy legibility at the head and tail by deleting it in between.
    # The two move together or not at all.
    lines = index_lines(traj, per_step=index_per_step)

    n = len(traj)
    bounds = [(lo, min(lo + window, n) - 1) for lo in range(0, n, window)]
    # step 0 has nothing before it; a window holding only step 0 has nothing
    # to ask about.
    bounds = [(lo, hi) for lo, hi in bounds if hi >= 1]
    if not bounds:
        return graph

    batches = map_concurrent(
        lambda b: _window_edges(traj, llm,
                                _fit("\n".join(lines[: b[1] + 1]), index_chars),
                                b[0], b[1], step_budget, origin_edges),
        bounds, workers=workers)

    # Deduplicate parents per (step, relation) before adding, so the fan-in cap
    # counts distinct parents rather than repeated mentions of one.
    seen: Dict[tuple, Set[int]] = {}
    kept = dropped = 0
    for raw in batches:
        for e in (raw or []):
            try:
                src, dst = int(e.get("src")), int(e.get("dst"))
            except (TypeError, ValueError):
                continue
            if not (0 <= src < n and 0 <= dst < n) or src >= dst:
                continue                      # parents must precede children
            kind = str(e.get("kind") or "").strip().lower()
            kind = DATA_FLOW if kind.startswith("d") else (
                CONTROL_FLOW if kind.startswith("c") else "")
            if not kind:
                continue
            bucket = seen.setdefault((dst, kind), set())
            if src not in bucket and len(bucket) >= MAX_PARENTS_PER_STEP:
                dropped += 1
                continue
            bucket.add(src)
            graph.add(FlowEdge(src=src, dst=dst, kind=kind,
                               entity=str(e.get("entity") or "")[:200],
                               rationale=str(e.get("rationale") or "")[:300]))
            kept += 1

    if dropped:
        logger.warning("%s: dropped %d parents past the %d-per-step fan-in cap",
                       traj.task_id, dropped, MAX_PARENTS_PER_STEP)
    data_roots = sum(1 for i in range(n) if not seen.get((i, DATA_FLOW)))
    # A relation whose edges are nearly all dst-src==1 has collapsed into the
    # step sequence: painting up from a late anchor then sweeps in the entire
    # prefix and compresses nothing. Control flow did exactly this at 86%
    # before the prompt was tightened, so the share is worth watching.
    for kind in (DATA_FLOW, CONTROL_FLOW):
        es = [e for e in graph.edges if e.kind == kind]
        if len(es) >= 20:
            adj = sum(1 for e in es if e.dst - e.src == 1) / len(es)
            if adj > 0.6:
                logger.warning("%s: %d%% of %s edges are step-to-step; that "
                               "relation has collapsed into the sequence",
                               traj.task_id, round(100 * adj), kind)
    logger.info("%s: forward pass over %d steps in %d windows -> %d edges "
                "(%d data roots); painted %d/%d steps on flow",
                traj.task_id, n, len(bounds), kept, data_roots,
                len(graph.on_flow()), n)
    return graph


# ---------------------------------------------------------------------------
# Stage 2b — top up the painted chains from the anchors
# ---------------------------------------------------------------------------
TOPUP_PROMPT = """You are tracing a FAILED agent trajectory backwards, one step
at a time.

TASK:
{task}

WHAT VISIBLY WENT WRONG:
{failure_summary}

ARTEFACTS THE FAILURE IS ABOUT: {entities}

TRAJECTORY DIGEST (indices 0..{last}, one line per step):
{index}

STEPS TO TRACE NOW (full text):
{frontier_text}

Each step above lies on the chain leading to the failure, and an earlier pass
found no earlier step it depends on. That pass looked at each step alongside
nineteen others; you are looking at these few closely. Its answer may have been
right — or it may have missed a dependency that is several steps back.

For EACH step above, look again for its parents, in two SEPARATE relations:

DATA FLOW — which earlier step produced, transformed, or dropped a concrete
  thing this step uses. Name that thing in "entity". The parent is often far
  back: the step that first read the file, ran the command, or chose the value.

CONTROL FLOW — which earlier step made the agent take this step: the plan it is
  carrying out, the failure it is reacting to, the thing it retries or works
  around. Plain sequence is NOT a parent — "the step before finished" explains
  nothing.

A step genuinely has NO parent when its information entered from OUTSIDE the
trajectory — the task statement, the environment, the first read of a file, the
raw output of a command — or when it states an opening plan that nothing
triggered. Those are real roots of the chain, and saying so is a correct
answer, not a failure to find something. List them under "origins".

Do not invent a parent to avoid an empty answer. An edge you cannot justify
from the text above is worse than no edge: it extends the chain into a step
that had nothing to do with the failure.

At most {k} parents per step per relation. src MUST be strictly less than dst.
Do NOT judge which step is the mistake.

Return ONLY JSON:
{{
  "edges": [
    {{"src": <int>, "dst": <int>, "kind": "data"|"control",
      "entity": "<what is carried, verbatim if possible>",
      "rationale": "<one short clause>"}}
  ],
  "origins": [<indices of the traced steps that genuinely have no parent>]
}}"""


def _painted_leaves(graph: FlowGraph, kind: str) -> Set[int]:
    """Painted steps with no parent in ``kind``: where the walk ran out.

    These, and only these, are worth re-asking about. A painted step that
    already has a parent continues on its own; an unpainted step is not on a
    path to the failure, so extending it buys nothing.
    """
    parents = {e.dst for e in graph.edges if e.kind == kind}
    return {i for i in graph.paint(kind) if i not in parents}


def topup_backward(traj: Trajectory, llm: LLM, graph: FlowGraph,
                   rounds: int = 3, batch: int = 4, max_calls: int = 12,
                   edges_per_step: int = 3, index_chars: int = 40000,
                   step_budget: int = 2500, workers: int = 4) -> FlowGraph:
    """Stage 2b: re-ask about the ends of the chains that reach the anchors.

    The forward sweep asks about every step once, in windows of twenty. That
    is what makes it parallel, and it is also its weakness: on a 120-step
    trajectory the model answers "no parent" for roughly half the steps, and
    every such answer truncates whatever chain ran through it. Measured across
    two corpora, the chain length that results is governed by how often the
    model declares a root, not by how long the trajectory is — so painting
    covers 23% of a 51-step run and 13% of a 119-step one.

    This stage asks again, but only about the steps where a chain to an anchor
    actually stopped, and only a few at a time. The cost therefore scales with
    chain length rather than trajectory length: a handful of calls per case,
    against the several dozen the forward sweep already spent.

    It is deliberately easy for the model to say "no parent" here. Painting
    more of the trajectory is not the goal — a chain extended into steps that
    had nothing to do with the failure is worse than a short one, because the
    localiser reads the chain as evidence.
    """
    if not graph.anchors:
        return graph
    index = _fit(build_index(traj, per_step=180), index_chars)
    ents = ", ".join(graph.failure_entities[:25]) or "(none identified)"

    asked: Set[int] = set()
    origins: Set[int] = set()
    calls = added = 0

    for rnd in range(rounds):
        frontier = sorted((_painted_leaves(graph, DATA_FLOW)
                           | _painted_leaves(graph, CONTROL_FLOW))
                          - asked - origins,
                          reverse=True)          # nearest the failure first
        frontier = [i for i in frontier if i > 0]
        if not frontier:
            break
        budget_left = max_calls - calls
        if budget_left <= 0:
            logger.info("%s: topup stopped at the %d-call cap with %d steps "
                        "still unexplored", traj.task_id, max_calls,
                        len(frontier))
            break
        groups = [frontier[i:i + batch]
                  for i in range(0, len(frontier), batch)][:budget_left]
        asked.update(i for g in groups for i in g)

        def ask(group):
            return llm.generate_obj(
                TOPUP_PROMPT.format(
                    task=clip_middle(traj.task_description, 2000),
                    failure_summary=graph.failure_summary or "(not summarised)",
                    entities=ents, last=len(traj) - 1, index=index,
                    frontier_text=render_steps(traj, group, step_budget),
                    k=edges_per_step),
                max_tokens=min(REASONING_HEADROOM + 200 * len(group)
                               * edges_per_step, 12000),
                system=TRACE_SYSTEM)

        results = map_concurrent(ask, groups, workers=workers)
        calls += len(groups)
        before = len(graph.edges)
        for obj in results:
            if not obj:
                continue
            for e in (obj.get("edges") or []):
                if isinstance(e, dict):
                    _add_edge(graph, e, len(traj))
            origins.update(_clean_indices(obj.get("origins"), len(traj)))
        added += len(graph.edges) - before
        logger.info("%s: topup round %d asked %d steps in %d calls -> %d edges,"
                    " %d origins; painted %d/%d", traj.task_id, rnd + 1,
                    sum(len(g) for g in groups), len(groups),
                    len(graph.edges) - before, len(origins),
                    len(graph.on_flow()), len(traj))

    if calls:
        logger.info("%s: topup added %d edges in %d calls; painted %d/%d steps",
                    traj.task_id, added, calls, len(graph.on_flow()), len(traj))
    return graph


def _add_edge(graph: FlowGraph, e: Dict, n: int) -> bool:
    """Validate and add one model-reported edge. Shared by both stages."""
    try:
        src, dst = int(e.get("src")), int(e.get("dst"))
    except (TypeError, ValueError):
        return False
    if not (0 <= src < n and 0 <= dst < n) or src >= dst:
        return False
    kind = str(e.get("kind") or "").strip().lower()
    kind = DATA_FLOW if kind.startswith("d") else (
        CONTROL_FLOW if kind.startswith("c") else "")
    if not kind:
        return False
    graph.add(FlowEdge(src=src, dst=dst, kind=kind,
                       entity=str(e.get("entity") or "")[:200],
                       rationale=str(e.get("rationale") or "")[:300]))
    return True


# ---------------------------------------------------------------------------
# Stage 2 (legacy) — backward expansion
# ---------------------------------------------------------------------------
TRACE_PROMPT = """You are tracing backwards through a FAILED agent trajectory to
find which EARLIER steps this step depends on.

TASK:
{task}

TRAJECTORY DIGEST (indices 0..{last}, one line per step):
{index}

STEPS CURRENTLY BEING TRACED (full text):
{frontier_text}

For EACH traced step, identify its upstream dependencies among steps with a
SMALLER index. Two complementary relations:

DATA FLOW — how the information used by this step was built by earlier steps.
  Report an edge when an earlier step INTRODUCED information this step relies
  on, or TRANSFORMED it, or DROPPED information that this step needed.
  In "entity", name the concrete thing carried: a file path, a value, a
  function name, a field name, a test id, an observation.

CONTROL FLOW — why the agent moved from the earlier step to this one.
  Report an edge when an earlier step caused this transition: it set the plan
  this step follows, its result triggered this step, it failed and this step
  is the reaction, or this step retries or abandons what it started.

Rules:
  - src MUST be strictly less than dst.
  - Only report dependencies you can justify from the shown text.
  - At most {k} edges per traced step per relation. Prefer the ones carrying
    information relevant to the artefacts above.
  - Do NOT judge which step is the mistake. Only report dependencies.

ORIGINS — a traced step is an ORIGIN when nothing earlier in the trajectory
explains it:
  - its information entered from outside the trajectory: the task statement,
    the first read of a file, the raw output of a command, the environment;
  - or it states an initial plan or goal that no earlier step triggered.
Tracing stops on a branch once it reaches an origin, so list a traced step here
only when it genuinely has no upstream left to find.

Return ONLY JSON:
{{
  "edges": [
    {{"src": <int>, "dst": <int>, "kind": "data"|"control",
      "entity": "<what is carried, verbatim if possible>",
      "rationale": "<one short clause>"}}
  ],
  "origins": [<indices of traced steps that have no upstream>]
}}"""


def trace_backward(traj: Trajectory, llm: LLM, graph: FlowGraph,
                   max_rounds: int = 20, batch_size: int = 4,
                   edges_per_step: int = 4, index_chars: int = 40000,
                   frontier_budget: int = 3000,
                   max_traced_steps: int = 60) -> FlowGraph:
    """Stage 2: expand upstream from the anchors until the flow reaches its
    origins.

    Every upstream step the model names is queued and eventually traced. The
    walk ends naturally when the queue drains — that is, when every branch has
    reached a step the model marked as an ORIGIN (information entering from
    outside the trajectory, or an untriggered initial plan) or a step with no
    reportable upstream.

    ``batch_size`` is how many steps share one LLM call, not a filter: steps
    that do not fit in this round wait in the queue rather than being dropped.
    ``max_rounds`` and ``max_traced_steps`` are only safety stops for very long
    trajectories; when either binds, the leftover queue is logged so the
    truncation is never silent.
    """
    index = _fit(build_index(traj, per_step=180), index_chars)
    ent_block = ", ".join(graph.failure_entities[:25]) or "(none identified)"
    visited: Set[int] = set()      # traced: we asked what these depend on
    origins: Set[int] = set()      # the model says nothing upstream remains
    queued: Set[int] = set()       # ever put on the queue, so we never re-add
    heap: List[int] = []           # negated indices: latest step first

    def push(i: int) -> None:
        if 0 < i < len(traj) and i not in queued:
            queued.add(i)
            heapq.heappush(heap, -i)

    for a in graph.anchors:
        push(a)

    rounds = 0
    while heap and rounds < max_rounds and len(visited) < max_traced_steps:
        batch: List[int] = []
        while heap and len(batch) < batch_size:
            i = -heapq.heappop(heap)
            if i in visited or i in origins:
                continue
            batch.append(i)
        if not batch:
            break
        visited.update(batch)
        rounds += 1

        prompt = TRACE_PROMPT.format(
            task=clip_middle(traj.task_description, 2000),
            failure_summary=graph.failure_summary or "(not summarised)",
            entities=ent_block, last=len(traj) - 1, index=index,
            frontier_text=render_steps(traj, batch, frontier_budget),
            k=edges_per_step)

        # Scale the answer budget with what we asked for. A batch of b steps
        # may return b*edges_per_step edges, each carrying an entity and a
        # rationale (~120 tokens together), plus the origins list. A fixed 2500
        # silently truncated the JSON on full batches — and a truncated body is
        # non-empty, so it used to be cached as if it were an answer.
        budget = 600 + 160 * len(batch) * edges_per_step
        obj = llm.generate_obj(prompt, max_tokens=min(budget, 8000),
                               system=TRACE_SYSTEM)
        new_nodes: Set[int] = set()
        for e in (obj.get("edges") or []):
            if not isinstance(e, dict):
                continue
            try:
                src, dst = int(e.get("src")), int(e.get("dst"))
            except (TypeError, ValueError):
                continue
            if not (0 <= src < len(traj) and 0 <= dst < len(traj)):
                continue
            if src >= dst:
                continue                      # must point backwards
            kind = str(e.get("kind") or "").strip().lower()
            kind = DATA_FLOW if kind.startswith("d") else (
                CONTROL_FLOW if kind.startswith("c") else "")
            if not kind:
                continue
            graph.add(FlowEdge(src=src, dst=dst, kind=kind,
                               entity=str(e.get("entity") or "")[:200],
                               rationale=str(e.get("rationale") or "")[:300]))
            new_nodes.add(src)

        # an origin ends its branch: never queue it, never trace it further
        origins.update(_clean_indices(obj.get("origins"), len(traj)))
        for s in sorted(new_nodes):
            if s not in visited and s not in origins:
                push(s)

        logger.info("round %d: batch=%s +%d edges, %d origins, queue=%d",
                    rounds, batch, len(obj.get("edges") or []),
                    len(origins), len(heap))

    if heap:
        logger.info("%s: tracing stopped early with %d steps still queued "
                    "(rounds=%d/%d, traced=%d/%d)", traj.task_id, len(heap),
                    rounds, max_rounds, len(visited), max_traced_steps)
    else:
        logger.info("%s: tracing reached its origins after %d rounds "
                    "(%d steps traced, %d origins)",
                    traj.task_id, rounds, len(visited), len(origins))
    return graph


def _clean_indices(vals, n: int) -> List[int]:
    out = []
    for v in (vals or []):
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in out:
            out.append(i)
    return out


def trace(traj: Trajectory, llm: LLM, max_anchors: int = 3,
          window: int = 20, workers: int = 4, backward: bool = False,
          max_rounds: int = 20, batch_size: int = 4,
          max_traced_steps: int = 60, topup_rounds: int = 0,
          topup_calls: int = 12, origin_edges: bool = False,
          index_per_step: int = 180, index_chars: int = 40000) -> FlowGraph:
    """Anchor the failure, then build the flow.

    Forward construction is the default: one wave of window calls covering
    every step. ``backward=True`` selects the origin-terminated upstream walk
    instead, which reaches only the steps it can chain back to but does so
    with a query per frontier rather than per step.

    ``topup_rounds`` adds a third stage after the forward sweep, re-asking
    about the steps where a chain to an anchor stopped. It is off by default:
    it changes the graph, so it has to be measured against a run without it
    before it becomes the default anywhere.

    Anchor RESELECTION does not happen here. ``keyword_anchors`` and
    ``failure_anchors`` are both applied by the pipeline at stage 2c, against
    ``anchors_located``, so that they also apply to a flow loaded from disk --
    which is the whole point of ``--reuse-flow``, and would be silently skipped
    by anything done inside this function.
    """
    graph = locate_anchors(traj, llm, max_anchors=max_anchors)
    logger.info("anchors=%s entities=%d", graph.anchors,
                len(graph.failure_entities))
    if backward:
        return trace_backward(traj, llm, graph, max_rounds=max_rounds,
                              batch_size=batch_size,
                              max_traced_steps=max_traced_steps)
    graph = build_forward(traj, llm, graph, window=window, workers=workers,
                          origin_edges=origin_edges,
                          index_per_step=index_per_step,
                          index_chars=index_chars)
    if topup_rounds > 0:
        graph = topup_backward(traj, llm, graph, rounds=topup_rounds,
                               max_calls=topup_calls, workers=workers)
    return graph
