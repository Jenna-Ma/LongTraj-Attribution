"""Forward commitment-ledger scan: a localisation agent with memory.

``span_prior.predict_spans`` guesses the region in ONE call from the phase
summaries alone, and its recall (0.6989 on TrajAudit) caps the whole method.
``localize_walk`` already tried splitting the read into per-segment calls; it
drove the late bias from 0.725 down to 0.549 and acc@0 did not move. So "look at
one segment at a time" is not, by itself, the missing piece.

Two things distinguish this module from that walk, and they are the only reasons
it is worth a run:

  * **Order.** ``localize_walk`` visits segments in RANKED order, so it has
    lookahead over the whole run. This scans in TRAJECTORY order and each call
    sees only what came before. A judgement therefore has to be made at the
    moment a commitment is entered into, without knowing how the run ends —
    which is why "prefer the earliest cause" stops being a post-hoc preference
    in a reduce stage and becomes a structural consequence of the scan order.
    On the rows both arms miss, 72% have the reference step in a segment neither
    arm nominated and 55.6% put both answers LATER than it, so the direction of
    the residual error is the one this ordering attacks.

  * **What is carried.** The walk accumulates a list of NOMINATIONS. This
    carries a world-state ledger: beliefs with a status, commitments with an
    in-force flag, and open questions. A belief entered as ``assumed`` at
    segment 2 and marked ``contradicted`` at segment 5 is the signature of a
    critical step, and a nomination list cannot express that retraction — the
    walk may not even have read segment 2 by the time it reads segment 5.

Output is a ``SpanPrior``, so this is a drop-in replacement for stage 5c and
every later stage is byte-identical. It deliberately reuses ``span_prior``'s own
guess budget and coverage bound: an arm that simply nominates more regions would
earn recall for free and the A/B would measure nothing.

Ground truth never enters this module: it only ever receives ``Trajectory``,
``Span``, ``FlowGraph`` and ``StepDigest``, none of which can hold a label.

VERDICT (TrajAudit, 98 rows, 2026-09-17): **it lost, and both arguments above
were wrong.** acc@0 0.5000 -> 0.4388; dev 0.5111 -> 0.4667, test 0.4906 ->
0.4151, so the pre-registered criterion failed on both halves (4 wins, 10
losses, McNemar p = 0.18). Two findings worth keeping:

  * **The ordering argument is refuted.** Late bias went 0.714 -> 0.709 — it did
    not move. Scanning in trajectory order with no lookahead does NOT make the
    answer come out earlier, while the RANKED walk (which has lookahead) drove
    the same number to 0.549. Whatever produces the late bias, it is not the
    visiting order. Do not reuse that inference.
  * **The ledger made the prior sharper and that is what killed it.** Top-guess
    recall rose (0.484 -> 0.516) and the localiser followed the prior more often
    (0.860 -> 0.871), but the nominated region got NARROWER (coverage 0.245 ->
    0.196) and total recall fell (0.699 -> 0.656). The cost lands on the rows
    where the prior is wrong: acc@0 given a prior miss halved, 0.179 -> 0.094.
    At this repo's 0.86 follow rate, narrowing the prior is a net loss — the
    thing to raise is recall AT UNCHANGED coverage, not top-guess precision.

Kept as an ablation, default off (``--memo-scan``). Full account in
``probe/RUNS.md``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .compress import render_compressed
from .llm import LLM
from .localize import (_hops_from_symptom, _names_failure_artefact,
                       _snap_to_attributable)
from .schema import (CONTROL_FLOW, DATA_FLOW, LIKELY, POSSIBLE, FlowGraph,
                     Span, SpanGuess, SpanPrior, StepDigest, Trajectory,
                     clip_middle)
from .span_prior import _best_overlap, _symptom_block

logger = logging.getLogger(__name__)

MEMO_MAX_CHAPTERS = 10      # hard cap on calls per trajectory
# The reply is a whole ledger, not a verdict: up to 8 beliefs, 6 commitments,
# 4 open questions and 3 suspects, each with prose. The first smoke run at 1600
# truncated 9 replies out of 15 and burned the retries, which cost three
# chapters their update and left the localiser with a fallback answer. Sized
# from what the caps can actually emit rather than from what a verdict costs.
MEMO_BUDGET = 3200

# The ledger is the whole point and it must stay bounded. Without a cap it grows
# monotonically and the final chapter's prompt is the whole trajectory again --
# which would cost what a single-shot read costs AND lose the locality that
# makes the scan different from one.
LEDGER_CAPS = {"beliefs": 8, "commitments": 6, "open": 4, "suspects": 3}
FIELD_CLIP = 200            # per-entry prose clip inside the ledger
GOAL_CLIP = 300

MEMO_SYSTEM = (
    "You read a failed agent trajectory one segment at a time, in order, "
    "carrying a running ledger of what the agent believes, what it has "
    "committed to, and what it has left unverified. You never see later "
    "segments. Your job each turn is to update that ledger and to flag any step "
    "in THIS segment that entered an unverified assumption or commitment the "
    "rest of the run then depended on. Answer strictly in JSON."
)

MEMO_PROMPT = """A coding agent FAILED this task. You are reading its run ONE
SEGMENT AT A TIME, in order. This is segment {pos} of {total}. You have not seen
— and will not see — anything after it.

THE TASK THE AGENT WAS GIVEN:
{task}

WHAT VISIBLY WENT WRONG (the symptom, reported at the END of the run):
{failure_summary}
{symptom_block}
Treat the symptom as the thing to be explained, not as a location. It surfaced
at the end; what caused it was entered into earlier, possibly in this segment.

THE LEDGER SO FAR (your own output from the previous segment; everything you
know about what came before is in here):
{ledger}

FLOW EVIDENCE FOR THIS SEGMENT
Mechanical facts, not judgements: how far each of this segment's steps is from
the symptom over the traced dependencies, which earlier steps it depends on, and
where the things it touches reappear elsewhere in the run.
{evidence}

THIS SEGMENT — steps {start}-{end}:
{body}

Do two things.

FIRST, update the ledger so it describes the world as of the END of this
segment:
  * beliefs — what the agent is now treating as true. Mark each "supported" if
    this run actually established it, "assumed" if the agent adopted it without
    checking, "contradicted" if something here disproves a belief it still acts
    on. CARRY FORWARD earlier entries unchanged unless this segment changes
    them; you will not get to see the earlier segments again.
  * commitments — choices already made that later work builds on (an approach, a
    file, an interface, a diagnosis). Set "in_force": false once the agent
    abandons one.
  * open — things the agent said it would verify, or must verify, and has not.

SECOND, decide whether any step IN THIS SEGMENT is a fix-point: a step that
entered a belief or commitment which was NOT established, and which later work
would then rest on. That is the thing to flag — not the step where trouble
becomes visible.
  * A step that merely reads, explores or reports is not a fix-point.
  * A well-formed action is still a fix-point if the DECISION behind it was
    unverified.
  * Marking a carried belief "contradicted" here is strong evidence that the
    step which ENTERED that belief was the fix-point. If that step is in an
    earlier segment, do not flag it here — instead say so in the belief's status
    and keep any existing suspect for it in "suspects".
  * You are reading one segment, not the evidence for the whole run. A later
    reader gets the full step-by-step text and will pick the single step; your
    suspects only tell that reader WHICH SEGMENTS to open first, and they can
    only open a few. So an EMPTY suspects list is not the safe answer — it
    spends a slot on nothing. If this segment contains a step that took anything
    for granted, record it at the strength it deserves rather than omitting it:
      - strength 3: a commitment that was plainly unverified and that later work
        in this segment already visibly rests on.
      - strength 2: an unverified commitment whose consequences are not yet
        visible here.
      - strength 1: the best candidate in this segment, kept because it is the
        best here, even though on its own it looks weak.
    Only a segment that is pure reading, exploring or reporting should add
    nothing.

Keep the ledger SHORT: at most {n_beliefs} beliefs, {n_commitments}
commitments, {n_open} open questions and {n_suspects} suspects. ONE clause each,
under 30 words — anything longer is truncated and wasted. When you are at the
limit, merge or drop the weakest rather than truncating the newest — except for
"suspects", where the limit is spent on BREADTH: keep at most one suspect per
segment, so that the {n_suspects} slots point at {n_suspects} DIFFERENT places in
the run. A second suspect from a segment that already has one displaces nothing;
merge it into that entry instead. Between two suspects of equal strength from
different segments, keep the EARLIER one.

Return ONLY JSON:
{{"goal": "<one clause: what success required; keep the previous wording if it
  is already there>",
  "beliefs": [{{"claim": "...", "since": <step index>,
                "status": "supported|assumed|contradicted"}}],
  "commitments": [{{"what": "...", "at": <step index>, "in_force": true|false}}],
  "open": [{{"question": "...", "raised_at": <step index>}}],
  "suspects": [{{"step": <step index>, "strength": <0-3>,
                 "why": "<one clause: what it took for granted, and what later
                 rests on it>"}}]}}"""


# ---------------------------------------------------------------------------
# Chapters: the unit of one call.
# ---------------------------------------------------------------------------
def chapters(spans: Sequence[Span],
             cap: int = MEMO_MAX_CHAPTERS) -> List[List[int]]:
    """Group span positions into at most ``cap`` contiguous chapters.

    Merging, never truncating. Dropping the tail of a long run would turn "has
    not read it yet" into "will never read it" and destroy the forward property
    the whole design rests on, so when there are more spans than the budget the
    shortest adjacent pair is merged until the count fits.
    """
    groups = [[i] for i in range(len(spans))]
    if cap < 1:
        cap = 1
    while len(groups) > cap:
        # merge the adjacent pair with the fewest steps between them
        def size(g):
            return sum(len(spans[i]) for i in g)
        j = min(range(len(groups) - 1),
                key=lambda k: (size(groups[k]) + size(groups[k + 1]), k))
        groups[j:j + 2] = [groups[j] + groups[j + 1]]
    return groups


# ---------------------------------------------------------------------------
# The ledger.
# ---------------------------------------------------------------------------
def _empty_ledger() -> Dict[str, Any]:
    return {"goal": "", "beliefs": [], "commitments": [], "open": [],
            "suspects": []}


def _int_or_none(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _merge_ledger(prev: Dict[str, Any], obj: Dict[str, Any],
                  n_steps: int) -> Dict[str, Any]:
    """Validate one chapter's reply into the next chapter's ledger.

    An unusable reply (the vveai gateway returns empty HTTP-200s under load)
    passes the previous ledger through unchanged rather than resetting it: a
    lost call should cost one segment's reading, not the memory of every segment
    before it.
    """
    if not isinstance(obj, dict) or not obj:
        return prev

    out = _empty_ledger()
    out["goal"] = (str(obj.get("goal") or "").strip()[:GOAL_CLIP]
                   or prev.get("goal", ""))

    def rows(key, fields):
        got = []
        for item in (obj.get(key) or []):
            if not isinstance(item, dict):
                continue
            row = {}
            ok = True
            for name, kind in fields.items():
                v = item.get(name)
                if kind == "step":
                    s = _int_or_none(v)
                    # The model invents indices. A row anchored outside the run
                    # cannot be mapped back to a span, so it is not kept.
                    if s is None or not (0 <= s < n_steps):
                        ok = False
                        break
                    row[name] = s
                elif kind == "bool":
                    row[name] = bool(v)
                elif kind == "grade":
                    g = _int_or_none(v)
                    row[name] = max(0, min(3, g if g is not None else 0))
                else:
                    row[name] = str(v or "").strip()[:FIELD_CLIP]
            if ok:
                got.append(row)
            if len(got) >= LEDGER_CAPS[key]:
                break
        return got

    out["beliefs"] = rows("beliefs", {"claim": "text", "since": "step",
                                      "status": "text"})
    out["commitments"] = rows("commitments", {"what": "text", "at": "step",
                                              "in_force": "bool"})
    out["open"] = rows("open", {"question": "text", "raised_at": "step"})
    out["suspects"] = rows("suspects", {"step": "step", "strength": "grade",
                                        "why": "text"})
    return out


def render_ledger(led: Dict[str, Any]) -> str:
    if not led or not any(led.get(k) for k in
                          ("goal", "beliefs", "commitments", "open", "suspects")):
        return "(empty — this is the first segment)"
    out = []
    if led.get("goal"):
        out.append("GOAL: %s" % led["goal"])
    if led.get("beliefs"):
        out.append("BELIEFS:")
        out += ["  - [%s, since step %d] %s"
                % (b.get("status") or "?", b["since"], b.get("claim") or "")
                for b in led["beliefs"]]
    if led.get("commitments"):
        out.append("COMMITMENTS:")
        out += ["  - [step %d, %s] %s"
                % (c["at"], "in force" if c.get("in_force") else "abandoned",
                   c.get("what") or "")
                for c in led["commitments"]]
    if led.get("open"):
        out.append("OPEN (said or needed to be verified, still unverified):")
        out += ["  - [raised at step %d] %s"
                % (o["raised_at"], o.get("question") or "")
                for o in led["open"]]
    if led.get("suspects"):
        out.append("SUSPECTS SO FAR:")
        out += ["  - step %d (strength %d): %s"
                % (s["step"], s.get("strength", 0), s.get("why") or "")
                for s in led["suspects"]]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Per-chapter evidence (mechanical; no call).
# ---------------------------------------------------------------------------
def _entity_index(graph: FlowGraph) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for e in graph.edges:
        if e.kind == DATA_FLOW and e.entity:
            out.setdefault(e.entity.strip().lower(), set()).update({e.src, e.dst})
    return out


def _chapter_evidence(idxs: set, graph: FlowGraph, ent_index: Dict[str, set],
                      data_hops: Dict[int, int], ctrl_hops: Dict[int, int],
                      max_steps: int = 12, max_refs: int = 6) -> str:
    """Flow facts for this chapter's steps only, in step order."""
    on = sorted(i for i in idxs if i in data_hops or i in ctrl_hops
                or graph.predecessors(i))
    if not on:
        return "(no traced data or control flow runs through this segment)"
    lines = []
    for i in on[:max_steps]:
        hops = [h[i] for h in (data_hops, ctrl_hops) if i in h]
        lines.append("step %d — %s hops from symptom"
                     % (i, min(hops) if hops else "no traced path"))
        for e in graph.predecessors(i)[:max_refs]:
            art = (" — names a failure artefact"
                   if _names_failure_artefact(e.entity, graph.failure_entities)
                   else "")
            lines.append("    depends on step %d%s (%s)%s"
                         % (e.src, (" [%s]" % e.entity) if e.entity else "",
                            e.kind, art))
        carried = {e.entity.strip().lower() for e in graph.edges
                   if e.kind == DATA_FLOW and e.entity
                   and (e.src == i or e.dst == i)}
        for ent in sorted(carried)[:max_refs]:
            # Only steps BEFORE this chapter: naming where an entity reappears
            # later would hand the scan the lookahead it is built to do without.
            earlier = sorted(s for s in ent_index.get(ent, ()) if s < min(idxs))
            if earlier:
                lines.append("    entity '%s' also at earlier steps %s"
                             % (ent, ", ".join(str(s) for s in earlier[:max_refs])))
    if len(on) > max_steps:
        lines.append("    ... %d more steps with traced flow in this segment"
                     % (len(on) - max_steps))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The scan.
# ---------------------------------------------------------------------------
def scan(traj: Trajectory, spans: List[Span], graph: FlowGraph, llm: LLM,
         digests: Optional[Sequence[StepDigest]] = None,
         max_chapters: int = MEMO_MAX_CHAPTERS, max_guesses: int = 3,
         max_coverage: float = 0.5,
         symptom: bool = True) -> Tuple[SpanPrior, Dict[str, Any]]:
    """Walk the run forward carrying a ledger; return the regions it suspects.

    Returns the prior AND the final ledger, so the step nominations the scan
    made are recorded in the artifact even though this round does not use them
    as the answer. Keeping the answer with ``localize`` is what makes any change
    in acc@0 attributable to the region and nothing else.
    """
    if len(spans or []) < 4:
        # Same bar as span_prior: with fewer phases than this, any guess covers
        # a large share of the run and the calls would buy nothing.
        return SpanPrior(), _empty_ledger()

    groups = chapters(spans, max_chapters)
    by_index = {d.index: d for d in (digests or ())}
    ent_index = _entity_index(graph)
    data_hops = _hops_from_symptom(graph, DATA_FLOW)
    ctrl_hops = _hops_from_symptom(graph, CONTROL_FLOW)
    sym = _symptom_block(traj) if symptom else ""
    n_steps = len(traj)

    led = _empty_ledger()
    empties = 0
    for pos, grp in enumerate(groups):
        chap = [spans[i] for i in grp]
        idxs = {i for sp in chap for i in sp.indices}
        chap_digests = [by_index[i] for i in sorted(idxs) if i in by_index]
        body = (render_compressed(traj, chap, chap_digests)
                if chap_digests else "")
        if not body.strip():
            # No compressed text for this chapter (digests not supplied, or all
            # steps dropped). Nothing to read; skip without spending a call.
            continue

        prompt = MEMO_PROMPT.format(
            pos=pos + 1, total=len(groups),
            task=clip_middle(traj.task_description, 2000),
            failure_summary=graph.failure_summary or "(not summarised)",
            symptom_block=sym,
            ledger=render_ledger(led),
            evidence=_chapter_evidence(idxs, graph, ent_index,
                                       data_hops, ctrl_hops),
            start=chap[0].start, end=chap[-1].end,
            body=clip_middle(body, 40000),
            n_beliefs=LEDGER_CAPS["beliefs"],
            n_commitments=LEDGER_CAPS["commitments"],
            n_open=LEDGER_CAPS["open"],
            n_suspects=LEDGER_CAPS["suspects"])

        obj = llm.generate_obj(prompt, max_tokens=MEMO_BUDGET,
                               system=MEMO_SYSTEM)
        if not obj:
            empties += 1
            logger.warning("%s: chapter %d/%d returned an empty response "
                           "(endpoint failure, not a skip); ledger carried "
                           "forward unchanged", traj.task_id, pos + 1,
                           len(groups))
        led = _merge_ledger(led, obj, n_steps)

    prior = _to_prior(led, traj, spans, max_guesses, max_coverage)
    led["chapters"] = len(groups)
    led["empty_replies"] = empties
    logger.info("%s: memo scan over %d chapters -> %d suspects, %d guesses "
                "(%d empty replies)", traj.task_id, len(groups),
                len(led["suspects"]), len(prior.guesses), empties)
    return prior, led


def _to_prior(led: Dict[str, Any], traj: Trajectory, spans: List[Span],
              max_guesses: int, max_coverage: float) -> SpanPrior:
    """Suspect steps -> the phases holding them, strongest and earliest first.

    Ties on strength break toward the LOWER step index. That is the same
    preference the scan order already encodes, and on the corpus it is the only
    tie-break with any discriminative power: over the rows where the two shipped
    arms disagree, "take the earlier answer" recovers 0.372 against a confidence
    comparison's 0.279.

    The guess budget and the coverage bound are span_prior's, unchanged. An arm
    that nominated more regions would earn recall for free, and the A/B is
    supposed to measure where it looks, not how widely.
    """
    budget = min(max_guesses, max(1, len(spans) // 3))
    limit = int(len(traj) * max_coverage) if len(traj) else 0

    ranked = sorted(led.get("suspects") or [],
                    key=lambda s: (-int(s.get("strength") or 0), int(s["step"])))
    guesses: List[SpanGuess] = []
    seen = set()
    covered: set = set()
    for s in ranked:
        step = _snap_to_attributable(int(s["step"]), traj)
        if step is None:
            continue
        sp = _best_overlap(step, step, spans)
        if sp is None or (sp.start, sp.end) in seen:
            continue
        steps = set(range(sp.start, sp.end + 1))
        if guesses and len(covered | steps) > limit:
            break
        seen.add((sp.start, sp.end))
        covered |= steps
        guesses.append(SpanGuess(
            start=sp.start, end=sp.end,
            likelihood=(LIKELY if int(s.get("strength") or 0) >= 2
                        else POSSIBLE),
            why=str(s.get("why") or "")[:300]))
        if len(guesses) >= budget:
            break
    return SpanPrior(guesses=guesses)
