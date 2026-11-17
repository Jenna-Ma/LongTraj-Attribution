"""Step-level, flow-aware compression.

Compression is decided per STEP, not per span. A span is only the unit of
*reading*: one LLM call per span both summarises the span and compresses the
off-flow steps inside it, because both jobs need the same context.

    step on a data or control flow
        -> KEEP   verbatim, no compression at all
        -> LIGHT  head+tail only if it is enormous (deterministic clip)

    step on neither flow
        -> KEEP        if already short: nothing to gain by compressing
        -> COMPRESSED  otherwise: what it did, what it targeted, what came
                       back. Condensed, but never reduced to "did something".

Two properties matter for downstream localisation:

  * **step indices survive.** Nothing is merged into span prose, so the
    localiser can always cite a concrete step, and evidence carried by one
    step cannot be silently attributed to its neighbours.
  * **on-flow text is never rewritten by a model.** It is copied, or clipped
    deterministically. A model cannot drop the one literal that mattered.

Off-flow compression is guarded: an implausibly short result falls back to
deterministic clipping, so a lazy model cannot erase a step.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm import LLM, map_concurrent
from .schema import (COMPRESSED, KEEP, LIGHT, LIKELY, POSSIBLE, WEAK, FlowGraph,
                     Span, StepDigest, Trajectory, clip_middle)

logger = logging.getLogger(__name__)

COMPRESS_SYSTEM = (
    "You compress steps of agent trajectories for downstream failure analysis. "
    "You never invent facts and never drop identifiers, values or error text. "
    "Answer strictly in JSON."
)

# -- budgets, in characters -------------------------------------------------
ON_FLOW_VERBATIM_LIMIT = 6000   # on-flow step at or under this: untouched
ON_FLOW_CLIP_CHARS = 5000       # on-flow step above it: minimal head+tail clip
OFF_FLOW_KEEP_BELOW = 800       # off-flow step already this short: untouched
OFF_FLOW_TARGET_CHARS = 700     # off-flow compression target
OFF_FLOW_FLOOR = 120            # below this we assume the model gave up
PROMPT_STEP_CLIP = 3000         # per-step clip when building the span prompt
PROMPT_SPAN_CLIP = 40000        # whole-span cap for the same prompt

# Appended to the span prompt only when an answer-space step is being protected.
# Empty otherwise, and spliced with no newline of its own, so an arm with the
# protection off sends a prompt that is byte-identical to every run that
# predates the option -- which is what keeps the cache warm and makes the A/B
# attribute its difference to the protection and to nothing else.
PROTECT_NOTE = ("\nSome of the steps you are not asked to compress are not on "
                "that path either: they are steps that CHANGED something, and "
                "are kept in their own words so the later analysis can quote "
                "them. Treat them the same way — context for Job A only.")

# An off-flow step at or under this keeps its own words even though a model was
# asked to paraphrase it. See ``restore_verbatim``.
RESTORE_VERBATIM_LIMIT = 2500

_ID_RE = re.compile(
    r"(?:/[\w.\-]+){2,}"                                  # paths
    r"|\b[\w./\-]+\.(?:py|js|ts|go|rs|java|rb|json|ya?ml|toml|cfg|md|rst|txt)\b"
    r"|\b(?:test_[\w]+|[\w]+_test)\b"                     # test ids
    r"|\b[A-Za-z_][\w]*\.[A-Za-z_][\w]*\(?"               # attr / call
    r"|\b\d+\.\d+\.\d+\b"                                 # versions
    r"|'[^']{2,40}'|\"[^\"]{2,40}\"")                     # quoted literals


def salient_identifiers(text: str, limit: int = 24) -> List[str]:
    """Identifiers worth preserving, in first-seen order."""
    out, seen = [], set()
    for m in _ID_RE.findall(text or ""):
        tok = m.strip().strip("'\"")
        if len(tok) < 3 or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
SPAN_PROMPT = """You are processing ONE segment of a failed agent trajectory.
Do two separate jobs.

JOB A — SUMMARISE THE SEGMENT
In 1-3 sentences: what sub-goal was the agent pursuing here, and how did the
segment end (what it achieved, found, or failed to do). Factual only; do not
analyse blame or guess at root causes.

JOB B — COMPRESS THESE STEPS: {off_ids}
These steps are NOT on the failure's dependency path, so they may lose detail.
They must still be informative. For each, keep:
  - what the agent did (the action and the argument that identifies its target)
  - what came back (succeeded / failed, what was found, key error text)
Aim for about {budget} characters each. Never reduce a step to a bare phrase
like "ran a command" or "explored the repo" — a reader must still be able to
tell what happened and to what.

Steps NOT in that list are on the dependency path. They are preserved verbatim
elsewhere, so do NOT compress them — read them only as context for Job A.{protect_note}

SEGMENT: steps {start}..{end}{label}
{content}

Return ONLY JSON:
{{"label": "<short sub-goal phrase>",
  "summary": "<1-3 sentences for Job A>",
  "steps": {{{example}}}}}"""


def _span_prompt_content(traj: Trajectory, span: Span, on_flow,
                         protect=()) -> str:
    parts = []
    for i in span.indices:
        if not (0 <= i < len(traj.steps)):
            continue
        st = traj.steps[i]
        if i in on_flow:
            tag = " [ON DEPENDENCY PATH - do not compress]"
        elif i in protect:
            tag = " [ANSWER-SPACE STEP - do not compress]"
        else:
            tag = ""
        body = clip_middle(st.content or "", PROMPT_STEP_CLIP)
        parts.append("step %d [%s]%s:\n%s" % (i, st.name or st.role, tag, body))
    return clip_middle("\n\n".join(parts), PROMPT_SPAN_CLIP)


def _on_flow_digest(traj: Trajectory, i: int, flows: str,
                    anchor: bool, limit: int = ON_FLOW_VERBATIM_LIMIT
                    ) -> StepDigest:
    """A step no model is allowed to rewrite: copied, or clipped by rule.

    Used for on-flow steps and, when the answer space is protected, for the
    candidate steps too. A protected step passes ``flows=""`` and
    ``anchor=False``, so ``StepDigest.on_flow`` stays False and the artifact
    does not claim a dependency edge that tracing never found -- it was kept
    verbatim for a different reason.

    ``limit`` is where the copy stops and a deterministic clip begins. It is a
    parameter rather than the constant because the clip is the ONLY way an
    on-flow step still loses text, so a caller that has a reason to rate this
    step higher -- both relations reach it, or the prior flagged its phase --
    needs somewhere to say so.
    """
    st = traj.steps[i]
    body = st.content or ""
    if len(body) <= limit:
        tier, text = KEEP, body
    else:
        tier, text = LIGHT, clip_middle(body, max(ON_FLOW_CLIP_CHARS,
                                                  limit - 1000))
    return StepDigest(index=i, role=st.role, name=st.name or "", text=text,
                      tier=tier, raw_chars=len(body), flows=flows, anchor=anchor)


def _accept_compression(out: str, raw: str) -> bool:
    """Reject a compression that threw the step away."""
    out = (out or "").strip()
    if not out:
        return False
    if len(out) < OFF_FLOW_FLOOR and len(raw) > 4 * OFF_FLOW_FLOOR:
        return False
    return True


# ---------------------------------------------------------------------------
# Prior-conditioned compression
# ---------------------------------------------------------------------------
#
# Compression currently decides what to keep from ONE input: did tracing put a
# flow through this step. That is a structural question, and it is answered
# before anybody has asked the semantic one -- which phases could plausibly
# contain the mistake. The span prior asks exactly that, but it has always run
# AFTER compression, so its answer arrives too late to protect anything.
#
# This inverts the order. The prior runs on the rule-derived span digests
# (``spans.segment_units`` supplies them; that is why the two changes are one
# change), and compression then reads its verdict:
#
#   * a phase the prior calls LIKELY      -> every step verbatim, flow or not
#   * a phase it calls POSSIBLE           -> on-flow verbatim, the rest clipped
#                                            deterministically, never rewritten
#   * a phase it did not name at all      -> today's behaviour, and the steps
#                                            with no flow edges at all compress
#                                            harder than today
#
# FLOW STILL GRADES INSIDE EACH BAND, and it grades by RELATION, which is the
# part the old policy threw away. ``on_flow()`` is a union, so a step reached
# only by control flow and a step reached by both look identical to it. Here:
#
#   * on BOTH data and control flow -> the verbatim limit doubles, because the
#     6000-char limit is the only route by which an on-flow step still gets
#     clipped, and these are the steps least worth clipping.
#   * off-flow but sharing an edge with an on-flow step -> "flow-adjacent";
#     tracing declined to paint it but did see it participate. Inside a POSSIBLE
#     phase that earns a full-size deterministic clip rather than a half one.
#   * no edges at all, in an unnamed phase -> the "completely harmless" case,
#     and the only band whose budget goes DOWN. Something has to pay for the
#     verbatim text above, and this is the band with the least evidence behind
#     it.
#
# THE INVARIANT. ``plan`` is None for every arm that does not ask for this, and
# then not one line below behaves differently -- the existing paths are reached
# unchanged, so old artifacts reproduce byte-for-byte. There is deliberately no
# "plan that means the default": a plan built from an empty prior would still
# have to route through the new branches, and that is a diff nobody can check.
ACT_KEEP = "keep"      # verbatim, or rule-clipped at its limit; no model
ACT_CLIP = "clip"      # rule-clipped; no model
ACT_MODEL = "model"    # eligible for model compression (today's behaviour)

PRIOR_KEEP_LIMIT = 12000     # verbatim limit inside a flagged phase
BOTH_FLOW_LIMIT = 12000      # verbatim limit for a step on data AND control
PRIOR_CLIP_CHARS = 2500      # deterministic clip in a POSSIBLE phase
PRIOR_CLIP_FAR = 1200        # ... for a step with no edge to the flow
HARMLESS_TARGET = 500        # model budget in a phase with no flow and no guess
# Kept text above this, and the localiser's prompt starts losing its tail to
# ``clip_middle`` -- which drops the MIDDLE of the trajectory, i.e. exactly the
# region a prior tends to flag. So degrade explicitly here instead, weakest
# band first, rather than letting a downstream clip choose silently.
PRIOR_BUDGET_CHARS = 90000


def _flow_adjacent(graph: FlowGraph, on_flow) -> set:
    """Off-flow steps that share an edge with an on-flow step.

    Tracing saw these participate in a dependency but did not put them on a
    root-to-anchor path -- usually because the chain switched relation part-way,
    which the default separate-forest painting drops. They are weaker evidence
    than an on-flow step and stronger than a step with no edges at all.
    """
    out = set()
    for e in graph.edges:
        if e.src in on_flow and e.dst not in on_flow:
            out.add(e.dst)
        elif e.dst in on_flow and e.src not in on_flow:
            out.add(e.src)
    return out


def _span_likelihood(prior, span: Span) -> str:
    """The strongest guess overlapping this span, or "" for none.

    Overlap rather than equality: ``span_prior._parse`` snaps guesses onto real
    phase boundaries, so they usually match exactly, but a guess spanning two
    phases must flag both rather than neither.
    """
    if not prior:
        return ""
    best = ""
    for g in getattr(prior, "guesses", ()) or ():
        if g.end < span.start or g.start > span.end:
            continue
        if g.likelihood == LIKELY:
            return LIKELY
        best = best or g.likelihood
    return best


def _span_judge_band(verdicts, span: Span) -> str:
    """The band this span's pointwise judge earned, or "" for none/absent.

    ``verdicts`` is ``span_judge.judge_spans``'s mapping, keyed by ``span.start``
    exactly as ``_spans`` below is. None -- every arm that does not ask for the
    stage -- returns "" without touching anything, so not one branch downstream
    behaves differently and old artifacts reproduce byte for byte.

    Equality on ``start`` rather than the overlap ``_span_likelihood`` uses: the
    judge is handed the span objects themselves and scores each one, so there is
    no snapping step that could have moved a boundary.
    """
    if not verdicts:
        return ""
    v = verdicts.get(span.start)
    if v is None:
        return ""
    return v.band()


def plan_compression(traj: Trajectory, spans: List[Span], graph: FlowGraph,
                     prior, protect=(),
                     budget_chars: int = PRIOR_BUDGET_CHARS,
                     verdicts=None,
                     possible_keep: bool = False,
                     full_keep: bool = False) -> Dict[int, Any]:
    """Decide, per step, who may touch it and how much survives.

    Returns ``{step: {"act", "limit", "why"}}`` plus the per-span model budget
    under key ``"_spans"``. The ``why`` is carried so the artifact can be asked
    afterwards which rule kept which step -- without it, a run that wins is
    unattributable to any of the three bands above.

    ``verdicts`` is the pointwise judge's per-phase score (``span_judge``). It
    adds two bands BELOW the prior's two, never above them: the prior compared
    the phases against each other before naming three, and the judge did not
    compare anything, so where they disagree the ranked reader wins. When it is
    None every branch below is the one that ran before the argument existed.

    The total budget is NOT raised to pay for the new bands. Whatever the judge
    keeps comes out of the ``unguessed`` band or out of the degrade ladder --
    five arms have now tried buying accuracy with a larger prompt and none has
    won.

    ``possible_keep`` promotes the prior's POSSIBLE band from a clip to a full
    keep, so that a step the flow never touched is still handed over verbatim
    whenever the phase it sits in was called suspicious at all. The two bands
    then differ only in who gives ground first: ``_DEGRADE_ORDER`` takes the
    POSSIBLE band back down to a clip before LIKELY loses a single character,
    so the budget still buys the better-justified text first. Off by default --
    with the flag down every branch below is byte-for-byte the one that ran
    before the argument existed. The judge's POSSIBLE band is deliberately NOT
    promoted with it: that would be a second variable, and no live arm runs the
    judge.

    ``full_keep`` is the three-rule scheme stated without a budget at all:

      1. on the flow                                 -> the WHOLE step
      2. off the flow, inside a phase the prior called suspicious
         (LIKELY *or* POSSIBLE)                      -> the WHOLE step
      3. off the flow, phase not suspicious          -> compressed

    "Whole" means no character limit, not a larger one: every band limit below
    is raised to the step's own length, so a 24508-character step on the flow
    survives intact instead of being clipped at 12000. The degrade ladder does
    not run and ``budget_chars`` is ignored -- that is the point of the flag.
    It implies ``possible_keep``.

    Two consequences the CALLER owns, neither of which this function can
    enforce:

    * ``PRIOR_BUDGET_CHARS`` existed to stop the rendered trajectory reaching
      ``localize.locate``'s 120000-char cap, where ``_fit_budget`` shrinks
      OFF-PATH steps to 400/220/120 characters and then ``clip_middle`` drops
      the middle of the run. Rule 2's steps are off-path by construction, so on
      a long row stage 6 can silently undo exactly what this flag kept. Hand
      them to ``_fit_budget``'s ``keep`` set, or raise the cap, before reading
      any localisation result produced under this flag.
    * Prompts get bigger, and "more verbatim text" is the direction this repo
      has refuted most often. The flag makes that trade deliberately and on
      request; nothing here predicts that it wins.
    """
    protect = set(protect or ())
    on_flow = graph.on_flow()
    data = graph.data_flow_steps()
    ctrl = graph.control_flow_steps()
    adj = _flow_adjacent(graph, on_flow)

    plan: Dict[int, Any] = {}
    span_budget: Dict[int, int] = {}
    for sp in spans:
        lik = _span_likelihood(prior, sp)
        jband = _span_judge_band(verdicts, sp)
        idxs = [i for i in sp.indices if 0 <= i < len(traj.steps)]
        flowed_here = any(i in on_flow for i in idxs)
        # "Completely harmless": not named by the prior AND carrying no traced
        # flow at all. Only this band pays for the verbatim text elsewhere.
        span_budget[sp.start] = (HARMLESS_TARGET if (not lik and not flowed_here)
                                 else OFF_FLOW_TARGET_CHARS)
        for i in idxs:
            # Under full_keep every keep-band limit is raised to the step's own
            # length, which is how "the whole step" is said in a vocabulary
            # whose only verb is a limit. `whole` is the identity otherwise, so
            # with the flag down each branch below is the expression it was.
            whole = ((lambda lim, n: max(lim, n)) if full_keep
                     else (lambda lim, n: lim))
            n_i = len(traj.steps[i].content or "")
            if i in on_flow:
                lim = (BOTH_FLOW_LIMIT if (i in data and i in ctrl)
                       else ON_FLOW_VERBATIM_LIMIT)
                if lik == LIKELY:
                    lim = max(lim, PRIOR_KEEP_LIMIT)
                plan[i] = {"act": ACT_KEEP, "limit": whole(lim, n_i),
                           "why": "flow"}
            elif i in protect:
                plan[i] = {"act": ACT_KEEP,
                           "limit": whole(ON_FLOW_VERBATIM_LIMIT, n_i),
                           "why": "answer-space"}
            elif lik == LIKELY:
                plan[i] = {"act": ACT_KEEP,
                           "limit": whole(PRIOR_KEEP_LIMIT, n_i),
                           "why": "prior-likely"}
            elif jband == "judge-likely":
                plan[i] = {"act": ACT_KEEP, "limit": PRIOR_KEEP_LIMIT,
                           "why": "judge-likely"}
            elif lik == POSSIBLE:
                plan[i] = ({"act": ACT_KEEP,
                            "limit": whole(PRIOR_KEEP_LIMIT, n_i),
                            "why": "prior-possible"}
                           if (possible_keep or full_keep) else
                           {"act": ACT_CLIP,
                            "limit": PRIOR_CLIP_CHARS if i in adj else PRIOR_CLIP_FAR,
                            "why": "prior-possible"})
            elif jband == "judge-possible":
                plan[i] = {"act": ACT_CLIP,
                           "limit": PRIOR_CLIP_CHARS if i in adj else PRIOR_CLIP_FAR,
                           "why": "judge-possible"}
            else:
                plan[i] = {"act": ACT_MODEL, "limit": span_budget[sp.start],
                           "why": "unguessed"}
    # Rule 3 is the only compression full_keep allows, and the ladder exists
    # solely to take text back off the other two rules. Running it would undo
    # the flag on exactly the rows the flag is for.
    if not full_keep:
        _degrade(traj, plan, budget_chars)
    plan["_spans"] = span_budget
    return plan


# Weakest justification first. On-flow and answer-space steps are absent: the
# repo's one structural guarantee is that a step tracing put on the path is
# never model-rewritten, and a budget overrun is not a reason to break it.
#
# Each judge band gives ground BEFORE the prior band of the same strength. The
# prior read every phase line in one call and chose three; the judge read one
# phase and never saw a second, so its score is uncalibrated across phases. When
# the budget forces a choice between two bands that both say "probably here",
# the one that actually compared the alternatives keeps its text.
#
# A run without the judge has no step whose ``why`` is a judge band, so the two
# extra passes find an empty row list and change nothing -- which is what keeps
# this constant shared rather than branched on. The same holds for the first
# rung and ``possible_keep``: with the flag down no prior-possible step is ever
# an ACT_KEEP.
#
# That first rung runs before anything else because it does not destroy text --
# it walks the POSSIBLE band back to the clip it would have had with the flag
# down. So an over-budget plan collapses onto the default plan BEFORE any band
# starts being model-rewritten, and the flag can only ever help within budget.
# (It lands on PRIOR_CLIP_FAR rather than the 2500 a flow-adjacent step would
# have had; the rung below then takes both to the same place anyway.)
_DEGRADE_ORDER = (
    ("prior-possible", ACT_KEEP, ACT_CLIP),
    ("judge-possible", ACT_CLIP, ACT_MODEL),
    ("prior-possible", ACT_CLIP, ACT_MODEL),
    ("judge-likely", ACT_KEEP, ACT_CLIP),
    ("prior-likely", ACT_KEEP, ACT_CLIP),
    ("judge-likely", ACT_CLIP, ACT_MODEL),
    ("prior-likely", ACT_CLIP, ACT_MODEL),
)


def _degrade(traj: Trajectory, plan: Dict[int, Any], budget_chars: int) -> None:
    """Spend the kept-text budget on the best-justified steps.

    A prior that flags half the run would otherwise keep half the run verbatim,
    which is not a compression scheme. Degradation is in place and in order:
    the LIKELY band only starts giving ground once the POSSIBLE band has given
    all of it, and within a band the longest steps go first because they buy the
    most budget per step sacrificed.
    """
    if budget_chars <= 0:
        return

    def kept() -> int:
        tot = 0
        for i, p in plan.items():
            if p["act"] == ACT_MODEL:
                tot += p["limit"]
            else:
                tot += min(len(traj.steps[i].content or ""), p["limit"])
        return tot

    total = kept()
    if total <= budget_chars:
        return
    for why, frm, to in _DEGRADE_ORDER:
        rows = [i for i, p in plan.items()
                if p["why"] == why and p["act"] == frm]
        rows.sort(key=lambda i: -len(traj.steps[i].content or ""))
        for i in rows:
            plan[i]["act"] = to
            plan[i]["limit"] = (PRIOR_CLIP_FAR if to == ACT_CLIP
                                else OFF_FLOW_TARGET_CHARS)
            total = kept()
            if total <= budget_chars:
                logger.info("compression plan degraded to %d chars", total)
                return
    logger.info("compression plan still %d chars over budget %d",
                total, budget_chars)


def plan_stats(plan: Optional[Dict[int, Any]]) -> Dict[str, Any]:
    """How many steps each band kept, for the artifact."""
    if not plan:
        return {}
    acts: Dict[str, int] = {}
    whys: Dict[str, int] = {}
    for i, p in plan.items():
        if i == "_spans":
            continue
        acts[p["act"]] = acts.get(p["act"], 0) + 1
        whys[p["why"]] = whys.get(p["why"], 0) + 1
    return {"acts": acts, "reasons": whys}


def compress_span(traj: Trajectory, span: Span, graph: FlowGraph, llm: LLM,
                  annot: Optional[Dict[int, str]] = None,
                  protect: Optional[Sequence[int]] = None,
                  plan: Optional[Dict[int, Any]] = None
                  ) -> Tuple[Span, List[StepDigest]]:
    """Summarise one span and compress the off-flow steps inside it.

    ``protect`` names steps that must survive in their own words even though
    tracing put no flow through them. See ``compress``.

    ``plan`` is ``plan_compression``'s verdict. When it is None every branch
    below is the one that ran before this argument existed.
    """
    on_flow = graph.on_flow()
    protect = set(protect or ())
    anchors = set(graph.anchors)
    annot = annot or {}
    idxs = [i for i in span.indices if 0 <= i < len(traj.steps)]

    if plan is None:
        off = [i for i in idxs if i not in on_flow and i not in protect]
        todo = [i for i in off
                if len(traj.steps[i].content or "") > OFF_FLOW_KEEP_BELOW]
        budget = OFF_FLOW_TARGET_CHARS
        held = protect
    else:
        # The plan already folded on-flow and answer-space status in, so the
        # model is asked about exactly the steps it left as ACT_MODEL.
        todo = [i for i in idxs
                if plan.get(i, {}).get("act") == ACT_MODEL
                and len(traj.steps[i].content or "") > OFF_FLOW_KEEP_BELOW]
        budget = plan.get("_spans", {}).get(span.start, OFF_FLOW_TARGET_CHARS)
        held = {i for i in idxs
                if i not in on_flow
                and plan.get(i, {}).get("act") in (ACT_KEEP, ACT_CLIP)}

    obj = {}
    if idxs:
        example = ", ".join('"%d": "<compressed>"' % i for i in todo[:2]) \
            or '"<step id>": "<compressed>"'
        prompt = SPAN_PROMPT.format(
            off_ids=(", ".join(str(i) for i in todo) if todo
                     else "(none — summarise only)"),
            budget=budget, start=span.start, end=span.end,
            label=(" — working label: %s" % span.label) if span.label else "",
            content=_span_prompt_content(traj, span, on_flow, held),
            protect_note=(PROTECT_NOTE if any(i in held for i in idxs)
                          else ""),
            example=example)
        obj = llm.generate_obj(
            prompt, max_tokens=min(8000, 400 + len(todo) * 320),
            system=COMPRESS_SYSTEM)

    span.summary = str(obj.get("summary") or "").strip()
    if obj.get("label"):
        span.label = str(obj["label"])[:120]
    if not span.summary:
        span.summary = _fallback_summary(traj, span)

    raw_steps = obj.get("steps") if isinstance(obj.get("steps"), dict) else {}
    digests: List[StepDigest] = []
    for i in idxs:
        if i in on_flow:
            lim = (plan.get(i, {}).get("limit", ON_FLOW_VERBATIM_LIMIT)
                   if plan else ON_FLOW_VERBATIM_LIMIT)
            digests.append(_on_flow_digest(traj, i, annot.get(i, ""),
                                           i in anchors, limit=lim))
            continue
        if plan is None:
            if i in protect:
                digests.append(_on_flow_digest(traj, i, "", False))
                continue
        else:
            act = plan.get(i, {}).get("act", ACT_MODEL)
            if act == ACT_KEEP:
                digests.append(_on_flow_digest(
                    traj, i, "", False,
                    limit=plan.get(i, {}).get("limit", ON_FLOW_VERBATIM_LIMIT)))
                continue
            if act == ACT_CLIP:
                st = traj.steps[i]
                body = st.content or ""
                lim = plan.get(i, {}).get("limit", PRIOR_CLIP_CHARS)
                digests.append(StepDigest(
                    index=i, role=st.role, name=st.name or "",
                    text=(body if len(body) <= lim else clip_middle(body, lim)),
                    tier=(KEEP if len(body) <= lim else LIGHT),
                    raw_chars=len(body)))
                continue
        st = traj.steps[i]
        body = st.content or ""
        if len(body) <= OFF_FLOW_KEEP_BELOW:
            digests.append(StepDigest(index=i, role=st.role, name=st.name or "",
                                      text=body, tier=KEEP, raw_chars=len(body)))
            continue
        got = raw_steps.get(str(i))
        if got is None:
            got = raw_steps.get(i)
        if isinstance(got, (dict, list)):
            got = None
        if _accept_compression(got, body):
            text, tier = str(got).strip(), COMPRESSED
        else:
            if got:
                logger.info("step %d compression rejected (%d chars from %d)",
                            i, len(str(got)), len(body))
            text, tier = clip_middle(body, OFF_FLOW_TARGET_CHARS), LIGHT
        digests.append(StepDigest(index=i, role=st.role, name=st.name or "",
                                  text=text, tier=tier, raw_chars=len(body)))
    return span, digests


def restore_verbatim(traj: Trajectory, digests: Sequence[StepDigest],
                     limit: int = RESTORE_VERBATIM_LIMIT) -> int:
    """Throw away a paraphrase wherever the original was affordable anyway.

    Off-flow compression exists to buy prompt budget, and for a long step it
    earns its keep. For a short one it does not: rewriting a 1200-character step
    into a 700-character summary saves 500 characters and pays for them with the
    step's own words. Measured on both corpora, a reference step that reaches the
    localiser as a paraphrase is found roughly half as often as one that arrives
    intact (SP 0.167 vs 0.375, TA 0.320 vs 0.507).

    That 2x is CONFOUNDED and is not evidence that paraphrasing halves
    findability: the steps arriving intact are largely the on-flow and KEEP
    steps, i.e. the ones tracing already judged important, so "verbatim" and
    "important" are the same variable in that comparison. It is a reason to try
    the intervention, not a prediction of its size -- and the intervention was
    tried and lost (see ``compress``). Treat the number as a hypothesis that
    has since been tested, not as a measured effect.

    So this runs AFTER compression and simply prefers the original text for every
    off-flow step at or under ``limit``. Three properties make it safe to do here
    rather than by lowering the compression threshold:

      * **no extra calls.** The span call still happens and still hits the cache;
        only its output for these steps is discarded. Re-tuning the compression
        threshold instead would change every span prompt and invalidate the lot.
      * **it cannot lose information.** The replacement is the step's own content,
        never a summary of it, so the worst case is a longer prompt.
      * **it is deterministic**, so an A/B against a stored run attributes the
        difference to this rule and not to a second sample from the compressor.

    On-flow steps are skipped: they were never paraphrased to begin with.
    Returns how many steps were restored.
    """
    if limit <= 0:
        return 0
    n = 0
    for d in digests:
        if d.on_flow or d.tier == KEEP:
            continue
        if not (0 <= d.index < len(traj.steps)):
            continue
        body = traj.steps[d.index].content or ""
        if not body or len(body) > limit or d.text == body:
            continue
        d.text, d.tier = body, KEEP
        n += 1
    if n:
        logger.info("restored %d step(s) to their own words (limit %d)", n, limit)
    return n


def restore_candidates(traj: Trajectory, digests: Sequence[StepDigest],
                       steps: Sequence[int]) -> Dict[str, int]:
    """Put every answer-space step back in its own words, unconditionally.

    The localiser is REQUIRED to answer with one of these steps, and on the SP
    baseline 437 of the 921 pooled steps reach it as something other than their
    own text -- 49 of the 66 rows whose reference step is in the pool at all.
    A step the answer must be chosen from, rendered as a summary of itself, is
    the one case where compression is spending the thing it is supposed to be
    protecting.

    This is NOT ``protect_candidates`` and the difference is the whole point of
    running it separately. That flag changes the PLAN: it moves the candidates
    into the answer-space band before compression, which also changes the span
    prompt (``PROTECT_NOTE``), the budget split between bands, and therefore
    what every OTHER step in the row looks like. It was tried in 2026-09 and
    lost on both corpora. This runs after compression and touches nothing but
    the candidates' own text, so every call upstream is byte-identical and
    comes back from the cache, and an A/B attributes the difference to the
    answer space's rendering and to nothing else.

    Three ways it goes further than ``restore_verbatim``, which is why that one
    cannot be reused by raising its limit:

      * **no length limit.** ``restore_verbatim`` keeps the original only when
        it is already short; a candidate is restored whatever it costs. Measured
        on SP: the 22 pooled steps over the 6000-char verbatim cap include 5 of
        the 66 reference steps.
      * **on-flow candidates too.** ``restore_verbatim`` skips anything on the
        flow because those were never paraphrased -- but they ARE clipped, at
        6000 or 12000 chars, and a clipped step is not the step.
      * **tier is forced to KEEP**, so the artifact says afterwards which steps
        arrived intact and the claim is checkable rather than assumed.

    Returns ``{"asked", "restored", "chars_added"}``. ``restored`` counts only
    steps whose text actually changed, so a row where compression had already
    kept everything contributes zero and the stat does not flatter the arm.
    """
    want = {int(i) for i in (steps or ())}
    out = {"asked": len(want), "restored": 0, "chars_added": 0}
    if not want:
        return out
    for d in digests:
        if d.index not in want or not (0 <= d.index < len(traj.steps)):
            continue
        body = traj.steps[d.index].content or ""
        if not body or d.text == body:
            continue
        out["chars_added"] += len(body) - len(d.text)
        d.text, d.tier = body, KEEP
        out["restored"] += 1
    if out["restored"]:
        logger.info("restored %d candidate step(s) verbatim (%+d chars)",
                    out["restored"], out["chars_added"])
    return out


def _fallback_summary(traj: Trajectory, span: Span) -> str:
    """Deterministic span summary when the model gives nothing."""
    bits = []
    for i in span.indices[:4]:
        if 0 <= i < len(traj.steps):
            line = " ".join((traj.steps[i].content or "").split())[:150]
            bits.append("step %d: %s" % (i, line))
    return " | ".join(bits) or "(no summary)"


def compress(traj: Trajectory, spans: List[Span], graph: FlowGraph, llm: LLM,
             workers: int = 8,
             protect: Optional[Sequence[int]] = None,
             plan: Optional[Dict[int, Any]] = None
             ) -> Tuple[List[Span], List[StepDigest]]:
    """Run every span concurrently; return the spans and one digest per step.

    ``protect`` is a second set of steps kept in their own words, alongside the
    steps tracing put a flow through. It exists because the protection set and
    the ANSWER SPACE were never the same thing: compression protects
    ``graph.on_flow()``, but the localiser has to answer with a step from the
    candidate list, and those two sets overlap only partly. Measured on both
    corpora, a reference step that reaches the localiser as a paraphrase is
    found roughly half as often as one that arrives intact -- SP 0.167 vs
    0.375, TA 0.320 vs 0.507 -- and 31.8% of candidate steps were arriving
    paraphrased. (That 2x is observational and confounded with on-flow status;
    see ``restore_verbatim``. Only the 31.8% is a clean count.)

    This is the targeted form of ``restore_verbatim``, which had the same
    motivation and lost: that one restores EVERY short off-flow step, trading
    fidelity for volume, and the noise grew faster than the signal. This one
    restores only the steps the answer may actually be, about eleven per
    trajectory, and does it before the paraphrase is ever written rather than
    after.

    ``plan`` comes from ``plan_compression`` and, when present, supersedes the
    flow-only policy for every step -- see there. None means every branch is
    the one that ran before the argument existed.
    """
    protect = set(protect or ())
    annot = _edge_annotations(graph)
    results = map_concurrent(
        lambda sp: compress_span(traj, sp, graph, llm, annot,
                                 protect=protect, plan=plan), spans,
        workers=workers)

    out_spans: List[Span] = []
    digests: List[StepDigest] = []
    on_flow = graph.on_flow()
    anchors = set(graph.anchors)
    for sp, res in zip(spans, results):
        if res is None:                      # the whole span call failed
            logger.info("span %d-%d failed; clipping deterministically",
                        sp.start, sp.end)
            sp.summary = sp.summary or _fallback_summary(traj, sp)
            out_spans.append(sp)
            for i in sp.indices:
                if not (0 <= i < len(traj.steps)):
                    continue
                act = (plan or {}).get(i, {}).get("act")
                lim = (plan or {}).get(i, {}).get("limit")
                if i in on_flow:
                    digests.append(_on_flow_digest(
                        traj, i, annot.get(i, ""), i in anchors,
                        limit=lim or ON_FLOW_VERBATIM_LIMIT))
                elif i in protect or act == ACT_KEEP:
                    digests.append(_on_flow_digest(
                        traj, i, "", False, limit=lim or ON_FLOW_VERBATIM_LIMIT))
                else:
                    # No call came back, so there is no paraphrase to use and
                    # the deterministic clip is all that is left. A planned
                    # ACT_CLIP step keeps its own, larger limit: the span call
                    # failing is not a reason to discard the prior's verdict.
                    st = traj.steps[i]
                    body = st.content or ""
                    n = lim if act == ACT_CLIP else OFF_FLOW_TARGET_CHARS
                    digests.append(StepDigest(
                        index=i, role=st.role, name=st.name or "",
                        text=(body if len(body) <= n else clip_middle(body, n)),
                        tier=LIGHT if len(body) > OFF_FLOW_KEEP_BELOW else KEEP,
                        raw_chars=len(body)))
            continue
        s2, d2 = res
        out_spans.append(s2)
        digests.extend(d2)

    digests.sort(key=lambda d: d.index)
    return out_spans, digests


def _edge_annotations(graph: FlowGraph) -> Dict[int, str]:
    """Which flows each step participates in: "data", "control", "data+control"."""
    data = graph.data_flow_steps()
    ctrl = graph.control_flow_steps()
    out: Dict[int, str] = {}
    for i in sorted(data | ctrl | set(graph.anchors)):
        bits = []
        if i in data:
            bits.append("data")
        if i in ctrl:
            bits.append("control")
        out[i] = "+".join(bits)
    return out


# ---------------------------------------------------------------------------
def render_compressed(traj: Trajectory, spans: List[Span],
                      digests: Sequence[StepDigest]) -> str:
    """Artifact 2: the step-level compressed trajectory."""
    by_index = {d.index: d for d in digests}
    blocks = []
    for sp in spans:
        blocks.append("----- steps %d-%d: %s -----"
                      % (sp.start, sp.end, sp.label or "(unlabelled)"))
        for i in sp.indices:
            d = by_index.get(i)
            if d is not None:
                blocks.append(d.render())
    return "\n\n".join(blocks)


def compression_stats(traj: Trajectory, spans: List[Span],
                      digests: Sequence[StepDigest]) -> Dict[str, float]:
    raw = traj.total_chars()
    comp = sum(len(d.text or "") for d in digests)
    outline_chars = sum(len(sp.summary or "") for sp in spans)
    tiers = {KEEP: 0, LIGHT: 0, COMPRESSED: 0}
    for d in digests:
        tiers[d.tier] = tiers.get(d.tier, 0) + 1
    on = sum(1 for d in digests if d.on_flow)
    return {
        "raw_chars": raw,
        "compressed_chars": comp,
        "outline_chars": outline_chars,
        "ratio": (comp / raw) if raw else 0.0,
        "ratio_with_outline": ((comp + outline_chars) / raw) if raw else 0.0,
        "spans": len(spans),
        "weak_spans": sum(1 for sp in spans if sp.tier == WEAK),
        "strong_spans": sum(1 for sp in spans if sp.tier != WEAK),
        "steps": len(digests),
        "steps_on_flow": on,
        "steps_kept": tiers.get(KEEP, 0),
        "steps_clipped": tiers.get(LIGHT, 0),
        "steps_compressed": tiers.get(COMPRESSED, 0),
    }
