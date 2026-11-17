"""Core types for LongTraj.

Ground-truth isolation is enforced structurally, not by convention:

  * ``Trajectory`` / ``Step`` have NO field capable of holding a critical-step
    label. Anything the pipeline sees is one of these objects.
  * Labels live in a separate ``GroundTruth`` object which loaders return
    alongside the trajectory, and which only ``longtraj.evaluate`` ever reads.
  * ``LoadedRecord`` is the only place the two co-exist, and it is unpacked
    immediately at the pipeline boundary.

So a bug in a prompt builder cannot leak the answer: the object it is handed
does not contain it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Roles that can never be the critical step: they are not agent decisions.
NON_ATTRIBUTABLE_ROLES = {"user", "system"}


@dataclass(frozen=True)
class Step:
    """One trajectory step. ``index`` always equals the position in the list."""

    index: int
    role: str                      # assistant | user | system | tool
    content: str                   # full text of the step
    name: Optional[str] = None     # fine-grained speaker (e.g. "ActionAgent")

    @property
    def attributable(self) -> bool:
        """Whether this step may be blamed as the critical step."""
        return self.role not in NON_ATTRIBUTABLE_ROLES

    def render(self, budget: Optional[int] = None) -> str:
        """Render for a prompt, optionally clipped to ``budget`` chars."""
        who = self.name or self.role
        body = self.content or ""
        if budget is not None and len(body) > budget:
            body = clip_middle(body, budget)
        return f"[step {self.index} | {who}]\n{body}"


@dataclass
class Trajectory:
    """A failed agent trajectory, with no label attached — by construction."""

    task_id: str
    source: str                     # longrca | trajaudit | trajdebug
    subset: str                     # e.g. swe_bench_pro / swe-agent-claude45
    task_description: str
    steps: List[Step]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def attributable_indices(self) -> List[int]:
        return [s.index for s in self.steps if s.attributable]

    def slice(self, lo: int, hi: int) -> List[Step]:
        return self.steps[max(0, lo):min(len(self.steps), hi)]

    def total_chars(self) -> int:
        return sum(len(s.content or "") for s in self.steps)


@dataclass(frozen=True)
class GroundTruth:
    """Held-out reference label. Never passed to the pipeline."""

    task_id: str
    critical_step: Optional[int]
    error_type: Optional[str] = None
    responsible_role: Optional[str] = None
    rationale: Optional[str] = None


@dataclass
class LoadedRecord:
    """Loader output. Unpacked at the pipeline boundary and never re-joined."""

    trajectory: Trajectory
    ground_truth: Optional[GroundTruth] = None


# --------------------------------------------------------------------------
# Flow graph produced by backward tracing
# --------------------------------------------------------------------------
DATA_FLOW = "data"
CONTROL_FLOW = "control"


@dataclass(frozen=True)
class FlowEdge:
    """One traced dependency.

    ``kind == DATA_FLOW``    : information produced at ``src`` is consumed,
                              transformed, or dropped at ``dst``.
    ``kind == CONTROL_FLOW`` : ``src`` is why the agent moved on to ``dst``.
    """

    src: int
    dst: int
    kind: str
    entity: str = ""        # for data flow: what was carried
    rationale: str = ""     # short LLM justification

    def key(self):
        return (self.src, self.dst, self.kind, self.entity)


@dataclass
class FlowGraph:
    # The anchors painting actually starts from. Downstream stages all read
    # this one (compression protects them, spans and the localiser rank by
    # distance to them), so re-selecting anchors needs no change anywhere else.
    anchors: List[int] = field(default_factory=list)
    # Provenance: what ``locate_anchors`` returned, verbatim. Kept separate and
    # serialised so that ``anchors`` may be re-derived per run without the
    # rewrite in ``write_complete`` destroying the original choice -- reset
    # ``anchors`` from this at the start of every run and the operation is
    # idempotent no matter how often an artifact has been written back.
    anchors_located: List[int] = field(default_factory=list)
    failure_summary: str = ""
    failure_entities: List[str] = field(default_factory=list)
    edges: List[FlowEdge] = field(default_factory=list)
    # How ``on_flow`` reads this graph -- see there. Deliberately NOT part of
    # ``to_dict``: it is a choice about reading, not a property of what was
    # traced, so the same stored flow can be read both ways and the A/B costs
    # no re-tracing.
    mixed_paint: bool = False
    # Diagnostics from ``tracing.keyword_anchors`` when rule anchoring is on:
    # the per-step scores, what the rule picked, and what the model had said.
    # Kept off ``to_dict`` for the same reason as ``mixed_paint`` -- it records
    # how ``anchors`` was chosen this run, not what was traced -- and written
    # straight into the artifact by the pipeline instead.
    anchor_rule: Optional[Dict] = None

    def add(self, edge: FlowEdge) -> None:
        seen = {e.key() for e in self.edges}
        if edge.key() not in seen:
            self.edges.append(edge)

    def paint(self, kind: Optional[str] = None) -> set:
        """Steps on a root→anchor path in one relation's forest.

        The graph holds a parent forest per relation: every edge says "dst
        depends on src". Painting walks from each anchor up its parent edges
        until it reaches a root, and returns everything on the way. A branch
        that never reaches an anchor is left unpainted — it belongs to the
        trajectory but not to the failure.

        ``kind`` restricts the walk to one relation; ``None`` walks every edge
        regardless of kind, which is what a graph built by backward tracing from
        the anchors already represents. Which of the two ``on_flow`` uses is a
        decision with consequences -- see there.
        """
        parents: Dict[int, List[int]] = {}
        for e in self.edges:
            if kind is None or e.kind == kind:
                parents.setdefault(e.dst, []).append(e.src)
        seen = {i for i in self.anchors if i >= 0}
        stack = list(seen)
        while stack:
            for p in parents.get(stack.pop(), ()):
                if p >= 0 and p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def on_flow(self) -> set:
        """Every step the failure depends on, however the chain gets there.

        Two readings, selected by ``mixed_paint``:

        SEPARATE (default). ``paint(data) | paint(control)`` -- each relation's
        forest is walked on its own. A chain must be homogeneous: you follow
        data dependencies, or control dependencies, but a chain that switches
        relation part-way is on neither path and the step is dropped.

        MIXED. ``paint(None)`` -- any edge may be followed, so a chain may
        alternate.

        THE ARGUMENT FOR MIXED, since the default was never argued for. The
        canonical bug story alternates: the agent read a wrong value (data), so
        it took a wrong branch (control), so it wrote the wrong file (data).
        Requiring a homogeneous chain excludes exactly the multi-hop causal
        stories this method exists to find: on SWE-Bench-Pro 21 of the 44 missed
        reference steps (24% of cases) are reachable only by a chain that
        switches relation, so for those this filter is what drops them.

        THE ARGUMENT AGAINST, which is why it is a flag and not a fix. Loosening
        the chain rule always costs enrichment -- recall over coverage -- and
        probe/paint_frontier.py measures eight variants on the stored flows
        without a single exception: 3.06x (this default) -> 2.93x (one
        control->data handover) -> 2.69x (any one switch) -> 2.28x (two) ->
        2.06x (mixed). Recall bought by covering more of the run is not recall,
        so none of them can be judged on ``flow_recall``; only acc@0 decides.
        Two of those points are Pareto-dominated: re-anchoring on the last 10%
        of the trajectory reaches mixed's recall 0.7326 at coverage 0.2956
        rather than 0.3550.

        AND LOOSENING CANNOT REACH THE BIGGEST BUCKET ANYWAY. This is the part
        an earlier version of this docstring got wrong: it claimed 100% of the
        reference steps the paint walks past are steps in the graph that this
        filter dropped. Measured, that is half. The other 22 SWE-Bench-Pro
        reference steps (26% of cases, against 6% on TrajAudit) are in the graph
        but reachable from NO anchor under ANY rule here, mixed included -- the
        median one sits 42 steps from the nearest anchor. That is an anchoring
        problem, not a painting one, and nothing in this method will move it.

        A MEASURED NULL, so nobody re-runs it. The cheapest loosening on that
        frontier (one control->data handover, flow_recall 0.4884 -> 0.5698 for
        coverage 0.1596 -> 0.1942) was implemented and A/B'd on SWE-Bench-Pro:
        acc@0 0.3023 -> 0.2791, and in the 7 cases where the reference step
        newly came on-flow it converted 0 -- won 0, lost 0, all 7 wrong both
        ways. See probe/RUNS.md and probe/attic/rejected_bridge_paint.patch.
        """
        if self.mixed_paint:
            return self.paint(None)
        return self.data_flow_steps() | self.control_flow_steps()

    def data_flow_steps(self) -> set:
        return self.paint(DATA_FLOW)

    def control_flow_steps(self) -> set:
        return self.paint(CONTROL_FLOW)

    def predecessors(self, node: int) -> List[FlowEdge]:
        return [e for e in self.edges if e.dst == node]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchors": self.anchors,
            "anchors_located": self.anchors_located or self.anchors,
            "failure_summary": self.failure_summary,
            "failure_entities": self.failure_entities,
            "edges": [
                {"src": e.src, "dst": e.dst, "kind": e.kind,
                 "entity": e.entity, "rationale": e.rationale}
                for e in self.edges
            ],
        }


# --------------------------------------------------------------------------
# Spans and compression
# --------------------------------------------------------------------------
WEAK = "weak"       # span touches a traced flow
STRONG = "strong"   # span lies off every flow

# Per-step compression tiers. Compression is decided per STEP, not per span:
# a step on a flow keeps its text, a step off every flow is condensed but must
# remain informative.
KEEP = "keep"              # verbatim, no compression at all
LIGHT = "light"            # on a flow but very long: head+tail kept
COMPRESSED = "compressed"  # off every flow: summarised, still substantive


@dataclass
class Span:
    """A contiguous run of steps forming one semantic unit."""

    start: int
    end: int                       # inclusive
    label: str = ""                # LLM-assigned sub-goal description
    tier: str = STRONG
    summary: str = ""              # what the agent did in this span
    # The dominant operation role inside the span ("edit", "test", ...), from
    # ``units.unit_chain``. Empty unless unit segmentation produced this span.
    role: str = ""
    # A rule-derived description of the span's operations, available BEFORE the
    # compressor has run. ``summary`` is written by ``compress``, so any stage
    # that must precede compression -- the span prior, once it conditions the
    # compressor -- reads this instead. Keeping both means the prior can move
    # earlier without the compressor losing the summary it still writes.
    digest: str = ""

    @property
    def indices(self) -> List[int]:
        return list(range(self.start, self.end + 1))

    def __len__(self) -> int:
        return self.end - self.start + 1

    def to_dict(self) -> Dict[str, Any]:
        d = {"start": self.start, "end": self.end, "label": self.label,
             "tier": self.tier, "summary": self.summary}
        # Only serialised when populated, so an artifact from an arm that does
        # not use unit segmentation stays byte-identical to one written before
        # these fields existed.
        if self.role:
            d["role"] = self.role
        if self.digest:
            d["digest"] = self.digest
        return d


@dataclass
class StepDigest:
    """One step after step-level compression.

    The step index survives compression, so the localiser can always cite a
    concrete step: nothing is merged away into span prose.
    """

    index: int
    role: str
    text: str                      # possibly compressed content
    name: str = ""
    tier: str = COMPRESSED
    raw_chars: int = 0
    flows: str = ""                # "" | "data" | "control" | "data+control"
    anchor: bool = False

    @property
    def on_flow(self) -> bool:
        """Whether this step must be protected from lossy compression.

        An anchor counts even with no edge attached: it is where the failure is
        visible, i.e. an endpoint of the flow, and tracing simply found nothing
        upstream of it yet.
        """
        return bool(self.flows) or self.anchor

    @property
    def ratio(self) -> float:
        return (len(self.text) / self.raw_chars) if self.raw_chars else 1.0

    def marker(self) -> str:
        """Short tag shown to the localiser."""
        bits = []
        if self.flows:
            bits.append("%s flow" % self.flows.replace("+", "+"))
        if self.anchor:
            bits.append("anchor")
        if self.tier == COMPRESSED:
            bits.append("compressed")
        elif self.tier == LIGHT:
            bits.append("clipped")
        return ", ".join(bits)

    def render(self) -> str:
        who = self.name or self.role
        mark = self.marker()
        head = "[step %d | %s]%s" % (self.index, who,
                                     (" (%s)" % mark) if mark else "")
        return "%s\n%s" % (head, self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "role": self.role, "name": self.name,
                "tier": self.tier, "raw_chars": self.raw_chars,
                "flows": self.flows, "anchor": self.anchor, "text": self.text}


@dataclass
class Outline:
    """Artifact 1: a high-level account of how the agent executed the task.

    ``overview`` describes the main phases end to end; ``spans`` carries the
    per-span sub-goal summaries it was built from.
    """

    overview: str = ""
    spans: List[Span] = field(default_factory=list)

    def render(self) -> str:
        lines = []
        if self.overview:
            lines.append(self.overview.strip())
            lines.append("")
        for sp in self.spans:
            flag = "on dependency path" if sp.tier == WEAK else "off path"
            lines.append("steps %d-%d [%s] %s" % (sp.start, sp.end, flag,
                                                  sp.label or "(unlabelled)"))
            if sp.summary:
                lines.append("    " + sp.summary.strip().replace("\n", "\n    "))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"overview": self.overview,
                "spans": [sp.to_dict() for sp in self.spans]}


# Confidence a span prior may attach to one of its guesses. Two levels, not a
# score: a number invites the localiser to arithmetic it cannot justify from
# phase summaries alone.
LIKELY = "likely"
POSSIBLE = "possible"


@dataclass
class SpanGuess:
    """One phase a second reader thinks may contain the critical step."""

    start: int
    end: int                       # inclusive; always a real Span's boundaries
    likelihood: str = POSSIBLE
    why: str = ""

    def covers(self, step: int) -> bool:
        return self.start <= step <= self.end

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start, "end": self.end,
                "likelihood": self.likelihood, "why": self.why}


@dataclass
class SpanPrior:
    """A second reader's guess at WHICH PHASES contain the critical step.

    Ordered: the first guess is the strongest, and evaluation reports recall at
    the top guess separately from recall over all of them. An empty prior is
    the normal degraded state — it renders to nothing and the localiser prompt
    is then byte-identical to a run without this stage.
    """

    guesses: List[SpanGuess] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.guesses)

    def covers(self, step: int) -> Optional[SpanGuess]:
        for g in self.guesses:
            if g.covers(step):
                return g
        return None

    def n_steps_covered(self) -> int:
        """Distinct steps inside any guess. Guesses may overlap after snapping."""
        covered: set = set()
        for g in self.guesses:
            covered.update(range(g.start, g.end + 1))
        return len(covered)

    def render(self) -> str:
        lines = []
        for g in self.guesses:
            head = "steps %d-%d (%s)" % (g.start, g.end, g.likelihood)
            lines.append("%s — %s" % (head, g.why.strip()) if g.why else head)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"guesses": [g.to_dict() for g in self.guesses]}


@dataclass
class Prediction:
    """Pipeline output."""

    task_id: str
    critical_step: Optional[int]
    confidence: float = 0.0
    reason: str = ""
    error_type: str = ""
    # diagnostics — not part of the answer, useful for analysis
    flow: Optional[Dict[str, Any]] = None
    spans: Optional[List[Dict[str, Any]]] = None
    outline: Optional[Dict[str, Any]] = None
    # {"guesses": [...], "chosen": <int|null>, "disagreement": "..."}
    # The guesses come from the prior stage; ``chosen`` and ``disagreement``
    # are the localiser's own account of whether it followed them. Kept
    # regardless of ``keep_diagnostics``: it is small and evaluation needs it.
    span_prior: Optional[Dict[str, Any]] = None
    # {"n": <int>, "steps": [...], "obeyed": <bool>, "moved_from": <int|null>}
    # The restricted answer space, when it was applied. ``obeyed`` records
    # whether the localiser answered inside it on its own or had to be snapped
    # in: an accuracy that only appears after snapping is a different claim from
    # one the model reached by itself, and the two must stay separable.
    candidates: Optional[Dict[str, Any]] = None
    # {"verdicts": [...], "fusion": {...}, "lead": "<expert key>",
    #  "fault_category": "..."}
    # Present only under the multi-expert localiser. Every expert's verdict is
    # kept verbatim, including the abstentions and the ones the fusion rule
    # overruled, so an alternative fusion can be scored against a stored run
    # without spending a call. Kept regardless of ``keep_diagnostics``: it is
    # small, and without it the answer cannot be attributed to a reader.
    experts: Optional[Dict[str, Any]] = None
    # {"n_called": <int>, "n_found": <int>, "n_declined": <int>,
    #  "distinct": <int>, "by_op": {"edit": {...}, ...}}
    # Present only when the per-operation-type analysts ran. Every analyst's
    # report is kept, declines included. ``distinct`` is the number that arm
    # exists to move: the four-expert localiser it replaces averaged 1.6
    # distinct answers out of 4 readers over SHARED evidence, and if five
    # readers over DISJOINT slices still converge to one step then the
    # partition did not work -- which is a result, and only readable from here.
    op_analysts: Optional[Dict[str, Any]] = None
    # {"acts": {"keep": n, "clip": n, "model": n},
    #  "reasons": {"flow": n, "prior-likely": n, "prior-possible": n,
    #              "unguessed": n, "answer-space": n}}
    # Present only under prior-conditioned compression. This is the only record
    # of WHICH rule kept which step: the digests downstream say a step survived,
    # not why, so without this an arm that wins cannot be attributed to the
    # prior rather than to the flow policy it sits on top of.
    compression_plan: Optional[Dict[str, Any]] = None
    # {"n_spans": n, "n_called": n, "n_skipped_flow": n, "n_abstained": n,
    #  "bands": {...}, "mean_strength": f, "spans": {"<start>": {...}}}
    # Present only under the pointwise span judge. Every per-phase score, its
    # confidence and its band are kept, not only the ones that earned a band:
    # the stage's claim is that a per-phase reader can CLEAR a phase, and a
    # record that dropped the cleared phases could not be asked whether it did.
    # It is also what makes the thresholds re-weightable offline -- the scores
    # are here, so a different cut can be recomputed without paying for the
    # calls again (as a new arm, swept on dev; see span_judge).
    span_judge: Optional[Dict[str, Any]] = None
    # {"steps": [...], "why": {...}, "n_c1": n, "n_c3": n, "n_new": n,
    #  "n_proposed": n, "n_dropped_known": n, "n_dropped_bad": n,
    #  "n_truncated": n, "abstained": <bool>}
    # Present only when the shortlist reader ran. Kept on abstentions as well,
    # because a row where it added nothing is a row where the prompt was
    # byte-identical and the answer provably not moved -- that is the per-row
    # control the arm is judged against, and it exists only here. ``n_truncated``
    # and ``n_dropped_known`` are what distinguish a reader that EXTENDED the
    # answer space from one that re-derived it at full price.
    shortlist: Optional[Dict[str, Any]] = None
    # {"scores": [...], "picked": [...], "located": [...]}
    # Present only under rule anchoring. ``located`` is what the model said, kept
    # alongside what the rule chose, so the two can be compared after the fact
    # without re-running the anchor call.
    anchor_rule: Optional[Dict[str, Any]] = None
    # {"cast": <int>, "votes": {"<step>": <int>}, "agreement": <float>}
    # Present only when the localiser was asked for more than one sample. The
    # whole tally is kept, not just the winner, because ``agreement`` is a
    # *measured* confidence — unlike the model's self-reported ``confidence``,
    # a run can be re-scored under a different vote rule from this field alone.
    samples: Optional[Dict[str, Any]] = None
    # {"first": <int>, "finalists": [...], "proposed": <int|null>,
    #  "accepted": <bool>, "confidence": <float>}
    # Present only when the second pass ran. The FIRST answer is kept here even
    # when the second pass overruled it, so a run can be re-scored offline under
    # a different acceptance rule without paying for the calls again.
    zoom: Optional[Dict[str, Any]] = None
    # {"first": <int>, "finalists": [...], "n_groups": <int>, "verdicts": [...],
    #  "abstained": <int>, "proposed": <int|null>, "accepted": <bool>}
    # Present only when the group-stage bracket ran. Every group's verdict is
    # kept, not just the one that won, so an alternative arbitration rule can be
    # scored against the run offline.
    bracket: Optional[Dict[str, Any]] = None
    # {"asked": [...], "answers": {"<name>": <int|null>}, "agreed": <int>,
    #  "winner": <int|null>, "incumbent": <int|null>, "changed": <bool>}
    # Present only when the localiser was asked the same question several
    # WAYS. Unlike ``samples`` -- k draws of one prompt -- each entry here is a
    # different decision procedure over identical evidence and an identical
    # answer space, so the answers are comparable by construction rather than
    # by assumption. Every framing's answer is kept, including the losers and
    # the incumbent's, so a different merge rule can be scored offline.
    framings: Optional[Dict[str, Any]] = None
    # {"first": <int>, "parents": [...], "proposed": <int|null>,
    #  "move": <bool>, "object": "...", "edge_kind": "data|control",
    #  "edge_kind_ok": <bool>, "accepted": <bool>}
    # Present only when the predecessor audit ran. Holds the answer BEFORE the
    # move and each of the four conditions separately, so a run can be
    # re-scored under a looser or stricter acceptance rule -- the part most
    # likely to change -- without paying for the calls again.
    audit: Optional[Dict[str, Any]] = None

    # {"asked": <int>, "verdicts": [{"step", "depends", "averts",
    #  "earlier_cause", "only_symptom", "confidence", "why", "score"}],
    #  "best_score", "n_tied", "incumbent", "incumbent_score",
    #  "winner_step", "changed"}
    # Present only when each candidate was judged POINTWISE -- one call
    # per candidate, each shown no rivals. Unlike ``framings`` (one
    # reader, several questions) these are several readings of several
    # steps against ONE fixed question set, so what is comparable here is
    # the steps, not the questions. Every per-candidate verdict is kept so
    # that a different weighting can be re-scored offline at zero cost.
    verify: Optional[Dict[str, Any]] = None
    # {"asked": <int>, "steps": [...], "returned": [...], "missed": [...],
    #  "coverage": <float>, "turning": [...], "answer_in_turning": <bool>,
    #  "skipped": "<reason or empty>"}
    # Present only when the localiser was made to call the roll -- one line per
    # traced step, written BEFORE it names an answer. The measured failure it
    # targets is not a wrong choice but an unmade one: in 0.830 of TrajAudit
    # misses the localiser's own stated reason never mentions the reference
    # step at all. ``missed`` is therefore the field that matters most here: it
    # is the list of steps the reader was told to account for and did not, and
    # a run where it is large is a run where the roll did not happen, whatever
    # the accuracy did.
    roll: Optional[Dict[str, Any]] = None
    # {"asked": <int>, "capped": <bool>, "skipped": "<reason or empty>",
    #  "rows": [{"step", "n", "met", "violated", "unclear",
    #            "specs": [{"id", "must", "why"}],
    #            "checks": [{"id", "verdict", "evidence"}]}]}
    # Present only when each candidate step was specified BEFORE it was read
    # and then checked against that specification. Unlike ``verify`` — which
    # asks a model to judge each candidate — no reader in this stage is ever
    # asked whether a step is critical, or told how the run ended, so what
    # lands here is a compliance count rather than a second opinion. The
    # requirements are kept verbatim alongside the verdicts: the counts are one
    # way to read them and the cheapest thing to change, and changing it must
    # not cost the ~1,800 calls that produced them.
    expect: Optional[Dict[str, Any]] = None
    # {"chapters": <int>, "goal": "...", "beliefs": [...], "commitments": [...],
    #  "open": [...], "suspects": [{"step", "strength", "why"}]}
    # Present only when the forward commitment-ledger scan supplied the region.
    # The scan's own step nominations live in ``suspects`` and are NOT the
    # answer this round -- the answer stays with the localiser so that any
    # change in accuracy is attributable to the region and nothing else. They
    # are kept so the next round's question ("should the scan answer directly?")
    # can be scored against this run without paying for the calls again.
    memo: Optional[Dict[str, Any]] = None
    usage: Dict[str, int] = field(default_factory=dict)
    compression_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "critical_step": self.critical_step,
            "confidence": self.confidence,
            "reason": self.reason,
            "error_type": self.error_type,
            "flow": self.flow,
            "spans": self.spans,
            "outline": self.outline,
            "span_prior": self.span_prior,
            "candidates": self.candidates,
            "experts": self.experts,
            "samples": self.samples,
            "zoom": self.zoom,
            "bracket": self.bracket,
            "framings": self.framings,
            "verify": self.verify,
            "roll": self.roll,
            "expect": self.expect,
            "memo": self.memo,
            "shortlist": self.shortlist,
            "usage": self.usage,
            "compression_ratio": self.compression_ratio,
        }


# --------------------------------------------------------------------------
def clip_middle(text: str, limit: int) -> str:
    """Deterministic fallback compression: keep head and tail, drop the middle."""
    s = str(text or "")
    if limit <= 0 or len(s) <= limit:
        return s
    if limit <= 48:
        return s[:limit]
    head = int(limit * 0.7)
    tail = max(0, limit - head - 24)
    return s[:head] + "\n...[clipped]...\n" + (s[-tail:] if tail else "")
