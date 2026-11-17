"""End-to-end pipeline.

    raw trajectory
      -> anchor + flow build   (anchor on visible failure, then relate steps)
      -> [persist flow]        <- written here, before anything can fail later
      -> semantic spans        (contiguous sub-goal phases)
      -> per-span summary + step-level compression
             on a painted path -> kept verbatim (or minimally clipped)
             off every path    -> condensed, but still says what happened
      -> outline               ARTIFACT 1: how the task was executed
      -> compressed steps      ARTIFACT 2: step-level trajectory, indices intact
      -> [span prior]          optional: which phases plausibly hold the cause
      -> localisation over both artifacts
      -> [persist full record]

The flow graph is saved as soon as it is built rather than at the end, because
building it is the expensive stage. A crash in compression or localisation
therefore costs nothing to recover, and ``reuse_flow`` lets you re-run
downstream stages against a stored graph.

The ground truth never enters this module: ``run`` takes a ``Trajectory``,
which has no field able to hold a label.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from . import alfworld_anchors as alfworld_anchors_mod
from . import alfworld_prompt as alfworld_prompt_mod
from . import alfworld_ledger as alfworld_ledger_mod
from . import alfworld_modules as alfworld_modules_mod
from . import alfworld_local3 as alfworld_local3_mod
from . import alfworld_obligations as alfworld_obligations_mod
from . import alfworld_space as alfworld_space_mod
from . import audit as audit_mod
from . import bracket as bracket_mod
from . import candidates as candidates_mod
from . import compress as compress_mod
from . import experts as experts_mod
from . import expect as expect_mod
from . import localize as localize_mod
from . import memo_scan as memo_scan_mod
from . import op_analysts as op_analysts_mod
from . import outline as outline_mod
from . import spans as spans_mod
from . import span_prior as span_prior_mod
from . import span_prior_enhanced as span_prior_enhanced_mod
from . import span_judge as span_judge_mod
from . import shortlist as shortlist_mod
from . import tracing as tracing_mod
from . import zoom as zoom_mod
from .artifacts import TraceStore
from .llm import LLM
from .schema import Prediction, Trajectory

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Pipeline knobs."""

    max_anchors: int = 3           # symptom steps to start from
    flow_window: int = 20          # steps per forward-construction call
    # How many chars of each step the parent-naming windows can see. A step
    # earns out-degree only by being CITED, and it is cited off these digest
    # lines, so this is the width of the aperture through which the tracer
    # decides what influenced what. Raising it costs forward-sweep prefill and
    # nothing else. Must move with flow_index_chars -- see build_forward.
    flow_index_per_step: int = 180
    flow_index_chars: int = 40000
    origin_edges: bool = False     # 0 extra calls: make the forward sweep cite
                                   #    where each value entered, not only the
                                   #    nearest mention (raises fan-in density)
    backward: bool = False         # use the legacy upstream walk instead
    max_rounds: int = 20           # backward only: safety cap on rounds
    batch_size: int = 4            # backward only: steps per tracing call
    max_traced_steps: int = 60     # backward only: cap on total steps traced
    topup_rounds: int = 0          # re-ask where a chain to an anchor stopped
    topup_calls: int = 12          # cap on the extra calls that costs
    target_spans: Optional[int] = None   # default: ~n/6
    weak_ratio: float = 0.0        # >0 requires several flow steps per weak span
    workers: int = 8               # concurrent span calls
    max_prompt_chars: int = 120000  # final localiser prompt cap. Over it,
                                   # localize._fit_budget clips every OFF-FLOW
                                   # digest to 400/220/120 and then drops the
                                   # trajectory's middle -- which is exactly the
                                   # set --full-keep just saved, so the two
                                   # fight. Raise it with --max-prompt-chars;
                                   # qwen3.7-plus took 2.0M chars in the
                                   # 2026-10-13 probe (see llm.MAX_PROMPT_CHARS)
    verbatim_limit: int = 0        # 0 calls: keep off-flow steps this short intact
    build_outline: bool = True     # artifact 1
    span_prior: bool = False       # +1 call: guess which phases hold the cause
    prior_samples: int = 1         # +k-1 calls: draw the prior prompt k times
                                   # and UNION the phases named, ranked by vote.
                                   # Repairs a measured arity deficit (qwen names
                                   # 1 phase where budget allows 3); _parse's
                                   # budget/coverage caps still bound the result.
    prior_signals: bool = False    # 0 calls: give that guess the phase counts
    prior_symptom: bool = False    # 0 calls: give it the harness's failing test
    prior_enhanced: bool = False   # +1 call: keep --span-prior's guesses and
                                   # APPEND a second tier until the nominated
                                   # region reaches prior_coverage_target. The
                                   # first guess is unchanged, so this cannot
                                   # move prior_top_recall by construction.
    prior_coverage_target: float = 0.30  # measured today: 0.185
    localize_symptom: bool = False  # 0 extra calls: give stage 6 the raw test
    memo_scan: bool = False        # +k calls: forward ledger scan, replaces 5c
    memo_chapters: int = 10        # hard cap on that scan's calls per record
    prior_unranked: bool = False   # 0 calls: hand the localiser the prior's
                                   # regions WITHOUT an order or a "likely"
                                   # label, so a wrong top guess cannot anchor
    prior_to_localizer: bool = True  # 0 extra calls on an arm that already ran
                                   # with --prior-compress: set False and the
                                   # prior still allocates the compression
                                   # budget at 3b but never appears in the
                                   # localiser prompt. Separates the prior's
                                   # two jobs, which ta_noprior removed together
    localize_thinking: bool = False  # 0 calls: add deep reasoning prompt to
                                   # localizer, forcing step-by-step analysis
                                   # before answering (CoT-style)
    # ALFWorld only. "" keeps every TA/SP run on the incumbent head and note,
    # and the call site below does not even pass the kwarg when it is "", so a
    # run without it sends a byte-identical prompt to the same cache key.
    alf_prompt: str = ""
    # ALFWorld only, and the OPPOSITE intervention to ``alf_anchors``: that one
    # repaints the flow graph wider, this one suppresses it. Both follow from
    # the same measurement -- the graph is rooted at step 59 on 99/100 records
    # because the failure is silent -- which supports two contrary readings
    # (too narrow vs actively misleading). On the stored baseline, dev rows
    # whose gold the graph covers score 0.1429 against 0.3548 for rows it does
    # not, which is the suppression reading's prediction. "" passes no kwarg at
    # the call site, so TA/SP prompts and cache keys are untouched.
    alf_flownote: str = ""
    # ALFWorld only. "" leaves ``cands`` exactly as candidates.py computed it,
    # so every TA/SP run is unaffected by the mere existence of this field.
    # Set to one of alfworld_space.SPACES to REPLACE the answer space with a
    # positional one. See probe/prereg_alf_040.md: the write-op parse degrades
    # to [] on this corpus (nothing is ever written) and the state-change port
    # of it was killed with zero calls, so without this the localiser runs
    # unrestricted in all 30 slots. Recall ceiling on dev is 0.7632.
    alf_space: str = ""
    alf_window: int = alfworld_space_mod.DEFAULT_WINDOW
    # ALFWorld only, 0 calls. False leaves every digest exactly as compression
    # left it, and the call site below is guarded, so no TA/SP prompt changes
    # by a byte. True hoists each assistant step's own <action> to the front of
    # its digest: measured over all 3000 assistant steps, the action is 1.0% of
    # the text and sits at the 98% mark, while 86% of steps cite a DIFFERENT
    # step by number. See probe/prereg_alf_fix.md section 1.4.
    alf_actions: bool = False
    # ALFWorld only, 0 calls. "" leaves graph.anchors exactly as located (or as
    # --rule-anchors/--hybrid-anchors left it), so no TA/SP run paints from a
    # different root or changes a cache key. Set to one of
    # alfworld_anchors.VARIANTS to UNION the located anchors with the steps that
    # revisit an already-visited fixture. Measured on 100/100: the located
    # anchor is step 59 on 99 of them and sits a median of 40 steps from the
    # reference, because an ALFWorld failure is silent -- there is no traceback
    # to anchor on. Repainting the stored graphs offline moves gold_on_flow
    # 0.3700 -> 0.6600 and |anchor - gold| 40 -> 4 at zero calls.
    # See probe/prereg_alf_anchor.md.
    alf_anchors: str = ""
    # ALFWorld only, 0 upstream calls. Replace only the final localiser's
    # trajectory body with 30 compact state-transition rows. The extractor
    # reads the trajectory text only; flow, spans, outline and prior stay the
    # same, and the kwarg is omitted entirely while this is False.
    alf_ledger: bool = False
    # ALFWorld only. Keep the incumbent answer, then audit memory, reflection,
    # plan and action/system over separate evidence views and arbitrate only
    # among their validated nominations. Default off and mutually exclusive
    # with the alternative ALF renderers in the intended experiments.
    alf_modules: bool = False
    # ALFWorld only. After the unchanged incumbent localiser, compare only its
    # immediate attributable neighbours using exact local transitions. One
    # call, closed three-step answer set, default off.
    alf_local3: bool = False
    # ALFWorld only, 0 extra calls. Add a code-derived objective obligation
    # timeline to the existing localiser prompt. It neither restricts the
    # answer space nor adds a post-hoc decision stage.
    alf_obligations: bool = False
    localize_rootcause: bool = False  # 0 calls: when the step the localiser is
                                   # about to name ACTED (an edit/create/patch),
                                   # make it test whether that step merely
                                   # executed a decision an earlier plan/reason
                                   # step had already got wrong -- and keep the
                                   # acting step unless the earlier one made the
                                   # fault unrecoverable. Two-sided by
                                   # measurement (probe/rootcause_headroom.py):
                                   # 15 edit-ish rows to win on TA against 25 it
                                   # could break, 24 against 21 on SP, so the
                                   # block is written as a test with a default,
                                   # not as a nudge to walk back
    localize_pointwise: bool = False  # +k calls: pointwise scoring instead of
                                   # listwise ranking. Each candidate scored
                                   # independently (no comparison), argmax wins.
                                   # Costs ~10-20 calls/case but avoids being
                                   # anchored by the most salient step
    localize_multiturn: bool = False  # +4 calls: coarse-to-fine localization.
                                   # Round 1: phase (6→1), Round 2: span (5→1),
                                   # Round 3: step (7→1), Round 4: verify.
                                   # Avoids information overload and salience
                                   # bias by narrowing context progressively
    op_analysts: bool = False      # +<=5 calls: one analyst per operation type
                                   # (edit/reason/read/search/test), each shown
                                   # ONLY its own steps, then the localiser
                                   # arbitrates. The partition is the point:
                                   # --multi-expert split by failure family
                                   # over SHARED evidence and converged (1.6
                                   # distinct answers of 4, acc 0.3333)
    mixed_paint: bool = False      # 0 calls: let a chain alternate data/control
    rule_anchors: bool = False     # -0 calls: pick the failure anchors by
                                   # keyword + recency instead of by the model's
                                   # read of the digest. The call still happens
                                   # (failure_summary/entities feed six prompts);
                                   # only the anchor indices change
    unit_spans: bool = False       # 0 extra calls: segment by grouping a
                                   # rule-built operation chain (observation ->
                                   # plan -> localization -> edit -> test) rather
                                   # than by cutting a per-step digest
    prior_compress: bool = False   # 0 extra calls: run the span prior BEFORE
                                   # compression and let it decide what survives
                                   # -- flagged phases verbatim, unflagged ones
                                   # compressed harder. Implies --unit-spans:
                                   # the prior reads the rule digest, which only
                                   # unit segmentation produces. The prior is
                                   # computed once and reused at stage 5c
    prior_budget_chars: int = 90000  # cap on kept text before the plan degrades
    possible_keep: bool = False    # 0 extra calls: hand over a flowless step
                                   # VERBATIM whenever the prior called its
                                   # phase suspicious at all, instead of only
                                   # when it called it likely. POSSIBLE stops
                                   # being a clip and becomes a keep; it is
                                   # still the first band to give ground when
                                   # the budget runs out, so LIKELY is never
                                   # paid for out of it. Needs --prior-compress
    full_keep: bool = False        # 0 extra calls, NO budget: the three-rule
                                   # scheme -- on the flow => whole step; off
                                   # the flow but inside a phase the prior
                                   # called suspicious (likely OR possible) =>
                                   # whole step; otherwise compressed. Every
                                   # band limit becomes the step's own length
                                   # and the degrade ladder does not run, so
                                   # prior_budget_chars is ignored. Implies
                                   # possible_keep. See plan_compression's
                                   # docstring for the stage-6 caveat
    span_judge: bool = False       # +1 call per FLOWLESS phase: score that phase
                                   # on its own (relevance x confidence) and let
                                   # the product pick a compression band. Phases
                                   # carrying flow are skipped -- their steps are
                                   # ACT_KEEP already and the degrade ladder may
                                   # not touch them, so a score buys nothing.
                                   # The score goes to the COMPRESSOR, where
                                   # arithmetic over a character budget is the
                                   # job; SpanPrior refuses a number for the
                                   # LOCALISER and that refusal still stands
    judge_to_localizer: bool = False  # 0 extra calls: ALSO render the scores
                                   # into the localiser's prompt. Separate arm,
                                   # run second and judged on its own, because
                                   # giving the localiser more pointed evidence
                                   # is the axis that has now lost four times
    judge_workers: int = 8         # concurrency for the per-phase judges
    hybrid_anchors: bool = False   # 0 calls: paint from the model's anchors
                                   # UNION the keyword rule's, instead of
                                   # letting the rule replace them. The two sets
                                   # are nearly disjoint (Jaccard 0.116 SP /
                                   # 0.267 TA) and the union only enlarges
                                   # on_flow, so gold_on_flow moves +4/-0 on
                                   # both corpora by construction. Mutually
                                   # exclusive with rule_anchors
    candidate_set: bool = False    # 0 calls: offer the answer the write steps
    candidate_strict: bool = False  # and refuse any answer outside them
    plan_candidates: bool = False  # 0 calls: ALSO offer the prose steps that
                                   #    named a file and stated a conclusion
                                   #    about it. A write-only answer space
                                   #    fits SWE-Bench-Pro (76.7% of its
                                   #    reference steps are writes) and
                                   #    half-fits TrajAudit (52.0%), whose
                                   #    label is often the step that committed
                                   #    to the wrong approach before any edit.
                                   #    Capped at plan_cap so the list stays a
                                   #    restriction. SP control: +0 rows
    plan_cap: int = 15             # keep the EARLIEST this many when it binds
    shortlist: bool = False        # +1 call: a reader of the whole compressed
                                   #    run names up to shortlist_k MORE
                                   #    candidate steps, merged into the answer
                                   #    space. The write-matching parse cannot
                                   #    see a step whose only act was to decide:
                                   #    on SP it misses the reference step on 20
                                   #    rows and on 15 of those the reference
                                   #    step is pure prose. Needs candidate_set
    shortlist_k: int = 5           # cap on the NEW steps one call may add.
                                   #    There is no flag for keeping the
                                   #    proposals out of the verify poll: that
                                   #    was tried, and it made answering with
                                   #    one self-defeating (68 rows chosen, 68
                                   #    overridden). Answerable and able to vote
                                   #    are the same set now
    protect_candidates: bool = False  # 0 calls: keep the answer space verbatim
    # 0 calls each. Two independent uses of candidates.handoffs(), which
    # reads only the author field. Separate flags on purpose: the first
    # changes what the localiser READS, the second what it may ANSWER, and
    # the second is answer-space restriction -- an axis measured net
    # negative on TrajAudit -- so they must be judgeable apart.
    handoff_protect: bool = False   # keep control-transfer steps verbatim
    handoff_candidates: bool = False  # and/or make them the answer space
    candidates_verbatim: bool = False  # 0 calls: restore it AFTER compression
    cand_evidence: bool = False    # 0 calls: the answer space as a comparison
    framings: int = 1              # +k-1 calls: ask the same question k ways
    verify: bool = False           # +1 call per candidate: judge each alone
    verify_graded: bool = False    # ...and grade it 0-3 instead of yes/no
    verify_chain: bool = False     # ...and show it the edges out of the step
    verify_reflect: bool = False   # ...REPLACE with 1 listwise reflection call
    verify_duel: bool = False      # ...and play a tied poll off pairwise
    verify_pool: bool = False      # ...and poll the traced path where no
                                   #    write-based candidate set was found
    verify_pool_union: bool = False  # ...and where there IS one, poll the set
                                   #    UNION the traced path. The only flag
                                   #    that widens the answer space without
                                   #    discarding the set: --candidates-strict
                                   #    off changes nothing, because the poll
                                   #    draws from the set and re-restricts.
    verify_outdeg: bool = False    # +0 calls: add capped traced out-degree to
                                   #    the rubric score when merging (SP only)
    multi_expert: bool = False     # +3 calls: four specialists, then a vote
    roll_call: bool = False        # +0 calls: make the localiser account for
                                   #    every traced step before it answers
    expect_spec: bool = False      # +2 calls per candidate: say what each one
                                   #    was required to produce BEFORE reading
                                   #    it, then check it against that
    samples: int = 1               # +k-1 calls: vote over k draws of one prompt
    zoom: bool = False             # +1 call: re-read the finalists verbatim
    zoom_window: int = 5           # half-width when there is no candidate list
    bracket: bool = False          # +~6 calls: contiguous groups, then a final
    bracket_window: int = 5        # window unioned with the candidate list
    bracket_max: int = 20          # cap on the finalist set
    bracket_group: int = 4         # finalists per group call
    audit: bool = False            # +1 call: ask whether a DIRECT flow parent
                                   #    of the answer already introduced the
                                   #    error. Moves earlier only, and only on
                                   #    all four conditions -- see audit.py
    audit_max: int = 6             # cap on the parents examined (mean is 2.0)
    error_anchors: bool = False    # re-anchor on the steps judged to be wrong
    keep_diagnostics: bool = True  # attach flow/spans/outline to the Prediction
    # persistence
    trace_dir: Optional[str] = None   # where per-trajectory artifacts go
    reuse_flow: bool = False          # reuse a stored flow instead of re-tracing
    save_spans: bool = True           # include span summaries in the artifact
    save_steps: bool = True           # include step digests in the artifact

    def fingerprint(self) -> dict:
        """The knobs that affect the traced flow, recorded in the artifact."""
        d = dataclasses.asdict(self)
        d.pop("trace_dir", None)
        d.pop("reuse_flow", None)
        return d


def run(traj: Trajectory, llm: LLM, config: Optional[Config] = None,
        store: Optional[TraceStore] = None,
        llm_localize: Optional[LLM] = None,
        llm_trace: Optional[LLM] = None) -> Prediction:
    """Locate the critical step of one trajectory.

    ``store`` overrides ``config.trace_dir``; pass one shared instance when
    running many trajectories concurrently.

    ``llm_localize`` runs stage 6 (and the 6d poll inside it) on a different
    model while every earlier stage keeps ``llm``. Measured reason for the
    seam: given the reference step is on the traced flow -- verbatim in the
    prompt, nothing left to find -- it is still picked only 0.571 of the time
    on SWE-Bench-Pro and 0.569 on TrajAudit. The two corpora agreeing to
    three decimals says that ceiling belongs to the reader, not the corpus,
    and in 0.70/0.83 of those misses the localiser's own stated reason never
    mentions the reference step at all: it is not rejected, it is not seen.
    Every other axis assumes a better-helped reader chooses better; this one
    changes the reader. Leaving it None keeps the single-model path and every
    prompt byte-identical, so all earlier stages stay on their cached model.

    ``llm_trace`` is the same seam one stage earlier: stage 1 (backward
    tracing) runs on a different model while every later stage keeps
    ``llm``. Measured reason for this seam: on a 258-step-median corpus
    the adjacent `assistant->tool` edge is built only 0.705 of the time,
    the graph fragments into 8-60 weakly-connected components, and
    `paint()` from the tail anchors reaches 0.055 of the steps -- so
    ``gold_on_flow`` is 0.000 there where TrajAudit gets 0.696. Whether
    that is the tracer's judgement or the 209-step backtrack distance
    compounding is exactly what swapping only this model answers.
    Leaving it None keeps the single-model path and every prompt
    byte-identical.
    """
    cfg = config or Config()
    # Stage 6's reader. Identical object when the seam is unused, so every
    # `lz is llm` path below collapses to the single-model behaviour.
    lz = llm_localize or llm
    # Stage 1's tracer. Same collapse rule: identical object when unused.
    tr = llm_trace or llm
    if store is None and cfg.trace_dir:
        store = TraceStore(cfg.trace_dir)
    t0 = time.time()
    before = llm.snapshot()
    before_lz = lz.snapshot() if lz is not llm else None
    before_tr = tr.snapshot() if tr is not llm and tr is not lz else None

    # 1. backward tracing: symptom -> upstream data and control dependencies
    graph = None
    if store is not None and cfg.reuse_flow:
        graph = store.load_graph_for(traj)
        if graph is not None:
            logger.info("%s: reusing stored flow (%d edges)",
                        traj.task_id, len(graph.edges))
    if graph is None:
        graph = tracing_mod.trace(traj, tr, max_anchors=cfg.max_anchors,
                                  window=cfg.flow_window,
                                  workers=cfg.workers,
                                  backward=cfg.backward,
                                  max_rounds=cfg.max_rounds,
                                  batch_size=cfg.batch_size,
                                  max_traced_steps=cfg.max_traced_steps,
                                  topup_rounds=cfg.topup_rounds,
                                  topup_calls=cfg.topup_calls,
                                  origin_edges=cfg.origin_edges,
                                  index_per_step=cfg.flow_index_per_step,
                                  index_chars=cfg.flow_index_chars)
        # 2. persist the flow now: it is the expensive artifact
        if store is not None:
            try:
                store.write_flow(traj, graph, model=tr.model,
                                 config=cfg.fingerprint())
            except Exception:                               # noqa: BLE001
                logger.exception("%s: failed to save flow", traj.task_id)

    # 2b. how the rest of the run reads that flow. Set AFTER tracing, so the
    #     graph is built identically either way and the two arms of the A/B
    #     differ only downstream -- which is also what lets a stored flow be
    #     reused for both (``to_dict`` does not carry this).
    graph.mixed_paint = cfg.mixed_paint

    # 2c. WHICH anchors painting starts from: the located (symptom) anchors AND
    #     the in-degree (confluence) anchors, both treated as failure anchors.
    #     Unlike ``mixed_paint`` this one IS serialised, because
    #     ``write_complete`` re-dumps the graph at the end of the run -- so reset
    #     from the located anchors FIRST. That makes the re-anchoring idempotent
    #     in both directions: a rerun over a trace dir a previous run already
    #     wrote back still starts from what ``locate_anchors`` actually said,
    #     instead of compounding a union onto a union.
    graph.anchors = list(graph.anchors_located or graph.anchors)
    #     Rule anchoring replaces those indices with ``keyword_anchors``: the
    #     preference order in ANCHOR_PROMPT is three surface patterns (a submit
    #     call, a traceback or FAILED line, recency), and the located anchors sit
    #     at relative position median 0.982, so the model is mostly answering
    #     "the end". This applies here rather than inside ``trace`` so that it
    #     also reaches a flow loaded with --reuse-flow -- and, like every other
    #     line in this block, it reads ``anchors_located``, so re-running over a
    #     trace dir a previous run wrote back is idempotent.
    if cfg.rule_anchors:
        picked, detail = tracing_mod.keyword_anchors(traj, k=cfg.max_anchors,
                                                     detail=True)
        if picked:
            detail["located"] = list(graph.anchors_located or [])
            graph.anchor_rule = detail
            graph.anchors = list(picked)
    #     Hybrid anchoring UNIONS the two instead of letting the rule replace the
    #     model. Measured reason: the two sets are nearly disjoint -- Jaccard
    #     0.116 on SWE-Bench-Pro, 0.267 on TrajAudit -- so they are complementary
    #     signals, not two implementations of one. 63/86 and 73/98 of the MODEL's
    #     anchors carry no symptom evidence at all (it is mostly answering "the
    #     end"), and the rule's anchors are exactly the evidence-bearing ones it
    #     missed.
    #
    #     The union can only ENLARGE on_flow, so re-painting stored graphs gives
    #     +4/-0 reference steps on both corpora (SP 42->46 of 86, TA 67->71 of
    #     98) -- guaranteed by construction rather than measured luck. Against
    #     --rule-anchors, which REPLACES, the SP figure is 25->46: the
    #     replacement cost 17 reference steps their protected band. Repairing
    #     that fully (probe/sp_hy_fx.jsonl) bought +1 row of acc@0, and the 21
    #     repaired rows themselves went 9->8, so the lost band is NOT that arm's
    #     cause of loss -- the whole +1 came from the 25-row control bucket whose
    #     reference step was on flow either way (14->17).
    #
    #     Cost: mean on-flow steps 18.1->24.8 (SP) and 15.3->19.5 (TA). Rows
    #     exceeding the 90000-char plan budget are 28->43 with mean 82187->92797
    #     chars (+12.9%) when counted the way _degrade itself counts, via
    #     kept() = min(len(content), limit); the 0->1 and 1->4 figures this
    #     comment used to carry summed the bare limits and understated it. The
    #     ladder does fire. Filtering the union down to evidence-bearing rule
    #     anchors only is WORSE (43/86, 69/98), so it is the full union or
    #     nothing. Mutually exclusive with --rule-anchors, which the CLI refuses.
    if cfg.hybrid_anchors:
        picked, detail = tracing_mod.keyword_anchors(traj, k=cfg.max_anchors,
                                                     detail=True)
        located = list(graph.anchors_located or graph.anchors)
        if picked:
            detail["located"] = list(located)
            detail["rule_only"] = list(picked)
            detail["mode"] = "hybrid-union"
            graph.anchor_rule = detail
            graph.anchors = sorted(set(located) | set(picked))
    #     The re-anchoring was adopted on flow_recall, at a time when acc@0 was
    #     explicitly not the criterion. Measured on acc@0 (E4) it is a net loss
    #     on both corpora and on every split -- SP 0.3023 -> 0.2558, TA 0.4804
    #     -> 0.4510 -- so it is off by default and kept only as an ablation.
    #     It also explains why every stored artifact predating it disagreed with
    #     a rerun: switching it off reproduces those files case for case.
    if cfg.error_anchors:
        graph.anchors = tracing_mod.failure_anchors(traj, graph,
                                                    k=cfg.max_anchors)
    #     ALFWorld only, 0 calls, and LAST so that the three branches above are
    #     reached exactly as before -- with the flag off, ``anchors`` returns
    #     None and this block cannot touch the graph. It unions rather than
    #     replaces, for the reason the hybrid branch records: a union can only
    #     enlarge the painted set, so no reference step loses its protected
    #     band. Like every line in this block it reads ``anchors_located``, so
    #     re-running over a trace dir a previous run wrote back is idempotent,
    #     and because it sits here rather than inside ``trace`` it also reaches
    #     a flow loaded with --reuse-flow -- which is what makes this arm free:
    #     the edges are already stored and only the painting changes.
    if cfg.alf_anchors:
        _alf_a = alfworld_anchors_mod.anchors(
            cfg.alf_anchors, traj, list(graph.anchors_located or graph.anchors))
        if _alf_a is not None and _alf_a["anchors"]:
            _alf_a["mode"] = "alf-revisit-union"
            graph.anchor_rule = _alf_a
            graph.anchors = list(_alf_a["anchors"])

    # 2d. the answer space, computed HERE rather than just before localisation.
    #     It is a parse of the trajectory -- no calls, no dependency on anything
    #     upstream -- and it used to sit at step 5b, AFTER compression. That
    #     ordering was the bug ``protect_candidates`` fixes: compression decides
    #     what survives in its own words from ``graph.on_flow()`` alone, so it
    #     had no idea which steps the localiser would be required to answer
    #     with, and 31.8% of them were reaching it as a paraphrase.
    #     Returns [] when the parse found too few to trust, and the localiser
    #     then runs unrestricted rather than being handed an empty answer space.
    #     Whether straying outside the set is corrected is a SEPARATE knob: the
    #     list helped on one corpus and the correction hurt on the other.
    cands = ((candidates_mod.candidates_with_plans(traj, cfg.plan_cap)
              if cfg.plan_candidates else candidates_mod.candidates(traj))
             if (cfg.candidate_set or cfg.protect_candidates
                 or cfg.candidates_verbatim) else None)
    #     2d''. ALFWorld only, and a REPLACEMENT rather than a union for the
    #     same reason ``handoff_candidates`` is: the write-op pool and the
    #     positional pool disagree about what kind of thing an answer is, and
    #     on this corpus the write-op pool is empty anyway (``candidates: null``
    #     on 38 of 38 records measured). Guarded on ``cands is not None`` so
    #     that the flag cannot conjure an answer space none of the three
    #     consuming knobs asked for.
    if cfg.alf_space and cands is not None:
        cands = alfworld_space_mod.answer_space(traj, cfg.alf_space,
                                                cfg.alf_window)
    #     2d'. The handoff pool, parsed from the author field rather than
    #     from what the step did. Computed once and used by either or both
    #     of the two flags below; `or None` keeps the "no handoff structure"
    #     case indistinguishable from the flag being off, which is what
    #     makes this a no-op on the four corpora that have no such authors.
    hands = (candidates_mod.handoffs(traj)
             if (cfg.handoff_protect or cfg.handoff_candidates)
             else None) or None
    if cfg.handoff_candidates and hands:
        # Replaces rather than unions: the write-op pool and the handoff
        # pool disagree about what kind of thing an answer is, and a union
        # would be a third object neither arm measured.
        cands = hands

    # 3. semantic spans, tagged by whether they touch a flow
    if cfg.unit_spans or cfg.prior_compress:
        sp = spans_mod.segment_units(traj, llm, graph,
                                     target_spans=cfg.target_spans)
    else:
        sp = spans_mod.segment(traj, llm, graph, target_spans=cfg.target_spans)
    sp = spans_mod.assign_tiers(sp, graph, weak_ratio=cfg.weak_ratio)

    protect = None
    if cfg.protect_candidates and cands:
        protect = sorted({int(c["step"]) for c in cands})
    if cfg.handoff_protect and hands:
        # Union, not replace: --protect-candidates and --handoff-protect
        # answer different questions about the same step list and stack.
        # Measured reason this stage needs help at all: compression keeps
        # 83.7% of steps but only 36.8% of handoff steps, because it
        # chooses by length and these are the longest class in the corpus.
        # The protection set it uses instead is graph.on_flow(), which on
        # that corpus paints 5.5% of steps and reaches the reference step
        # 0.000 of the time -- so there is nothing for it to protect.
        protect = sorted(set(protect or ())
                         | {int(c["step"]) for c in hands})

    # 3b. optional: the span prior, moved AHEAD of compression so its verdict can
    #     decide what survives compression rather than arriving after the text is
    #     already gone. Phases it calls likely keep every step verbatim; phases it
    #     did not name compress harder than they do today.
    #
    #     The reason this could not be done before is an ordering cycle, not a
    #     missing feature: ``predict_spans`` read ``Span.summary``, which the
    #     COMPRESSOR writes. Unit segmentation breaks it -- ``Span.digest`` says
    #     what each phase did, is rule-derived, and exists before any call. That
    #     is why ``prior_compress`` implies ``unit_spans`` above: without the
    #     digest the prior would read "(no summary)" for every phase and be
    #     guessing off labels alone.
    #
    #     No second call at 5c. The localiser gets THIS prior, so the arm costs
    #     nothing over a plain --span-prior run and the only variable is where in
    #     the pipeline the prior was asked.
    pre_prior, plan = None, None
    verdicts = None
    #     3a. optional: a pointwise judge, one call per phase that carries NO
    #     traced flow. It runs here because its output is an input to the plan
    #     below, and it judges only those phases because a phase that carries
    #     flow already has those steps at ACT_KEEP and the degrade ladder may
    #     not touch them -- a score could not change their fate, so buying one
    #     would buy nothing. That restriction is also what makes the stage
    #     affordable: ~11 calls per SP row of 19 phases, ~3.4 per TA row of 9.
    #
    #     The position it addresses: 11 of SP's 19 median phases hold no flow at
    #     all, the listwise prior may name at most three phases of nineteen
    #     (coverage 0.0655, recall 0.163), and the 34 rows whose reference step
    #     sits in a flowless phase score acc@0 0.147 against 0.442 for the rest.
    #     TA's equivalent position is 18 rows at 0.222, which is near the noise
    #     floor; it runs there for reproducibility, not because it is expected
    #     to convert.
    if cfg.span_judge:
        verdicts = span_judge_mod.judge_spans(traj, sp, graph, llm,
                                              workers=cfg.judge_workers)
    if cfg.prior_compress:
        pre_prior = span_prior_mod.predict_spans(
            traj, sp, graph, llm,
            signals=cfg.prior_signals, symptom=cfg.prior_symptom,
            cands=(cands if cfg.prior_signals else None),
            samples=cfg.prior_samples)
        plan = compress_mod.plan_compression(
            traj, sp, graph, pre_prior, protect=(protect or ()),
            budget_chars=cfg.prior_budget_chars,
            possible_keep=cfg.possible_keep,
            full_keep=cfg.full_keep,
            # Withheld entirely when the judge did not run, so the plan a
            # --prior-compress arm produces is the one it produced before this
            # stage existed and its stored runs reproduce exactly.
            **({"verdicts": verdicts} if verdicts else {}))

    # 4. per-span summary + step-level compression (one call per span)
    # Forwarded only when asked for, so an arm with the knob off sends a
    # byte-identical span prompt and pays nothing for the option existing.
    kw = {}
    if protect:
        kw["protect"] = protect
    if plan is not None:
        kw["plan"] = plan
    sp, digests = compress_mod.compress(traj, sp, graph, llm,
                                        workers=cfg.workers, **kw)
    # 4b. optional: prefer the step's own words wherever the paraphrase was not
    #     buying anything. Costs no calls -- the span call already happened and
    #     still hits the cache -- so an A/B against a stored run changes only the
    #     localiser's prompt. Off by default: at 0 the digests are untouched.
    restored = compress_mod.restore_verbatim(traj, digests, cfg.verbatim_limit)
    # 4b'. ALFWorld only, 0 calls, and guarded so that a run without the flag
    #      does not even call it: every TA/SP digest keeps the bytes 4b left.
    #      Runs AFTER restore_verbatim on purpose -- restoring a step to its
    #      own words puts the <action> back at the 98% mark, so hoisting has to
    #      be the last word on the digest's text.
    if cfg.alf_actions:
        stats_led = alfworld_prompt_mod.lead_with_action(digests, traj)
    else:
        stats_led = 0
    stats = compress_mod.compression_stats(traj, sp, digests)
    stats["steps_restored"] = restored
    if cfg.alf_actions:
        stats["steps_action_led"] = stats_led
    ledger_body = None
    if cfg.alf_ledger:
        ledger_body, ledger_stats = alfworld_ledger_mod.render_ledger(traj)
        stats.update(ledger_stats.to_dict())
    obligation_block = None
    if cfg.alf_obligations:
        obligation_block, obligation_stats = \
            alfworld_obligations_mod.build_obligation_block(traj)
        stats.update(obligation_stats.to_dict())

    # 4c. optional: the ANSWER SPACE, unconditionally in its own words. The
    #     localiser is required to answer with one of these steps, so a
    #     candidate that reaches it as a summary of itself is the one case
    #     where compression spent the thing it exists to protect. Measured on
    #     the sp_rcb baseline: 437 of 921 pooled steps arrive altered, and 49
    #     of the 66 rows whose reference step is pooled at all.
    #
    #     Deliberately AFTER compression rather than as a plan band. The plan
    #     version is ``--protect-candidates`` and it also changes the span
    #     prompt and the budget split, so it changes every other step in the
    #     row too; it lost on both corpora in 2026-09. This changes the
    #     candidates' text and nothing else, which means every call above is
    #     byte-identical and comes back from the cache -- the arm costs only
    #     the localiser and verify calls, and the A/B has one variable.
    cv_steps: List[int] = []
    if cfg.candidates_verbatim and cands:
        cv_steps = [int(c["step"]) for c in cands]
        stats["candidates_verbatim"] = compress_mod.restore_candidates(
            traj, digests, cv_steps)

    # 5. ARTIFACT 1: how the task was executed, end to end
    outline = (outline_mod.build_outline(traj, sp, llm, graph)
               if cfg.build_outline else None)

    # 5b. the answer space was computed at 2d, before compression could rewrite
    #     it. Hand it to the localiser only when the list is meant to restrict
    #     the answer -- protecting it from compression is a separate decision.
    #
    #     ``handoff_candidates`` is listed here for the same reason
    #     ``candidate_set`` is: both mean "this list IS the answer space".
    #     Without it the handoff pool would be built at 2d', assigned to
    #     ``cands``, and then dropped on this line -- a flag that costs a
    #     parse and changes nothing. The condition is a disjunction rather
    #     than a replacement so that every arm with the new flag off keeps
    #     reaching this line with exactly the behaviour it had before.
    if not (cfg.candidate_set or cfg.handoff_candidates):
        cands = None

    # 5c. optional: a second reader's guess at WHICH PHASES hold the cause.
    #     Two ways to make it. ``span_prior`` reads the phase summaries in ONE
    #     call. ``memo_scan`` walks the run forward one chapter at a time
    #     carrying a ledger of the agent's beliefs and commitments, and never
    #     sees a chapter before it has read the ones before it. Both return a
    #     SpanPrior and nothing downstream can tell them apart, so an A/B
    #     between them measures region selection and nothing else. The localiser
    #     is free to overrule either.
    memo = None
    if cfg.memo_scan:
        prior, memo = memo_scan_mod.scan(traj, sp, graph, llm,
                                         digests=digests,
                                         max_chapters=cfg.memo_chapters,
                                         symptom=cfg.prior_symptom)
    elif cfg.span_prior:
        if pre_prior is not None:
            # Already asked at 3b, on the rule digests, and already spent. Asking
            # again on the post-compression summaries would cost a call to
            # produce a second opinion the compressor could no longer act on.
            prior = pre_prior
        else:
            prior_fn = (span_prior_enhanced_mod.enhanced_predict_spans
                        if cfg.prior_enhanced else span_prior_mod.predict_spans)
            prior = prior_fn(traj, sp, graph, llm,
                             signals=cfg.prior_signals,
                             symptom=cfg.prior_symptom,
                             cands=(cands if cfg.prior_signals
                                    else None),
                             **({"coverage_target": cfg.prior_coverage_target}
                                if cfg.prior_enhanced
                                else {"samples": cfg.prior_samples}))
    else:
        prior = pre_prior

    # 5b-2. optional: a reader that has read the whole compressed run names a
    #       few more steps the write-matching parse cannot see. C1 is built by
    #       pattern-matching writes, so a step whose only act was to decide is
    #       invisible to it -- on SP, 15 of the 20 rows where C1 misses the
    #       reference step miss it for exactly that reason, and snapping cannot
    #       recover them (median distance to the nearest C1 entry is 5 steps).
    #       One call. It is shown C1 and told not to repeat it, so the whole
    #       budget buys new entries.
    #
    #       Placed after 5c, not before it: with --prior-signals the span prior
    #       is shown the candidate list, and letting C3 reach it would change
    #       the region as well as the answer space and leave neither
    #       attributable. Placed before 5d for the opposite reason: 5d's
    #       referent must be the set the localiser is actually handed.
    #
    #       Abstention -- unparseable JSON, an empty list, or every proposal
    #       already in C1 -- leaves C3 == C1, every prompt below byte-identical
    #       and the whole row answered from the cache. Those rows are a control
    #       group that cost nothing to assemble.
    shortlist = None
    new_steps: List[int] = []
    cands_c1 = list(cands or [])
    if cfg.shortlist and cands:
        shortlist = shortlist_mod.propose(traj, sp, graph, llm, digests=digests,
                                         cands=cands, k=cfg.shortlist_k,
                                         max_prompt_chars=cfg.max_prompt_chars)
        cands, new_steps = shortlist_mod.merge(cands, shortlist)

    # 5d. optional: what each candidate step was REQUIRED to produce, decided
    #     before that step was read, and how much of it the step actually did.
    #     Two calls per candidate -- the expensive stage on this axis by a wide
    #     margin -- and it runs here, after 5b, so that "the candidate steps"
    #     means exactly the set the localiser is about to be handed. With no
    #     candidate set there is no referent and the stage records that it
    #     skipped rather than substituting a pool of its own.
    #
    #     It is given the flow graph and shown none of it: neither reader is
    #     told what went wrong. A reader that knows the symptom writes the
    #     requirement the symptom violates, and the compliance count then
    #     measures hindsight instead of the step. See ``expect``.
    expect = (expect_mod.assess(traj, graph, llm, cands, digests=digests,
                                workers=cfg.workers)
              if cfg.expect_spec else None)

    # 5e. optional: five analysts, one per operation type, each shown only the
    #     steps of its own type and asked whether the critical step is among
    #     them. Runs after compression because the slices are rendered from the
    #     digests -- an analyst sees a step in its own words exactly when the
    #     localiser would -- and before stage 6, whose prompt consumes the
    #     findings. Declining is allowed and expected: most runs fail in one
    #     kind of operation, so four of the five are reading clean material.
    findings = (op_analysts_mod.run_analysts(traj, sp, graph, llm,
                                             digests=digests,
                                             workers=cfg.workers)
                if cfg.op_analysts else None)
    analyst_block = (op_analysts_mod.render_findings(findings)
                     if findings else "")

    # 5f. the judge's block, rendered only for the arm that asks the localiser
    #     to read it. Empty when no phase earned a band, which keeps that arm's
    #     prompt identical to the first arm's on rows where the judge cleared
    #     everything -- so the comparison between the two is over the rows where
    #     the block actually exists, and the rest are not quietly counted as
    #     evidence either way.
    judge_block = (span_judge_mod.render_verdicts(verdicts, sp)
                   if (cfg.judge_to_localizer and verdicts) else "")

    # 6. localisation over artifact 1 + artifact 2. One general reader, or four
    #    specialists in parallel followed by a rule-based vote. The two take the
    #    same arguments and return the same Prediction; when ``multi_expert`` is
    #    off not one byte of the single localiser's prompt changes, so an A/B
    #    against a stored run measures the architecture and nothing else.
    if cfg.alf_modules:
        incumbent = localize_mod.localize(
            traj, sp, graph, lz, digests=digests, outline=outline,
            prior=prior, cands=cands, enforce_cands=cfg.candidate_strict,
            max_prompt_chars=cfg.max_prompt_chars)
        pred = alfworld_modules_mod.refine(traj, lz, incumbent,
                                           workers=cfg.workers)
    elif cfg.localize_multiturn:
        # Multi-turn: coarse-to-fine localization (phase → span → step → verify)
        from . import localize_multiturn
        pred = localize_multiturn.multiturn_localize(
            traj, graph, outline, digests, lz)
    elif cfg.localize_pointwise:
        # Pointwise: score each candidate independently, take argmax
        from . import localize_pointwise
        pred = localize_pointwise.pointwise_localize(
            traj, graph, lz, outline=outline, max_candidates=20)
    elif cfg.multi_expert:
        pred = experts_mod.localize_multi(traj, sp, graph, lz, digests=digests,
                                          outline=outline, prior=prior,
                                          cands=cands,
                                          enforce_cands=cfg.candidate_strict,
                                          max_prompt_chars=cfg.max_prompt_chars,
                                          workers=cfg.workers)
    else:
        pred = localize_mod.localize(traj, sp, graph, lz, digests=digests,
                                     outline=outline, prior=prior, cands=cands,
                                     enforce_cands=cfg.candidate_strict,
                                     max_prompt_chars=cfg.max_prompt_chars,
                                     samples=cfg.samples,
                                     **({"prior_unranked": True}
                                        if cfg.prior_unranked else {}),
                                     **({"hide_prior": True}
                                        if not cfg.prior_to_localizer else {}),
                                     **({"cand_evidence": True}
                                        if cfg.cand_evidence else {}),
                                     **({"framings": cfg.framings}
                                        if cfg.framings != 1 else {}),
                                     **({"verify": True}
                                        if cfg.verify else {}),
                                     **({"verify_graded": True}
                                        if cfg.verify_graded else {}),
                                     **({"verify_chain": True}
                                        if cfg.verify_chain else {}),
                                     **({"verify_reflect": True}
                                        if cfg.verify_reflect else {}),
                                     **({"verify_duel": True}
                                        if cfg.verify_duel else {}),
                                     **({"verify_pool": True}
                                        if cfg.verify_pool else {}),
                                     **({"verify_pool_union": True}
                                        if cfg.verify_pool_union else {}),
                                     **({"verify_outdeg": True}
                                        if cfg.verify_outdeg else {}),
                                     # 0 new calls: the same one call, required
                                     # to enumerate the traced steps before it
                                     # commits. Aimed at the measured failure
                                     # mode -- in 0.830 of TrajAudit misses the
                                     # reference step is never mentioned in the
                                     # localiser's own reason -- rather than at
                                     # the choice, which eight arms have now
                                     # tried to improve and none has.
                                     **({"roll": True}
                                        if cfg.roll_call else {}),
                                     # a COUNT per candidate, not a verdict:
                                     # the shape that converted on this corpus
                                     # (--verify-outdeg) as against the five
                                     # arms that handed the reader a second
                                     # judgement about criticality and lost.
                                     **({"expect": expect}
                                        if expect else {}),
                                     **({"symptom_evidence": True}
                                        if cfg.localize_symptom else {}),
                                     **({"thinking": True}
                                        if cfg.localize_thinking else {}),
                                     # Empty when no analyst nominated anything,
                                     # and the kwarg is withheld entirely when
                                     # the stage is off, so the prompt and the
                                     # cache key are byte-identical to a run
                                     # that predates this stage.
                                     **({"analyst_findings": analyst_block}
                                        if analyst_block else {}),
                                     # The second arm of the judge axis, and the
                                     # one on the losing side of the ledger:
                                     # handing the LOCALISER more pointed
                                     # evidence has now failed four times
                                     # (memo-scan, prior-unranked,
                                     # localize-symptom, op-analysts). Arm one
                                     # routes the same scores to the compressor
                                     # only and leaves this block absent, so the
                                     # two arms differ by exactly this kwarg and
                                     # the first one's prompt is byte-identical
                                     # to a run that predates the stage.
                                     **({"judge_block": judge_block}
                                        if judge_block else {}),
                                     # Asked for directly: make the localiser
                                     # test whether an edit step it is about to
                                     # name only executed an already-wrong
                                     # plan/reason step. Procedure, not
                                     # evidence -- it adds no fact about the
                                     # trajectory and names no step, which is
                                     # what separates it from the four arms
                                     # that lost on the evidence axis. Withheld
                                     # when off, so the prompt and cache key are
                                     # byte-identical to a run without it.
                                     **({"rootcause": True}
                                        if cfg.localize_rootcause else {}),
                                     **({"alf_prompt": cfg.alf_prompt}
                                        if cfg.alf_prompt else {}),
                                     **({"alf_flownote": cfg.alf_flownote}
                                        if cfg.alf_flownote else {}),
                                     **({"obligation_block": obligation_block}
                                        if obligation_block else {}),
                                     **({"trajectory_override": ledger_body}
                                        if ledger_body is not None else {}),
                                     # Only so the budget trimmer knows not to
                                     # clip back what 4c just restored; it
                                     # changes no text by itself.
                                     **({"cand_verbatim": cv_steps}
                                        if cv_steps else {}),
                                     # The merged space gets prose that
                                     # describes it: CANDIDATE_BLOCK is frozen
                                     # text and it says in so many words that
                                     # steps which only reasoned are not
                                     # listed, which C2 makes false. The
                                     # ENTRIES are rendered the same way either
                                     # way, one flat list in trajectory order
                                     # with nothing marking which came from
                                     # where. Withheld when the reader proposed
                                     # nothing, so those rows' prompts are
                                     # byte-identical to a run that predates
                                     # the stage -- the per-row control this
                                     # arm is judged against.
                                     **({"cands_mixed": True}
                                        if new_steps else {}))

    # 6a'. ALFWorld only: a deliberately local temporal-boundary check. It is
    # after the normal localiser so stages 1--6 and the incumbent cache key are
    # unchanged; off means the function is not called at all.
    if cfg.alf_local3:
        pred = alfworld_local3_mod.refine(traj, lz, pred)

    # The multi-expert localiser above does not take the compliance block, so
    # under that arm the stage is measured but not consulted. Recording it
    # anyway keeps the calls from being paid for and thrown away, and makes the
    # combination legible in the artifact instead of silently degrading.
    if expect is not None and pred.expect is None:
        pred.expect = expect

    # 6b. optional second pass: re-read the finalists in their own words. Runs
    #     after the answer exists because the finalist set is built around it,
    #     and the first answer stays in the record either way, so an acceptance
    #     rule can be changed offline without re-calling.
    if cfg.zoom:
        pred = zoom_mod.zoom(traj, graph, llm, pred, cands=cands,
                             window=cfg.zoom_window)

    # 6c. optional: the same gap attacked by changing the question instead of
    #     repeating it. The finalists are cut into contiguous groups judged
    #     independently and in parallel, then the winners meet. Mutually
    #     exclusive with 6b in practice -- both revise the same answer, and
    #     stacking them would make neither attributable -- but not forbidden,
    #     because each keeps the answer it received in its own record.
    if cfg.bracket:
        pred = bracket_mod.bracket(traj, graph, llm, pred, cands=cands,
                                   outline=outline,
                                   window=cfg.bracket_window,
                                   cap=cfg.bracket_max,
                                   size=cfg.bracket_group,
                                   workers=cfg.workers)

    # 6d. optional: audit the answer's DIRECT causal predecessors. Unlike 6b
    #     and 6c this does not re-ask which step is critical -- it asks the one
    #     question the graph can pose that the search cannot, "did a parent
    #     already introduce this", and it can only move the answer EARLIER.
    #     Stacks safely with 6b/6c: it reads whatever answer it is handed and
    #     keeps that answer in its own record.
    if cfg.audit:
        pred = audit_mod.audit(traj, graph, llm, pred, cap=cfg.audit_max)

    after = llm.snapshot()
    pred.usage = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    if before_lz is not None:
        after_lz = lz.snapshot()
        for k, v in after_lz.items():
            pred.usage[k] = pred.usage.get(k, 0) + v - before_lz.get(k, 0)
    if before_tr is not None:
        after_tr = tr.snapshot()
        for k, v in after_tr.items():
            pred.usage[k] = pred.usage.get(k, 0) + v - before_tr.get(k, 0)
    pred.usage["seconds"] = round(time.time() - t0, 1)
    pred.compression_ratio = round(stats["ratio"], 4)
    # The scan's own step nominations, kept but deliberately unused this round:
    # the answer stays with the localiser so a change in accuracy is
    # attributable to the region and nothing else.
    if memo is not None:
        pred.memo = dict(memo)
    # The analysts' own nominations, kept but deliberately not used as the
    # answer: the localiser arbitrates, so a change in accuracy is attributable
    # to the arbitration and not to a rule that picked among five readers.
    if findings is not None:
        pred.op_analysts = op_analysts_mod.stats(findings)
    # Which band of the compression plan kept which step, and which steps the
    # keyword rule anchored on. Diagnostics only -- nothing reads them back --
    # but without them an arm that wins is unattributable to any one of the
    # three bands, which is the whole question this design poses.
    if plan is not None:
        pred.compression_plan = compress_mod.plan_stats(plan)
    # Every per-phase score, cleared phases included. Without the cleared ones
    # the record could not answer the stage's own claim -- that a pointwise
    # reader CAN say "nothing here", which the listwise prior structurally
    # cannot -- and it is also what makes the two thresholds re-cuttable offline
    # instead of by paying for the calls again.
    if verdicts:
        pred.span_judge = span_judge_mod.stats(verdicts)
    # What the reader proposed, what was dropped and why, and how big C1 and C3
    # ended up. Kept on abstentions too: "it was asked and added nothing" is the
    # per-row control this axis is judged against, and a record that dropped
    # those rows could not say which rows the stage could possibly have moved.
    if shortlist is not None:
        pred.shortlist = shortlist_mod.stats(shortlist, cands_c1, cands,
                                             new_steps)
    if getattr(graph, "anchor_rule", None):
        pred.anchor_rule = graph.anchor_rule
    if cfg.keep_diagnostics:
        pred.flow = graph.to_dict()
        pred.spans = [s.to_dict() for s in sp]
        pred.outline = outline.to_dict() if outline is not None else None

    # 7. persist the full record
    if store is not None:
        try:
            store.write_complete(
                traj, graph, (sp if cfg.save_spans else []), pred,
                digests=(digests if cfg.save_steps else []), outline=outline,
                compression=stats,
                # Recorded distinguishably on purpose: the invalid ta_vpool run
                # proved the stored config is the only reliable way to tell two
                # arms apart after the fact, and this seam is invisible in
                # cfg.fingerprint() because the model lives on the LLM.
                # Same for the stage-1 seam: a flow built by another
                # model is a different artifact, and the composed id is
                # the only place on disk that says so. Both seams unused
                # collapses to bare llm.model, byte-identical to before.
                model=("%s%s%s" % (
                    ("trace:%s+" % tr.model) if tr is not llm else "",
                    llm.model,
                    ("+localize:%s" % lz.model) if lz is not llm else "")),
                config=cfg.fingerprint())
        except Exception:                                   # noqa: BLE001
            logger.exception("%s: failed to save artifact", traj.task_id)

    logger.info("%s: n=%d -> step %s (conf %.2f) ratio=%.3f spans=%d "
                "steps kept/clip/comp=%d/%d/%d flow=%d edges calls=%d %.1fs",
                traj.task_id, len(traj), pred.critical_step, pred.confidence,
                stats["ratio"], stats["spans"], stats["steps_kept"],
                stats["steps_clipped"], stats["steps_compressed"],
                len(graph.edges), pred.usage.get("calls", 0),
                pred.usage["seconds"])
    return pred
