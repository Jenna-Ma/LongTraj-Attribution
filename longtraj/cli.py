"""Command-line entry point.

    # inspect what the loaders see (no LLM needed)
    python -m longtraj.cli stats
    python -m longtraj.cli peek --source trajaudit --limit 2

    # run the pipeline; per-trajectory flow graphs land in traces/
    python -m longtraj.cli run --source trajaudit --subset swe-agent --limit 10 \
        --out results/trajaudit.json --trace-dir traces

    # dry run: exercises every stage with the deterministic fallbacks only
    python -m longtraj.cli run --source longrca --limit 3 --dry-run

    # inspect what was saved
    python -m longtraj.cli flows --trace-dir traces
    python -m longtraj.cli flows --trace-dir traces --task-id <id> --full
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from typing import List

from .artifacts import TraceStore, render_flow_text
from .evaluate import (TOLERANCES, aggregate, format_report, group_by,
                       score_one, write_results)
from .llm import LLM, account_is_dead, map_concurrent
from .loaders import LOADERS, find_repo_root, load
from .pipeline import Config, run as run_one


def _add_common(p):
    p.add_argument("--root", default=None, help="repo root holding the corpora")
    p.add_argument("--source", default="trajaudit", choices=sorted(LOADERS))
    p.add_argument("--subset", default=None)
    p.add_argument("--limit", type=int, default=None)


def cmd_stats(args) -> int:
    root = args.root or find_repo_root()
    print("repo root: %s" % root)
    for name in sorted(LOADERS):
        try:
            recs = list(LOADERS[name](root=root, subset=None, limit=args.limit))
        except Exception as exc:                              # noqa: BLE001
            print("  %-10s ERROR %s" % (name, exc))
            continue
        if not recs:
            print("  %-10s (nothing found)" % name)
            continue
        lens = sorted(len(r.trajectory) for r in recs)
        labelled = sum(1 for r in recs
                       if r.ground_truth and r.ground_truth.critical_step is not None)
        subsets = sorted({r.trajectory.subset for r in recs})
        print("  %-10s n=%-5d steps: median=%-4d max=%-4d  labelled=%-5d"
              % (name, len(recs), lens[len(lens) // 2], lens[-1], labelled))
        print("             chars/traj median=%d"
              % sorted(r.trajectory.total_chars() for r in recs)[len(recs) // 2])
        print("             subsets: %s" % ", ".join(subsets[:8]))
    return 0


def cmd_peek(args) -> int:
    for rec in load(args.source, root=args.root, subset=args.subset,
                    limit=args.limit or 1):
        t = rec.trajectory
        print("=" * 78)
        print("%s  [%s/%s]  steps=%d chars=%d"
              % (t.task_id[:60], t.source, t.subset, len(t), t.total_chars()))
        print("task: %s" % (t.task_description or "")[:300].replace("\n", " "))
        for s in t.steps[:3]:
            print("-" * 70)
            print(s.render(300))
        if len(t) > 4:
            print("-" * 70)
            print("... %d steps omitted ..." % (len(t) - 4))
            print(t.steps[-1].render(300))
        print("(held-out label not shown)")
    return 0


def _alf_half(task_id: str) -> str:
    """Dev/test half of one task id -- the SAME hash probe/split_eval.py uses.

    Kept byte-compatible with that file on purpose: if the run side and the
    scoring side disagreed about which half a record is in, "tuned on dev,
    revealed test once" would silently stop being true. A test pins the two
    implementations against each other. Only the task id is read; no label is
    involved, so this is a sampling decision, not a look at the answer.
    """
    h = hashlib.sha256(("longtraj/split/v1|" + task_id).encode("utf-8")).hexdigest()
    return "dev" if int(h[:8], 16) % 2 == 0 else "test"


def _load_records(args):
    """Load records, routing the one source that takes extra kwargs.

    ``loaders.load`` has a fixed three-argument signature shared by every
    source, and it stays that way: a flag only alfworld understands must not
    become an argument every loader has to tolerate. So the alfworld loader is
    called directly here and everything else goes through ``load`` unchanged.
    """
    if args.source != "trajdebug_alf":
        return list(load(args.source, root=args.root, subset=args.subset,
                         limit=args.limit))
    half = getattr(args, "alf_half", None)
    # --limit is applied AFTER the half filter, not by the loader, or "the
    # first N" and "the first N in this half" would be different corpora.
    recs = list(LOADERS["trajdebug_alf"](
        root=args.root, subset=args.subset,
        limit=None if half else args.limit,
        dedup=bool(getattr(args, "alf_dedup", False))))
    if half:
        recs = [r for r in recs if _alf_half(r.trajectory.task_id) == half]
        if args.limit:
            recs = recs[:args.limit]
    return recs


def cmd_run(args) -> int:
    # Flag conflicts first, before the corpus is read and before credentials are
    # touched: refusing a bad combination must not depend on a corpus being
    # present or an account being in good standing.
    #
    # Two anchor policies, one ``graph.anchors``. Replacing and unioning are
    # opposite answers to the same question and the second assignment would
    # silently win, so refuse rather than pick.
    if args.rule_anchors and args.hybrid_anchors:
        print("--rule-anchors replaces the model's anchors, --hybrid-anchors "
              "unions with them; pick one", file=sys.stderr)
        return 2
    if args.judge_to_localizer and not args.span_judge:
        print("--judge-to-localizer needs --span-judge: there are no scores to "
              "show the localiser otherwise", file=sys.stderr)
        return 2
    if args.possible_keep and not (args.prior_compress or args.span_judge):
        print("--possible-keep needs --prior-compress: the possible band it "
              "promotes only exists in the pre-compression plan", file=sys.stderr)
        return 2
    if args.full_keep and not (args.prior_compress or args.span_judge):
        print("--full-keep needs --prior-compress: without the pre-compression "
              "plan there is no prior verdict to keep whole steps for",
              file=sys.stderr)
        return 2
    if args.full_keep and args.prior_budget_chars != 90000:
        print("--full-keep ignores --prior-budget-chars (the ladder does not "
              "run); drop one of the two")
    if args.shortlist and not (args.candidates or args.candidates_strict):
        print("--shortlist adds to an answer space; without --candidates or "
              "--candidates-strict there is nothing to add to, and the one call "
              "would be the only thing restricting the answer", file=sys.stderr)
        return 2
    if args.plan_candidates and not (args.candidates or args.candidates_strict):
        print("--plan-candidates widens an answer space; without --candidates "
              "or --candidates-strict no answer space is built at all",
              file=sys.stderr)
        return 2
    if args.plan_cap < 2:
        print("--plan-cap below MIN_CANDIDATES leaves nothing to restrict with",
              file=sys.stderr)
        return 2
    if args.span_judge and not args.prior_compress:
        print("--span-judge implies --prior-compress (and so --unit-spans)")
    if args.verify_reflect and not args.verify:
        print("--verify-reflect requires --verify", file=sys.stderr)
        return 2
    if args.verify_reflect and (args.verify_graded or args.verify_chain
                                or args.verify_duel):
        print("--verify-reflect is mutually exclusive with --verify-graded, "
              "--verify-chain, and --verify-duel", file=sys.stderr)
        return 2

    records = _load_records(args)
    if not records:
        print("no trajectories matched", file=sys.stderr)
        return 1

    llm = LLM(model=args.model, base_url=args.base_url, api_key=args.api_key,
              temperature=args.temperature, cache_dir=args.cache_dir,
              dry_run=args.dry_run, config=args.config,
              config_required=not args.dry_run and not args.api_key
              and not os.getenv("OPENAI_API_KEY")
              and not os.getenv("LONGTRAJ_API_KEY"))
    # Stage 6 only. Same endpoint, same credentials, same temperature -- the
    # model id is the single difference, so stages 1-5c keep hitting the cache
    # they were filled with and the run pays only for the localiser.
    llm_localize = None
    if args.localize_model and args.localize_model != llm.model:
        llm_localize = LLM(model=args.localize_model, base_url=args.base_url,
                           api_key=args.api_key, temperature=args.temperature,
                           cache_dir=args.cache_dir, dry_run=args.dry_run,
                           config=args.config, config_required=False)
    # Stage 1 only, built the same way. Unlike the localiser seam this one
    # is NOT cache-neutral downstream: the flow it writes feeds every later
    # prompt, so stages 2-6 are paid again even though their model is
    # unchanged. That is the price of the experiment, not a bug.
    llm_trace = None
    if args.trace_model and args.trace_model != llm.model:
        llm_trace = LLM(model=args.trace_model, base_url=args.base_url,
                        api_key=args.api_key, temperature=args.temperature,
                        cache_dir=args.cache_dir, dry_run=args.dry_run,
                        config=args.config, config_required=False)
        if args.reuse_flow:
            print("*** --trace-model given with --reuse-flow: stored "
                  "flows win, so the new tracer will not be called on any "
                  "record that already has one. ***", file=sys.stderr)
    # ``--alf-space`` is consumed at pipeline §2d'', which is guarded on
    # ``cands is not None``; and ``cands`` is reset to None at §3 unless
    # ``--candidates`` (or ``--handoff-candidates``) asked for an answer
    # space. So the flag ON ITS OWN is a silent no-op -- and a silent no-op
    # is the one failure mode that cannot be repaired after the fact, because
    # the artifact would record the restricted arm as having run while the
    # localiser actually ran unrestricted. Refuse rather than run.
    if args.alf_space and not (args.candidates or args.handoff_candidates):
        print("*** --alf-space %s requires --candidates: without it the "
              "answer space is discarded before the localiser sees it, and "
              "the run would be recorded as the restricted arm while running "
              "unrestricted. ***" % args.alf_space, file=sys.stderr)
        return 2
    cfg = Config(max_anchors=args.anchors, target_spans=args.spans,
                 flow_window=args.flow_window, backward=args.backward,
                 origin_edges=args.origin_edges,
                 flow_index_per_step=args.flow_index_per_step,
                 flow_index_chars=args.flow_index_chars,
                 max_rounds=args.max_rounds, batch_size=args.batch_size,
                 max_traced_steps=args.max_traced_steps,
                 topup_rounds=args.topup_rounds, topup_calls=args.topup_calls,
                 workers=args.span_workers, trace_dir=args.trace_dir,
                 reuse_flow=args.reuse_flow,
                 build_outline=not args.no_outline,
                 span_prior=args.span_prior,
                 prior_samples=args.prior_samples,
                 prior_signals=args.prior_signals,
                 prior_enhanced=args.prior_enhanced,
                 prior_symptom=args.prior_symptom,
                 localize_symptom=args.localize_symptom,
                 localize_thinking=args.localize_thinking,
                 localize_rootcause=args.localize_rootcause,
                 alf_prompt=args.alf_prompt,
                 alf_flownote=args.alf_flownote,
                 alf_space=args.alf_space,
                 alf_window=args.alf_window,
                 alf_actions=args.alf_actions,
                 alf_anchors=args.alf_anchors,
                 alf_ledger=args.alf_ledger,
                 alf_modules=args.alf_modules,
                 alf_local3=args.alf_local3,
                 alf_obligations=args.alf_obligations,
                 localize_pointwise=args.localize_pointwise,
                 localize_multiturn=args.localize_multiturn,
                 memo_scan=args.memo_scan,
                 memo_chapters=args.memo_chapters,
                 prior_unranked=args.prior_unranked,
                 prior_to_localizer=not args.no_prior_to_localizer,
                 op_analysts=args.op_analysts,
                 candidate_set=args.candidates or args.candidates_strict,
                 candidate_strict=args.candidates_strict,
                 plan_candidates=args.plan_candidates,
                 plan_cap=args.plan_cap,
                 shortlist=args.shortlist,
                 shortlist_k=args.shortlist_k,
                 protect_candidates=args.protect_candidates,
                 handoff_protect=args.handoff_protect,
                 handoff_candidates=args.handoff_candidates,
                 candidates_verbatim=args.candidates_verbatim,
                 cand_evidence=args.cand_evidence,
                 framings=args.framings,
                 verify=args.verify,
                 verify_graded=args.verify_graded,
                 verify_chain=args.verify_chain,
                 verify_reflect=args.verify_reflect,
                 verify_duel=args.verify_duel,
                 verify_pool=args.verify_pool,
                 verify_pool_union=args.verify_pool_union,
                 verify_outdeg=args.verify_outdeg,
                 roll_call=args.roll_call,
                 expect_spec=args.expect_spec,
                 multi_expert=args.multi_expert,
                 verbatim_limit=args.verbatim_limit,
                 samples=args.samples,
                 zoom=args.zoom,
                 zoom_window=args.zoom_window,
                 bracket=args.bracket,
                 bracket_window=args.bracket_window,
                 bracket_max=args.bracket_max,
                 bracket_group=args.bracket_group,
                 audit=args.audit,
                 audit_max=args.audit_max,
                 error_anchors=args.error_anchors,
                 mixed_paint=args.mixed_paint,
                 rule_anchors=args.rule_anchors,
                 hybrid_anchors=args.hybrid_anchors,
                 unit_spans=args.unit_spans,
                 # The judge's scores are consumed by the compression plan and
                 # by nothing else, so asking for the judge without the plan
                 # would pay for ~11 calls a row and discard every one of them.
                 prior_compress=(args.prior_compress or args.span_judge),
                 span_judge=args.span_judge,
                 judge_to_localizer=args.judge_to_localizer,
                 judge_workers=args.judge_workers,
                 prior_budget_chars=args.prior_budget_chars,
                 possible_keep=args.possible_keep,
                 max_prompt_chars=args.max_prompt_chars,
                 full_keep=args.full_keep)
    # one store shared by all trajectories: writes are atomic and per-file
    store = TraceStore(args.trace_dir) if args.trace_dir else None

    print("running %d trajectories through %s%s"
          % (len(records), llm.model, " [DRY RUN]" if args.dry_run else ""))
    if llm.config_path and not args.dry_run:
        print("credentials: %s" % llm.config_path)
    if store is not None:
        print("flow/span artifacts -> %s" % store.root)
    if llm_trace is not None:
        print("stage 1 (tracing) -> %s" % llm_trace.model)
    if llm_localize is not None:
        print("stage 6 (localise) -> %s" % llm_localize.model)

    def work(rec):
        # ground truth is dropped here and never reaches the pipeline
        pred = run_one(rec.trajectory, llm, cfg, store=store,
                       llm_localize=llm_localize, llm_trace=llm_trace)
        return rec, pred

    pairs = map_concurrent(work, records, workers=args.workers)

    rows: List[dict] = []
    for item in pairs:
        if item is None:
            continue
        rec, pred = item
        row = score_one(pred, rec.ground_truth) if rec.ground_truth else {
            "task_id": pred.task_id, "pred": pred.critical_step, "gold": None}
        row["source"] = rec.trajectory.source
        row["subset"] = rec.trajectory.subset
        row["n_steps"] = len(rec.trajectory)
        rows.append(row)
        if args.verbose:
            print("  %-44s pred=%-5s gold=%-5s %s"
                  % (str(row["task_id"])[:44], row.get("pred"), row.get("gold"),
                     "HIT" if row.get("hit@0") else ""))

    summary = aggregate(rows)
    by_subset = group_by(rows, "subset")
    print()
    print(format_report(summary, by_subset))
    print()
    print("llm usage: %s" % llm.snapshot())
    if account_is_dead():
        # A result file written after the account was refused looks exactly
        # like a finished run -- same fields, same case count, a plausible
        # acc@0 built from fallbacks. Saying so on stdout is the only thing
        # standing between that file and someone scoring it.
        print("*** ENDPOINT REFUSED MID-RUN -- %d calls were skipped. This "
              "result file is NOT a run and must not be scored. ***"
              % llm.snapshot().get("blocked", 0))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        # What produced this file, recorded in the file. `fingerprint()` drops
        # only trace_dir/reuse_flow, so a knob added to Config lands here
        # without anyone remembering to add it.
        run_block = {"source": args.source, "subset": args.subset,
                     "model": llm.model, "n_records": len(records),
                     "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                     "llm": llm.snapshot(), "config": cfg.fingerprint()}
        # Which stage ran on which model, in the file. Without this a
        # mixed-model run is indistinguishable from a single-model one.
        if llm_trace is not None:
            run_block["trace_model"] = llm_trace.model
            run_block["llm_trace"] = llm_trace.snapshot()
        if llm_localize is not None:
            run_block["localize_model"] = llm_localize.model
            run_block["llm_localize"] = llm_localize.snapshot()
        if account_is_dead():
            # stdout scrolls away; this does not. Anything that scores a result
            # file can look for this key and refuse.
            run_block["account_refused"] = True
        write_results(args.out, rows, summary, by_subset, run=run_block)
        print("wrote %s" % args.out)
    return 0


def cmd_flows(args) -> int:
    """Inspect the flow graphs written by `run --trace-dir`."""
    store = TraceStore(args.trace_dir)
    arts = list(store.iter_artifacts())
    if args.source:
        arts = [a for a in arts if a.get("source") == args.source]
    if args.task_id:
        arts = [a for a in arts if args.task_id in str(a.get("task_id", ""))]
    if not arts:
        print("no artifacts under %s" % store.root, file=sys.stderr)
        return 1
    arts.sort(key=lambda a: (str(a.get("source")), str(a.get("task_id"))))

    if args.full:
        for a in arts[: args.limit or len(arts)]:
            print(render_flow_text(a))
            print()
        return 0

    print("%-42s %5s %6s %5s %5s %5s %6s %6s"
          % ("task_id", "steps", "edges", "data", "ctrl", "onfl", "cover", "pred"))
    print("-" * 92)
    for a in arts[: args.limit or len(arts)]:
        st = a.get("flow_stats") or {}
        pred = (a.get("prediction") or {}).get("critical_step")
        print("%-42s %5s %6s %5s %5s %5s %6.2f %6s"
              % (str(a.get("task_id"))[:42], a.get("n_steps"),
                 st.get("n_edges", 0), st.get("n_data_edges", 0),
                 st.get("n_control_edges", 0), st.get("steps_on_flow", 0),
                 st.get("coverage", 0.0), pred))

    n = len(arts)
    complete = sum(1 for a in arts if a.get("stage") == "complete")
    cov = [(a.get("flow_stats") or {}).get("coverage", 0.0) for a in arts]
    print("-" * 92)
    print("%d artifacts (%d complete, %d flow-only), mean coverage %.3f"
          % (n, complete, n - complete, sum(cov) / n if n else 0.0))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="longtraj")
    ap.add_argument("--log", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="corpus statistics")
    _add_common(p)
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("peek", help="print a trajectory as the pipeline sees it")
    _add_common(p)
    p.set_defaults(fn=cmd_peek)

    p = sub.add_parser("run", help="locate critical steps")
    _add_common(p)
    p.add_argument("--config", default=None,
                   help="JSON file with base_url / api_key / model_name "
                        "(default: LongTraj/config.json)")
    p.add_argument("--model", default=None)
    p.add_argument("--localize-model", default=None,
                   help="run stage 6 (and its 6d poll) on this model while "
                        "every earlier stage keeps --model. Given the "
                        "reference step is already on the traced flow it is "
                        "still picked only 0.571/0.569 of the time (SP/TA), "
                        "and in 0.70/0.83 of those misses the localiser's own "
                        "reason never mentions it -- that ceiling is the "
                        "reader's. Off by default; leaving it off keeps every "
                        "prompt and cache key byte-identical.")
    p.add_argument("--trace-model", default=None,
                   help="run stage 1 (backward tracing) on this model "
                        "while every later stage keeps --model. Same seam "
                        "as --localize-model, one end of the pipeline "
                        "instead of the other. Measured reason: on a "
                        "258-step-median corpus the adjacent "
                        "assistant->tool edge is built only 0.705 of the "
                        "time, the graph breaks into 8-60 components, and "
                        "gold_on_flow is 0.000 against 0.696 on "
                        "TrajAudit. Off by default. NOTE it invalidates "
                        "every downstream cache key, because the flow it "
                        "writes changes the prompts all later stages "
                        "build -- --reuse-flow must be off for it to do "
                        "anything at all.")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cache-dir", default=".longtraj_cache")
    p.add_argument("--dry-run", action="store_true",
                   help="skip all LLM calls; exercises fallback paths only")
    p.add_argument("--anchors", type=int, default=3)
    p.add_argument("--flow-window", type=int, default=20,
                   help="steps whose parents are asked for in one call; all "
                        "windows run concurrently")
    p.add_argument("--flow-index-per-step", type=int, default=180,
                   help="chars of each step the parent-naming windows see. A "
                        "step earns out-degree only by being cited, and it is "
                        "cited off this digest, so content past this width "
                        "cannot be cited. Raise with --flow-index-chars")
    p.add_argument("--flow-index-chars", type=int, default=40000,
                   help="cap on the whole digest handed to one window; past it "
                        "the MIDDLE is dropped, so this must grow whenever "
                        "--flow-index-per-step does")
    p.add_argument("--origin-edges", action="store_true",
                   help="forward sweep cites where each value entered as well "
                        "as the nearest mention; no extra calls, denser graph")
    p.add_argument("--backward", action="store_true",
                   help="use the legacy origin-terminated upstream walk "
                        "instead of the forward sweep")
    p.add_argument("--max-rounds", type=int, default=20,
                   help="--backward only: safety cap on tracing rounds")
    p.add_argument("--batch-size", type=int, default=4,
                   help="--backward only: steps traced per LLM call")
    p.add_argument("--max-traced-steps", type=int, default=60,
                   help="--backward only: cap on how many steps get traced")
    p.add_argument("--topup-rounds", type=int, default=0,
                   help="after the forward sweep, re-ask about the steps "
                        "where a chain to an anchor stopped; 0 disables")
    p.add_argument("--topup-calls", type=int, default=12,
                   help="cap on the extra calls --topup-rounds may spend")
    p.add_argument("--spans", type=int, default=None)
    p.add_argument("--workers", type=int, default=4,
                   help="trajectories in parallel")
    p.add_argument("--span-workers", type=int, default=8,
                   help="spans summarised/compressed in parallel, per trajectory")
    p.add_argument("--no-outline", action="store_true",
                   help="skip artifact 1; localise from the compressed steps only")
    p.add_argument("--alf-half", default=None, choices=["dev", "test"],
                   help="alfworld only: keep just one half of the fixed "
                        "dev/test split (same hash as probe/split_eval.py). "
                        "Applied before --limit, so --limit N means the first "
                        "N of that half, not the first N of the corpus")
    p.add_argument("--alf-flownote", default="",
                   choices=["", "none", "anchorless"],
                   help="alfworld only: SUPPRESS the flow-graph evidence in "
                        "the localiser prompt (longtraj/alfworld_flownote.py). "
                        "The opposite intervention to --alf-anchors, and it "
                        "tests the opposite reading of the same measurement: "
                        "the graph is rooted at the last step on 99/100 "
                        "records because ALFWorld failures are silent, and on "
                        "the stored baseline the dev rows whose gold the graph "
                        "covers score 0.1429 against 0.3548 for rows it does "
                        "not. 'none' replaces both flow blocks and the symptom "
                        "line with a statement that no symptom anchor exists; "
                        "'anchorless' keeps the dependency edges and drops only "
                        "the anchors line and the hops-from-symptom marks, "
                        "separating 'the anchors are harmful' from 'the whole "
                        "graph is harmful'. Off leaves every prompt byte "
                        "identical. See probe/prereg_alf_flownote.md.")
    p.add_argument("--alf-prompt", default="",
                   choices=["", "null", "domain", "rubric", "full"],
                   help="alfworld only: swap the localiser's head and note for "
                        "an ALFWorld-specific framing (longtraj/"
                        "alfworld_prompt.py). 'domain' fixes the vocabulary "
                        "only, 'rubric' adds the corpus's annotation guideline, "
                        "'full' adds how to read the simulator's replies. "
                        "'null' is the noise control: same words as the "
                        "incumbent, one trailing space, so it misses the cache "
                        "and re-draws the localiser alone. Default \"\" sends "
                        "the incumbent prompt byte for byte")
    p.add_argument("--alf-space", default="",
                   choices=["", "early", "early+blank", "blank"],
                   help="alfworld only: REPLACE the answer space with a "
                        "positional one (longtraj/alfworld_space.py). Needs "
                        "--candidates (or --protect-candidates / "
                        "--candidates-verbatim) to have asked for an answer "
                        "space at all; on this corpus the write-op parse in "
                        "candidates.py degrades to [] because nothing is ever "
                        "written. 'early' offers the first --alf-window steps "
                        "(dev: 11 of 30 slots, recall 0.7632 with blank, lift "
                        "2.01x); 'blank' offers only the turns that emitted an "
                        "empty <action> (lift 2.08x, median pool size 0); "
                        "'early+blank' is their union. This FITS AN ANNOTATION "
                        "CONVENTION measured on dev -- see prereg_alf_040.md "
                        "%s5 -- and 0.7632 is a recall ceiling, not a bug. "
                        "Default \"\" leaves the answer space untouched"
                        % "\xa7")
    p.add_argument("--alf-window", type=int, default=21, metavar="N",
                   help="alfworld only: last step index offered by "
                        "--alf-space early. Frozen at 21 by prereg_alf_040.md "
                        "%s4.6; retuning it against a dev reading is "
                        "overfitting the answer space" % "\xa7")
    p.add_argument("--alf-anchors", default="",
                   choices=["", "union", "first2", "first3", "only"],
                   help="alfworld only, 0 calls: union the located symptom "
                        "anchors with the steps that revisit an "
                        "already-visited fixture (longtraj/"
                        "alfworld_anchors.py). An ALFWorld failure is silent, "
                        "so locate_anchors returns step 59 on 99/100 records "
                        "and the backward walk starts a median of 40 steps "
                        "from the reference. Repainting the stored graphs "
                        "offline moves gold_on_flow 0.3700 to 0.6600 and "
                        "|anchor-gold| 40 to 4. 'first2' caps the additions at "
                        "two, landing at 3.0 anchors/record against TA's 2.6, "
                        "because the full union paints 56%% of all steps and "
                        "'on flow' stops discriminating once it is the "
                        "majority. 'only' drops the located anchor entirely "
                        "and is the ablation, not a shipping choice")
    p.add_argument("--alf-actions", action="store_true",
                   help="alfworld only, 0 calls: hoist each assistant step's "
                        "own <action> to the front of its digest. Measured "
                        "over all 3000 assistant steps, the action is 1.0%% "
                        "of the text and starts at the 98%% mark, while 86%% "
                        "of steps cite a DIFFERENT step by number -- so the "
                        "localiser reads mostly prose about other steps. Adds "
                        "no new information: the text is copied from the "
                        "step's own content. See probe/prereg_alf_fix.md")
    p.add_argument("--alf-ledger", action="store_true",
                   help="alfworld only, 0 upstream calls: replace only the "
                        "final localiser's trajectory body with one compact "
                        "state-transition row per agent turn. Each row keeps "
                        "the original step index, before/action/after text, "
                        "short verbatim thought excerpts and code-derived "
                        "admissibility/repetition/refusal flags. Default off; "
                        "see probe/prereg_alf_ledger.md")
    p.add_argument("--alf-modules", action="store_true",
                   help="alfworld only: preserve the incumbent localiser, "
                        "then run separate memory/reflection/plan/action-system "
                        "audits and arbitrate only among their valid step "
                        "nominations. Default off; see "
                        "probe/prereg_alf_modules.md")
    p.add_argument("--alf-local3", action="store_true",
                   help="alfworld only: after the unchanged localiser, compare "
                        "its previous/current/next assistant turns using exact "
                        "local observations and introduced-vs-inherited "
                        "attribution. One extra call, closed set, default off; "
                        "see probe/prereg_alf_local3.md")
    p.add_argument("--alf-obligations", action="store_true",
                   help="alfworld only, 0 extra calls: add a code-derived "
                        "objective obligation timeline to the existing "
                        "localiser prompt and ask for the first turn where "
                        "the agent treated a still-unmet obligation as done, "
                        "unnecessary or abandoned. Does not restrict the "
                        "answer space; default off. See "
                        "probe/prereg_alf_obligations.md")
    p.add_argument("--alf-dedup", action="store_true",
                   help="alfworld only (--source trajdebug_alf): strip the "
                        "replayed history from each user turn, keeping just "
                        "the new observation. 43.5%% of that corpus is replay "
                        "and the user side is non-attributable, so none of it "
                        "can be the answer. Ignored by every other source")
    p.add_argument("--span-prior", action="store_true",
                   help="one extra call: guess which phases hold the cause, and "
                        "hand that guess to the localiser as an overrulable hint")
    p.add_argument("--prior-samples", type=int, default=1, metavar="K",
                   help="draw --span-prior's prompt K times and UNION the "
                        "phases named, ranked by how many draws named each. "
                        "Repairs an arity deficit, not a disagreement: the "
                        "budget allows 3 phases on every SP row and 2-3 on 71 "
                        "TA rows, but qwen3-235b names exactly one on 78/86 SP "
                        "and 85/98 TA rows where claude-opus named three. The "
                        "parse was cleared first -- on those rows neither the "
                        "coverage rule nor the budget cut a second guess. This "
                        "is a union, not a vote: --samples 5 on the LOCALISER "
                        "picked one of k and lost on both corpora, whereas the "
                        "budget and coverage caps in _parse still bound this, "
                        "so it cannot nominate more than the design already "
                        "allowed. K=1 is byte-identical and fully cached.")
    p.add_argument("--prior-signals", action="store_true",
                   help="no extra call: give --span-prior the per-phase counts "
                        "(on-flow steps, hops from the symptom, and — with "
                        "--candidates — how many steps in the phase changed "
                        "anything, since the answer must be one of those)")
    p.add_argument("--prior-enhanced", action="store_true",
                   help="one extra call: keep --span-prior's guesses and APPEND "
                        "a second tier of 'possible' phases until the nominated "
                        "region reaches ~30%% of the run (measured today: 18.5%%). "
                        "The strongest guess is left untouched, so this widens "
                        "recall without moving the top guess")
    p.add_argument("--prior-symptom", action="store_true",
                   help="no extra call: give --span-prior the harness's own "
                        "failing test and traceback, instead of only the "
                        "anchoring model's two-sentence summary of it. Empty on "
                        "corpora that do not ship one (trajdebug), so the prompt "
                        "there is unchanged")
    p.add_argument("--localize-symptom", action="store_true",
                   help="give the final localiser the harness's raw failing "
                        "test and error output; no-op on corpora without it")
    p.add_argument("--localize-thinking", action="store_true",
                   help="add deep reasoning prompt to localizer: forces "
                        "step-by-step analysis (trace backwards, review "
                        "candidates, consider alternatives) before answering")
    p.add_argument("--localize-rootcause", action="store_true",
                   help="when the step the localiser is about to name ACTED "
                        "(edit/create/patch), make it test whether that step "
                        "only executed a decision an earlier plan/reason step "
                        "had already got wrong, and keep the acting step "
                        "unless the earlier one made a correct edit "
                        "unreachable. Adds no evidence, 0 extra calls")
    p.add_argument("--localize-pointwise", action="store_true",
                   help="use pointwise scoring instead of listwise ranking: "
                        "score each candidate independently (no comparison), "
                        "take argmax. Costs ~10-20 calls/case but avoids "
                        "being anchored by the most salient step")
    p.add_argument("--localize-multiturn", action="store_true",
                   help="use multi-turn coarse-to-fine localization: Round 1 "
                        "narrows to one phase (6→1), Round 2 to one span (5→1), "
                        "Round 3 to one step (7→1), Round 4 verifies or corrects. "
                        "Costs 4 calls/case but avoids information overload and "
                        "salience bias by progressively narrowing context")
    p.add_argument("--memo-scan", action="store_true",
                   help="replace --span-prior with a forward scan that reads the "
                        "run one chapter at a time, in order, carrying a ledger "
                        "of what the agent believes, what it has committed to "
                        "and what it left unverified. Never sees a later "
                        "chapter, so it must judge a commitment when it is made "
                        "rather than from the symptom backwards. Costs one call "
                        "per chapter; takes precedence over --span-prior")
    p.add_argument("--memo-chapters", type=int, default=10,
                   help="hard cap on --memo-scan's calls per record. Phases "
                        "beyond it are MERGED, never dropped: truncating the "
                        "tail would turn 'not read yet' into 'never read' and "
                        "destroy the forward property (default 10)")
    p.add_argument("--op-analysts", action="store_true",
                   help="up to five extra calls: one localisation analyst per "
                        "OPERATION TYPE — modification, reasoning, reading, "
                        "search, execution/testing — each shown ONLY the steps "
                        "of its own type and asked whether the critical step is "
                        "among them, with the final localiser arbitrating over "
                        "the compressed trajectory, the span prior and all five "
                        "reports. The partition is the whole claim: "
                        "--multi-expert split four readers by FAILURE FAMILY "
                        "over SHARED evidence and they converged (1.6 distinct "
                        "answers of 4, TrajAudit 0.3333, lowest of 39 arms). "
                        "Measured headroom before the first call: the answer "
                        "already lands in the right operation type 80/102, so "
                        "a perfect type selector is worth only 0.716; the gain, "
                        "if any, is WITHIN a type, where the 30 right-type "
                        "wrong-step rows miss by a median of 4 trajectory steps "
                        "but only 1 same-type step. Adds no new evidence: 86%% "
                        "of reference steps already arrive uncompressed")
    p.add_argument("--prior-unranked", action="store_true",
                   help="no extra call: show the localiser the prior's regions "
                        "in trajectory order with no 'likely' label and no "
                        "ranking. Measured on TrajAudit: when the prior's top "
                        "guess is wrong the localiser scores 0/22, while rows "
                        "just as hard where it answered outside the prior score "
                        "0.538 and rows with no prior at all score 0.778 — the "
                        "top guess anchors it. 20 rows sit in a correctly "
                        "nominated but lower-ranked region")
    p.add_argument("--no-prior-to-localizer", action="store_true",
                   help="no extra call upstream: keep the prior allocating the "
                        "compression budget at 3b, but never show it to the "
                        "localiser, which then reads only the compressed "
                        "trajectory and the task description. ta_noprior "
                        "(-13 rows) turned off the flag feeding BOTH jobs, on "
                        "claude-opus-4-8 and before 3b existed, so it cannot "
                        "say which job earned those rows. This is the cut that "
                        "separates them. _fallback_step still receives the "
                        "prior, so exactly one thing changes")
    p.add_argument("--candidates", action="store_true",
                   help="no extra call: parse the steps that changed something "
                        "and offer them as the answer space, without forcing "
                        "the answer into them (falls back to the unrestricted "
                        "space when the parse finds too few)")
    p.add_argument("--candidates-strict", action="store_true",
                   help="as --candidates, and move any answer outside the set "
                        "to the nearest entry. Helped on swebenchpro, cost "
                        "three correct answers on trajaudit — measure per corpus")
    p.add_argument("--plan-candidates", action="store_true",
                   help="no extra call: with --candidates, ALSO offer the prose "
                        "steps that named a file or method and stated a "
                        "conclusion about it. The write-only parse fits "
                        "swebenchpro, whose reference step is a write 76.7%% of "
                        "the time, and half-fits trajaudit, where it is 52.0%% "
                        "and the label often sits on the step that committed to "
                        "the wrong approach before any edit was made. Offered "
                        "list capped at --plan-cap, earliest kept")
    p.add_argument("--plan-cap", type=int, default=15, metavar="N",
                   help="cap on the --plan-candidates answer space (default 15). "
                        "When it binds the EARLIEST steps are kept: trajaudit's "
                        "reference step sits at median position 0.39, and on the "
                        "24 rows where the cap bites, keeping the first 15 loses "
                        "no reference step while keeping the last 15 loses every "
                        "row this stage gained")
    p.add_argument("--shortlist", action="store_true",
                   help="+1 call per trajectory: show a reader the same "
                        "compressed run the localiser gets, plus the answer "
                        "space itself, and ask it for up to --shortlist-k MORE "
                        "steps. The answer space is parsed by matching file "
                        "WRITES, so a step whose only act was to decide or "
                        "misdiagnose can never be in it -- on swebenchpro it "
                        "misses the reference step on 20 rows and on 15 of those "
                        "the reference step is pure prose. The proposals are "
                        "merged into the answer space, which is shown as ONE "
                        "list in trajectory order with nothing marking which "
                        "entries came from where. The frozen candidate block is "
                        "not touched: a merged space gets a separate text that "
                        "describes it, since the frozen one says steps which "
                        "only reasoned are not listed. An unparseable or empty "
                        "answer leaves the run byte-identical. Needs "
                        "--candidates or --candidates-strict")
    p.add_argument("--shortlist-k", type=int, default=5, metavar="K",
                   help="cap on the NEW steps --shortlist may add (default 5). "
                        "Accuracy falls as the answer space grows (|C1| 2-5: "
                        "0.714; 9-12: 0.231), so the names are expensive; 5 was "
                        "fixed before the arm ran and sweeping it is a new arm "
                        "on the dev half, not a re-read of this one")
    p.add_argument("--protect-candidates", action="store_true",
                   help="no extra call: keep the steps that changed something "
                        "in their own words through compression, the way "
                        "on-flow steps already are. The two sets are not the "
                        "same -- compression protects what tracing reached, "
                        "the localiser must answer with a candidate -- and "
                        "31.8%% of candidates were reaching it paraphrased, "
                        "which halves the rate at which they are found")
    p.add_argument("--handoff-protect", action="store_true",
                   help="no extra call: keep steps whose AUTHOR marks a "
                        "transfer of control (a name like 'DiagnostAgent "
                        "(-> ActionAgent)') in their own words through "
                        "compression. Repairs a measured bias of our own "
                        "compressor: it keeps 83.7%% of steps but only "
                        "36.8%% of these, because it chooses by length and "
                        "they are the longest class (median 2780 chars vs "
                        "298). It cannot fall back on on_flow() there -- "
                        "that paints 5.5%% of steps and reaches the "
                        "reference step 0.000 of the time. No-op on a "
                        "corpus without such authors, which is three of "
                        "longrca-bench's five subsets")
    p.add_argument("--handoff-candidates", action="store_true",
                   help="no extra call: make those same steps the ANSWER "
                        "SPACE (replacing the write-op pool, not unioned "
                        "with it). Read the warning: this is answer-space "
                        "restriction, measured net negative on TrajAudit, "
                        "and the set was chosen off the label distribution "
                        "(1.6%% of steps hold 57.0%% of reference steps on "
                        "SWE-bench Pro) -- it is fitting an annotation "
                        "convention of ONE subset and must never be "
                        "reported as a method-level gain. --handoff-protect "
                        "is the defensible half; this is the gamble")
    p.add_argument("--candidates-verbatim", action="store_true",
                   help="no extra call: AFTER compression, replace every "
                        "answer-space step's text with the step's own content "
                        "from the raw trajectory, whatever compression did to "
                        "it and at any length. Distinct from "
                        "--protect-candidates, which does it by changing the "
                        "compression PLAN and so also changes the span prompt, "
                        "the budget split, and every other step in the row "
                        "(and lost on both corpora in 2026-09). This changes "
                        "the candidates' text and nothing else, so every call "
                        "upstream hits the cache")
    p.add_argument("--cand-evidence", action="store_true",
                   help="no extra call: set the answer space out as a "
                        "comparison -- each entry with its own text and the "
                        "phase/tracing facts already computed for it -- "
                        "instead of a list of bare step numbers whose text "
                        "sits scattered through PART 2. Needs --candidates or "
                        "--candidates-strict to have an answer space at all")
    p.add_argument("--framings", type=int, default=1, metavar="K",
                   help="+k-1 calls per trajectory: ask the same question as k "
                        "different decision procedures over identical evidence "
                        "and an identical answer space (root cause / one-step "
                        "repair / point of no return), then take the majority. "
                        "Unlike --multi-expert the framings all choose from the "
                        "same set and return the same fields, so the answers "
                        "are comparable by construction; a tie resolves to the "
                        "incumbent framing, so the answer can only move when "
                        "two framings agree against it")
    p.add_argument("--verify", action="store_true",
                   help="+1 call per CANDIDATE: instead of choosing one step "
                        "out of the offered list in a single call, judge each "
                        "candidate on its own, shown no rivals, against four "
                        "fixed yes/no questions (does the failure depend on it; "
                        "would redoing it alone avert the failure; was the run "
                        "already doomed before it; does it merely report a "
                        "fault introduced elsewhere). Every call is the same "
                        "template over the same evidence with only the focal "
                        "step changed, so the verdicts are comparable by "
                        "construction -- the defect that sank --bracket. The "
                        "listwise answer is displaced only by a candidate "
                        "scoring strictly higher, so a noisy verifier leaves it "
                        "standing. Needs --candidates-strict to have an answer "
                        "space; costs about 11 calls per trajectory on "
                        "SWE-Bench-Pro")
    p.add_argument("--verify-graded", action="store_true",
                   help="with --verify, grade the two positive questions 0-3 "
                        "instead of asking them as binaries. The binary rubric "
                        "was measured informative but blunt: all four questions "
                        "separate the reference step in the right direction, "
                        "yet 'averts' comes back true on 0.7%% of candidates, "
                        "so four binaries give six score levels for eleven "
                        "candidates and 55%% of polls end in a tie the rubric "
                        "did not decide. Grading widens the range to -4..9 and "
                        "drops the self-reported confidence field, which was "
                        "measured actively harmful as a tie-break. No effect "
                        "without --verify")
    p.add_argument("--verify-chain", action="store_true",
                   help="with --verify, show each focal step the dependency "
                        "edges traced OUT of it -- which later steps used what "
                        "it produced, which later steps happened because of it, "
                        "and how many of those sit on a chain reaching a "
                        "symptom. The pointwise rubric is length-robust where "
                        "the listwise reader is not (0.548/0.571 against "
                        "0.419/0.250 on 80-119 and 120+ step runs), but its "
                        "whole view of consequence was two hop counts. This is "
                        "structural evidence, not more verbatim text. No effect "
                        "without --verify")
    p.add_argument("--verify-reflect", action="store_true",
                   help="with --verify, REPLACE the per-candidate pointwise "
                        "rubric with ONE listwise reflection call that sees all "
                        "candidates side-by-side with structural annotations "
                        "(out-degree, in-degree, hops to symptom). Mutually "
                        "exclusive with --verify-graded/--verify-chain/"
                        "--verify-duel. No effect without --verify")
    p.add_argument("--verify-duel", action="store_true",
                   help="with --verify, settle a tied poll by playing the tied "
                        "candidates off against each other pairwise. The poll's "
                        "defect on SWE-Bench-Pro is resolution: the reference "
                        "step is in the top-scoring group on 40 of 66 polls but "
                        "alone on top only 12 times, and the tie-break wins 14 "
                        "of the 28 contested -- a coin flip over the largest "
                        "block of recoverable rows left. Each pair is asked "
                        "twice with the labels swapped and a win counts only if "
                        "both askings agree, so label bias shows up as a "
                        "recorded draw instead of deciding the answer. Calls "
                        "land only on contested polls. No effect without "
                        "--verify")
    p.add_argument("--verify-pool", action="store_true",
                   help="with --verify, poll the assistant steps on the "
                        "traced path when the write parse found no "
                        "candidate set. On TrajAudit that is 43 of 102 "
                        "rows, and they are not under-parsed: 32 contain "
                        "no write-looking step at all and across all 43 "
                        "the reference step is a parsed write exactly "
                        "once. The traced pool is the same size as the "
                        "write sets elsewhere (13.8 steps) and holds the "
                        "reference step 72%% of the time. The listwise "
                        "answer joins the pool so the merge keeps its "
                        "meaning. No effect without --verify, and none on "
                        "rows that already have a candidate set")
    p.add_argument("--verify-pool-union", action="store_true",
                   help="with --verify, poll the write-based candidate set "
                        "UNION the assistant steps on the traced path, on the "
                        "rows that HAVE a candidate set. This is the only way "
                        "to widen the answer space without throwing the set "
                        "away: --candidates-strict off does nothing on its "
                        "own, because the poll draws from the set and so "
                        "re-imposes the restriction a stage later (measured: "
                        "84 fresh calls, 0 of 86 rows moved). On "
                        "SWE-Bench-Pro the set holds the reference step on "
                        "66 of the 84 parsed rows, so 18 rows are locked out "
                        "of a correct answer; dropping the set outright woke "
                        "2 of those 18 but cost 7 of the 66, net -5. The "
                        "union adds only, so the rows the set was getting "
                        "right keep every signal they had. No effect without "
                        "--verify, and none on rows with no candidate set -- "
                        "use --verify-pool for those")
    p.add_argument("--verify-outdeg", action="store_true",
                   help="with --verify, add the capped traced out-degree to "
                        "each candidate's rubric score when merging. Costs no "
                        "calls -- the graph is already built. On "
                        "SWE-Bench-Pro the reference step has mean "
                        "out-degree 4.47 against rivals' 2.06, a wider gap "
                        "than any judgement measured here; argmax out-degree "
                        "alone scores 0.3372 and the rubric alone 0.3023, "
                        "but summed 0.3605. Weight and cap were swept on the "
                        "dev half only. DOES NOT TRANSFER TO TRAJAUDIT "
                        "(dev 0.4583 against its shipped 0.5000), so this is "
                        "off by default and claimed on SP only")
    p.add_argument("--roll-call", action="store_true",
                   help="0 extra calls: require the localiser to write one "
                        "line for EVERY traced step -- what it did, and "
                        "whether the run turned on it -- before it names an "
                        "answer. Targets a different failure than the eight "
                        "arms before it: given the reference step is on the "
                        "traced flow it is still picked only 0.569 of the "
                        "time (TA) / 0.571 (SP), and in 0.830 / 0.696 of "
                        "those misses the localiser's own reason never "
                        "mentions it at all -- it is not rejected, it is not "
                        "seen. On TA the rows where the reference step IS on "
                        "the flow split 0.6481 when the traced set is <=20 "
                        "steps (n=54) against 0.3333 when it is larger "
                        "(n=18), which is what a reader that skims rather "
                        "than enumerates looks like. Adds no evidence to the "
                        "prompt and asks no new question; it makes the "
                        "enumeration an output. Skipped on rows with fewer "
                        "than 2 or more than 40 traced steps -- a truncated "
                        "roll would read as compliance in the record")
    p.add_argument("--expect-spec", action="store_true",
                   help="+2 calls PER CANDIDATE (~1,840 on SP): for each "
                        "candidate step, one reader is shown the task, the "
                        "agent's design (meta.framework) and everything that "
                        "reached that step -- but NOT the step -- and writes "
                        "3-6 checkable requirements it must satisfy; a second "
                        "reader then marks the step met/violated/unclear "
                        "against each. The localiser is shown the COUNTS. "
                        "Neither reader is told how the run ended and neither "
                        "is ever asked which step is critical: a reader that "
                        "knows the symptom writes the requirement the symptom "
                        "violates, and 'critical' is the concept ta_roll "
                        "showed this model and the annotation disagree about. "
                        "Requires --candidates or --candidates-strict; on "
                        "TrajAudit there is no "
                        "candidate set and every row records "
                        "skipped=no_candidates, so this is an SP-only claim")
    p.add_argument("--multi-expert", action="store_true",
                   help="+3 calls per trajectory: replace the single localiser "
                        "with four specialists run in parallel (compliance, "
                        "logic, flow, context), each free to abstain, fused by "
                        "a confidence-weighted vote. Every verdict is kept in "
                        "the record, so alternative fusion rules can be scored "
                        "offline against the run without re-calling")
    p.add_argument("--verbatim-limit", type=int, default=0, metavar="CHARS",
                   help="no extra call: after compression, put back the original "
                        "text of every off-flow step at or under CHARS, "
                        "discarding the model's paraphrase of it. The span calls "
                        "still run and still hit the cache, so this changes only "
                        "the localiser's prompt. 0 disables it and leaves the "
                        "digests byte-identical. 2500 is where the measured "
                        "trade-off sits on both corpora")
    p.add_argument("--samples", type=int, default=1, metavar="K",
                   help="+K-1 calls per trajectory: ask the localiser the same "
                        "prompt K times and keep the majority step, ties going "
                        "to the earliest. Sample 0 is the configured "
                        "temperature and reuses the stored answer, so only the "
                        "extra draws are paid for. 1 disables it")
    p.add_argument("--error-anchors", dest="error_anchors",
                   action="store_true", default=False,
                   help="+1 call: after the symptom anchors are located, "
                        "re-anchor on the steps a model judges to be wrong. "
                        "Adopted on flow_recall, but measured on acc@0 it "
                        "costs 0.05 on swebenchpro and 0.03 on trajaudit, so "
                        "it is off by default and kept as an ablation")
    p.add_argument("--no-error-anchors", dest="error_anchors",
                   action="store_false",
                   help="explicit form of the default: paint from the located "
                        "symptom anchors only")
    p.add_argument("--zoom", action="store_true",
                   help="+1 call per trajectory: after the localiser answers, "
                        "show it the finalists — the candidate list, or a "
                        "window around its own answer — in their ORIGINAL text "
                        "and let it revise. Targets the measured gap between "
                        "having the right region and naming the right step. "
                        "The first answer is kept in the record either way")
    p.add_argument("--zoom-window", type=int, default=5, metavar="W",
                   help="half-width of the finalist window used when there is "
                        "no candidate list to inherit (default 5)")
    p.add_argument("--bracket", action="store_true",
                   help="+~6 calls per trajectory: replace the single n-way "
                        "choice with a group stage. The finalists -- the "
                        "candidate list unioned with a window around the first "
                        "answer -- are cut into contiguous groups of four, each "
                        "judged on its own and in parallel with every member "
                        "shown in its original text, and the group winners then "
                        "meet in one final call. Targets the measured gap "
                        "between holding the right shortlist (0.78/0.80) and "
                        "picking the right entry out of it (0.30/0.48)")
    p.add_argument("--bracket-window", type=int, default=5, metavar="W",
                   help="half-width of the window unioned into the finalist "
                        "set (default 5)")
    p.add_argument("--bracket-max", type=int, default=20, metavar="N",
                   help="cap on the finalist set; over it, window entries are "
                        "dropped and the candidate list is kept whole")
    p.add_argument("--bracket-group", type=int, default=4, metavar="G",
                   help="finalists shown per group call (default 4)")
    p.add_argument("--audit", action="store_true",
                   help="+1 call per trajectory: after the answer exists, show "
                        "the model that step and its DIRECT flow parents in "
                        "their original text and ask only whether a parent had "
                        "already introduced the error, leaving the answer to "
                        "carry it out. Moves the answer EARLIER or not at all, "
                        "and only when all four conditions hold (a named wrong "
                        "object, a real edge of the claimed kind carrying it, "
                        "no new error at the answer, and repairing the parent "
                        "sufficing). Aimed at the measured late bias: 35 of "
                        "TrajAudit's 49 misses answer later than the reference "
                        "step and 14 of them name one of its children")
    p.add_argument("--audit-max", type=int, default=6, metavar="N",
                   help="cap on the parents examined; nearest to the answer "
                        "kept (default 6, against a measured mean of 2.0)")
    p.add_argument("--mixed-paint", action="store_true",
                   help="no extra call: let a dependency chain alternate between "
                        "data and control edges instead of walking each relation "
                        "separately. Roughly doubles what counts as on-flow, so "
                        "judge it on acc@0, not on flow_recall")
    p.add_argument("--rule-anchors", action="store_true",
                   help="no extra call: pick the failure anchors by keyword and "
                        "recency instead of by the model's read of the digest. "
                        "The anchor call still happens -- failure_summary and "
                        "failure_entities feed six downstream prompts -- so the "
                        "only thing that changes is where painting starts")
    p.add_argument("--unit-spans", action="store_true",
                   help="no extra call: build spans by grouping a rule-derived "
                        "operation chain (observation -> plan -> localization -> "
                        "edit -> test) instead of cutting a per-step digest. The "
                        "model is asked to group ~20 typed units, not to cut 117 "
                        "lines")
    p.add_argument("--prior-compress", action="store_true",
                   help="no extra call: run the span prior BEFORE compression "
                        "and let its verdict decide what survives -- flagged "
                        "phases verbatim, unflagged ones compressed harder. "
                        "Implies --unit-spans (the prior reads the rule digest) "
                        "and reuses the same prior at stage 5c")
    p.add_argument("--prior-budget-chars", type=int, default=90000, metavar="N",
                   help="cap on text kept verbatim by --prior-compress before "
                        "the plan degrades, weakest band first (default 90000)")
    p.add_argument("--possible-keep", action="store_true",
                   help="no extra call: keep a flowless step VERBATIM whenever "
                        "--prior-compress's prior called its phase suspicious "
                        "at all, not only when it called it likely -- the "
                        "possible band becomes a keep instead of a 2500/1200 "
                        "clip. It is still the first band degraded when the "
                        "budget runs out, so the likely band never pays for it")
    p.add_argument("--full-keep", action="store_true",
                   help="no extra call and NO budget: on the flow => the whole "
                        "step; off the flow but inside a phase the prior called "
                        "suspicious (likely or possible) => the whole step; "
                        "otherwise compressed. Band limits become the step's "
                        "own length and the degrade ladder does not run, so "
                        "--prior-budget-chars is ignored. Implies "
                        "--possible-keep. Prompts get much bigger")
    p.add_argument("--max-prompt-chars", type=int, default=120000,
                   help="cap on the final localiser prompt body. Above it "
                        "localize._fit_budget clips every OFF-FLOW digest to "
                        "400/220/120 and then drops the trajectory's middle. "
                        "The 120000 default was sized for a model the gateway "
                        "dropped on 2026-10-10; qwen3.7-plus accepted 2.0M "
                        "chars when re-measured on 2026-10-13. Raising this "
                        "past llm.MAX_PROMPT_CHARS does nothing good -- the "
                        "blind middle-clip there takes over from the "
                        "structured one here")
    p.add_argument("--hybrid-anchors", action="store_true",
                   help="no extra call: paint from the model's anchors UNION "
                        "the keyword rule's, instead of letting the rule "
                        "replace them. The two sets are nearly disjoint "
                        "(Jaccard 0.116 SP / 0.267 TA) and a union can only "
                        "enlarge on_flow, so gold_on_flow moves +4/-0 on both "
                        "corpora by construction. Refuses to combine with "
                        "--rule-anchors, which replaces rather than unions")
    p.add_argument("--span-judge", action="store_true",
                   help="+1 call per FLOWLESS phase: score that phase on its "
                        "own for relevance AND confidence, and let the product "
                        "pick its compression band. Phases that carry a traced "
                        "path are skipped -- their steps are kept verbatim "
                        "already and the degrade ladder may not touch them. "
                        "The score goes to the COMPRESSOR, which allocates a "
                        "character budget; the localiser is not shown it "
                        "unless --judge-to-localizer. Implies --prior-compress "
                        "(and so --unit-spans): without the plan there is "
                        "nothing to consume the scores")
    p.add_argument("--judge-to-localizer", action="store_true",
                   help="no extra call: ALSO render --span-judge's scores into "
                        "the localiser's prompt. Run and judged as a SEPARATE "
                        "arm, second: handing the localiser more pointed "
                        "evidence is the axis that has lost four times "
                        "(memo-scan, prior-unranked, localize-symptom, "
                        "op-analysts). Costs almost nothing after the first "
                        "arm -- the judge and compression calls are "
                        "byte-identical and hit the cache")
    p.add_argument("--judge-workers", type=int, default=8, metavar="N",
                   help="concurrent per-phase judge calls (default 8)")
    p.add_argument("--out", default=None)
    p.add_argument("--trace-dir", default=None,
                   help="save each trajectory's flow/spans/prediction here")
    p.add_argument("--reuse-flow", action="store_true",
                   help="reuse a stored flow instead of re-tracing")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("flows", help="inspect saved flow graphs")
    p.add_argument("--trace-dir", default="traces")
    p.add_argument("--source", default=None, choices=sorted(LOADERS))
    p.add_argument("--task-id", default=None, help="substring match")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--full", action="store_true",
                   help="dump every edge and span instead of one line each")
    p.set_defaults(fn=cmd_flows)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
