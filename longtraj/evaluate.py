"""Scoring. This is the ONLY module that reads ``GroundTruth``.

Reported metrics follow the conventions of the source corpora so numbers are
comparable with their published baselines:

  * exact / tol-1 / tol-2 / tol-3 step accuracy (RootSE uses tolerances 0-3)
  * mean absolute step distance over the cases where a prediction was made
  * a breakdown by source and subset
  * flow recall: was the reference step on a traced flow, and was it inside a
    weakly-compressed span. These two diagnose the pipeline: if flow recall is
    low, tracing is the bottleneck, not the final localiser.
  * span prior recall against span prior coverage, when that stage ran. Recall
    alone cannot say whether the prior is informative — nominating half the run
    recalls half the gold steps for free — so the pair is always reported
    together, alongside whether the localiser followed the hint at all.
  * candidate-set ceiling against candidate-set coverage, when the answer space
    was restricted, plus the accuracy before and after the answer was snapped
    into the set. The restriction can only help on the cases where it fired, so
    the cases it declined are reported separately rather than averaged in.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .schema import GroundTruth, Prediction

TOLERANCES = (0, 1, 2, 3)


def score_one(pred: Prediction, gt: GroundTruth,
              tolerances: Iterable[int] = TOLERANCES) -> Dict[str, object]:
    """Score a single prediction against its reference label."""
    out: Dict[str, object] = {
        "task_id": pred.task_id,
        "pred": pred.critical_step,
        "gold": gt.critical_step,
        "confidence": pred.confidence,
        "error_type": pred.error_type,
        "gold_error_type": gt.error_type,
        "compression_ratio": pred.compression_ratio,
        "calls": pred.usage.get("calls", 0),
    }
    have = pred.critical_step is not None and gt.critical_step is not None
    dist = abs(pred.critical_step - gt.critical_step) if have else None
    out["distance"] = dist
    for t in tolerances:
        out["hit@%d" % t] = bool(have and dist <= t)

    # diagnostics: did tracing and compression retain the reference step?
    if gt.critical_step is not None and pred.flow:
        painted = _painted(pred.flow)
        out["gold_on_flow"] = gt.critical_step in painted["union"]
        out["gold_on_data_flow"] = gt.critical_step in painted["data"]
        out["gold_on_control_flow"] = gt.critical_step in painted["control"]
        n_flow = len(painted["union"]) or None
        out["flow_size"] = len(painted["union"])
        out["data_flow_size"] = len(painted["data"])
        out["control_flow_size"] = len(painted["control"])
    else:
        out["gold_on_flow"] = None
        out["gold_on_data_flow"] = None
        out["gold_on_control_flow"] = None
    if gt.critical_step is not None and pred.spans:
        out["gold_in_weak_span"] = _in_weak_span(gt.critical_step, pred.spans)
    else:
        out["gold_in_weak_span"] = None

    # the span prior: recall is only meaningful against the region it nominated
    guesses = (pred.span_prior or {}).get("guesses") or []
    if guesses:
        out["prior_span_steps"] = _prior_steps(guesses)
        # the ranges themselves, so a later analysis can ask the counterfactual
        # this row cannot: would a run that never saw the prior have answered
        # inside it anyway? Without that baseline ``pred_in_prior_span`` cannot
        # distinguish the localiser adopting the hint from the hint agreeing
        # with where the localiser was already headed.
        out["prior_spans"] = [[g.get("start"), g.get("end")] for g in guesses]
        out["pred_in_prior_span"] = (pred.critical_step is not None
                                     and _in_guess(pred.critical_step, guesses))
        out["prior_disagreed"] = bool((pred.span_prior or {}).get("disagreement"))
        if gt.critical_step is not None:
            out["gold_in_prior_span"] = _in_guess(gt.critical_step, guesses)
            out["gold_in_top_prior_span"] = _in_guess(gt.critical_step,
                                                      guesses[:1])

    # the pointwise phase judge. The band is what the compressor acted on, so
    # the band the reference step's phase got is the only honest way to ask
    # whether the stage reached the answer -- a high score on some other phase
    # is not a near miss, it is spent budget. ``gold_phase_judged`` separates
    # the two ways of failing to help: the judge looked and cleared the phase,
    # or it was never asked because tracing had already put flow there.
    jspans = (pred.span_judge or {}).get("spans") or {}
    if jspans:
        vs = list(jspans.values())
        out["judge_called"] = sum(1 for v in vs if v.get("called"))
        out["judge_abstained"] = sum(1 for v in vs if v.get("called")
                                     and v.get("score") is None)
        out["judge_skipped_flow"] = sum(1 for v in vs
                                        if v.get("skipped") == "carries flow")
        # Over the CALLED phases only. A phase that was never asked about is
        # not a phase the judge cleared, and folding the two together would
        # read a cheap run as a confident one.
        bands = {}
        for v in vs:
            if not v.get("called"):
                continue
            b = v.get("band") or "none"
            bands[b] = bands.get(b, 0) + 1
        out["judge_bands"] = bands

        def _host(step):
            for v in vs:
                if v.get("start") is not None and v.get("end") is not None \
                        and v["start"] <= step <= v["end"]:
                    return v
            return None

        if pred.critical_step is not None:
            hv = _host(pred.critical_step)
            out["pred_judge_band"] = (hv.get("band") or "") if hv else None
        if gt.critical_step is not None:
            hv = _host(gt.critical_step)
            out["gold_judge_band"] = (hv.get("band") or "") if hv else None
            out["gold_judge_strength"] = hv.get("strength") if hv else None
            out["gold_phase_judged"] = bool(hv.get("called")) if hv else None

    # the restricted answer space, when one was applied. The ceiling of the
    # restriction IS ``gold_in_candidates``: no chooser, however good, can beat
    # it. And because the localiser is snapped into the set when it strays,
    # accuracy has to be split by whether it obeyed on its own — an accuracy
    # produced by snapping is a claim about the parse, not about the model.
    cand = pred.candidates or {}
    if cand:
        steps = [int(s) for s in (cand.get("steps") or [])]
        out["n_candidates"] = len(steps)
        # the set itself, for the confound the row cannot answer alone: how
        # often would a run that never saw the restriction have answered inside
        # it anyway? Without that baseline, ``candidate_obeyed`` cannot tell
        # the localiser adopting the constraint from the constraint agreeing
        # with where the localiser was already going.
        out["candidate_steps"] = steps
        out["candidate_obeyed"] = cand.get("obeyed") is True
        # whether straying was corrected. Without it, ``hit@0_presnap`` equal to
        # ``hit@0`` is ambiguous: it means either that enforcement never fired
        # or that there was no enforcement to fire.
        out["candidate_enforced"] = cand.get("enforced") is True
        out["candidate_moved_from"] = cand.get("moved_from")
        if gt.critical_step is not None:
            out["gold_in_candidates"] = gt.critical_step in steps
            # what the answer would have been without the snap, scored the same
            # way: the difference between this and hit@0 is exactly what forcing
            # the answer into the set bought.
            pre = (cand.get("moved_from") if cand.get("moved_from") is not None
                   else pred.critical_step)
            out["hit@0_presnap"] = (pre is not None
                                    and pre == gt.critical_step)

    # The multi-expert localiser. Recorded per case so the fusion can be
    # re-scored offline: ``expert_steps`` is what each specialist actually said,
    # and whether the gold step was named by ANY of them is the ceiling the
    # fusion rule was working against. A run where ``expert_ceiling`` is far
    # above ``hit@0`` is losing answers in the vote, not in the experts.
    exp = pred.experts or {}
    if exp:
        verdicts = exp.get("verdicts") or []
        fusion = exp.get("fusion") or {}
        named = {v.get("step") for v in verdicts
                 if v.get("found") and v.get("step") is not None}
        out["expert_steps"] = {str(v.get("expert")): v.get("step")
                               for v in verdicts}
        out["expert_found"] = sorted(str(v.get("expert")) for v in verdicts
                                     if v.get("found"))
        out["expert_n_voting"] = fusion.get("n_voting", 0)
        out["expert_backers"] = fusion.get("backers") or []
        out["expert_agreement"] = fusion.get("agreement", 0.0)
        out["expert_lead"] = exp.get("lead")
        out["expert_fault_category"] = exp.get("fault_category") or ""
        out["expert_distinct_steps"] = len(named)
        if gt.critical_step is not None:
            out["expert_ceiling"] = gt.critical_step in named
            # which specialist, if any, was on it — the per-remit hit rate is
            # what says whether a weak expert is weak or merely rarely relevant
            out["expert_hits"] = sorted(
                str(v.get("expert")) for v in verdicts
                if v.get("found") and v.get("step") == gt.critical_step)

    # Self-consistency. ``vote_ceiling`` is whether ANY draw named the reference
    # step: the gap between it and ``hit@0`` is what the majority rule threw
    # away, and is the only thing that says whether a different rule could do
    # better on the draws already paid for.
    smp = pred.samples or {}
    if smp:
        votes = {int(k): int(v) for k, v in (smp.get("votes") or {}).items()}
        out["vote_cast"] = smp.get("cast", 0)
        out["vote_agreement"] = smp.get("agreement", 0.0)
        out["vote_distinct_steps"] = len(votes)
        out["vote_unanimous"] = len(votes) == 1 and bool(votes)
        if gt.critical_step is not None:
            out["vote_ceiling"] = gt.critical_step in votes
            out["vote_gold_share"] = (
                votes.get(gt.critical_step, 0) / float(smp.get("cast") or 1))

    # The second pass. What has to be separable here is the answer BEFORE the
    # revision from the answer after it: a stage that fixes four cases and
    # breaks four is worth nothing, and the two are invisible in acc@0 alone.
    zm = pred.zoom or {}
    if zm:
        fin = list(zm.get("finalists") or [])
        out["zoom_n_finalists"] = len(fin)
        out["zoom_changed"] = bool(zm.get("accepted"))
        out["zoom_in_set"] = bool(zm.get("in_set"))
        out["zoom_confidence"] = zm.get("confidence")
        if gt.critical_step is not None:
            first = zm.get("first")
            out["hit@0_prezoom"] = (first is not None
                                    and first == gt.critical_step)
            # the ceiling this stage was working against: it can only ever
            # answer with a finalist, so a reference step outside the set was
            # unreachable no matter how well the model read
            out["zoom_ceiling"] = gt.critical_step in fin
            if zm.get("accepted"):
                out["zoom_fixed"] = (not out["hit@0_prezoom"]
                                     and bool(out.get("hit@0")))
                out["zoom_broke"] = (out["hit@0_prezoom"]
                                     and not bool(out.get("hit@0")))

    # The group-stage bracket, scored on exactly the same net principle: what it
    # fixed against what it broke, plus the ceiling it was working against.
    # ``bracket_abstained`` is the diagnostic that tells a null result apart
    # from a stage that never fired.
    br = pred.bracket or {}
    if br:
        fin = list(br.get("finalists") or [])
        out["bracket_n_finalists"] = len(fin)
        out["bracket_n_groups"] = br.get("n_groups")
        out["bracket_abstained"] = br.get("abstained")
        out["bracket_n_verdicts"] = len(br.get("verdicts") or [])
        out["bracket_changed"] = bool(br.get("accepted"))
        out["bracket_confidence"] = br.get("confidence")
        if gt.critical_step is not None:
            first = br.get("first")
            out["hit@0_prebracket"] = (first is not None
                                       and first == gt.critical_step)
            out["bracket_ceiling"] = gt.critical_step in fin
            # a separate ceiling for the final call: once the groups have
            # spoken, only a group winner can still be answered, so this is
            # where a loss is the GROUP stage's fault rather than the final's
            wins = [v.get("step") for v in (br.get("verdicts") or [])]
            out["bracket_group_ceiling"] = gt.critical_step in wins
            if br.get("accepted"):
                out["bracket_fixed"] = (not out["hit@0_prebracket"]
                                        and bool(out.get("hit@0")))
                out["bracket_broke"] = (out["hit@0_prebracket"]
                                        and not bool(out.get("hit@0")))

    # The predecessor audit, same net principle. Two extra diagnostics it needs
    # and the others do not: ``audit_ceiling`` is the ONLY thing this stage
    # could ever have reached (it may answer nothing but a parent), and
    # ``audit_move_refused`` separates a model that never wanted to move from
    # one whose four conditions kept refusing it -- opposite lessons.
    au = pred.audit or {}
    if au:
        par = list(au.get("parents") or [])
        out["audit_n_parents"] = len(par)
        out["audit_changed"] = bool(au.get("accepted"))
        out["audit_move_refused"] = bool(au.get("move")
                                         and not au.get("accepted"))
        out["audit_edge_kind"] = au.get("edge_kind")
        out["audit_confidence"] = au.get("confidence")
        if gt.critical_step is not None:
            first = au.get("first")
            out["hit@0_preaudit"] = (first is not None
                                     and first == gt.critical_step)
            out["audit_ceiling"] = gt.critical_step in par
            if au.get("accepted"):
                out["audit_fixed"] = (not out["hit@0_preaudit"]
                                      and bool(out.get("hit@0")))
                out["audit_broke"] = (out["hit@0_preaudit"]
                                      and not bool(out.get("hit@0")))
    return out


def _in_guess(step: int, guesses: List[Dict]) -> bool:
    return any(g.get("start", 0) <= step <= g.get("end", -1) for g in guesses)


def _prior_steps(guesses: List[Dict]) -> int:
    """Distinct steps the prior nominated. Guesses can overlap after snapping."""
    covered: set = set()
    for g in guesses:
        try:
            covered.update(range(int(g["start"]), int(g["end"]) + 1))
        except (KeyError, TypeError, ValueError):
            continue
    return len(covered)


def _painted(flow: Dict) -> Dict[str, set]:
    """The steps each relation actually paints, rebuilt from the stored graph.

    This must match ``FlowGraph.paint`` exactly, and specifically must NOT be
    "every endpoint of every edge". Under backward tracing those coincided,
    because only the traced steps had edges at all. Forward construction gives
    every step a parent, so the endpoint definition marks the whole trajectory
    and reports ``flow_recall = 1.0`` no matter how badly the painting did.

    NOTE, because it has misled once already: this is ALWAYS the separate
    reading. The graph is rebuilt from the stored dict, and how a run chose to
    read its flow (``mixed_paint`` / any later relaxation) is deliberately not
    serialised -- see ``FlowGraph.on_flow``. So the reported ``flow_recall`` and
    the ``painting:`` line do NOT move when a paint flag is switched on, and an
    unchanged flow_recall is NOT evidence that the flag failed to take effect.
    Recompute the reading you actually want from the stored flows; that is what
    probe/paint_frontier.py is for. Evidence a paint flag took effect is that
    the run made uncached calls where the control made none.

    THE CONVERSE, so the warning above is not over-read: ``anchors`` IS
    serialised, so a flag that changes WHERE the anchors go -- ``--indeg-anchors``
    -- does move the reported flow_recall, and is the one intervention so far
    that may be read off this number directly. The blindness is to how the flow
    is read, not to what it is painted from.
    """
    from .schema import CONTROL_FLOW, DATA_FLOW, FlowEdge, FlowGraph

    g = FlowGraph(anchors=[i for i in (flow.get("anchors") or [])
                           if isinstance(i, int)])
    for e in (flow.get("edges") or []):
        src, dst, kind = e.get("src"), e.get("dst"), e.get("kind")
        if isinstance(src, int) and isinstance(dst, int) and kind in (
                DATA_FLOW, CONTROL_FLOW):
            g.add(FlowEdge(src=src, dst=dst, kind=kind))
    data, ctrl = g.data_flow_steps(), g.control_flow_steps()
    return {"data": data, "control": ctrl, "union": data | ctrl}


def _in_weak_span(step: int, spans: List[Dict]) -> bool:
    for sp in spans:
        if sp.get("start", 0) <= step <= sp.get("end", -1):
            return sp.get("tier") == "weak"
    return False


def aggregate(rows: List[Dict[str, object]],
              tolerances: Iterable[int] = TOLERANCES) -> Dict[str, object]:
    """Aggregate per-case rows into a summary."""
    scored = [r for r in rows if r.get("gold") is not None]
    n = len(scored)
    summary: Dict[str, object] = {"n_cases": len(rows), "n_scored": n}
    if not n:
        return summary
    for t in tolerances:
        k = "hit@%d" % t
        summary["acc@%d" % t] = round(sum(1 for r in scored if r.get(k)) / n, 4)
    dists = [r["distance"] for r in scored if r.get("distance") is not None]
    summary["mean_distance"] = round(sum(dists) / len(dists), 2) if dists else None
    summary["median_distance"] = (sorted(dists)[len(dists) // 2] if dists else None)

    onf = [r for r in scored if r.get("gold_on_flow") is not None]
    if onf:
        summary["flow_recall"] = round(
            sum(1 for r in onf if r["gold_on_flow"]) / len(onf), 4)
        # Recall alone cannot say whether the flow is useful: painting the whole
        # trajectory scores 1.0 and localises nothing. Report each relation's
        # recall next to the fraction of the trajectory it paints, so a gain in
        # recall bought purely by painting more is visible as such.
        for name, key in (("data", "gold_on_data_flow"),
                          ("control", "gold_on_control_flow")):
            vals = [r for r in onf if r.get(key) is not None]
            if vals:
                summary["%s_recall" % name] = round(
                    sum(1 for r in vals if r[key]) / len(vals), 4)
        for name, key in (("flow", "flow_size"), ("data_flow", "data_flow_size"),
                          ("control_flow", "control_flow_size")):
            cov = [r[key] / r["n_steps"] for r in onf
                   if isinstance(r.get(key), int) and r.get("n_steps")]
            if cov:
                summary["%s_coverage" % name] = round(sum(cov) / len(cov), 4)
    inw = [r for r in scored if r.get("gold_in_weak_span") is not None]
    if inw:
        summary["weak_span_recall"] = round(
            sum(1 for r in inw if r["gold_in_weak_span"]) / len(inw), 4)

    # Span prior. Recall must be read against coverage: a prior that nominates
    # half the run and recalls half the gold steps carries no information, and
    # only the pair of numbers shows that. ``prior_followed`` is separate again
    # — it says whether the localiser went where it was pointed, which is what
    # distinguishes "the hint was wrong" from "the hint was ignored".
    pri = [r for r in scored if r.get("gold_in_prior_span") is not None]
    if pri:
        summary["prior_recall"] = round(
            sum(1 for r in pri if r["gold_in_prior_span"]) / len(pri), 4)
        summary["prior_top_recall"] = round(
            sum(1 for r in pri if r.get("gold_in_top_prior_span")) / len(pri), 4)
        cov = [r["prior_span_steps"] / r["n_steps"] for r in pri
               if isinstance(r.get("prior_span_steps"), int) and r.get("n_steps")]
        if cov:
            summary["prior_coverage"] = round(sum(cov) / len(cov), 4)
        fol = [r for r in pri if r.get("pred_in_prior_span") is not None]
        if fol:
            summary["prior_followed"] = round(
                sum(1 for r in fol if r["pred_in_prior_span"]) / len(fol), 4)
        # the number the stage was built to produce: does landing in the
        # nominated region actually coincide with getting the answer right?
        inside = [r for r in pri if r.get("gold_in_prior_span")]
        if inside:
            summary["acc@0_given_prior_hit"] = round(
                sum(1 for r in inside if r.get("hit@0")) / len(inside), 4)
        outside = [r for r in pri if not r.get("gold_in_prior_span")]
        if outside:
            summary["acc@0_given_prior_miss"] = round(
                sum(1 for r in outside if r.get("hit@0")) / len(outside), 4)

    # The restricted answer space. ``candidate_recall`` is the ceiling and
    # ``candidate_coverage`` is what that ceiling cost in answer space — the
    # same pairing as the prior. ``candidate_obeyed`` and ``acc@0_presnap``
    # separate what the localiser did from what the snap did for it: if acc@0
    # only exceeds acc@0_presnap, the win belongs to the parse, not the prompt.
    cnd = [r for r in scored if r.get("n_candidates") is not None]
    if cnd:
        summary["n_with_candidates"] = len(cnd)
        summary["mean_candidates"] = round(
            sum(r["n_candidates"] for r in cnd) / len(cnd), 2)
        cov = [r["n_candidates"] / r["n_steps"] for r in cnd if r.get("n_steps")]
        if cov:
            summary["candidate_coverage"] = round(sum(cov) / len(cov), 4)
        summary["candidate_obeyed"] = round(
            sum(1 for r in cnd if r.get("candidate_obeyed")) / len(cnd), 4)
        summary["candidate_enforced"] = round(
            sum(1 for r in cnd if r.get("candidate_enforced")) / len(cnd), 4)
        rec = [r for r in cnd if r.get("gold_in_candidates") is not None]
        if rec:
            summary["candidate_recall"] = round(
                sum(1 for r in rec if r["gold_in_candidates"]) / len(rec), 4)
            summary["acc@0_on_restricted"] = round(
                sum(1 for r in rec if r.get("hit@0")) / len(rec), 4)
            summary["acc@0_presnap"] = round(
                sum(1 for r in rec if r.get("hit@0_presnap")) / len(rec), 4)
            inside = [r for r in rec if r["gold_in_candidates"]]
            if inside:
                summary["acc@0_given_candidate_hit"] = round(
                    sum(1 for r in inside if r.get("hit@0")) / len(inside), 4)
            outside = [r for r in rec if not r["gold_in_candidates"]]
            if outside:
                # gold outside the set: every hit here is a case the restriction
                # could only have taken away, so this should be ~0 by
                # construction. A nonzero value means the snap is not enforcing.
                summary["acc@0_given_candidate_miss"] = round(
                    sum(1 for r in outside if r.get("hit@0")) / len(outside), 4)
        # cases the parse declined to restrict keep their unrestricted
        # behaviour, and must be scored apart rather than folded in
        rest = [r for r in scored if r.get("n_candidates") is None]
        if rest:
            summary["acc@0_unrestricted_fallback"] = round(
                sum(1 for r in rest if r.get("hit@0")) / len(rest), 4)
            summary["n_unrestricted_fallback"] = len(rest)

    # The multi-expert localiser, when it ran. Three things have to be
    # separable here, because they fail differently:
    #   expert_ceiling  — the gold step was named by SOME expert. This is what
    #                     the four readers between them could reach.
    #   acc@0           — the fusion picked it. The gap between the two is the
    #                     vote's loss, and it is the only part a rule change can
    #                     recover without spending a call.
    #   per-expert hits — which remits are carrying the run. An expert that
    #                     never hits is either mis-specified or covering a
    #                     failure family this corpus does not contain, and
    #                     ``found`` separates those: one that nominates often
    #                     and hits rarely is wrong, one that abstains is not.
    exps = [r for r in scored if r.get("expert_n_voting") is not None]
    if exps:
        n = float(len(exps))
        summary["n_with_experts"] = len(exps)
        summary["expert_mean_voting"] = round(
            sum(r.get("expert_n_voting", 0) for r in exps) / n, 2)
        summary["expert_mean_distinct"] = round(
            sum(r.get("expert_distinct_steps", 0) for r in exps) / n, 2)
        summary["expert_mean_agreement"] = round(
            sum(float(r.get("expert_agreement") or 0.0) for r in exps) / n, 4)
        # unanimity among those who spoke, i.e. one region not several
        summary["expert_unanimous"] = round(
            sum(1 for r in exps if r.get("expert_distinct_steps", 0) == 1) / n, 4)
        summary["expert_all_abstained"] = round(
            sum(1 for r in exps if r.get("expert_n_voting", 0) == 0) / n, 4)
        ceil = [r for r in exps if r.get("expert_ceiling") is not None]
        if ceil:
            summary["expert_ceiling"] = round(
                sum(1 for r in ceil if r["expert_ceiling"]) / len(ceil), 4)
            # what the vote kept of what the experts reached
            hit_and_reach = sum(1 for r in ceil
                                if r["expert_ceiling"] and r.get("hit@0"))
            reached = sum(1 for r in ceil if r["expert_ceiling"])
            if reached:
                summary["expert_fusion_kept"] = round(
                    hit_and_reach / float(reached), 4)
        per: Dict[str, Dict[str, int]] = {}
        for r in exps:
            for key in (r.get("expert_steps") or {}):
                d = per.setdefault(key, {"found": 0, "hit": 0, "lead": 0})
                if key in (r.get("expert_found") or []):
                    d["found"] += 1
                if key in (r.get("expert_hits") or []):
                    d["hit"] += 1
                if r.get("expert_lead") == key:
                    d["lead"] += 1
        summary["expert_per_remit"] = {
            k: {"found_rate": round(v["found"] / n, 4),
                "hit_rate": round(v["hit"] / n, 4),
                "lead_rate": round(v["lead"] / n, 4),
                # of the times it did nominate, how often it was right
                "precision": (round(v["hit"] / float(v["found"]), 4)
                              if v["found"] else None)}
            for k, v in sorted(per.items())}

    # Self-consistency, summarised the same way: ``vote_ceiling`` is the
    # accuracy an oracle picking among the draws would reach, so the gap to
    # ``acc@0`` is the headroom left in the vote rule rather than in the model.
    smps = [r for r in scored if r.get("vote_cast") is not None]
    if smps:
        n = float(len(smps))
        summary["n_with_samples"] = len(smps)
        summary["vote_mean_cast"] = round(
            sum(r.get("vote_cast", 0) for r in smps) / n, 2)
        summary["vote_mean_distinct"] = round(
            sum(r.get("vote_distinct_steps", 0) for r in smps) / n, 2)
        summary["vote_mean_agreement"] = round(
            sum(float(r.get("vote_agreement") or 0.0) for r in smps) / n, 4)
        summary["vote_unanimous"] = round(
            sum(1 for r in smps if r.get("vote_unanimous")) / n, 4)
        ceil = [r for r in smps if r.get("vote_ceiling") is not None]
        if ceil:
            summary["vote_ceiling"] = round(
                sum(1 for r in ceil if r["vote_ceiling"]) / len(ceil), 4)
            reached = sum(1 for r in ceil if r["vote_ceiling"])
            if reached:
                summary["vote_kept"] = round(
                    sum(1 for r in ceil
                        if r["vote_ceiling"] and r.get("hit@0")) / float(reached),
                    4)

    # The second pass, scored as a net: ``acc@0_prezoom`` is what the run would
    # have reported without it, ``zoom_fixed`` and ``zoom_broke`` are the two
    # halves of the difference. A stage whose two halves are equal is a stage
    # that did nothing but cost a call per trajectory.
    zms = [r for r in scored if r.get("zoom_n_finalists") is not None]
    if zms:
        n = float(len(zms))
        summary["n_with_zoom"] = len(zms)
        summary["zoom_mean_finalists"] = round(
            sum(r.get("zoom_n_finalists", 0) for r in zms) / n, 2)
        summary["zoom_changed"] = round(
            sum(1 for r in zms if r.get("zoom_changed")) / n, 4)
        summary["zoom_out_of_set"] = round(
            sum(1 for r in zms if not r.get("zoom_in_set")) / n, 4)
        pre = [r for r in zms if r.get("hit@0_prezoom") is not None]
        if pre:
            summary["acc@0_prezoom"] = round(
                sum(1 for r in pre if r["hit@0_prezoom"]) / len(pre), 4)
            summary["zoom_ceiling"] = round(
                sum(1 for r in pre if r.get("zoom_ceiling")) / len(pre), 4)
            summary["n_zoom_fixed"] = sum(1 for r in pre if r.get("zoom_fixed"))
            summary["n_zoom_broke"] = sum(1 for r in pre if r.get("zoom_broke"))

    # The bracket, same net. ``bracket_group_ceiling`` sits between the two
    # other ceilings and says where a loss happened: above it the shortlist was
    # wrong, between it and acc@0 the final call was wrong, below it the group
    # stage dropped the step before the final ever saw it.
    brs = [r for r in scored if r.get("bracket_n_finalists") is not None]
    if brs:
        n = float(len(brs))
        summary["n_with_bracket"] = len(brs)
        summary["bracket_mean_finalists"] = round(
            sum(r.get("bracket_n_finalists", 0) for r in brs) / n, 2)
        summary["bracket_mean_groups"] = round(
            sum(r.get("bracket_n_groups") or 0 for r in brs) / n, 2)
        summary["bracket_abstain_rate"] = round(
            sum(r.get("bracket_abstained") or 0 for r in brs)
            / max(1.0, sum(r.get("bracket_n_groups") or 0 for r in brs)), 4)
        summary["bracket_changed"] = round(
            sum(1 for r in brs if r.get("bracket_changed")) / n, 4)
        pre = [r for r in brs if r.get("hit@0_prebracket") is not None]
        if pre:
            summary["acc@0_prebracket"] = round(
                sum(1 for r in pre if r["hit@0_prebracket"]) / len(pre), 4)
            summary["bracket_ceiling"] = round(
                sum(1 for r in pre if r.get("bracket_ceiling")) / len(pre), 4)
            summary["bracket_group_ceiling"] = round(
                sum(1 for r in pre if r.get("bracket_group_ceiling"))
                / len(pre), 4)
            summary["n_bracket_fixed"] = sum(
                1 for r in pre if r.get("bracket_fixed"))
            summary["n_bracket_broke"] = sum(
                1 for r in pre if r.get("bracket_broke"))

    # The predecessor audit, same net. Read ``audit_ceiling`` FIRST: it is the
    # share of rows whose reference step was a direct parent at all, and the
    # stage cannot beat it no matter how well it reads. ``audit_changed``
    # against ``audit_move_refused`` says which half of the design is doing the
    # work -- a stage that never moves and a stage whose four conditions keep
    # refusing it both score zero and mean opposite things.
    aus = [r for r in scored if r.get("audit_n_parents") is not None]
    if aus:
        n = float(len(aus))
        summary["n_with_audit"] = len(aus)
        summary["audit_mean_parents"] = round(
            sum(r.get("audit_n_parents", 0) for r in aus) / n, 2)
        summary["audit_changed"] = round(
            sum(1 for r in aus if r.get("audit_changed")) / n, 4)
        summary["audit_move_refused"] = round(
            sum(1 for r in aus if r.get("audit_move_refused")) / n, 4)
        pre = [r for r in aus if r.get("hit@0_preaudit") is not None]
        if pre:
            summary["acc@0_preaudit"] = round(
                sum(1 for r in pre if r["hit@0_preaudit"]) / len(pre), 4)
            summary["audit_ceiling"] = round(
                sum(1 for r in pre if r.get("audit_ceiling")) / len(pre), 4)
            summary["n_audit_fixed"] = sum(
                1 for r in pre if r.get("audit_fixed"))
            summary["n_audit_broke"] = sum(
                1 for r in pre if r.get("audit_broke"))

    ratios = [r["compression_ratio"] for r in scored
              if isinstance(r.get("compression_ratio"), (int, float))
              and r["compression_ratio"] > 0]
    if ratios:
        summary["mean_compression_ratio"] = round(sum(ratios) / len(ratios), 4)
    calls = [r["calls"] for r in scored if isinstance(r.get("calls"), int)]
    if calls:
        summary["mean_llm_calls"] = round(sum(calls) / len(calls), 2)
    return summary


def group_by(rows: List[Dict[str, object]], key: str,
             tolerances: Iterable[int] = TOLERANCES) -> Dict[str, Dict]:
    buckets = defaultdict(list)
    for r in rows:
        buckets[str(r.get(key, "?"))].append(r)
    return {k: aggregate(v, tolerances) for k, v in sorted(buckets.items())}


def format_report(summary: Dict[str, object],
                  by_subset: Optional[Dict[str, Dict]] = None) -> str:
    lines = []
    lines.append("cases=%s scored=%s" % (summary.get("n_cases"),
                                         summary.get("n_scored")))
    accs = " ".join("acc@%d=%.3f" % (t, summary["acc@%d" % t])
                    for t in TOLERANCES if "acc@%d" % t in summary)
    lines.append(accs)
    extra = []
    for k, label in (("mean_distance", "mean_dist"),
                     ("median_distance", "median_dist"),
                     ("flow_recall", "flow_recall"),
                     ("weak_span_recall", "weak_recall"),
                     ("mean_compression_ratio", "compress_ratio"),
                     ("mean_llm_calls", "llm_calls")):
        if summary.get(k) is not None:
            extra.append("%s=%s" % (label, summary[k]))
    if extra:
        lines.append(" ".join(extra))
    # recall is only meaningful against how much got painted to earn it
    rel = [("%s_recall" % r, "%s_flow_coverage" % r, r)
           for r in ("data", "control")]
    parts = ["union rec=%s cov=%s" % (_f(summary.get("flow_recall")),
                                      _f(summary.get("flow_coverage")))]
    for rk, ck, name in rel:
        if summary.get(rk) is not None:
            parts.append("%s rec=%s cov=%s"
                         % (name, _f(summary.get(rk)), _f(summary.get(ck))))
    if len(parts) > 1:
        lines.append("painting: " + " | ".join(parts))
    if summary.get("prior_recall") is not None:
        lines.append("span prior: rec=%s top=%s cov=%s followed=%s | "
                     "acc@0 in=%s out=%s"
                     % (_f(summary.get("prior_recall")),
                        _f(summary.get("prior_top_recall")),
                        _f(summary.get("prior_coverage")),
                        _f(summary.get("prior_followed")),
                        _f(summary.get("acc@0_given_prior_hit")),
                        _f(summary.get("acc@0_given_prior_miss"))))
    if summary.get("candidate_recall") is not None:
        lines.append("candidates: n=%s ceiling=%s cov=%s obeyed=%s (%s) | "
                     "acc@0 here=%s presnap=%s | fallback %s on %s"
                     % (summary.get("mean_candidates"),
                        _f(summary.get("candidate_recall")),
                        _f(summary.get("candidate_coverage")),
                        _f(summary.get("candidate_obeyed")),
                        ("enforced" if summary.get("candidate_enforced") == 1.0
                         else "offered" if not summary.get("candidate_enforced")
                         else "mixed"),
                        _f(summary.get("acc@0_on_restricted")),
                        _f(summary.get("acc@0_presnap")),
                        _f(summary.get("acc@0_unrestricted_fallback")),
                        summary.get("n_unrestricted_fallback", 0)))
    if summary.get("expert_mean_voting") is not None:
        lines.append("experts: voting=%s distinct=%s agree=%s unanimous=%s "
                     "silent=%s | ceiling=%s fusion kept=%s"
                     % (summary.get("expert_mean_voting"),
                        summary.get("expert_mean_distinct"),
                        _f(summary.get("expert_mean_agreement")),
                        _f(summary.get("expert_unanimous")),
                        _f(summary.get("expert_all_abstained")),
                        _f(summary.get("expert_ceiling")),
                        _f(summary.get("expert_fusion_kept"))))
        per = summary.get("expert_per_remit") or {}
        for name in sorted(per):
            d = per[name]
            lines.append("  %-11s found=%s hit=%s lead=%s precision=%s"
                         % (name, _f(d.get("found_rate")), _f(d.get("hit_rate")),
                            _f(d.get("lead_rate")), _f(d.get("precision"))))
    if summary.get("vote_mean_cast") is not None:
        lines.append("samples: cast=%s distinct=%s agree=%s unanimous=%s | "
                     "ceiling=%s vote kept=%s"
                     % (summary.get("vote_mean_cast"),
                        summary.get("vote_mean_distinct"),
                        _f(summary.get("vote_mean_agreement")),
                        _f(summary.get("vote_unanimous")),
                        _f(summary.get("vote_ceiling")),
                        _f(summary.get("vote_kept"))))
    if summary.get("zoom_mean_finalists") is not None:
        lines.append("zoom: finalists=%s changed=%s out-of-set=%s | "
                     "acc@0 before=%s after=%s (fixed %s, broke %s) "
                     "ceiling=%s"
                     % (summary.get("zoom_mean_finalists"),
                        _f(summary.get("zoom_changed")),
                        _f(summary.get("zoom_out_of_set")),
                        _f(summary.get("acc@0_prezoom")),
                        _f(summary.get("acc@0")),
                        summary.get("n_zoom_fixed", 0),
                        summary.get("n_zoom_broke", 0),
                        _f(summary.get("zoom_ceiling"))))
    if summary.get("bracket_mean_finalists") is not None:
        lines.append("bracket: finalists=%s groups=%s abstain=%s changed=%s | "
                     "acc@0 before=%s after=%s (fixed %s, broke %s)"
                     % (summary.get("bracket_mean_finalists"),
                        summary.get("bracket_mean_groups"),
                        _f(summary.get("bracket_abstain_rate")),
                        _f(summary.get("bracket_changed")),
                        _f(summary.get("acc@0_prebracket")),
                        _f(summary.get("acc@0")),
                        summary.get("n_bracket_fixed", 0),
                        summary.get("n_bracket_broke", 0)))
        lines.append("bracket ceilings: shortlist=%s group winners=%s "
                     "(between them is the final call's loss, below is the "
                     "group stage's)"
                     % (_f(summary.get("bracket_ceiling")),
                        _f(summary.get("bracket_group_ceiling"))))
    if by_subset:
        lines.append("")
        lines.append("%-28s %6s %7s %7s %7s %7s" %
                     ("subset", "n", "acc@0", "acc@1", "acc@3", "flow"))
        for name, s in by_subset.items():
            lines.append("%-28s %6s %7s %7s %7s %7s" % (
                name[:28], s.get("n_scored", 0),
                _f(s.get("acc@0")), _f(s.get("acc@1")),
                _f(s.get("acc@3")), _f(s.get("flow_recall"))))
    return "\n".join(lines)


def _f(v):
    return "-" if v is None else ("%.3f" % v)


def write_results(path: str, rows: List[Dict], summary: Dict,
                  by_subset: Optional[Dict] = None,
                  run: Optional[Dict] = None) -> None:
    """Write a scored run. ``run`` records WHICH CONFIGURATION produced it.

    Without that block the filename is the only record of what was run, and a
    filename is not a record: two of these artifacts turned out to be the same
    run under different names, and the config that produced the best TrajAudit
    score had to be reverse-engineered out of a trace directory -- where it is
    only written when the flow is traced fresh, so every ``--reuse-flow`` run
    left its downstream knobs (span_prior, candidate_set, candidate_strict)
    recorded nowhere at all. Those are exactly the knobs an A/B varies.

    ``run`` is written first so ``head`` on the file shows it. It is never read
    back by anything in the pipeline; it exists to be read by a human comparing
    two results months apart, and to make a mislabelled comparison detectable.
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"run": run or {}, "summary": summary,
                   "by_subset": by_subset or {}, "cases": rows},
                  fh, indent=2, ensure_ascii=False)
