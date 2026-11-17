"""Ensemble localization: combine multiple strategies and vote.

Strategy:
1. Run baseline localization with different framings
2. Add flow-guided localization (prioritize on-flow steps)
3. Add span-guided localization (use prior more heavily)
4. Combine with weighted voting based on confidence

This addresses the 48.8% flow recall by having dedicated flow-aware voters.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .llm import LLM
from .schema import (
    Config,
    FlowGraph,
    Outline,
    Prediction,
    SpanPrior,
    StepDigest,
    Trajectory,
)

logger = logging.getLogger(__name__)


def ensemble_localize(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    span_prior: Optional[SpanPrior],
    candidate_steps: Optional[List[int]],
    llm: LLM,
    cfg: Config,
) -> Prediction:
    """Ensemble approach: multiple strategies with weighted voting."""

    votes = []

    # Vote 1: Baseline causal framing (weight 2.0 - strongest)
    pred1 = _localize_causal(traj, graph, outline, compressed,
                             span_prior, candidate_steps, llm, cfg)
    if pred1.critical_step is not None:
        votes.append((pred1.critical_step, 2.0 * pred1.confidence, "causal"))

    # Vote 2: Flow-prioritized (weight 1.5)
    pred2 = _localize_flow_prioritized(traj, graph, outline, compressed,
                                        span_prior, candidate_steps, llm, cfg)
    if pred2.critical_step is not None:
        votes.append((pred2.critical_step, 1.5 * pred2.confidence, "flow"))

    # Vote 3: Span-prior-guided (weight 1.2)
    if span_prior and span_prior.spans:
        pred3 = _localize_span_guided(traj, graph, outline, compressed,
                                       span_prior, candidate_steps, llm, cfg)
        if pred3.critical_step is not None:
            votes.append((pred3.critical_step, 1.2 * pred3.confidence, "span"))

    if not votes:
        # Fallback
        return pred1 if pred1.critical_step is not None else Prediction(
            task_id=traj.task_id, critical_step=None
        )

    # Weighted voting
    vote_scores = {}
    for step, weight, source in votes:
        vote_scores[step] = vote_scores.get(step, 0) + weight

    best_step = max(vote_scores.items(), key=lambda x: x[1])[0]
    best_score = vote_scores[best_step]
    total_weight = sum(w for _, w, _ in votes)

    logger.info(f"Ensemble votes: {votes} -> chose {best_step} (score {best_score:.2f})")

    return Prediction(
        task_id=traj.task_id,
        critical_step=best_step,
        confidence=min(1.0, best_score / total_weight),
        reason=f"Ensemble of {len(votes)} strategies",
    )


def _localize_causal(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    span_prior: Optional[SpanPrior],
    candidate_steps: Optional[List[int]],
    llm: LLM,
    cfg: Config,
) -> Prediction:
    """Standard causal framing (current baseline)."""
    from .localize import localize
    return localize(traj, graph, outline, compressed, span_prior,
                   candidate_steps, llm, cfg)


def _localize_flow_prioritized(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    span_prior: Optional[SpanPrior],
    candidate_steps: Optional[List[int]],
    llm: LLM,
    cfg: Config,
) -> Prediction:
    """Flow-prioritized localization: focus on steps that are on traced flows."""

    # Build prompt emphasizing flow
    from .localize import build_localize_prompt

    base_prompt = build_localize_prompt(
        traj, graph, outline, compressed, span_prior, candidate_steps, cfg
    )

    # Add flow emphasis
    on_flow_steps = [s.index for s in compressed if s.on_flow]

    flow_emphasis = f"""

CRITICAL GUIDANCE - FLOW ANALYSIS:
Steps on traced data/control flows: {on_flow_steps}

These steps are directly connected to the failure through dependency chains.
While off-flow steps CAN be critical, prioritize on-flow steps unless you have
strong evidence that the root cause is elsewhere.

When choosing between two plausible candidates:
- If one is on-flow and one is off-flow, prefer the on-flow candidate
- If both are on-flow, apply standard but-for reasoning
- Only choose off-flow if it clearly introduces information that flows downstream
"""

    enhanced_prompt = base_prompt.replace(
        "HOW TO CHOOSE:",
        flow_emphasis + "\n\nHOW TO CHOOSE:"
    )

    result = llm.generate_obj(
        enhanced_prompt,
        system="You locate critical steps in failed agent trajectories.",
        max_tokens=1500,
    )

    return Prediction(
        task_id=traj.task_id,
        critical_step=result.get("critical_step"),
        confidence=result.get("confidence", 0.5) * 0.9,  # Slight discount
        reason=result.get("reason", ""),
    )


def _localize_span_guided(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    span_prior: Optional[SpanPrior],
    candidate_steps: Optional[List[int]],
    llm: LLM,
    cfg: Config,
) -> Prediction:
    """Span-prior-guided: heavily weight spans marked as suspicious."""

    if not span_prior or not span_prior.spans:
        return Prediction(task_id=traj.task_id, critical_step=None)

    # Get high-priority spans
    high_priority_spans = [s for s in span_prior.spans if s.score > 0.6]
    if not high_priority_spans:
        high_priority_spans = span_prior.spans[:2]  # Top 2

    # Build restricted candidate set from these spans
    restricted_candidates = []
    for span in high_priority_spans:
        span_steps = [s.index for s in compressed
                     if span.start <= s.index <= span.end]
        restricted_candidates.extend(span_steps)

    if candidate_steps:
        # Intersect with write candidates
        restricted_candidates = [s for s in restricted_candidates
                                if s in candidate_steps]

    if not restricted_candidates:
        return Prediction(task_id=traj.task_id, critical_step=None)

    from .localize import build_localize_prompt

    base_prompt = build_localize_prompt(
        traj, graph, outline, compressed, span_prior,
        restricted_candidates, cfg  # Use restricted set
    )

    result = llm.generate_obj(
        base_prompt,
        system="You locate critical steps in failed agent trajectories.",
        max_tokens=1500,
    )

    return Prediction(
        task_id=traj.task_id,
        critical_step=result.get("critical_step"),
        confidence=result.get("confidence", 0.5) * 0.85,  # Discount for restriction
        reason=result.get("reason", ""),
    )
