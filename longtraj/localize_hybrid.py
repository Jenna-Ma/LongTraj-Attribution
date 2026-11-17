"""Hybrid localization: combine single-shot with focused refinement.

Strategy:
  1. Single-shot to get top-K candidates (use existing strong baseline)
  2. For each candidate, do focused verification with local context
  3. Re-rank based on verification confidence

This preserves the baseline's strong first-pass while adding refinement.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .compress import render_compressed
from .llm import LLM
from .localize import localize as single_shot_localize
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


def hybrid_localize(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    span_prior: Optional[SpanPrior],
    candidate_steps: Optional[List[int]],
    llm: LLM,
    cfg: Config,
) -> Prediction:
    """Hybrid approach: single-shot + focused refinement."""

    # Step 1: Get baseline prediction
    baseline_pred = single_shot_localize(
        traj, graph, outline, compressed, span_prior, candidate_steps, llm, cfg
    )

    if baseline_pred.critical_step is None:
        return baseline_pred

    # Step 2: Generate top-K candidates from single-shot context
    # Use the same prompt but ask for top-3 instead of top-1
    candidates = _get_topk_candidates(
        traj, graph, outline, compressed, span_prior,
        candidate_steps, llm, cfg, k=3
    )

    if not candidates:
        # Fallback to baseline if we can't get candidates
        return baseline_pred

    # Step 3: Refine each candidate with focused verification
    verified = []
    for cand_step in candidates:
        score = _verify_candidate(
            traj, graph, compressed, cand_step, llm
        )
        verified.append((cand_step, score))

    # Step 4: Re-rank and pick best
    verified.sort(key=lambda x: x[1], reverse=True)
    best_step, best_score = verified[0]

    # If verification strongly prefers a different candidate, use it
    if best_step != baseline_pred.critical_step and best_score > 0.7:
        logger.info(
            f"Hybrid override: {baseline_pred.critical_step} → {best_step} "
            f"(score {best_score:.2f})"
        )
        return Prediction(
            task_id=traj.task_id,
            critical_step=best_step,
            confidence=best_score,
            reason=f"Refined from baseline {baseline_pred.critical_step}",
            error_type=baseline_pred.error_type,
            usage=baseline_pred.usage,
            compression_ratio=baseline_pred.compression_ratio,
        )
    else:
        # Stay with baseline
        return baseline_pred


def _get_topk_candidates(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    span_prior: Optional[SpanPrior],
    candidate_steps: Optional[List[int]],
    llm: LLM,
    cfg: Config,
    k: int = 3,
) -> List[int]:
    """Get top-K candidate steps from single-shot localization prompt."""

    # Reuse the existing localization prompt but ask for top-K
    from .localize import build_localize_prompt

    prompt = build_localize_prompt(
        traj, graph, outline, compressed, span_prior, candidate_steps, cfg
    )

    # Modify to ask for top-K
    prompt += f"\n\nProvide your top {k} candidate steps in order of likelihood."
    prompt += f"\n\nOutput JSON with:\n{{"
    prompt += f'\n  "candidates": [<step1>, <step2>, <step3>],'
    prompt += f'\n  "reasoning": "<brief explanation>"'
    prompt += f"\n}}"

    try:
        result = llm.generate_obj(prompt, max_tokens=1000)
        candidates = result.get("candidates", [])
        if isinstance(candidates, list) and len(candidates) > 0:
            return candidates[:k]
    except Exception as e:
        logger.warning(f"Failed to get top-K candidates: {e}")

    return []


def _verify_candidate(
    traj: Trajectory,
    graph: FlowGraph,
    compressed: List[StepDigest],
    candidate_step: int,
    llm: LLM,
) -> float:
    """Verify a candidate step with focused local context.

    Returns confidence score 0-1.
    """

    # Get the candidate step
    step_digest = next((s for s in compressed if s.index == candidate_step), None)
    if not step_digest:
        return 0.0

    # Get local context: predecessors and successors in flow
    preds = graph.predecessors(candidate_step)
    succs = graph.successors(candidate_step)

    # Build focused context
    context_steps = []
    for edge in preds[:3]:
        pred = next((s for s in compressed if s.index == edge.src), None)
        if pred:
            context_steps.append((edge.src, pred, "predecessor"))

    context_steps.append((candidate_step, step_digest, "CANDIDATE"))

    for edge in succs[:3]:
        succ = next((s for s in compressed if s.index == edge.dst), None)
        if succ:
            context_steps.append((edge.dst, succ, "successor"))

    # Render context
    context_text = "\n\n".join([
        f"Step {idx} [{label}]:\n{digest.content[:500]}"
        for idx, digest, label in context_steps
    ])

    prompt = f"""FOCUSED VERIFICATION

FAILURE SYMPTOM:
{graph.failure_summary}

LOCAL CONTEXT (candidate and its flow neighbors):
{context_text}

Question: Is the CANDIDATE step the root cause of the failure?

Consider:
- Does this step introduce wrong information that flows downstream?
- Is there a predecessor that better explains the root cause?
- Does the error only surface here but originate elsewhere?

Output JSON with:
{{
  "is_root_cause": <boolean>,
  "confidence": <float 0-1>,
  "reasoning": "<brief explanation>"
}}"""

    try:
        result = llm.generate_obj(
            prompt,
            system="You verify root cause hypotheses. Answer in JSON.",
            max_tokens=600,
        )

        if result.get("is_root_cause", False):
            return result.get("confidence", 0.5)
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"Verification failed for step {candidate_step}: {e}")
        return 0.0
