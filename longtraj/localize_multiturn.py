"""Multi-turn coarse-to-fine localization: narrow down range progressively.

Instead of one-shot localization from 100+ steps, this approach:
  Round 1: Phase-level (6 phases → 1)
  Round 2: Span-level (5-7 spans → 1)
  Round 3: Step-level (5-10 steps → 1)
  Round 4: Verification (confirm or walk back 1-2 steps)

Each round focuses on a smaller context, avoiding information overload and
salience bias that makes edit steps dominate single-shot decisions.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .compress import render_compressed
from .llm import LLM
from .schema import (
    FlowGraph,
    Outline,
    Prediction,
    Span,
    StepDigest,
    Trajectory,
)

logger = logging.getLogger(__name__)

MULTITURN_SYSTEM = (
    "You perform root-cause localization on failed agent trajectories. "
    "You identify the earliest step whose error causally led to the observed "
    "failure. Answer strictly in JSON."
)


def multiturn_localize(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    llm: LLM,
) -> Prediction:
    """Multi-turn coarse-to-fine localization."""

    # Round 1: Phase-level localization
    phase_result = _round1_phase(traj, graph, outline, llm)
    chosen_phase_idx = phase_result["phase_idx"]

    # Round 2: Span-level localization within chosen phase
    phase_spans = _get_phase_spans(outline.spans, chosen_phase_idx)
    if not phase_spans:
        # Fallback: no spans in this phase, pick first span
        logger.warning(f"No spans in phase {chosen_phase_idx}, falling back")
        phase_spans = outline.spans[:1] if outline.spans else []

    span_result = _round2_span(traj, graph, phase_spans, phase_result, llm)
    chosen_span = span_result["span"]

    # Round 3: Step-level localization within chosen span
    span_steps = _get_span_steps(compressed, chosen_span)
    step_result = _round3_step(traj, graph, span_steps, span_result, llm)
    chosen_step = step_result["step"]

    # Round 4: Verification
    final_result = _round4_verify(traj, graph, compressed, chosen_step, llm)

    return Prediction(
        task_id=traj.task_id,
        critical_step=final_result["actual_step"],
        confidence=final_result.get("confidence", 0.5),
        reason=final_result.get("reasoning", ""),
    )


def _round1_phase(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    llm: LLM,
) -> Dict:
    """Round 1: Which phase contains the critical mistake?"""

    # Group spans into phases (use overview or create phases from spans)
    phases = _extract_phases(outline)

    prompt = f"""ROUND 1: PHASE-LEVEL LOCALIZATION

TASK:
{traj.task_description}

FAILURE SYMPTOM:
{graph.failure_summary}

PHASES (high-level execution flow):
{_render_phases(phases)}

Question: Which phase most likely contains the critical mistake that caused this failure?

Consider:
- Which phase's PURPOSE aligns with what went wrong
- Temporal causality: earlier phases affect later ones
- The failure symptom may surface late, but the cause is usually earlier

Output JSON with:
{{
  "phase_idx": <int, 0-indexed>,
  "reasoning": "<why this phase>"
}}"""

    result = llm.generate_obj(
        prompt,
        system=MULTITURN_SYSTEM,
        max_tokens=500,
    )

    logger.info(f"Round 1: chose phase {result['phase_idx']}")
    return result


def _round2_span(
    traj: Trajectory,
    graph: FlowGraph,
    phase_spans: List[Span],
    phase_result: Dict,
    llm: LLM,
) -> Dict:
    """Round 2: Which span within the phase contains the mistake?"""

    prompt = f"""ROUND 2: SPAN-LEVEL LOCALIZATION

You chose phase {phase_result['phase_idx']}: {phase_result['reasoning']}

FAILURE SYMPTOM:
{graph.failure_summary}

SPANS in this phase:
{_render_spans(phase_spans, graph)}

Question: Which span within this phase contains the critical mistake?

Consider:
- What each span accomplished
- Which span's actions relate to the failure entities: {", ".join(graph.failure_entities)}
- Dependency markers (on/off flow) indicate relevance

Output JSON with:
{{
  "span_idx": <int, index into the spans list above>,
  "reasoning": "<why this span>"
}}"""

    result = llm.generate_obj(
        prompt,
        system=MULTITURN_SYSTEM,
        max_tokens=500,
    )

    chosen_idx = result["span_idx"]
    if 0 <= chosen_idx < len(phase_spans):
        result["span"] = phase_spans[chosen_idx]
    else:
        logger.warning(f"Invalid span_idx {chosen_idx}, falling back to first span")
        result["span"] = phase_spans[0]

    logger.info(f"Round 2: chose span {result['span'].start}-{result['span'].end}")
    return result


def _round3_step(
    traj: Trajectory,
    graph: FlowGraph,
    span_steps: List[StepDigest],
    span_result: Dict,
    llm: LLM,
) -> Dict:
    """Round 3: Which specific step in the span is the critical mistake?"""

    span = span_result["span"]

    # Render steps with flow information
    steps_text = _render_steps_with_flow(span_steps, graph)

    prompt = f"""ROUND 3: STEP-LEVEL LOCALIZATION

You chose span {span.start}-{span.end}: {span_result['reasoning']}

FAILURE SYMPTOM:
{graph.failure_summary}

STEPS in this span (verbatim):
{steps_text}

RELEVANT DATA FLOW EDGES (within and entering this span):
{_render_local_flow(span_steps, graph)}

Question: Which specific step is the critical mistake?

Consider:
- Which step INTRODUCED the error (not just where it surfaced)
- Data flow edges show where information came from
- Steps marked as "anchor" or on flow are where failure is visible
- Look for: wrong logic, wrong value, wrong file, missing check

Output JSON with:
{{
  "step": <int, the step index>,
  "reasoning": "<why this step>"
}}"""

    result = llm.generate_obj(
        prompt,
        system=MULTITURN_SYSTEM,
        max_tokens=800,
    )

    logger.info(f"Round 3: chose step {result['step']}")
    return result


def _round4_verify(
    traj: Trajectory,
    graph: FlowGraph,
    compressed: List[StepDigest],
    chosen_step: int,
    llm: LLM,
) -> Dict:
    """Round 4: Verify or walk back to find the EARLIEST error."""

    # Get context around chosen step
    step_digest = next((s for s in compressed if s.index == chosen_step), None)
    if not step_digest:
        logger.warning(f"Step {chosen_step} not in compressed, skipping verification")
        return {
            "actual_step": chosen_step,
            "confidence": 0.3,
            "reasoning": "Could not verify (step not found in compressed trajectory)",
        }

    # Get predecessors from flow
    preds = graph.predecessors(chosen_step)
    pred_steps = []
    for edge in preds[:5]:  # Limit to 5 predecessors
        pred_digest = next((s for s in compressed if s.index == edge.src), None)
        if pred_digest:
            pred_steps.append((edge, pred_digest))

    prompt = f"""ROUND 4: VERIFICATION

You identified step {chosen_step} as the critical mistake.

CHOSEN STEP:
{step_digest.render()}

FAILURE SYMPTOM:
{graph.failure_summary}

PREDECESSORS (steps that led to this one):
{_render_predecessors(pred_steps)}

Question: Is step {chosen_step} the EARLIEST step that introduced the error,
or is it just where the consequence became visible?

Consider:
- If this step used wrong information, which predecessor produced it?
- If this step made a wrong decision, was it based on wrong earlier reasoning?
- If this step failed, was it because an earlier step set it up incorrectly?

Output JSON with:
{{
  "confirmed": <bool, true if step {chosen_step} is correct>,
  "actual_step": <int, {chosen_step} if confirmed, else the real earliest step>,
  "reasoning": "<explain your decision>",
  "confidence": <float 0.0-1.0>
}}"""

    result = llm.generate_obj(
        prompt,
        system=MULTITURN_SYSTEM,
        max_tokens=800,
    )

    if result["confirmed"]:
        logger.info(f"Round 4: confirmed step {chosen_step}")
    else:
        logger.info(f"Round 4: corrected {chosen_step} → {result['actual_step']}")

    return result


# Helper functions for data extraction and rendering

def _extract_phases(outline: Outline) -> List[Dict]:
    """Extract phases from outline spans.

    For now, create coarse phases by grouping spans.
    Later can use outline.overview parsing if it has phase structure.
    """
    if not outline.spans:
        return [{"idx": 0, "name": "Entire trajectory", "start": 0, "end": 999, "summary": "No spans available"}]

    # Simple heuristic: group every 3-5 spans into a phase
    phases = []
    n_spans = len(outline.spans)
    phase_size = max(3, n_spans // 6)  # Target ~6 phases

    for i in range(0, n_spans, phase_size):
        chunk = outline.spans[i:i+phase_size]
        phases.append({
            "idx": len(phases),
            "name": f"Phase {len(phases)+1}",
            "start": chunk[0].start,
            "end": chunk[-1].end,
            "spans": chunk,
            "summary": _summarize_phase(chunk),
        })

    return phases


def _summarize_phase(spans: List[Span]) -> str:
    """Summarize what a group of spans accomplished."""
    labels = [sp.label for sp in spans if sp.label]
    if not labels:
        return f"Steps {spans[0].start}-{spans[-1].end}"

    # Concatenate first 3 span labels
    summary = "; ".join(labels[:3])
    if len(labels) > 3:
        summary += f" (and {len(labels)-3} more)"
    return summary


def _render_phases(phases: List[Dict]) -> str:
    """Render phases for Round 1 prompt."""
    lines = []
    for p in phases:
        lines.append(f"{p['idx']}. {p['name']} (steps {p['start']}-{p['end']})")
        lines.append(f"   {p['summary']}")
    return "\n".join(lines)


def _get_phase_spans(all_spans: List[Span], phase_idx: int) -> List[Span]:
    """Get spans belonging to a phase."""
    phases = _extract_phases(Outline(spans=all_spans))
    if 0 <= phase_idx < len(phases):
        return phases[phase_idx]["spans"]
    return []


def _render_spans(spans: List[Span], graph: FlowGraph) -> str:
    """Render spans for Round 2 prompt."""
    lines = []
    for i, sp in enumerate(spans):
        flag = "on dependency path" if sp.tier == "weak" else "off path"
        lines.append(f"{i}. Steps {sp.start}-{sp.end} [{flag}]")
        lines.append(f"   {sp.label or '(no label)'}")
        if sp.summary:
            lines.append(f"   {sp.summary[:200]}")
    return "\n".join(lines)


def _get_span_steps(compressed: List[StepDigest], span: Span) -> List[StepDigest]:
    """Get steps within a span."""
    return [s for s in compressed if span.start <= s.index <= span.end]


def _render_steps_with_flow(steps: List[StepDigest], graph: FlowGraph) -> str:
    """Render steps with flow markers for Round 3."""
    return "\n\n".join(s.render() for s in steps)


def _render_local_flow(steps: List[StepDigest], graph: FlowGraph) -> str:
    """Render data flow edges within/entering a span."""
    step_indices = {s.index for s in steps}

    # Find edges where dst is in span
    relevant_edges = [
        e for e in graph.edges
        if e.dst in step_indices and e.kind == "data"
    ]

    if not relevant_edges:
        return "(no data flow edges into this span)"

    lines = []
    for e in relevant_edges[:10]:  # Limit to 10
        lines.append(f"step {e.src} → step {e.dst} [{e.entity}]")
        if e.rationale:
            lines.append(f"  {e.rationale}")

    return "\n".join(lines)


def _render_predecessors(pred_steps: List[Tuple]) -> str:
    """Render predecessor steps for Round 4."""
    if not pred_steps:
        return "(no predecessors in flow graph)"

    lines = []
    for edge, digest in pred_steps:
        lines.append(f"Step {edge.src} → {edge.dst} [{edge.entity}]:")
        lines.append(digest.render())
        lines.append("")

    return "\n".join(lines)
