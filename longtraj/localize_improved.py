"""Improved single-shot localization with better flow utilization.

Key insight from evaluation:
- Flow recall is only 48.8% - half the gold steps are OFF the traced flows
- But when we DO hit the flow, accuracy is much higher
- Current prompt mentions flows but doesn't emphasize them strongly enough

Improvements:
1. Explicitly mark which steps are on data/control flows
2. Add "flow distance" - how many hops from failure symptom
3. Better guidance on when to trust flow vs when to look off-flow
4. Add explicit "alternative hypothesis" check
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from .compress import render_compressed
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


def improved_localize(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    span_prior: Optional[SpanPrior],
    candidate_steps: Optional[List[int]],
    llm: LLM,
    cfg: Config,
) -> Prediction:
    """Improved localization with better flow analysis."""

    # Calculate flow distances from anchors
    flow_info = _analyze_flow_distances(graph, compressed)

    # Build enhanced prompt
    prompt = _build_improved_prompt(
        traj, graph, outline, compressed, span_prior,
        candidate_steps, flow_info, cfg
    )

    result = llm.generate_obj(
        prompt,
        system="You locate critical steps in failed agent trajectories.",
        max_tokens=2000,
    )

    return Prediction(
        task_id=traj.task_id,
        critical_step=result.get("critical_step"),
        confidence=result.get("confidence", 0.5),
        error_type=result.get("error_type", "unknown"),
        reason=result.get("reason", ""),
    )


def _analyze_flow_distances(
    graph: FlowGraph,
    compressed: List[StepDigest],
) -> Dict[int, Dict]:
    """Calculate each step's distance from failure anchors via flows."""

    flow_info = {}

    # Get anchor steps
    anchors = [s.index for s in compressed if s.is_anchor]

    # BFS from anchors backwards through flows
    from collections import deque

    distances = {}
    queue = deque()

    for anchor in anchors:
        distances[anchor] = 0
        queue.append(anchor)

    while queue:
        curr = queue.popleft()
        curr_dist = distances[curr]

        # Follow edges backwards
        for edge in graph.predecessors(curr):
            if edge.src not in distances:
                distances[edge.src] = curr_dist + 1
                queue.append(edge.src)

    # Build info dict
    for step in compressed:
        flow_info[step.index] = {
            "on_flow": step.index in distances,
            "distance": distances.get(step.index, 999),
            "is_anchor": step.is_anchor,
        }

    return flow_info


def _build_improved_prompt(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    span_prior: Optional[SpanPrior],
    candidate_steps: Optional[List[int]],
    flow_info: Dict[int, Dict],
    cfg: Config,
) -> str:
    """Build improved prompt with flow analysis."""

    # Use base localize infrastructure
    from .localize import build_localize_prompt

    base = build_localize_prompt(
        traj, graph, outline, compressed, span_prior,
        candidate_steps, cfg
    )

    # Add flow analysis section
    on_flow_steps = [idx for idx, info in flow_info.items() if info["on_flow"]]
    close_steps = [idx for idx, info in flow_info.items()
                   if info["distance"] <= 3 and info["on_flow"]]

    flow_analysis = f"""

═══════════════════════════════════════════════════════════════════════
FLOW ANALYSIS (NEW)
═══════════════════════════════════════════════════════════════════════

The dependency tracing identified {len(on_flow_steps)} steps connected to the
failure through data or control flow edges.

Steps within 3 hops of failure symptoms: {close_steps}
All steps on flows: {on_flow_steps}

WHEN TO TRUST THE FLOW:
✓ If a step is on-flow AND matches the error pattern, it's likely correct
✓ Steps close to symptoms (distance ≤ 3) are strong candidates
✓ Flow tracing is good at finding where bad information propagated FROM

WHEN TO LOOK OFF-FLOW:
✗ If the error is a MISSING action (agent should have done X but never did)
✗ If the error is in PLANNING (wrong approach decided but not yet executed)
✗ If on-flow steps all look correct and you need to check earlier decisions

VERIFICATION CHECKLIST:
1. Is your chosen step on-flow? If YES, explain why this step over closer ones.
   If NO, explain why all on-flow steps are insufficient.

2. Could an off-flow step be responsible? Check:
   - Early planning/reasoning steps that set wrong direction
   - Missing validation checks
   - Incorrect assumptions stated but not acted on

3. Apply but-for test: if this step was corrected, would failure be prevented?

4. Is this the EARLIEST step in the causal chain?
"""

    # Insert before "HOW TO CHOOSE"
    enhanced = base.replace(
        "HOW TO CHOOSE:",
        flow_analysis + "\n\nHOW TO CHOOSE:"
    )

    # Enhance JSON output requirements
    enhanced = enhanced.replace(
        'Return ONLY JSON:',
        '''Return ONLY JSON (with flow analysis):
{{
  "on_flow": <boolean, is your answer on the traced flow?>,
  "flow_justification": "<if off-flow, explain why on-flow steps are insufficient>",'''
    )

    return enhanced
