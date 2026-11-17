"""Enhanced localization with better prompt engineering and error type awareness.

Key improvements:
1. Add explicit error type classification before localization
2. Use error-type-specific reasoning prompts
3. Add explicit "red flags" checklist for common mistake patterns
4. Better handling of candidate sets with scoring
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

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

ERROR_TYPE_PROMPTS = {
    "wrong_file": """RED FLAGS FOR WRONG FILE ERRORS:
- Agent edited/read file X when the task required file Y
- Agent created a new file instead of modifying existing one
- Agent worked in wrong directory/module
Look for: file path decisions, ls/find commands, navigation steps""",

    "wrong_logic": """RED FLAGS FOR WRONG LOGIC ERRORS:
- Agent understood the requirement but implemented wrong algorithm
- Off-by-one errors, wrong operators, inverted conditions
- Correct file but wrong implementation
Look for: the step where code logic was decided or written""",

    "wrong_value": """RED FLAGS FOR WRONG VALUE ERRORS:
- Agent used wrong constant, variable name, or parameter
- Typo in identifier that wasn't caught
- Wrong function/class name
Look for: where the wrong value was first introduced (not just propagated)""",

    "incomplete": """RED FLAGS FOR INCOMPLETE ERRORS:
- Agent stopped too early, missing required steps
- Partial implementation that looks complete but isn't
- Forgot to handle edge case or special condition
Look for: where agent decided the task was done""",

    "wrong_understanding": """RED FLAGS FOR MISUNDERSTANDING ERRORS:
- Agent misread the requirement or task description
- Solved a different problem than what was asked
- Made incorrect assumption about system behavior
Look for: early planning/reasoning steps where intent was formed""",
}


def enhanced_localize(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    compressed: List[StepDigest],
    span_prior: Optional[SpanPrior],
    candidate_steps: Optional[List[int]],
    llm: LLM,
    cfg: Config,
) -> Prediction:
    """Enhanced localization with error type classification."""

    # Step 1: Classify error type first
    error_type = _classify_error_type(traj, graph, outline, llm)
    logger.info(f"Classified error type: {error_type}")

    # Step 2: Get error-type-specific red flags
    red_flags = ERROR_TYPE_PROMPTS.get(error_type, "")

    # Step 3: Run localization with enhanced prompt
    from .localize import build_localize_prompt

    # Build base prompt
    base_prompt = build_localize_prompt(
        traj, graph, outline, compressed, span_prior, candidate_steps, cfg
    )

    # Enhance with error-type awareness
    enhanced_prompt = f"""{base_prompt}

CLASSIFIED ERROR TYPE: {error_type}

{red_flags}

STEP-BY-STEP REASONING:
1. Review the failure symptom and identify what specifically went wrong
2. Consider the error type ({error_type}) and check for its red flags
3. Trace backwards from symptom through data/control flow
4. For each candidate, ask: does this match the error type pattern?
5. Pick the EARLIEST step that introduced this specific type of error

Answer with the same JSON format as requested above."""

    result = llm.generate_obj(
        enhanced_prompt,
        system="You locate critical steps in failed agent trajectories.",
        max_tokens=1500,
    )

    return Prediction(
        task_id=traj.task_id,
        critical_step=result.get("critical_step"),
        confidence=result.get("confidence", 0.5),
        error_type=result.get("error_type", "unknown"),
        reason=result.get("reason", ""),
    )


def _classify_error_type(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Outline,
    llm: LLM,
) -> str:
    """First pass: classify what type of error occurred."""

    prompt = f"""Classify the type of error in this failed agent run.

TASK:
{traj.task_description}

FAILURE SYMPTOM:
{graph.failure_summary}

HIGH-LEVEL EXECUTION:
{outline.overview}

ERROR TYPES:
- wrong_file: worked on wrong file/directory
- wrong_logic: correct file but wrong algorithm/implementation
- wrong_value: typo, wrong constant, wrong identifier
- incomplete: stopped too early, missing required steps
- wrong_understanding: misunderstood the requirement
- other: doesn't fit above categories

Output JSON with:
{{
  "error_type": "<one of the types above>",
  "reasoning": "<brief explanation>"
}}"""

    try:
        result = llm.generate_obj(
            prompt,
            system="You classify errors in agent trajectories.",
            max_tokens=400,
        )
        return result.get("error_type", "other")
    except Exception as e:
        logger.warning(f"Error type classification failed: {e}")
        return "other"
