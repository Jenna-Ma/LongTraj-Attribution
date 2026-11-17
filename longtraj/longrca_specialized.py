"""LongRCA SWE-bench Pro episode-first locator (recorded 63/128 arm).

This is intentionally separate from the shared data/control-flow pipeline.
The integration facade routes here only when the trajectory contains the exact
LongRCA diagnose-to-action handoff speaker. No evaluation label is accepted or
read by this module.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple

from .llm import LLM
from .schema import Trajectory


HANDOFF_NAME = "DiagnostAgent (-> ActionAgent)"
METHOD_NAME = "longrca_sp_episode_guarded_v1"
LENGTH_RATIO = 3.0
OVERLAP_GAIN = 0.10
_TOKEN = re.compile(r"[a-z_][a-z0-9_]{2,}", re.I)

EP_SYSTEM = """You trace a failed coding-agent run backward from concrete test
and reviewer evidence to the first repair episode that committed the surviving
defect. Return only JSON."""

EP_PROMPT = """Select the execution episode that FIRST committed the defect
responsible for final failure.

An episode starts at a diagnose-to-action HANDOFF. Default to episode 0. Move
later only when earlier work was reasonable or addressed a different issue and
the later handoff introduces the surviving wrong API, omission, revert, or
repair direction. A later failure report is evidence, not automatically cause.

ORIGINAL TASK:
{task}

FINAL FAILURE / REVIEW EVIDENCE:
{outcome}

EPISODES:
{episodes}

Return {{"episode": <listed ordinal>, "handoff_step": <its listed step>,
"surviving_defect": "precise defect supported by outcome evidence",
"why_not_earlier": "causal comparison", "confidence": <0..1>}}.
"""


def is_longrca_trajectory(traj: Trajectory) -> bool:
    """Whether the exact routing marker occurs in the uploaded trajectory."""
    return any((step.name or "") == HANDOFF_NAME for step in traj.steps)


def _clip_middle(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    half = max(1, (limit - 80) // 2)
    return (text[:half] + "\n... [%d chars omitted] ...\n" %
            (len(text) - 2 * half) + text[-half:])


def _handoffs(traj: Trajectory) -> List[int]:
    return [step.index for step in traj.steps
            if (step.name or "") == HANDOFF_NAME]


def _bounds(traj: Trajectory, handoffs: List[int]) -> List[Tuple[int, int]]:
    return [(step, handoffs[i + 1] - 1
             if i + 1 < len(handoffs) else len(traj.steps) - 1)
            for i, step in enumerate(handoffs)]


def _outcome_evidence(traj: Trajectory) -> str:
    eligible = []
    for step in traj.steps:
        text = step.content or ""
        if (step.name or "") == "JudgeAgent" or (
                (step.name or "") == "Computer_terminal" and
                re.search(r"delegated agent finished|test|fail|error|reward",
                          text, re.I)):
            eligible.append(step)
    blocks = ["EVIDENCE step %d | %s\n%s" %
              (step.index, step.name or step.role,
               _clip_middle(step.content, 1800))
              for step in eligible[-8:]]
    return "\n\n".join(blocks) or "No explicit final report; infer from trace."


def _episode_ledger(traj: Trajectory, handoffs: List[int]) -> str:
    blocks = []
    for ordinal, (lo, hi) in enumerate(_bounds(traj, handoffs)):
        handoff = traj.steps[lo]
        tail = [step for step in traj.steps[lo + 1:hi + 1]
                if (step.name or "") in ("DiagnostAgent", "JudgeAgent")][-2:]
        text = "EPISODE %d | HANDOFF step %d\n%s" % (
            ordinal, lo, _clip_middle(handoff.content, 4200))
        for step in tail:
            text += "\nEPISODE-END step %d | %s\n%s" % (
                step.index, step.name, _clip_middle(step.content, 900))
        blocks.append(text)
    return "\n\n".join(blocks)


def _tokens(text: str) -> set:
    return {token.lower() for token in _TOKEN.findall(str(text or ""))}


def _mechanical_prediction(traj: Trajectory,
                           handoffs: List[int]) -> Tuple[int, Dict[str, Any]]:
    task = _tokens(traj.task_description)
    overlap = [len(task & _tokens(traj.steps[index].content)) /
               max(1, len(task)) for index in handoffs]
    later = max(
        range(1, len(handoffs)),
        key=lambda rank: (overlap[rank],
                          len(traj.steps[handoffs[rank]].content)),
        default=0,
    )
    ratio = (len(traj.steps[handoffs[later]].content) /
             max(1, len(traj.steps[handoffs[0]].content)))
    gain = overlap[later] - overlap[0]
    accepted = later > 0 and ratio >= LENGTH_RATIO and gain >= OVERLAP_GAIN
    prediction = handoffs[later] if accepted else handoffs[0]
    return prediction, {
        "default": handoffs[0],
        "proposed": handoffs[later],
        "accepted": accepted,
        "length_ratio": ratio,
        "task_overlap_gain": gain,
        "handoff_rank": later,
    }


def _usage_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in after.items():
        if isinstance(value, (int, float)):
            result[key] = value - before.get(key, 0)
    return result


def locate(traj: Trajectory, llm: LLM) -> Dict[str, Any]:
    """Run the exact episode gate + deterministic fallback composition."""
    handoffs = _handoffs(traj)
    if not handoffs:
        raise ValueError("LongRCA specialised locator requires a handoff")

    prompt = EP_PROMPT.format(
        task=_clip_middle(traj.task_description, 9000),
        outcome=_outcome_evidence(traj),
        episodes=_episode_ledger(traj, handoffs),
    )
    before = llm.snapshot()
    started = time.perf_counter()
    response = llm.generate_obj(prompt, max_tokens=1000, system=EP_SYSTEM)
    try:
        episode = int(response.get("episode", 0))
    except (AttributeError, TypeError, ValueError):
        episode = 0
    if not 0 <= episode < len(handoffs):
        episode = 0

    default = handoffs[0]
    proposed = handoffs[episode]
    episode_move_accepted = (
        episode == 0 or
        len(traj.steps[proposed].content or "") >=
        len(traj.steps[default].content or ""))
    episode_prediction = proposed if episode_move_accepted else default

    mechanical_prediction, mechanical_audit = _mechanical_prediction(
        traj, handoffs)
    final_prediction = (mechanical_prediction
                        if episode_prediction == default
                        else episode_prediction)
    return {
        "critical_step": final_prediction,
        "confidence": response.get("confidence"),
        "reason": str(response.get("surviving_defect") or ""),
        "error_type": "episode_handoff",
        "usage": _usage_delta(before, llm.snapshot()),
        "compression_ratio": None,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "diagnostics": {
            "handoff_steps": handoffs,
            "default": default,
            "episode": episode,
            "episode_proposed": proposed,
            "episode_move_accepted": episode_move_accepted,
            "episode_prediction": episode_prediction,
            "mechanical_prediction": mechanical_prediction,
            "mechanical_audit": mechanical_audit,
            "model_response": response,
            "prompt_chars": len(prompt),
        },
    }
