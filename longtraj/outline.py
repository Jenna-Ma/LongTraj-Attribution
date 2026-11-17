"""Artifact 1: a high-level account of how the agent executed the task.

Built from the per-span sub-goal summaries, this answers "what were the main
steps taken to complete this task" in one place. The localiser reads it before
the step-level trajectory, so it starts with the shape of the whole run rather
than reconstructing it from fragments.

It is descriptive, never diagnostic: it says what the agent did, not what went
wrong. Diagnosis is the localiser's job, and pre-empting it here would bias the
answer toward whichever step this stage happened to find noteworthy.
"""
from __future__ import annotations

import logging
from typing import List

from .llm import LLM
from .schema import WEAK, FlowGraph, Outline, Span, Trajectory, clip_middle

logger = logging.getLogger(__name__)

OUTLINE_SYSTEM = ("You describe what an agent did, phase by phase, without "
                  "judging it.")

OUTLINE_PROMPT = """Below are the phases of one agent's attempt at a task, in
order. Write a high-level description of HOW THE AGENT EXECUTED THE TASK.

TASK:
{task}

PHASES ({n} spans over {steps} steps):
{phases}

Write 4-8 sentences covering, in order:
  - what the agent set out to do and how it oriented itself
  - the main things it actually did, in sequence, naming the concrete targets
    (files, commands, tests) it worked on
  - where it changed direction, retried, or worked around an obstacle
  - how the run ended

Refer to phases by their step ranges (e.g. "steps 10-13") so the description
can be lined up against the trajectory.

Describe only. Do NOT say which step was wrong, do NOT diagnose the failure,
and do NOT speculate about causes — that judgement is made later by a separate
reader. Return plain prose, no headings, no bullet points."""


def build_outline(traj: Trajectory, spans: List[Span], llm: LLM,
                  graph: FlowGraph = None) -> Outline:
    """Summarise the span summaries into one end-to-end description."""
    if not spans:
        return Outline(overview="", spans=[])

    phases = []
    for sp in spans:
        phases.append("steps %d-%d — %s\n    %s"
                      % (sp.start, sp.end, sp.label or "(unlabelled)",
                         (sp.summary or "(no summary)").strip()))
    body = clip_middle("\n".join(phases), 40000)

    text = llm.generate(
        OUTLINE_PROMPT.format(task=clip_middle(traj.task_description, 2000),
                              n=len(spans), steps=len(traj), phases=body),
        max_tokens=1200, system=OUTLINE_SYSTEM)
    text = (text or "").strip()
    if not text:
        logger.info("%s: outline fell back to concatenated span summaries",
                    traj.task_id)
        text = _fallback_overview(spans)
    return Outline(overview=text, spans=spans)


def _fallback_overview(spans: List[Span]) -> str:
    """Deterministic outline: the span summaries in order."""
    bits = []
    for sp in spans:
        bits.append("Steps %d-%d: %s" % (sp.start, sp.end,
                                         (sp.summary or sp.label or "").strip()))
    return " ".join(bits)


def outline_stats(outline: Outline) -> dict:
    return {
        "overview_chars": len(outline.overview or ""),
        "spans": len(outline.spans),
        "weak_spans": sum(1 for sp in outline.spans if sp.tier == WEAK),
        "summary_chars": sum(len(sp.summary or "") for sp in outline.spans),
    }
