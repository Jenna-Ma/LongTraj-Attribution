#!/usr/bin/env python3
"""
Pointwise localize: 给每步独立打分，取最高分。

与 listwise（现有）的核心差异：
- listwise: 给候选集 [A, B, C, D] → 模型排序 → 取第一个
  问题：模型容易被最显著的步骤（通常是 edit）钉住
- pointwise: 逐步打分 score(A), score(B), score(C), score(D) → argmax
  优势：每步独立评估，不受其他步干扰

实现策略：
1. 从 outline 提取候选步（~10-20 步）
2. 对每个候选，调用模型打分（0.0-1.0）
3. 取最高分作为答案

成本：每行 ~10-20 次调用（vs listwise 的 1 次）
"""
from __future__ import annotations
import re
from typing import List, Dict, Optional
from longtraj.schema import Trajectory, Prediction, Span, FlowGraph, Outline
from longtraj.llm import LLM, map_concurrent

POINTWISE_PROMPT = """You are analyzing a software engineering trajectory to find the critical mistake.

## Task
{task}

## Failure summary
{failure_summary}

## Outline (phase-level overview)
{outline}

## This candidate step (#{step_idx})
{step_text}

## Question
On a scale of 0.0 to 1.0, how likely is THIS SPECIFIC STEP the critical mistake that caused the failure?

Consider:
1. Does this step introduce a logical error, incorrect API usage, or wrong assumption?
2. Is the timing appropriate (did the agent have enough information)?
3. Apply the but-for test: if this step were correct, would the failure still occur?
4. Is this the EARLIEST step in the causal chain, or just where the consequence became visible?

Be calibrated:
- Most steps should score < 0.3 (not critical)
- Steps that propagate an earlier error: 0.2-0.4
- Plausible candidates: 0.5-0.7
- Clear critical mistakes: 0.8-1.0

Output JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


def pointwise_localize(
    traj: Trajectory,
    graph: FlowGraph,
    llm: LLM,
    outline: Optional[Outline] = None,
    max_candidates: int = 20,
) -> Prediction:
    """
    Pointwise 定位：给每个候选步独立打分。

    Args:
        traj: 轨迹
        graph: 流图
        llm: 模型
        outline: outline artifact（用于提取候选）
        max_candidates: 最多评估多少步

    Returns:
        Prediction with pred = argmax(scores)
    """
    # 1. 提取候选步：从 outline 里提到的步 + traced flow 上的步
    candidates = _extract_candidates(traj, graph, outline, max_candidates)

    if not candidates:
        # fallback: 第一个助理步
        first_assistant = next((i for i, s in enumerate(traj.steps) if s.role == "assistant"), 0)
        pred = Prediction(task_id=traj.task_id, critical_step=first_assistant, confidence=0.0)
        pred.guesses = []
        return pred

    # 2. 并发打分
    outline_text = outline.overview if outline else "(no outline)"

    def score_step(idx: int) -> Dict:
        step = traj.steps[idx]
        prompt = POINTWISE_PROMPT.format(
            task=traj.task_description[:500],
            failure_summary=graph.failure_summary or "(not summarized)",
            outline=outline_text[:1000],
            step_idx=idx,
            step_text=step.content[:800],
        )

        try:
            response = llm.generate(prompt, max_tokens=200, force_json=True)
            # 简单 JSON 提取
            import json
            data = json.loads(response)
            score = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", "")
            return {"idx": idx, "score": score, "reasoning": reasoning}
        except Exception as e:
            return {"idx": idx, "score": 0.0, "reasoning": f"Error: {e}"}

    # 并发打分（最多 20 个并发）
    scores = map_concurrent(score_step, candidates, workers=min(20, len(candidates)))

    # 3. 取最高分
    valid_scores = [s for s in scores if isinstance(s, dict) and "score" in s]
    if not valid_scores:
        first_assistant = next((i for i, s in enumerate(traj.steps) if s.role == "assistant"), 0)
        pred = Prediction(task_id=traj.task_id, critical_step=first_assistant, confidence=0.0)
        pred.guesses = []
        return pred

    best = max(valid_scores, key=lambda s: s["score"])

    # 4. 构造 guesses（按分数排序）
    sorted_scores = sorted(valid_scores, key=lambda s: s["score"], reverse=True)
    guesses = [
        {"step": s["idx"], "score": s["score"], "reasoning": s["reasoning"]}
        for s in sorted_scores[:5]
    ]

    pred = Prediction(
        task_id=traj.task_id,
        critical_step=best["idx"],
        confidence=best["score"],
        reason=best.get("reasoning", ""),
    )
    pred.guesses = guesses
    return pred


def _extract_candidates(
    traj: Trajectory,
    graph: FlowGraph,
    outline: Optional[Outline],
    max_candidates: int,
) -> List[int]:
    """
    从 outline 和 traced flow 提取候选步。

    策略：
    1. outline 里提到的步（通过 "step X" / "步骤 X" 关键词）
    2. traced flow 上的步（graph.anchors + 它们的前驱）
    3. 限制在 max_candidates 内
    """
    candidates = set()

    # 从 outline 提取
    if outline and outline.overview:
        text = outline.overview
        # 匹配 "step 5", "步骤 5", "Step 5:"
        matches = re.findall(r"(?:step|步骤)\s*(\d+)", text, re.IGNORECASE)
        for m in matches:
            idx = int(m)
            if 0 <= idx < len(traj.steps):
                candidates.add(idx)

    # 从 traced flow 提取
    candidates.update(graph.anchors)

    # 取前 max_candidates 个
    candidates = sorted(candidates)[:max_candidates]

    # 如果还是太少，补充助理步
    if len(candidates) < 5:
        assistant_steps = [i for i, s in enumerate(traj.steps) if s.role == "assistant"]
        # 均匀采样
        step = len(assistant_steps) // 5 if len(assistant_steps) >= 5 else 1
        sampled = assistant_steps[::step][:max_candidates]
        candidates = sorted(set(candidates) | set(sampled))

    return candidates


if __name__ == "__main__":
    print(__doc__)
