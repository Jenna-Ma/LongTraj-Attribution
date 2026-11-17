"""LongTraj — critical-step localisation in long agent trajectories.

Pipeline: backward tracing of data/control flow -> semantic spans -> per-span
summaries and step-level compression -> localisation over two artifacts:

  1. an ``Outline`` — how the agent executed the task, phase by phase;
  2. a list of ``StepDigest`` — every step, indices intact, with on-flow steps
     kept in full and off-flow steps condensed.

    from longtraj import load, LLM, Config, run

    llm = LLM(cache_dir=".cache")
    for rec in load("trajaudit", subset="swe-agent", limit=5):
        pred = run(rec.trajectory, llm, Config())
        print(pred.critical_step)

``rec.trajectory`` carries no label; ``rec.ground_truth`` is only ever read by
``longtraj.evaluate``.
"""
from .artifacts import (TraceStore, digests_from_list, flow_stats,
                        graph_from_dict, outline_from_dict, render_flow_text)
from .evaluate import aggregate, format_report, group_by, score_one
from .llm import LLM
from .loaders import (find_repo_root, load, load_all, load_longrca,
                      load_trajaudit, load_trajdebug)
from .outline import build_outline
from .pipeline import Config, run
from .schema import (COMPRESSED, CONTROL_FLOW, DATA_FLOW, KEEP, LIGHT, STRONG,
                     WEAK, FlowEdge, FlowGraph, GroundTruth, LoadedRecord,
                     Outline, Prediction, Span, Step, StepDigest, Trajectory)

__version__ = "0.2.0"

__all__ = [
    "LLM", "Config", "run", "build_outline",
    "load", "load_all", "load_longrca", "load_trajaudit", "load_trajdebug",
    "find_repo_root",
    "Trajectory", "Step", "GroundTruth", "LoadedRecord", "Prediction",
    "FlowGraph", "FlowEdge", "Span", "StepDigest", "Outline",
    "DATA_FLOW", "CONTROL_FLOW", "WEAK", "STRONG",
    "KEEP", "LIGHT", "COMPRESSED",
    "TraceStore", "graph_from_dict", "digests_from_list", "outline_from_dict",
    "flow_stats", "render_flow_text",
    "score_one", "aggregate", "group_by", "format_report",
]
