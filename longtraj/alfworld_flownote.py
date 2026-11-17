# -*- coding: utf-8 -*-
"""ALFWorld-only flow-evidence suppression. Zero extra LLM calls, default off.

WHY THIS EXISTS. It is the OTHER half of a two-sided intervention, and it
exists because the observation that motivated ``alfworld_anchors`` has two
mutually exclusive readings that no amount of looking can separate.

The diagnosis both share: the flow graph is traced BACKWARD FROM SYMPTOM
ANCHORS, and on ALFWorld the anchor is the last step on 99/100 records because
the failure is silent (see alfworld_anchors for the full measurement). So the
graph is rooted at a fact carrying no information: the run ended.

    READING A -- the graph is too NARROW and mis-rooted. Repaint it from
    better anchors; gold_on_flow goes 0.37 -> 0.66 offline. This is what
    ``--alf-anchors`` does.

    READING B -- the graph is HARMFUL EVIDENCE here. A backward walk from a
    zero-information root produces a chain that looks justified but is
    arbitrary, and the localiser is led by it. Then painting it wider only
    hands over more wrong evidence.

Measured on the stored baseline artifacts, dev 38 (probe/alf_anchor_settle.py,
reading the real ``FlowGraph.on_flow()`` backward walk):

    gold_on_flow                     7/38 = 0.1842
    acc@0 | gold ON flow             1/7  = 0.1429
    acc@0 | gold OFF flow           11/31 = 0.3548

Rows whose answer the graph actually covers are answered WORSE. That is
Reading B's prediction and Reading A has to call it confounding -- on-flow rows
do carry denser graphs (21.8 vs 16.0 edges) and accuracy RISES with flow size
by quartile (0.08/0.24/0.32/0.28), which is a real Simpson-shaped objection.
Observation cannot settle it; only intervening in BOTH directions can, which is
why this module ships alongside the one that does the opposite.

MODES.
    ``none``       -- replace both flow blocks and the SYMPTOM STEPS line with
                      one sentence saying the failure is silent and no symptom
                      anchor is available. The direct test of Reading B.
    ``anchorless`` -- keep the dependency edges, drop only the anchors line and
                      the hops-from-symptom annotations. Separates "the anchors
                      are harmful" from "the whole graph is harmful", which the
                      ``none`` arm alone would conflate.

WHY A NOTE AND NOT AN EMPTY STRING. An empty {data_block} would leave the
section headers standing over nothing, and the prompt's own text ("Follow it to
find where the information FIRST became wrong") would then instruct the reader
to follow a list that is not there. Saying plainly that the evidence is
unavailable is the honest rendering, and it is also what the prompt does for
every other absent stage.

See probe/prereg_alf_flownote.md, written before any arm's accuracy was
readable, for the arms, criteria and predictions.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MODES = ("none", "anchorless")

# Replaces both flow blocks under ``none``. Phrased as an absence of evidence,
# not as a hint about where to look: a sentence that nudged toward any region
# would make this arm a prompt intervention as well as an ablation, and the
# whole point is to remove one source of evidence while changing nothing else.
SILENT_NOTE = (
    "  (not available for this corpus: the failure is silent -- the run ends\n"
    "   by exhausting its step budget rather than by reporting an error, so\n"
    "   there is no symptom step to trace backward from, and no dependency\n"
    "   chain rooted at one.)")

ANCHORS_NONE = ("(none -- this run has no error-reporting step; the failure is "
                "that the task was never completed)")

# ``_flow_block`` annotates each edge with its distance from the symptom via
# ``_hops_from_symptom``, rendering it as the first entry of a "; "-joined mark
# list: "step 3 -> step 5 [x] (2 hops from symptom; names a failure artefact)".
# Under ``anchorless`` the edges stay but that annotation must go: it is a
# statement ABOUT the anchor, so leaving it would keep the anchor's influence
# in the prompt while claiming to have removed it. Both variants the renderer
# can emit are stripped, including the "not on a chain" branch, which is just
# as much a statement about the anchor as a hop count is.
_HOPS_MARK = re.compile(
    r"(?:\d+ hops? from symptom|not on a chain to any symptom)(?:; )?")
# A line whose only mark was the hop count is left with an empty "()", which
# would read as a rendering bug rather than as a removed annotation.
_EMPTY_MARKS = re.compile(r" \(\)")


def _strip_hops(block: str) -> str:
    return _EMPTY_MARKS.sub("", _HOPS_MARK.sub("", block))


def blocks(mode: Optional[str], data_block: str, control_block: str,
           anchors_line: str) -> Optional[Dict[str, str]]:
    """Return substitute prompt slots, or None when off.

    None means "use the values the caller already computed", which is what
    every TA/SP run gets: the prompt stays byte-identical and the cache key
    with it. An unknown mode raises rather than falling back silently, for the
    reason alfworld_anchors records -- a typo that quietly ran the incumbent
    would be recorded in the artifact as the mode having run, which is the one
    failure mode that corrupts an arm comparison after the fact.
    """
    if not mode:
        return None
    if mode not in MODES:
        raise ValueError("unknown alf_flownote mode %r; known: %s"
                         % (mode, ", ".join(MODES)))
    if mode == "none":
        return {"data_block": SILENT_NOTE, "control_block": SILENT_NOTE,
                "anchors": ANCHORS_NONE, "mode": mode}
    return {"data_block": _strip_hops(data_block),
            "control_block": _strip_hops(control_block),
            "anchors": ANCHORS_NONE, "mode": mode}
