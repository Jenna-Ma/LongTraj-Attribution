# -*- coding: utf-8 -*-
"""ALFWorld-only symptom anchors. Zero LLM calls, additive, default off.

WHY THIS EXISTS. The flow graph is traced BACKWARD FROM SYMPTOM ANCHORS, and
``locate_anchors`` finds the step where the failure becomes VISIBLE. On a coding
corpus that is a traceback or a failing test. ALFWorld failures are silent: the
run hits its step limit or simply stops short of the goal state, so there is
nothing to see. Measured on all 100 records of the unified corpus:

    anchor == 59 (the last step)   99/100
    |anchor - gold|                median 40, mean 32.7
    anchor IS gold                 10/100  (and those 10 are the step_limit rows)
    anchors per record             1.5     (TA 2.6, SP 2.9)

``tracing.py`` says as much about the model it replaced -- located anchors sit
at relative position median 0.982 -- but on TA/SP that is harmless because the
end of the run is also where the error is reported. Here it means the whole
backward walk, every "N hops from symptom" edge annotation, and HOW TO CHOOSE
item 7 are rooted at the one fact that carries no information: the run ended.

``tracing.keyword_anchors`` cannot repair it either. Every pattern in
``_RX_SYMPTOM`` is code-specific (``Traceback``, pytest's ``E   ``, exit codes,
shell errors); none fires on household text, so the score degenerates to pure
recency and the rule re-picks the last few steps.

WHAT THIS USES INSTEAD. The corpus's own marking scheme makes "inefficient plan
/ memory over-simplification -- it revisits places it already searched" the most
common critical subtype, and ``go to <fixture>`` is explicit in the action text.
So the revisit set needs no model. Priced over the 100 records:

    pool               recall  size  lift
    first revisit      0.1500   1.0  5.49x
    all revisits       0.2600   6.3  1.51x
    (for contrast, measured in earlier rounds:)
    "Nothing happens."    --     --  0.91x
    rejected action       --     --  0.91x

5.49x is the first signal above 1.1x found on this corpus. Its recall is far
too low to gate an answer space -- that route is dead anyway, see below -- but
an ANCHOR does not have to be the answer, only close to it. Repainting the 100
stored graphs offline (``FlowGraph.paint`` is pure, so this costs nothing):

    anchor set                    gold_on_flow  anchors  steps_on_flow  med|d|
    as shipped                          0.3700     1.51           21.5      40
    revisits UNION located              0.6600     6.41           33.7       4
    first-2 revisits UNION located      0.6000     3.01           27.5       4

For scale, the same quantity is 0.6961 on TrajAudit (acc@0 0.5000) and 0.4884
on SWE-Bench-Pro (acc@0 0.3953), against ALFWorld's 0.34 (acc@0 0.2300).

WHY ANCHORS AND NOT A CANDIDATE POOL. Six rounds of pool-shaped interventions
all failed for one reason: ``acc = recall*pick_in + (1-recall)*pick_out`` with
``pick_in`` about 0.345, so recall does not convert. --verify-pool reached
recall 0.4737 and scored 0.2105 BELOW the no-pool baseline; the duel fired on
21/38 rows and moved hits 3->3; a 21-arm plurality vote scored 11/38 against
the best single arm's 12/38. Anchors are not an answer space -- they change
which EVIDENCE the localiser is shown, which is the one layer never touched.

HONEST COUNTER-EVIDENCE, recorded here because it predicts failure. Within
ALFWorld the conditional is INVERTED: rows whose gold is on the flow score
0.1176 and rows whose gold is off it score 0.2879 (n=34/66). All three graph
metrics agree, while ``gold_in_prior_span`` runs the expected way round
(0.3488/0.1404), so it is not a sign error. It is confounded -- on-flow rows
carry denser graphs (21.8 vs 16.0 edges) and accuracy RISES with flow_size by
quartile (0.08/0.24/0.32/0.28) -- but it cannot be dissolved by observation.
See probe/prereg_alf_anchor.md section 2.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ``go to <fixture>`` is the only navigation verb ALFWorld offers, and the
# fixture name carries its instance number ("cabinet 4"), which is what makes a
# revisit decidable without a model. Trailing digits are kept: "cabinet 4" and
# "cabinet 1" are different places, and collapsing them would invent revisits.
_GO = re.compile(r"go to ([a-z]+ ?\d*)")
# The digest hoists other steps' text around, so the step's OWN action is read
# from its <action> block when present and only then from the whole body.
_ACTION = re.compile(r"<action>(.*?)</action>", re.S)

# How many revisits each variant contributes. The cap exists because the full
# union puts 33.7 of 60 steps on the flow (56%), and "on flow" stops
# discriminating once it is the majority -- the same dilution risk pipeline.py
# records for the SP hybrid union. ``first2`` lands at 3.01 anchors/record,
# which matches TA's 2.6, and is the conservative variant.
VARIANTS = ("union", "first2", "first3", "only")
DEFAULT_VARIANT = "union"


def revisit_steps(traj) -> List[int]:
    """Steps whose action returns to a fixture already visited. No LLM call.

    Reads ``step.content`` only. There is no field on ``Trajectory`` that could
    carry a label in the first place (see schema.py), and the test suite
    asserts this module never names a label-side key.
    """
    seen: set = set()
    out: List[int] = []
    for s in getattr(traj, "steps", ()) or ():
        if not getattr(s, "attributable", True):
            continue
        body = (getattr(s, "content", "") or "").lower()
        m = _ACTION.search(body)
        g = _GO.search(m.group(1) if m else body)
        if not g:
            continue
        fx = g.group(1).strip()
        if fx in seen:
            out.append(int(s.index))
        else:
            seen.add(fx)
    return out


def anchors(variant: Optional[str], traj,
            located: Sequence[int]) -> Optional[Dict]:
    """Return a dict carrying the new anchor list, or None when off.

    None means "leave ``graph.anchors`` exactly as it was", which is what every
    TA/SP run gets, so those runs paint from the same roots as before and hit
    the same cache keys.

    ``located`` is what ``locate_anchors`` chose. Every variant but ``only``
    UNIONS with it rather than replacing it, for the reason pipeline.py records
    for the SP hybrid: a union can only enlarge the painted set, so no
    reference step can lose the protected band it had. ``only`` is kept as the
    ablation that answers "is the located anchor contributing anything".

    An unknown variant raises rather than silently falling back: a typo that
    quietly ran the incumbent would be recorded in the artifact as the variant
    having run, which is the one failure mode that corrupts an arm comparison
    after the fact.
    """
    if not variant:
        return None
    if variant not in VARIANTS:
        raise ValueError("unknown alf_anchors variant %r; known: %s"
                         % (variant, ", ".join(VARIANTS)))
    rv = sorted(set(revisit_steps(traj)))
    base = sorted({int(x) for x in (located or []) if int(x) >= 0})
    if variant == "first2":
        rv = rv[:2]
    elif variant == "first3":
        rv = rv[:3]
    picked = sorted(set(rv)) if variant == "only" else sorted(set(rv) | set(base))
    if not rv:
        # No ``go to`` ever repeated (18/100 records). Falling back to the
        # located anchors keeps those records on the incumbent behaviour
        # rather than painting from nothing, which would empty the flow block.
        # ``fired`` must report the RULE, not whether an anchor list came back
        # non-empty: prereg criterion 5 reads it to separate "the mechanism ran
        # and did not convert" from "the mechanism never ran", and the fallback
        # list is always non-empty.
        logger.debug("alf_anchors: no revisit on %s, keeping located anchors",
                     getattr(traj, "task_id", "?"))
        return {"anchors": list(base), "revisits": [], "variant": variant,
                "located": list(base), "fired": False}
    return {"anchors": picked, "revisits": rv, "variant": variant,
            "located": list(base), "fired": True}
