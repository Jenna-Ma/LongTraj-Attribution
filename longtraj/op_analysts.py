"""Five localisation analysts, one per OPERATION TYPE, each reading only its own.

This is the second multi-reader design in the repo and it exists because the
first one lost. ``experts.py`` runs four readers split by FAILURE FAMILY over
one shared body of evidence -- "They differ in remit, not in evidence" is its
stated design commitment -- and it scored 0.3333 on TrajAudit, the lowest of 39
arms. The recorded cause was that all four prompts read the same material and
were pulled to the same salient step: distinct answers averaged 1.6 out of 4.

This module inverts exactly that commitment. Each analyst sees ONLY the steps
of its own operation type. The edit analyst cannot be pulled by the search step
that pulled all four experts, because it never sees it. That is the whole
mechanical claim, and it is the only reason this is worth a second run.

WHAT THE MEASUREMENT SAYS BEFORE ANY CALL IS MADE. On the 102 stored TrajAudit
rows of the best arm (``ta_repro2``, acc@0 0.5000), with gold read on the
evaluation side only:

  * the answer already lands in the RIGHT operation type 80/102 = 0.784 of the
    time, and the marginal distribution is near-perfect (predicted edit 45 /
    reason 24 / search 15 / read 10 / test 8 against gold 40 / 24 / 15 / 15 /
    8). So a perfect type SELECTOR is worth at most 0.716 and this design is
    not aiming there;
  * the gain, if any, is WITHIN a type. On the 30 rows where the type is right
    and the step is wrong, the miss is a median of +4 trajectory steps but only
    **+1 same-type step**, and 9 of the 30 are off by a single same-type step.
    Partitioning turns a median 52-step search into a median 11-step one;
  * this design adds NO new evidence. 88 of 102 gold steps already arrive
    uncompressed (keep 66, light 22), so the analysts mostly re-read what the
    localiser could already see. The novelty is the partition, not the volume.

WHY THE ANALYSTS SEE COMPRESSED STEPS TOO. The brief was "the steps of this
type that were not compressed". Taken literally that blinds the reason analyst:
only 143 of 891 reasoning steps survive uncompressed (0.16), while reasoning
holds 24 gold rows at acc 0.375. So each analyst reads every step of its type,
with compressed ones marked as such and the prompt telling it to say when its
judgement rests on one. Verbatim where verbatim exists; not silence elsewhere.

An analyst may decline. ``found: false`` costs it its nomination rather than
forcing a guess out of a slice that genuinely holds nothing.

FUSION IS A MODEL, NOT A RULE -- the other inversion from ``experts.py``, whose
docstring argues a model arbitrating verdicts "is a general-purpose judge again,
which is the thing being replaced". Here the arbiter is the existing localiser,
reading the compressed trajectory, the span prior and these findings, and free
to reject all five. Its prompt is byte-identical to a run without this stage
when the stage is off.

Ground truth is not reachable from here: ``run_analysts`` is handed a
``Trajectory``, which by construction has no field that can hold a label.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import candidates as candidates_mod
from .candidates import _ACTION, _first_call, classify
from .compress import PROMPT_STEP_CLIP
from .llm import LLM, map_concurrent
from .localize import (_flow_block, _snap_to_attributable, _with_rejection)
from . import units as units_mod
from .schema import (COMPRESSED, CONTROL_FLOW, DATA_FLOW, FlowGraph, Span,
                     StepDigest, Trajectory, clip_middle)

logger = logging.getLogger(__name__)

# Output budget per analyst. Smaller than ANSWER_BUDGET (4000) because an
# analyst returns one nomination and two or three sentences, not the six-field
# answer the localiser returns; larger than it looks necessary because the
# ceiling being hit means an unparseable answer, i.e. a silent abstention.
ANALYST_BUDGET = 1500

# Per-analyst prompt cap. The whole slice of one type is far smaller than the
# trajectory -- median 11 steps -- so this is a guard against the pathological
# tail (largest observed slice: 42 steps), not the normal path.
ANALYST_PROMPT_CHARS = 60000

ANALYST_SYSTEM = (
    "You review one kind of operation inside a failed agent trajectory. "
    "You see only the steps of that kind and you report only on them. "
    "Answer strictly in JSON."
)


# ---------------------------------------------------------------------------
# Operation-type classification
# ---------------------------------------------------------------------------
# The classifier itself now lives in ``longtraj/units.py``, which is a near-leaf
# (it imports ``candidates`` and ``schema`` only). It moved there because a
# second caller appeared: ``spans.segment_units`` needs the same five types to
# build its operation chain, and ``spans`` cannot import this module -- this one
# sits above ``localize``, which would make the cycle
# ``spans -> op_analysts -> localize -> compress``.
#
# Two copies of a classifier drift, and this one is load-bearing for the SP
# tables, so there is exactly one. The names are re-exported here because tests
# and probes import them from this module.
OP_TYPES = units_mod.OP_TYPES
optype = units_mod.optype
_RX_SEARCH = units_mod._RX_SEARCH
_RX_TEST = units_mod._RX_TEST
_RX_READ = units_mod._RX_READ
_RX_RUN = units_mod._RX_RUN

OP_TITLES = {
    "edit": "MODIFICATION",
    "reason": "REASONING",
    "read": "READING",
    "search": "SEARCH",
    "test": "EXECUTION / TESTING",
}


def slice_steps(traj: Trajectory) -> Dict[str, List[int]]:
    """Attributable step indices grouped by operation type, in trajectory order.

    Non-attributable turns (user/system) are excluded here rather than in the
    prompt: they can never be the answer, so showing them to an analyst that is
    asked to nominate one would only invite an answer that gets snapped away.
    """
    out: Dict[str, List[int]] = {k: [] for k in OP_TYPES}
    for s in traj.steps:
        if not s.attributable:
            continue
        out[optype(s.content)].append(s.index)
    return out


# ---------------------------------------------------------------------------
@dataclass
class AnalystFinding:
    """One analyst's report. An empty slice yields one of these with no call."""

    op: str
    n_steps: int = 0
    found: bool = False
    step: Optional[int] = None
    confidence: float = 0.0
    reason: str = ""
    called: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "n_steps": self.n_steps, "found": self.found,
                "step": self.step, "confidence": round(self.confidence, 3),
                "reason": self.reason, "called": self.called}


ANALYST_PROMPT = """You are the {title} analyst for a failed agent trajectory.

TASK THE AGENT WAS GIVEN
{task}

HOW THE RUN FAILED
{failure_summary}

============================================================
YOUR SLICE — every {label} step of this run, and nothing else
============================================================
You are shown ONLY the {label} steps. The run also contains steps of other
kinds; they are deliberately withheld, because four other analysts are reading
those and your job is to judge yours without being pulled towards whatever is
loudest elsewhere. Step indices are the ORIGINAL ones and are not contiguous —
the gaps are the other analysts' steps.

Steps marked [compressed] were summarised by an earlier stage and lost detail.
That marking says where evidence was preserved, not where the mistake is. If
your judgement rests on a compressed step, say so in your reason.
{slice_body}
{flow_block}
============================================================
WHAT TO DECIDE
============================================================
Is one of the steps above the CRITICAL step — the earliest step whose error
causally led to the failure described at the top?

1. Exploration is not an error. The error begins when the agent COMMITS to a
   flawed direction — acts on it, or states it as the single path forward.
2. Apply the but-for test: had this step been done correctly, would the failure
   still have happened? If yes, it is not the critical step.
3. Prefer the EARLIEST step that starts the causal chain, not a later step
   where the consequence became visible.
4. Judge only your own steps. Do not speculate about what the withheld steps
   contained, and do not nominate an index that is not in the list above.

**You are expected to decline when your slice holds nothing.** Most failures
are caused by one kind of operation, so four of the five analysts on any given
run are looking at steps that are merely fine. Answering "found": false is the
correct answer in that case and costs you nothing. A forced guess from a clean
slice is worse than silence, because a later stage has to weigh it.

Return ONLY JSON:
{{
  "found": <true if one of YOUR steps is the critical step, else false>,
  "critical_step": <int from the list above, or null>,
  "confidence": <float 0..1>,
  "reason": "<2-3 sentences: what this step did and how it caused the failure;
              empty string if found is false>"
}}"""


def _render_slice(traj: Trajectory, idxs: Sequence[int],
                  dmap: Dict[int, StepDigest], budget: int) -> str:
    """The analyst's steps, verbatim where verbatim survives.

    A step is shown in its own words when compression kept it (tier keep or
    light, which is 86% of gold steps); otherwise the compressed summary is
    shown under a [compressed] marker. Falling back to the raw step for a
    compressed one would undo the compression stage inside this prompt and make
    the arm measure two changes at once.
    """
    out: List[str] = []
    for i in idxs:
        d = dmap.get(i)
        st = traj.steps[i] if 0 <= i < len(traj) else None
        if d is not None and d.tier == COMPRESSED:
            body = d.text or ""
            mark = " [compressed]"
        elif d is not None:
            body, mark = (d.text or ""), ""
        elif st is not None:
            body, mark = (st.content or ""), ""
        else:
            continue
        who = (st.name or st.role) if st is not None else "assistant"
        out.append("[step %d | %s]%s\n%s"
                   % (i, who, mark, clip_middle(body, PROMPT_STEP_CLIP)))
    body = "\n\n".join(out)
    return body if len(body) <= budget else clip_middle(body, budget)


def _slice_flow_block(graph: FlowGraph, idxs: Sequence[int]) -> str:
    """The traced edges that touch this analyst's steps, and only those.

    Reuses ``localize._flow_block`` and filters its lines by the slice, so the
    rendering stays identical to what the localiser sees rather than becoming a
    second, subtly different edge format.
    """
    keep = set(idxs)
    chunks: List[str] = []
    for kind, label in ((DATA_FLOW, "DATA"), (CONTROL_FLOW, "CONTROL")):
        lines = [ln for ln in _flow_block(graph, kind, max_edges=60).splitlines()
                 if any(("step %d" % i) in ln for i in keep)]
        if lines:
            chunks.append("%s FLOW EDGES touching your steps:\n%s"
                          % (label, "\n".join(lines[:30])))
    return ("\n" + "\n\n".join(chunks) + "\n") if chunks else ""


def _parse(op: str, obj: Any, allowed: Sequence[int],
           traj: Trajectory) -> AnalystFinding:
    """Read one analyst's JSON. Anything malformed becomes an abstention.

    An analyst losing its nomination degrades the fusion; it does not stop it.
    An index outside the analyst's own slice is dropped rather than snapped,
    because a nomination of a step the analyst was never shown is not a finding
    about that step -- it is the model ignoring its remit.
    """
    f = AnalystFinding(op=op, n_steps=len(allowed), called=True,
                       raw=(obj if isinstance(obj, dict) else {}))
    if not isinstance(obj, dict) or not obj:
        return f

    step = obj.get("critical_step")
    try:
        step = int(step)
    except (TypeError, ValueError):
        step = None
    if step is not None and step not in set(allowed):
        logger.info("%s analyst named step %d outside its slice; dropped",
                    op, step)
        step = None
    step = _snap_to_attributable(step, traj)

    found = obj.get("found")
    if not isinstance(found, bool):
        found = step is not None

    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0

    f.found = bool(found and step is not None)
    f.step = step if f.found else None
    f.confidence = conf if f.found else 0.0
    f.reason = _with_rejection(obj) if f.found else ""
    return f


def run_analysts(traj: Trajectory, spans: List[Span], graph: FlowGraph,
                 llm: LLM,
                 digests: Optional[Sequence[StepDigest]] = None,
                 max_prompt_chars: int = ANALYST_PROMPT_CHARS,
                 workers: int = 5) -> List[AnalystFinding]:
    """One call per non-empty operation type, concurrently. Empty slice: no call.

    Returns one finding per entry of ``OP_TYPES``, always in that order, so the
    record has a fixed shape whether or not a type occurred in this run.
    """
    dmap = {d.index: d for d in (digests or [])}
    sl = slice_steps(traj)
    task = clip_middle(traj.task_description, 3000)
    summary = graph.failure_summary or "(not summarised)"

    todo = [op for op in OP_TYPES if sl[op]]
    empty = {op: AnalystFinding(op=op, n_steps=0) for op in OP_TYPES
             if not sl[op]}

    def one(op: str) -> AnalystFinding:
        idxs = sl[op]
        prompt = ANALYST_PROMPT.format(
            title=OP_TITLES[op], label=OP_TITLES[op].lower(),
            task=task, failure_summary=summary,
            slice_body=_render_slice(traj, idxs, dmap, max_prompt_chars),
            flow_block=_slice_flow_block(graph, idxs))
        obj = llm.generate_obj(prompt, max_tokens=ANALYST_BUDGET,
                               system=ANALYST_SYSTEM)
        return _parse(op, obj, idxs, traj)

    got = map_concurrent(one, todo, workers=max(1, min(workers, len(todo)))) \
        if todo else []
    done = {op: (f if f is not None else AnalystFinding(op=op,
                                                       n_steps=len(sl[op]),
                                                       called=True))
            for op, f in zip(todo, got)}
    done.update(empty)
    return [done[op] for op in OP_TYPES]


ANALYST_BLOCK = """
============================================================
FIVE INDEPENDENT ANALYSES, ONE PER KIND OF OPERATION
============================================================
Five analysts read this run in parallel. Each was shown ONLY the steps of one
kind of operation — modification, reasoning, reading, search, execution — and
was asked whether the critical step is among its own. None of them saw the
others' steps, and none of them saw the outline, the flow graph or the run as a
whole. **You are the only reader with the whole picture.**

Read these as evidence from a narrow viewpoint, not as votes:

  * Two analysts naming steps near each other is NOT corroboration. They were
    looking at different material and could not have checked each other.
  * Disagreement is information, not noise. It usually means the run has more
    than one plausible fault and the choice between them needs the global view
    that only you have.
  * A "declined" analyst means its slice looked clean to a reader who saw
    nothing else. That is weak evidence, not proof of innocence.
  * **You may reject all of them.** They could not apply the but-for test
    properly, because the consequences of a step usually land in another
    analyst's slice. You can. If the trajectory says the fault is a step none
    of them named, answer with that step.

{findings}
"""


def render_findings(findings: Sequence[AnalystFinding]) -> str:
    """The arbiter's block. Returns '' when nothing was nominated at all.

    An empty return keeps the localiser's prompt byte-identical to a run
    without this stage, which is what lets a corpus where no analyst ever fires
    be compared against the baseline directly.
    """
    lines: List[str] = []
    for f in findings:
        head = "%s analyst" % OP_TITLES.get(f.op, f.op)
        if f.n_steps == 0:
            lines.append("%s — no steps of this kind in this run." % head)
        elif not f.found:
            lines.append("%s — read %d step%s, declined to nominate."
                         % (head, f.n_steps, "" if f.n_steps == 1 else "s"))
        else:
            lines.append("%s — read %d step%s, nominates STEP %d "
                         "(confidence %.2f):\n    %s"
                         % (head, f.n_steps, "" if f.n_steps == 1 else "s",
                            f.step, f.confidence, f.reason.strip()))
    if not any(f.found for f in findings):
        return ""
    return ANALYST_BLOCK.format(findings="\n".join(lines))


def stats(findings: Sequence[AnalystFinding]) -> Dict[str, Any]:
    """Diagnostics for the arm's record. Not used by any prompt.

    ``distinct`` is the number this design exists to move: ``experts.py``
    averaged 1.6 distinct answers out of 4 readers, which is the recorded cause
    of its loss. If five analysts reading five disjoint slices still converge to
    one or two steps, the partition did not work and that is the finding.
    """
    named = [f.step for f in findings if f.found]
    return {
        "n_called": sum(1 for f in findings if f.called),
        "n_found": len(named),
        "n_declined": sum(1 for f in findings if f.called and not f.found),
        "distinct": len(set(named)),
        "by_op": {f.op: f.to_dict() for f in findings},
    }
