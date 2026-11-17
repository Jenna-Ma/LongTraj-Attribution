"""Stage 3a: a pointwise judge that scores ONE phase at a time, before compression.

The span prior (stage 5c, and 3b under ``--prior-compress``) is a *listwise*
reader: one call sees every phase summary and names at most three. That shape
has two measured consequences on SWE-Bench-Pro, where the median run has 19
phases (all figures from ``probe/judge_tab.py``, zero calls, recomputed over the
stored artifacts of the ``sp_qbase`` arm):

  * its coverage is 0.0655 of the run and its recall 0.163 -- it is allowed to
    name three phases out of nineteen, so most phases are never considered at
    all, not considered and cleared;
  * ELEVEN of those nineteen phases (59%) contain no step tracing put on a flow.
    Those phases get the compression plan's ``unguessed`` band by default, which
    is the one band whose budget goes DOWN, and the reference step lands in such
    a phase in 34 of 86 rows. Those 34 rows score acc@0 **0.147**, against
    **0.442** for the rows whose phase carries flow. That gap is the largest
    position found anywhere in this analysis and it is what this stage addresses.

Two corrections to that reading, both from the same probe, because the arm this
stage ships in is not ``sp_qbase``:

  * Inside ``sp_qarm`` -- the arm being modified -- the split is **45 rows at
    0.200** against 41 at 0.415. More rows, a smaller gap (0.215): its rule
    anchors paint less, so more phases are flowless and the ones that are are
    less selected. The union restores the painting, which moves the median
    flowless count back to 9 of 19 and is the reason the two changes ship in one
    arm.
  * On TrajAudit the same split is 19 rows at 0.368 against 79 at 0.443 -- a
    0.075 gap, inside the noise floor. TA runs for reproducibility, not because
    there is a position there.

So: one call per phase, each judging its own phase in isolation and returning a
score with a confidence. Three design points, each with a reason that can be
checked rather than argued:

**Only phases with no on-flow step are judged.** A phase that carries flow
already has those steps at ACT_KEEP, and ``_degrade`` may never touch them --
that is the repo's one structural guarantee. A score could not change their fate,
so paying for it would be paying for nothing. This is also what makes the stage
affordable: under the union painting, 9.4 calls per SP row instead of 19 and 2.4
per TA row instead of 9 -- 811 and 233 calls for the two corpora.

**Pointwise, not listwise.** No judge sees another judge's verdict or another
phase's text. That is what lets all of them run concurrently, and it removes the
ordering effect a ranked reader has -- the listwise prior must spend its three
names somewhere, while a pointwise judge is free to say "nothing here" for every
phase of a run, and the 59% figure above says that is often the true answer.

**The score goes to the COMPRESSOR, not to the localiser.** ``schema.SpanPrior``
deliberately refuses a numeric score: *"a number invites the localiser to
arithmetic it cannot justify from phase summaries alone."* That constraint is
about the LOCALISER. The compressor's job already IS arithmetic -- it allocates a
90000-character budget across steps -- so a number is the right shape there. This
is the mechanical difference between this stage and the four arms that have lost
on the region-selection axis (memo-scan, prior-unranked, localize-symptom,
op-analysts); every one of those handed the LOCALISER more pointed evidence, and
the damage concentrated in ``acc@0_given_prior_miss`` (0.179 -> 0.032).
``render_verdicts`` exists for the arm that does hand it over, and that arm is
run second and judged separately, precisely because its prior is the losing one.

Abstention is free and expected: ``relevant: false``, a missing field or
unparseable JSON all land on strength 0.0, which is the band the step would have
had without this stage.

Like every other stage, the ground truth is unreachable from here: this module
sees ``Trajectory``, ``Span`` and ``FlowGraph``, none of which can hold a label.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .compress import PROMPT_STEP_CLIP
from .llm import LLM, map_concurrent
from .schema import FlowGraph, Span, Trajectory, clip_middle

logger = logging.getLogger(__name__)

# score * confidence thresholds. Fixed here and named in the pre-registration
# BEFORE the arm runs: no score exists until the run happens, so any later
# re-tuning would be selection on the reference labels. Re-tuning is allowed, as
# a new arm swept on the dev half with test read once -- not as a re-read of this
# one.
JUDGE_HI = 0.60      # >= this: keep the phase verbatim (band "judge-likely")
JUDGE_LO = 0.35      # >= this: deterministic clip  (band "judge-possible")

# Whole-phase cap for one judge prompt. Smaller than compress.PROMPT_SPAN_CLIP
# (40000) because this stage issues one call per phase rather than one per run,
# and the phases it judges are by construction the ones tracing found nothing in.
JUDGE_PROMPT_CHARS = 24000

# One verdict and two sentences. Same reasoning as op_analysts.ANALYST_BUDGET:
# hitting the ceiling arrives as unparseable JSON, i.e. as a silent abstention.
JUDGE_BUDGET = 700

JUDGE_SYSTEM = (
    "You judge whether one phase of a failed agent trajectory could contain the "
    "step that caused the failure. You see one phase and nothing else. "
    "Answer strictly in JSON."
)

JUDGE_PROMPT = """A coding agent FAILED this task. Its run was divided into
phases. You are shown ONE phase, in full. Judge that phase alone.

ORIGINAL TASK:
{task}

WHAT VISIBLY WENT WRONG (the symptom, not the cause):
{failure_summary}

THE FAILURE BECAME VISIBLE AT: {anchors}
That is where the causal chain ENDS, and it is almost never where it starts.

============================================================
THE PHASE YOU ARE JUDGING — steps {start}..{end}{label}
============================================================
{content}
============================================================

WHAT TO DECIDE
Could the step that CAUSED the failure above be one of the steps in this phase?

1. Exploration is not an error. The error begins where the agent COMMITS to a
   direction — acts on it, or states it as the only way forward.
2. Apply the but-for test: if everything in this phase had been done correctly,
   would the failure described above still have happened? If yes, this phase is
   not where the cause is.
3. A phase that merely TOUCHES the files or symbols the failure is about is
   on-topic, not guilty. The agent had to touch them to attempt the task at all.
4. You are judging this phase against the symptom, not against the other phases.
   You cannot see them and you are not ranking. "Nothing here" is the correct
   answer for most phases of most runs, and it costs you nothing.

TWO NUMBERS, AND THEY MEAN DIFFERENT THINGS
  * "score" — how likely this phase is to contain the cause. 0.0 means the phase
    is visibly fine; 1.0 means the mistake is plainly here.
  * "confidence" — how much the evidence in front of you can support that score
    at all. A phase that is short, routine, or shown mostly as tool output gives
    you little to judge; say so with a LOW confidence rather than by moving the
    score to the middle. A low score you are sure of and a middling score you are
    unsure of are different answers and the reader after you treats them
    differently.

Return ONLY JSON:
{{
  "relevant": <true if the cause could be in this phase, else false>,
  "score": <float 0..1>,
  "confidence": <float 0..1>,
  "why": "<1-2 sentences: what this phase did, and how it does or does not
           connect to the symptom>"
}}"""


@dataclass
class SpanVerdict:
    """One judge's report on one phase. A skipped phase gets one with no call."""

    span_start: int
    span_end: int
    relevant: bool = False
    score: Optional[float] = None        # None = abstained
    confidence: Optional[float] = None
    why: str = ""
    called: bool = False
    skipped: str = ""                    # why no call was made, if none was
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def strength(self) -> float:
        """``score * confidence``, and 0.0 for anything that abstained.

        The product rather than the score alone: a phase the judge rates 0.9 on
        evidence it rates 0.2 is not a phase worth spending 12000 verbatim
        characters on, and the whole point of asking for two numbers is to be
        able to tell that case from a confident 0.9.
        """
        if not self.relevant or self.score is None or self.confidence is None:
            return 0.0
        return self.score * self.confidence

    def band(self, hi: float = JUDGE_HI, lo: float = JUDGE_LO) -> str:
        """The compression band this verdict earns, or "" for none."""
        s = self.strength
        if s >= hi:
            return "judge-likely"
        if s >= lo:
            return "judge-possible"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.span_start, "end": self.span_end,
                "relevant": self.relevant,
                "score": (None if self.score is None else round(self.score, 3)),
                "confidence": (None if self.confidence is None
                               else round(self.confidence, 3)),
                "strength": round(self.strength, 4),
                "band": self.band(), "why": self.why,
                "called": self.called, "skipped": self.skipped}


def _phase_content(traj: Trajectory, span: Span) -> str:
    """The phase's steps in their own words.

    Raw content, not digests: this stage runs BEFORE compression -- that is the
    entire point of it -- so there is nothing else to read, and the judge sees
    the text whose survival it is being asked to decide.
    """
    parts: List[str] = []
    for i in span.indices:
        if not (0 <= i < len(traj.steps)):
            continue
        st = traj.steps[i]
        parts.append("[step %d | %s]\n%s"
                     % (i, st.name or st.role,
                        clip_middle(st.content or "", PROMPT_STEP_CLIP)))
    return clip_middle("\n\n".join(parts), JUDGE_PROMPT_CHARS)


def _parse(span: Span, obj: Any) -> SpanVerdict:
    """Read one judge's JSON. Anything malformed becomes an abstention.

    An abstention is not a failure: strength 0.0 puts the phase in exactly the
    band it would have had if this stage had never run, so a judge that cannot
    answer costs the run a call and changes nothing else.
    """
    v = SpanVerdict(span_start=span.start, span_end=span.end, called=True,
                    raw=(obj if isinstance(obj, dict) else {}))
    if not isinstance(obj, dict) or not obj:
        return v

    def num(key: str) -> Optional[float]:
        try:
            return max(0.0, min(1.0, float(obj.get(key))))
        except (TypeError, ValueError):
            return None

    score, conf = num("score"), num("confidence")
    rel = obj.get("relevant")
    if not isinstance(rel, bool):
        # A model that answered with numbers but no verdict has still answered;
        # a model that gave neither has not.
        rel = score is not None
    if score is None or conf is None:
        # Half an answer is an abstention. Defaulting the missing number would
        # invent the very quantity the stage exists to measure.
        return v

    v.relevant = bool(rel)
    v.score, v.confidence = score, conf
    v.why = str(obj.get("why") or "")[:400]
    return v


def judge_spans(traj: Trajectory, spans: List[Span], graph: FlowGraph, llm: LLM,
                workers: int = 8,
                min_spans: int = 4) -> Dict[int, SpanVerdict]:
    """One call per phase that carries no traced flow. Returns {span.start: verdict}.

    Phases that DO carry flow are returned too, marked ``skipped``, so the
    artifact records that they were considered and why they were not asked
    about. They cost no call: their on-flow steps are ACT_KEEP already and the
    degrade ladder is forbidden to touch them, so no score could change what
    happens to them.

    ``min_spans`` mirrors ``span_prior.predict_spans``: below it there is nothing
    to narrow and the calls would buy a judgement about most of the run.
    """
    out: Dict[int, SpanVerdict] = {}
    if len(spans or []) < min_spans:
        return out

    on_flow = graph.on_flow()
    todo: List[Span] = []
    for sp in spans:
        idxs = [i for i in sp.indices if 0 <= i < len(traj.steps)]
        if any(i in on_flow for i in idxs):
            out[sp.start] = SpanVerdict(span_start=sp.start, span_end=sp.end,
                                        skipped="carries flow")
        elif not idxs:
            out[sp.start] = SpanVerdict(span_start=sp.start, span_end=sp.end,
                                        skipped="empty")
        else:
            todo.append(sp)
    if not todo:
        return out

    task = clip_middle(traj.task_description, 2000)
    summary = graph.failure_summary or "(not summarised)"
    anchors = (", ".join("step %d" % a for a in sorted(graph.anchors))
               or "(none identified)")

    def one(sp: Span) -> SpanVerdict:
        prompt = JUDGE_PROMPT.format(
            task=task, failure_summary=summary, anchors=anchors,
            start=sp.start, end=sp.end,
            label=(" — working label: %s" % sp.label) if sp.label else "",
            content=_phase_content(traj, sp))
        obj = llm.generate_obj(prompt, max_tokens=JUDGE_BUDGET,
                               system=JUDGE_SYSTEM)
        return _parse(sp, obj)

    got = map_concurrent(one, todo, workers=max(1, min(workers, len(todo))))
    for sp, v in zip(todo, got):
        out[sp.start] = (v if v is not None
                         else SpanVerdict(span_start=sp.start, span_end=sp.end,
                                          called=True))
    return out


JUDGE_BLOCK = """
============================================================
A PER-PHASE RELEVANCE JUDGEMENT
============================================================
Each phase below was read on its own by a separate reader, which saw that phase
and the symptom and nothing else — not the other phases, not the outline, not
the flow graph. **You are the only reader with the whole picture.**

Read these as per-phase opinions, not as a ranking:

  * The scores are NOT comparable across phases. No reader saw a second phase,
    so nothing here was calibrated against anything. A 0.8 and a 0.6 do not mean
    the first phase is more likely than the second; they mean two readers,
    looking at different material, each formed a view.
  * "confidence" is about the EVIDENCE, not about the phase. A low confidence
    means the reader had little to go on, which is a statement about what was
    shown to it — often tool output — and not a statement that the phase is fine.
  * Phases carrying a traced dependency path are absent from this list. They
    were not judged, because their steps are preserved in their own words
    regardless. Absence here is not a clearance.
  * **You may disregard all of it.** These readers could not apply the but-for
    test properly: the consequences of a step usually land in another phase. You
    can.

{rows}
"""


def render_verdicts(verdicts: Dict[int, SpanVerdict],
                    spans: Optional[Sequence[Span]] = None,
                    hi: float = JUDGE_HI, lo: float = JUDGE_LO) -> str:
    """The localiser's block. '' when no phase earned a band.

    An empty return keeps the localiser's prompt byte-identical to a run without
    this stage -- which is what lets the first arm (scores to the compressor
    only) be compared against the second (scores to the localiser as well) with
    the block as the single difference.
    """
    rows: List[str] = []
    for start in sorted(verdicts):
        v = verdicts[start]
        if v.skipped or not v.relevant or not v.band(hi, lo):
            continue
        rows.append("steps %d-%d — score %.2f, confidence %.2f:\n    %s"
                    % (v.span_start, v.span_end, v.score or 0.0,
                       v.confidence or 0.0, v.why.strip() or "(no reason given)"))
    if not rows:
        return ""
    return JUDGE_BLOCK.format(rows="\n".join(rows))


def stats(verdicts: Optional[Dict[int, SpanVerdict]],
          hi: float = JUDGE_HI, lo: float = JUDGE_LO) -> Dict[str, Any]:
    """Diagnostics for the arm's record. Read by no prompt.

    ``n_abstained`` is the number to watch first. The stage's claim is that a
    pointwise reader can say "nothing here" for most phases, which the listwise
    prior structurally cannot. If almost nothing abstains, the judges are doing
    what the prior does, only once per phase and at full price.
    """
    if not verdicts:
        return {}
    vs = list(verdicts.values())
    called = [v for v in vs if v.called]
    bands: Dict[str, int] = {}
    for v in called:
        b = v.band(hi, lo) or "none"
        bands[b] = bands.get(b, 0) + 1
    strengths = [v.strength for v in called]
    return {
        "n_spans": len(vs),
        "n_called": len(called),
        "n_skipped_flow": sum(1 for v in vs if v.skipped == "carries flow"),
        "n_abstained": sum(1 for v in called if v.score is None),
        "n_irrelevant": sum(1 for v in called
                            if v.score is not None and not v.relevant),
        "bands": bands,
        "mean_strength": (round(sum(strengths) / len(strengths), 4)
                          if strengths else 0.0),
        "max_strength": (round(max(strengths), 4) if strengths else 0.0),
        "spans": {str(v.span_start): v.to_dict() for v in vs},
    }
