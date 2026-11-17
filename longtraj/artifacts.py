"""Per-trajectory artifact store.

Tracing is the expensive stage — one LLM call to anchor plus one per hop — so
its output is written to disk as soon as it exists, before span segmentation
and compression run. If a later stage crashes, or you want to re-run
compression with different budgets, the flow graph is already there.

Layout, one file per trajectory:

    <root>/<source>/<slug>.json

    {
      "artifact_version": 2,
      "task_id": ..., "source": ..., "subset": ..., "n_steps": ...,
      "stage": "flow" | "complete",
      "model": ..., "config": {...}, "created": ..., "updated": ...,
      "flow": {
        "anchors": [...],
        "failure_summary": "...",
        "failure_entities": [...],
        "edges": [{"src","dst","kind","entity","rationale"}]
      },
      "spans": [{"start","end","label","tier","summary"}],
      "outline": {"overview": "...", "spans": [...]},        <- artifact 1
      "steps": [{"index","role","name","tier","flows",        <- artifact 2
                 "anchor","raw_chars","text"}],
      "span_prior": {"guesses":[{"start","end","likelihood","why"}],
                     "chosen": ..., "claimed": ..., "disagreement": "..."},
      "candidates": {"n": ..., "steps": [...], "obeyed": ...,
                     "moved_from": ...},
      "prediction": {...},
      "compression": {...}
    }

``outline`` and ``steps`` are the two products handed to the localiser, stored
verbatim so a run can be audited — or the localiser re-run — without repeating
tracing and compression.

Artifacts never contain ground truth: they are built from ``Trajectory``,
``FlowGraph``, ``Span``, ``StepDigest``, ``Outline`` and ``Prediction``, none
of which can hold a label.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

from .schema import (COMPRESSED, CONTROL_FLOW, DATA_FLOW, KEEP, LIGHT, LIKELY,
                     POSSIBLE, STRONG, WEAK, FlowEdge, FlowGraph, Outline,
                     Prediction, Span, SpanGuess, SpanPrior, StepDigest,
                     Trajectory)

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 2
_UNSAFE = re.compile(r"[^\w.\-]+")


def slugify(task_id: str, maxlen: int = 110) -> str:
    """Filesystem-safe name, hash-suffixed when truncated to stay unique."""
    raw = str(task_id or "unknown")
    s = _UNSAFE.sub("_", raw).strip("_") or "unknown"
    if len(s) > maxlen:
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        s = s[: maxlen - 11] + "_" + h
    return s


class TraceStore:
    """Reads and writes per-trajectory flow/span/prediction artifacts."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def path(self, task_id: str, source: str = "") -> str:
        sub = os.path.join(self.root, slugify(source)) if source else self.root
        return os.path.join(sub, slugify(task_id) + ".json")

    def exists(self, task_id: str, source: str = "") -> bool:
        return os.path.exists(self.path(task_id, source))

    # -- read --------------------------------------------------------------
    def read(self, task_id: str, source: str = "") -> Optional[Dict[str, Any]]:
        p = self.path(task_id, source)
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:                                   # noqa: BLE001
            logger.warning("unreadable artifact: %s", p)
            return None

    def load_graph(self, task_id: str, source: str = "") -> Optional[FlowGraph]:
        d = self.read(task_id, source)
        return graph_from_dict(d.get("flow")) if d else None

    def load_graph_for(self, traj) -> Optional[FlowGraph]:
        """The stored flow for THIS trajectory, or None if the file holds another.

        Artifacts are named by ``task_id`` alone, and on TrajAudit that is not
        unique: the same SWE instance appears once per agent that attempted it,
        so ``swe-agent/claude45`` and ``swe-agent/gpt5`` are different runs of
        different lengths with different reference steps under one id. Four ids
        collide there, eight rows in all. Without this guard and with
        ``--reuse-flow``, whichever row runs second reads the other's anchors,
        on-flow set and digests -- graph indices that need not even exist in its
        own trajectory. SWE-Bench-Pro has no collisions, so its stored runs are
        unaffected either way.

        The mismatch is caught on read rather than fixed by renaming, so every
        stored artifact stays valid and only the colliding rows re-trace. They
        re-trace on every run, since the two still share a filename; at four
        collisions that is cheaper than invalidating the whole cache.
        """
        d = self.read(traj.task_id, traj.source)
        if not d:
            return None
        stored = d.get("subset")
        if stored not in (None, "") and stored != traj.subset:
            logger.info("%s: stored artifact belongs to subset %r, this row is "
                        "%r -- re-tracing", traj.task_id, stored, traj.subset)
            return None
        n = d.get("n_steps")
        if isinstance(n, int) and n != len(traj):
            logger.info("%s: stored artifact has %d steps, this row has %d -- "
                        "re-tracing", traj.task_id, n, len(traj))
            return None
        return graph_from_dict(d["flow"]) if d.get("flow") else None

    def load_spans(self, task_id: str, source: str = "") -> Optional[List[Span]]:
        d = self.read(task_id, source)
        if not d or not d.get("spans"):
            return None
        return spans_from_list(d["spans"])

    def load_steps(self, task_id: str,
                   source: str = "") -> Optional[List[StepDigest]]:
        """Artifact 2: the step-level compressed trajectory."""
        d = self.read(task_id, source)
        if not d or not d.get("steps"):
            return None
        return digests_from_list(d["steps"])

    def load_outline(self, task_id: str, source: str = "") -> Optional[Outline]:
        """Artifact 1: the high-level account of how the task was executed."""
        d = self.read(task_id, source)
        if not d or not d.get("outline"):
            return None
        return outline_from_dict(d["outline"])

    def load_span_prior(self, task_id: str,
                        source: str = "") -> Optional[SpanPrior]:
        """The guessed phases, if the span-prior stage ran for this record."""
        d = self.read(task_id, source)
        if not d or not d.get("span_prior"):
            return None
        return span_prior_from_dict(d["span_prior"])

    def load_candidates(self, task_id: str,
                        source: str = "") -> Optional[Dict[str, Any]]:
        """The restricted answer space, if one was applied to this record."""
        d = self.read(task_id, source)
        if not d or not d.get("candidates"):
            return None
        return candidates_from_dict(d["candidates"])

    # -- write -------------------------------------------------------------
    def _dump(self, payload: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, ensure_ascii=False)
            os.replace(tmp, path)                # atomic: no partial artifacts
        except Exception:                                   # noqa: BLE001
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _base(self, traj: Trajectory, model: str,
              config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "task_id": traj.task_id,
            "source": traj.source,
            "subset": traj.subset,
            "n_steps": len(traj),
            "raw_chars": traj.total_chars(),
            "model": model,
            "config": config or {},
        }

    def write_flow(self, traj: Trajectory, graph: FlowGraph,
                   model: str = "", config: Optional[Dict[str, Any]] = None) -> str:
        """Persist the traced flow immediately after tracing."""
        p = self.path(traj.task_id, traj.source)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        prev = self.read(traj.task_id, traj.source) or {}
        payload = self._base(traj, model, config)
        payload.update({
            "stage": "flow",
            "created": prev.get("created", now),
            "updated": now,
            "flow": graph.to_dict(),
            "flow_stats": flow_stats(graph, len(traj)),
        })
        self._dump(payload, p)
        return p

    def write_complete(self, traj: Trajectory, graph: FlowGraph,
                       spans: List[Span], pred: Prediction,
                       digests: Optional[List[StepDigest]] = None,
                       outline: Optional[Outline] = None,
                       compression: Optional[Dict[str, Any]] = None,
                       model: str = "",
                       config: Optional[Dict[str, Any]] = None) -> str:
        """Persist the full record once localisation is done."""
        p = self.path(traj.task_id, traj.source)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        prev = self.read(traj.task_id, traj.source) or {}
        payload = self._base(traj, model, config)
        payload.update({
            "stage": "complete",
            "created": prev.get("created", now),
            "updated": now,
            "flow": graph.to_dict(),
            "flow_stats": flow_stats(graph, len(traj)),
            "spans": [s.to_dict() for s in spans],
            "outline": outline.to_dict() if outline is not None else None,
            "steps": [d.to_dict() for d in (digests or [])],
            "compression": compression or {},
            # None unless the span-prior stage ran; older artifacts simply
            # lack the key, which reads back as None.
            "span_prior": pred.span_prior,
            # None unless the answer space was restricted. Kept because an
            # accuracy reached inside a restriction is a different claim from
            # one reached freely, and the artifact has to say which it was.
            "candidates": pred.candidates,
            # None unless the multi-expert localiser ran. Every verdict is here,
            # abstentions included, because the artifact has to say which reader
            # the answer came from and what the others said instead.
            "experts": pred.experts,
            # None unless the per-operation-type analysts ran. Their declines
            # are kept alongside their nominations: "the edit analyst read six
            # edits and found nothing" is the evidence that separates a clean
            # slice from an absent one, and the arbiter was shown both.
            "op_analysts": pred.op_analysts,
            # None unless prior-conditioned compression ran. Counts per band,
            # which is the only place the run records WHY a step survived --
            # the digests say that it did, never on whose authority.
            "compression_plan": pred.compression_plan,
            # None unless the pointwise phase judge ran. EVERY verdict is here,
            # including the phases it cleared and the ones it was never asked
            # about, because the stage's own claim -- that a pointwise reader
            # can say "nothing here" where the listwise prior structurally
            # cannot -- is only checkable if the clearances are on disk. It is
            # also what lets JUDGE_HI / JUDGE_LO be re-cut offline, and the
            # compression plan be recomputed under {anchors} x {judge} to tell
            # the two changes of this arm apart, without paying for a call.
            "span_judge": pred.span_judge,
            # None unless the shortlist reader ran. Its proposals are here
            # whether or not they entered the answer space -- the ones already in
            # C1, the out-of-range ones and the ones lost to the K cap included --
            # because "asked and added nothing" is the control case for this arm,
            # and because a different K can then be re-scored offline against a
            # stored run instead of by paying for the calls again.
            "shortlist": pred.shortlist,
            # None unless the anchors were picked by rule. Holds the model's own
            # picks alongside the rule's, so the two can be compared offline
            # without re-running the anchor call.
            "anchor_rule": pred.anchor_rule,
            # None unless the localiser was sampled more than once. The whole
            # tally is kept, so a different vote rule can be scored against a
            # stored run without spending a call.
            "samples": pred.samples,
            # None unless the second pass ran. Holds the FIRST answer as well as
            # the revision, because "right before the revision" and "right after
            # it" are different results and the artifact has to say which.
            "zoom": pred.zoom,
            # every group's verdict, not only the winning one: the arbitration
            # rule is the part most likely to be changed, and keeping the losers
            # is what lets it be changed without re-running the calls
            "bracket": pred.bracket,
            # each framing's own answer, including the incumbent's when the
            # merge overruled it. Kept for the same reason as ``bracket``: the
            # merge rule is the part most likely to change, and this is also
            # the only way to ask what a single framing would have scored ON
            # ITS OWN -- a question the vote itself cannot answer.
            "framings": pred.framings,
            # the answer BEFORE the predecessor audit moved it, and each of the
            # four acceptance conditions separately. The acceptance rule is the
            # part most likely to be loosened or tightened, and this is what
            # lets that be scored offline instead of re-run.
            "audit": pred.audit,
            # the whole pointwise poll, not just who won: the only way to
            # ask what another weighting of the four questions would have
            # scored, which the winner alone cannot answer.
            "verify": pred.verify,
            # what the reader actually accounted for before answering, and --
            # the field that decides how this arm is read -- what it was told
            # to account for and skipped. Without ``missed`` on disk, a losing
            # run cannot be told apart from one where the instruction was
            # simply ignored, and those two have different next moves.
            "roll": pred.roll,
            # the requirements themselves, not only the counts the localiser
            # was shown. This is the field that makes the arm re-readable: the
            # rule for turning compliance into evidence is the part most
            # likely to be wrong, and without the specs and verdicts on disk,
            # asking what a different rule would have scored means paying for
            # every spec and check call again.
            "expect": pred.expect,
            # None unless the forward commitment-ledger scan supplied the
            # region. Holds the scan's own step nominations, which this round
            # deliberately does not use as the answer.
            "memo": pred.memo,
            "prediction": {
                "critical_step": pred.critical_step,
                "confidence": pred.confidence,
                "error_type": pred.error_type,
                "reason": pred.reason,
                "usage": pred.usage,
            },
        })
        self._dump(payload, p)
        return p

    # -- iteration ---------------------------------------------------------
    def iter_artifacts(self):
        for dirpath, _dirs, files in os.walk(self.root):
            for fn in sorted(files):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                        yield json.load(fh)
                except Exception:                           # noqa: BLE001
                    continue


# ---------------------------------------------------------------------------
# (de)serialisation helpers
# ---------------------------------------------------------------------------
def graph_from_dict(d: Optional[Dict[str, Any]]) -> Optional[FlowGraph]:
    if not isinstance(d, dict):
        return None
    g = FlowGraph(
        anchors=[int(x) for x in (d.get("anchors") or [])
                 if isinstance(x, (int, float))],
        # Older artifacts predate the split and carry only "anchors"; there the
        # two are the same thing by definition.
        anchors_located=[int(x) for x in
                         (d.get("anchors_located") or d.get("anchors") or [])
                         if isinstance(x, (int, float))],
        failure_summary=str(d.get("failure_summary") or ""),
        failure_entities=[str(x) for x in (d.get("failure_entities") or [])],
    )
    for e in (d.get("edges") or []):
        if not isinstance(e, dict):
            continue
        try:
            src, dst = int(e["src"]), int(e["dst"])
        except (KeyError, TypeError, ValueError):
            continue
        kind = str(e.get("kind") or "")
        if kind not in (DATA_FLOW, CONTROL_FLOW):
            continue
        g.add(FlowEdge(src=src, dst=dst, kind=kind,
                       entity=str(e.get("entity") or ""),
                       rationale=str(e.get("rationale") or "")))
    return g


def spans_from_list(items: List[Dict[str, Any]]) -> List[Span]:
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        try:
            start, end = int(it["start"]), int(it["end"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(Span(start=start, end=end,
                        label=str(it.get("label") or ""),
                        tier=(WEAK if it.get("tier") == WEAK else STRONG),
                        summary=str(it.get("summary") or "")))
    return out


def digests_from_list(items: List[Dict[str, Any]]) -> List[StepDigest]:
    """Rebuild artifact 2. Unknown tiers fall back to COMPRESSED, the
    conservative reading: it tells the localiser the text may have lost detail."""
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        try:
            index = int(it["index"])
        except (KeyError, TypeError, ValueError):
            continue
        tier = str(it.get("tier") or "")
        out.append(StepDigest(
            index=index,
            role=str(it.get("role") or ""),
            name=str(it.get("name") or ""),
            text=str(it.get("text") or ""),
            tier=(tier if tier in (KEEP, LIGHT, COMPRESSED) else COMPRESSED),
            raw_chars=int(it.get("raw_chars") or 0),
            flows=str(it.get("flows") or ""),
            anchor=bool(it.get("anchor")),
        ))
    out.sort(key=lambda d: d.index)
    return out


def outline_from_dict(d: Optional[Dict[str, Any]]) -> Optional[Outline]:
    if not isinstance(d, dict):
        return None
    return Outline(overview=str(d.get("overview") or ""),
                   spans=spans_from_list(d.get("spans") or []))


def span_prior_from_dict(d: Optional[Dict[str, Any]]) -> Optional[SpanPrior]:
    """Rebuild the guessed phases. ``chosen``/``claimed`` are not part of the
    prior itself — they record what the localiser did with it — so they stay in
    the stored dict and are not carried onto the object."""
    if not isinstance(d, dict):
        return None
    out = []
    for it in (d.get("guesses") or []):
        if not isinstance(it, dict):
            continue
        try:
            start, end = int(it["start"]), int(it["end"])
        except (KeyError, TypeError, ValueError):
            continue
        lk = str(it.get("likelihood") or "")
        out.append(SpanGuess(start=start, end=end,
                             likelihood=(LIKELY if lk == LIKELY else POSSIBLE),
                             why=str(it.get("why") or "")))
    return SpanPrior(guesses=out)


def candidates_from_dict(
        d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Rebuild the restricted-answer-space record.

    ``obeyed`` is read strictly: anything that is not literally ``True`` reads
    back as False. A missing flag must not be mistaken for "the localiser
    answered inside the set on its own" — that is the very distinction the
    field exists to preserve.
    """
    if not isinstance(d, dict):
        return None
    steps = []
    for x in (d.get("steps") or []):
        try:
            steps.append(int(x))
        except (TypeError, ValueError):
            continue
    moved = d.get("moved_from")
    try:
        moved = int(moved) if moved is not None else None
    except (TypeError, ValueError):
        moved = None
    try:
        n = int(d.get("n"))
    except (TypeError, ValueError):
        n = len(steps)
    return {"n": n, "steps": steps, "obeyed": d.get("obeyed") is True,
            "enforced": d.get("enforced") is True,
            "moved_from": moved}


def flow_stats(graph: FlowGraph, n_steps: int) -> Dict[str, Any]:
    data = graph.data_flow_steps()
    ctrl = graph.control_flow_steps()
    on = graph.on_flow()
    return {
        "n_edges": len(graph.edges),
        "n_data_edges": sum(1 for e in graph.edges if e.kind == DATA_FLOW),
        "n_control_edges": sum(1 for e in graph.edges if e.kind == CONTROL_FLOW),
        "n_anchors": len(graph.anchors),
        "steps_on_flow": len(on),
        "steps_on_data_flow": len(data),
        "steps_on_control_flow": len(ctrl),
        "coverage": round(len(on) / n_steps, 4) if n_steps else 0.0,
        "earliest_on_flow": (min(on) if on else None),
    }


# ---------------------------------------------------------------------------
def render_flow_text(artifact: Dict[str, Any], max_edges: int = 200) -> str:
    """Human-readable dump of a saved artifact."""
    lines = []
    lines.append("=" * 78)
    lines.append("%s  [%s/%s]" % (artifact.get("task_id"), artifact.get("source"),
                                  artifact.get("subset")))
    lines.append("steps=%s raw_chars=%s stage=%s model=%s"
                 % (artifact.get("n_steps"), artifact.get("raw_chars"),
                    artifact.get("stage"), artifact.get("model")))
    st = artifact.get("flow_stats") or {}
    if st:
        lines.append("flow: %d edges (%d data / %d control), %d/%s steps on flow "
                     "(coverage %.2f), earliest=%s"
                     % (st.get("n_edges", 0), st.get("n_data_edges", 0),
                        st.get("n_control_edges", 0), st.get("steps_on_flow", 0),
                        artifact.get("n_steps"), st.get("coverage", 0.0),
                        st.get("earliest_on_flow")))
    flow = artifact.get("flow") or {}
    lines.append("")
    lines.append("anchors (failure visible here): %s" % (flow.get("anchors") or []))
    if flow.get("failure_summary"):
        lines.append("failure: %s" % flow["failure_summary"])
    if flow.get("failure_entities"):
        lines.append("entities: %s" % ", ".join(flow["failure_entities"][:20]))

    edges = flow.get("edges") or []
    for kind, title in ((DATA_FLOW, "DATA FLOW  (information built upstream)"),
                        (CONTROL_FLOW, "CONTROL FLOW  (why the agent moved on)")):
        sel = [e for e in edges if e.get("kind") == kind]
        if not sel:
            continue
        lines.append("")
        lines.append("-- %s: %d edges" % (title, len(sel)))
        for e in sorted(sel, key=lambda x: (-int(x.get("dst", 0)),
                                            int(x.get("src", 0))))[:max_edges]:
            ent = (" [%s]" % e["entity"]) if e.get("entity") else ""
            why = (" — %s" % e["rationale"]) if e.get("rationale") else ""
            lines.append("   %3s -> %3s%s%s" % (e.get("src"), e.get("dst"), ent, why))

    spans = artifact.get("spans") or []
    if spans:
        lines.append("")
        lines.append("-- SPANS: %d (%d weak / %d strong)"
                     % (len(spans),
                        sum(1 for s in spans if s.get("tier") == WEAK),
                        sum(1 for s in spans if s.get("tier") != WEAK)))
        for s in spans:
            lines.append("   %3s-%-3s %-6s %-34s summary=%d chars"
                         % (s.get("start"), s.get("end"), s.get("tier"),
                            str(s.get("label") or "")[:34],
                            len(s.get("summary") or "")))
    outline = artifact.get("outline") or {}
    if outline.get("overview"):
        lines.append("")
        lines.append("-- OUTLINE (how the task was executed)")
        for para in str(outline["overview"]).strip().splitlines():
            lines.append("   " + para)

    prior = artifact.get("span_prior") or {}
    if prior.get("guesses"):
        lines.append("")
        lines.append("-- SPAN PRIOR (a second reader's guess, from summaries only)")
        for g in prior["guesses"]:
            lines.append("   steps %s-%s  %-8s %s"
                         % (g.get("start"), g.get("end"),
                            g.get("likelihood"), str(g.get("why") or "")[:60]))
        lines.append("   localiser answered inside phase: %s (it claimed %s)"
                     % (prior.get("chosen"), prior.get("claimed")))
        if prior.get("disagreement"):
            lines.append("   disagreement: %s" % prior["disagreement"])

    cands = artifact.get("candidates") or {}
    if cands:
        lines.append("")
        lines.append("-- CANDIDATE SET (answer restricted to steps that changed "
                     "something)")
        lines.append("   %s steps: %s"
                     % (cands.get("n"), cands.get("steps") or []))
        if cands.get("obeyed"):
            lines.append("   localiser answered inside the set unaided")
        elif cands.get("enforced"):
            lines.append("   localiser answered OUTSIDE the set at step %s and "
                         "was snapped in" % cands.get("moved_from"))
        else:
            lines.append("   localiser answered OUTSIDE the set and was left "
                         "there (set offered, not enforced)")

    comp = artifact.get("compression") or {}
    if comp:
        lines.append("")
        lines.append("-- COMPRESSION: %s -> %s chars (ratio %.4f)"
                     % (comp.get("raw_chars"), comp.get("compressed_chars"),
                        comp.get("ratio", 0.0)))
        if comp.get("steps"):
            lines.append("   steps: %s total, %s on flow | kept %s / clipped %s "
                         "/ compressed %s"
                         % (comp.get("steps"), comp.get("steps_on_flow"),
                            comp.get("steps_kept"), comp.get("steps_clipped"),
                            comp.get("steps_compressed")))

    steps = artifact.get("steps") or []
    if steps:
        lines.append("")
        lines.append("-- STEPS: %d digests" % len(steps))
        for s in steps[:max_edges]:
            mark = []
            if s.get("flows"):
                mark.append("%s flow" % s["flows"])
            if s.get("anchor"):
                mark.append("anchor")
            lines.append("   %3s %-10s %-10s %6s -> %-6s %s"
                         % (s.get("index"), str(s.get("role"))[:10],
                            s.get("tier"), s.get("raw_chars"),
                            len(s.get("text") or ""), ", ".join(mark)))

    pred = artifact.get("prediction") or {}
    if pred:
        lines.append("")
        lines.append("-- PREDICTION: step %s (confidence %.2f) %s"
                     % (pred.get("critical_step"), pred.get("confidence") or 0.0,
                        pred.get("error_type") or ""))
        if pred.get("reason"):
            lines.append("   %s" % pred["reason"])
    return "\n".join(lines)
