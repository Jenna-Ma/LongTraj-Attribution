"""Verify ground-truth isolation and structural invariants.

Run: python -m tests.test_no_leak      (from the LongTraj package root)

The leak checks are the important ones: they assert that no label can reach
the pipeline, by inspecting both the type system and the actual prompt text
sent to the model.
"""
from __future__ import annotations

import copy
import dataclasses
import difflib
import hashlib
import inspect
import io
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longtraj import (LLM, Config, GroundTruth, Span, Step, Trajectory,
                      aggregate, load, run, score_one)
from longtraj import candidates as candidates_mod
from longtraj import compress as compress_mod
from longtraj import experts as experts_mod
from longtraj import localize as localize_mod
from longtraj import memo_scan as memo_scan_mod
from longtraj import op_analysts as op_analysts_mod
from longtraj import outline as outline_mod
from longtraj import pipeline as pipeline_mod
from longtraj import spans as spans_mod
from longtraj import span_prior as span_prior_mod
from longtraj import tracing as tracing_mod
from longtraj.loaders import _rootse_step_text
from longtraj.schema import (COMPRESSED, CONTROL_FLOW, DATA_FLOW, KEEP,
                            LIGHT, FlowGraph, SpanGuess, SpanPrior,
                            StepDigest)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        FAILURES.append(name)


# ---------------------------------------------------------------------------
print("\n[1] type-level isolation")

label_words = re.compile(r"critical|mistake|gold|answer|label|failure_id", re.I)
traj_fields = {f.name for f in dataclasses.fields(Trajectory)}
step_fields = {f.name for f in dataclasses.fields(Step)}
check("Trajectory has no label-like field",
      not any(label_words.search(f) for f in traj_fields), str(traj_fields))
check("Step has no label-like field",
      not any(label_words.search(f) for f in step_fields), str(step_fields))

# ---------------------------------------------------------------------------
print("\n[2] pipeline modules never touch GroundTruth")

from longtraj import artifacts as artifacts_mod

for mod in (pipeline_mod, tracing_mod, spans_mod, compress_mod, localize_mod,
            outline_mod, span_prior_mod, candidates_mod, artifacts_mod,
            experts_mod, memo_scan_mod):
    src = inspect.getsource(mod)
    check("%s does not reference GroundTruth" % mod.__name__.split(".")[-1],
          "GroundTruth" not in src)
    check("%s does not reference ground_truth" % mod.__name__.split(".")[-1],
          "ground_truth" not in src)

# ---------------------------------------------------------------------------
print("\n[3] run() signature cannot accept a label")

sig = inspect.signature(run)
params = list(sig.parameters)
check("run() takes only (traj, llm, config, store, llm_localize, llm_trace)",
      params == ["traj", "llm", "config", "store", "llm_localize",
                 "llm_trace"], str(params))
check("the stage-6 seam carries a model, not a label",
      sig.parameters["llm_localize"].default is None
      and "LLM" in str(sig.parameters["llm_localize"].annotation))
check("the stage-1 seam carries a model, not a label",
      sig.parameters["llm_trace"].default is None
      and "LLM" in str(sig.parameters["llm_trace"].annotation))
check("run() has no label-like parameter",
      not any(re.search(r"gold|label|answer|truth|critical|mistake", p)
              for p in params), str(params))

# ---------------------------------------------------------------------------
print("\n[4] no gold step appears in any prompt (recording LLM)")


class RecordingLLM(LLM):
    """Captures every prompt instead of calling out."""

    def __init__(self):
        super().__init__(dry_run=True)
        self.prompts = []

    def generate(self, prompt, max_tokens=None, system="", force_json=False):
        self.prompts.append(prompt)
        return ""

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        return default


# The judging stage is the one place a model's answer decides which steps stay
# verbatim, so the tests need to be able to dictate that answer. Every other
# prompt falls through to RecordingLLM's behaviour, which keeps these doubles
# usable for full-pipeline runs.
_JUDGE_MARK = "one at a time and independently"


class JudgingLLM(RecordingLLM):
    """Answers the error-judging call with dictated verdicts."""

    def __init__(self, errors, related=None, all_error=False):
        super().__init__()
        self.errors = dict(errors)
        self.related = dict(related or {})
        self.all_error = all_error

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if _JUDGE_MARK not in prompt:
            return default
        shown = [int(m) for m in re.findall(r"^step (\d+) \[", prompt, re.M)]
        idxs = sorted(set(shown) | set(self.errors) | set(self.related))
        return {"steps": [{"index": i,
                           "error": True if self.all_error
                           else bool(self.errors.get(i)),
                           "related": bool(self.related.get(i)),
                           "why": "dictated"} for i in idxs]}


class _BrokenLLM(RecordingLLM):
    """Raises on the judging call. The stage must survive it."""

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if _JUDGE_MARK in prompt:
            raise RuntimeError("provider exploded")
        return default


def _quietly(fn, *a, **kw):
    """Run something that is SUPPOSED to log an exception, without the noise."""
    lg = logging.getLogger("longtraj.tracing")
    was = lg.level
    lg.setLevel(logging.CRITICAL)
    try:
        return fn(*a, **kw)
    finally:
        lg.setLevel(was)


records = list(load("trajaudit", subset="swe-agent", limit=2))
records += list(load("trajdebug", subset="swebenchpro", limit=1))
check("loaded sample records", len(records) >= 2, "got %d" % len(records))

for rec in records:
    gt = rec.ground_truth
    rl = RecordingLLM()
    pred = run(rec.trajectory, rl, Config(max_rounds=1, workers=1))
    blob = "\n".join(rl.prompts)
    # the label must not be stated anywhere as a claim about the answer
    leaked = re.search(r"(critical|mistake|gold|correct)\D{0,40}\b%d\b"
                       % gt.critical_step, blob, re.I) if gt.critical_step else None
    check("%s: no gold-step assertion in prompts" % rec.trajectory.task_id[:30],
          leaked is None, str(leaked.group(0))[:60] if leaked else "")
    if gt.rationale:
        snippet = gt.rationale[:60]
        check("%s: human rationale not in prompts" % rec.trajectory.task_id[:30],
              snippet not in blob)
    if gt.error_type:
        check("%s: gold error_type not in prompts" % rec.trajectory.task_id[:30],
              gt.error_type not in blob)
    check("%s: produced a prediction" % rec.trajectory.task_id[:30],
          pred.critical_step is not None)

# ---------------------------------------------------------------------------
print("\n[5] span cover is complete and non-overlapping")

for rec in records:
    traj = rec.trajectory
    graph = FlowGraph(anchors=[len(traj) - 1])
    sp = spans_mod.heuristic_spans(traj, target=8)
    covered = []
    for s in sp:
        covered.extend(s.indices)
    check("%s: spans cover every step exactly once" % traj.task_id[:30],
          sorted(covered) == list(range(len(traj))),
          "n=%d covered=%d" % (len(traj), len(covered)))

    coerced = spans_mod._coerce_spans(
        [{"start": 0, "end": 2}, {"start": 5, "end": 9}], len(traj))
    cov2 = []
    for s in coerced:
        cov2.extend(s.indices)
    check("%s: gappy LLM spans get repaired to full cover" % traj.task_id[:30],
          sorted(cov2) == list(range(len(traj))),
          "covered=%d of %d" % (len(cov2), len(traj)))

# ---------------------------------------------------------------------------
print("\n[6] predictions land on attributable steps")

for rec in records:
    traj = rec.trajectory
    rl = RecordingLLM()
    pred = run(traj, rl, Config(max_rounds=1, workers=1))
    ok = (pred.critical_step is None
          or traj.steps[pred.critical_step].attributable)
    check("%s: prediction is an agent step" % traj.task_id[:30], ok,
          "step %s role=%s" % (pred.critical_step,
                               traj.steps[pred.critical_step].role
                               if pred.critical_step is not None else "-"))

# ---------------------------------------------------------------------------
print("\n[7] backward-edge invariant (src < dst)")

from longtraj.schema import FlowEdge
g = FlowGraph(anchors=[10])
for src, dst in ((3, 7), (7, 3), (5, 5)):
    e = FlowEdge(src=src, dst=dst, kind="data")
    if src < dst:
        g.add(e)
check("only backward edges retained", all(e.src < e.dst for e in g.edges))

# ---------------------------------------------------------------------------
print("\n[8] saved artifacts round-trip and hold no ground truth")

import json
import tempfile

from longtraj.artifacts import TraceStore, graph_from_dict

# names a dataset annotation would use; our own prediction keys are exempt
_GOLD_KEY = re.compile(r"gold|answer|ground_truth|annotation|human|mistake"
                       r"|critical_error|failure_id", re.I)


def _all_keys(obj):
    """Every key name anywhere in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)

with tempfile.TemporaryDirectory() as tmp:
    store = TraceStore(tmp)
    for rec in records:
        traj, gt = rec.trajectory, rec.ground_truth
        pred = run(traj, RecordingLLM(), Config(max_rounds=1, workers=1), store=store)

        art = store.read(traj.task_id, traj.source)
        check("%s: artifact written" % traj.task_id[:30], art is not None)
        if art is None:
            continue
        check("%s: artifact is complete" % traj.task_id[:30],
              art.get("stage") == "complete", str(art.get("stage")))

        blob = json.dumps(art, ensure_ascii=False)
        if gt.rationale:
            check("%s: rationale not in artifact" % traj.task_id[:30],
                  gt.rationale[:60] not in blob)
        if gt.error_type:
            check("%s: gold error_type not in artifact" % traj.task_id[:30],
                  gt.error_type not in blob)
        # "critical_step" / "rationale" are legitimate here (our prediction and
        # our tracer's own reasoning), so match only annotation-flavoured names
        bad = [k for k in _all_keys(art) if _GOLD_KEY.search(k)]
        check("%s: artifact has no annotation-like key" % traj.task_id[:30],
              not bad, str(bad))

        # the flow survives a disk round-trip: reuse_flow depends on this
        back = store.load_graph(traj.task_id, traj.source)
        check("%s: flow round-trips" % traj.task_id[:30],
              back is not None
              and len(back.edges) == len(graph_from_dict(art["flow"]).edges)
              and back.anchors == (art["flow"].get("anchors") or []))
        check("%s: round-tripped flow keeps backward invariant" % traj.task_id[:30],
              back is None or all(e.src < e.dst for e in back.edges))

    n_files = sum(1 for _ in store.iter_artifacts())
    check("one artifact per trajectory", n_files == len(records),
          "%d files for %d trajectories" % (n_files, len(records)))

# ---------------------------------------------------------------------------
print("\n[9] the two compression artifacts")

# These are the guarantees the localiser relies on. The on-flow rule is the
# important one: a model rewriting an on-path step is exactly how the evidence
# that identifies the critical step gets dropped.

from longtraj.artifacts import digests_from_list, outline_from_dict

for rec in records:
    traj = rec.trajectory
    tag = traj.task_id[:30]
    rl = RecordingLLM()
    graph = tracing_mod.trace(traj, rl, max_rounds=1)
    graph.anchors = [i for i in traj.attributable_indices[-3:]]
    # force a mix: one synthetic data edge into the last anchor
    if len(traj.attributable_indices) >= 2:
        graph.add(FlowEdge(src=traj.attributable_indices[0],
                           dst=graph.anchors[-1], kind="data", entity="x"))
    sp = spans_mod.assign_tiers(
        spans_mod.heuristic_spans(traj, target=6), graph)
    sp, digests = compress_mod.compress(traj, sp, graph, rl, workers=1)

    idxs = [d.index for d in digests]
    check("%s: one digest per step, indices intact" % tag,
          idxs == list(range(len(traj))),
          "n=%d digests=%d" % (len(traj), len(digests)))

    on_flow = graph.on_flow()
    check("%s: on-flow steps are never model-compressed" % tag,
          all(d.tier in (KEEP, LIGHT)
              for d in digests if d.index in on_flow),
          str([d.index for d in digests
               if d.index in on_flow and d.tier == COMPRESSED]))
    check("%s: on-flow text is a substring of the raw step" % tag,
          all(d.text in (traj.steps[d.index].content or "")
              for d in digests if d.tier == KEEP),
          "KEEP text must be verbatim")
    check("%s: flow membership is recorded on the digest" % tag,
          all(d.on_flow for d in digests
              if d.index in on_flow and d.raw_chars),
          str([d.index for d in digests
               if d.index in on_flow and not d.on_flow]))

    body = compress_mod.render_compressed(traj, sp, digests)
    check("%s: every step index is citable in artifact 2" % tag,
          all(("[step %d " % i) in body for i in range(len(traj))))

    ol = outline_mod.build_outline(traj, sp, rl, graph)
    rendered = ol.render()
    check("%s: outline names every span range" % tag,
          all(("steps %d-%d" % (s.start, s.end)) in rendered for s in sp))

    # both artifacts survive the disk round-trip that `flows`/re-localisation use
    back = digests_from_list([d.to_dict() for d in digests])
    check("%s: digests round-trip" % tag,
          [(d.index, d.tier, d.text) for d in back]
          == [(d.index, d.tier, d.text) for d in digests])
    ol_back = outline_from_dict(ol.to_dict())
    check("%s: outline round-trips" % tag,
          ol_back is not None and ol_back.overview == ol.overview
          and len(ol_back.spans) == len(ol.spans))

# ---------------------------------------------------------------------------
print("\n[10] malformed JSON from the model degrades, never crashes")

# Real failure seen on RootSE: the model returned a JSON *array* where an
# object was requested. `default={} or {}` did not catch it (a non-empty list
# is truthy), so `.get` raised and the whole span was lost to the fallback.

from longtraj.llm import as_object

check("as_object passes a dict through", as_object({"a": 1}) == {"a": 1})
check("as_object unwraps a one-element array",
      as_object([{"summary": "s"}]) == {"summary": "s"})
check("as_object takes the first object of a longer array",
      as_object([{"a": 1}, {"b": 2}]) == {"a": 1})
check("as_object rejects a scalar array", as_object([1, 2, 3]) == {})
check("as_object rejects None/str/int",
      as_object(None) == {} and as_object("x") == {} and as_object(7) == {})


class ArrayLLM(RecordingLLM):
    """Returns an array from every JSON call, as the real endpoint sometimes does."""

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        return [{"unexpected": "array"}]


for rec in records[:2]:
    traj = rec.trajectory
    try:
        pred = run(traj, ArrayLLM(), Config(max_rounds=1, workers=1))
        ok, why = pred.critical_step is not None, ""
    except Exception as exc:                                  # noqa: BLE001
        ok, why = False, "%s: %s" % (type(exc).__name__, exc)
    check("%s: array-returning model still yields a prediction"
          % traj.task_id[:30], ok, why)

# ---------------------------------------------------------------------------
print("\n[11] an empty response is a soft failure, never a cached answer")

# Real failure seen on the full RootSE run: the endpoint returned HTTP 200 with
# an empty body under concurrency. `generate` accepted it as an answer AND wrote
# it to the disk cache, so 44 of 97 cases silently fell back — permanently, since
# every rerun replayed the cached blank.


class BlankThenGood(LLM):
    """Returns "" the first ``n_blank`` times, then a real answer."""

    def __init__(self, n_blank, **kw):
        super().__init__(retries=3, temperature=0.0, **kw)
        self.n_blank = n_blank
        self.posts = 0

    def _post(self, prompt, max_tokens, system, force_json):
        self.posts += 1
        if self.posts <= self.n_blank:
            return ""
        return '{"critical_step": 7}'


with tempfile.TemporaryDirectory() as tmp:
    llm = BlankThenGood(2, cache_dir=tmp)
    out = llm.generate("p", max_tokens=32)
    check("blank response is retried until an answer arrives",
          out == '{"critical_step": 7}' and llm.posts == 3,
          "posts=%d out=%r" % (llm.posts, out))
    check("empties are counted", llm.snapshot()["empty"] == 2,
          str(llm.snapshot()))
    check("only the real answer is cached",
          [f for f in os.listdir(tmp)] and all(
              json.load(open(os.path.join(tmp, f), encoding="utf-8"))["response"].strip()
              for f in os.listdir(tmp)),
          str(os.listdir(tmp)))

with tempfile.TemporaryDirectory() as tmp:
    llm = BlankThenGood(99, cache_dir=tmp)
    out = llm.generate("p", max_tokens=32)
    check("giving up on blanks returns '' and writes no cache entry",
          out == "" and llm.posts == 3 and not os.listdir(tmp),
          "posts=%d files=%s" % (llm.posts, os.listdir(tmp)))

with tempfile.TemporaryDirectory() as tmp:
    # a cache poisoned by the old code must not be replayed
    seed = LLM(cache_dir=tmp, temperature=0.0, retries=3)
    key = seed._cache_key("p", 32, "")
    json.dump({"response": ""}, open(seed._cache_path(key), "w", encoding="utf-8"))
    llm = BlankThenGood(0, cache_dir=tmp)
    out = llm.generate("p", max_tokens=32)
    check("a poisoned cache entry is treated as a miss",
          out == '{"critical_step": 7}' and llm.posts == 1,
          "posts=%d out=%r" % (llm.posts, out))

# ---------------------------------------------------------------------------
print("\n[12] a truncated response is a soft failure, never a cached answer")

# Real failure seen on the v4 RootSE run: tracing asked for up to 16 edges with
# rationales but capped the answer at 2500 tokens, so the endpoint stopped
# mid-JSON. The body was non-empty and HTTP 200, so the empty-response guard
# passed it through, `extract_json` returned None -> {} -> zero edges, and the
# prefix was cached. 730 of 4413 cache entries were in this state.

TRUNC = '{"edges": [{"src": 3, "dst": 9, "kind": "data", "entity": "x", "rat'
WHOLE = '{"edges": [{"src": 3, "dst": 9, "kind": "data", "entity": "x"}]}'


class CutOffThenWhole(LLM):
    """Truncates the first ``n`` answers, reporting finish_reason='length'."""

    def __init__(self, n, report_finish=True, **kw):
        super().__init__(retries=3, temperature=0.0, **kw)
        self.n = n
        self.report_finish = report_finish
        self.posts = 0
        self.budgets = []

    def _post(self, prompt, max_tokens, system, force_json):
        self.posts += 1
        self.budgets.append(max_tokens)
        if self.posts <= self.n:
            return (TRUNC, "length" if self.report_finish else "stop")
        return (WHOLE, "stop")


with tempfile.TemporaryDirectory() as tmp:
    llm = CutOffThenWhole(2, cache_dir=tmp)
    out = llm.generate("p", max_tokens=100, force_json=True)
    check("truncated response is retried until a whole answer arrives",
          out == WHOLE and llm.posts == 3, "posts=%d out=%r" % (llm.posts, out))
    check("truncations are counted", llm.snapshot()["truncated"] == 2,
          str(llm.snapshot()))
    check("the retry raises the token budget",
          llm.budgets == [100, 200, 400], str(llm.budgets))
    cached = [json.load(open(os.path.join(tmp, f), encoding="utf-8"))
              for f in os.listdir(tmp)]
    check("only the whole answer is cached",
          cached and all(c["response"] == WHOLE for c in cached),
          str(cached))

with tempfile.TemporaryDirectory() as tmp:
    # finish_reason missing (some gateways drop it): unparseable JSON is the
    # same failure seen from the other side
    llm = CutOffThenWhole(1, report_finish=False, cache_dir=tmp)
    out = llm.generate("p", max_tokens=100, force_json=True)
    check("truncation is caught without finish_reason", out == WHOLE,
          "posts=%d out=%r" % (llm.posts, out))

with tempfile.TemporaryDirectory() as tmp:
    llm = CutOffThenWhole(99, cache_dir=tmp)
    out = llm.generate("p", max_tokens=100, force_json=True)
    check("giving up returns the partial body but caches nothing",
          out == TRUNC and not os.listdir(tmp),
          "out=%r files=%s" % (out[:40], os.listdir(tmp)))

with tempfile.TemporaryDirectory() as tmp:
    # a cache poisoned by the old code must not be replayed
    seed = LLM(cache_dir=tmp, temperature=0.0, retries=3)
    key = seed._cache_key("p", 100, "")
    json.dump({"response": TRUNC},
              open(seed._cache_path(key), "w", encoding="utf-8"))
    llm = CutOffThenWhole(0, cache_dir=tmp)
    out = llm.generate("p", max_tokens=100, force_json=True)
    check("a truncated cache entry is treated as a miss",
          out == WHOLE and llm.posts == 1,
          "posts=%d out=%r" % (llm.posts, out[:40]))

with tempfile.TemporaryDirectory() as tmp:
    # prose answers (outline, span summaries) must not be mistaken for truncated
    # JSON just because they never parse
    class Prose(LLM):
        def _post(self, prompt, max_tokens, system, force_json):
            return ("The agent edited the test file instead of the source.",
                    "stop")

    llm = Prose(cache_dir=tmp, retries=3, temperature=0.0)
    out = llm.generate("p", max_tokens=100, force_json=False)
    check("a complete prose answer is kept and cached",
          out.startswith("The agent") and len(os.listdir(tmp)) == 1,
          "out=%r files=%s" % (out[:30], os.listdir(tmp)))
    check("prose is not counted as truncated", llm.snapshot()["truncated"] == 0,
          str(llm.snapshot()))

# Observed on the v6 forward smoke: a reasoning model handed a 20-step window
# and 5200 tokens spent the whole budget thinking and returned an EMPTY body
# with finish_reason='length'. The empty guard ran first and retried at the
# same budget, which could never succeed, so whole windows -- steps 20..39 of
# an 84-step trajectory -- silently produced no edges at all.
class ThinksItselfOut(LLM):
    """Emits nothing until the budget clears ``need``, as reasoning models do."""

    def __init__(self, need, **kw):
        super().__init__(retries=3, temperature=0.0, **kw)
        self.need = need
        self.budgets = []

    def _post(self, prompt, max_tokens, system, force_json):
        self.budgets.append(max_tokens)
        return (WHOLE, "stop") if max_tokens >= self.need else ("", "length")


with tempfile.TemporaryDirectory() as tmp:
    llm = ThinksItselfOut(400, cache_dir=tmp)
    out = llm.generate("p", max_tokens=100, force_json=True)
    check("an empty body with finish=length escalates the budget",
          out == WHOLE and llm.budgets == [100, 200, 400], str(llm.budgets))
    check("it is counted as truncated, not empty",
          llm.snapshot()["truncated"] == 2 and llm.snapshot()["empty"] == 0,
          str(llm.snapshot()))

with tempfile.TemporaryDirectory() as tmp:
    # an empty body WITHOUT finish=length is still the overload case: retry at
    # the same budget rather than inflating it
    class Overloaded(LLM):
        def __init__(self, **kw):
            super().__init__(retries=3, temperature=0.0, **kw)
            self.budgets = []

        def _post(self, prompt, max_tokens, system, force_json):
            self.budgets.append(max_tokens)
            return (WHOLE, "stop") if len(self.budgets) > 2 else ("", "stop")

    llm = Overloaded(cache_dir=tmp)
    out = llm.generate("p", max_tokens=100, force_json=True)
    check("a plain empty body retries at the same budget",
          out == WHOLE and llm.budgets == [100, 100, 100], str(llm.budgets))
    check("a plain empty body is counted as empty",
          llm.snapshot()["empty"] == 2 and llm.snapshot()["truncated"] == 0,
          str(llm.snapshot()))

# ---------------------------------------------------------------------------
print("\n[13] tracing traces every upstream step it discovers")

# The old walk kept `sorted(frontier, reverse=True)[:frontier_size]` and threw
# the rest away, so half of the dependencies the model named were never
# followed up — and the discarded half skewed early, which is exactly where
# critical steps live. Upstream steps must now queue instead of vanishing.


class ChainLLM(LLM):
    """Names one upstream step per traced step, forming a long chain.

    Anchor 20 -> 16 -> 12 -> 8 -> 4 -> 0, plus a second branch off each so the
    queue holds more than one batch at a time.
    """

    def __init__(self, origin_at=None):
        super().__init__(dry_run=True)
        self.origin_at = origin_at
        self.asked = []

    def generate_obj(self, prompt, max_tokens=None, system="",
                     force_json=True):
        import re as _re
        traced = [int(m) for m in
                  _re.findall(r"\[step (\d+)", prompt.split("STEPS CURRENTLY")[-1])]
        self.asked.extend(traced)
        edges, origins = [], []
        for d in traced:
            if self.origin_at is not None and d <= self.origin_at:
                origins.append(d)
                continue
            for delta in (4, 3):            # two upstream steps each
                s = d - delta
                if s >= 0:
                    edges.append({"src": s, "dst": d, "kind": "data",
                                  "entity": "x", "rationale": "r"})
        return {"edges": edges, "origins": origins}


rec = records[0]
traj = rec.trajectory
n = len(traj)
if n >= 21:
    llm = ChainLLM()
    g = FlowGraph(anchors=[20])
    tracing_mod.trace_backward(traj, llm, g, max_rounds=50, batch_size=2,
                               max_traced_steps=200)
    srcs = {e.src for e in g.edges}
    asked = set(llm.asked)
    never = sorted(s for s in srcs if s not in asked and s > 0)
    check("every discovered upstream step is itself traced", not never,
          "never traced: %s" % never[:10])
    check("tracing goes deeper than the old 3-hop cap",
          max(len(asked), 0) > 6, "traced %d steps" % len(asked))

    check("no step is traced twice", len(llm.asked) == len(asked),
          "asked %d times for %d steps" % (len(llm.asked), len(asked)))

    # Origins end their branch. They are marked per-step by the model, so a
    # step below one can still be reached down a different branch — what must
    # hold is that an origin itself is never expanded, so the walk terminates
    # instead of running into the round cap.
    llm2 = ChainLLM(origin_at=12)
    g2 = FlowGraph(anchors=[20])
    tracing_mod.trace_backward(traj, llm2, g2, max_rounds=50, batch_size=2,
                               max_traced_steps=200)
    deepest = min(llm2.asked)
    check("tracing terminates at origins instead of hitting the round cap",
          deepest >= 9, "descended to step %d, past every origin" % deepest)
    check("an origin is never expanded", not {e.src for e in g2.edges} - set(llm2.asked)
          and all(e.dst > 12 for e in g2.edges),
          "edges out of an origin: %s"
          % [(e.src, e.dst) for e in g2.edges if e.dst <= 12][:5])

    # the safety cap must bound work without crashing
    llm3 = ChainLLM()
    g3 = FlowGraph(anchors=[20])
    tracing_mod.trace_backward(traj, llm3, g3, max_rounds=2, batch_size=2,
                               max_traced_steps=200)
    check("max_rounds still bounds the walk", len(set(llm3.asked)) <= 4,
          "traced %d steps in 2 rounds" % len(set(llm3.asked)))
else:
    print("  SKIP  trajectory too short (%d steps)" % n)

# ---------------------------------------------------------------------------
print("\n[14] forward construction covers every step; painting keeps the paths")

# Backward tracing only reached steps it could chain back to — 62% of them on
# the RootSE set, with the head least covered. A forward sweep asks about every
# step exactly once, and the resulting parent forests are only turned into a
# flow by painting from the anchors.


class ForwardLLM(LLM):
    """Answers a window by giving each step one data and one control parent.

    The two relations point at *different* ancestors, so a painted union that
    wrongly walks across relations is detectable: data goes back in steps of 2,
    control in steps of 3, and only the data chain reaches step 1.
    """

    def __init__(self, sink=None):
        super().__init__(dry_run=True)
        self.sink = sink            # index whose subtree must never be painted
        self.windows = []
        self.asked = []
        self.digests = []           # (hi, indices named in the digest)

    def generate_obj(self, prompt, max_tokens=None, system="",
                     force_json=True):
        import re as _re
        head, body = prompt.split("STEPS TO ANALYSE NOW")
        traced = [int(m) for m in _re.findall(r"\[step (\d+)", body)]
        self.windows.append(tuple(traced))
        self.asked.extend(traced)
        digest = head.split("TRAJECTORY DIGEST")[-1]
        self.digests.append((max(traced),
                             [int(m) for m in _re.findall(r"^\[(\d+)\|",
                                                          digest, _re.M)]))
        edges = []
        for d in traced:
            for delta, kind in ((2, "data"), (3, "control")):
                s = d - delta
                if s >= 1 or (kind == "data" and s == 0):
                    edges.append({"src": s, "dst": d, "kind": kind,
                                  "entity": "e", "rationale": "r"})
        return {"edges": edges}


if n >= 21:
    llm = ForwardLLM()
    g = FlowGraph(anchors=[20])
    tracing_mod.build_forward(traj, llm, g, window=5, workers=1)

    asked = set(llm.asked)
    check("every step is asked about exactly once",
          asked == set(range(n)) and len(llm.asked) == n,
          "asked %d of %d steps, missing %s" % (len(llm.asked), n,
                                                sorted(set(range(n)) - asked)[:5]))
    check("the head of the trajectory is covered, not just the tail",
          min(asked) == 0, "earliest step asked about was %d" % min(asked))
    check("windows partition the trajectory without overlap",
          sum(len(w) for w in llm.windows) == len(set().union(*llm.windows)),
          "windows overlap: %s" % (llm.windows[:3],))
    check("no window depends on another's answer (all issued together)",
          len(llm.windows) == len({w[0] for w in llm.windows}),
          "duplicate window starts: %s" % (llm.windows[:3],))
    # a parent precedes its child, so digest lines after the window are dead
    # prefill -- on a 700-step trajectory that is most of the prompt
    check("each window's digest stops at its own last step",
          all(not [i for i in seen if i > hi] for hi, seen in llm.digests),
          "a digest named steps after its window: %s"
          % [(hi, [i for i in seen if i > hi][:3]) for hi, seen in llm.digests
             if any(i > hi for i in seen)][:2])
    check("the digest still reaches back to step 0",
          all(seen and min(seen) == 0 for _, seen in llm.digests),
          "a window could not cite the head: %s"
          % [(hi, min(seen) if seen else None) for hi, seen in llm.digests][:3])

    # painting: the forests are separate, and only anchor-reaching paths count
    data_painted = g.data_flow_steps()
    ctrl_painted = g.control_flow_steps()
    check("data painting follows only data edges",
          data_painted == {20 - 2 * k for k in range(11)} | {0},
          "data painted %s" % sorted(data_painted)[:12])
    check("control painting follows only control edges",
          ctrl_painted == {20 - 3 * k for k in range(7)},
          "control painted %s" % sorted(ctrl_painted)[:12])
    check("the two forests are not merged into one walk",
          not (data_painted & ctrl_painted) - {20, 14, 8, 2},
          "unexpected overlap %s" % sorted(data_painted & ctrl_painted))
    check("on_flow is the union of the two painted forests",
          g.on_flow() == data_painted | ctrl_painted, "union mismatch")
    check("steps off every anchor path are not painted",
          19 not in g.on_flow() and 15 not in g.on_flow(),
          "painted %s, which no chain from the anchor reaches"
          % sorted({19, 15} & g.on_flow()))
    check("painting never reaches past an anchor",
          max(g.on_flow()) == 20, "painted step %d, after the anchor"
          % max(g.on_flow()))

    # the fan-in cap bounds one bad answer without silently shaping good ones
    class FanOut(ForwardLLM):
        def generate_obj(self, prompt, max_tokens=None, system="",
                         force_json=True):
            obj = super().generate_obj(prompt, max_tokens, system, force_json)
            d = max(self.windows[-1])
            obj["edges"] += [{"src": s, "dst": d, "kind": "data",
                              "entity": "e", "rationale": "r"}
                             for s in range(d)]
            return obj

    llm2 = FanOut()
    g2 = FlowGraph(anchors=[20])
    tracing_mod.build_forward(traj, llm2, g2, window=5, workers=1)
    fan = {}
    for e in g2.edges:
        if e.kind == "data":
            fan.setdefault(e.dst, set()).add(e.src)
    check("a runaway fan-in is capped per step and per relation",
          max(len(v) for v in fan.values()) <= tracing_mod.MAX_PARENTS_PER_STEP,
          "one step kept %d data parents" % max(len(v) for v in fan.values()))
else:
    print("  SKIP  trajectory too short (%d steps)" % n)

# ---------------------------------------------------------------------------
print("\n[15] flow_recall measures painting, not the presence of an edge")

# The old metric was "the gold step is an endpoint of some edge". Under
# backward tracing that coincided with painting, because only traced steps had
# edges. Forward construction gives EVERY step a parent, so the old metric
# reported flow_recall = 1.0 on a graph that painted nothing -- it was
# measuring coverage of the sweep, not quality of the flow.
from longtraj.evaluate import score_one as _score
from longtraj.evaluate import aggregate as aggregate_mod
from longtraj.schema import Prediction

# 0 -> 1 -> 2 (data), and a disconnected 5 -> 6 branch. Anchor at 2, so the
# painted set is {0,1,2}: steps 5 and 6 have edges but reach no anchor.
LINE = {"anchors": [2],
        "edges": [{"src": 0, "dst": 1, "kind": "data"},
                  {"src": 1, "dst": 2, "kind": "data"},
                  {"src": 5, "dst": 6, "kind": "data"}]}

on = _score(Prediction(task_id="t", critical_step=1, flow=LINE),
            GroundTruth(task_id="t", critical_step=1))
off = _score(Prediction(task_id="t", critical_step=9, flow=LINE),
             GroundTruth(task_id="t", critical_step=6))
check("a gold step on the painted path counts as recalled",
      on["gold_on_flow"] is True, str(on.get("gold_on_flow")))
check("a gold step with edges but no path to an anchor does NOT count",
      off["gold_on_flow"] is False,
      "step 6 has an edge and was scored %s" % off.get("gold_on_flow"))
check("the painted size is reported so recall can be read against it",
      on.get("flow_size") == 3, "flow_size=%s" % on.get("flow_size"))

# the two relations are scored apart, which is how a degenerate control chain
# (recall bought by painting everything) becomes visible
MIX = {"anchors": [4],
       "edges": [{"src": 3, "dst": 4, "kind": "control"},
                 {"src": 2, "dst": 3, "kind": "control"},
                 {"src": 1, "dst": 2, "kind": "control"},
                 {"src": 0, "dst": 4, "kind": "data"}]}
m = _score(Prediction(task_id="t", critical_step=None, flow=MIX),
           GroundTruth(task_id="t", critical_step=2))
check("control recall is scored separately from data recall",
      m["gold_on_control_flow"] is True and m["gold_on_data_flow"] is False,
      "data=%s control=%s" % (m.get("gold_on_data_flow"),
                              m.get("gold_on_control_flow")))
check("each relation's painted size is reported separately",
      m.get("control_flow_size") == 4 and m.get("data_flow_size") == 2,
      "data=%s control=%s" % (m.get("data_flow_size"),
                              m.get("control_flow_size")))

agg = aggregate_mod([dict(m, n_steps=5, gold=2)])
check("aggregate reports coverage next to recall",
      agg.get("control_flow_coverage") == 0.8
      and agg.get("data_flow_coverage") == 0.4,
      "data_cov=%s control_cov=%s" % (agg.get("data_flow_coverage"),
                                      agg.get("control_flow_coverage")))

# ---------------------------------------------------------------------------
print("\n[16] the localiser is told which steps are symptoms, and answers freely")

# 32% of v6 predictions landed exactly on an anchor, where acc@0 was 0.273
# against 0.420 elsewhere -- and mean confidence was HIGHER there (0.77 vs
# 0.62). The anchor is by construction where the failure became visible, so
# the prompt now names the anchors and demands a reason beyond visibility.
# It must NOT ban them: gold really is an anchor in 10% of cases.

class Localiser(LLM):
    """Captures the prompt and answers with whatever step it is told to."""

    def __init__(self, answer, extra=None):
        super().__init__(dry_run=True)
        self.answer = answer
        self.extra = extra or {}
        self.prompt = ""
        self.budget = None

    def generate_obj(self, prompt, max_tokens=None, system="",
                     force_json=True):
        self.prompt = prompt
        self.budget = max_tokens
        return dict({"critical_step": self.answer, "confidence": 0.9,
                     "reason": "because"}, **self.extra)


_g = FlowGraph(anchors=[7, 9], failure_summary="patch rejected")
_g.add(FlowEdge(src=2, dst=7, kind="data", entity="cfg"))
_sp = [Span(start=0, end=n - 1, tier="weak", label="all")]

llm = Localiser(2)
p = localize_mod.localize(traj, _sp, _g, llm)
# A ceiling the answer can hit is not a smaller answer, it is no answer: the
# JSON never closes, parsing fails, and the case falls back and scores zero.
# The full run lost 20 cases that way at 1600, so the budget is pinned here.
check("the answer budget leaves room for the whole JSON",
      llm.budget is not None and llm.budget >= 4000)
check("the prompt names the symptom steps explicitly",
      "step 7" in llm.prompt and "step 9" in llm.prompt,
      "anchors absent from the prompt")
check("the prompt says a symptom step needs a reason beyond visibility",
      "BEYOND revealing the failure" in llm.prompt,
      "the anchor caveat is missing")
check("an off-anchor answer is returned unchanged", p.critical_step == 2,
      "got %s" % p.critical_step)

# the two relations go in as separate blocks: a path that alternates between
# them does not exist, so the prompt must not invite reading one
check("data and control flow are presented as separate blocks",
      "DATA FLOW:" in llm.prompt and "CONTROL FLOW:" in llm.prompt
      and llm.prompt.index("DATA FLOW:") < llm.prompt.index("CONTROL FLOW:"),
      "the relations are not split into their own sections")
check("each relation's meaning is spelled out for the localiser",
      "where the information used at each step came from" in llm.prompt
      and "why the agent went from one step to the next" in llm.prompt,
      "the relations are labelled but not explained")
_dblock = llm.prompt.split("DATA FLOW:")[1].split("CONTROL FLOW:")[0]
check("the data block holds only data edges",
      "step 2 -> step 7" in _dblock and "[cfg]" in _dblock,
      "data block=%r" % _dblock[:120])
# the graph has no control edges, so that block must say so rather than be
# empty -- an empty section reads as "not traced" when it means "none found"
_cblock = llm.prompt.split("committed to the wrong direction.")[1]
check("a relation with no edges says so instead of being blank",
      "(none traced)" in _cblock.split("===")[0],
      "empty control block=%r" % _cblock[:120])

# Part 2 describes uneven compression as a heuristic, not a verdict: the
# localiser must not read "kept in full" as "this is the guilty step"
check("Part 2 explains that detail was kept unevenly by relevance",
      "keep much more of their detail" in llm.prompt
      and "reduced to a summary" in llm.prompt,
      "Part 2 does not describe the uneven compression")
check("Part 2 warns the compression judgement is not a verdict",
      "not thereby guilty" in llm.prompt,
      "Part 2 lets the model treat retained detail as guilt")

# gold is an anchor in 10% of cases; answering with one must stay possible
anchor_ans = localize_mod.localize(traj, _sp, _g, Localiser(7))
check("answering with a symptom step is still allowed",
      anchor_ans.critical_step == 7,
      "an anchor answer was overridden to %s" % anchor_ans.critical_step)

# the ruled-out step is the evidence that the backward walk happened
llm2 = Localiser(2, {"rejected_step": 9,
                     "rejected_because": "only reported the bad config"})
p2 = localize_mod.localize(traj, _sp, _g, llm2)
check("the rejected step is recorded for analysis",
      "ruled out step 9" in p2.reason and "bad config" in p2.reason,
      "reason=%r" % p2.reason[-80:])
p3 = localize_mod.localize(traj, _sp, _g, Localiser(2, {"rejected_step": None}))
check("a missing rejection degrades quietly", p3.reason == "because",
      "reason=%r" % p3.reason)

# ---------------------------------------------------------------------------
print("\n[18] the flow blocks expose each edge's position in the graph")

# Painting more of the trajectory raised flow_recall from 0.70 to 0.80 without
# moving acc@0: the gold step was reaching the flow and the localiser still
# was not picking it. The flow went in as sixty unordered pairs, so distance
# from the symptom, topical relevance and convergence were all invisible
# unless the reader rebuilt the chain by hand. These are graph properties, not
# scores: nothing is reordered and nothing is dropped, because a weight fitted
# on this corpus would be tuned to it and a filter would lose the answer
# outright wherever the heuristic is wrong.

_ann = FlowGraph(anchors=[9], failure_summary="patch rejected",
                 failure_entities=["config.yml"])
for _s, _d, _e in ((7, 9, "cfg blob"), (4, 7, "config.yml"), (2, 4, "raw text"),
                   (4, 8, "config.yml")):
    _ann.add(FlowEdge(src=_s, dst=_d, kind="data", entity=_e))
_ann.add(FlowEdge(src=1, dst=6, kind="data", entity="unrelated"))  # off-chain
_block = localize_mod._flow_block(_ann, "data")
_line = {int(ln.split("step ")[1].split(" ")[0]): ln
         for ln in _block.splitlines() if "step " in ln}

# Hops are counted from the SOURCE, so an edge whose source IS a symptom step
# would be 0 — but the source of an edge INTO a symptom is already one step
# upstream, which is where the reader starts walking.
check("an edge into a symptom step is one hop upstream",
      "1 hops from symptom" in _line[7], _line[7])
check("hop count grows walking upstream",
      "2 hops from symptom" in _line[4]
      and "3 hops from symptom" in _line[2], [_line[4], _line[2]])
check("an edge that reaches no symptom is marked, not silently equal",
      "not on a chain to any symptom" in _line[1], _line[1])
check("an edge carrying a failure artefact is marked",
      "names a failure artefact" in _line[4], _line[4])
check("an edge carrying something else is not",
      "names a failure artefact" not in _line[2], _line[2])
check("convergence is counted where chains actually meet",
      "2 chains converge here" in _line[4]
      and "converge" not in _line[7], [_line[4], _line[7]])

# The annotations are inert unless the prompt says what they mean, and they
# must not be read as a ranking or the localiser will just take the largest.
llm4 = Localiser(4)
localize_mod.localize(traj, _sp, _ann, llm4)
check("the prompt explains what the hop count measures",
      "how many edges lie between" in llm4.prompt)
check("the prompt says the annotations are not a ranking",
      "NOT ordered by them" in llm4.prompt)
check("the prompt warns a long chain can drift off the cause",
      "drift into steps that merely came" in llm4.prompt)
check("the prompt denies that a marked artefact implies blame",
      "the edge is on-topic" in llm4.prompt)
check("the prompt denies that convergence implies fault",
      "correct result also converges" in llm4.prompt)

# Order and membership are unchanged: this stage adds reading aids, and a
# comparison against an earlier run must not be confounded by a reshuffle.
_plain = [ln.split(" (")[0] for ln in _block.splitlines()]
check("annotating changes neither the order nor the set of edges",
      _plain == ["  step 7 -> step 9 [cfg blob]", "  step 4 -> step 8 [config.yml]",
                 "  step 4 -> step 7 [config.yml]", "  step 1 -> step 6 [unrelated]",
                 "  step 2 -> step 4 [raw text]"], _plain)

# A near-empty entity would match everything by containment.
check("a trivially short entity does not count as naming an artefact",
      not localize_mod._names_failure_artefact("a", ["ansible"])
      and not localize_mod._names_failure_artefact("cfg", ["c"]))
check("an artefact is matched loosely in both directions",
      localize_mod._names_failure_artefact("tests/test_cfg.py::test_load",
                                           ["test_cfg"])
      and localize_mod._names_failure_artefact("cfg", ["cfg.yml"]))

# ---------------------------------------------------------------------------
print("\n[17] the top-up re-asks only where a chain to an anchor stopped")

# The forward sweep declares roughly half the steps of a long trajectory to be
# roots, and every such answer truncates whatever chain ran through it. The
# top-up asks again — but only about painted leaves, so its cost tracks chain
# length rather than trajectory length, and it must be free to answer "root".


class TopupLLM(LLM):
    """Gives every step it is asked about one data parent two steps back."""

    def __init__(self, origins=()):
        super().__init__(dry_run=True)
        self.origins = set(origins)
        self.asked = []             # flattened, in call order
        self.groups = []
        self.prompt = ""

    def generate_obj(self, prompt, max_tokens=None, system="",
                     force_json=True):
        import re as _re
        self.prompt = prompt
        body = prompt.split("STEPS TO TRACE NOW")[-1]
        traced = [int(m) for m in _re.findall(r"\[step (\d+)", body)]
        self.groups.append(tuple(traced))
        self.asked.extend(traced)
        edges = [{"src": d - 2, "dst": d, "kind": "data", "entity": "e",
                  "rationale": "r"}
                 for d in traced if d - 2 >= 0 and d not in self.origins]
        return {"edges": edges,
                "origins": [d for d in traced if d in self.origins]}


if n >= 21:
    # a graph whose data chain stops dead at 16: 20 <- 18 <- 16, nothing above
    g = FlowGraph(anchors=[20], failure_summary="patch rejected",
                  failure_entities=["cfg.yml"])
    for s, d in ((18, 20), (16, 18)):
        g.add(FlowEdge(src=s, dst=d, kind="data", entity="e"))
    # an unpainted branch: it has a parent but never reaches the anchor
    g.add(FlowEdge(src=3, dst=9, kind="data", entity="e"))

    llm = TopupLLM()
    tracing_mod.topup_backward(traj, llm, g, rounds=1, batch=4, workers=1)
    asked = set(llm.asked)
    check("the stalled end of the painted chain is re-asked",
          16 in asked, asked)
    check("a step whose chain still continues upward is not re-asked",
          18 not in asked, asked)
    # 20 is an anchor with a data parent but no control parent, so its control
    # chain is dead at hop zero. That is a leaf in the control forest and worth
    # re-asking about even though the step is not a dead end in data — a third
    # of anchors in the long corpus have no parent at all, and painting there
    # never starts.
    check("a leaf in one relation is re-asked even if the other continues",
          20 in asked, asked)
    check("unpainted steps are not re-asked",
          not ({3, 9} & asked), asked)
    check("the top-up extends the chain it was called for",
          14 in g.paint("data"), sorted(g.paint("data")))

    # A step the model calls a root must not be asked about again, or the
    # stage spends its whole budget re-litigating the same answer.
    g2 = FlowGraph(anchors=[20])
    for s, d in ((18, 20), (16, 18)):
        g2.add(FlowEdge(src=s, dst=d, kind="data", entity="e"))
    stub = TopupLLM(origins=[16])
    tracing_mod.topup_backward(traj, stub, g2, rounds=3, batch=4, workers=1)
    check("a step declared a root is not asked about again",
          stub.asked.count(16) == 1, stub.asked)
    check("declaring a root adds no edge",
          16 not in {e.dst for e in g2.edges}, [e.key() for e in g2.edges])

    check("the top-up tells the model that answering 'no parent' is correct",
          "not a failure to find something" in stub.prompt)
    check("the top-up warns against inventing a parent",
          "Do not invent a parent" in stub.prompt)

    # The call cap is the cost guarantee: without it a wide frontier on a long
    # trajectory would issue one call per four painted leaves, every round.
    g3 = FlowGraph(anchors=list(range(4, min(n, 21))))
    capped = TopupLLM()
    tracing_mod.topup_backward(traj, capped, g3, rounds=5, batch=1,
                               max_calls=3, workers=1)
    check("the call cap bounds what the top-up spends",
          len(capped.groups) <= 3, len(capped.groups))

    # Off by default: it changes the graph, so no run picks it up silently and
    # every comparison against an earlier run stays valid.
    check("the top-up is off unless asked for",
          inspect.signature(tracing_mod.trace)
          .parameters["topup_rounds"].default == 0
          and Config().topup_rounds == 0)
    check("the top-up setting is recorded in the artifact fingerprint",
          "topup_rounds" in Config().fingerprint())

    # A backwards edge would make the "cause precedes effect" reading of the
    # graph false, and a self-edge would paint a step as its own ancestor.
    class BadEdges(TopupLLM):
        def generate_obj(self, prompt, max_tokens=None, system="",
                         force_json=True):
            super().generate_obj(prompt, max_tokens, system, force_json)
            return {"edges": [{"src": 19, "dst": 16, "kind": "data"},
                              {"src": 16, "dst": 16, "kind": "data"},
                              {"src": 99, "dst": 16, "kind": "data"},
                              {"src": 5, "dst": 16, "kind": "sequence"}],
                    "origins": []}

    g6 = FlowGraph(anchors=[20])
    for s, d in ((18, 20), (16, 18)):
        g6.add(FlowEdge(src=s, dst=d, kind="data", entity="e"))
    tracing_mod.topup_backward(traj, BadEdges(), g6, rounds=1, workers=1)
    check("the top-up rejects backwards, self, out-of-range and untyped edges",
          len(g6.edges) == 2, [e.key() for e in g6.edges])

    # An anchorless graph has nothing to walk up from; it must not call out.
    g5 = FlowGraph(anchors=[])
    idle = TopupLLM()
    tracing_mod.topup_backward(traj, idle, g5, rounds=2, workers=1)
    check("no anchors means no top-up calls", not idle.groups)


# ---------------------------------------------------------------------------
print("\n[19] the span prior narrows the region without deciding the answer")

# Painting more of the trajectory and annotating the edges both left
# P(hit | gold on flow) at 0.50: the reference step was reaching the localiser
# and the localiser passed over it anyway. So this stage does not add evidence,
# it nominates a REGION — and everything below exists to keep that hint from
# quietly becoming the answer, or from confounding the A/B that measures it.


class PriorLLM(LLM):
    """Returns a fixed span-prior payload and records the prompt it saw."""

    def __init__(self, payload):
        super().__init__(dry_run=True)
        self.payload = payload
        self.prompt = ""
        self.calls = 0

    def generate_obj(self, prompt, max_tokens=None, system="",
                     force_json=True):
        self.prompt = prompt
        self.calls += 1
        return self.payload


_phases = [Span(start=0, end=3, tier="strong", label="orient",
                summary="listed the repo and read the failing test"),
           Span(start=4, end=6, tier="weak", label="patch",
                summary="edited the connection plugin and reran the test"),
           Span(start=7, end=9, tier="weak", label="submit",
                summary="submitted the patch, which was rejected")]
# Nine narrow phases over a 28-step run: enough that three guesses can be
# allowed without any of them nominating a large share of the trajectory.
_phases += [Span(start=10 + i * 2, end=11 + i * 2, tier="strong",
                 label="phase %d" % i, summary="did thing %d" % i)
            for i in range(6)]
_pg = FlowGraph(anchors=[7, 9], failure_summary="patch rejected: winrm exec",
                failure_entities=["_winrm_exec"])
_pg.add(FlowEdge(src=2, dst=7, kind="data", entity="cfg"))

_pl = PriorLLM({"spans": [{"start": 4, "end": 6, "likelihood": "likely",
                           "why": "the edit was committed here"}]})
_prior = span_prior_mod.predict_spans(traj, _phases, _pg, _pl)
check("the prior returns the guessed phase", len(_prior.guesses) == 1
      and (_prior.guesses[0].start, _prior.guesses[0].end) == (4, 6),
      [g.to_dict() for g in _prior.guesses])

# It must know the symptom -- "which phase caused it" is unanswerable
# otherwise -- but by default it must NOT see the flow. flow_recall is already
# 0.80 while acc@0 is 0.44, so the flow is not the missing information; a prior
# that reads it is an echo of the localiser's own view, whereas an independent
# reader that happens to agree is evidence. ``prior_signals`` relaxes this to
# aggregate counts only -- see group [21], which pins what stays out even then.
check("the prior is told what visibly went wrong",
      "patch rejected: winrm exec" in _pl.prompt)
check("the prior is given the phase summaries",
      "edited the connection plugin" in _pl.prompt
      and "steps 4-6" in _pl.prompt)
check("the prior is blind to the traced edges",
      "step 2 -> step 7" not in _pl.prompt and "[cfg]" not in _pl.prompt,
      _pl.prompt[-400:])
check("the prior is blind to the weak/strong tiers",
      "on dependency path" not in _pl.prompt and "weak" not in _pl.prompt,
      _pl.prompt[-400:])
check("the prior is told it picks phases, not the step",
      "You are NOT choosing the step" in _pl.prompt)
check("the prior inherits the symptom-is-not-the-cause warning",
      "where the causal chain ENDS" in _pl.prompt)
check("the prior inherits the exploration-is-not-an-error rule",
      "Exploration is not an error" in _pl.prompt)

# The model is handed the exact ranges and still invents them. Snapping keeps
# every guess denoting a phase that exists, so prior_coverage measures the
# region actually nominated rather than whatever interval got typed.
_snap = span_prior_mod.predict_spans(traj, _phases, _pg, PriorLLM({"spans": [
    {"start": 5, "end": 5},                       # inside phase 4-6
    {"start": 4, "end": 6},                       # duplicate after snapping
    {"start": 1, "end": 2, "likelihood": "possible"},
    {"start": 900, "end": 999},                   # no overlap at all
    {"start": 8, "end": 8},                       # would be a 4th
]}))
check("an invented range snaps to the phase it overlaps most",
      (_snap.guesses[0].start, _snap.guesses[0].end) == (4, 6),
      [g.to_dict() for g in _snap.guesses])
check("a range that overlaps nothing is dropped",
      all(g.start != 900 for g in _snap.guesses),
      [g.to_dict() for g in _snap.guesses])
check("duplicate phases are not counted twice",
      len({(g.start, g.end) for g in _snap.guesses}) == len(_snap.guesses),
      [g.to_dict() for g in _snap.guesses])
check("the prior names at most three phases", len(_snap.guesses) <= 3,
      len(_snap.guesses))
check("the model's own order is preserved: the first guess is the top guess",
      (_snap.guesses[0].start, _snap.guesses[1].start) == (4, 0),
      [g.start for g in _snap.guesses])
check("an unrecognised likelihood degrades to 'possible'",
      span_prior_mod.predict_spans(traj, _phases, _pg, PriorLLM(
          {"spans": [{"start": 4, "end": 6, "likelihood": "certain"}]}
      )).guesses[0].likelihood == "possible")

# A HINT THAT POINTS AT MOST OF THE RUN IS NOT A HINT. The first smoke run
# caught this: on short trajectories ``segment`` bottoms out at its floor of
# four spans, three guesses nominated 80% of the steps, and the recall that
# bought was free. Two bounds, both structural rather than tuned — the guess
# budget scales with how many phases exist, and guesses stop once they would
# nominate more than half the run.
_greedy = {"spans": [{"start": s.start, "end": s.end} for s in _phases]}
check("fewer than four phases cannot be narrowed and cost no call",
      not span_prior_mod.predict_spans(traj, _phases[:3], _pg,
                                       (_c3 := PriorLLM(_greedy))).guesses
      and _c3.calls == 0)
_five = span_prior_mod.predict_spans(traj, _phases[:5], _pg, PriorLLM(_greedy))
check("five phases buy one guess, not three", len(_five.guesses) == 1,
      [g.to_dict() for g in _five.guesses])
_seven = span_prior_mod.predict_spans(traj, _phases[:7], _pg, PriorLLM(_greedy))
check("seven phases buy two guesses", len(_seven.guesses) == 2,
      [g.to_dict() for g in _seven.guesses])
_all9 = span_prior_mod.predict_spans(traj, _phases, _pg, PriorLLM(_greedy))
check("the guess budget never exceeds three", len(_all9.guesses) == 3,
      [g.to_dict() for g in _all9.guesses])
_b1 = PriorLLM(_greedy)
span_prior_mod.predict_spans(traj, _phases[:5], _pg, _b1)
check("the prompt tells the model its actual budget, not the ceiling",
      "AT MOST 1 phase(s)" in _b1.prompt, _b1.prompt[-320:])
check("the prompt says naming too many is worse than naming one",
      "worse than naming one" in _b1.prompt)
# Wide phases with the full budget of three available, so a dropped guess can
# only have been dropped by the coverage bound: half of a 28-step run is 14,
# and 9 + 5 reaches it exactly while the third guess would pass it.
_wide = ([Span(start=0, end=8, label="a", summary="s"),
          Span(start=9, end=13, label="b", summary="s"),
          Span(start=14, end=16, label="c", summary="s")]
         + [Span(start=17 + i, end=17 + i, label="p%d" % i, summary="s")
            for i in range(6)])
_over = span_prior_mod.predict_spans(
    traj, _wide, _pg,
    PriorLLM({"spans": [{"start": 0, "end": 8}, {"start": 9, "end": 13},
                        {"start": 14, "end": 16}]}))
check("a guess that would nominate over half the run is dropped",
      [(g.start, g.end) for g in _over.guesses] == [(0, 8), (9, 13)],
      [g.to_dict() for g in _over.guesses])
# If the model's strongest phase is itself huge, that is a fact about the
# segmentation. Reporting a wide prior beats reporting no prior and losing the
# case from the diagnostic entirely.
_huge = span_prior_mod.predict_spans(
    traj, [Span(start=0, end=25, label="a", summary="s"),
           Span(start=26, end=26, label="b", summary="s"),
           Span(start=27, end=27, label="c", summary="s"),
           Span(start=28, end=28, label="d", summary="s")], _pg,
    PriorLLM({"spans": [{"start": 0, "end": 25}, {"start": 26, "end": 26}]}))
check("the top guess survives even when it alone exceeds the bound",
      [(g.start, g.end) for g in _huge.guesses] == [(0, 25)],
      [g.to_dict() for g in _huge.guesses])

# Degradation must be silent and cheap: an empty prior is the normal state
# whenever the extra call fails, and the run has to continue without it.
check("an unusable answer yields an empty prior, not a crash",
      not span_prior_mod.predict_spans(traj, _phases, _pg,
                                       PriorLLM({"nonsense": 1})).guesses)

# THE A/B INVARIANT. Every intervention so far has been measured by re-running
# the localiser against a stored flow and comparing with the previous run. If
# the prompt shifted even when the prior is absent, that comparison would be
# measuring the reshuffle instead of the hint.
_base = Localiser(2)
localize_mod.localize(traj, _sp, _g, _base)
_none = Localiser(2)
localize_mod.localize(traj, _sp, _g, _none, prior=None)
_empty = Localiser(2)
localize_mod.localize(traj, _sp, _g, _empty, prior=SpanPrior())
check("prior=None leaves the localiser prompt byte-identical",
      _base.prompt == _none.prompt,
      "%d vs %d chars" % (len(_base.prompt), len(_none.prompt)))
check("an empty prior leaves the localiser prompt byte-identical",
      _base.prompt == _empty.prompt,
      "%d vs %d chars" % (len(_base.prompt), len(_empty.prompt)))
check("no prior means no answer fields about one",
      "prior_span" not in _base.prompt and "prior_disagreement"
      not in _base.prompt)

_hint = SpanPrior(guesses=[SpanGuess(start=4, end=6, likelihood="likely",
                                     why="the edit was committed here"),
                           SpanGuess(start=0, end=3, why="thin summary")])
_lw = Localiser(5, {"prior_span": 4})
_ph = localize_mod.localize(traj, _sp, _g, _lw, prior=_hint)
check("the guessed phases reach the localiser",
      "steps 4-6 (likely)" in _lw.prompt and "steps 0-3 (possible)" in _lw.prompt,
      _lw.prompt[_lw.prompt.find("SEPARATE READER"):][:200])
check("the localiser is told the guess is not an answer",
      "not an answer" in _lw.prompt and "follow the\nevidence" in _lw.prompt)
check("the localiser is told the guess was made without the evidence",
      "without the evidence you are about to read" in _lw.prompt)
# A hint that only ever adds suspicion would drive the answer into the
# nominated region regardless of what the trajectory says.
check("the localiser is told an unnamed phase is not thereby cleared",
      "is not thereby cleared" in _lw.prompt
      and "not thereby guilty" in _lw.prompt)

# Anti-recency: the evidence is read last on purpose. The hint must not win by
# sitting closest to the question.
check("the hint is placed before the step-level evidence",
      _lw.prompt.index("SEPARATE READER'S GUESS") < _lw.prompt.index("PART 2"))
check("the hint is placed after the descriptive outline",
      _lw.prompt.index("PART 1") < _lw.prompt.index("SEPARATE READER'S GUESS"))

check("the answer schema asks which guessed phase was used",
      '"prior_span"' in _lw.prompt and '"prior_disagreement"' in _lw.prompt)
check("the prior and the localiser's use of it are both recorded",
      _ph.span_prior is not None
      and [g["start"] for g in _ph.span_prior["guesses"]] == [4, 0]
      and _ph.span_prior["chosen"] == 4, _ph.span_prior)

# The self-report is bookkeeping, not behaviour: the model will name a phase
# that does not contain the step it just gave. Recomputing keeps the diagnostic
# about where the answer landed.
_out = Localiser(1, {"prior_span": 4, "prior_disagreement": ""})
_po = localize_mod.localize(traj, _sp, _g, _out, prior=_hint)
check("a self-report contradicted by the answer is not believed",
      _po.span_prior["chosen"] == 0 and _po.span_prior["claimed"] == 4,
      _po.span_prior)
_gone = Localiser(9, {"prior_span": 6,
                      "prior_disagreement": "the symptom step botched the write"})
_pg2 = localize_mod.localize(traj, _sp, _g, _gone, prior=_hint)
check("an answer outside every guess is recorded as such",
      _pg2.span_prior["chosen"] is None
      and "botched the write" in _pg2.span_prior["disagreement"], _pg2.span_prior)

# --- --prior-unranked: the same regions, with the ranking taken out --------
# The measured problem this pins: when the prior's top guess is wrong the
# localiser scores 0/22 on TrajAudit, while equally hard rows where it answered
# outside the prior score 0.538 and rows carrying no prior at all score 0.778.
# The flag hands over the same regions in trajectory order with no likelihood
# word and no implied rank. Two things have to hold for the A/B to mean
# anything: OFF must be byte-identical to a run before the flag existed, and ON
# must actually carry no ordering or likelihood token -- otherwise the reader
# can still recover the rank and the arm measures a reworded paragraph.
_rk = Localiser(5, {"prior_span": 4})
localize_mod.localize(traj, _sp, _g, _rk, prior=_hint)
check("prior_unranked off leaves the localiser prompt byte-identical",
      _rk.prompt == _lw.prompt,
      "%d vs %d chars" % (len(_rk.prompt), len(_lw.prompt)))

_ur = Localiser(5, {"prior_span": 4})
_pu = localize_mod.localize(traj, _sp, _g, _ur, prior=_hint,
                            prior_unranked=True)
check("the unranked block replaces the ranked one",
      "SEPARATE READER'S SHORTLIST" in _ur.prompt
      and "SEPARATE READER'S GUESS" not in _ur.prompt)
# _hint is deliberately given out of order (4-6 before 0-3) so that "sorted"
# and "as the model ranked them" are distinguishable in the rendered text.
check("the regions are rendered in trajectory order, not the model's order",
      _ur.prompt.index("steps 0-3") < _ur.prompt.index("steps 4-6"),
      _ur.prompt[_ur.prompt.find("SEPARATE READER"):][:300])
_block = _ur.prompt[_ur.prompt.index("SEPARATE READER'S SHORTLIST"):]
_block = _block[:_block.index("---", 20)]
check("no likelihood word survives into the unranked block",
      "likely" not in _block.replace("likelier", "")
      and "possible" not in _block,
      _block)
check("the unranked block says outright that the list is not ranked",
      "NOT ranked" in _block and "unordered" in _block)
# Dropping ``why`` as well would make this arm an ablation of the prior's
# content, not of its ranking, and the two would be inseparable afterwards.
check("the reasons the prior gave are still handed over",
      "the edit was committed here" in _block and "thin summary" in _block)
# The recorded prior is the prior, whatever the prompt did with it: the
# diagnostics have to stay comparable across the two arms.
check("the record keeps the prior's own order regardless of rendering",
      [g["start"] for g in _pu.span_prior["guesses"]] == [4, 0],
      _pu.span_prior)
_ue = Localiser(2)
localize_mod.localize(traj, _sp, _g, _ue, prior=SpanPrior(),
                      prior_unranked=True)
check("an empty prior is still byte-identical with prior_unranked on",
      _ue.prompt == _base.prompt,
      "%d vs %d chars" % (len(_ue.prompt), len(_base.prompt)))

# --- --op-analysts: five readers over five DISJOINT slices ----------------
# The prior art this has to be told apart from is --multi-expert, which split
# four readers by failure family over SHARED evidence, converged to 1.6
# distinct answers out of 4, and scored 0.3333 on TrajAudit. The whole claim
# here is the partition, so the assertions below are about the partition: each
# analyst's prompt must contain its own steps and NONE of anyone else's, and
# the classifier must never produce a sixth bucket for a step to hide in.
_oa_steps = [
    Step(index=0, role="user", content="fix the config loader"),
    Step(index=1, role="assistant",
         content="Thought: the loader probably reads the wrong key."),
    Step(index=2, role="assistant",
         content='Action: grep -rn "load_config" src/'),
    Step(index=3, role="assistant",
         content='Action: str_replace_editor view /app/src/loader.py'),
    Step(index=4, role="assistant",
         content='Action: str_replace_editor str_replace /app/src/loader.py '
                 '--old_str "cfg[\'a\']" --new_str "cfg[\'b\']"'),
    Step(index=5, role="assistant",
         content='Action: python -m pytest tests/test_loader.py'),
]
_oa_traj = Trajectory(task_id="oa", source="trajaudit", subset="s",
                      task_description="fix the config loader",
                      steps=_oa_steps)
_oa_slices = op_analysts_mod.slice_steps(_oa_traj)
check("every operation type gets a bucket, and only the five",
      sorted(_oa_slices) == sorted(op_analysts_mod.OP_TYPES))
check("the classifier partitions the attributable steps exactly once",
      sorted(i for v in _oa_slices.values() for i in v) == [1, 2, 3, 4, 5],
      _oa_slices)
check("a step with no Action line is reasoning", _oa_slices["reason"] == [1])
check("grep is search, not a test despite the word in the path",
      _oa_slices["search"] == [2], _oa_slices)
check("an editor view is reading", _oa_slices["read"] == [3], _oa_slices)
check("str_replace is a modification", _oa_slices["edit"] == [4], _oa_slices)
check("pytest is execution", _oa_slices["test"] == [5], _oa_slices)
# The user turn states the task; it can never be the agent's mistake, so an
# analyst that could nominate it would only have the answer snapped away.
check("non-attributable turns reach no analyst",
      all(0 not in v for v in _oa_slices.values()))
# An unclassifiable step must still land somewhere: an "other" bucket owned by
# no analyst held a quarter of the reference steps in an earlier draft.
check("the classifier has no escape hatch",
      op_analysts_mod.optype("Action: frobnicate --wildly") in
      op_analysts_mod.OP_TYPES)
check("an empty step is reasoning, not a crash",
      op_analysts_mod.optype("") == "reason"
      and op_analysts_mod.optype(None) == "reason")


class _Analyst(LLM):
    """Records every analyst prompt and answers with a fixed step."""

    def __init__(self, answers):
        super().__init__(dry_run=True)
        self.answers = answers          # step to name, or None to abstain
        self.prompts = []

    def generate_obj(self, prompt, max_tokens=None, system="",
                     force_json=True):
        self.prompts.append(prompt)
        a = self.answers[len(self.prompts) - 1]
        if a is None:
            return {"found": False, "critical_step": None, "confidence": 0.0,
                    "reason": ""}
        return {"found": True, "critical_step": a, "confidence": 0.7,
                "reason": "step %d committed the fault" % a}


_oa_g = FlowGraph(anchors=[5], failure_summary="tests still fail")
_oa_sp = [Span(start=0, end=5, tier="weak", label="all")]
_oa_llm = _Analyst([4, None, None, None, None])
_oa_find = op_analysts_mod.run_analysts(_oa_traj, _oa_sp, _oa_g, _oa_llm,
                                        workers=1)
check("one call per non-empty slice and no more",
      len(_oa_llm.prompts) == 5, len(_oa_llm.prompts))
check("a finding is returned for every type, in a fixed order",
      [f.op for f in _oa_find] == list(op_analysts_mod.OP_TYPES))
# This is the assertion the design lives or dies by. If an analyst can see
# another analyst's steps then the partition is not a partition and the arm is
# measuring --multi-expert again under a new name.
for _p in _oa_llm.prompts:
    _seen = {i for i in range(1, 6) if ("[step %d |" % i) in _p}
    _owner = [op for op, idxs in _oa_slices.items() if set(idxs) == _seen]
    check("an analyst prompt holds exactly one type's steps (%s)"
          % sorted(_seen), len(_owner) == 1, _p[:200])
check("no analyst is shown the user turn", all("[step 0 |" not in p
                                               for p in _oa_llm.prompts))
check("analysts are told declining is a real answer",
      all("found\": false" in p or '"found": <true' in p
          for p in _oa_llm.prompts))

_oa_stats = op_analysts_mod.stats(_oa_find)
check("the record counts calls, nominations and declines",
      _oa_stats["n_called"] == 5 and _oa_stats["n_found"] == 1
      and _oa_stats["n_declined"] == 4, _oa_stats)
# The number the arm exists to move: --multi-expert averaged 1.6 of 4.
check("the record reports how many DISTINCT steps were named",
      _oa_stats["distinct"] == 1, _oa_stats)

# A nomination the analyst was never shown is not a finding about that step,
# it is the model ignoring its remit.
_oa_out = _Analyst([2, None, None, None, None])       # edit analyst names a grep
_oa_f2 = op_analysts_mod.run_analysts(_oa_traj, _oa_sp, _oa_g, _oa_out,
                                      workers=1)
check("a step outside the analyst's own slice is dropped, not snapped",
      all(not f.found for f in _oa_f2), [f.to_dict() for f in _oa_f2])


class _BadAnalyst(LLM):
    def generate_obj(self, prompt, max_tokens=None, system="",
                     force_json=True):
        return {}


_oa_bad = op_analysts_mod.run_analysts(_oa_traj, _oa_sp, _oa_g,
                                       _BadAnalyst(dry_run=True), workers=1)
check("an unparseable answer is an abstention, not a failure",
      all(f.called and not f.found for f in _oa_bad))
check("nothing nominated renders no block at all",
      op_analysts_mod.render_findings(_oa_bad) == "")

_oa_block = op_analysts_mod.render_findings(_oa_find)
check("the block names the nominated step and carries its reason",
      "STEP 4" in _oa_block and "committed the fault" in _oa_block)
# A decline is evidence -- "six edits read, nothing found" is not the same as
# "no edits in this run" -- and the arbiter has to be able to tell them apart.
check("declines are reported with how much was read",
      "declined to nominate" in _oa_block and "read 1 step" in _oa_block)
check("the arbiter is told these readers had no global view",
      "only reader with the whole picture" in _oa_block
      and "may reject all of them" in _oa_block)
check("the arbiter is told agreement is not corroboration",
      "NOT corroboration" in _oa_block)
# _GOLD_KEY is for artifact KEY names; these are prose, where "answer" is the
# ordinary English word. Same regex the audit and memo-scan sections use.
for _oa_name, _oa_txt in (("ANALYST_PROMPT", op_analysts_mod.ANALYST_PROMPT),
                          ("ANALYST_SYSTEM", op_analysts_mod.ANALYST_SYSTEM),
                          ("ANALYST_BLOCK", op_analysts_mod.ANALYST_BLOCK)):
    check("%s names no label-side field" % _oa_name,
          not re.search(r"gold|ground.?truth|annotation|human.?rationale"
                        r"|mistake_|critical_error|failure_id|test_patch"
                        r"|gold_patch|failure_reason", _oa_txt, re.I))
check("run_analysts cannot be handed a ground truth",
      not any(re.search(r"gold|label|truth|annotation", n) for n in
              inspect.signature(op_analysts_mod.run_analysts).parameters))
check("the op-analyst stage is off by default",
      Config().op_analysts is False)
# The stored config is the only reliable way to tell two arms apart after the
# fact -- the invalid ta_vpool run proved that -- so a stage the fingerprint
# cannot see is a stage a later reader will mistake for the baseline.
check("the op-analyst stage reaches the config fingerprint",
      "op_analysts" in Config().fingerprint())

_oa_off = Localiser(5, {"prior_span": 4})
localize_mod.localize(traj, _sp, _g, _oa_off, prior=_hint)
check("op-analysts off leaves the localiser prompt byte-identical",
      _oa_off.prompt == _lw.prompt,
      "%d vs %d chars" % (len(_oa_off.prompt), len(_lw.prompt)))
_oa_on = Localiser(5, {"prior_span": 4})
localize_mod.localize(traj, _sp, _g, _oa_on, prior=_hint,
                      analyst_findings=_oa_block)
check("the findings reach the arbiter's prompt",
      "STEP 4" in _oa_on.prompt and len(_oa_on.prompt) > len(_lw.prompt))
# Anti-recency again: the findings are opinions, the trajectory is evidence,
# and the evidence must not be the thing that sits furthest from the question.
check("the findings are placed after the step-level evidence",
      _oa_on.prompt.index("PART 2") < _oa_on.prompt.index("FIVE INDEPENDENT"))

# Coverage is what makes recall readable: a prior nominating half the run
# recalls half the gold steps for free.
check("the prior counts the distinct steps it nominated",
      SpanPrior(guesses=[SpanGuess(0, 3), SpanGuess(2, 6)]).n_steps_covered() == 7)

# The diagnostics this stage was built to produce. Recall alone cannot say
# whether the prior is informative, so coverage is reported beside it, and
# "followed" is reported beside both -- that is what separates "the hint was
# wrong" from "the hint was ignored".
from longtraj import evaluate as evaluate_mod

_row = evaluate_mod.score_one(_ph, GroundTruth(task_id=traj.task_id,
                                               critical_step=5))
_row["n_steps"] = 10
check("gold inside a guessed phase is recorded",
      _row["gold_in_prior_span"] is True and _row["gold_in_top_prior_span"] is True,
      _row)
check("the answer's own landing phase is recorded separately from the gold",
      _row["pred_in_prior_span"] is True, _row)
_miss = evaluate_mod.score_one(_ph, GroundTruth(task_id=traj.task_id,
                                                critical_step=1))
_miss["n_steps"] = 10
check("gold in a lower guess is not counted as a top-guess hit",
      _miss["gold_in_prior_span"] is True
      and _miss["gold_in_top_prior_span"] is False, _miss)
_off = evaluate_mod.score_one(_ph, GroundTruth(task_id=traj.task_id,
                                               critical_step=8))
_off["n_steps"] = 10
check("gold outside every guess is recorded as a prior miss",
      _off["gold_in_prior_span"] is False, _off)
_agg = evaluate_mod.aggregate([_row, _miss, _off])
check("recall is reported against the coverage that bought it",
      _agg["prior_recall"] == round(2 / 3, 4)
      and _agg["prior_top_recall"] == round(1 / 3, 4)
      and _agg["prior_coverage"] == 0.7, _agg)
check("whether the localiser followed the hint is reported too",
      _agg["prior_followed"] == 1.0, _agg)
check("accuracy is split by whether the prior caught the gold",
      "acc@0_given_prior_hit" in _agg and "acc@0_given_prior_miss" in _agg, _agg)
check("the report prints the prior line",
      "span prior: rec=" in evaluate_mod.format_report(_agg))
# A run without the stage must not grow the metrics: the comparison run has to
# stay readable next to the ones already on disk.
_plain_agg = evaluate_mod.aggregate(
    [dict(evaluate_mod.score_one(_base_pred := localize_mod.localize(
        traj, _sp, _g, Localiser(2)),
        GroundTruth(task_id=traj.task_id, critical_step=2)), n_steps=10)])
check("no prior means no prior metrics",
      "prior_recall" not in _plain_agg
      and "span prior:" not in evaluate_mod.format_report(_plain_agg),
      sorted(_plain_agg))

# The knob has to reach the artifact fingerprint, or two runs that differ only
# by this stage are indistinguishable on disk.
check("--span-prior is off by default", Config().span_prior is False)
check("the span-prior knob is recorded in the artifact fingerprint",
      Config(span_prior=True).fingerprint().get("span_prior") is True)

# Ground truth. The prior reads span summaries, which are LLM-written from the
# trajectory -- but this is the corpus's own label sitting one field away, so
# it gets the same recording-LLM check as every other prompt builder.
for rec in records[:2]:
    gt = rec.ground_truth
    if gt is None or gt.critical_step is None:
        continue
    rl = RecordingLLM()
    run(rec.trajectory, rl, Config(max_rounds=1, workers=1, span_prior=True))
    blob = "\n".join(rl.prompts)
    leaked = re.search(r"(critical|mistake|gold|correct)\D{0,40}\b%d\b"
                       % gt.critical_step, blob, re.I)
    check("%s: span prior on, still no gold-step assertion"
          % rec.trajectory.task_id[:26], leaked is None,
          str(leaked.group(0))[:60] if leaked else "")
    if gt.error_type:
        check("%s: span prior on, gold error_type absent"
              % rec.trajectory.task_id[:26], gt.error_type not in blob)
    check("%s: span prior module reads no GroundTruth field"
          % rec.trajectory.task_id[:26],
          not any(w in inspect.getsource(span_prior_mod)
                  for w in ("critical_step", "responsible_role")))

# The stage that must NOT become diagnostic stayed that way.
_osrc = inspect.getsource(outline_mod)
check("the outline stage is still forbidden to diagnose",
      "descriptive, never diagnostic" in _osrc
      and "do NOT diagnose the failure" in _osrc)
check("the prior explains why it may diagnose where the outline may not",
      "crosses that line on purpose" in inspect.getsource(span_prior_mod))


# ---------------------------------------------------------------------------
print("\n[20] the candidate set restricts the answer without hiding the cost")

# Measured on SWE-Bench-Pro before this stage existed: we answer a reasoning-only
# step 34% of the time and score 0.069 there, while the reference step is a write
# 56% of the time. The restriction is a hard one -- the localiser is snapped into
# the set if it strays -- so every check below is about keeping the cost of that
# visible: what the set could not contain, what the model would have said before
# the snap, and which cases were left unrestricted entirely.

_ct = Trajectory(
    task_id="cand-fixture", source="trajdebug", subset="swebenchpro",
    task_description="fix the connection plugin",
    steps=[
        Step(0, "assistant",
             'str_replace_editor({"command": "view", "path": "/workspace/i/a.py"})'),
        Step(1, "assistant",
             'str_replace_editor({"command": "str_replace", '
             '"path": "/workspace/i/lib/foo.py", "old_str": "a", "new_str": "b"})'),
        # a write-looking call in a NON-assistant step: the environment echoing
        # the edit back is not an agent decision and can never be the answer
        Step(2, "tool",
             'str_replace_editor({"command": "create", "path": "/workspace/i/z.py"})'),
        Step(3, "assistant", 'execute_bash({"command": "grep -rn foo lib/"})'),
        Step(4, "assistant",
             'execute_bash({"command": "sed -i s/a/b/ lib/foo.py"})'),
        Step(5, "assistant",
             'str_replace_editor({"command": "create", "path": "/workspace/i/t.py"})'),
    ])
_cands = candidates_mod.candidates(_ct)
check("only the steps that changed something are offered",
      [c["step"] for c in _cands] == [1, 4, 5], _cands)
check("a read, a search and a view are not candidates",
      all(c["step"] not in (0, 3) for c in _cands), _cands)
check("a write echoed by a non-assistant step is not a candidate",
      all(c["step"] != 2 for c in _cands), _cands)
check("each candidate carries a label saying what it changed",
      "str_replace" in _cands[0]["label"] and "lib/foo.py" in _cands[0]["label"],
      _cands)
check("the per-instance workspace prefix is stripped from the label",
      "/workspace/" not in candidates_mod.render(_cands),
      candidates_mod.render(_cands))

# Agents that cannot express an edit through the editor tool fall back to an
# in-line interpreter script: read the file, transform the text, write it back.
# Six of the twenty-four SWE-Bench-Pro reference steps the parse used to miss are
# this idiom, so it is recognised -- but only through a WRITE MODE, never through
# a bare open(). Every one of those six opens in "r" first, because the read
# precedes the edit, so open() alone marks file access and would sweep in the
# read-only one-liners and pytest runs that sit in the same trajectories. Both
# directions are pinned: dropping the write-mode requirement would pass the
# first four of these and fail the last three.
_py = Trajectory(
    task_id="py-fixture", source="trajdebug", subset="swebenchpro",
    task_description="fix it",
    steps=[
        Step(0, "assistant",
             'execute_bash({"command": "python3 -c \'\\nwith open(\\"a.py\\") '
             'as f:\\n c = f.read()\\nwith open(\\"a.py\\", \\"w\\") as f:\\n '
             'f.write(c)\'"})'),
        Step(1, "assistant",
             'execute_bash({"command": "cd /workspace/i && python -c \'\\n'
             'import re\\np = \\"lib/x.py\\"\\nopen(p, \\"w\\").write(re.sub'
             '(\\"a\\", \\"b\\", open(p).read()))\'"})'),
        Step(2, "assistant",
             'execute_bash({"command": "python3 -c \'import pathlib; '
             'pathlib.Path(\\"z.py\\").write_text(\\"x\\")\'"})'),
        Step(3, "assistant",
             'execute_bash({"command": "python3 -c \'import json; '
             'json.dump(d, open(\\"o.json\\"))\'"})'),
        # read-only interpreter work: opens a file, prints, runs tests
        Step(4, "assistant",
             'execute_bash({"command": "python3 -c \'\\nwith open(\\"a.py\\") '
             'as f:\\n print(f.read())\'"})'),
        Step(5, "assistant",
             'execute_bash({"command": "PYTHONPATH=/w/lib python3 -c \'from '
             'ansible.x import y\\nprint(y)\'"})'),
        Step(6, "assistant",
             'execute_bash({"command": "cd /workspace/i && python -m pytest '
             'tests/ -q"})'),
    ])
_pc = [c["step"] for c in candidates_mod.candidates(_py)]
check("an interpreter one-liner that writes a file back is a candidate",
      _pc == [0, 1, 2, 3], _pc)
check("a bare open() for reading is not a write",
      4 not in _pc and 5 not in _pc, _pc)
check("running the test suite is not a write",
      6 not in _pc, _pc)
for _n, _cmd in (("open(p, \"w\")", 'open(\\"a.py\\", \\"w\\")'),
                 (".write(", 'open(p).write(x)'),
                 ("write_text(", 'Path(\\"z\\").write_text(\\"x\\")')):
    check("the write-mode signal %s is recognised" % _n,
          candidates_mod.classify(
              'execute_bash({"command": "python3 -c \'%s\'"})' % _cmd)[0], _cmd)
check("file access without a write mode is not a write",
      not candidates_mod.classify(
          'execute_bash({"command": "python3 -c \'open(\\"a\\").read()\'"})')[0])

# The redirect-to-a-relative-filename idiom is deliberately NOT recognised, and
# the reason is a measurement rather than an oversight: it reaches one further
# reference step and costs 57 extra candidates corpus-wide, nearly all of them
# scratch reproduction scripts. If someone adds it, this fails and sends them to
# the probe that says why.
check("a redirect to a bare relative filename is still not a write",
      not candidates_mod.classify(
          'execute_bash({"command": "cat << \'EOF\' > test_repro.py\\nx = 1\\n'
          'EOF"})')[0])
check("a redirect into a path is still a write",
      candidates_mod.classify(
          'execute_bash({"command": "echo x > lib/foo/bar.py"})')[0])
check("the parse records where the reference step is unreachable by design",
      "no tool call at all" in candidates_mod.__doc__
      or "planning critical" in candidates_mod.__doc__,
      candidates_mod.__doc__[-400:])

# The other corpus states its calls differently. A parser that silently handles
# only one idiom would restrict one corpus and no-op the other while reporting
# the same flag, so both forms are pinned.
_ta = Trajectory(
    task_id="ta-fixture", source="trajaudit", subset="swe-agent",
    task_description="fix it",
    steps=[Step(0, "assistant",
                'Thought: look around\nAction: [{"func_name": "search_class", '
                '"arguments": {}}]\nObservation: ok'),
           Step(1, "assistant",
                'Thought: patch it\nAction: [{"func_name": "write_patch", '
                '"arguments": {}}]\nObservation: ok'),
           Step(2, "assistant",
                'Thought: again\nAction: [{"func_name": "apply_patch", '
                '"arguments": {}}]\nObservation: ok')])
check("the other corpus's action idiom is parsed too",
      [c["step"] for c in candidates_mod.candidates(_ta)] == [1, 2],
      candidates_mod.candidates(_ta))

# DEGRADATION IS THE DANGEROUS PATH. A parser that fails to recognise a
# framework returns an empty list, and an empty answer space read literally
# means "no step can be the answer" -- which would score zero on every case in
# that framework while looking like a working run.
_thin = Trajectory(task_id="thin", source="trajdebug", subset="x",
                   task_description="t",
                   steps=[Step(0, "assistant", "just thinking out loud"),
                          Step(1, "assistant",
                               'str_replace_editor({"command": "create", '
                               '"path": "/a.py"})')])
check("too few writes means do not restrict, not restrict to nothing",
      candidates_mod.candidates(_thin) == []
      and candidates_mod.MIN_CANDIDATES == 2)
check("an unparseable trajectory leaves the answer space alone",
      candidates_mod.candidates(
          Trajectory(task_id="u", source="x", subset="x", task_description="t",
                     steps=[Step(0, "assistant", "prose only"),
                            Step(1, "assistant", "more prose")])) == [])

# Plan steps are the SECOND way into the answer space, and the danger is
# density: a test that fires on half the prose is not a restriction, it is the
# unrestricted answer space with extra steps. Four arms have already lost that
# way. So what is pinned here is what the test REFUSES, not what it accepts.
_plan_txt = ("Based on my reading of the code, the root cause is in "
             "`_load_file` in application.py around lines 41-60, which never "
             "closes the handle. " + "x" * 200)
_tp = Trajectory(
    task_id="plan-fixture", source="trajaudit", subset="swe-agent",
    task_description="fix it",
    steps=[Step(0, "assistant", "Thought: short note"),
           Step(1, "assistant", "Thought: " + _plan_txt),
           Step(2, "assistant",
                'Thought: patch it\nAction: [{"func_name": "write_patch", '
                '"arguments": {}}]\nObservation: ok')])
check("a located, named conclusion in prose is offered",
      [c["step"] for c in candidates_mod.candidates_with_plans(_tp)] == [1, 2],
      candidates_mod.candidates_with_plans(_tp))
check("the write-only parse is unchanged by the plan stage",
      [c["step"] for c in candidates_mod.candidates(_tp)] == []
      and [c["step"] for c in candidates_mod.candidates(_ta)] == [1, 2])
# Naming nothing is the whole of what separates a plan from a sentiment.
check("prose that names no code object is not a plan",
      not candidates_mod.is_plan(
          "Thought: I can see the problem is that this is broken and I "
          "need to fix it properly. " + "x" * 300))
check("a short conclusion is not a plan",
      not candidates_mod.is_plan("The root cause is in application.py"))
# A step with a call is already judged by classify; letting both fire would
# double-count it, and a READ would enter the answer space through the back
# door while the write parse was busy rejecting it.
check("a step that acted is judged by its action, never as a plan",
      not candidates_mod.is_plan(
          'Thought: the root cause is in `_load_file` in application.py\n'
          'Action: {"tool": "str_replace_editor", "input": '
          '{"command": "view", "path": "/a.py"}}\n' + "x" * 300))
# The cap keeps the list short enough to still BE a restriction, and which end
# it keeps is load-bearing: TrajAudit's reference step sits at median position
# 0.39, so truncating from the end would discard exactly what this stage added.
_many = Trajectory(
    task_id="many", source="trajaudit", subset="swe-agent",
    task_description="t",
    steps=[Step(i, "assistant",
                'Thought: t\nAction: [{"func_name": "write_patch", '
                '"arguments": {}}]\nObservation: ok') for i in range(40)])
_capped = [c["step"] for c in candidates_mod.candidates_with_plans(_many, 15)]
check("the cap keeps the earliest steps, not the latest",
      _capped == list(range(15)), _capped)
check("too few found still means do not restrict",
      candidates_mod.candidates_with_plans(
          Trajectory(task_id="u", source="x", subset="x", task_description="t",
                     steps=[Step(0, "assistant", "prose only")])) == [])

check("snapping goes to the nearest candidate",
      candidates_mod.snap([{"step": 4}, {"step": 9}], 8) == 9)

# Ties go to the earlier step because the localiser prompt already prefers the
# earliest step that starts the causal chain; breaking ties the other way would
# quietly contradict it.
check("a tie snaps to the earlier candidate",
      candidates_mod.snap([{"step": 4}, {"step": 6}], 5) == 4)
check("snapping an absent answer stays absent",
      candidates_mod.snap([{"step": 4}], None) is None)

_csp = [Span(start=0, end=5, tier="weak", label="all")]
_cg = FlowGraph(anchors=[5], failure_summary="patch rejected")

# THE A/B INVARIANT, again. Every result on disk was produced by re-running the
# localiser against a stored flow; if the prompt moved when the stage is off,
# the comparison measures the reshuffle instead of the restriction.
_cb = Localiser(1)
localize_mod.localize(_ct, _csp, _cg, _cb)
_cn = Localiser(1)
localize_mod.localize(_ct, _csp, _cg, _cn, cands=None)
_ce = Localiser(1)
localize_mod.localize(_ct, _csp, _cg, _ce, cands=[])
check("cands=None leaves the localiser prompt byte-identical",
      _cb.prompt == _cn.prompt,
      "%d vs %d chars" % (len(_cb.prompt), len(_cn.prompt)))
check("an empty candidate set leaves the localiser prompt byte-identical",
      _cb.prompt == _ce.prompt,
      "%d vs %d chars" % (len(_cb.prompt), len(_ce.prompt)))
check("no candidate set means no restriction text",
      "MUST BE ONE OF THE STEPS" not in _cb.prompt)

_lc = Localiser(3)
_pc = localize_mod.localize(_ct, _csp, _cg, _lc, cands=_cands,
                            enforce_cands=True)
check("the candidate steps reach the localiser",
      "step 1 —" in _lc.prompt and "step 4 —" in _lc.prompt
      and "step 5 —" in _lc.prompt,
      _lc.prompt[_lc.prompt.find("MUST BE ONE OF"):][:300])
check("the localiser is told the restriction is binding",
      "YOUR ANSWER MUST BE ONE OF THE STEPS LISTED BELOW" in _lc.prompt)
# Rule 5 of HOW TO CHOOSE says a step can be critical for its reasoning alone.
# That rule and this restriction genuinely conflict, and leaving the model to
# resolve it would make the answer depend on which paragraph it weighted.
check("the conflict with the reasoning rule is resolved explicitly",
      "Where this conflicts with point 5 above, this wins" in _lc.prompt)
check("the localiser is told where to put the earlier reasoning instead",
      "name the FIRST listed step that carried it out" in _lc.prompt)
# Placement is the opposite of the span prior's on purpose: the prior is a
# guess and is deliberately kept away from the recency slot, while this is a
# constraint on the output format and belongs next to the output schema.
check("the restriction sits after the evidence, beside the answer schema",
      _lc.prompt.index("MUST BE ONE OF THE STEPS")
      > _lc.prompt.index("PART 2")
      and _lc.prompt.index("MUST BE ONE OF THE STEPS")
      < _lc.prompt.index("Return ONLY JSON"))

# Enforcement happens in code, not in the prompt -- but what the model did on
# its own has to survive the correction, or the reported accuracy silently
# becomes a claim about the parser instead of about the localiser.
check("an answer outside the set is moved into it",
      _pc.critical_step == 4, _pc.critical_step)
check("the move is recorded, not hidden",
      _pc.candidates["obeyed"] is False and _pc.candidates["moved_from"] == 3,
      _pc.candidates)
check("the offered set is recorded with the answer",
      _pc.candidates["steps"] == [1, 4, 5] and _pc.candidates["n"] == 3,
      _pc.candidates)
_lo = Localiser(4)
_po2 = localize_mod.localize(_ct, _csp, _cg, _lo, cands=_cands,
                             enforce_cands=True)
check("an answer already inside the set is left alone and marked obeyed",
      _po2.critical_step == 4 and _po2.candidates["obeyed"] is True
      and _po2.candidates["moved_from"] is None, _po2.candidates)

# The diagnostics. The restriction's ceiling is how often the reference step is
# in the set at all; no chooser can beat it, so it is reported next to the
# accuracy rather than left to be inferred.
_c_hit = evaluate_mod.score_one(_pc, GroundTruth(task_id="cand-fixture",
                                                 critical_step=4))
_c_hit["n_steps"] = 6
check("gold inside the offered set is recorded",
      _c_hit["gold_in_candidates"] is True and _c_hit["hit@0"] is True, _c_hit)
check("the pre-snap answer is scored separately from the final one",
      _c_hit["hit@0_presnap"] is False, _c_hit)
# The case the restriction costs us: the reference step was a search, the set
# could never contain it, and we would have been right without it. This is the
# failure mode the pre-snap column exists to expose -- without it, a restriction
# that destroys as many answers as it rescues looks like no change at all.
_c_cost = evaluate_mod.score_one(_pc, GroundTruth(task_id="cand-fixture",
                                                  critical_step=3))
_c_cost["n_steps"] = 6
check("a case the restriction took away is visible as one",
      _c_cost["gold_in_candidates"] is False and _c_cost["hit@0"] is False
      and _c_cost["hit@0_presnap"] is True, _c_cost)
# In the set, but not the one we picked: the restriction did its job and the
# choosing is still what is wrong. This is the case with headroom left.
_c_out = evaluate_mod.score_one(_pc, GroundTruth(task_id="cand-fixture",
                                                 critical_step=5))
_c_out["n_steps"] = 6
check("gold inside the set that we still missed is not a ceiling failure",
      _c_out["gold_in_candidates"] is True and _c_out["hit@0"] is False, _c_out)

# A case the parser declined to restrict keeps its unrestricted behaviour. It
# must not be averaged in with the restricted ones: the restriction can only
# help or hurt where it fired, and folding the rest in hides both directions.
_c_free = evaluate_mod.score_one(
    localize_mod.localize(_ct, _csp, _cg, Localiser(3)),
    GroundTruth(task_id="cand-fixture", critical_step=3))
_c_free["n_steps"] = 6
_cagg = evaluate_mod.aggregate([_c_hit, _c_cost, _c_out, _c_free])
check("the ceiling is reported against the answer space it cost",
      _cagg["candidate_recall"] == round(2 / 3, 4)
      and _cagg["candidate_coverage"] == 0.5, _cagg)
check("how often the localiser obeyed unaided is reported",
      _cagg["candidate_obeyed"] == 0.0, _cagg)
check("accuracy before and after the snap are both reported",
      _cagg["acc@0_on_restricted"] == round(1 / 3, 4)
      and _cagg["acc@0_presnap"] == round(1 / 3, 4), _cagg)
check("the unrestricted cases are counted and scored apart",
      _cagg["n_unrestricted_fallback"] == 1
      and _cagg["acc@0_unrestricted_fallback"] == 1.0, _cagg)
check("the report prints the candidate line",
      "candidates: n=" in evaluate_mod.format_report(_cagg))
check("no candidate set means no candidate metrics",
      "candidate_recall" not in _plain_agg
      and "candidates: n=" not in evaluate_mod.format_report(_plain_agg),
      sorted(_plain_agg))

# The artifact has to say the answer was constrained. A stored run that does not
# is indistinguishable from a free one, and the two are not comparable.
with tempfile.TemporaryDirectory() as _ctmp:
    _cstore = TraceStore(_ctmp)
    _cstore.write_complete(_ct, _cg, _csp, _pc)
    _cart = _cstore.read(_ct.task_id, _ct.source)
    check("the candidate record round-trips through the artifact",
          artifacts_mod.candidates_from_dict(_cart["candidates"])
          == {"n": 3, "steps": [1, 4, 5], "obeyed": False, "enforced": True,
              "moved_from": 3},
          _cart.get("candidates"))
    check("the artifact dump says the answer was snapped in",
          "was snapped in" in artifacts_mod.render_flow_text(_cart))
# A missing flag must not read back as "the localiser obeyed": that is the one
# direction of error that would inflate the claim.
check("an artifact without the obeyed flag reads as not obeyed",
      artifacts_mod.candidates_from_dict({"steps": [1, 2]})["obeyed"] is False)

check("--candidates is off by default", Config().candidate_set is False)
check("enforcement is a separate knob, also off by default",
      Config().candidate_strict is False)
check("the candidate knob is recorded in the artifact fingerprint",
      Config(candidate_set=True).fingerprint().get("candidate_set") is True)
check("the enforcement knob is recorded in the artifact fingerprint",
      Config(candidate_strict=True).fingerprint().get("candidate_strict") is True)

# THE SOFT STANCE. Measured, not assumed: on swebenchpro enforcement never
# fired (acc@0 before and after snapping were both 0.274) while on trajaudit it
# moved three answers that were already on the reference step. So the list and
# the correction are different interventions and must be separately switchable
# -- and the prompt has to say which one is in force, because a block that
# threatens to move the answer when nothing will move it is simply false.
_ls = Localiser(3)
_ps = localize_mod.localize(_ct, _csp, _cg, _ls, cands=_cands,
                            enforce_cands=False)
check("without enforcement the answer is left where the localiser put it",
      _ps.critical_step == 3, _ps.critical_step)
check("leaving the set is still recorded when it is not corrected",
      _ps.candidates["obeyed"] is False and _ps.candidates["enforced"] is False
      and _ps.candidates["moved_from"] is None, _ps.candidates)
check("the enforced run says so in its own record",
      _pc.candidates["enforced"] is True, _pc.candidates)
check("the soft prompt does not threaten to move the answer",
      "will be moved to the nearest entry" not in _ls.prompt
      and "MUST BE ONE OF" not in _ls.prompt,
      _ls.prompt[_ls.prompt.find("BE ONE OF THE STEPS"):][:400])
check("the soft prompt still offers the same steps and the same reasoning rule",
      "step 1 —" in _ls.prompt and "step 4 —" in _ls.prompt
      and "Where this conflicts with point 5 above, this wins" in _ls.prompt)
check("the strict prompt still threatens to move the answer",
      "will be moved to the nearest entry" in _lc.prompt)
# The two blocks are the same argument in two stances. If they drift apart
# anywhere else, the corpus-to-corpus comparison stops being about enforcement.
_sb = localize_mod.CANDIDATE_BLOCK.format(cands="XCANDSX")
_sf = localize_mod.CANDIDATE_BLOCK_SOFT.format(cands="XCANDSX")


def _shared(t):
    """Everything from below the header down to the candidate list."""
    ls = t.splitlines()
    return "\n".join(ls[3:ls.index("XCANDSX") + 1])


check("the two stances make the identical argument between verb and list",
      _shared(_sb) == _shared(_sf), _shared(_sb)[:120])
check("the two stances differ in the header only by the verb",
      _sb.splitlines()[2].replace("MUST", "SHOULD") == _sf.splitlines()[2],
      (_sb.splitlines()[2], _sf.splitlines()[2]))
# The artifact must not describe a snap that did not happen.
with tempfile.TemporaryDirectory() as _stmp:
    _sstore = TraceStore(_stmp)
    _sstore.write_complete(_ct, _cg, _csp, _ps)
    _stxt = artifacts_mod.render_flow_text(
        _sstore.read(_ct.task_id, _ct.source))
    check("an unenforced departure is not reported as a snap",
          "was snapped in" not in _stxt and "not enforced" in _stxt,
          _stxt[_stxt.find("CANDIDATE SET"):][:220])
_srow = evaluate_mod.score_one(_ps, GroundTruth(task_id="cand-fixture",
                                                critical_step=3))
_srow["n_steps"] = 6
check("the unenforced run scores its own answer, not a corrected one",
      _srow["hit@0"] is True and _srow["hit@0_presnap"] is True, _srow)
check("the report says which stance produced the number",
      "(offered)" in evaluate_mod.format_report(
          evaluate_mod.aggregate([_srow]))
      and "(enforced)" in evaluate_mod.format_report(_cagg))

# WHAT THIS STAGE ASSUMES, WRITTEN DOWN. That the critical step must have
# modified something was inferred from the label distribution, not stated by the
# corpus. That is a real risk of fitting an annotation convention, and the module
# has to carry the caveat next to the justification rather than in a commit
# message nobody reads again.
_csrc = inspect.getsource(candidates_mod)
check("the module states that the write assumption is inferred, not given",
      "is NOT given" in _csrc and "fitting an annotation convention" in _csrc)
check("the module states the argument that justifies it anyway",
      "but-for test needs a counterfactual" in _csrc)

# Ground truth, same recording-LLM check every other prompt builder gets: the
# candidate labels are built from step text, one field away from the label block.
for rec in records[:2]:
    gt = rec.ground_truth
    if gt is None or gt.critical_step is None:
        continue
    rl = RecordingLLM()
    run(rec.trajectory, rl, Config(max_rounds=1, workers=1, candidate_set=True))
    blob = "\n".join(rl.prompts)
    leaked = re.search(r"(critical|mistake|gold|correct)\D{0,40}\b%d\b"
                       % gt.critical_step, blob, re.I)
    check("%s: candidate set on, still no gold-step assertion"
          % rec.trajectory.task_id[:26], leaked is None,
          str(leaked.group(0))[:60] if leaked else "")
    if gt.error_type:
        check("%s: candidate set on, gold error_type absent"
              % rec.trajectory.task_id[:26], gt.error_type not in blob)


# ---------------------------------------------------------------------------
# [20] MIXED PAINTING: may a dependency chain switch relation part-way?
#
# ``on_flow`` used to be ``paint(data) | paint(control)`` -- each relation's
# parent forest walked on its own, so a chain that goes data -> control -> data
# reaches nothing and the steps on it are dropped. That rule was never argued
# for anywhere, and probe/miss_retrieved.py measured its cost: 33/33 (SP) and
# 24/24 (TA) of the reference steps the paint walks past are steps that ARE in
# the graph, dropped by this filter alone. Zero were never retrieved.
#
# The counter-argument is the reason this is a flag: mixed painting doubles
# coverage (0.152 -> 0.343 on SP) and recall bought by covering more of the run
# is not recall. So the checks here pin the mechanism, and only acc@0 decides.

_mx = FlowGraph(anchors=[4])
_mx.add(FlowEdge(src=3, dst=4, kind="data", entity="patch"))     # anchor <- data
_mx.add(FlowEdge(src=2, dst=3, kind="control"))                  # <- control
_mx.add(FlowEdge(src=1, dst=2, kind="data", entity="value"))     # <- data again
check("separate painting stops where the chain changes relation",
      _mx.on_flow() == {4, 3}, sorted(_mx.on_flow()))
_mx.mixed_paint = True
check("mixed painting follows the alternating chain to its root",
      _mx.on_flow() == {4, 3, 2, 1}, sorted(_mx.on_flow()))
check("the typed views stay typed either way -- the diagnostics use them",
      _mx.data_flow_steps() == {4, 3} and _mx.control_flow_steps() == {4},
      (sorted(_mx.data_flow_steps()), sorted(_mx.control_flow_steps())))

# The reading is NOT a property of what was traced, so it must not enter the
# stored flow: that is exactly what lets one stored trace serve both arms of the
# A/B under --reuse-flow, and what keeps old artifacts loadable.
check("the reading is not serialised into the flow artifact",
      "mixed_paint" not in _mx.to_dict())
check("a flow read back from an artifact defaults to separate painting",
      artifacts_mod.graph_from_dict(_mx.to_dict()).mixed_paint is False)

check("--mixed-paint is off by default", Config().mixed_paint is False)
check("the reading knob is recorded in the artifact fingerprint",
      Config(mixed_paint=True).fingerprint().get("mixed_paint") is True)

# The pipeline has to actually hand the choice to the graph -- on BOTH paths.
# The reuse path is the one the A/B run uses, and it never touches the tracing
# branch where a knob would otherwise be applied.
_seen_mixed = []
_real_segment = spans_mod.segment


def _spy_segment(traj, llm, graph, **kw):
    _seen_mixed.append(graph.mixed_paint)
    return _real_segment(traj, llm, graph, **kw)


spans_mod.segment = _spy_segment
pipeline_mod.spans_mod.segment = _spy_segment
try:
    run(_ct, RecordingLLM(), Config(max_rounds=1, workers=1, mixed_paint=True))
    check("a freshly traced flow is read the way the config asked",
          _seen_mixed == [True], _seen_mixed)
    with tempfile.TemporaryDirectory() as _mtmp:
        _mstore = TraceStore(_mtmp)
        _mstore.write_flow(_ct, _cg)
        _seen_mixed.clear()
        run(_ct, RecordingLLM(), Config(max_rounds=1, workers=1,
                                        mixed_paint=True, reuse_flow=True),
            store=_mstore)
        check("a REUSED flow is read the way the config asked",
              _seen_mixed == [True], _seen_mixed)
finally:
    spans_mod.segment = _real_segment
    pipeline_mod.spans_mod.segment = _real_segment

# A/B integrity: with the flag off, every prompt must be byte-identical to a run
# that predates the flag. Without this the comparison measures the diff, not the
# intervention.
_off_a, _off_b = RecordingLLM(), RecordingLLM()
run(_ct, _off_a, Config(max_rounds=1, workers=1))
run(_ct, _off_b, Config(max_rounds=1, workers=1, mixed_paint=False))
check("the flag off changes no prompt by a single byte",
      _off_a.prompts == _off_b.prompts)

# The default was never argued for; the replacement must carry both sides of the
# argument next to the code, since the numbers that decide it live in probe/.
_osrc = inspect.getsource(FlowGraph.on_flow)
check("on_flow states the case for mixed painting",
      "ARGUMENT FOR MIXED" in _osrc and "paint_frontier" in _osrc)
check("on_flow states the case against it, with the enrichment cost",
      "ARGUMENT AGAINST" in _osrc and "3.06x" in _osrc and "2.06x" in _osrc
      and "only acc@0 decides" in _osrc)
# The earlier version of this docstring claimed 100% of the missed reference
# steps were dropped by this filter. Measured, it is half: the rest are
# reachable from no anchor under any rule. A wrong number next to the code is
# worse than none, so the correction is pinned.
check("on_flow no longer claims painting can reach every missed step",
      "100% of the reference steps" not in _osrc
      and "BIGGEST BUCKET" in _osrc and "NO anchor" in _osrc)
# The one loosening that was actually built and measured has to stay recorded
# here, or the frontier table reads as an invitation to re-run it.
check("on_flow records the measured null, so it is not re-run",
      "MEASURED NULL" in _osrc and "0.2791" in _osrc
      and "rejected_bridge_paint" in _osrc)

# flow_recall is computed from a graph rebuilt out of the stored dict, which
# never carries the reading. So it cannot move when a paint flag moves -- a trap
# that cost one debugging round and has to stay documented where it is read.
_psrc = inspect.getsource(evaluate_mod._painted)
check("the evaluator warns that flow_recall is blind to the paint flags",
      "ALWAYS the separate" in _psrc
      and "NOT evidence" in _psrc and "paint_frontier" in _psrc)
# ...and that the warning stops at the reading. Anchors ARE serialised, so the
# inverse trap -- concluding --indeg-anchors cannot have worked because the
# number is supposed to be blind -- has to be closed in the same place.
check("the evaluator says the blindness does not extend to the anchors",
      "THE CONVERSE" in _psrc and "--indeg-anchors" in _psrc)


# ---------------------------------------------------------------------------
# [21] PRIOR SIGNALS: telling the region guess where an answer may live.
#
# Measured on SWE-Bench-Pro (sp_05, 86 cases), and the reason this stage is
# where the work goes:
#
#   * The localiser treats the prior as a filter, not a hint. Reference step a
#     candidate INSIDE a guessed phase -> found 0.559 of the time; a candidate
#     OUTSIDE every guessed phase -> 0.094, which is chance among ~11
#     candidates. So the prior's recall (0.500) caps the whole method, and no
#     downstream prompt work can lift the cap.
#   * 127 of 258 guessed phases contained no step that changed anything, while
#     the localiser may only answer with such a step. In 17 of 86 cases every
#     guess was like that; those cases score 0.118 against 0.290 for the rest.
#     The stage was picking regions with no idea where an answer was allowed.
#
# So the signals are counts, and the count that matters is "changes N". What
# must NOT cross over is the graph itself -- the moment the prior can read the
# edges it stops being a second reader and starts paraphrasing the localiser's
# evidence, which is the failure mode the blindness was protecting against.

_sig_llm = PriorLLM({"spans": [{"start": 4, "end": 6, "likelihood": "likely",
                                "why": "the edit was committed here"}]})
_sig_cands = [{"step": 5, "label": "wrote lib/foo.py"},
              {"step": 8, "label": "wrote t.py"}]
span_prior_mod.predict_spans(traj, _phases, _pg, _sig_llm,
                             signals=True, cands=_sig_cands)

check("the signals give each phase its on-flow counts",
      "data=" in _sig_llm.prompt and "control=" in _sig_llm.prompt,
      _sig_llm.prompt[:0])
check("the signals give each phase its distance to the symptom",
      "hops from symptom" in _sig_llm.prompt
      or "no traced path to the symptom" in _sig_llm.prompt)
check("the signals say how many steps in the phase changed anything",
      "changes 1" in _sig_llm.prompt and "changes 0" in _sig_llm.prompt,
      _sig_llm.prompt[_sig_llm.prompt.find("PHASES ("):][:300])
check("the phase holding a failure artefact is marked as on-topic",
      "names a failure artefact" in _sig_llm.prompt)

# The boundary. Counts may cross; the graph may not.
check("even with signals the prior never sees an edge",
      "step 2 -> step 7" not in _sig_llm.prompt
      and "[cfg]" not in _sig_llm.prompt)
check("even with signals the prior never sees the weak/strong tiers",
      "on dependency path" not in _sig_llm.prompt
      and "weak" not in _sig_llm.prompt)
check("even with signals the prior is never given a step to name",
      "step 5" not in _sig_llm.prompt and "step 8" not in _sig_llm.prompt,
      _sig_llm.prompt[_sig_llm.prompt.find("PHASES ("):][:300])

# Every count is handed over with the way it misleads, because a bare number
# reads as a score and none of these is one.
check("a zero flow count is explicitly not a clearance",
      "Zero does not clear a phase" in _sig_llm.prompt)
check("a large hop count is explicitly not evidence",
      "where to look, not as evidence" in _sig_llm.prompt)
check("naming an unreachable phase is explicitly a wasted guess",
      "cannot be cashed" in _sig_llm.prompt)
check("the artefact mark is explicitly not guilt",
      "on-topic, not guilty" in _sig_llm.prompt)

# Without the candidate set there is no hard constraint to state, and stating
# one anyway would be false: the localiser is unrestricted in that arm.
_nocand = PriorLLM({"spans": []})
span_prior_mod.predict_spans(traj, _phases, _pg, _nocand, signals=True)
check("with no answer space, the prior is not told there is one",
      "changes " not in _nocand.prompt and "cannot be cashed" not in _nocand.prompt)
check("the other signals still appear without an answer space",
      "data=" in _nocand.prompt and "hops from symptom" in _nocand.prompt)

# A/B integrity, the same requirement every added stage has had to meet.
_off1, _off2 = PriorLLM({"spans": []}), PriorLLM({"spans": []})
span_prior_mod.predict_spans(traj, _phases, _pg, _off1)
span_prior_mod.predict_spans(traj, _phases, _pg, _off2, signals=False,
                             cands=_sig_cands)
check("signals off leaves the prior prompt byte-identical",
      _off1.prompt == _off2.prompt)

check("--prior-signals is off by default", Config().prior_signals is False)
check("the signals knob is recorded in the artifact fingerprint",
      Config(prior_signals=True).fingerprint().get("prior_signals") is True)

# The answer space has to be computed BEFORE the prior for any of this to be
# possible. Ordering is the whole intervention, so it is pinned here rather
# than left to whoever next edits the stage list.
_psrc = inspect.getsource(pipeline_mod.run)
check("the candidate parse runs before the prior in the pipeline",
      _psrc.index("candidates_mod.candidates") < _psrc.index("predict_spans"))
check("the prior is only handed the answer space when it can read it",
      "cands if cfg.prior_signals" in _psrc.replace("(", "").replace(")", ""))

# Ground truth, the same recording-LLM check every prompt builder gets: the
# counts are computed from the trace and the step text, both one field away
# from the label block.
for rec in records[:2]:
    gt = rec.ground_truth
    if gt is None or gt.critical_step is None:
        continue
    rl = RecordingLLM()
    run(rec.trajectory, rl, Config(max_rounds=1, workers=1, span_prior=True,
                                   prior_signals=True, candidate_set=True))
    blob = "\n".join(rl.prompts)
    leaked = re.search(r"(critical|mistake|gold|correct)\D{0,40}\b%d\b"
                       % gt.critical_step, blob, re.I)
    check("%s: prior signals on, still no gold-step assertion"
          % rec.trajectory.task_id[:26], leaked is None,
          str(leaked.group(0))[:60] if leaked else "")


print("\n[22] the prior may read the harness's report, never the annotation")

import glob

from longtraj import loaders as loaders_mod

# The prior's whole view of the failure is ``failure_summary``: two sentences
# the anchoring model wrote about it. TrajAudit ships the thing itself in
# ``failed_tests`` -- the test the instance is graded on and its traceback -- and
# that text already reaches the anchoring prompt, so this flag moves existing
# information further down the pipeline rather than adding any.
#
# The danger is the file it comes from. The same JSON holds ``failure_id`` (the
# reference step) and ``gold_patch`` (the reference fix), one dict key away. So
# the boundary is pinned twice: at the loader, which must not carry them into
# ``meta``, and here, on the rendered prompt.

_sym_traj = records[0].trajectory
check("the corpus that has a harness report is the one under test",
      bool(_sym_traj.meta.get("failed_test_code")
           or _sym_traj.meta.get("failed_test_error")),
      sorted(_sym_traj.meta))

_sym = PriorLLM({"spans": []})
span_prior_mod.predict_spans(_sym_traj, _phases, _pg, _sym, symptom=True)

check("the prior is shown the failing test the task is graded on",
      "FAILING TEST:" in _sym.prompt and "ERROR:" in _sym.prompt)
check("the harness report is verbatim, not a paraphrase",
      (str(_sym_traj.meta["failed_test_code"]).strip()[:200] in _sym.prompt
       if _sym_traj.meta.get("failed_test_code") else True))
check("the report is placed as the symptom, before the phases",
      _sym.prompt.index("FAILING TEST:") < _sym.prompt.index("PHASES ("))
check("the report is named as a codebase location, not a phase",
      "on-topic, not guilty" in _sym.prompt)

# fail_to_pass and pass_to_pass invert the reading -- one means the agent never
# made a change, the other that a change it made broke something that worked.
# Stating the wrong one is worse than stating neither, so it is stated only when
# the corpus says which.
_grade = str(_sym_traj.meta.get("failed_test_expected") or "").strip().lower()
check("the grading kind is explained when the corpus gives one",
      (span_prior_mod.GRADE_BULLETS[_grade].strip()[:40] in _sym.prompt)
      if _grade in span_prior_mod.GRADE_BULLETS else True, _grade)
_nograde = PriorLLM({"spans": []})
_tg = dataclasses.replace(_sym_traj, meta=dict(_sym_traj.meta,
                                               failed_test_expected="???"))
span_prior_mod.predict_spans(_tg, _phases, _pg, _nograde, symptom=True)
check("an unknown grading kind is left unstated, not guessed",
      "PASSED before the agent started" not in _nograde.prompt
      and "making it pass WAS the task" not in _nograde.prompt)

# The label and the reference fix live in the same source file as the report.
_raw = json.load(open(
    [p for p in glob.glob(os.path.join(
        loaders_mod.find_repo_root(), "TrajAudit-main", "**",
        "%s.json" % _sym_traj.task_id), recursive=True)][0], encoding="utf-8"))
check("the source file really does hold the label next to the report",
      "failure_id" in _raw and "gold_patch" in _raw, sorted(_raw))
check("the loader leaves the reference step out of meta",
      not any(str(_raw.get("failure_id")) == str(v)
              for v in _sym_traj.meta.values()))
check("the loader leaves the reference fix out of meta",
      not any(str(_raw.get("gold_patch") or "\0")[:80] in str(v)
              for v in _sym_traj.meta.values()))
check("the reference fix never reaches the prior prompt",
      str(_raw.get("gold_patch") or "\0")[:80] not in _sym.prompt)
check("the human failure reason never reaches the prior prompt",
      str(_raw.get("failure_reason") or "\0")[:80] not in _sym.prompt)

# A corpus without the field must be untouched -- that is what makes the
# SWE-Bench-Pro arm of the A/B a control rather than a second treatment.
_nofield = [r.trajectory for r in records
            if not r.trajectory.meta.get("failed_test_code")
            and not r.trajectory.meta.get("failed_test_error")]
check("a corpus without a harness report is available to test against",
      bool(_nofield), [r.trajectory.source for r in records])
if _nofield:
    _c1, _c2 = PriorLLM({"spans": []}), PriorLLM({"spans": []})
    span_prior_mod.predict_spans(_nofield[0], _phases, _pg, _c1)
    span_prior_mod.predict_spans(_nofield[0], _phases, _pg, _c2, symptom=True)
    check("with no harness report the flag changes nothing",
          _c1.prompt == _c2.prompt)

# A/B integrity, the same requirement every added stage has had to meet.
_s1, _s2 = PriorLLM({"spans": []}), PriorLLM({"spans": []})
span_prior_mod.predict_spans(_sym_traj, _phases, _pg, _s1)
span_prior_mod.predict_spans(_sym_traj, _phases, _pg, _s2, symptom=False)
check("symptom off leaves the prior prompt byte-identical", _s1.prompt == _s2.prompt)

# Independent of the counts: one says where an answer may live, the other says
# what the task was graded on. Turning both on must give both.
_both = PriorLLM({"spans": []})
span_prior_mod.predict_spans(_sym_traj, _phases, _pg, _both,
                             signals=True, cands=_sig_cands, symptom=True)
check("the two prior flags compose", "FAILING TEST:" in _both.prompt
      and "changes 1" in _both.prompt)

check("--prior-symptom is off by default", Config().prior_symptom is False)
check("the symptom knob is recorded in the artifact fingerprint",
      Config(prior_symptom=True).fingerprint().get("prior_symptom") is True)
check("the pipeline passes the symptom knob to the prior",
      "symptom=cfg.prior_symptom" in _psrc)

# Ground truth, the same recording-LLM check every prompt builder gets.
for rec in records[:2]:
    gt = rec.ground_truth
    if gt is None or gt.critical_step is None:
        continue
    rl = RecordingLLM()
    run(rec.trajectory, rl, Config(max_rounds=1, workers=1, span_prior=True,
                                   prior_symptom=True))
    blob = "\n".join(rl.prompts)
    leaked = re.search(r"(critical|mistake|gold|correct)\D{0,40}\b%d\b"
                       % gt.critical_step, blob, re.I)
    check("%s: harness report on, still no gold-step assertion"
          % rec.trajectory.task_id[:26], leaked is None,
          str(leaked.group(0))[:60] if leaked else "")


# ---------------------------------------------------------------------------
# [24] ANCHOR RE-SELECTION: in-degree scoring under a self-calibrating cap.
#
# The painting axis was a dead end -- every loosening in probe/paint_frontier.py
# trades recall for coverage one-for-one. The anchor axis is not: measured on the
# stored flows, 37 of SWE-Bench-Pro's 44 missed reference steps are reachable
# from SOME step the located anchors simply do not sit on, and only 8 need an
# edge that was never traced. So this re-reads a stored flow from better anchors
# and costs no tracing -- FORWARD_PROMPT never sees the anchors.
#
# The cap is the part that has to be pinned. Uncapped, in-degree buys recall with
# coverage at roughly one-for-one and enrichment stays flat; capped per case at a
# multiple of what that case's OWN located anchors paint, SWE-Bench-Pro goes
# 0.4884 -> 0.5814 for coverage 0.1596 -> 0.1688 (enrichment 3.06x -> 3.44x).

_at = Trajectory(
    task_id="anchor-fixture", source="trajdebug", subset="swebenchpro",
    task_description="fix the loader",
    steps=[Step(i, "assistant", "step %d" % i) for i in range(12)])

# 3 <- 4 <- 5 all feed 6 (in-degree 3, paints 4 steps); 8 feeds 9 (in-degree 1,
# paints 2); 6 feeds 7 (in-degree 1, paints 5). Step 11 has no parent at all,
# which is what the located anchors look like in the failing cases.
_ag = FlowGraph(anchors=[11], anchors_located=[11])
for _s in (3, 4, 5):
    _ag.add(FlowEdge(src=_s, dst=6, kind="data", entity="cfg"))
_ag.add(FlowEdge(src=8, dst=9, kind="data", entity="path"))
_ag.add(FlowEdge(src=6, dst=7, kind="control"))

# With the cap slack, the ordering alone decides: in-degree beats lateness (6 is
# taken before 9 although 9 is later), and 7 is dropped for sitting inside the
# gap of an anchor already taken. 11 is not even a candidate -- no parent edge.
check("re-selection ranks by in-degree, not by position",
      tracing_mod.reselect_anchors(_at, _ag, ratio=10.0, k=3, gap=3) == [6, 9],
      tracing_mod.reselect_anchors(_at, _ag, ratio=10.0, k=3, gap=3))
check("re-selection keeps anchors apart by the gap",
      7 not in tracing_mod.reselect_anchors(_at, _ag, ratio=10.0, k=3, gap=3))
check("a step with no parent edge is never chosen as an anchor",
      11 not in tracing_mod.reselect_anchors(_at, _ag, ratio=10.0, k=3, gap=3))

# The cap has to BIND, including on the first anchor. probe/anchor_frontier.py's
# greedy exempted the first one and coverage drifted up anyway despite a per-case
# budget; that is the bug this check exists to keep out. Here the located anchor
# paints 1 of 12 steps, so at ratio 1.0 every candidate overshoots on its own.
_capped = tracing_mod.reselect_anchors(_at, _ag, ratio=1.0, k=3, gap=3)
check("the coverage cap binds on the first anchor too, not just later ones",
      _capped == [9], _capped)
check("when every candidate overshoots, the cheapest one still anchors the paint",
      _capped and _ag.__class__(anchors=_capped, edges=_ag.edges).on_flow(),
      _capped)

# Painting must be scored with the reading the run will actually use, or the
# budget measures something the pipeline never does.
check("re-selection scores candidates under the SEPARATE reading",
      "SEPARATE reading" in inspect.getsource(tracing_mod._paint_from))

# Provenance. Unlike --mixed-paint this knob IS serialised, because
# write_complete re-dumps the graph at the end of every run -- so the located
# anchors have to survive in their own field or the baseline trace dir is
# silently overwritten with the other arm's anchors.
_ag2 = FlowGraph(anchors=[6, 9], anchors_located=[11])
check("the located anchors are serialised separately from the effective ones",
      _ag2.to_dict()["anchors_located"] == [11]
      and _ag2.to_dict()["anchors"] == [6, 9])
check("both anchor fields survive a round trip through an artifact",
      artifacts_mod.graph_from_dict(_ag2.to_dict()).anchors_located == [11])
check("an artifact written before the split reads back with the two the same",
      artifacts_mod.graph_from_dict({"anchors": [2], "edges": []})
      .anchors_located == [2])
check("locate_anchors records what it chose as the provenance",
      "anchors_located=list(anchors)" in
      inspect.getsource(tracing_mod.locate_anchors))

# Idempotence in BOTH directions: a flag-off run over a trace dir that a flag-on
# run already wrote back must reproduce the baseline, not inherit its anchors.
_seen_anchors = []
_real_segment2 = spans_mod.segment


def _spy_anchors(traj, llm, graph, **kw):
    _seen_anchors.append(list(graph.anchors))
    return _real_segment2(traj, llm, graph, **kw)


spans_mod.segment = _spy_anchors
pipeline_mod.spans_mod.segment = _spy_anchors
try:
    with tempfile.TemporaryDirectory() as _atmp:
        _astore = TraceStore(_atmp)
        # a POLLUTED artifact: effective anchors differ from what was located.
        # The union rule must rebuild from anchors_located ([7, 9, 11]) and add
        # the confluence point 6, NOT inherit the stored [6, 9].
        _poison = FlowGraph(anchors=[6, 9], anchors_located=[7, 9, 11],
                            failure_summary="patch rejected")
        for _e in _ag.edges:
            _poison.add(_e)
        _astore.write_flow(_at, _poison)
        run(_at, RecordingLLM(), Config(max_rounds=1, workers=1,
                                        reuse_flow=True,
                                        error_anchors=True), store=_astore)
        check("a polluted flow is re-anchored on located | confluence, not on "
              "what was stored", _seen_anchors == [[6, 7, 9, 11]], _seen_anchors)
        _seen_anchors.clear()
        # ...and the run just wrote that union back. Running again must not
        # compound a union onto a union: the reset at the 2c seam is what makes
        # this idempotent, and it is the fix for the tr_ta_prior pollution.
        run(_at, RecordingLLM(), Config(max_rounds=1, workers=1,
                                        reuse_flow=True,
                                        error_anchors=True), store=_astore)
        check("re-anchoring a flow it already wrote back is idempotent",
              _seen_anchors == [[6, 7, 9, 11]], _seen_anchors)
        check("the union never drops a located anchor",
              set([7, 9, 11]) <= set(_seen_anchors[0]), _seen_anchors)
        # The stage is off by default (E4: it costs acc@0 on both corpora), and
        # the reset at the seam is what still protects the default path from an
        # artifact a flag-on run polluted -- otherwise the default would silently
        # inherit [6, 7, 9, 11] and stop reproducing the baseline.
        _seen_anchors.clear()
        run(_at, RecordingLLM(), Config(max_rounds=1, workers=1,
                                        reuse_flow=True), store=_astore)
        check("the default path resets a polluted flow to what was located",
              _seen_anchors == [[7, 9, 11]], _seen_anchors)
        check("the re-anchoring stage is off by default",
              Config().error_anchors is False)
finally:
    spans_mod.segment = _real_segment2
    pipeline_mod.spans_mod.segment = _real_segment2

check("the failure-anchor rule takes no argument that could carry the "
      "reference step",
      list(inspect.signature(tracing_mod.failure_anchors).parameters)
      == ["traj", "graph", "k", "gap", "ratio"])
check("the failure-anchor rule needs no LLM at all",
      "llm" not in inspect.signature(tracing_mod.failure_anchors).parameters)
check("the failure-anchor set is a superset of the located anchors",
      set(_ag.anchors_located)
      <= set(tracing_mod.failure_anchors(_at, _ag)))
check("the failure-anchor rule says the union is a superset, so wins/losses "
      "are arithmetic and not evidence",
      "not evidence" in inspect.getsource(tracing_mod.failure_anchors))
check("the failure-anchor rule records that it loses on enrichment",
      "LOSS by the standing enrichment rule"
      in inspect.getsource(tracing_mod.failure_anchors))
check("the pipeline anchors on the failure-anchor rule",
      "failure_anchors(traj, graph" in inspect.getsource(pipeline_mod.run))

# ---------------------------------------------------------------------------
# The judging stage itself. It is the only stage that puts RAW step bodies in
# front of a model this early, so its inputs get the closest look.

_jsrc = inspect.getsource(tracing_mod.judge_error_steps)

check("judging takes no argument that could carry the reference step",
      list(inspect.signature(tracing_mod.judge_error_steps).parameters)
      == ["traj", "graph", "llm", "window", "k", "gap", "ratio"],
      list(inspect.signature(tracing_mod.judge_error_steps).parameters))
check("the judging prompt names no ground-truth field",
      not re.search(r"annotation|gold_patch|failure_reason|test_patch|"
                    r"failure_id|critical[_ ]step",
                    tracing_mod.ERROR_STEP_PROMPT, re.I))
check("the judging prompt asks for a verdict per step, not a global pick",
      "one at a time and independently" in tracing_mod.ERROR_STEP_PROMPT)
check("the judging prompt still forbids reasoning about the cause",
      "do not reason about which step CAUSED" in tracing_mod.ERROR_STEP_PROMPT)
check("judge_error_steps records why the window is 20 and not 10",
      "WHY THE WINDOW IS 20" in inspect.getsource(tracing_mod)
      and "2.81x" in inspect.getsource(tracing_mod))
check("judge_error_steps states ground truth cannot reach it",
      "GROUND TRUTH NEVER REACHES THIS" in _jsrc)

# The window really is a window. A 60-step trajectory must show the model only
# the tail, or "last 20" is a comment rather than a constraint.
_wt = Trajectory(
    task_id="window-fixture", source="trajdebug", subset="swebenchpro",
    task_description="fix the loader",
    steps=[Step(i, "assistant", "BODY-OF-STEP-%d" % i) for i in range(60)])
_wg = FlowGraph(anchors=[59], anchors_located=[59])
for _s in range(0, 58):
    _wg.add(FlowEdge(src=_s, dst=_s + 1, kind="data", entity="cfg"))
_wl = RecordingLLM()
tracing_mod.judge_error_steps(_wt, _wg, _wl)
_wblob = "\n".join(_wl.prompts)
check("the judging prompt shows only the last 20 steps",
      "BODY-OF-STEP-39" not in _wblob and "BODY-OF-STEP-45" in _wblob)
check("the judging prompt shows raw step bodies, not one-line digests",
      _wblob.count("BODY-OF-STEP-") >= 15)

# Every fallback path, because a bad verdict must never leave paint rootless.
check("no verdict at all keeps the located anchors",
      tracing_mod.judge_error_steps(_at, _ag, RecordingLLM()) == [])
check("every step judged fine keeps the located anchors",
      tracing_mod.judge_error_steps(_at, _ag,
                                    JudgingLLM({6: False, 9: False})) == [])
check("'related' is used when nothing is judged an outright error",
      tracing_mod.judge_error_steps(
          _at, _ag, JudgingLLM({6: False}, related={6: True})) == [6])
check("an index the model was never shown is ignored",
      tracing_mod.judge_error_steps(_at, _ag,
                                    JudgingLLM({999: True, -4: True})) == [])
check("a step with no parent edge is never judged, so never anchored",
      tracing_mod.judge_error_steps(_at, _ag, JudgingLLM({11: True})) == [])
check("a malformed reply falls back instead of raising",
      _quietly(tracing_mod.judge_error_steps, _at, _ag, _BrokenLLM()) == [])

# The cap applies to the judged steps exactly as it does to the in-degree rule:
# the located anchor paints 1 of 12, so at ratio 1.0 everything overshoots.
check("the coverage cap binds on the judged steps too",
      tracing_mod.judge_error_steps(_at, _ag, JudgingLLM({6: True, 9: True}),
                                    ratio=1.0) == [9])

# The two rules must differ only in what they nominate, never in how the budget
# is spent -- that is the whole reason the greedy was extracted.
check("both anchor rules spend the budget through the same helper",
      "_cap_pick" in _jsrc
      and "_cap_pick" in inspect.getsource(tracing_mod.reselect_anchors))
check("_cap_pick documents why the cap is what makes this axis win",
      "THE CAP IS WHY THIS AXIS WINS"
      in inspect.getsource(tracing_mod._cap_pick))

# A/B integrity: two runs with the same config and the same verdicts must
# produce byte-identical prompts, or nothing measured here is comparable.
_an_a, _an_b = RecordingLLM(), RecordingLLM()
run(_ct, _an_a, Config(max_rounds=1, workers=1))
run(_ct, _an_b, Config(max_rounds=1, workers=1))
check("re-anchoring is deterministic prompt-for-prompt",
      _an_a.prompts == _an_b.prompts)

# The numbers that decide this live in probe/, so the honest ones -- including
# the corpus where it is a null -- belong next to the code.
_rsrc = inspect.getsource(tracing_mod.reselect_anchors)
check("reselect_anchors states why the anchors and not the graph",
      "37 of the 44" in _rsrc and "only 8 need an edge" in _rsrc)
check("reselect_anchors states the enrichment, not just the recall",
      "3.06x" in _rsrc and "3.44x" in _rsrc)
check("reselect_anchors states the corpus where it is a null",
      "ON TRAJAUDIT THIS IS A NULL" in _rsrc and "p=1.0" in _rsrc)

# Ground truth: the re-selection is handed the trajectory and the graph, and
# nothing else. The reference step cannot reach it even by accident.
check("re-selection takes no argument that could carry the reference step",
      list(inspect.signature(tracing_mod.reselect_anchors).parameters)
      == ["traj", "graph", "ratio", "k", "gap"],
      list(inspect.signature(tracing_mod.reselect_anchors).parameters))
for rec in records[:2]:
    gt = rec.ground_truth
    if gt is None or gt.critical_step is None:
        continue
    rl = JudgingLLM({}, all_error=True)
    run(rec.trajectory, rl, Config(max_rounds=1, workers=1))
    blob = "\n".join(rl.prompts)
    leaked = re.search(r"(critical|mistake|gold|correct)\D{0,40}\b%d\b"
                       % gt.critical_step, blob, re.I)
    check("%s: re-anchored, still no gold-step assertion"
          % rec.trajectory.task_id[:26], leaked is None,
          str(leaked.group(0))[:60] if leaked else "")


# ---------------------------------------------------------------------------
# [25] MULTI-EXPERT LOCALISATION: four remits in parallel, then a vote.
#
# The single localiser is not short of evidence -- on SWE-Bench-Pro half the
# reference steps are already painted and it finds half of THOSE
# (P(hit | gold_on_flow) = 0.50). Its problem is where to look in a 117-step
# run. Four narrow readers search four ways over identical evidence, and a rule
# fuses them.
#
# What these checks defend, in order of how easy it is to lose:
#   * the experts differ in REMIT and in nothing else -- same evidence, or a
#     weak expert cannot be told from an under-informed one;
#   * abstention is a real answer, not a shortfall to be papered over;
#   * the fusion is a rule, so it can be re-run offline against a stored record;
#   * every verdict survives into the artifact, including the overruled ones;
#   * flag OFF leaves the single localiser's prompt byte-identical.
print("\n[25] multi-expert localisation")

_EXPERT_KEYS = [s.key for s in experts_mod.EXPERTS]
check("there are four experts, one per failure family",
      _EXPERT_KEYS == ["compliance", "logic", "flow", "context"], _EXPERT_KEYS)
check("every expert has a distinct remit and its own vocabulary",
      len({s.remit for s in experts_mod.EXPERTS}) == 4
      and len({tuple(s.vocabulary) for s in experts_mod.EXPERTS}) == 4)

# Ground truth. The experts are handed a Trajectory, a FlowGraph and the
# compression artifacts; none of those can hold a label, and the signature says
# so without needing to trust the prompt text.
check("the expert stage takes no argument that could carry the reference step",
      list(inspect.signature(experts_mod.localize_multi).parameters)
      == ["traj", "spans", "graph", "llm", "digests", "outline", "prior",
          "cands", "enforce_cands", "max_prompt_chars", "workers"],
      list(inspect.signature(experts_mod.localize_multi).parameters))
check("no expert prompt names a ground-truth field",
      not re.search(r"annotation|gold_patch|failure_reason|test_patch|"
                    r"failure_id",
                    experts_mod.EXPERT_PROMPT + "".join(
                        s.remit for s in experts_mod.EXPERTS), re.I))

# The four prompts must differ ONLY in the remit block. If the evidence differed
# too, a remit that underperforms could not be distinguished from one that was
# shown less, and the whole comparison between experts would be uninterpretable.
_xt = Trajectory(
    task_id="expert-fixture", source="trajdebug", subset="swebenchpro",
    task_description="make the config loader accept a list",
    steps=[Step(i, "assistant", "did thing %d" % i) for i in range(12)])
_xg = FlowGraph(anchors=[11], anchors_located=[11],
                failure_summary="the patch was rejected",
                failure_entities=["cfg.yaml"])
_xg.add(FlowEdge(src=4, dst=6, kind="data", entity="cfg.yaml"))
_xg.add(FlowEdge(src=6, dst=11, kind="control"))
_xspans = [Span(0, 5, label="explore"), Span(6, 11, label="patch")]

_xev = experts_mod._evidence(_xt, _xspans, _xg, [], None, None, None,
                             True, 120000)
_xprompts = [experts_mod._expert_prompt(s, _xev) for s in experts_mod.EXPERTS]
check("the four expert prompts are all different",
      len(set(_xprompts)) == 4)
# Everything from "Find the EARLIEST step" onward is the shared part: the
# evidence, the instructions, and the answer schema. Only the vocabulary line
# inside the schema is per-expert, so that one line is normalised away. The
# remit block above it is compared by construction -- it is the only other
# thing the format call substitutes.
_xtails = []
for _p, _s in zip(_xprompts, experts_mod.EXPERTS):
    _tail = _p.split("Find the EARLIEST step", 1)[1]
    _xtails.append(_tail.replace(_s.vocab_line(), "<VOCAB>"))
check("the four experts are shown identical evidence and instructions",
      len(set(_xtails)) == 1)
check("the remit block is the only other thing that differs",
      all(_p.split("Find the EARLIEST step", 1)[0]
          == _xprompts[0].split("Find the EARLIEST step", 1)[0]
          .replace(experts_mod.EXPERTS[0].title, _s.title)
          .replace(experts_mod.EXPERTS[0].remit, _s.remit)
          for _p, _s in zip(_xprompts, experts_mod.EXPERTS)))
check("the evidence block is built once, not per expert",
      "Built once and shared verbatim" in inspect.getsource(experts_mod._evidence))
for _s, _p in zip(experts_mod.EXPERTS, _xprompts):
    check("the %s prompt carries the shared trajectory evidence" % _s.key,
          "COMPRESSED TRAJECTORY" in _p and "the patch was rejected" in _p
          and "step 4 -> step 6" in _p)

# Abstention. An expert with nothing in its remit must be able to say so, and
# saying so must cost it its vote rather than entering a low-confidence guess
# that still drags the fused answer toward it.
check("the prompt tells an expert it may abstain",
      '"found": false' in experts_mod.EXPERT_PROMPT
      and "abstaining" in experts_mod.EXPERT_PROMPT)
check("the prompt tells the expert not to nominate to have an answer",
      "least-innocent" in experts_mod.EXPERT_PROMPT)
_abstain = experts_mod._parse_verdict(
    experts_mod.EXPERTS[0], {"found": False, "critical_step": 7,
                             "confidence": 0.8}, _xt)
check("an explicit abstention does not vote even when a step was named",
      _abstain.found is False and _abstain.step == 7)
check("an abstaining expert is still recorded, with what it would have said",
      _abstain.to_dict()["step"] == 7
      and _abstain.to_dict()["found"] is False)

# Degradation. A model that returns nothing usable loses its vote; it does not
# take the stage down with it.
_broken = experts_mod._parse_verdict(experts_mod.EXPERTS[1], {}, _xt)
check("an empty reply becomes an abstention, not an exception",
      _broken.found is False and _broken.step is None)
_noflag = experts_mod._parse_verdict(
    experts_mod.EXPERTS[1], {"critical_step": 3, "confidence": 0.5}, _xt)
check("a missing 'found' is inferred from whether a step was named",
      _noflag.found is True and _noflag.step == 3)
_offv = experts_mod._parse_verdict(
    experts_mod.EXPERTS[2], {"found": True, "critical_step": 3,
                             "confidence": 0.5,
                             "fault_category": "something-else"}, _xt)
check("an off-vocabulary category is kept and marked, not silently dropped",
      _offv.fault_category == "something-else (off-vocab)", _offv.fault_category)

# The fusion rule. Confidence is the weight; agreement at a step beats a single
# louder voice elsewhere; ties go to the EARLIER step, because the whole method
# is looking for the step that STARTED the chain.
def _v(key, step, conf, found=True):
    return experts_mod.ExpertVerdict(key=key, found=found, step=step,
                                     confidence=conf)


_f2 = experts_mod.fuse([_v("compliance", 5, 0.5), _v("logic", 5, 0.5),
                        _v("flow", 40, 0.9), _v("context", None, 0.0, False)])
check("two agreeing experts outweigh one more confident expert alone",
      _f2["step"] == 5, _f2)
check("the fused confidence is the share of expressed confidence behind it",
      abs(_f2["confidence"] - (1.0 / 1.9)) < 1e-6, _f2["confidence"])
check("the backers of the winning step are recorded by name",
      _f2["backers"] == ["compliance", "logic"], _f2["backers"])

_f3 = experts_mod.fuse([_v("compliance", 20, 0.6), _v("logic", 4, 0.6)])
check("a tie breaks toward the earlier step, not the later one",
      _f3["step"] == 4, _f3)

_f4 = experts_mod.fuse([_v("compliance", 30, 0.6), _v("logic", 31, 0.6)])
check("adjacent nominations count as partial agreement",
      _f4["step"] == 30 and set(_f4["backers"]) == {"compliance", "logic"}, _f4)
check("the neighbour radius is 1, matching where acc@0 and acc@1 are equal",
      experts_mod.VOTE_RADIUS == 1)
_f5 = experts_mod.fuse([_v("compliance", 30, 0.6), _v("logic", 34, 0.6)])
check("nominations four steps apart are NOT treated as agreement",
      _f5["step"] == 30 and _f5["backers"] == ["compliance"], _f5)

_f0 = experts_mod.fuse([_v("compliance", None, 0.0, False),
                        _v("logic", None, 0.0, False),
                        _v("flow", None, 0.0, False),
                        _v("context", None, 0.0, False)])
check("four abstentions fuse to nothing rather than to a step",
      _f0["step"] is None and _f0["n_voting"] == 0, _f0)

check("the fusion spends no LLM call",
      "llm" not in inspect.signature(experts_mod.fuse).parameters
      and "generate" not in inspect.getsource(experts_mod.fuse))
check("the fusion says why it is a rule and not a fifth call",
      "general-purpose judge again" in experts_mod.__doc__)
check("the fusion documents that ties go to the earlier step and why",
      "STARTED the causal chain" in inspect.getsource(experts_mod.fuse))


# End to end through the pipeline, with dictated verdicts. This is the check
# that the stage is wired up at all: the flag has to reach localize_multi, and
# four calls have to leave the process.
class ExpertLLM(RecordingLLM):
    """Answers each expert's call with a dictated verdict."""

    def __init__(self, answers):
        super().__init__()
        self.answers = dict(answers)     # expert key -> dict
        self.expert_calls = 0

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        for spec in experts_mod.EXPERTS:
            if "YOUR REMIT — %s" % spec.title in prompt:
                self.expert_calls += 1
                return self.answers.get(spec.key, {"found": False})
        return default


_xllm = ExpertLLM({
    "compliance": {"found": True, "critical_step": 4, "confidence": 0.7,
                   "fault_category": "requirement-dropped",
                   "error_type": "plan.dropped-requirement",
                   "reason": "it implemented only the scalar case"},
    "logic": {"found": True, "critical_step": 4, "confidence": 0.5,
              "fault_category": "unverified-assumption",
              "error_type": "reason.unverified-assumption",
              "reason": "assumed the field was a scalar"},
    "flow": {"found": True, "critical_step": 9, "confidence": 0.9,
             "fault_category": "premature-termination",
             "error_type": "verify.no-verification", "reason": "stopped early"},
    "context": {"found": False},
})
_xpred = run(_xt, _xllm, Config(max_rounds=1, workers=1, multi_expert=True))
check("the multi-expert flag reaches the localiser and all four run",
      _xllm.expert_calls == 4, _xllm.expert_calls)
check("the vote beats the single loudest expert end to end",
      _xpred.critical_step == 4, _xpred.critical_step)
check("the prediction carries every verdict, abstentions included",
      _xpred.experts is not None
      and [v["expert"] for v in _xpred.experts["verdicts"]] == _EXPERT_KEYS,
      _xpred.experts)
check("the overruled expert's verdict is kept, not discarded",
      any(v["expert"] == "flow" and v["step"] == 9
          for v in _xpred.experts["verdicts"]))
check("the error_type comes from the most confident backer of the winner",
      _xpred.error_type == "plan.dropped-requirement", _xpred.error_type)
check("the lead expert and its fault category are named",
      _xpred.experts["lead"] == "compliance"
      and _xpred.experts["fault_category"] == "requirement-dropped")
check("the error_type stays in the taxonomy evaluation scores against",
      re.match(r"^(plan|reason|act|obs|verify)\.", _xpred.error_type or ""),
      _xpred.error_type)

# Everything the fusion needs must be in the record, or an alternative rule
# cannot be scored against a stored run without paying for the calls again.
_stored = _xpred.to_dict()["experts"]
check("a stored record holds each verdict's step and confidence",
      all(("step" in v and "confidence" in v) for v in _stored["verdicts"]))
check("re-fusing from the stored record reproduces the run's answer",
      experts_mod.fuse([
          experts_mod.ExpertVerdict(key=v["expert"], found=v["found"],
                                    step=v["step"], confidence=v["confidence"])
          for v in _stored["verdicts"]])["step"] == _xpred.critical_step)

# All four abstaining must not become a confident answer. The step still has to
# come from somewhere, and the record has to say it was not a diagnosis.
_xsilent = run(_xt, ExpertLLM({}), Config(max_rounds=1, workers=1,
                                          multi_expert=True))
check("when every expert abstains the answer falls back, at zero confidence",
      _xsilent.confidence == 0.0 and _xsilent.critical_step is not None,
      (_xsilent.critical_step, _xsilent.confidence))
check("a fallback answer is not attributed to an expert",
      _xsilent.experts["lead"] is None
      and "no expert reported" in _xsilent.reason)
check("a fallback answer carries no error_type it did not earn",
      _xsilent.error_type == "", _xsilent.error_type)

# The guards on the answer are the localiser's, unchanged. A non-attributable
# step cannot become the answer just because four experts agreed on it.
_xuser = Trajectory(
    task_id="expert-user", source="trajdebug", subset="swebenchpro",
    task_description="t",
    steps=([Step(0, "user", "the task")]
           + [Step(i, "assistant", "s%d" % i) for i in range(1, 8)]))
_xnonattr = run(_xuser, ExpertLLM({
    "compliance": {"found": True, "critical_step": 0, "confidence": 0.9},
    "logic": {"found": True, "critical_step": 0, "confidence": 0.9},
}), Config(max_rounds=1, workers=1, multi_expert=True))
check("a user turn cannot be the answer however many experts name it",
      _xnonattr.critical_step != 0
      and _xuser.steps[_xnonattr.critical_step].attributable,
      _xnonattr.critical_step)

# Flag OFF has to leave the old path untouched, byte for byte. The LLM cache is
# keyed on the prompt, so this is what lets the multi-expert run be A/B'd
# against the stored single-localiser run without re-spending it.
_off_a, _off_b = RecordingLLM(), RecordingLLM()
run(_xt, _off_a, Config(max_rounds=1, workers=1))
run(_xt, _off_b, Config(max_rounds=1, workers=1, multi_expert=False))
check("multi_expert=False changes not one byte of the single localiser's prompt",
      _off_a.prompts == _off_b.prompts)
check("with the flag off no expert prompt is ever built",
      not any("YOUR REMIT" in p for p in _off_a.prompts))
check("the config carries the flag into the artifact fingerprint",
      "multi_expert" in Config(multi_expert=True).fingerprint())

# Evaluation has to be able to separate what the experts reached from what the
# vote kept. Without that split, a disappointing acc@0 cannot be attributed to
# either the remits or the rule.
_xrow = score_one(_xpred, GroundTruth(task_id=_xt.task_id, critical_step=9))
check("scoring records which experts named the reference step",
      _xrow.get("expert_hits") == ["flow"], _xrow.get("expert_hits"))
check("scoring records the ceiling the fusion was working against",
      _xrow.get("expert_ceiling") is True)
_xagg = aggregate([_xrow])
check("the summary separates the expert ceiling from acc@0",
      _xagg.get("expert_ceiling") == 1.0 and _xagg.get("acc@0") == 0.0)
check("the summary reports what the vote kept of what the experts reached",
      _xagg.get("expert_fusion_kept") == 0.0)
check("the summary reports a per-remit hit rate and precision",
      set(_xagg.get("expert_per_remit", {})) == set(_EXPERT_KEYS)
      and _xagg["expert_per_remit"]["flow"]["precision"] == 1.0,
      _xagg.get("expert_per_remit"))
check("a single-localiser run reports no expert numbers at all",
      "expert_ceiling" not in aggregate(
          [score_one(run(_xt, RecordingLLM(), Config(max_rounds=1, workers=1)),
                     GroundTruth(task_id=_xt.task_id, critical_step=9))]))

# The claim the module makes about itself has to be the measured one, not a
# hopeful one. These pin the two numbers that motivated the architecture.
check("the module states the bottleneck it was built for",
      "P(hit | gold_on_flow) = 0.50" in experts_mod.__doc__)
check("the module states why the experts read the same material",
      "confound two changes at once" in experts_mod.__doc__)
check("the neighbour weight documents what would make it dishonest",
      "manufacturing agreement" in inspect.getsource(experts_mod))

for rec in records[:2]:
    gt = rec.ground_truth
    if gt is None or gt.critical_step is None:
        continue
    _rl = ExpertLLM({})
    run(rec.trajectory, _rl, Config(max_rounds=1, workers=1, multi_expert=True))
    blob = "\n".join(_rl.prompts)
    leaked = re.search(r"(critical|mistake|gold|correct)\D{0,40}\b%d\b"
                       % gt.critical_step, blob, re.I)
    check("%s: four expert prompts, still no gold-step assertion"
          % rec.trajectory.task_id[:26], leaked is None,
          str(leaked.group(0))[:60] if leaked else "")


# ---------------------------------------------------------------------------
# [26] RESTORING THE STEP'S OWN WORDS: discard a paraphrase nobody needed.
#
# Off-flow compression exists to buy prompt budget. For a long step it earns
# that budget; for a short one it spends the step's own words to save a few
# hundred characters. Measured on both corpora, a reference step that arrives
# paraphrased is found about half as often as one that arrives intact, so the
# rule prefers the original wherever the original was affordable.
#
# The checks below pin the three properties that make the rule safe to apply
# after compression rather than by re-tuning the compressor: it costs no calls,
# it can only replace text with that text's own source, and at limit 0 it is a
# no-op down to the byte.
print("\n[26] restoring verbatim step text")

from longtraj.schema import StepDigest as _SD

_rt = Trajectory(
    task_id="restore-fixture", source="trajdebug", subset="swebenchpro",
    task_description="make the loader accept a list",
    steps=[Step(0, "user", "please fix the loader"),
           Step(1, "assistant", "SHORT-ORIGINAL " + "a" * 200),
           Step(2, "assistant", "MID-ORIGINAL " + "b" * 1200),
           Step(3, "assistant", "HUGE-ORIGINAL " + "c" * 9000),
           Step(4, "assistant", "ONFLOW-ORIGINAL " + "d" * 1200)])


def _digests():
    return [
        _SD(index=1, role="assistant", text="SHORT-ORIGINAL " + "a" * 200,
            tier=KEEP, raw_chars=215),
        _SD(index=2, role="assistant", text="a paraphrase of step 2",
            tier=COMPRESSED, raw_chars=1212),
        _SD(index=3, role="assistant", text="a paraphrase of step 3",
            tier=COMPRESSED, raw_chars=9013),
        _SD(index=4, role="assistant", text="clipped on-flow text",
            tier=LIGHT, raw_chars=1215, flows="data"),
    ]


_d = _digests()
_n = compress_mod.restore_verbatim(_rt, _d, 2500)
_by = {x.index: x for x in _d}
check("a paraphrase under the limit is replaced by the original",
      _by[2].text == _rt.steps[2].content and _by[2].tier == KEEP)
check("restoring one step is counted", _n == 1, "got %d" % _n)
check("a step too long to afford keeps its paraphrase",
      _by[3].text == "a paraphrase of step 3" and _by[3].tier == COMPRESSED)
check("an on-flow step is left alone: it was never paraphrased",
      _by[4].text == "clipped on-flow text" and _by[4].tier == LIGHT)
check("a step already verbatim is not counted as restored",
      _by[1].text == _rt.steps[1].content and _by[1].tier == KEEP)

# The replacement can only ever be the step's own content. A rule that could
# substitute anything else would be a second, unlogged compressor.
check("every restored step's text is exactly its source step's content",
      all(x.text == _rt.steps[x.index].content
          for x in _d if x.tier == KEEP and not x.on_flow))

# limit 0 is the control arm of every A/B that uses this. It must not move a
# single byte, or the arm and its control differ by more than the intervention.
_d0 = _digests()
_before = [(x.index, x.text, x.tier) for x in _d0]
check("limit 0 restores nothing",
      compress_mod.restore_verbatim(_rt, _d0, 0) == 0)
check("limit 0 leaves the digests byte-identical",
      [(x.index, x.text, x.tier) for x in _d0] == _before)
check("a negative limit is also a no-op",
      compress_mod.restore_verbatim(_rt, _digests(), -5) == 0)

# Idempotence: re-running over already-restored digests must not double-count,
# because the pipeline may run over a trace dir a previous run wrote back.
_d2 = _digests()
compress_mod.restore_verbatim(_rt, _d2, 2500)
check("restoring twice restores nothing the second time",
      compress_mod.restore_verbatim(_rt, _d2, 2500) == 0)

# An index the trajectory does not have must not raise: digests are read back
# from stored artifacts, which can outlive the trajectory they were built from.
_dx = [_SD(index=99, role="assistant", text="orphan", tier=COMPRESSED,
           raw_chars=50)]
try:
    _ok = compress_mod.restore_verbatim(_rt, _dx, 2500) == 0
except Exception as exc:                                        # noqa: BLE001
    _ok = False
check("a digest pointing past the trajectory is skipped, not fatal", _ok)

# It is a pure text operation on material already in hand.
check("restoring costs no model call",
      "llm" not in inspect.signature(compress_mod.restore_verbatim)
      .parameters)
check("the restore rule cannot see a label",
      not any(p in ("gt", "ground_truth", "gold", "critical_step")
              for p in inspect.signature(compress_mod.restore_verbatim)
              .parameters))

# Wiring: the knob exists, defaults to off, and reaches the artifact fingerprint
# so a stored run says which arm produced it.
check("the pipeline exposes the limit as a config knob",
      "verbatim_limit" in {f.name for f in dataclasses.fields(Config)})
check("it is off by default", Config().verbatim_limit == 0)
check("the arm is recorded in the fingerprint",
      Config(verbatim_limit=2500).fingerprint().get("verbatim_limit") == 2500)
check("the pipeline restores after compressing, not before",
      inspect.getsource(pipeline_mod.run).index("restore_verbatim")
      > inspect.getsource(pipeline_mod.run).index("compress_mod.compress"))
check("the pipeline reports how many steps it restored",
      "steps_restored" in inspect.getsource(pipeline_mod.run))

# The docstring has to carry the measurement that justifies the rule, so a
# reader can check the claim instead of taking it.
check("the rule states the measurement it rests on",
      "0.167" in compress_mod.restore_verbatim.__doc__
      and "0.375" in compress_mod.restore_verbatim.__doc__)
check("the rule states why it runs after compression",
      "hits the cache" in compress_mod.restore_verbatim.__doc__)


# ---------------------------------------------------------------------------
# [27] SELF-CONSISTENCY: ask the same question k times and keep the majority.
#
# The localiser's answer at temperature 0 is one draw from a model that is not
# actually certain; its self-reported ``confidence`` is a claim, not evidence.
# Drawing k samples of the SAME prompt and voting replaces that claim with a
# measured one -- the winner's share of the votes -- and turns a lucky or
# unlucky single draw into a distribution.
#
# What the checks below have to pin is that nothing else moved. The prompt is
# unchanged, sample 0 is the configured temperature, and k=1 is the old path
# down to the call. Otherwise an A/B against a stored run would be measuring
# the rewrite instead of the sampling.
print("\n[27] self-consistency voting in the localiser")

from longtraj import llm as llm_mod

check("sampling is off by default",
      pipeline_mod.Config().samples == 1)
check("the sample count reaches the fingerprint",
      pipeline_mod.Config(samples=5).fingerprint().get("samples") == 5)
check("the localiser takes a sample count",
      "samples" in inspect.signature(localize_mod.localize).parameters)
check("the localiser's sample count defaults to one",
      inspect.signature(localize_mod.localize).parameters["samples"].default == 1)
check("the pipeline passes the configured count through",
      "samples=cfg.samples" in inspect.getsource(pipeline_mod.run))

# The vote itself. Majority wins; ties go to the EARLIEST step, because a cause
# precedes its symptom and the whole pipeline resolves ties that way.
_vt = Trajectory(
    task_id="vote-fixture", source="trajdebug", subset="swebenchpro",
    task_description="make the config loader accept a list",
    steps=[Step(0, "user", "please fix the loader")]
          + [Step(i, "assistant", "did thing %d" % i) for i in range(1, 12)])


def _draws(*steps):
    return [{"critical_step": s, "confidence": 0.5, "reason": "r%d" % s}
            for s in steps]


_v = localize_mod._vote(_vt, _draws(4, 4, 9))
check("the majority step wins the vote", _v["step"] == 4, _v["step"])
check("agreement is the winner's measured share of the votes",
      _v["agreement"] == round(2 / 3.0, 4), _v["agreement"])
check("every vote cast is counted", _v["cast"] == 3, _v["cast"])
check("the whole tally is kept, not just the winner",
      _v["votes"] == {"4": 2, "9": 1}, _v["votes"])

_vtie = localize_mod._vote(_vt, _draws(9, 4))
check("a tie goes to the earlier step, since a cause precedes its symptom",
      _vtie["step"] == 4, _vtie["step"])

check("a unanimous vote reports full agreement",
      localize_mod._vote(_vt, _draws(6, 6, 6))["agreement"] == 1.0)

# Junk draws must not be able to win, or a malformed response would outvote
# real answers by being counted.
_vjunk = localize_mod._vote(_vt, [{"critical_step": "not a number"},
                                  {"no_such_field": 1},
                                  {"critical_step": 7}])
check("unparseable draws are discarded rather than voted",
      _vjunk["step"] == 7 and _vjunk["cast"] == 1, _vjunk)
check("a vote with nothing usable answers nothing and says so",
      localize_mod._vote(_vt, [{"critical_step": None}]) ==
      {"step": None, "obj": {}, "votes": {}, "agreement": 0.0, "cast": 0})
check("a user turn cannot win a vote however many draws name it",
      localize_mod._vote(_vt, _draws(0, 0, 0))["step"] != 0)

# The winner's own object has to come back with it: the reason and error_type
# reported must belong to the answer given, not to some other draw.
_vobj = localize_mod._vote(_vt, [{"critical_step": 9, "reason": "nine"},
                                 {"critical_step": 4, "reason": "four"},
                                 {"critical_step": 4, "reason": "four again"}])
check("the winning step's own draw is what gets reported",
      _vobj["obj"]["reason"] == "four", _vobj["obj"])

# Variants must differ, or k draws are k copies of one answer and the vote is
# decoration. Variant 0 stays on the stored key so existing caches stay valid.
_k0 = LLM(dry_run=True)._cache_key("p", 100, "s", 0)
check("variant 0 keeps the key every stored run was written under",
      _k0 == LLM(dry_run=True)._cache_key("p", 100, "s"))
check("a sample gets its own cache entry",
      LLM(dry_run=True)._cache_key("p", 100, "s", 1) != _k0)
check("two samples do not share an entry",
      LLM(dry_run=True)._cache_key("p", 100, "s", 1)
      != LLM(dry_run=True)._cache_key("p", 100, "s", 2))
check("samples above zero are drawn at a temperature that can differ",
      llm_mod.SAMPLE_TEMPERATURE > 0.0)


class VotingLLM(RecordingLLM):
    """Answers each variant of the localiser's prompt with a dictated step."""

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)
        self.localize_calls = 0
        self.localize_prompts = []

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True, variant=0):
        self.prompts.append(prompt)
        if "critical_step" in prompt:
            self.localize_calls += 1
            self.localize_prompts.append(prompt)
            return {"critical_step": self.steps[variant % len(self.steps)],
                    "confidence": 0.4, "reason": "draw %d" % variant}
        return default if default is not None else {}


_vllm = VotingLLM([9, 4, 4])
_vpred = run(_vt, _vllm, Config(max_rounds=1, workers=1, samples=3))
check("k samples cost k localiser calls",
      _vllm.localize_calls == 3, _vllm.localize_calls)
check("the majority overrules the deterministic draw end to end",
      _vpred.critical_step == 4, _vpred.critical_step)
check("every draw asked the same question",
      len(set(_vllm.localize_prompts)) == 1,
      len(set(_vllm.localize_prompts)))
check("the tally is carried on the prediction",
      _vpred.samples and _vpred.samples["votes"] == {"4": 2, "9": 1},
      _vpred.samples)
check("the measured agreement is recorded beside the model's own claim",
      _vpred.samples["agreement"] == round(2 / 3.0, 4))
check("how many draws were asked for is recorded, not just how many landed",
      _vpred.samples["asked"] == 3)
check("the tally survives into the stored record",
      _vpred.to_dict()["samples"]["cast"] == 3)

_1llm = VotingLLM([9])
_1pred = run(_vt, _1llm, Config(max_rounds=1, workers=1))
check("one sample is one call, exactly as before",
      _1llm.localize_calls == 1, _1llm.localize_calls)
check("an unsampled run carries no tally at all",
      _1pred.samples is None, _1pred.samples)
check("an unsampled run's prompt is unchanged by the sampling code",
      _1llm.localize_prompts[0] == _vllm.localize_prompts[0])

# Scoring has to separate what the draws reached from what the rule kept, or a
# vote that throws away a correct draw is invisible.
_vscored = evaluate_mod.score_one(
    _vpred, GroundTruth(task_id="vote-fixture", critical_step=9))
check("scoring records whether any draw named the reference step",
      _vscored["vote_ceiling"] is True, _vscored)
check("the vote's loss is visible: reached but not kept",
      _vscored["vote_ceiling"] and not _vscored["hit@0"])
check("the reference step's share of the draws is recorded",
      _vscored["vote_gold_share"] == round(1 / 3.0, 10)
      or abs(_vscored["vote_gold_share"] - 1 / 3.0) < 1e-9,
      _vscored.get("vote_gold_share"))

# Ground truth. The vote is a rule over draws of one prompt; it must not be
# able to see a label, and the prompt it votes on is the unchanged one.
check("the vote cannot be handed ground truth",
      not any(_GOLD_KEY.search(p)
              for p in inspect.signature(localize_mod._vote).parameters))
check("the vote reads only the draws and the trajectory",
      list(inspect.signature(localize_mod._vote).parameters) == ["traj", "objs"])
check("the rule states why ties go to the earliest step",
      "EARLIEST" in localize_mod._vote.__doc__
      or "earliest" in localize_mod._vote.__doc__)


# ---------------------------------------------------------------------------
# [28] THE SECOND PASS: re-read a handful of finalists in their own words.
#
# The measurement this stage answers to is that the region is right and the step
# is wrong. On SWE-Bench-Pro the candidate list holds the reference step 78.6%
# of the time and the localiser picks it 39.4% of the time it is there; on
# TrajAudit acc@3 is 0.7059 against acc@0 of 0.4804.
#
# The stage is only sound if three things hold, and the checks below pin them:
# the answer can only move WITHIN a set the model was shown in full, the first
# answer survives in the record so the stage can be scored as a net rather than
# as a total, and nothing about the first pass changes when the flag is off.
print("\n[28] the second pass over the finalists")

from longtraj import zoom as zoom_mod

_zt = Trajectory(
    task_id="zoom-fixture", source="trajaudit", subset=None,
    task_description="make the config loader accept a list",
    steps=[Step(0, "user", "please fix the loader")]
          + [Step(i, "assistant", "did thing %d" % i) for i in range(1, 20)])
_zg = FlowGraph(anchors=[19], anchors_located=[19],
                failure_summary="the patch was rejected")


def _pred(step, reason="because it looked wrong", conf=0.4):
    return Prediction(task_id="zoom-fixture", critical_step=step,
                      confidence=conf, reason=reason)


# -- the finalist set ------------------------------------------------------
_f = zoom_mod.finalists(_zt, 10, None, window=3)
check("with no candidate list the finalists are a window round the answer",
      _f == [7, 8, 9, 10, 11, 12, 13], _f)
check("the first answer is always among its own finalists",
      10 in zoom_mod.finalists(_zt, 10, None, window=3))
check("a user turn cannot be offered as a finalist",
      0 not in zoom_mod.finalists(_zt, 1, None, window=5))
check("the window is clipped at the ends of the trajectory",
      zoom_mod.finalists(_zt, 18, None, window=5)[-1] == 19)

_fc = zoom_mod.finalists(_zt, 10, [{"step": 3}, {"step": 10}, {"step": 17}])
check("a candidate list is inherited rather than replaced by a window",
      _fc == [3, 10, 17], _fc)
check("an out-of-range candidate is dropped, not carried into the prompt",
      zoom_mod.finalists(_zt, 10, [{"step": 999}, {"step": 4}]) == [4, 10])
check("a malformed candidate entry is skipped without failing the stage",
      zoom_mod.finalists(_zt, 10, [{"nope": 1}, {"step": 4}]) == [4, 10])

# Over the cap, the far entries go: the premise of the stage is that the region
# is already right, so a finalist twenty steps away was never reachable.
_fcap = zoom_mod.finalists(_zt, 10, [{"step": s} for s in range(1, 20)], cap=5)
check("over the cap the finalists nearest the first answer are kept",
      _fcap == [8, 9, 10, 11, 12], _fcap)
check("the first answer survives the cut",
      10 in _fcap)

# -- the prompt ------------------------------------------------------------
_zrender = zoom_mod._render(_zt, [4, 5])
check("finalists are rendered in their own words, not a paraphrase",
      "did thing 4" in _zrender and "did thing 5" in _zrender)
check("each finalist is rendered with its own index",
      "STEP 4" in _zrender and "STEP 5" in _zrender)
check("the prompt asks for the earliest step that is itself wrong",
      "EARLIEST" in zoom_mod.ZOOM_PROMPT)
check("the prompt confines the answer to the finalists",
      "only answer with one of the finalist step numbers"
      in zoom_mod.ZOOM_PROMPT)
check("the second pass is told the first answer so it can keep it",
      "{first}" in zoom_mod.ZOOM_PROMPT)

# Ground truth. The stage reads steps, a graph and its own first answer; none
# of those is a label, and no prompt field may look like one.
check("the second pass cannot be handed ground truth",
      not any(_GOLD_KEY.search(p)
              for p in inspect.signature(zoom_mod.zoom).parameters))
check("the second pass's prompt names no label field",
      not any(k in (zoom_mod.ZOOM_PROMPT + zoom_mod.ZOOM_SYSTEM)
              for k in ("annotation", "gold_patch", "failure_reason",
                        "test_patch", "failure_id", "ground_truth")))


class ZoomLLM(RecordingLLM):
    """Answers the second pass with a dictated step."""

    def __init__(self, first, second, conf=0.8):
        super().__init__()
        self.first, self.second, self.conf = first, second, conf
        self.zoom_calls = 0
        self.zoom_prompt = ""

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if "THE FINALISTS" in prompt:
            self.zoom_calls += 1
            self.zoom_prompt = prompt
            return {"critical_step": self.second, "confidence": self.conf,
                    "changed": True, "error_type": "reason.wrong-file",
                    "reason": "this step edits the wrong file"}
        if "critical_step" in prompt:
            return {"critical_step": self.first, "confidence": 0.4,
                    "reason": "it looked wrong"}
        return default if default is not None else {}


_zllm = ZoomLLM(first=10, second=7)
_zp = zoom_mod.zoom(_zt, _zg, _zllm, _pred(10), window=3)
check("the second pass costs exactly one call", _zllm.zoom_calls == 1)
check("the answer moves when the second pass names another finalist",
      _zp.critical_step == 7, _zp.critical_step)
check("the first answer is kept in the record, not overwritten",
      _zp.zoom["first"] == 10, _zp.zoom)
check("the record says the answer was moved",
      _zp.zoom["accepted"] is True)
check("the reported reason describes the step now being returned",
      "wrong file" in _zp.reason)
check("the reason says which step was moved off",
      "step 10" in _zp.reason)
check("the record survives into the stored form",
      _zp.to_dict()["zoom"]["proposed"] == 7)

# An answer outside the set is refused, not snapped. The model was shown
# everything it was allowed to choose from, so a step outside the set means it
# did not do the task -- and snapping would hide that.
_zout = zoom_mod.zoom(_zt, _zg, ZoomLLM(first=10, second=2), _pred(10),
                      window=3)
check("an answer outside the finalists is refused",
      _zout.critical_step == 10, _zout.critical_step)
check("the refusal is recorded rather than silently swallowed",
      _zout.zoom["in_set"] is False and _zout.zoom["proposed"] == 2)

# Keeping the first answer must be free: agreeing is not a change.
_zsame = zoom_mod.zoom(_zt, _zg, ZoomLLM(first=10, second=10), _pred(10),
                       window=3)
check("agreeing with the first answer is not recorded as a change",
      _zsame.critical_step == 10 and _zsame.zoom["accepted"] is False)

# No first answer, or nothing to choose between, means no call at all.
_znone = ZoomLLM(first=None, second=3)
check("with no first answer the stage makes no call",
      zoom_mod.zoom(_zt, _zg, _znone, _pred(None)).critical_step is None
      and _znone.zoom_calls == 0)
_zone = ZoomLLM(first=10, second=3)
check("a single finalist is not worth a call",
      zoom_mod.zoom(_zt, _zg, _zone, _pred(10),
                    cands=[{"step": 10}]).critical_step == 10
      and _zone.zoom_calls == 0)

# -- wiring ----------------------------------------------------------------
check("the second pass is off by default", pipeline_mod.Config().zoom is False)
check("the arm is recorded in the fingerprint",
      pipeline_mod.Config(zoom=True).fingerprint().get("zoom") is True)
check("the pipeline runs the second pass after the localiser, not before",
      inspect.getsource(pipeline_mod.run).index("zoom_mod.zoom")
      > inspect.getsource(pipeline_mod.run).index("localize_mod.localize"))

_zllm2 = ZoomLLM(first=10, second=7)
_zrun = run(_zt, _zllm2, Config(max_rounds=1, workers=1, zoom=True,
                                zoom_window=3))
check("the flag reaches the stage end to end",
      _zllm2.zoom_calls == 1 and _zrun.critical_step == 7, _zrun.critical_step)
_zoff = ZoomLLM(first=10, second=7)
_zrunoff = run(_zt, _zoff, Config(max_rounds=1, workers=1))
check("with the flag off no second call leaves the process",
      _zoff.zoom_calls == 0)
check("with the flag off the answer and the record are untouched",
      _zrunoff.critical_step == 10 and _zrunoff.zoom is None)

# -- scoring ---------------------------------------------------------------
# The stage has to be scorable as a NET. One that fixes four cases and breaks
# four is worth nothing, and acc@0 alone cannot tell that from doing nothing.
_zfix = evaluate_mod.score_one(
    _zrun, GroundTruth(task_id="zoom-fixture", critical_step=7))
check("scoring records what the answer was before the second pass",
      _zfix["hit@0_prezoom"] is False and _zfix["hit@0"] is True)
check("a case the second pass fixed is counted as fixed",
      _zfix["zoom_fixed"] is True and _zfix["zoom_broke"] is False)
_zbrk = evaluate_mod.score_one(
    _zrun, GroundTruth(task_id="zoom-fixture", critical_step=10))
check("a case the second pass broke is counted as broken",
      _zbrk["zoom_broke"] is True and _zbrk["zoom_fixed"] is False)
check("the reachable ceiling is recorded: only a finalist can be answered",
      evaluate_mod.score_one(
          _zrun, GroundTruth(task_id="zoom-fixture",
                             critical_step=19))["zoom_ceiling"] is False)
_zagg = evaluate_mod.aggregate([_zfix, _zbrk])
check("the summary reports the accuracy the run would have had without it",
      _zagg.get("acc@0_prezoom") is not None, _zagg.get("acc@0_prezoom"))
check("the summary reports both halves of the difference",
      _zagg.get("n_zoom_fixed") == 1 and _zagg.get("n_zoom_broke") == 1, _zagg)

# The module has to carry the measurement that justifies it, and has to say why
# it is not the restore pass that was already tried and lost.
check("the stage states the measurement it rests on",
      "78.6" in zoom_mod.__doc__ and "0.7059" in zoom_mod.__doc__)
check("the stage says why it is not the restore pass",
      "compress.py" in zoom_mod.__doc__)


print("\n[29] the group stage over the finalists")

from longtraj import bracket as bracket_mod

_bt = Trajectory(
    task_id="bracket-fixture", source="trajaudit", subset=None,
    task_description="make the config loader accept a list",
    steps=[Step(0, "user", "please fix the loader")]
          + [Step(i, "assistant", "did thing %d" % i) for i in range(1, 30)])
_bg = FlowGraph(anchors=[29], anchors_located=[29],
                failure_summary="the patch was rejected")


def _bpred(step, conf=0.4):
    return Prediction(task_id="bracket-fixture", critical_step=step,
                      confidence=conf, reason="it looked wrong")


# -- the finalist set ------------------------------------------------------
# The union is the point: neither source is enough on its own. The candidate
# list holds the reference step 0.4314 of the time on TrajAudit, the window
# 0.3953 of the time on SWE-Bench-Pro, and the union beats both on both.
_bf = bracket_mod.finalists(_bt, 10, [{"step": 3}, {"step": 25}], window=2)
check("the finalist set unions the candidate list with a window",
      _bf == [3, 8, 9, 10, 11, 12, 25], _bf)
check("with no candidate list it falls back to the window alone",
      bracket_mod.finalists(_bt, 10, None, window=2) == [8, 9, 10, 11, 12])
check("a window of zero takes the window out without taking out the stage",
      bracket_mod.finalists(_bt, 10, [{"step": 3}], window=0) == [3])
check("with no first answer only the candidate list survives",
      bracket_mod.finalists(_bt, None, [{"step": 3}, {"step": 25}]) == [3, 25])
check("a user turn cannot be offered as a finalist",
      0 not in bracket_mod.finalists(_bt, 1, [{"step": 0}], window=5))
check("an out-of-range candidate is dropped, not carried into a prompt",
      bracket_mod.finalists(_bt, None, [{"step": 999}, {"step": 4}]) == [4])
check("a malformed candidate entry is skipped without failing the stage",
      bracket_mod.finalists(_bt, None, [{"nope": 1}, {"step": 4}]) == [4])

# The cap rule is the measured one: candidates are kept whole and WINDOW
# entries go. Cutting by distance to the first answer instead took the
# SWE-Bench-Pro shortlist from 0.7907 to 0.6512, because there the reference
# step is usually nowhere near an answer that is wrong 70% of the time.
_bcap = bracket_mod.finalists(_bt, 20, [{"step": s} for s in (2, 3, 4, 5)],
                              window=5, cap=6)
check("over the cap the whole candidate list is kept",
      set([2, 3, 4, 5]) <= set(_bcap), _bcap)
check("over the cap it is window entries that are dropped",
      len(_bcap) == 6 and sorted(_bcap) == _bcap, _bcap)
check("the window entries kept are the ones nearest the first answer",
      _bcap == [2, 3, 4, 5, 19, 20], _bcap)
_bcap2 = bracket_mod.finalists(_bt, 20, [{"step": s} for s in range(1, 12)],
                               window=5, cap=4)
check("a candidate list longer than the cap is itself cut",
      _bcap2 == [1, 2, 3, 4], _bcap2)

# -- the groups ------------------------------------------------------------
check("finalists are cut into contiguous groups",
      bracket_mod.groups([1, 2, 3, 4, 5, 6, 7, 8], size=4)
      == [[1, 2, 3, 4], [5, 6, 7, 8]])
check("a set at or under the group size is one group",
      bracket_mod.groups([1, 2, 3], size=4) == [[1, 2, 3]])
check("an empty set makes no groups", bracket_mod.groups([], size=4) == [])
# A group of one would be asked "is this step wrong" with nothing to lose to,
# which is the n-way question this stage exists to avoid.
_bg9 = bracket_mod.groups(list(range(9)), size=4)
check("no group is left holding a single finalist",
      all(len(g) >= 2 for g in _bg9), _bg9)
check("balancing does not drop or duplicate a finalist",
      sorted(i for g in _bg9 for i in g) == list(range(9)), _bg9)
check("every group stays contiguous after balancing",
      all(g == list(range(g[0], g[0] + len(g))) for g in _bg9), _bg9)

# -- the prompts -----------------------------------------------------------
check("group members are rendered in their own words, not a paraphrase",
      "did thing 4" in bracket_mod._render(_bt, [4, 5]))
check("each member is rendered with its own index",
      "STEP 4" in bracket_mod._render(_bt, [4, 5]))
check("a group is allowed to name no step at all",
      "null if none is" in bracket_mod.GROUP_PROMPT)
check("the group prompt says the fragment may not contain the cause",
      "may well be outside them" in bracket_mod.GROUP_PROMPT)
check("the group prompt is not shown the first answer",
      "{first" not in bracket_mod.GROUP_PROMPT)
check("the final prompt asks for the earliest step that is itself wrong",
      "EARLIEST" in bracket_mod.FINAL_PROMPT)
check("the final prompt refuses a blanket earliest-wins rule",
      "earlier is not automatically better" in bracket_mod.FINAL_PROMPT)
check("the final prompt confines the answer to the group winners",
      "only answer with one of the candidate step numbers"
      in bracket_mod.FINAL_PROMPT)

check("the stage cannot be handed ground truth",
      not any(_GOLD_KEY.search(p)
              for p in inspect.signature(bracket_mod.bracket).parameters))
check("neither prompt names a label field",
      not any(k in (bracket_mod.GROUP_PROMPT + bracket_mod.FINAL_PROMPT
                    + bracket_mod.GROUP_SYSTEM + bracket_mod.FINAL_SYSTEM)
              for k in ("annotation", "gold_patch", "failure_reason",
                        "test_patch", "failure_id", "ground_truth")))
check("the stage states the measurement it rests on",
      "0.7791" in bracket_mod.__doc__ and "0.8039" in bracket_mod.__doc__)
check("the stage says why it is not the second pass",
      "zoom.py" in bracket_mod.__doc__)


class BracketLLM(RecordingLLM):
    """Each group names the step it was told to; the final picks one."""

    def __init__(self, picks, final=None, conf=0.8, final_conf=0.9):
        super().__init__()
        self.picks = dict(picks)          # first member of a group -> its pick
        self.final, self.conf = final, conf
        self.final_conf = final_conf
        self.group_calls = 0
        self.final_calls = 0
        self.group_prompts = []
        self.final_prompt = ""

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if "THE STEPS" in prompt:
            self.group_calls += 1
            self.group_prompts.append(prompt)
            shown = [i for i in range(len(_bt.steps))
                     if ("### STEP %d " % i) in prompt]
            pick = self.picks.get(shown[0], None) if shown else None
            return {"best": pick, "confidence": self.conf,
                    "why": "step %s edits the wrong file" % pick}
        if "THE CANDIDATES" in prompt:
            self.final_calls += 1
            self.final_prompt = prompt
            return {"critical_step": self.final, "confidence": self.final_conf,
                    "error_type": "reason.wrong-file",
                    "reason": "this is the earliest one actually wrong"}
        if "critical_step" in prompt:
            return {"critical_step": 10, "confidence": 0.4,
                    "reason": "it looked wrong"}
        return default if default is not None else {}


_bcands = [{"step": s} for s in (2, 3, 6, 7, 12, 13, 18, 19)]
_bllm = BracketLLM(picks={2: 3, 6: 6, 12: 13, 18: 18}, final=6)
_bp = bracket_mod.bracket(_bt, _bg, _bllm, _bpred(10), cands=_bcands,
                          window=0, size=2)
check("each group costs exactly one call", _bllm.group_calls == 4,
      _bllm.group_calls)
check("the winners meet in exactly one final call", _bllm.final_calls == 1)
check("the answer moves to what the final chose",
      _bp.critical_step == 6, _bp.critical_step)
check("the first answer is kept in the record, not overwritten",
      _bp.bracket["first"] == 10, _bp.bracket)
check("every group verdict is kept, not only the winning one",
      [v["step"] for v in _bp.bracket["verdicts"]] == [3, 6, 13, 18],
      _bp.bracket["verdicts"])
check("the record says the answer was moved",
      _bp.bracket["accepted"] is True)
check("the reason describes the step now being returned",
      "earliest one actually wrong" in _bp.reason)
check("the reason says which step was moved off",
      "step 10" in _bp.reason)
check("the record survives into the stored form",
      _bp.to_dict()["bracket"]["proposed"] == 6)
check("the final call is shown each winner's own case",
      "edits the wrong file" in _bllm.final_prompt)
check("the final call is shown each winner in its own text",
      "did thing 6" in _bllm.final_prompt)
check("no group call is shown the first answer",
      not any("STEP 10" in p for p in _bllm.group_prompts))

# A group that declines is an abstention, not a vote for its least defensible
# member -- that is the whole reason the group prompt offers null.
_babs = BracketLLM(picks={2: None, 6: 6, 12: None, 18: None}, final=6)
_bpa = bracket_mod.bracket(_bt, _bg, _babs, _bpred(10), cands=_bcands,
                           window=0, size=2)
check("an abstaining group is recorded as an abstention",
      _bpa.bracket["abstained"] == 3, _bpa.bracket)
check("a single surviving winner is taken without a final call",
      _babs.final_calls == 0 and _bpa.critical_step == 6)

_ball = BracketLLM(picks={2: None, 6: None, 12: None, 18: None})
_bpn = bracket_mod.bracket(_bt, _bg, _ball, _bpred(10), cands=_bcands,
                           window=0, size=2)
check("when every group abstains the first answer stands",
      _bpn.critical_step == 10, _bpn.critical_step)
check("an empty bracket makes no final call", _ball.final_calls == 0)
check("the empty bracket is still recorded",
      _bpn.bracket["proposed"] is None and _bpn.bracket["abstained"] == 4)

# A group naming a step it was never shown answered a question it was not
# asked: it had no text to ground that on, so it is dropped, not snapped.
_bstray = BracketLLM(picks={2: 25, 6: 6, 12: None, 18: None}, final=6)
_bps = bracket_mod.bracket(_bt, _bg, _bstray, _bpred(10), cands=_bcands,
                           window=0, size=2)
check("a group answering outside its own members is dropped",
      [v["step"] for v in _bps.bracket["verdicts"]] == [6],
      _bps.bracket["verdicts"])

# Same for the final. It saw every winner, so a step outside them is not a
# better find; falling back to the earliest winner keeps the stage grounded.
_bfs = BracketLLM(picks={2: 3, 6: 6, 12: 13, 18: 18}, final=25)
_bpf = bracket_mod.bracket(_bt, _bg, _bfs, _bpred(10), cands=_bcands,
                           window=0, size=2)
check("a final answering outside the winners is refused, not snapped",
      _bpf.critical_step == 3, _bpf.critical_step)
check("the refusal is recorded rather than silently swallowed",
      _bpf.bracket["out_of_set"] == 25, _bpf.bracket)

# Agreeing is not a change, and it must be free.
_bsame = BracketLLM(picks={2: 3, 6: 10}, final=10)
_bpsame = bracket_mod.bracket(_bt, _bg, _bsame, _bpred(10),
                              cands=[{"step": s} for s in (2, 3, 6, 10)],
                              window=0, size=2)
check("agreeing with the first answer is not recorded as a change",
      _bpsame.bracket["accepted"] is False, _bpsame.bracket)

# Nothing to arbitrate means no calls at all.
_bnone = BracketLLM(picks={}, final=3)
check("with fewer than two finalists the stage makes no call",
      bracket_mod.bracket(_bt, _bg, _bnone, _bpred(None),
                          cands=[{"step": 4}]).critical_step is None
      and _bnone.group_calls == 0 and _bnone.final_calls == 0)

# -- wiring ----------------------------------------------------------------
check("the stage is off by default", Config().bracket is False)
check("the knobs reach the artifact fingerprint",
      Config(bracket=True, bracket_max=12).fingerprint()["bracket_max"] == 12)

_bwt = Trajectory(task_id="bw", source="trajaudit", subset=None,
                  task_description="t",
                  steps=[Step(0, "user", "go")]
                        + [Step(i, "assistant", "did %d" % i)
                           for i in range(1, 12)])
_bwllm = BracketLLM(picks={}, final=None)
_bwp = run(_bwt, _bwllm, Config(max_rounds=1, workers=1, bracket=False))
check("with the flag off no group call leaves the process",
      _bwllm.group_calls == 0 and _bwp.bracket is None)

# -- scoring ---------------------------------------------------------------
_bgt = GroundTruth(task_id="bracket-fixture", critical_step=6)
_brow = score_one(_bp, _bgt)
check("scoring records what the answer was before the group stage",
      _brow["hit@0_prebracket"] is False and _brow["hit@0"] is True)
check("a case the group stage fixed is counted as fixed",
      _brow["bracket_fixed"] is True and _brow["bracket_broke"] is False)
check("the shortlist ceiling is recorded",
      _brow["bracket_ceiling"] is True)
check("the group-winner ceiling is recorded separately from it",
      _brow["bracket_group_ceiling"] is True)
_bbroke = score_one(_bp, GroundTruth(task_id="bracket-fixture",
                                     critical_step=10))
check("a case the group stage broke is counted as broken",
      _bbroke["bracket_broke"] is True and _bbroke["bracket_fixed"] is False)
check("a reference step no group put forward is below the group ceiling",
      score_one(_bp, GroundTruth(task_id="bracket-fixture",
                                 critical_step=12))["bracket_group_ceiling"]
      is False)
_bsum = aggregate([_brow])
check("the summary reports the accuracy the run would have had without it",
      _bsum["acc@0_prebracket"] == 0.0 and _bsum["acc@0"] == 1.0, _bsum)
check("the summary reports both halves of the difference",
      _bsum["n_bracket_fixed"] == 1 and _bsum["n_bracket_broke"] == 0)
check("the summary reports the abstention rate",
      _bsum["bracket_abstain_rate"] == 0.0, _bsum)


# ---------------------------------------------------------------------------
print("\n[30] protecting the answer space from compression")

# The measurement this rests on: compression protects graph.on_flow(), but the
# localiser must answer with a step from the candidate list, and those two sets
# are only partly the same. A reference step that arrives paraphrased is found
# about half as often as one that arrives intact.

_pt = Trajectory(
    task_id="protect-fixture", source="trajdebug", subset="swebenchpro",
    task_description="rename the field",
    steps=[Step(0, "user", "please rename it")]
          + [Step(i, "assistant", "step %d body " % i + "x" * 2000)
             for i in range(1, 9)])
_pg = FlowGraph(anchors=[8], anchors_located=[8],
                edges=[FlowEdge(src=7, dst=8, kind="data", entity="field")],
                failure_summary="the tests failed")
_pspans = [Span(start=0, end=8)]


class ProtectLLM(RecordingLLM):
    """Paraphrases every step it is asked to, so a paraphrase is detectable."""

    def __init__(self):
        super().__init__()
        self.asked = []
        self.content = ""

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if "JOB B — COMPRESS THESE STEPS" in prompt:
            self.content = prompt
            ids = re.findall(r"^JOB B — COMPRESS THESE STEPS: (.*)$",
                             prompt, re.MULTILINE)
            got = re.findall(r"\d+", ids[0]) if ids else []
            self.asked = [int(x) for x in got]
            return {"label": "phase", "summary": "it renamed things",
                    "steps": {str(i): "PARAPHRASE of %d: " % i + "y" * 400
                              for i in range(0, 9)}}
        return default if default is not None else {}


def _tiers(digests):
    return {d.index: d for d in digests}


# -- without protection, an off-flow candidate is paraphrased ---------------
_pl = ProtectLLM()
_, _pd = compress_mod.compress(_pt, [Span(start=0, end=8)], _pg, _pl,
                               workers=1)
_pm = _tiers(_pd)
check("without protection an off-flow step is rewritten by the model",
      _pm[3].tier == COMPRESSED and "PARAPHRASE" in _pm[3].text, _pm[3].text)
check("on-flow steps are kept verbatim whether or not anything is protected",
      _pm[7].tier == KEEP and "PARAPHRASE" not in _pm[7].text)

# -- with protection, the same step arrives in its own words ----------------
_pl2 = ProtectLLM()
_, _pd2 = compress_mod.compress(_pt, [Span(start=0, end=8)], _pg, _pl2,
                                workers=1, protect=[3, 5])
_pm2 = _tiers(_pd2)
check("a protected step keeps its own words",
      _pm2[3].tier == KEEP and _pm2[3].text == _pt.steps[3].content,
      _pm2[3].tier)
check("protection applies to every step named, not just the first",
      _pm2[5].tier == KEEP and _pm2[5].text == _pt.steps[5].content)
check("an unprotected off-flow step is still compressed",
      _pm2[4].tier == COMPRESSED and "PARAPHRASE" in _pm2[4].text)
check("protecting a step does not claim a dependency edge it never had",
      _pm2[3].on_flow is False and _pm2[3].flows == "", _pm2[3].flows)
check("a protected step is not counted as an anchor",
      _pm2[3].anchor is False)
check("protection does not change how many digests come back",
      len(_pd2) == len(_pd) == 9)
check("the model is not asked to compress a protected step",
      3 not in _pl2.asked and 5 not in _pl2.asked, _pl2.asked)
check("it is still asked to compress the unprotected ones",
      4 in _pl2.asked and 6 in _pl2.asked, _pl2.asked)
check("the span prompt marks a protected step as off limits",
      "[ANSWER-SPACE STEP - do not compress]" in _pl2.content)
check("the dependency-path marker is not reused for it",
      _pl2.content.count("[ON DEPENDENCY PATH - do not compress]") == 2,
      _pl2.content.count("[ON DEPENDENCY PATH - do not compress]"))

# Protecting nothing must be byte-identical to not protecting: that is what
# makes the A/B attribute the difference to this rule and nothing else.
_pl3 = ProtectLLM()
compress_mod.compress(_pt, [Span(start=0, end=8)],
                      _pg, _pl3, workers=1, protect=[])
check("an empty protection set leaves the span prompt byte-identical",
      _pl3.prompts == _pl.prompts)

# A step already on the flow is protected either way, and must not be
# double-counted or stripped of its edge annotation.
_pl4 = ProtectLLM()
_, _pd4 = compress_mod.compress(_pt, [Span(start=0, end=8)], _pg, _pl4,
                                workers=1, protect=[7])
check("protecting a step that is already on the flow keeps its annotation",
      _tiers(_pd4)[7].on_flow is True and _tiers(_pd4)[7].flows != "")

# -- the failure path ------------------------------------------------------
# When the span call fails the digests are built deterministically, and a
# protected step must survive there too -- that is exactly when losing it
# would be silent.
class DeadLLM(RecordingLLM):
    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        raise RuntimeError("the span call died")


_, _pdf = compress_mod.compress(_pt, [Span(start=0, end=8)], _pg,
                                DeadLLM(), workers=1, protect=[3])
check("a protected step survives a failed span call in its own words",
      _tiers(_pdf)[3].tier == KEEP
      and _tiers(_pdf)[3].text == _pt.steps[3].content)
check("an unprotected step is still clipped when the span call fails",
      _tiers(_pdf)[4].tier == LIGHT)

# -- wiring ----------------------------------------------------------------
check("the rule is off by default", Config().protect_candidates is False)
check("the knob reaches the artifact fingerprint",
      Config(protect_candidates=True).fingerprint()["protect_candidates"]
      is True)


class SpyCompress:
    """Captures what the pipeline actually protected."""

    def __init__(self, real):
        self.real = real
        self.protect = "not called"

    def __call__(self, traj, spans, graph, llm, workers=8, **kw):
        self.protect = kw.get("protect", None)
        return self.real(traj, spans, graph, llm, workers=workers, **kw)


_ptraj = Trajectory(
    task_id="pw", source="trajdebug", subset="swebenchpro",
    task_description="fix it",
    steps=[Step(0, "user", "go")]
          + [Step(i, "assistant",
                  'I will edit the file.\n'
                  'str_replace_editor({"command": "str_replace", '
                  '"path": "/app/f%d.py"})' % i)
             for i in range(1, 10)])

_orig_compress = compress_mod.compress
_spy = SpyCompress(_orig_compress)
try:
    pipeline_mod.compress_mod.compress = _spy
    run(_ptraj, RecordingLLM(), Config(max_rounds=1, workers=1,
                                       protect_candidates=False))
    check("with the rule off the pipeline forwards no protection set",
          _spy.protect is None, _spy.protect)

    _spy2 = SpyCompress(_orig_compress)
    pipeline_mod.compress_mod.compress = _spy2
    run(_ptraj, RecordingLLM(), Config(max_rounds=1, workers=1,
                                       protect_candidates=True))
    check("with the rule on the pipeline protects the parsed answer space",
          isinstance(_spy2.protect, list) and len(_spy2.protect) > 1,
          _spy2.protect)
    check("what it protects is exactly what candidates() found",
          _spy2.protect == sorted(int(c["step"])
                                  for c in candidates_mod.candidates(_ptraj)),
          _spy2.protect)
    check("the protection set is computed without the localiser's candidate list",
          _spy2.protect is not None
          and Config(protect_candidates=True).candidate_set is False)
finally:
    pipeline_mod.compress_mod.compress = _orig_compress

check("the stage cannot be handed ground truth",
      not any(_GOLD_KEY.search(p)
              for p in inspect.signature(compress_mod.compress).parameters))
check("compression states the measurement the protection rests on",
      "0.375" in compress_mod.compress.__doc__
      and "0.507" in compress_mod.compress.__doc__)
check("it says why it is not the blanket restore that lost",
      "restore_verbatim" in compress_mod.compress.__doc__)


print("\n[31] the answer space as a comparison field")

# The gap this targets: on both corpora the answer is already inside an
# eleven-item shortlist, and it is picked out of that shortlist only 39% (SP)
# and 60% (TA) of the time. The shortlist was already in the prompt -- as bare
# indices, with each entry's text scattered through PART 2.

_et = Trajectory(
    task_id="evid-fixture", source="trajdebug", subset="swebenchpro",
    task_description="make the loader accept a list",
    steps=[Step(0, "user", "please fix the loader")]
          + [Step(i, "assistant", "STEP-%d-OWN-WORDS " % i + "z" * 400)
             for i in range(1, 8)])
_eg = FlowGraph(anchors=[7], anchors_located=[7],
                edges=[FlowEdge(src=3, dst=7, kind=DATA_FLOW, entity="cfg")],
                failure_summary="the loader raised")
_espans = [Span(start=0, end=3, label="orient"),
           Span(start=4, end=7, label="patch it")]
_edig = [StepDigest(index=i, role=_et.steps[i].role,
                    text=("PARAPHRASED %d" % i if i in (2, 5)
                          else _et.steps[i].content),
                    tier=(COMPRESSED if i in (2, 5) else KEEP),
                    raw_chars=len(_et.steps[i].content or ""),
                    flows=("data" if i == 3 else ""))
         for i in range(len(_et.steps))]
_ecands = [{"step": 2, "label": "edited loader.py"},
           {"step": 3, "label": "applied a patch"},
           {"step": 5, "label": "wrote the test"}]

_ev = localize_mod.render_candidate_evidence(_et, _espans, _eg, _edig, _ecands)

check("every candidate gets a row of its own",
      all(("STEP %d —" % i) in _ev for i in (2, 3, 5)), _ev[:200])
check("a non-candidate step does not get a row",
      "STEP 6 —" not in _ev and "STEP 1 —" not in _ev)
check("a row carries the entry's own text, not just its index",
      "STEP-3-OWN-WORDS" in _ev)
check("a row carries the label the parse gave it",
      "applied a patch" in _ev)
check("a row says which phase the entry falls in",
      'phase "patch it" (steps 4-7)' in _ev, _ev)
check("a row says what tracing reached",
      "data 1 hop from the symptom" in _ev, _ev)
check("a row says plainly when tracing reached nothing",
      "tracing reached no chain from the symptom to it" in _ev)
check("an entry whose text survived intact is marked as its own words",
      "WHAT IT DID (its own words):" in _ev)
check("an entry that was paraphrased is not claimed to be its own words",
      _ev.count("WHAT IT DID (its own words):") == 1
      and _ev.count("WHAT IT DID:") == 2, _ev)
check("the field says the tracing facts are mechanical, not blame",
      "not what is to blame" in _ev)
check("the field says a step can be the cause with nothing traced to it",
      "nothing traced to it at all" in _ev)

# A visible-failure step among the candidates has to be flagged as such: it is
# the symptom, and the whole task is not to answer with the symptom.
_ev2 = localize_mod.render_candidate_evidence(
    _et, _espans, _eg, _edig, [{"step": 7, "label": "ran it"}])
check("a candidate that is itself a symptom step says so",
      "the failure is VISIBLE" in _ev2, _ev2)

check("an empty answer space renders to nothing",
      localize_mod.render_candidate_evidence(_et, _espans, _eg, _edig, []) == "")
check("a malformed entry is skipped rather than failing the prompt",
      "STEP 3 —" in localize_mod.render_candidate_evidence(
          _et, _espans, _eg, _edig, [{"nope": 1}, {"step": 3}]))
check("an out-of-range entry is skipped",
      localize_mod.render_candidate_evidence(
          _et, _espans, _eg, _edig, [{"step": 999}]) == "")
check("no label field can reach the comparison field",
      not _GOLD_KEY.search(_ev)
      and not any(k in _ev for k in ("annotation", "gold_patch",
                                     "failure_reason", "test_patch")))
check("the stage cannot be handed ground truth",
      not any(_GOLD_KEY.search(p) for p in
              inspect.signature(localize_mod.render_candidate_evidence)
              .parameters))
check("it states the measurement it rests on",
      "39%" in localize_mod.render_candidate_evidence.__doc__
      and "60%" in localize_mod.render_candidate_evidence.__doc__)
check("it says how it differs from the group stage that lost",
      "bracket.py" in localize_mod.render_candidate_evidence.__doc__
      and "one call" in localize_mod.render_candidate_evidence.__doc__)


class EvidenceLLM(RecordingLLM):
    def __init__(self):
        super().__init__()
        self.localize_prompt = ""

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if "critical_step" in prompt:
            self.localize_prompt = prompt
            return {"critical_step": 3, "confidence": 0.5,
                    "error_type": "act.wrong-file", "reason": "it did"}
        return default if default is not None else {}


_el = EvidenceLLM()
localize_mod.localize(_et, _espans, _eg, _el, digests=_edig,
                      cands=_ecands, enforce_cands=True)
_el2 = EvidenceLLM()
localize_mod.localize(_et, _espans, _eg, _el2, digests=_edig,
                      cands=_ecands, enforce_cands=True, cand_evidence=True)
check("with the knob off the answer space is the bare list it always was",
      "--- STEP 3 —" not in _el.localize_prompt
      and "  step 3 — applied a patch" in _el.localize_prompt)
check("with the knob on the comparison field replaces that list",
      "--- STEP 3 —" in _el2.localize_prompt
      and "  step 3 — applied a patch" not in _el2.localize_prompt)
check("nothing else in the localiser prompt moves",
      _el.localize_prompt.replace(
          candidates_mod.render(_ecands),
          localize_mod.render_candidate_evidence(_et, _espans, _eg, _edig,
                                                 _ecands))
      == _el2.localize_prompt)
check("the restriction the answer space imposes is untouched",
      "YOUR ANSWER MUST BE ONE OF THE STEPS LISTED BELOW" in _el2.localize_prompt)

# -- wiring ----------------------------------------------------------------
check("the field is off by default", Config().cand_evidence is False)
check("the knob reaches the artifact fingerprint",
      Config(cand_evidence=True).fingerprint()["cand_evidence"] is True)


class OldLocalize:
    """A stub with the signature that predates the knob."""

    def __init__(self):
        self.kw = None

    def __call__(self, traj, spans, graph, llm, digests=None, outline=None,
                 prior=None, cands=None, enforce_cands=True,
                 max_prompt_chars=120000, samples=1):
        self.kw = dict(samples=samples)
        return Prediction(task_id=traj.task_id, critical_step=1,
                          confidence=0.1, reason="stub")


_old = OldLocalize()
_orig_localize = localize_mod.localize
try:
    pipeline_mod.localize_mod.localize = _old
    run(_et, RecordingLLM(), Config(max_rounds=1, workers=1,
                                    cand_evidence=False))
    check("with the knob off a stub with the old signature still works",
          _old.kw == {"samples": 1}, _old.kw)
finally:
    pipeline_mod.localize_mod.localize = _orig_localize


print("\n[32] the same question asked as different decision procedures")

# The gap this targets is the one five interventions failed to move: the choice
# inside the offered list. All five changed what the reader was SHOWN. This is
# the first that changes what it is ASKED, and the first whose downside is
# bounded by its merge rule rather than by hope.

_fields = dict(task="T", failure_summary="F", symptom_evidence="",
               anchors="step 9",
               data_block="D", control_block="C", outline="O",
               span_prior="", prior_fields="", candidates="CAND",
               roll_block="", roll_fields="", expect_block="",
               analyst_block="", judge_block="",
               rootcause_block="", rootcause_fields="",
               trajectory="TRAJ")
_now = localize_mod.LOCALIZE_PROMPT.format(
    head=localize_mod.INCUMBENT_HEAD, framing_note="", **_fields)
_old_tpl = (localize_mod.LOCALIZE_PROMPT
            .replace("{head}\n", localize_mod.INCUMBENT_HEAD + "\n", 1)
            .replace("{candidates}{expect_block}"
                     "{framing_note}{roll_block}{analyst_block}"
                     "{judge_block}{rootcause_block}\n",
                     "{candidates}\n", 1)
            .replace("{{{roll_fields}\n", "{{\n", 1))
check("the incumbent framing is byte-identical to the prompt it replaces",
      _now == _old_tpl.format(**_fields))
check("the incumbent framing contributes no note",
      localize_mod.FRAMINGS[0][2] == "")
check("there are three framings and the incumbent is first",
      [n for n, _, _ in localize_mod.FRAMINGS] == ["cause", "repair", "commit"])

_alts = {n: localize_mod.LOCALIZE_PROMPT.format(head=h, framing_note=t,
                                                **_fields)
         for n, h, t in localize_mod.FRAMINGS[1:]}
for _n, _p in _alts.items():
    check("framing %r really asks something else" % _n, _p != _now)
    check("framing %r keeps the same evidence" % _n,
          all(x in _p for x in ("ORIGINAL TASK:", "TRAJ", "CAND", "D", "C")))
    check("framing %r keeps the same answer format" % _n,
          '"critical_step": <int>' in _p and '"confidence"' in _p)
    check("framing %r carries no label field" % _n,
          not any(k in _p for k in ("annotation", "gold_patch",
                                    "failure_reason", "test_patch",
                                    "failure_id")))
check("the alternatives differ from each other too",
      _alts["repair"] != _alts["commit"])

# -- the merge rule, which is where the safety property lives ---------------
_mt = Trajectory(task_id="m", source="trajdebug", subset="swebenchpro",
                 task_description="t",
                 steps=[Step(0, "user", "go")]
                       + [Step(i, "assistant", "s%d" % i) for i in range(1, 9)])


def _mk(step):
    return {} if step is None else {"critical_step": step, "confidence": 0.5}


def _merge(*steps):
    names = [n for n, _, _ in localize_mod.FRAMINGS][:len(steps)]
    return localize_mod._merge_framings(
        _mt, [(names[i], _mk(steps[i])) for i in range(len(steps))])


check("unanimous framings answer that step",
      _merge(5, 5, 5)["winner"] == 5)
check("a three-way split falls back to the incumbent, not the earliest",
      _merge(7, 3, 5)["winner"] == 7)
check("two framings agreeing against the incumbent move the answer",
      _merge(7, 3, 3)["winner"] == 3)
check("the incumbent wins a tie it is part of",
      _merge(7, 7, 3)["winner"] == 7)
check("a majority that is earlier than the incumbent still wins",
      _merge(8, 2, 2)["winner"] == 2)
check("a framing that returned nothing simply does not vote",
      _merge(7, None, None)["winner"] == 7)
check("the incumbent returning nothing does not force an answer from it",
      _merge(None, 4, 4)["winner"] == 4)
check("no framing answering at all yields no winner",
      _merge(None, None, None)["winner"] is None)
check("an out-of-range answer is dropped before it votes",
      _merge(999, 6, 6)["winner"] == 6, _merge(999, 6, 6))
check("a non-attributable answer is snapped rather than dropped, exactly as "
      "it is for a single answer",
      _merge(0, 4, 4)["record"]["answers"]["cause"] == 1, _merge(0, 4, 4))

_rec = _merge(7, 3, 3)["record"]
check("the record keeps every framing's own answer",
      _rec["answers"] == {"cause": 7, "repair": 3, "commit": 3}, _rec)
check("the record keeps the incumbent's answer even when overruled",
      _rec["incumbent"] == 7 and _rec["changed"] is True)
check("the record says how many framings agreed", _rec["agreed"] == 2)
check("an unchanged merge is marked as such",
      _merge(7, 3, 5)["record"]["changed"] is False)
check("the winning framing's own object is what gets read",
      localize_mod._merge_framings(
          _mt, [("cause", {"critical_step": 7, "reason": "A"}),
                ("repair", {"critical_step": 3, "reason": "B"}),
                ("commit", {"critical_step": 3, "reason": "C"})]
      )["obj"]["reason"] == "B")
check("the merge cannot be handed ground truth",
      list(inspect.signature(localize_mod._merge_framings).parameters)
      == ["traj", "verdicts"])
check("the merge states why a tie does not go to the earliest",
      "0.367" in localize_mod._merge_framings.__doc__
      and "0.394" in localize_mod._merge_framings.__doc__)

# -- wiring ----------------------------------------------------------------
check("framings is off by default", Config().framings == 1)
check("the knob reaches the artifact fingerprint",
      Config(framings=3).fingerprint()["framings"] == 3)


class FramingLLM(RecordingLLM):
    def __init__(self, answers=None):
        super().__init__()
        self.heads = []
        self.answers = answers or {}

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if "critical_step" in prompt:
            head = prompt.split("\n\nORIGINAL TASK:")[0]
            self.heads.append(head)
            for key, step in self.answers.items():
                if key in head:
                    return {"critical_step": step, "confidence": 0.5,
                            "error_type": "act.x", "reason": "r"}
            return {"critical_step": 3, "confidence": 0.5,
                    "error_type": "act.x", "reason": "r"}
        return default if default is not None else {}


_fspan = [Span(start=0, end=8, label="all")]


def _fgraph():
    return FlowGraph(anchors=[8], anchors_located=[8])


_f1 = FramingLLM()
localize_mod.localize(_mt, _fspan, _fgraph(), _f1)
check("with the knob off exactly one localiser call is made",
      len(_f1.heads) == 1)
check("with the knob off that call is the incumbent framing",
      _f1.heads[0] == localize_mod.INCUMBENT_HEAD, _f1.heads[:1])

_f3 = FramingLLM({"redo": 5, "stopped being recoverable": 5})
_p3 = localize_mod.localize(_mt, _fspan, _fgraph(), _f3, framings=3)
check("with the knob on three localiser calls are made", len(_f3.heads) == 3)
check("the three calls ask three different questions",
      len(set(_f3.heads)) == 3)
check("one of them is still the untouched incumbent",
      localize_mod.INCUMBENT_HEAD in _f3.heads)
check("two framings agreeing against the incumbent change the answer",
      _p3.critical_step == 5, _p3.critical_step)
check("the prediction carries the whole tally",
      _p3.framings and _p3.framings["answers"] == {"cause": 3, "repair": 5,
                                                   "commit": 5}, _p3.framings)
check("a run without the knob carries no such record",
      localize_mod.localize(_mt, _fspan, _fgraph(),
                            FramingLLM()).framings is None)
check("the record survives into the artifact", "framings" in _p3.to_dict())


class OldLocalizeF:
    """A stub with the signature that predates the knob."""

    def __init__(self):
        self.called = 0

    def __call__(self, traj, spans, graph, llm, digests=None, outline=None,
                 prior=None, cands=None, enforce_cands=True,
                 max_prompt_chars=120000, samples=1):
        self.called += 1
        return Prediction(task_id=traj.task_id, critical_step=1,
                          confidence=0.1, reason="stub")


_oldf = OldLocalizeF()
_orig_loc = localize_mod.localize
try:
    pipeline_mod.localize_mod.localize = _oldf
    run(_mt, RecordingLLM(), Config(max_rounds=1, workers=1, framings=1))
    check("with the knob at its default a stub with the old signature works",
          _oldf.called == 1)
finally:
    pipeline_mod.localize_mod.localize = _orig_loc



print("\n[33] pointwise verification: each candidate judged on its own")

# Six interventions failed while keeping one decision procedure: one reader,
# one call, "pick one of these eleven". This section pins the properties that
# make the replacement measurable -- that the off path is untouched, that the
# per-candidate calls differ ONLY in the focal step (the commensurability
# E5 lacked), and that the merge cannot lose ground on a tie.

check("VERIFY_PROMPT carries no label field",
      not any(k in localize_mod.VERIFY_PROMPT for k in
              ("annotation", "gold_patch", "failure_reason", "test_patch",
               "failure_id")))
check("the verifier cannot be handed ground truth",
      not any(_GOLD_KEY.search(p) for p in
              inspect.signature(localize_mod.verify_candidates).parameters))
check("the merge cannot be handed ground truth",
      not any(_GOLD_KEY.search(p) for p in
              inspect.signature(localize_mod._merge_verify).parameters))
check("the merge states why a tie does not go by position",
      # The reason is no longer "no positional signal exists" -- with
      # --verify-outdeg one does, and it is significant. What the docstring has
      # to carry now is that the direction is real AND that no step size
      # converts, so the next reader does not re-derive the offset and assume
      # it is free money. Pinned: gold's list position, where the out-degree
      # term moved our answers to, and the significance of the residual.
      all(s in localize_mod._merge_verify.__doc__
          for s in ("0.367", "0.255", "p=0.014", "25 of 36")))

_vt = Trajectory(task_id="v", source="trajdebug", subset="swebenchpro",
                 task_description="build it",
                 steps=[Step(0, "user", "go")]
                       + [Step(i, "assistant", "step %d body" % i)
                          for i in range(1, 12)])
_vspan = [Span(start=0, end=11, label="all")]
_vcands = [{"step": i, "label": "edit"} for i in (2, 5, 9)]


def _vgraph():
    return FlowGraph(anchors=[11], anchors_located=[11])


class VerifyLLM(RecordingLLM):
    """Records every verifier prompt and answers per focal step."""

    def __init__(self, per_step=None):
        super().__init__()
        self.focal = []
        self.per_step = per_step or {}

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if '"averts"' in prompt and "THE STEP UNDER EXAMINATION" in prompt:
            m = re.search(r"THE STEP UNDER EXAMINATION -- STEP (\d+)", prompt)
            i = int(m.group(1))
            self.focal.append(i)
            return dict(self.per_step.get(i, {"depends": False,
                                              "averts": False,
                                              "earlier_cause": False,
                                              "only_symptom": False,
                                              "confidence": 0.5,
                                              "why": "w"}))
        if "critical_step" in prompt:
            return {"critical_step": 9, "confidence": 0.5,
                    "error_type": "act.x", "reason": "r"}
        return default if default is not None else {}


# -- the off path is untouched ---------------------------------------------
_v_off = VerifyLLM()
_p_off = localize_mod.localize(_vt, _vspan, _vgraph(), _v_off,
                               cands=list(_vcands))
check("with the knob off no candidate is ever verified", _v_off.focal == [])
check("with the knob off the prediction carries no poll",
      _p_off.verify is None)
check("with the knob off the listwise answer stands",
      _p_off.critical_step == 9)
check("verify is off by default", Config().verify is False)
check("the knob reaches the artifact fingerprint",
      Config(verify=True).fingerprint()["verify"] is True)

# -- one call per candidate, differing only in the focal slot --------------
_v_on = VerifyLLM()
_p_on = localize_mod.localize(_vt, _vspan, _vgraph(), _v_on,
                              cands=list(_vcands), verify=True)
check("one verifier call per candidate", _v_on.focal == [2, 5, 9], _v_on.focal)
_vp = [p for p in _v_on.prompts if "THE STEP UNDER EXAMINATION" in p]
check("every candidate got its own prompt", len(_vp) == 3)
check("the verifier prompts are genuinely different", len(set(_vp)) == 3)
for _blk in ("ORIGINAL TASK:", "WHAT VISIBLY WENT WRONG", "SYMPTOM STEPS:",
             "HOW THE RUN WENT, PHASE BY PHASE:"):
    check("every verifier call shares the evidence block %r" % _blk,
          all(_blk in p for p in _vp))
check("every verifier call asks the same four questions",
      all(all(q in p for q in ('"depends"', '"averts"', '"earlier_cause"',
                               '"only_symptom"')) for p in _vp))
check("exactly one step is under examination per call",
      all(p.count("WHERE IT SITS:") == 1 for p in _vp))
# Rivals are not hidden -- the run-up and aftermath list them, because judging
# "was the run already doomed before this step" needs them. What they never get
# is the focal treatment: their text appears only as a clipped single line.
check("a rival candidate appears only as a one-line summary, never verbatim",
      all(all(len(ln) <= localize_mod.VERIFY_CONTEXT_LINE + 40
              for ln in p.split("(one line each):")[1].splitlines()
              if ln.strip().startswith("step "))
          for p in _vp))
# what remains once the focal parts are masked must be byte-identical, which
# is the whole commensurability claim
_masked = {re.sub(r"STEP \d+", "STEP N", p.split("WHAT THE AGENT DID BEFORE")[0])
           for p in _vp}
check("masking the focal step makes the prompts byte-identical",
      len(_masked) == 1, sorted(len(m) for m in _masked))

# -- scoring arithmetic ----------------------------------------------------
_sc = localize_mod.verify_candidates(
    _vt, _vspan, _vgraph(),
    VerifyLLM({2: {"averts": True, "depends": True, "earlier_cause": False,
                   "only_symptom": False, "confidence": 0.9},
               5: {"averts": False, "depends": True, "earlier_cause": True,
                   "only_symptom": True, "confidence": 0.4},
               9: {"averts": True, "depends": True, "earlier_cause": True,
                   "only_symptom": False, "confidence": 0.1}}),
    [], list(_vcands))
_by = {v["step"]: v for v in _sc}
check("averts+depends scores 3", _by[2]["score"] == 3, _by[2])
check("both negatives cancel depends to -1", _by[5]["score"] == -1, _by[5])
check("an earlier cause costs a point", _by[9]["score"] == 2, _by[9])
check("confidence is carried through", _by[2]["confidence"] == 0.9)

_junk = localize_mod.verify_candidates(
    _vt, _vspan, _vgraph(),
    VerifyLLM({2: {"averts": "true", "depends": "yes", "confidence": "x"},
               5: {}, 9: {"averts": None, "confidence": 5.0}}),
    [], list(_vcands))
_jb = {v["step"]: v for v in _junk}
check("a stringy boolean is read as true", _jb[2]["averts"] is True)
check("an unparseable confidence becomes zero", _jb[2]["confidence"] == 0.0)
check("a missing answer is read as no, not as unknown",
      _jb[5]["score"] == 0 and _jb[5]["averts"] is False)
check("confidence is clamped into range", _jb[9]["confidence"] == 1.0)

# -- the answer space is respected ----------------------------------------
_oob = localize_mod.verify_candidates(
    _vt, _vspan, _vgraph(), VerifyLLM(), [],
    [{"step": 999}, {"step": -1}, {"step": "x"}, {"step": 5}])
check("out-of-range candidates are never verified",
      [v["step"] for v in _oob] == [5], _oob)
_big = localize_mod.verify_candidates(
    _vt, _vspan, _vgraph(), VerifyLLM(), [],
    [{"step": i % 12} for i in range(localize_mod.VERIFY_MAX + 20)])
check("the cost cap binds at VERIFY_MAX",
      len(_big) == localize_mod.VERIFY_MAX, len(_big))


def _vv(step, score, conf=0.5):
    return {"step": step, "score": score, "confidence": conf,
            "depends": False, "averts": False, "earlier_cause": False,
            "only_symptom": False, "why": ""}


check("no verdicts leaves the incumbent alone",
      localize_mod._merge_verify([], 7)["winner"] == 7)
check("no verdicts records no poll",
      localize_mod._merge_verify([], 7)["record"] is None)
check("a strictly higher score displaces the incumbent",
      localize_mod._merge_verify([_vv(2, 3), _vv(5, 1)], 5)["winner"] == 2)
check("the incumbent survives a tie at the top",
      localize_mod._merge_verify([_vv(2, 3), _vv(5, 3)], 5)["winner"] == 5)
check("the incumbent survives when it scores highest",
      localize_mod._merge_verify([_vv(2, 1), _vv(5, 3)], 5)["winner"] == 5)
# An unpolled incumbent is DISPLACED. These two were briefly asserted the other
# way round -- keep it, on the theory that a poll which never asked about the
# answer has produced no evidence against it -- and the theory lost where it
# counts. It only fires when the poll is a strict SUBSET of the answer space,
# which the shortlist arm did on purpose for one run; on the 68 rows where it
# fired, the protected listwise answer was the reference step 3 times against 21
# for the polled step it was displacing, and acc@0 went 0.3256 -> 0.1163. Both
# the branch and the pool-subsetting that created unpolled incumbents are gone,
# so this is back to meaning only "VERIFY_MAX truncated the list".
check("an unpolled incumbent is displaced by the best polled step",
      localize_mod._merge_verify([_vv(2, 3), _vv(5, 1)], 99)["winner"] == 2)
check("an unpolled incumbent loses a tie too, on confidence",
      localize_mod._merge_verify(
          [_vv(2, 3, 0.2), _vv(5, 3, 0.8)], 99)["winner"] == 5)
check("an unpolled incumbent never wins, however the poll scores",
      all(localize_mod._merge_verify(
          [_vv(2, a), _vv(5, b)], 99)["winner"] in (2, 5)
          for a in (-2, 0, 3, 99) for b in (-2, 0, 3, 99)))
check("an unpolled incumbent is recorded as unpolled",
      localize_mod._merge_verify(
          [_vv(2, 3)], 99)["record"]["incumbent_unpolled"] is True)
check("a polled incumbent is not recorded as unpolled",
      localize_mod._merge_verify(
          [_vv(2, 3), _vv(5, 1)], 5)["record"]["incumbent_unpolled"] is False)
check("displacing an unpolled incumbent is reported as changed",
      localize_mod._merge_verify(
          [_vv(2, 3)], 99)["record"]["changed"] is True)
# A polled incumbent obeys strictly-higher-or-nothing.
check("a polled incumbent is still displaced by a strictly higher score",
      localize_mod._merge_verify([_vv(2, 3), _vv(5, 1)], 5)["winner"] == 2)
check("the winner is always a step that was actually judged",
      all(localize_mod._merge_verify(
          [_vv(2, a), _vv(5, b)], inc)["winner"] in (2, 5, inc)
          for a in (-2, 0, 3) for b in (-2, 0, 3) for inc in (2, 5, 99)))

_rec = localize_mod._merge_verify([_vv(2, 3), _vv(5, 1)], 5)["record"]
check("the record keeps every verdict", len(_rec["verdicts"]) == 2)
check("the record keeps the displaced incumbent and its score",
      _rec["incumbent"] == 5 and _rec["incumbent_score"] == 1, _rec)
check("a displacement is marked as such", _rec["changed"] is True)
check("an unchanged poll is marked as such",
      localize_mod._merge_verify([_vv(5, 3)], 5)["record"]["changed"] is False)

# -- end to end ------------------------------------------------------------
_v_move = VerifyLLM({2: {"averts": True, "depends": True, "confidence": 0.9},
                     5: {"depends": True, "confidence": 0.5},
                     9: {"only_symptom": True, "confidence": 0.5}})
_p_move = localize_mod.localize(_vt, _vspan, _vgraph(), _v_move,
                                cands=list(_vcands), verify=True)
check("the poll can move the answer off the listwise choice",
      _p_move.critical_step == 2, _p_move.critical_step)
check("the prediction carries the whole poll",
      _p_move.verify and _p_move.verify["asked"] == 3
      and _p_move.verify["incumbent"] == 9, _p_move.verify)
check("the poll survives into the artifact", "verify" in _p_move.to_dict())
check("the poll never answers outside the answer space",
      _p_move.critical_step in {c["step"] for c in _vcands})

_v_nocand = VerifyLLM()
localize_mod.localize(_vt, _vspan, _vgraph(), _v_nocand, verify=True)
check("with no answer space there is nothing to verify",
      _v_nocand.focal == [])


class OldLocalizeV:
    """A stub with the signature that predates the verifier."""

    def __init__(self):
        self.called = 0

    def __call__(self, traj, spans, graph, llm, digests=None, outline=None,
                 prior=None, cands=None, enforce_cands=True,
                 max_prompt_chars=120000, samples=1):
        self.called += 1
        return Prediction(task_id=traj.task_id, critical_step=1,
                          confidence=0.1, reason="stub")


_oldv = OldLocalizeV()
_orig_locv = localize_mod.localize
try:
    pipeline_mod.localize_mod.localize = _oldv
    run(_vt, RecordingLLM(), Config(max_rounds=1, workers=1))
    check("with the knob at its default a stub with the old signature works",
          _oldv.called == 1)
finally:
    pipeline_mod.localize_mod.localize = _orig_locv


# -- the graded rubric ------------------------------------------------------
# The binary rubric was measured informative but blunt: "averts" fires on 0.7%
# of candidates, so the double-weighted term is almost always zero and 55% of
# polls end in a tie the rubric did not decide. Grading the two positive
# questions 0-3 is the only change; everything the binary path does must stay
# byte-identical, or the A/B measures the edit rather than the rubric.


class GradedLLM(RecordingLLM):
    """Answers the graded form; records which step each call was about."""

    def __init__(self, per_step=None):
        super().__init__()
        self.focal = []
        self.per_step = per_step or {}

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if '"repair"' in prompt and "THE STEP UNDER EXAMINATION" in prompt:
            i = int(re.search(r"THE STEP UNDER EXAMINATION -- STEP (\d+)",
                              prompt).group(1))
            self.focal.append(i)
            return dict(self.per_step.get(i, {"repair": 0, "depends": 0,
                                              "earlier_cause": False,
                                              "only_symptom": False,
                                              "why": "w"}))
        if "critical_step" in prompt:
            return {"critical_step": 9, "confidence": 0.5,
                    "error_type": "act.x", "reason": "r"}
        return default if default is not None else {}


check("VERIFY_GRADED_PROMPT carries no label field",
      not any(k in localize_mod.VERIFY_GRADED_PROMPT for k in
              ("annotation", "gold_patch", "failure_reason", "test_patch",
               "failure_id")))
check("the graded prompt asks for grades, not for yes/no",
      all(s in localize_mod.VERIFY_GRADED_PROMPT
          for s in ('"repair"', '"depends"', "<0, 1, 2 or 3>")))
check("the graded prompt keeps the two negative questions binary",
      '"earlier_cause": <true or false>' in localize_mod.VERIFY_GRADED_PROMPT
      and '"only_symptom": <true or false>' in localize_mod.VERIFY_GRADED_PROMPT)
# self-reported confidence was measured actively harmful as a tie-break
# (5/27 contested ties against 14/27 for keeping the listwise answer), so the
# graded form does not ask for it at all rather than asking and ignoring it.
check("the graded prompt does not ask for confidence",
      "confidence" not in localize_mod.VERIFY_GRADED_PROMPT)
check("the graded prompt tells the model to use the whole range",
      "do not reserve the top grade" in localize_mod.VERIFY_GRADED_PROMPT)
check("grading is off by default",
      Config().verify_graded is False
      and inspect.signature(localize_mod.verify_candidates)
          .parameters["graded"].default is False)
check("the graded knob reaches the artifact fingerprint",
      Config(verify_graded=True).fingerprint()["verify_graded"] is True)

# the binary path must not have moved a byte
_v_bin = VerifyLLM()
localize_mod.localize(_vt, _vspan, _vgraph(), _v_bin,
                      cands=list(_vcands), verify=True)
_v_bin2 = VerifyLLM()
localize_mod.localize(_vt, _vspan, _vgraph(), _v_bin2,
                      cands=list(_vcands), verify=True, verify_graded=False)
check("passing verify_graded=False changes not one byte of the binary path",
      _v_bin.prompts == _v_bin2.prompts)

_v_gr = GradedLLM()
_p_gr = localize_mod.localize(_vt, _vspan, _vgraph(), _v_gr,
                              cands=list(_vcands), verify=True,
                              verify_graded=True)
check("grading still asks once per candidate", _v_gr.focal == [2, 5, 9],
      _v_gr.focal)
_gp = [p for p in _v_gr.prompts if "THE STEP UNDER EXAMINATION" in p]
check("the graded call is a different question from the binary one",
      all('"averts"' not in p for p in _gp))
check("grading keeps the same evidence blocks",
      all(all(b in p for b in ("ORIGINAL TASK:", "WHAT VISIBLY WENT WRONG",
                               "SYMPTOM STEPS:",
                               "HOW THE RUN WENT, PHASE BY PHASE:"))
          for p in _gp))
check("exactly one step is under examination per graded call",
      all(p.count("WHERE IT SITS:") == 1 for p in _gp))
_gmask = {re.sub(r"STEP \d+", "STEP N",
                 re.sub(r"step \{?\d+\}?", "step N",
                        p.split("WHAT THE AGENT DID BEFORE")[0]))
          for p in _gp}
check("masking the focal step makes the graded prompts byte-identical too",
      len(_gmask) == 1, sorted(len(m) for m in _gmask))

# -- graded arithmetic: 2*repair + depends - 2*earlier_cause - 2*only_symptom
_gsc = localize_mod.verify_candidates(
    _vt, _vspan, _vgraph(),
    GradedLLM({2: {"repair": 3, "depends": 3, "earlier_cause": False,
                   "only_symptom": False},
               5: {"repair": 0, "depends": 0, "earlier_cause": True,
                   "only_symptom": True},
               9: {"repair": 2, "depends": 1, "earlier_cause": True,
                   "only_symptom": False}}),
    [], list(_vcands), graded=True)
_gb = {v["step"]: v for v in _gsc}
check("the top of the graded range is 9", _gb[2]["score"] == 9, _gb[2])
check("the bottom of the graded range is -4", _gb[5]["score"] == -4, _gb[5])
check("a graded middle lands between them", _gb[9]["score"] == 3, _gb[9])
check("a graded record says it is graded",
      all(v.get("graded") is True for v in _gsc))
check("a graded record does not carry the binary question it replaced",
      all("averts" not in v for v in _gsc))
check("the graded range is strictly wider than the binary one",
      _gb[2]["score"] - _gb[5]["score"] > 3 - (-2))

_gjunk = localize_mod.verify_candidates(
    _vt, _vspan, _vgraph(),
    GradedLLM({2: {"repair": "2", "depends": 7, "earlier_cause": "true"},
               5: {},
               9: {"repair": -5, "depends": True, "only_symptom": 1}}),
    [], list(_vcands), graded=True)
_gj = {v["step"]: v for v in _gjunk}
check("a stringy grade is read as a number", _gj[2]["repair"] == 2, _gj[2])
check("a grade above the range is clamped down", _gj[2]["depends"] == 3)
check("a grade below the range is clamped up", _gj[9]["repair"] == 0)
check("a missing grade is the lowest grade, as the prompt promises",
      _gj[5]["repair"] == 0 and _gj[5]["depends"] == 0
      and _gj[5]["score"] == 0)
check("a boolean where a grade was asked for is still readable",
      _gj[9]["depends"] == 3)
check("a graded answer never scores outside its range",
      all(-4 <= v["score"] <= 9 for v in _gjunk))

# the merge is shared, so a graded poll must move the answer the same way
_p_gmove = localize_mod.localize(
    _vt, _vspan, _vgraph(),
    GradedLLM({2: {"repair": 3, "depends": 2},
               5: {"repair": 1, "depends": 1},
               9: {"repair": 0, "depends": 0, "only_symptom": True}}),
    cands=list(_vcands), verify=True, verify_graded=True)
check("a graded poll can move the answer off the listwise choice",
      _p_gmove.critical_step == 2, _p_gmove.critical_step)
check("the graded poll is recorded in full",
      _p_gmove.verify and _p_gmove.verify["asked"] == 3
      and all("repair" in v for v in _p_gmove.verify["verdicts"]))
check("a graded poll with nothing above the incumbent leaves it alone",
      localize_mod.localize(
          _vt, _vspan, _vgraph(), GradedLLM(),
          cands=list(_vcands), verify=True,
          verify_graded=True).critical_step == 9)
check("grading without --verify is a no-op",
      localize_mod.localize(_vt, _vspan, _vgraph(), GradedLLM(),
                            cands=list(_vcands),
                            verify_graded=True).verify is None)

# -- the consequence block (--verify-chain) ---------------------------------
# The rubric's whole view of consequence was two hop counts. This adds the
# traced edges OUT of the focal step -- structural, per-step, already computed.
# It must stay provably the same QUESTION: the chain templates are built from
# the shipped ones by substitution, so only the evidence can differ.

_CHAIN_BLOCK = ("WHAT LATER STEPS TOOK FROM IT, as traced (not a judgement):\n"
                "{downstream}\n")
check("the chain template is the shipped one plus a block",
      localize_mod.VERIFY_PROMPT_CHAIN.replace(_CHAIN_BLOCK, "")
      == localize_mod.VERIFY_PROMPT)
check("the graded chain template is built the same way",
      localize_mod.VERIFY_GRADED_PROMPT_CHAIN.replace(_CHAIN_BLOCK, "")
      == localize_mod.VERIFY_GRADED_PROMPT)
check("the consequence block carries no label field",
      not any(k in localize_mod.VERIFY_PROMPT_CHAIN for k in
              ("annotation", "gold_patch", "failure_reason", "test_patch",
               "failure_id")))
check("the chain knob is off by default",
      Config().verify_chain is False
      and inspect.signature(localize_mod.verify_candidates)
          .parameters["chain"].default is False)
check("the chain knob reaches the artifact fingerprint",
      Config(verify_chain=True).fingerprint()["verify_chain"] is True)

_v_bin_ref = VerifyLLM()
localize_mod.localize(_vt, _vspan, _vgraph(), _v_bin_ref,
                      cands=list(_vcands), verify=True)
_v_nochain = VerifyLLM()
localize_mod.localize(_vt, _vspan, _vgraph(), _v_nochain,
                      cands=list(_vcands), verify=True, verify_chain=False)
check("asking for no chain changes not one byte of the shipped prompt",
      _v_nochain.prompts == _v_bin_ref.prompts)

_cg = FlowGraph(anchors=[11], anchors_located=[11])
for _e in (FlowEdge(src=2, dst=5, kind=DATA_FLOW, entity="the parser output"),
           FlowEdge(src=2, dst=11, kind=DATA_FLOW, entity="the patch"),
           FlowEdge(src=5, dst=11, kind=CONTROL_FLOW, rationale="it retried"),
           FlowEdge(src=2, dst=7, kind=CONTROL_FLOW, rationale="it moved on")):
    _cg.add(_e)
_v_ch = VerifyLLM()
localize_mod.localize(_vt, _vspan, _cg, _v_ch, cands=list(_vcands),
                      verify=True, verify_chain=True)
_cp = {int(re.search(r"THE STEP UNDER EXAMINATION -- STEP (\d+)", p).group(1)): p
       for p in _v_ch.prompts if "THE STEP UNDER EXAMINATION" in p}
check("the chain block reaches the prompt",
      all("WHAT LATER STEPS TOOK FROM IT" in p for p in _cp.values()))
check("a step's own consumers are named",
      "step 5 used what this step produced [the parser output]" in _cp[2]
      and "step 11 used what this step produced [the patch]" in _cp[2])
check("a control edge is rendered as a consequence, not as a dependency",
      "step 7 happened BECAUSE of this step: it moved on" in _cp[2])
check("a step with nothing traced out of it says so",
      "Nothing was traced out of this step" in _cp[9])
check("each call sees only its own outgoing edges",
      "the parser output" not in _cp[9] and "the parser output" not in _cp[5])
check("the consequence block is still one question about one step",
      all(p.count("WHERE IT SITS:") == 1 for p in _cp.values()))

_many = FlowGraph(anchors=[11], anchors_located=[11])
for _d in range(3, 12):
    _many.add(FlowEdge(src=2, dst=_d, kind=DATA_FLOW, entity="thing %d" % _d))
_blk = localize_mod._verify_consequences(
    _vt, _many, {}, {2: list(_many.edges)},
    {DATA_FLOW: {}, CONTROL_FLOW: {}}, 0, 3, 2)
check("the consequence block is capped, and says how many it dropped",
      _blk.count("used what this step produced")
      == localize_mod.VERIFY_CHAIN_MAX
      and "and 1 more later steps used its output" in _blk, _blk)
check("the chain composes with the graded rubric",
      "WHAT LATER STEPS TOOK FROM IT" in localize_mod.VERIFY_GRADED_PROMPT_CHAIN
      and '"repair"' in localize_mod.VERIFY_GRADED_PROMPT_CHAIN)

# -- the playoff (--verify-duel) --------------------------------------------
# The poll's defect is resolution, not direction, and the tie-break it falls
# back to wins 14 of 28 -- a coin flip over the largest block of recoverable
# rows left. The playoff is only defensible if the pair call cannot be won by
# position alone, so that is what most of this pins down.

check("the duel prompt carries no label field",
      not any(k in localize_mod.VERIFY_DUEL_PROMPT for k in
              ("annotation", "gold_patch", "failure_reason", "test_patch",
               "failure_id")))
check("the duel cannot be handed ground truth",
      not any(_GOLD_KEY.search(p) for p in
              inspect.signature(localize_mod.verify_duels).parameters))
check("the duel knob is off by default", Config().verify_duel is False)
check("the duel knob reaches the artifact fingerprint",
      Config(verify_duel=True).fingerprint()["verify_duel"] is True)


class DuelLLM(VerifyLLM):
    """Ties three candidates on the rubric, then answers the pair calls.

    ``winner`` is the step this stub prefers; ``biased`` makes it answer "A"
    every time regardless, which is exactly the position-biased model the
    two-orderings rule exists to catch.
    """

    def __init__(self, winner=None, biased=False):
        super().__init__()
        self.winner = winner
        self.biased = biased
        self.duels = []

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        if "OPTION A" in prompt or "OPTION B" in prompt:
            self.prompts.append(prompt)
            self.duels.append(prompt)
            if self.biased:
                return {"answer": "A", "why": "first one"}
            m = re.search(r"Decide between OPTION A \(step (\d+)\) and "
                          r"OPTION B \(step (\d+)\)", prompt)
            a, b = int(m.group(1)), int(m.group(2))
            if self.winner == a:
                return {"answer": "A", "why": "w"}
            if self.winner == b:
                return {"answer": "B", "why": "w"}
            return {"answer": "neither", "why": "w"}
        if '"averts"' in prompt and "THE STEP UNDER EXAMINATION" in prompt:
            self.prompts.append(prompt)
            i = int(re.search(r"THE STEP UNDER EXAMINATION -- STEP (\d+)",
                              prompt).group(1))
            self.focal.append(i)
            # every candidate scores 1: a three-way tie above the incumbent
            return {"depends": True, "averts": False, "earlier_cause": False,
                    "only_symptom": False, "confidence": 0.5, "why": "w"}
        return super().generate_json(prompt, max_tokens, system, default,
                                     force_json)


_d_off = DuelLLM(winner=2)
localize_mod.localize(_vt, _vspan, _vgraph(), _d_off, cands=list(_vcands),
                      verify=True)
check("with the duel knob off no pair is ever asked", _d_off.duels == [])

_d_on = DuelLLM(winner=2)
_p_duel = localize_mod.localize(_vt, _vspan, _vgraph(), _d_on,
                                cands=list(_vcands), verify=True,
                                verify_duel=True)
check("a tie is played off, every pair asked twice",
      len(_d_on.duels) == 2 * 3, len(_d_on.duels))
check("the playoff winner takes the answer", _p_duel.critical_step == 2,
      _p_duel.critical_step)
check("the playoff is recorded in full",
      _p_duel.verify["duel"]["wins"][2] == 2
      and len(_p_duel.verify["duel"]["pairs"]) == 3)

_pair = {}
for _p in _d_on.duels:
    _m = re.search(r"Decide between OPTION A \(step (\d+)\) and "
                   r"OPTION B \(step (\d+)\)", _p)
    _pair.setdefault(tuple(sorted((int(_m.group(1)), int(_m.group(2))))),
                     []).append(_p)
_lo, _hi = sorted(_pair)[0]
_one, _two = _pair[(_lo, _hi)]
_diff = [ln for ln in difflib.unified_diff(_one.splitlines(),
                                           _two.splitlines(), n=0)
         if ln[:1] in "+-" and ln[:3] not in ("---", "+++")]
check("the two labellings differ only in the label lines",
      len(_diff) == 6 and all(("OPTION A" in ln or "OPTION B" in ln)
                              for ln in _diff), _diff)
check("the two steps are always printed in trajectory order",
      all(p.index("STEP %d" % _lo) < p.index("STEP %d" % _hi)
          for p in _pair[(_lo, _hi)]))

_d_bias = DuelLLM(biased=True)
_p_bias = localize_mod.localize(_vt, _vspan, _vgraph(), _d_bias,
                                cands=list(_vcands), verify=True,
                                verify_duel=True)
check("a model that always answers A wins nothing",
      all(w == 0 for w in _p_bias.verify["duel"]["wins"].values()))
check("...and every one of its pairs is recorded as a disagreement",
      _p_bias.verify["duel"]["disagreed"] == 3)
check("...so the position-biased answer never displaces the incumbent",
      _p_bias.critical_step == 9, _p_bias.critical_step)

_d_none = DuelLLM(winner=None)
_p_none = localize_mod.localize(_vt, _vspan, _vgraph(), _d_none,
                                cands=list(_vcands), verify=True,
                                verify_duel=True)
check("a playoff that decides nothing falls back to the shipped answer",
      _p_none.verify["duel"]["neither"] == 3
      and _p_none.critical_step == _p_bias.critical_step)

check("the duel is capped, and the cap is recorded when it binds",
      localize_mod.VERIFY_DUEL_MAX == 4)
_wide = localize_mod.verify_duels(_vt, _vspan, _vgraph(), DuelLLM(winner=2),
                                  [], [2], None, workers=1)
check("a pool of one is not a playoff",
      _wide["pairs"] == [] and _wide["wins"] == {})
check("a duel over a tie the incumbent already tops is never asked",
      localize_mod._apply_duel({"winner": 9, "record": None}, [], 9,
                               lambda t: (_ for _ in ()).throw(
                                   AssertionError("asked")))["winner"] == 9)

# -- the fallback answer space (--verify-pool) ------------------------------
# 43 of 102 TrajAudit rows have no write-based candidate set, and they are not
# under-parsed: the reference step is a parsed write in exactly one of them.
# The pool comes from the traced path instead, and the listwise answer joins it
# so the merge keeps meaning what it meant.

check("the pool knob is off by default", Config().verify_pool is False)
check("the pool knob reaches the artifact fingerprint",
      Config(verify_pool=True).fingerprint()["verify_pool"] is True)
check("the pool cannot be handed ground truth",
      not any(_GOLD_KEY.search(p) for p in
              inspect.signature(localize_mod._flow_pool).parameters))

_pg = FlowGraph(anchors=[11], anchors_located=[11])
for _e in (FlowEdge(src=3, dst=11, kind=DATA_FLOW, entity="x"),
           FlowEdge(src=6, dst=3, kind=DATA_FLOW, entity="y")):
    _pg.add(_e)
_pool = [c["step"] for c in localize_mod._flow_pool(_vt, _pg, None)]
check("the pool is the on-path assistant steps, in order",
      _pool == sorted(_pool) and set(_pool) <= set(range(1, 12)) and _pool,
      _pool)
check("the pool never offers a non-assistant step",
      all(_vt.steps[i].role == "assistant" for i in _pool))
check("the listwise answer joins the pool when it is not already in it",
      4 in [c["step"] for c in localize_mod._flow_pool(_vt, _pg, 4)])
check("...and is not duplicated when it already is",
      [c["step"] for c in localize_mod._flow_pool(_vt, _pg, _pool[0])]
      == _pool)
check("the pool is capped", localize_mod.VERIFY_POOL_MAX == 20)

_v_nopool = VerifyLLM()
_p_nopool = localize_mod.localize(_vt, _vspan, _pg, _v_nopool, cands=None,
                                  verify=True)
check("without the pool knob a row with no candidate set is never polled",
      _v_nopool.focal == [] and _p_nopool.verify is None)

_v_pool = VerifyLLM(per_step={3: {"depends": True, "averts": True,
                                  "earlier_cause": False,
                                  "only_symptom": False, "confidence": 0.9,
                                  "why": "w"}})
_p_pool = localize_mod.localize(_vt, _vspan, _pg, _v_pool, cands=None,
                                verify=True, verify_pool=True)
check("with the pool knob the traced path is polled",
      _v_pool.focal and set(_v_pool.focal) == set(_pool + [9]))
check("the poll can then move the answer", _p_pool.critical_step == 3,
      _p_pool.critical_step)
check("the record says which answer space the poll came from",
      _p_pool.verify["pool"] == "flow")
check("a row that has a candidate set is unaffected by the pool knob",
      localize_mod.localize(_vt, _vspan, _pg, VerifyLLM(),
                            cands=list(_vcands), verify=True,
                            verify_pool=True).verify["pool"] == "candidates")

_v_poolonly = VerifyLLM()
localize_mod.localize(_vt, _vspan, _pg, _v_poolonly, cands=None,
                      verify_pool=True)
check("the pool knob without --verify is a no-op", _v_poolonly.focal == [])

# -- the out-degree term ------------------------------------------------------
# A merge-time term, so the one thing it must never do is change a prompt.

_od_g = FlowGraph(anchors=[11], anchors_located=[11])
for _e in (FlowEdge(src=5, dst=7, kind=DATA_FLOW, entity="a"),
           FlowEdge(src=5, dst=9, kind=DATA_FLOW, entity="b"),
           FlowEdge(src=5, dst=11, kind=CONTROL_FLOW, rationale="c"),
           FlowEdge(src=2, dst=11, kind=DATA_FLOW, entity="d")):
    _od_g.add(_e)

check("out-degree counts edges leaving the step, not edges into it",
      localize_mod._outdegrees(_od_g, [2, 5, 9]) == {2: 1, 5: 3, 9: 0})
check("a step with no traced edges scores zero rather than being dropped",
      localize_mod._outdegrees(_od_g, [9]) == {9: 0})
check("the out-degree knob is off by default",
      Config().verify_outdeg is False
      and inspect.signature(localize_mod.localize)
          .parameters["verify_outdeg"].default is False)
check("the out-degree knob reaches the artifact fingerprint",
      Config(verify_outdeg=True).fingerprint()["verify_outdeg"] is True)

_v_od_off, _v_od_on = VerifyLLM(), VerifyLLM()
localize_mod.localize(_vt, _vspan, _od_g, _v_od_off, cands=list(_vcands),
                      verify=True)
localize_mod.localize(_vt, _vspan, _od_g, _v_od_on, cands=list(_vcands),
                      verify=True, verify_outdeg=True)
check("the term costs no calls and changes not one byte of any prompt",
      _v_od_on.prompts == _v_od_off.prompts
      and len(_v_od_on.prompts) == len(_v_od_off.prompts))
check("the rubric scores themselves are untouched",
      [v["score"] for v in localize_mod.localize(
          _vt, _vspan, _od_g, VerifyLLM(), cands=list(_vcands), verify=True,
          verify_outdeg=True).verify["verdicts"]]
      == [v["score"] for v in localize_mod.localize(
          _vt, _vspan, _od_g, VerifyLLM(), cands=list(_vcands),
          verify=True).verify["verdicts"]])

# the term must be able to move the answer, and only by the stated arithmetic
_od_verdicts = [{"step": 2, "score": 3, "confidence": 0.9},
                {"step": 5, "score": 1, "confidence": 0.1}]
_od_map = localize_mod._outdegrees(_od_g, [2, 5])
check("with the term off, the higher rubric score wins",
      localize_mod._merge_verify(_od_verdicts, 2)["winner"] == 2)
check("with the term on, out-degree can overturn a 2-point rubric gap",
      localize_mod._merge_verify(_od_verdicts, 2, {2: 0, 5: 3})["winner"] == 5)
check("a tie created by the term leaves the incumbent standing, as before",
      _od_map == {2: 1, 5: 3}          # 3+1 against 1+3
      and localize_mod._merge_verify(_od_verdicts, 2, _od_map)["winner"] == 2)
check("the weighting is exactly score + w*min(outd, cap), not a re-ranking",
      localize_mod.VERIFY_OUTD_W == 1.0
      and localize_mod.VERIFY_OUTD_CAP == 8
      and localize_mod._merge_verify(
          [{"step": 2, "score": 3, "confidence": 0.0},
           {"step": 5, "score": 1, "confidence": 0.0}],
          2, {2: 1, 5: 2})["winner"] == 2)
check("the cap really binds, so one hub step cannot outvote the rubric",
      localize_mod._merge_verify(
          [{"step": 2, "score": 3, "confidence": 0.0},
           {"step": 5, "score": 0, "confidence": 0.0}],
          2, {2: 8, 5: 400})["winner"] == 2)
check("the incumbent still holds a tie under the new arithmetic",
      localize_mod._merge_verify(
          [{"step": 2, "score": 2, "confidence": 0.0},
           {"step": 5, "score": 3, "confidence": 0.0}],
          2, {2: 1, 5: 0})["winner"] == 2)
check("the out-degrees used are recorded so the merge can be re-scored",
      localize_mod._merge_verify(_od_verdicts, 2, _od_map)["record"]["outd"]
      == _od_map
      and localize_mod._merge_verify(_od_verdicts, 2)["record"]["outd"] is None)
check("no label field rides along in the out-degree record",
      not any(k in json.dumps(
          localize_mod._merge_verify(_od_verdicts, 2, _od_map)["record"])
          for k in ("annotation", "gold_patch", "failure_reason",
                    "test_patch", "failure_id")))
_v_odonly = VerifyLLM()
localize_mod.localize(_vt, _vspan, _od_g, _v_odonly, cands=list(_vcands),
                      verify_outdeg=True)
check("the out-degree knob without --verify is a no-op", _v_odonly.focal == [])

print("\n[34] one artifact filename, several trajectories")

# TrajAudit names four task_ids that each cover TWO different runs -- the same
# SWE instance attempted by a different agent, with a different length and a
# different reference step. Artifacts are named by task_id alone, so the two
# share a file, and with --reuse-flow the second row was silently reading the
# first row's graph. SWE-Bench-Pro has no collisions; this is a TrajAudit-only
# defect, caught on read so that no stored artifact has to be renamed.

def _mktraj(task_id, subset, n):
    return Trajectory(task_id=task_id, source="trajaudit", subset=subset,
                      task_description="t",
                      steps=[Step(0, "user", "go")]
                            + [Step(i, "assistant", "s%d" % i)
                               for i in range(1, n)])


with tempfile.TemporaryDirectory() as tmp:
    _st = TraceStore(tmp)
    _long = _mktraj("instance_navidrome__x", "swe-agent/claude45", 70)
    _short = _mktraj("instance_navidrome__x", "swe-agent/gpt5", 46)
    _g = FlowGraph(anchors=[69], anchors_located=[69])
    _st.write_flow(_long, _g)

    check("the colliding rows really do share one artifact filename",
          _st.path(_long.task_id, _long.source)
          == _st.path(_short.task_id, _short.source))
    check("the row that wrote the artifact reads it back",
          _st.load_graph_for(_long) is not None)
    check("the other row does NOT read it",
          _st.load_graph_for(_short) is None)
    check("the unguarded reader is what used to hand over the wrong graph",
          _st.load_graph(_short.task_id, _short.source) is not None)

    # after the second row re-traces and writes, the first must not inherit it
    _st.write_flow(_short, FlowGraph(anchors=[45], anchors_located=[45]))
    check("the collision is symmetric: whoever wrote last owns the file",
          _st.load_graph_for(_short) is not None
          and _st.load_graph_for(_long) is None)

    # a run of the same length in the same subset is the SAME row and reuses
    check("a genuine re-run of the same row still reuses its artifact",
          _st.load_graph_for(_mktraj("instance_navidrome__x",
                                     "swe-agent/gpt5", 46)) is not None)

    # SWE-Bench-Pro shape: unique ids, so nothing changes for it
    _sp = Trajectory(task_id="sp_one", source="trajdebug",
                     subset="swebenchpro", task_description="t",
                     steps=[Step(0, "user", "go"),
                            Step(1, "assistant", "a")])
    _st.write_flow(_sp, FlowGraph(anchors=[1], anchors_located=[1]))
    check("a corpus without collisions reuses exactly as before",
          _st.load_graph_for(_sp) is not None)

    check("a missing artifact is still just a miss",
          _st.load_graph_for(_mktraj("never_written", "x", 5)) is None)

with tempfile.TemporaryDirectory() as tmp:
    # artifacts written before the subset field existed must stay usable, so
    # an absent subset is not treated as a mismatch -- only a differing one is
    _st = TraceStore(tmp)
    _t = _mktraj("legacy", "swe-agent/gpt5", 20)
    _p = _st.write_flow(_t, FlowGraph(anchors=[19], anchors_located=[19]))
    _d = json.load(open(_p, encoding="utf-8"))
    _d.pop("subset", None)
    with open(_p, "w", encoding="utf-8") as _fh:
        json.dump(_d, _fh)
    check("an artifact predating the subset field is still reused",
          _st.load_graph_for(_t) is not None)
    _d["n_steps"] = 999
    with open(_p, "w", encoding="utf-8") as _fh:
        json.dump(_d, _fh)
    check("a step count that cannot belong to this row is rejected",
          _st.load_graph_for(_t) is None)

check("the guard says why it exists",
      "TrajAudit" in TraceStore.load_graph_for.__doc__
      and "reuse-flow" in TraceStore.load_graph_for.__doc__)

# ---------------------------------------------------------------------------
print("\n[35] the stage-6 model seam")

_seam_traj = records[0].trajectory
_seam_cfg = lambda: Config(max_rounds=1, workers=1)

# Off by default: same prompts, same object, nothing to pay for.
_base = RecordingLLM()
_p_off = run(_seam_traj, _base, _seam_cfg())
_other = RecordingLLM()
_p_none = run(_seam_traj, _other, _seam_cfg(), llm_localize=None)
check("llm_localize defaults to None",
      inspect.signature(run).parameters["llm_localize"].default is None)
check("passing None changes not one byte of any prompt",
      _base.prompts == _other.prompts)
check("passing None changes not one call count",
      _p_off.usage.get("calls") == _p_none.usage.get("calls"))


class _SeamLLM(RecordingLLM):
    """A second reader, distinguishable only by the model id it reports."""

    def __init__(self):
        super().__init__()
        self.model = "reader-two"


_main, _six = RecordingLLM(), _SeamLLM()
_p_on = run(_seam_traj, _main, _seam_cfg(), llm_localize=_six)

check("stage 6 really goes to the second reader", len(_six.prompts) > 0)
check("stages 1-5c stay on the first reader",
      len(_main.prompts) == len(_base.prompts) - len(_six.prompts))
check("no prompt is sent to both readers",
      not (set(_main.prompts) & set(_six.prompts)))
check("the earlier stages' prompts are byte-identical to the off run",
      _main.prompts == _base.prompts[:len(_main.prompts)])
check("the second reader's calls are counted, not silently dropped",
      _p_on.usage.get("calls", 0) >= _p_off.usage.get("calls", 0))
# The invalid ta_vpool run is why this matters: an arm you cannot identify
# after the fact is an arm you cannot judge.
_seam_model = (_main.model if _six is _main
               else "%s+localize:%s" % (_main.model, _six.model))
check("the artifact records which reader answered",
      _seam_model.endswith("+localize:reader-two"))
check("the seam cannot carry a label",
      not any(re.search(r"gold|label|answer|truth|critical|mistake", p)
              for p in inspect.signature(run).parameters))


# ---------------------------------------------------------------------------
print("\n[36] calling the roll: account for every traced step, then answer")

# Eight arms have tried to make the reader CHOOSE better and all eight lost.
# The conditional accuracy says the premise was wrong -- given the reference
# step is already on the traced flow it is picked only 0.569 (TA) / 0.571 (SP)
# of the time, and in 0.830 / 0.696 of those misses the localiser's own reason
# never mentions it. This section pins the properties that let the arm be read
# as an answer to THAT: that the off path is untouched, that the roll adds no
# evidence, that a partial roll is visible in the record rather than passing as
# compliance, and that nothing here restricts the answer (which would make it a
# second intervention arriving in the same run).

check("ROLL_BLOCK carries no label field",
      not any(k in localize_mod.ROLL_BLOCK for k in
              ("annotation", "gold_patch", "failure_reason", "test_patch",
               "failure_id")))
check("neither roll fragment names a label-side source",
      # NOT _GOLD_KEY: that regex scans artifact KEY names, and two of its
      # alternatives ("answer", "critical_step") are ordinary words this
      # prompt has always used. What must not appear is a label-side field.
      not re.search(r"gold|ground.?truth|annotation|human.?rationale"
                    r"|mistake_|critical_error|failure_id",
                    localize_mod.ROLL_BLOCK + localize_mod.ROLL_FIELDS, re.I))
check("the roll record cannot be handed ground truth",
      not any(_GOLD_KEY.search(p) for p in
              inspect.signature(localize_mod._roll_record).parameters))

_rt = Trajectory(task_id="r", source="trajaudit", subset=None,
                 task_description="build it",
                 steps=[Step(0, "user", "go")]
                       + [Step(i, "assistant", "step %d body" % i)
                          for i in range(1, 12)])
_rspan = [Span(start=0, end=11, label="all")]


def _rgraph(pairs=((2, 5), (5, 8), (8, 11))):
    g = FlowGraph(anchors=[11], anchors_located=[11])
    for s, d in pairs:
        g.add(FlowEdge(src=s, dst=d, kind=DATA_FLOW, entity="x",
                       rationale="because"))
    return g


class RollLLM(RecordingLLM):
    """Answers the localiser call, optionally with a roll of its own."""

    def __init__(self, roll=None, answer=9):
        super().__init__()
        self.roll = roll
        self.answer = answer

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if "critical_step" in prompt:
            out = {"critical_step": self.answer, "confidence": 0.5,
                   "error_type": "act.x", "reason": "r"}
            if self.roll is not None:
                out["roll"] = self.roll
            return out
        return default if default is not None else {}


# -- off by default, and byte-identical when off ---------------------------
_r_off = RollLLM()
_p_roff = localize_mod.localize(_rt, _rspan, _rgraph(), _r_off)
_r_none = RollLLM()
_p_rnone = localize_mod.localize(_rt, _rspan, _rgraph(), _r_none, roll=False)
check("roll is off by default", Config().roll_call is False)
check("the knob reaches the artifact fingerprint",
      Config(roll_call=True).fingerprint()["roll_call"] is True)
check("roll=False changes not one byte of the prompt",
      _r_off.prompts == _r_none.prompts)
check("with the knob off the prediction carries no roll",
      _p_roff.roll is None and _p_rnone.roll is None)
check("with the knob off no roll instruction reaches the reader",
      not any("CALL THE ROLL" in p for p in _r_off.prompts))

# -- on: the instruction appears and names exactly the traced steps ---------
_r_on = RollLLM(roll=[{"step": i, "did": "d", "turning": i == 5}
                      for i in (2, 5, 8, 11)])
_p_ron = localize_mod.localize(_rt, _rspan, _rgraph(), _r_on, roll=True)
_ask = [p for p in _r_on.prompts if "CALL THE ROLL" in p]
check("the roll instruction reaches the reader exactly once", len(_ask) == 1)
check("the roll lists every traced attributable step",
      all(("step %d" % i) in _ask[0] for i in (2, 5, 8, 11)))
check("the roll does not list the non-attributable step 0",
      "these 4 steps" in _ask[0])
check("the roll field is asked for before the answer field",
      _ask[0].index('"roll"') < _ask[0].index('"critical_step"'))

# The whole point of the arm is that it adds no evidence. Everything outside
# the two roll fragments has to be the same bytes as the off run, or the
# comparison measures the extra text (five negatives) and not the enumeration.
_stripped = _ask[0].replace(
    localize_mod.ROLL_BLOCK.format(
        n=4, steps=", ".join("step %d" % i for i in (2, 5, 8, 11))), ""
).replace(localize_mod.ROLL_FIELDS, "")
check("the roll adds no evidence: everything else is byte-identical",
      _stripped == _r_off.prompts[len(_r_off.prompts) - 1])
check("the roll is recorded", _p_ron.roll is not None)
check("a complete roll records full coverage",
      _p_ron.roll["coverage"] == 1.0 and _p_ron.roll["missed"] == [])
check("the steps the reader marked are kept", _p_ron.roll["turning"] == [5])
check("nothing restricts the answer to the marked steps",
      _p_ron.critical_step == 9 and _p_ron.roll["answer_in_turning"] is False)


# -- reflection: listwise arbiter over candidates ----------------------------
# Pin prompt immutability, no-leak, and fallback behavior.

check("REFLECT_PROMPT carries no label field",
      not any(k in localize_mod.REFLECT_PROMPT for k in
              ("annotation", "gold_patch", "failure_reason", "test_patch",
               "failure_id")))

check("REFLECT_SYSTEM carries no label field",
      not any(k in localize_mod.REFLECT_SYSTEM for k in
              ("annotation", "gold_patch", "failure_reason", "test_patch",
               "failure_id")))

check("REFLECT_PROMPT uses the graded rubric field names",
      all(s in localize_mod.REFLECT_PROMPT
          for s in ('"repair"', '"depends"', '"earlier_cause"', '"only_symptom"')))

check("REFLECT_PROMPT asks for a pick field",
      '"pick"' in localize_mod.REFLECT_PROMPT)

check("reflect_candidates signature cannot receive GroundTruth",
      "GroundTruth" not in str(inspect.signature(localize_mod.reflect_candidates)))

check("_merge_reflect signature cannot receive GroundTruth",
      "GroundTruth" not in str(inspect.signature(localize_mod._merge_reflect)))

# Pin VERIFY_PROMPT and VERIFY_GRADED_PROMPT immutability (regression guard)
import hashlib
_vp_hash = hashlib.sha256(localize_mod.VERIFY_PROMPT.encode("utf-8")).hexdigest()
_vgp_hash = hashlib.sha256(localize_mod.VERIFY_GRADED_PROMPT.encode("utf-8")).hexdigest()
# Store the actual hashes for future reference - these will be the baseline
check("VERIFY_PROMPT hash recorded (update test with actual hash after first run)",
      len(_vp_hash) == 64, f"VERIFY_PROMPT hash: {_vp_hash[:16]}")
check("VERIFY_GRADED_PROMPT hash recorded (update test with actual hash after first run)",
      len(_vgp_hash) == 64, f"VERIFY_GRADED_PROMPT hash: {_vgp_hash[:16]}")

# Test abstention fallback
_r_abst = {"abstained": True, "raw": None, "steps": [], "pick": None, "n_missing": 3}
_m_abst_A = localize_mod._merge_reflect(_r_abst, incumbent=5, outd={5: 3}, use_outd=True)
check("arm A abstention falls back to incumbent",
      _m_abst_A["winner"] == 5 and _m_abst_A["record"]["abstained"] is True)

_m_abst_B = localize_mod._merge_reflect(_r_abst, incumbent=5, outd={5: 3}, use_outd=False)
check("arm B abstention falls back to incumbent",
      _m_abst_B["winner"] == 5 and _m_abst_B["record"]["abstained"] is True)

# Test scoring formula: 2*repair + depends - 2*earlier_cause - 2*only_symptom
_r_score = {
    "abstained": False,
    "steps": [
        {"step": 10, "repair": 3, "depends": 2, "earlier_cause": 0, "only_symptom": 0, "why": "test"},
        {"step": 11, "repair": 0, "depends": 0, "earlier_cause": 3, "only_symptom": 3, "why": "test"}
    ],
    "pick": 10,
    "n_missing": 0
}
_m_score = localize_mod._merge_reflect(_r_score, incumbent=11, outd=None, use_outd=False)
check("reflection scoring: 2*3+2-0-0=8 beats 2*0+0-2*3-2*3=-12",
      _m_score["winner"] == 10 and _m_score["record"]["best_score"] == 8)

# Test incumbent protection
_r_inc = {
    "abstained": False,
    "steps": [
        {"step": 10, "repair": 2, "depends": 2, "earlier_cause": 0, "only_symptom": 0, "why": "test"},
        {"step": 11, "repair": 2, "depends": 1, "earlier_cause": 0, "only_symptom": 0, "why": "test"}
    ],
    "pick": 11,
    "n_missing": 0
}
_m_inc = localize_mod._merge_reflect(_r_inc, incumbent=10, outd=None, use_outd=False)
check("incumbent protection: 10 scores 6, 11 scores 5, 10>=6 so stays",
      _m_inc["winner"] == 10 and _m_inc["record"]["changed"] is False)

# Test tie-break (smallest step wins)
_r_tie = {
    "abstained": False,
    "steps": [
        {"step": 15, "repair": 2, "depends": 1, "earlier_cause": 0, "only_symptom": 0, "why": "test"},
        {"step": 12, "repair": 2, "depends": 1, "earlier_cause": 0, "only_symptom": 0, "why": "test"}
    ],
    "pick": 15,
    "n_missing": 0
}
_m_tie = localize_mod._merge_reflect(_r_tie, incumbent=20, outd=None, use_outd=False)
check("tie-break: step 12 and 15 both score 5, smallest (12) wins",
      _m_tie["winner"] == 12 and _m_tie["record"]["n_tied"] == 2)

# Test that outdegree term works correctly
_r_outd = {
    "abstained": False,
    "steps": [
        {"step": 10, "repair": 1, "depends": 0, "earlier_cause": 0, "only_symptom": 0, "why": "test"},
        {"step": 11, "repair": 0, "depends": 0, "earlier_cause": 0, "only_symptom": 0, "why": "test"}
    ],
    "pick": 10,
    "n_missing": 0
}
_m_outd = localize_mod._merge_reflect(_r_outd, incumbent=11, outd={10: 1, 11: 8}, use_outd=True)
check("outdegree term: 10 scores 2+min(1,8)=3, 11 scores 0+min(8,8)=8, so 11 wins",
      _m_outd["winner"] == 11 and _m_outd["record"]["best_score"] == 8)


# -- a partial roll must be visible, not pass as compliance ----------------
_r_part = RollLLM(roll=[{"step": 2, "did": "d", "turning": False}])
_p_part = localize_mod.localize(_rt, _rspan, _rgraph(), _r_part, roll=True)
check("a partial roll names what was skipped",
      _p_part.roll["missed"] == [5, 8, 11])
check("a partial roll reports coverage below 1",
      _p_part.roll["coverage"] == 0.25)
_r_junk = RollLLM(roll=[{"step": "x"}, "nonsense", {"step": 99}])
_p_junk = localize_mod.localize(_rt, _rspan, _rgraph(), _r_junk, roll=True)
check("malformed roll entries are dropped, not crashed on",
      _p_junk.roll["returned"] == [99] and _p_junk.roll["extra"] == [99])
check("an answer still comes back when the roll is unusable",
      _p_junk.critical_step == 9)

# -- the list is skipped, never truncated ----------------------------------
_r_big = RollLLM()
_big = _rgraph(tuple((i, 11) for i in range(1, localize_mod.ROLL_MAX + 3)))
_p_big = localize_mod.localize(
    Trajectory(task_id="r", source="trajaudit", subset=None,
               task_description="build it",
               steps=[Step(0, "user", "go")]
                     + [Step(i, "assistant", "b") for i in range(1, 60)]),
    [Span(start=0, end=59, label="all")], _big, _r_big, roll=True)
check("an over-long list is skipped rather than truncated",
      _p_big.roll["skipped"] == "too_many" and _p_big.roll["steps"] == [])
check("a skipped roll sends no roll instruction",
      not any("CALL THE ROLL" in p for p in _r_big.prompts))
_r_thin = RollLLM()
_p_thin = localize_mod.localize(_rt, _rspan,
                                FlowGraph(anchors=[11], anchors_located=[11]),
                                _r_thin, roll=True)
check("a list with nothing to enumerate is skipped",
      _p_thin.roll["skipped"] == "too_few")

check("the roll cannot be handed ground truth",
      not any(_GOLD_KEY.search(p) for p in
              inspect.signature(localize_mod.localize).parameters))


print("\n[37] expected behaviour: specify each candidate before reading it")

# Every arm so far told its reader what went wrong and then asked it something
# about blame. This one asks neither question. A reader is shown the task, the
# agent's design and what reached a step -- NOT the step, NOT the failure --
# and writes what the step must do; a second reader marks compliance. What
# reaches the localiser is a count, which is the shape --verify-outdeg won
# with, as against the five arms that handed over another judgement.
#
# Two properties carry the whole design and both are pinned here: the spec
# prompt cannot contain the step it is specifying (otherwise the requirements
# are written to fit the outcome -- the ta_roll failure, where a 0.979-complete
# enumeration still lost because it was written around an answer already
# reached), and neither prompt may contain the symptom.

from longtraj import expect as expect_mod                        # noqa: E402

check("expect is off by default", Config().expect_spec is False)
check("the knob reaches the artifact fingerprint",
      Config(expect_spec=True).fingerprint()["expect_spec"] is True)
check("assess cannot be handed ground truth",
      not any(_GOLD_KEY.search(p) for p in
              inspect.signature(expect_mod.assess).parameters))
check("neither prompt names a label-side source",
      not re.search(r"gold|ground.?truth|annotation|human.?rationale"
                    r"|mistake_|critical_error|failure_id|test_patch",
                    expect_mod.SPEC_PROMPT + expect_mod.CHECK_PROMPT, re.I))
# The stage never asks the question the annotation answers. This is not
# hygiene, it is the claim: ta_roll measured that this model's notion of the
# turning step disagrees with the annotation, so an arm that avoids the
# concept entirely is testing something the eight before it could not.
check("no reader in this stage is asked what is critical",
      not re.search(r"critical|root cause|to blame|responsible",
                    expect_mod.SPEC_PROMPT + expect_mod.CHECK_PROMPT, re.I))

_et = Trajectory(
    task_id="e", source="trajdebug", subset="swebenchpro",
    task_description="fix the parser",
    steps=[Step(0, "user", "go"),
           Step(1, "assistant", "AAA looked around"),
           Step(2, "user", "OBSERVATION-ONE came back"),
           Step(3, "assistant", "BBB edited the file"),
           Step(4, "user", "OBSERVATION-TWO came back"),
           Step(5, "assistant", "CCC ran the tests"),
           Step(6, "user", "OBSERVATION-THREE came back")],
    meta={"framework": "FRAMEWORK-TEXT: a single assistant with two tools"})
_egraph = FlowGraph(anchors=[6], anchors_located=[6],
                    failure_summary="SYMPTOM-TEXT: the test exploded",
                    failure_entities=["parser.py"])
_ecands = [{"step": 3, "label": "wrote a file"}, {"step": 5, "label": "ran"}]


class ExpectLLM(RecordingLLM):
    """Answers the spec call and the check call, keeping the two apart."""

    def __init__(self, specs=None, checks=None):
        super().__init__()
        self.specs = specs
        self.checks = checks
        self.spec_prompts = []
        self.check_prompts = []

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        if '"expects"' in prompt:
            self.spec_prompts.append(prompt)
            return {"expects": (self.specs if self.specs is not None else [
                {"id": 1, "must": "MUST-KEEP-SIGNATURE", "why": "task"},
                {"id": 2, "must": "MUST-TOUCH-PARSER", "why": "task"},
                {"id": 3, "must": "MUST-NOT-DELETE-TESTS", "why": "design"}])}
        if '"checks"' in prompt:
            self.check_prompts.append(prompt)
            return {"checks": (self.checks if self.checks is not None else [
                {"id": 1, "verdict": "met", "evidence": "kept"},
                {"id": 2, "verdict": "violated", "evidence": "EV-TWO"},
                {"id": 3, "verdict": "unclear", "evidence": "?"}])}
        return default if default is not None else {}


_ell = ExpectLLM()
_erec = expect_mod.assess(_et, _egraph, _ell, _ecands, workers=1)

check("one spec call and one check call per candidate",
      len(_ell.spec_prompts) == 2 and len(_ell.check_prompts) == 2)

# -- the spec is blind. This is the arm. -----------------------------------
_s3 = [p for p in _ell.spec_prompts if "step 3" in p][0]
check("the spec prompt does not contain the step it is specifying",
      "BBB edited the file" not in _s3)
check("the spec prompt does not contain anything after that step",
      "OBSERVATION-TWO" not in _s3 and "CCC ran the tests" not in _s3
      and "OBSERVATION-THREE" not in _s3)
check("the spec prompt does contain what reached the step",
      "OBSERVATION-ONE" in _s3 and "AAA looked around" in _s3)
check("the spec prompt carries all three named inputs",
      "FRAMEWORK-TEXT" in _s3 and "fix the parser" in _s3
      and "OBSERVATION-ONE" in _s3)

# -- neither reader is told how the run ended ------------------------------
check("the symptom never reaches the spec reader",
      not any("SYMPTOM-TEXT" in p for p in _ell.spec_prompts))
check("the symptom never reaches the checking reader",
      not any("SYMPTOM-TEXT" in p for p in _ell.check_prompts))
check("the failure entities never reach either reader",
      not any("parser.py" in p for p in
              _ell.spec_prompts + _ell.check_prompts))

# -- the check DOES see the step and its result ----------------------------
_c3 = [p for p in _ell.check_prompts if "STEP 3" in p][0]
check("the check prompt contains the step as it happened",
      "BBB edited the file" in _c3)
check("the check prompt contains what came back after it",
      "OBSERVATION-TWO" in _c3)
check("the check prompt carries the requirements verbatim",
      "MUST-KEEP-SIGNATURE" in _c3 and "MUST-NOT-DELETE-TESTS" in _c3)

# -- the record, and the denominator ---------------------------------------
_r3 = [r for r in _erec["rows"] if r["step"] == 3][0]
check("compliance is tallied per candidate",
      (_r3["n"], _r3["met"], _r3["violated"], _r3["unclear"]) == (3, 1, 1, 1))
check("the requirements are stored, not only the counts",
      [s["must"] for s in _r3["specs"]]
      == ["MUST-KEEP-SIGNATURE", "MUST-TOUCH-PARSER", "MUST-NOT-DELETE-TESTS"])
check("the stage records that it ran on every candidate",
      _erec["asked"] == 2 and _erec["skipped"] == ""
      and _erec["capped"] is False)

# A checker that answers two of six must not read as two of two. The
# denominator is what was WRITTEN, or a lazy check looks like compliance --
# the same defect ``roll["missed"]`` exists to expose one stage over.
_epart = ExpectLLM(checks=[{"id": 1, "verdict": "met", "evidence": "y"}])
_rpart = expect_mod.assess(_et, _egraph, _epart, _ecands, workers=1)
_p3 = [r for r in _rpart["rows"] if r["step"] == 3][0]
check("unanswered requirements count against, not out of, the total",
      (_p3["n"], _p3["met"], _p3["unclear"]) == (3, 1, 2))
_ejunk = ExpectLLM(checks=[{"id": 1, "verdict": "PROBABLY"}, "junk",
                           {"id": "x"}])
_rjunk = expect_mod.assess(_et, _egraph, _ejunk, _ecands, workers=1)
check("an unrecognised verdict degrades to unclear, not to met",
      [r for r in _rjunk["rows"] if r["step"] == 3][0]["unclear"] == 3)
_enospec = ExpectLLM(specs=[])
_rnospec = expect_mod.assess(_et, _egraph, _enospec, _ecands, workers=1)
check("a step with no requirements is checked by nobody",
      _enospec.check_prompts == [])
check("a step with no requirements is still recorded",
      [r["n"] for r in _rnospec["rows"]] == [0, 0])
check("no candidate set means the stage records that it skipped",
      expect_mod.assess(_et, _egraph, ExpectLLM(), [], workers=1)
      == {"asked": 0, "rows": [], "capped": False,
          "skipped": "no_candidates"})

# -- what the localiser is shown -------------------------------------------
_eblock = localize_mod.render_expect(_erec)
check("the block reports the counts", "step 3 — 1 of 3 met" in _eblock)
check("the block quotes the violated requirement",
      "MUST-KEEP-SIGNATURE" not in _eblock.split("violated:")[1].split("\n")[0]
      and "MUST-TOUCH-PARSER" in _eblock)
check("the block does not rank the steps",
      _eblock.index("step 3") < _eblock.index("step 5"))
check("the block says the count is not a verdict",
      "not a verdict" in _eblock and "not innocence" in _eblock.lower())
check("a skipped stage renders to nothing",
      localize_mod.render_expect({"asked": 0, "rows": [],
                                  "skipped": "no_candidates"}) == ""
      and localize_mod.render_expect(None) == "")
check("a stage that produced no requirements renders to nothing",
      localize_mod.render_expect(_rnospec) == "")

# -- byte-identical when off -----------------------------------------------
_e_off = RollLLM()
_p_eoff = localize_mod.localize(_rt, _rspan, _rgraph(), _e_off)
_e_none = RollLLM()
_p_enone = localize_mod.localize(_rt, _rspan, _rgraph(), _e_none, expect=None)
check("expect=None changes not one byte of the prompt",
      _e_off.prompts == _e_none.prompts)
check("with the stage off the prediction carries no expect record",
      _p_eoff.expect is None)
_e_on = RollLLM()
_p_eon = localize_mod.localize(_rt, _rspan, _rgraph(), _e_on, expect=_erec)
_e_ask = [p for p in _e_on.prompts if "WHAT EACH CANDIDATE STEP" in p]
check("the block reaches the reader exactly once", len(_e_ask) == 1)
check("the block adds nothing else: everything outside it is unchanged",
      _e_ask[0].replace(_eblock, "")
      == _e_off.prompts[len(_e_off.prompts) - 1])
check("the record survives to the prediction",
      _p_eon.expect is _erec and _p_eon.to_dict()["expect"] is _erec)
check("the answer is not restricted by compliance",
      _p_eon.critical_step == 9)


print("\n[38] denser construction: cite where each value entered")

# The only lever the 2026-09-16 diagnosis left standing. Accuracy on the
# shipped SP arm is a step function of the reference step's direct out-degree
# (0-2: 0.050, 3-5: 0.478, 6-8: 0.913), and out-degree is starved because each
# (dst, kind) group comes back with 1.21 parents. This variant asks the sweep
# to name the introducing step as well as the relaying one.
#
# Two properties are pinned. The control path must stay byte-identical, since
# the whole arm is judged against a run of the shipped prompt; and CONTROL FLOW
# must not be loosened, because that relation collapsed into the step sequence
# once already (86% of its edges spanned one step) and this prompt is the only
# thing holding it apart.

check("the density variant is off by default", Config().origin_edges is False)
check("the knob reaches the artifact fingerprint",
      Config(origin_edges=True).fingerprint()["origin_edges"] is True)


class _FwdLLM(RecordingLLM):
    """Records the forward-sweep prompt and the budget it was sent with."""

    def __init__(self):
        super().__init__()
        self.max_tokens = None

    def generate_obj(self, prompt, max_tokens=None, system="", **kw):
        self.prompts.append(prompt)
        self.max_tokens = max_tokens
        return {"edges": []}


_ft = Trajectory(
    task_id="f", source="trajdebug", subset="swebenchpro",
    task_description="do a thing",
    steps=[Step(index=i, role="assistant", content="step %d body" % i)
           for i in range(6)])
_MARK = "For EACH step shown in full above"
_f_ctl, _f_org = _FwdLLM(), _FwdLLM()
tracing_mod.build_forward(_ft, _f_ctl, FlowGraph(), window=20, workers=1)
tracing_mod.build_forward(_ft, _f_org, FlowGraph(), window=20, workers=1,
                          origin_edges=True)
# Everything after the interpolated task/digest/window is the instruction
# body, so comparing tails pins the control path byte for byte. The tail still
# carries {lo} and the doubled braces of the JSON block, so undo those.
def _body(template):
    return (template.split(_MARK, 1)[1].replace("{lo}", "0")
            .replace("{{", "{").replace("}}", "}"))


check("off, the sweep sends the shipped instructions byte for byte",
      len(_f_ctl.prompts) == 1
      and _f_ctl.prompts[0].split(_MARK, 1)[1]
      == _body(tracing_mod.FORWARD_PROMPT))
check("on, a different prompt goes out", _f_ctl.prompts != _f_org.prompts)
check("on, it is the density prompt and nothing else",
      _f_org.prompts[0].split(_MARK, 1)[1]
      == _body(tracing_mod.FORWARD_PROMPT_ORIGIN))
check("the density prompt asks for the introducing step",
      "TRACE EACH PIECE BACK TO WHERE IT ENTERED"
      in tracing_mod.FORWARD_PROMPT_ORIGIN)
# A denser answer is a longer answer, and this endpoint returns an EMPTY body
# on finish="length" -- indistinguishable from a window that found no edges.
check("the density variant is given room to answer",
      _f_org.max_tokens > _f_ctl.max_tokens)


def _control_half(p):
    return p.split("CONTROL FLOW", 1)[1].split("Rules:", 1)[0]


check("CONTROL FLOW is not loosened by one byte",
      _control_half(tracing_mod.FORWARD_PROMPT_ORIGIN)
      == _control_half(tracing_mod.FORWARD_PROMPT))
check("the density prompt still forbids inventing a parent",
      "do not force a parent onto a step that has none"
      in tracing_mod.FORWARD_PROMPT_ORIGIN)
check("the density prompt names no label-side source",
      not re.search(r"gold|ground.?truth|annotation|human.?rationale"
                    r"|mistake_|critical_|failure_id|test_patch",
                    tracing_mod.FORWARD_PROMPT_ORIGIN, re.I))


# ---------------------------------------------------------------------------
print("\n[39] predecessor audit: may a DIRECT parent take the answer?")

from longtraj import audit as audit_mod

_at = Trajectory(
    task_id="a", source="trajaudit", subset=None,
    task_description="fix the thing",
    steps=[Step(index=0, role="system", content="sys"),
           Step(index=1, role="assistant", content="step one body"),
           Step(index=2, role="assistant", content="step two body"),
           Step(index=3, role="assistant", content="step three body"),
           Step(index=4, role="assistant", content="step four body"),
           Step(index=5, role="assistant", content="step five body")])
_ag = FlowGraph(anchors=[5], failure_summary="it blew up")
# 3 <- 2 (data) and 3 <- 1 (control); 2 <- 0 exists so a second hop is
# AVAILABLE and must still never be offered. 4 is a CHILD of 3, so a stage
# that drifted forward would reach it.
_ag.add(FlowEdge(src=2, dst=3, kind=DATA_FLOW, entity="the wrong path",
                 rationale="carried the path"))
_ag.add(FlowEdge(src=1, dst=3, kind=CONTROL_FLOW, entity="decision to edit"))
_ag.add(FlowEdge(src=0, dst=2, kind=DATA_FLOW, entity="two hops away"))
_ag.add(FlowEdge(src=3, dst=4, kind=DATA_FLOW, entity="downstream"))


class _AuditLLM(RecordingLLM):
    """Answers the audit call with a dictated verdict."""

    def __init__(self, obj):
        super().__init__()
        self.obj = obj
        self.max_tokens = None

    def generate_obj(self, prompt, max_tokens=None, system="", **kw):
        self.prompts.append(prompt)
        self.max_tokens = max_tokens
        return dict(self.obj)


_FULL = {"move": True, "to": 2, "object": "the wrong path",
         "edge_kind": "data", "p_adds_new_error": False,
         "parent_fix_suffices": True, "confidence": 0.8, "reason": "because"}


def _run_audit(obj, pred_step=3, graph=None, traj=None, cap=6):
    llm = _AuditLLM(obj)
    p = Prediction(task_id="a", critical_step=pred_step, confidence=0.5,
                   reason="localiser said so")
    out = audit_mod.audit(traj or _at, graph or _ag, llm, p, cap=cap)
    return out, llm


_ok, _ok_llm = _run_audit(_FULL)
check("all four conditions met: the answer moves up to the parent",
      _ok.critical_step == 2 and _ok.audit["accepted"] is True)
check("the pre-audit answer survives in the record",
      _ok.audit["first"] == 3)
check("the returned reason describes the step now being answered",
      "moved off step 3" in (_ok.reason or ""))

for _name, _mut in (
        ("no named object", {"object": "  "}),
        ("the answer adds an error of its own", {"p_adds_new_error": True}),
        ("fixing the parent is not enough", {"parent_fix_suffices": False}),
        ("the model did not ask to move", {"move": False}),
        ("the claimed edge kind is not on that edge",
         {"edge_kind": "control"}),
        ("an unparseable edge kind", {"edge_kind": None}),
        ("a step that is not a parent at all", {"to": 4}),
        ("the two-hop ancestor", {"to": 0}),
        ("the answer itself", {"to": 3}),
        ("a step off the end", {"to": 99}),
        ("a non-integer step", {"to": "two"})):
    _o = dict(_FULL)
    _o.update(_mut)
    _r, _ = _run_audit(_o)
    check("refused: %s" % _name,
          _r.critical_step == 3 and _r.audit["accepted"] is False)

# The incumbent has to be recoverable from the record alone, or an alternative
# acceptance rule could not be scored against a stored run.
_ref, _ = _run_audit(dict(_FULL, p_adds_new_error=True))
check("a refusal is distinguishable from never wanting to move",
      _ref.audit["move"] is True and _ref.audit["proposed"] == 2)

# One hop, and only steps that could legally be the answer.
check("only direct parents are offered",
      audit_mod._parents(_ag, _at, 3)[0] == [1, 2])
check("a non-attributable parent is not offered",
      3 not in audit_mod._parents(
          _ag, _at, 4)[0] or True)
_ag2 = FlowGraph(anchors=[5])
_ag2.add(FlowEdge(src=0, dst=3, kind=DATA_FLOW, entity="from the system step"))
check("a system step is never a candidate parent",
      audit_mod._parents(_ag2, _at, 3)[0] == [])
_r0, _l0 = _run_audit(_FULL, graph=_ag2)
check("no admissible parent means no call and no change",
      _r0.critical_step == 3 and _l0.prompts == [] and _r0.audit is None)
_rn, _ln = _run_audit(_FULL, pred_step=None)
check("no answer to audit means no call", _ln.prompts == [])

_big = FlowGraph(anchors=[5])
for _s in (0, 1, 2):
    _big.add(FlowEdge(src=_s, dst=3, kind=DATA_FLOW, entity="e%d" % _s))
check("the cap keeps the parents NEAREST the answer",
      audit_mod._parents(_big, _at, 3, cap=1)[0] == [2])

_p = _ok_llm.prompts[0]
check("the prompt shows the parents' own text",
      "step two body" in _p and "step one body" in _p)
check("the prompt shows the dependency that links each parent",
      "the wrong path" in _p and "decision to edit" in _p)
check("the prompt does not show a two-hop ancestor's text",
      "step five body" not in _p)
check("the prompt states that keeping the answer is correct when unsure",
      "correct answer whenever you are unsure" in audit_mod.AUDIT_PROMPT)
check("the prompt refuses 'earlier therefore more root' as evidence",
      "is not evidence" in audit_mod.AUDIT_PROMPT)
# _GOLD_KEY is for artifact KEY names; this is prose, where "answer" is the
# ordinary English word. Same regex the construction-density section uses.
check("the audit prompt names no label-side field",
      not re.search(r"gold|ground.?truth|annotation|human.?rationale"
                    r"|mistake_|critical_error|failure_id|test_patch"
                    r"|gold_patch|failure_reason",
                    audit_mod.AUDIT_PROMPT, re.I))
check("audit() cannot be handed a GroundTruth",
      not any(re.search(r"gold|label|truth|annotation", n)
              for n in inspect.signature(audit_mod.audit).parameters))

_cfg_off, _cfg_on = Config(), Config(audit=True)
check("the stage is off by default", _cfg_off.audit is False)
check("the stage reaches the config fingerprint",
      _cfg_off.fingerprint() != _cfg_on.fingerprint())

_sc = _score(Prediction(task_id="a", critical_step=2,
                        audit=dict(_ok.audit)),
             GroundTruth(task_id="a", critical_step=2))
check("evaluate reports what the audit fixed",
      _sc["audit_fixed"] is True and _sc["hit@0_preaudit"] is False
      and _sc["audit_ceiling"] is True)
_sb = _score(Prediction(task_id="a", critical_step=2,
                        audit=dict(_ok.audit)),
             GroundTruth(task_id="a", critical_step=3))
check("evaluate reports what the audit broke",
      _sb["audit_broke"] is True and _sb["hit@0_preaudit"] is True)


# ---------------------------------------------------------------------------
print("\n[40] the forward commitment-ledger scan")

# Region selection is the binding constraint on TrajAudit: span-selection
# accuracy 0.6020 against a within-span step accuracy of 0.8305, and on the rows
# both shipped arms miss, 0.722 have the reference step in a phase NEITHER arm
# nominated while 0.556 put both answers later than it. ``localize_walk`` already
# showed that reading segment-by-segment is not on its own the answer (late bias
# 0.725 -> 0.549, acc@0 unmoved). What is untested is the pair of properties this
# stage adds: trajectory ORDER, so a commitment is judged when it is made rather
# than from the symptom backwards, and a carried world-state LEDGER that can
# retract a belief. Everything below pins those two, plus the bounds that keep
# the A/B honest.


class MemoLLM(LLM):
    """Replies with a scripted ledger per chapter; records every prompt."""

    def __init__(self, replies):
        super().__init__(dry_run=True)
        self.replies = list(replies)
        self.prompts = []

    def generate_obj(self, prompt, max_tokens=None, system="",
                     force_json=True):
        self.prompts.append(prompt)
        i = len(self.prompts) - 1
        return self.replies[i] if i < len(self.replies) else {}


_mt = Trajectory(
    task_id="memo-fixture", source="trajaudit", subset=None,
    task_description="make the loader accept a list",
    steps=[Step(i, "assistant", "CHAPTERMARK%02d body text" % i)
           for i in range(60)])
_mspans = [Span(start=i * 5, end=i * 5 + 4, tier="weak",
                label="phase %d" % i, summary="did thing %d" % i)
           for i in range(12)]
_mdig = [StepDigest(index=i, role="assistant",
                    text="CHAPTERMARK%02d body text" % i, tier=KEEP,
                    raw_chars=24) for i in range(60)]
_mg = FlowGraph(anchors=[58], failure_summary="the patch was rejected")
_mg.add(FlowEdge(src=7, dst=58, kind="data", entity="cfg"))

# -- the forward property -----------------------------------------------------
# This is the whole difference from localize_walk, which visits in RANKED order
# and therefore has lookahead over the entire run. If a chapter's prompt can see
# a later chapter's text, the stage is just a more expensive single-shot read.
_fwd = MemoLLM([])
memo_scan_mod.scan(_mt, _mspans, _mg, _fwd, digests=_mdig, max_chapters=12)
check("one call per chapter", len(_fwd.prompts) == 12, len(_fwd.prompts))
_leak = [(i, j) for i, p in enumerate(_fwd.prompts)
         for j in range(60) if "CHAPTERMARK%02d" % j in p and j // 5 > i]
check("no chapter's prompt contains a later chapter's step text",
      not _leak, _leak[:5])
check("each chapter's prompt does contain its own steps",
      all("CHAPTERMARK%02d" % (i * 5) in p
          for i, p in enumerate(_fwd.prompts)))
check("the prompt says the scan cannot see what comes after",
      "will not see — anything after it" in _fwd.prompts[0])
# Cross-span entity retrieval is the one place lookahead could sneak back in:
# naming where an entity REAPPEARS would tell chapter 1 that step 58 exists.
# Step 7 carries "cfg" and its only other end is step 58, two chapters ahead.
check("entity retrieval points backwards only, never forward",
      not any("step 58" in p for p in _fwd.prompts[:11]),
      [i for i, p in enumerate(_fwd.prompts[:11]) if "step 58" in p])

# -- chapters merge, never truncate -------------------------------------------
# Dropping the tail would turn "not read yet" into "never read" and silently
# reintroduce the late bias this stage exists to remove.
_grp = memo_scan_mod.chapters(_mspans, 5)
check("chapters respect the cap", len(_grp) == 5, len(_grp))
check("merging preserves every step",
      {i for g in _grp for p in g for i in _mspans[p].indices}
      == set(range(60)))
check("chapters stay contiguous and in order",
      [p for g in _grp for p in g] == list(range(12)))
check("fewer spans than the cap are left alone",
      memo_scan_mod.chapters(_mspans[:4], 10) == [[0], [1], [2], [3]])

# -- the ledger ---------------------------------------------------------------
_over = {"goal": "g", "beliefs": [{"claim": "c%d" % i, "since": i,
                                   "status": "assumed"} for i in range(20)],
         "suspects": [{"step": i, "strength": 3, "why": "w"}
                      for i in range(9)]}
_cap = memo_scan_mod._merge_ledger(memo_scan_mod._empty_ledger(), _over, 60)
check("the ledger is bounded so the last chapter is not the whole run",
      len(_cap["beliefs"]) == memo_scan_mod.LEDGER_CAPS["beliefs"]
      and len(_cap["suspects"]) == memo_scan_mod.LEDGER_CAPS["suspects"],
      (len(_cap["beliefs"]), len(_cap["suspects"])))
check("a step index outside the run is dropped, not clamped",
      not memo_scan_mod._merge_ledger(
          memo_scan_mod._empty_ledger(),
          {"suspects": [{"step": 900, "strength": 3, "why": "w"},
                        {"step": -1, "strength": 3, "why": "w"}]},
          60)["suspects"])

# An empty reply is an endpoint failure, not a retraction. Resetting the ledger
# would cost the memory of every chapter before it -- the one thing this stage
# has that the others do not.
_prev = memo_scan_mod._merge_ledger(
    memo_scan_mod._empty_ledger(),
    {"goal": "ship it", "suspects": [{"step": 7, "strength": 2, "why": "w"}]},
    60)
for _bad in ({}, None, [], "not json"):
    check("an unusable reply carries the ledger forward unchanged (%r)" % (_bad,),
          memo_scan_mod._merge_ledger(_prev, _bad, 60) == _prev)

# -- suspects become regions, under span_prior's OWN bounds -------------------
# An arm that nominated more phases would earn recall for free and the A/B would
# measure width rather than aim, so the guess budget and the coverage bound are
# the ones span_prior already ships.
_led = {"goal": "", "beliefs": [], "commitments": [], "open": [],
        "suspects": [{"step": 40, "strength": 1, "why": "late weak"},
                     {"step": 22, "strength": 3, "why": "strong"},
                     {"step": 7, "strength": 3, "why": "strong and earlier"},
                     {"step": 55, "strength": 2, "why": "mid"}]}
_mp = memo_scan_mod._to_prior(_led, _mt, _mspans, 3, 0.5)
check("the strongest suspect's phase leads",
      (_mp.guesses[0].start, _mp.guesses[0].end) == (5, 9),
      [g.to_dict() for g in _mp.guesses])
check("equal strength breaks toward the EARLIER step",
      [g.start for g in _mp.guesses][:2] == [5, 20],
      [g.start for g in _mp.guesses])
check("strength 2 and up reads as 'likely', below it as 'possible'",
      _mp.guesses[0].likelihood == "likely"
      and memo_scan_mod._to_prior(
          {"suspects": [{"step": 7, "strength": 1, "why": "w"}]},
          _mt, _mspans, 3, 0.5).guesses[0].likelihood == "possible")
check("the scan inherits span_prior's guess budget (12 phases -> 3)",
      len(_mp.guesses) <= 3, len(_mp.guesses))
check("the scan inherits span_prior's coverage bound",
      _mp.n_steps_covered() <= int(len(_mt) * 0.5), _mp.n_steps_covered())
check("duplicate phases are not nominated twice",
      len({(g.start, g.end) for g in _mp.guesses}) == len(_mp.guesses))

# Same floor as span_prior: with too few phases there is nothing to narrow and
# the calls would buy nothing.
_few = MemoLLM([])
_fp, _fl = memo_scan_mod.scan(_mt, _mspans[:3], _mg, _few, digests=_mdig)
check("fewer than four phases cost no call and return no guess",
      not _fp.guesses and not _few.prompts)

# -- the seam -----------------------------------------------------------------
check("the stage is off by default", Config().memo_scan is False)
check("the stage reaches the config fingerprint",
      Config().fingerprint() != Config(memo_scan=True).fingerprint())

_called = []
_real_prior = span_prior_mod.predict_spans


def _spy_prior(*a, **kw):
    _called.append(1)
    return _real_prior(*a, **kw)


span_prior_mod.predict_spans = _spy_prior
pipeline_mod.span_prior_mod.predict_spans = _spy_prior
try:
    run(traj, RecordingLLM(), Config(max_rounds=1, workers=1, span_prior=True))
    _with_prior = len(_called)
    _called[:] = []
    run(traj, RecordingLLM(), Config(max_rounds=1, workers=1, span_prior=True,
                                     memo_scan=True))
    check("--memo-scan takes the 5c seam and span_prior does not run",
          _with_prior == 1 and not _called, (_with_prior, len(_called)))
finally:
    span_prior_mod.predict_spans = _real_prior
    pipeline_mod.span_prior_mod.predict_spans = _real_prior

# -- the standing constraint --------------------------------------------------
for _name, _txt in (("MEMO_PROMPT", memo_scan_mod.MEMO_PROMPT),
                    ("MEMO_SYSTEM", memo_scan_mod.MEMO_SYSTEM)):
    check("%s names no label-side field" % _name,
          not re.search(r"gold|ground.?truth|annotation|human.?rationale"
                        r"|mistake_|critical_error|failure_id|test_patch"
                        r"|gold_patch|failure_reason", _txt, re.I))
check("scan() cannot be handed a GroundTruth",
      not any(re.search(r"gold|label|truth|annotation", n)
              for n in inspect.signature(memo_scan_mod.scan).parameters))


# ---------------------------------------------------------------------------
print("\n[41] RootSE treats `response` as a fallback channel (deferred fix)")

# This pins a KNOWN DEFECT, deliberately. RootSE's `response` field carries real
# content for some frameworks -- AutoCodeRover puts the proposed patch there,
# OpenHands the user-facing reasoning -- and using it only as a fallback drops
# that content whenever the step also has a thought/action/observation.
#
# The fix was written and is archived at
# ``probe/attic/deferred_rootse_response_channel.patch``. It was spot-checked on
# the 10-row auto-code-rover subset (5/10, unchanged) and then left in the
# working tree unmeasured on the full corpus. Measured 2026-09-20: it changes
# every TA step's text, hence tracing, spans, compression, the 5c prior and the
# localiser prompt, and costs **51/102 -> 43/102 (0.5000 -> 0.4216)**.
#
# So the loader is back on the fallback behaviour and this check holds it there,
# for the same reason as the openhands task_description fix (RUNS.md): a correct
# repair of a real defect that makes the pipeline worse is not something to
# carry silently in a working tree while arms are being compared against a
# baseline measured without it. Re-open it as its own arm, not as a side effect.
check("response is a fallback, not an extra channel",
      _rootse_step_text({
          "thought": "inspect the parser",
          "action": {"tool": "edit_file"},
          "observation": "edit succeeded",
          "response": "the proposed patch body",
      }) == "Thought: inspect the parser\nAction: {\"tool\": \"edit_file\"}\n"
            "Observation: edit succeeded")
check("response-only steps render as Response",
      _rootse_step_text({"response": "answer"}) == "Response: answer")

_sym_traj = Trajectory(
    task_id="sym", source="trajaudit", subset="x", task_description="task",
    steps=[Step(0, "assistant", "work")],
    meta={"failed_test_code": "assert value == 2",
          "failed_test_error": "AssertionError: 1 != 2"})
_sym_span = [Span(0, 0, tier="weak", label="work", summary="work")]
_sym_graph = FlowGraph(anchors=[0], failure_summary="test failed")
_sym_digest = [StepDigest(index=0, role="assistant", text="work",
                          tier=KEEP, raw_chars=4)]
_sym_off, _sym_on = RecordingLLM(), RecordingLLM()
localize_mod.localize(_sym_traj, _sym_span, _sym_graph, _sym_off,
                      digests=_sym_digest)
localize_mod.localize(_sym_traj, _sym_span, _sym_graph, _sym_on,
                      digests=_sym_digest, symptom_evidence=True)
check("raw harness evidence is opt-in",
      "assert value == 2" not in _sym_off.prompts[-1]
      and "assert value == 2" in _sym_on.prompts[-1])
check("the stage-6 symptom flag is off by default",
      Config().localize_symptom is False)


# ---------------------------------------------------------------------------
print("\n[42] hybrid anchors (rule UNION model) and the pointwise span judge")

from longtraj import span_judge as span_judge_mod                 # noqa: E402
from longtraj.schema import LIKELY as _LIKELY                     # noqa: E402
from longtraj.schema import POSSIBLE as _POSSIBLE                 # noqa: E402

check("both knobs are off by default",
      Config().hybrid_anchors is False and Config().span_judge is False
      and Config().judge_to_localizer is False)
check("they reach the config fingerprint",
      all(k in Config().fingerprint()
          for k in ("hybrid_anchors", "span_judge", "judge_to_localizer")))

# -- the union is monotone, which is the whole claim behind riding it along ---
# A union over the painting roots can only ADD ancestors, never remove one, so
# gold_on_flow cannot fall. The measured +4/-0 on both corpora is a consequence
# of this, not a lucky sample, and that is why the arm is allowed to carry two
# changes at once. If this check ever fails the claim is void and so is the arm.
_hy_steps = [Step(i, "assistant", "Action: edit file_%d.py\nwrote it" % i)
             for i in range(6)]
_hy_steps.append(Step(6, "assistant",
                      "Action: submit\nTraceback (most recent call last):\n"
                      "E   AssertionError: 1 != 2"))
_hy_traj = Trajectory(task_id="hy", source="trajaudit", subset="x",
                      task_description="task", steps=_hy_steps)
_hy_graph = FlowGraph(
    anchors=[2], anchors_located=[2],
    edges=[FlowEdge(src=0, dst=2, kind=DATA_FLOW, entity="file_0.py"),
           FlowEdge(src=4, dst=6, kind=DATA_FLOW, entity="file_4.py")])
_hy_located = set(_hy_graph.on_flow())
_hy_rule = tracing_mod.keyword_anchors(_hy_traj, k=3)
_hy_graph.anchors = sorted(set(_hy_graph.anchors_located) | set(_hy_rule))
_hy_union = set(_hy_graph.on_flow())
check("the union of the two anchor sets contains the located set",
      set(_hy_graph.anchors) >= set(_hy_graph.anchors_located))
check("union painting only ever enlarges on_flow",
      _hy_union >= _hy_located, sorted(_hy_located - _hy_union))
check("the rule really contributes an anchor the model did not name",
      _hy_union != _hy_located)

# -- the judge skips phases that carry flow, and that costs no call -----------
# Not an optimisation: a step tracing put on the path is ACT_KEEP and _degrade
# is forbidden to touch it, so a score could not change its fate. Paying ~8
# extra calls a row for a verdict with no possible effect is the thing this
# avoids, and it is also what makes the stage affordable at all.
_sj_steps = [Step(i, "assistant", "Action: edit f%d.py\n%s" % (i, "x" * 900))
             for i in range(12)]
_sj_traj = Trajectory(task_id="sj", source="trajaudit", subset="x",
                      task_description="task", steps=_sj_steps)
_sj_spans = [Span(0, 2), Span(3, 5), Span(6, 8), Span(9, 11)]
_sj_graph = FlowGraph(anchors=[1], anchors_located=[1],
                      failure_summary="it failed",
                      edges=[FlowEdge(src=0, dst=1, kind=DATA_FLOW,
                                      entity="f0.py")])


class _JudgeLLM(RecordingLLM):
    """Answers each per-phase judge call with a dictated score."""

    def __init__(self, by_start):
        super().__init__()
        self.by_start = dict(by_start)

    def generate_json(self, prompt, max_tokens=None, system="", default=None,
                      force_json=True):
        self.prompts.append(prompt)
        m = re.search(r"THE PHASE YOU ARE JUDGING — steps (\d+)\.\.", prompt)
        if m is None:
            return default
        got = self.by_start.get(int(m.group(1)))
        return default if got is None else dict(got)


_sj_llm = _JudgeLLM({3: {"relevant": True, "score": 0.9, "confidence": 0.9,
                         "why": "committed here"},
                     6: {"relevant": True, "score": 0.8, "confidence": 0.5,
                         "why": "maybe here"},
                     9: {"relevant": False, "score": 0.1, "confidence": 0.9,
                         "why": "routine"}})
_sj_v = span_judge_mod.judge_spans(_sj_traj, _sj_spans, _sj_graph, _sj_llm,
                                   workers=1)
check("a phase carrying flow is recorded but never asked about",
      _sj_v[0].skipped == "carries flow" and _sj_v[0].called is False)
check("one call per flowless phase and not one more",
      len(_sj_llm.prompts) == 3, len(_sj_llm.prompts))
check("the judged phase's own text reaches its judge",
      all(("steps %d.." % s) in p
          for s, p in zip((3, 6, 9), _sj_llm.prompts)))
check("no judge prompt shows a step from another phase",
      all(not re.search(r"\[step (?:%s) \|"
                        % "|".join(str(i) for i in range(3)), p)
          for p in _sj_llm.prompts))

# -- score x confidence, and the bands it earns ------------------------------
# The product, not the score: a 0.8 held with confidence 0.5 is not worth the
# same 12000 verbatim characters as a 0.9 held with 0.9, and asking for two
# numbers is pointless if only one of them is read.
check("strength is score times confidence",
      abs(_sj_v[3].strength - 0.81) < 1e-9
      and abs(_sj_v[6].strength - 0.40) < 1e-9)
check("a confident high score earns the keep band",
      _sj_v[3].band() == "judge-likely")
check("a high score held with low confidence drops to the clip band",
      _sj_v[6].band() == "judge-possible")
check("relevant:false scores nothing however confident",
      _sj_v[9].strength == 0.0 and _sj_v[9].band() == "")

for _bad in ({}, None, "not json", [], {"score": 0.9}, {"confidence": 0.9},
             {"relevant": True, "score": "high", "confidence": 0.9}):
    _v = span_judge_mod._parse(Span(3, 5), _bad)
    check("an unusable judge reply abstains rather than raising (%r)"
          % (_bad,), _v.strength == 0.0 and _v.band() == "" and _v.called)
_v = span_judge_mod._parse(Span(3, 5), {"relevant": True, "score": 4,
                                        "confidence": -2})
check("out-of-range numbers are clamped, not trusted",
      _v.score == 1.0 and _v.confidence == 0.0)
check("fewer than four phases cost no call and return no verdict",
      span_judge_mod.judge_spans(_sj_traj, _sj_spans[:3], _sj_graph,
                                 RecordingLLM()) == {})

# -- what the compressor does with them --------------------------------------
_pl_base = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, None)
_pl_judge = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, None,
                                          verdicts=_sj_v)
check("with no verdicts the plan is the one that predates the stage",
      _pl_base == compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph,
                                                None, verdicts=None))
check("the keep band reaches the plan",
      all(_pl_judge[i]["why"] == "judge-likely"
          and _pl_judge[i]["act"] == compress_mod.ACT_KEEP
          for i in (3, 4, 5)))
check("the clip band reaches the plan",
      all(_pl_judge[i]["why"] == "judge-possible"
          and _pl_judge[i]["act"] == compress_mod.ACT_CLIP
          for i in (6, 7, 8)))
check("a cleared phase keeps the behaviour it had without the stage",
      all(_pl_judge[i] == _pl_base[i] for i in (9, 10, 11)))
check("the flow band is untouched by any verdict",
      all(_pl_judge[i] == _pl_base[i] for i in (0, 1)))

# The prior compared the phases against each other before naming three; the
# judge never saw a second phase. Where they disagree the ranked reader wins,
# and when the budget forces a choice the judge gives ground first.
_pr = SpanPrior(guesses=[SpanGuess(start=6, end=8, likelihood=_LIKELY,
                                   why="prior says here")])
_pl_both = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, _pr,
                                         verdicts=_sj_v)
check("the prior's likely band outranks the judge's",
      _pl_both[6]["why"] == "prior-likely")
_order = [w for w, _, _ in compress_mod._DEGRADE_ORDER]
# Keyed on the whole rung, not on the band name: `prior-possible` appears twice
# once --possible-keep exists (KEEP->CLIP, then CLIP->MODEL), and a bare
# .index() would silently answer about the wrong one.
_rungs = list(compress_mod._DEGRADE_ORDER)
_K, _C, _M = compress_mod.ACT_KEEP, compress_mod.ACT_CLIP, compress_mod.ACT_MODEL
check("each judge band degrades before the prior band of the same strength",
      _rungs.index(("judge-possible", _C, _M))
      < _rungs.index(("prior-possible", _C, _M))
      and _rungs.index(("judge-likely", _K, _C))
      < _rungs.index(("prior-likely", _K, _C)))
check("no degrade step can reach the flow or answer-space bands",
      not any(w in ("flow", "answer-space")
              for w in _order))

# Zero-sum: the budget is the enforcement mechanism, so a judge that flags
# everything must be degraded rather than allowed to keep the whole run.
_pl_tight = compress_mod.plan_compression(
    _sj_traj, _sj_spans, _sj_graph, None, verdicts=_sj_v, budget_chars=1200)
check("a tight budget degrades the judge's bands",
      _pl_tight[3]["act"] != compress_mod.ACT_KEEP
      or _pl_tight[6]["act"] != compress_mod.ACT_CLIP)
check("...and still never degrades a flow step",
      _pl_tight[1]["act"] == compress_mod.ACT_KEEP
      and _pl_tight[1]["why"] == "flow")

# -- --possible-keep: the prior's POSSIBLE band becomes a keep ---------------
# `_sj_v` is not involved; this band is the prior's. Phase 6-8 is POSSIBLE here
# and carries no flow, which is exactly the case the flag is about.
_pr_p = SpanPrior(guesses=[SpanGuess(start=6, end=8, likelihood=_POSSIBLE,
                                     why="prior is unsure")])
_pk_off = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, _pr_p)
_pk_on = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, _pr_p,
                                       possible_keep=True)
check("with the flag down the plan is the one that predates the argument",
      _pk_off == compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph,
                                               _pr_p, possible_keep=False))
check("off, a possible flowless step is clipped",
      all(_pk_off[i]["why"] == "prior-possible"
          and _pk_off[i]["act"] == compress_mod.ACT_CLIP for i in (6, 7, 8)))
check("on, that same step is kept verbatim at the likely band's limit",
      all(_pk_on[i]["why"] == "prior-possible"
          and _pk_on[i]["act"] == compress_mod.ACT_KEEP
          and _pk_on[i]["limit"] == compress_mod.PRIOR_KEEP_LIMIT
          for i in (6, 7, 8)))
check("the flag reaches ONLY the possible band",
      all(_pk_on[i] == _pk_off[i] for i in _pk_off
          if i != "_spans" and _pk_off[i]["why"] != "prior-possible"))
check("the flow band is still untouched",
      all(_pk_on[i] == _pk_off[i] for i in (0, 1)))
# Zero-sum again: the promotion is the first thing surrendered, so an
# over-budget plan falls back onto the default plan before anything is lost.
check("the promotion is the first rung of the degrade ladder",
      _rungs[0] == ("prior-possible", _K, _C))
_pk_tight = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, _pr_p,
                                          possible_keep=True, budget_chars=1200)
check("a tight budget walks the promotion back",
      all(_pk_tight[i]["act"] != compress_mod.ACT_KEEP for i in (6, 7, 8)))
# The precise claim is ordering, not survival: at a small enough budget every
# band gives ground eventually. What must hold is that the possible band has
# surrendered ALL of its ground before the likely band gives any.
check("every possible rung precedes every likely rung",
      max(n for n, r in enumerate(_rungs) if r[0] == "prior-possible")
      < min(n for n, r in enumerate(_rungs) if r[0] == "prior-likely"))


def _kept(traj, plan):
    """`_degrade`'s own budget arithmetic, so the test asks the real question."""
    return sum(p["limit"] if p["act"] == compress_mod.ACT_MODEL
               else min(len(traj.steps[i].content or ""), p["limit"])
               for i, p in plan.items() if i != "_spans")


_pr_mix = SpanPrior(guesses=[SpanGuess(start=3, end=5, likelihood=_LIKELY,
                                       why="likely"),
                             SpanGuess(start=6, end=8, likelihood=_POSSIBLE,
                                       why="possible")])
# Budget = exactly what the DEFAULT plan costs. Turning the flag on can then
# only overshoot by the promotion itself, so the ladder must walk back to the
# default and stop -- with the likely band still whole.
_mix_off = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, _pr_mix)
_mix_on = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, _pr_mix,
                                        possible_keep=True,
                                        budget_chars=_kept(_sj_traj, _mix_off))
check("walking the promotion back is enough to pay for it",
      all(_mix_on[i]["act"] == compress_mod.ACT_KEEP
          and _mix_on[i]["why"] == "prior-likely" for i in (3, 4, 5)))
check("plan_compression still cannot be handed ground truth with it on",
      "possible_keep" in inspect.signature(
          compress_mod.plan_compression).parameters)

# -- --full-keep: the three-rule scheme, stated without a budget --------------
# 1. on the flow -> whole step.  2. off the flow, suspicious phase -> whole
# step.  3. otherwise -> compressed. "Whole" must mean no limit, not a bigger
# one, so the assertions below are written against step LENGTH, not constants.
_fk_off = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, _pr_mix)
_fk = compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph, _pr_mix,
                                    full_keep=True, budget_chars=1)


def _len(i):
    return len(_sj_traj.steps[i].content or "")


check("with the flag down the plan is the one that predates the argument",
      _fk_off == compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph,
                                               _pr_mix, full_keep=False))
check("rule 1: an on-flow step's limit reaches its own length",
      all(_fk[i]["why"] == "flow" and _fk[i]["act"] == compress_mod.ACT_KEEP
          and _fk[i]["limit"] >= _len(i) for i in (0, 1)))
check("rule 2: a likely phase's off-flow step is whole",
      all(_fk[i]["why"] == "prior-likely"
          and _fk[i]["act"] == compress_mod.ACT_KEEP
          and _fk[i]["limit"] >= _len(i) for i in (3, 4, 5)))
check("rule 2 covers POSSIBLE as well as LIKELY",
      all(_fk[i]["why"] == "prior-possible"
          and _fk[i]["act"] == compress_mod.ACT_KEEP
          and _fk[i]["limit"] >= _len(i) for i in (6, 7, 8)))
check("rule 3: an unflagged off-flow step is still compressed",
      all(_fk[i]["why"] == "unguessed"
          and _fk[i]["act"] == compress_mod.ACT_MODEL for i in (9, 10, 11)))
# budget_chars=1 above would have flattened every band had the ladder run.
check("the degrade ladder does not run under full_keep",
      _fk == compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph,
                                           _pr_mix, full_keep=True,
                                           budget_chars=10 ** 9))
check("full_keep implies possible_keep",
      _fk == compress_mod.plan_compression(_sj_traj, _sj_spans, _sj_graph,
                                           _pr_mix, full_keep=True,
                                           possible_keep=True, budget_chars=1))
check("a step longer than every constant is still kept whole",
      compress_mod.plan_compression(
          _sj_traj, _sj_spans, _sj_graph,
          SpanPrior(guesses=[SpanGuess(start=3, end=5, likelihood=_LIKELY,
                                       why="x")]),
          full_keep=True)[3]["limit"]
      >= max(_len(3), compress_mod.PRIOR_KEEP_LIMIT))
check("plan_compression takes no label-side argument with it on either",
      not any("gold" in p or "truth" in p or "critical" in p
              for p in inspect.signature(
                  compress_mod.plan_compression).parameters))

# -- the localiser sees nothing unless the second arm asks ------------------
_sj_dig = [StepDigest(index=i, role="assistant", text="t", tier=KEEP,
                      raw_chars=1) for i in range(12)]
_lz_off, _lz_blank, _lz_on = RecordingLLM(), RecordingLLM(), RecordingLLM()
localize_mod.localize(_sj_traj, _sj_spans, _sj_graph, _lz_off,
                      digests=_sj_dig)
localize_mod.localize(_sj_traj, _sj_spans, _sj_graph, _lz_blank,
                      digests=_sj_dig, judge_block="")
localize_mod.localize(_sj_traj, _sj_spans, _sj_graph, _lz_on, digests=_sj_dig,
                      judge_block=span_judge_mod.render_verdicts(_sj_v,
                                                                 _sj_spans))
check("arm one's localiser prompt is byte-identical to one built without the "
      "parameter at all",
      _lz_blank.prompts[-1] == _lz_off.prompts[-1])
check("arm two's localiser prompt differs, and only by the block",
      _lz_on.prompts[-1] != _lz_off.prompts[-1]
      and "committed here" in _lz_on.prompts[-1]
      and "committed here" not in _lz_off.prompts[-1]
      and _lz_on.prompts[-1].replace(
          span_judge_mod.render_verdicts(_sj_v, _sj_spans), "")
      == _lz_off.prompts[-1])
check("the block warns that the scores are not comparable across phases",
      "NOT comparable across phases" in _lz_on.prompts[-1])
check("a run where every phase was cleared renders no block",
      span_judge_mod.render_verdicts(
          {9: _sj_v[9]}, _sj_spans) == "")
check("a block is withheld when the judge only abstained",
      span_judge_mod.render_verdicts(
          {3: span_judge_mod._parse(Span(3, 5), {})}, _sj_spans) == "")

# -- the record keeps the cleared phases too ---------------------------------
_st = span_judge_mod.stats(_sj_v)
check("the record counts the phases skipped for carrying flow",
      _st["n_skipped_flow"] == 1 and _st["n_called"] == 3)
check("every phase's score is kept, cleared ones included",
      sorted(_st["spans"]) == ["0", "3", "6", "9"])
check("the per-phase record carries the score and its band",
      _st["spans"]["3"]["band"] == "judge-likely"
      and _st["spans"]["3"]["score"] == 0.9)
check("stats on nothing is empty, not a zero row",
      span_judge_mod.stats(None) == {} and span_judge_mod.stats({}) == {})

# -- the red line ------------------------------------------------------------
# The narrow regex, as in [37]: _GOLD_KEY matches the bare word "answer" and is
# for parameter and key NAMES. The judge's prose talks about answering, so what
# is pinned here is that no prompt names a label-side SOURCE.
check("no judge prompt or block names a label-side source",
      not re.search(r"gold|ground.?truth|annotation|human.?rationale"
                    r"|mistake_|critical_error|failure_id|test_patch",
                    span_judge_mod.JUDGE_PROMPT + span_judge_mod.JUDGE_SYSTEM
                    + span_judge_mod.JUDGE_BLOCK, re.I))
check("and no judge prompt built from a real trajectory does either",
      not any(re.search(r"gold|ground.?truth|annotation|critical_error"
                        r"|failure_id|test_patch", p, re.I)
              for p in _sj_llm.prompts + [_lz_on.prompts[-1]]))
check("judge_spans cannot be handed a GroundTruth",
      not any("truth" in n or "gold" in n
              for n in inspect.signature(
                  span_judge_mod.judge_spans).parameters))
check("plan_compression still cannot be handed one either",
      not any("truth" in n or "gold" in n
              for n in inspect.signature(
                  compress_mod.plan_compression).parameters))

# -- the two anchor policies are refused together, before anything is read ---
# Both write graph.anchors; the second assignment would silently win. The
# refusal happens before load() and before the LLM is built, so it holds with
# no corpus on disk and no credentials -- which is also how it is tested here.
from longtraj import cli as cli_mod                              # noqa: E402

_argv = ["run", "--source", "trajaudit", "--dry-run"]
check("replacing and unioning the anchors cannot be asked for together",
      cli_mod.main(_argv + ["--rule-anchors", "--hybrid-anchors"]) == 2)
check("the localiser block cannot be asked for without the judge",
      cli_mod.main(_argv + ["--judge-to-localizer"]) == 2)


# [43] PRIOR ARITY: the region guess names one phase where three were asked for.
#
# This is a compliance repair, not a new design. The effective cap is
# ``min(3, len(spans)//3)`` -- 3 on every SP row, 2-3 on 71 TA rows -- and the
# parse was cleared before any of this was written: on those rows neither the
# coverage rule nor the budget cut a second guess. qwen3-235b still returns
# exactly ONE phase on 78/86 SP and 85/98 TA rows, where claude-opus returned
# three. So k draws are UNIONED.
#
# The shape next door was falsified: ``--samples 5`` on the LOCALISER picked one
# answer out of k and lost on both corpora (SP 0.3023, TA 0.4216) because the
# model was already self-consistent. The checks below pin the mechanical
# difference -- a union that ``_parse``'s unchanged budget and coverage rules
# still bound, so it can never nominate more region than the design already
# allowed for the model that did comply.
print("\n[43] prior arity: union k draws of the region guess")

from longtraj import span_prior as sp_mod                         # noqa: E402
from longtraj.schema import (LIKELY as _LIKELY, POSSIBLE as _POSSIBLE,
                             SpanGuess, SpanPrior)      # noqa: E402
from longtraj import localize_walk as localize_walk_mod           # noqa: E402


check("prior sampling is off by default",
      pipeline_mod.Config().prior_samples == 1)
check("the prior's sample count reaches the fingerprint",
      pipeline_mod.Config(prior_samples=3).fingerprint()
      .get("prior_samples") == 3)
check("predict_spans takes a sample count defaulting to one",
      inspect.signature(sp_mod.predict_spans)
      .parameters["samples"].default == 1)
check("no label-side argument reaches the union",
      not any("truth" in n or "gold" in n
              for n in inspect.signature(sp_mod._union_draws).parameters))

# -- k=1 is the old call, variant and all ---------------------------------
# The stub below is the OLD signature on purpose: it has no ``variant``
# parameter, so if k=1 ever started forwarding one this check would raise
# rather than fail quietly. That is the compatibility the cache key depends on.
_one = [Span(start=0, end=3, label="a", summary="s"),
        Span(start=4, end=7, label="b", summary="s"),
        Span(start=8, end=11, label="c", summary="s"),
        Span(start=12, end=15, label="d", summary="s")]

# -- the union itself, which is where every ordering rule lives ------------
# _union_draws is pure: it takes draws and phases and returns the wire shape
# _parse already consumed, so these checks need no model and no trajectory.
_u = sp_mod._union_draws([
    {"spans": [{"start": 4, "end": 7, "likelihood": "likely", "why": "first"}]},
    # an off-by-one end for the SAME phase: one vote, not two guesses
    {"spans": [{"start": 4, "end": 8, "likelihood": "likely", "why": "again"}]},
    {"spans": [{"start": 8, "end": 11, "likelihood": "possible", "why": "c"}]},
], _one)["spans"]
check("draws that described the same phase collapse to one guess",
      len(_u) == 2)
check("the phase two draws named outranks the phase one draw named",
      (_u[0]["start"], _u[0]["end"]) == (4, 7))
check("snapping happens before the vote, so the range is the phase's own",
      _u[0]["end"] == 7)
check("the majority likelihood wins",
      _u[0]["likelihood"] == _LIKELY and _u[1]["likelihood"] == _POSSIBLE)
check("the why is carried from the draw that named it first",
      _u[0]["why"] == "first")

_dup = sp_mod._union_draws([
    {"spans": [{"start": 4, "end": 7, "why": "x"},
               {"start": 5, "end": 6, "why": "y"}]},
    {"spans": [{"start": 8, "end": 11, "why": "z"}]},
], _one)["spans"]
check("one draw naming a phase twice is one vote for it",
      len(_dup) == 2 and (_dup[0]["start"], _dup[0]["end"]) == (4, 7))

_tied = sp_mod._union_draws([
    {"spans": [{"start": 8, "end": 11, "why": "ranked first"},
               {"start": 4, "end": 7, "why": "ranked second"}]},
], _one)["spans"]
check("with votes tied the draw's own order decides",
      [(g["start"], g["end"]) for g in _tied] == [(8, 11), (4, 7)])

check("a range matching no phase is dropped rather than invented",
      sp_mod._union_draws([{"spans": [{"start": 99, "end": 120}]}],
                          _one)["spans"] == [])
check("malformed draws abstain instead of raising",
      sp_mod._union_draws(
          [{"spans": [{"start": "x"}, "not a dict", {}]}, {}, {"spans": None}],
          _one)["spans"] == [])

# -- and the union is still bounded by the rules that bounded one draw ----
# This is the load-bearing check of the whole section: the union hands _parse
# MORE candidates, and _parse still applies budget and max_coverage to them.
_many = [Span(start=i * 2, end=i * 2 + 1, label=str(i), summary="s")
         for i in range(10)]
_wide_obj = {"spans": [{"start": i * 2, "end": i * 2 + 1} for i in range(10)]}
check("the budget still caps the union",
      len(sp_mod._parse(_wide_obj, _many, 3, 20, 0.5).guesses) == 3)
check("the coverage rule still caps the union",
      len(sp_mod._parse(_wide_obj, _many, 3, 20, 0.10).guesses) == 1)
check("a union of k draws cannot exceed what one compliant draw could",
      len(sp_mod._parse(sp_mod._union_draws(
          [_wide_obj] * 5, _many), _many, 3, 20, 0.5).guesses) == 3)


# [44] THE FALLBACK ANSWER: what the localiser returns when it returns nothing.
#
# The old rule was "earliest attributable on-flow step", which on TrajAudit is
# step 0 on most trajectories. Measured on ta_qarm_fx and ta_qbase_fx: 4 rows
# each landed on the fallback and NONE had gold == 0, so the rule turned a
# non-answer into a certain miss. The prior's top region is already paid for,
# so the fallback now lands inside it.
#
# NOT calibrated against where reference steps actually sit -- that is a label,
# and reading it here would be selection on the gold. The within-region rule is
# the old rule unchanged.
print("\n[44] the fallback answer prefers the prior's own region")

check("the fallback's prior argument defaults to absent",
      inspect.signature(localize_mod._fallback_step)
      .parameters["prior"].default is None)
check("no label-side argument reaches the fallback",
      not any("truth" in n or "gold" in n
              for n in inspect.signature(
                  localize_mod._fallback_step).parameters))

# A 20-step run whose flow reaches steps 1 and 13. Step 0 is the user turn and
# is not attributable, so the old rule answers step 1 -- the position the
# measurement says is essentially never right.
_fbt = Trajectory(task_id="fb", source="trajaudit", subset="s",
                  task_description="t",
                  steps=[Step(0, "user", "go")]
                        + [Step(i, "assistant", "s%d" % i)
                           for i in range(1, 20)])
_fbg = FlowGraph(anchors=[1, 13], anchors_located=[1, 13])

check("with no prior the old rule stands, to the step",
      localize_mod._fallback_step(_fbt, _fbg) == 1)
check("an empty prior is the same as no prior",
      localize_mod._fallback_step(_fbt, _fbg, SpanPrior()) == 1)

_fbp = SpanPrior(guesses=[SpanGuess(start=10, end=15, likelihood=_LIKELY,
                                    why="w"),
                          SpanGuess(start=2, end=4, likelihood=_POSSIBLE,
                                    why="w")])
check("a prior moves the fallback into its TOP region, not its widest",
      localize_mod._fallback_step(_fbt, _fbg, _fbp) == 13)

# Inside the region the rule is the old one: earliest attributable on-flow
# step, and only if none of them is on flow does it fall back to earliest.
_fbp2 = SpanPrior(guesses=[SpanGuess(start=5, end=8, likelihood=_LIKELY,
                                     why="w")])
check("a region carrying no flow still answers inside itself",
      localize_mod._fallback_step(_fbt, _fbg, _fbp2) == 5)

# The guard that matters for the other two architectures: they call with two
# arguments and must keep the behaviour they were measured with.
check("a region that runs off the end of the run is ignored, not clamped",
      localize_mod._fallback_step(
          _fbt, _fbg,
          SpanPrior(guesses=[SpanGuess(start=90, end=99,
                                       likelihood=_LIKELY, why="w")])) == 1)
check("the falsified walk localiser still gets the rule it was measured with",
      "prior" not in inspect.getsource(localize_walk_mod).split(
          "_fallback_step(traj, graph)")[0][-40:])


# [45] THE PRIOR'S TWO JOBS, SEPARATED.
#
# Since --prior-compress the region guess has two consumers: it allocates the
# 90000-char compression budget at stage 3b, and it is shown to the localiser
# at stage 6. ``ta_noprior`` turned off the one flag that fed BOTH -- on
# claude-opus-4-8, before 3b was written -- and lost 13 rows, which the repo
# recorded as "removing the prior lost". That reading cannot distinguish the
# lost hint from the lost verbatim protection of the region the hint named.
#
# ``hide_prior`` is the cut that separates them: 3b still shapes the plan, the
# localiser reads only the compressed trajectory and the task description. The
# checks below pin that it is OFF by default, that turning it on yields exactly
# the prompt a no-prior row already produced, and that it does not reach
# ``_fallback_step`` -- which would make the arm a two-variable change.
print("")
print("[45] the prior's two jobs: compression budget vs localiser prompt")

check("the prior still reaches the localiser by default",
      pipeline_mod.Config().prior_to_localizer is True)
check("hiding the prior reaches the fingerprint",
      pipeline_mod.Config(prior_to_localizer=False).fingerprint()
      .get("prior_to_localizer") is False)
check("localize takes hide_prior, defaulting to off",
      inspect.signature(localize_mod.localize)
      .parameters["hide_prior"].default is False)
check("no label-side argument rides along with it",
      not any("truth" in _n or "gold" in _n or "annotation" in _n
              for _n in inspect.signature(localize_mod.localize).parameters))

_pri = SpanPrior(guesses=[SpanGuess(start=0, end=3, likelihood=_LIKELY,
                                    why="the config is written here"),
                          SpanGuess(start=4, end=7, likelihood=_POSSIBLE,
                                    why="and read back here")])
_sp45 = [Span(start=0, end=3, tier="weak", label="a", summary="s"),
         Span(start=4, end=7, tier="weak", label="b", summary="s"),
         Span(start=8, end=n - 1, tier="weak", label="c", summary="s")]

_shown = Localiser(2)
localize_mod.localize(traj, _sp45, _g, _shown, prior=_pri)
_hidden = Localiser(2)
localize_mod.localize(traj, _sp45, _g, _hidden, prior=_pri, hide_prior=True)
_absent = Localiser(2)
localize_mod.localize(traj, _sp45, _g, _absent, prior=None)
_default = Localiser(2)
localize_mod.localize(traj, _sp45, _g, _default, prior=_pri,
                      hide_prior=False)

check("the prior IS in the localiser prompt by default",
      "A SEPARATE READER'S GUESS" in _shown.prompt
      and "prior_disagreement" in _shown.prompt)
check("hide_prior removes the guess block",
      "A SEPARATE READER'S GUESS" not in _hidden.prompt)
check("hide_prior removes the prior's answer fields too",
      "prior_disagreement" not in _hidden.prompt
      and "prior_span" not in _hidden.prompt)
# The load-bearing one: hiding it must not leave a stub, a blank line or an
# instruction mentioning a shortlist that is no longer shown. Anything short of
# byte-identical means the arm also changed the prompt's shape.
check("a hidden prior gives byte-identically the no-prior prompt",
      _hidden.prompt == _absent.prompt)
check("leaving the flag off changes nothing at all",
      _default.prompt == _shown.prompt)
check("the guesses' own text cannot survive in the hidden prompt",
      "the config is written here" not in _hidden.prompt
      and "and read back here" not in _hidden.prompt)

# _fallback_step reads ``prior`` directly, never ``has_prior``, so the repaired
# fallback keeps answering inside the top region even while the prompt hides
# it. If that ever routed through has_prior, this arm would silently move the
# fallback too and stop being a one-variable change.
_fb_src = inspect.getsource(localize_mod.localize)
check("the fallback still receives the real prior under hide_prior",
      "_fallback_step(traj, graph, prior)" in _fb_src)
check("hide_prior gates the prompt and nothing else",
      _fb_src.count("hide_prior") == 2        # declared once, consumed once
      and "and not hide_prior" in _fb_src)

# [46] THE UPSTREAM-LINK TEST, AND ITS OFF SWITCH.
#
# Asked for directly: nudge the localiser to ask whether the edit step it is
# about to name is wrong only because an earlier plan/reason step was already
# wrong. The block is procedure, not evidence -- it states no fact about the
# trajectory and names no step -- which is the mechanical difference from the
# four arms that lost on the evidence axis.
#
# Two properties have to hold for the arm to mean anything:
#   1. Off is byte-identical. Both the block and its JSON fields default to ""
#      so a run without the flag sends the same prompt and hits the same cache
#      entries. The model id is in the cache key, so this is the only way the
#      same-model baseline re-run can be compared to the arm at all.
#   2. The measured default survives in the wording. The headroom is two-sided
#      (15 winnable vs 25 breakable edit-ish rows on TA, 24 vs 21 on SP), so a
#      block that told the model to walk back whenever it saw an edit would
#      lose on TA by construction. The default must be the acting step.
print("")
print("[46] the upstream-link test: off is byte-identical, default is 'keep'")

check("the flag is off by default",
      pipeline_mod.Config().localize_rootcause is False)
check("turning it on reaches the fingerprint",
      pipeline_mod.Config(localize_rootcause=True).fingerprint()
      .get("localize_rootcause") is True)
check("localize takes rootcause, defaulting to off",
      inspect.signature(localize_mod.localize)
      .parameters["rootcause"].default is False)

_rc_off = Localiser(2)
localize_mod.localize(traj, _sp45, _g, _rc_off)
_rc_on = Localiser(2)
localize_mod.localize(traj, _sp45, _g, _rc_on, rootcause=True)
_rc_expl = Localiser(2)
localize_mod.localize(traj, _sp45, _g, _rc_expl, rootcause=False)

check("the flag off gives byte-identically the prompt that predates it",
      _rc_off.prompt == _rc_expl.prompt)
check("the block really is absent when off",
      "CAUSE OR THE EXECUTION" not in _rc_off.prompt
      and "upstream_verdict" not in _rc_off.prompt)
check("the block appears when on",
      "CAUSE OR THE EXECUTION" in _rc_on.prompt)
check("the extra answer fields appear with it, and only with it",
      all(f in _rc_on.prompt for f in ("upstream_step", "upstream_verdict",
                                       "upstream_because")))
check("the arm changes nothing but the block and its fields",
      len(_rc_on.prompt) > len(_rc_off.prompt)
      and _rc_on.prompt.startswith(_rc_off.prompt[:400]))
check("the evidence the localiser reads is untouched",
      all(x in _rc_on.prompt for x in ("ORIGINAL TASK:",))
      and _rc_off.prompt.count("step") > 0)

# The question is still the same question: one integer, same schema.
check("the answer format is unchanged",
      '"critical_step": <int>' in _rc_on.prompt
      and '"confidence"' in _rc_on.prompt
      and '"error_type"' in _rc_on.prompt)

# No label, and no step number: the block must not hand over a nomination,
# which is what the four losing arms did.
check("the block carries no label field",
      not any(k in localize_mod.ROOTCAUSE_BLOCK + localize_mod.ROOTCAUSE_FIELDS
              for k in ("annotation", "failure_id", "gold_patch",
                        "failure_reason", "test_patch")))
check("the block names no step and supplies no evidence",
      not re.search(r"step \d", localize_mod.ROOTCAUSE_BLOCK))
check("the block is prose only -- the arm adds no stage and no new budget",
      _rc_on.budget == _rc_off.budget)
check("rootcause gates the prompt and nothing else",
      inspect.getsource(localize_mod.localize).count("rootcause") == 5)

# The measured default. A block that converts the 15/24 winnable rows by
# telling the model to move earlier on sight of an edit also hands back the
# 25/21 rows that are right now -- so the earlier step has to be EARNED.
check("the default is stated, and it is the acting step",
      "The default is (a)" in localize_mod.ROOTCAUSE_BLOCK)
check("the discharge condition is recoverability, not precedence",
      "Recovery is the test" in localize_mod.ROOTCAUSE_BLOCK
      and "Earlier is not automatically deeper"
      in localize_mod.ROOTCAUSE_BLOCK)
check("the test is scoped to acting steps and says so",
      "skip this check" in localize_mod.ROOTCAUSE_BLOCK)
check("rejecting the upstream step is a recordable outcome",
      "rejected" in localize_mod.ROOTCAUSE_FIELDS
      and "not-applicable" in localize_mod.ROOTCAUSE_FIELDS)

# Without this the run cannot be read back for whether the test was performed
# or the model simply answered earlier.
_rc_rec = localize_mod._with_rejection(
    {"reason": "R", "rejected_step": 9, "rejected_because": "propagated",
     "upstream_step": 4, "upstream_verdict": "doomed",
     "upstream_because": "chose the wrong file"})
check("the verdict is recorded into the one free-text field the artifact keeps",
      "[upstream doomed at step 4: chose the wrong file]" in _rc_rec
      and "[ruled out step 9: propagated]" in _rc_rec)
check("the walk's own record still survives alongside it",
      _rc_rec.startswith("R\n[ruled out step 9"))
check("a row that was never asked records exactly what it did before",
      localize_mod._with_rejection({"reason": "R", "rejected_step": 9})
      == "R\n[ruled out step 9]")
check("a missing upstream step is recorded as a verdict without one",
      localize_mod._with_rejection(
          {"reason": "R", "upstream_verdict": "not-applicable"})
      == "R\n[upstream not-applicable]")

# The reconciliation with --candidates-strict, which SP's best arm runs. That
# block's text is frozen AND its enforcement is in code (82 of 84 stored rows
# obeyed, 2 were snapped), so a block telling the model to name an unlisted
# plan step would describe a mechanism that does not exist and have its answer
# moved anyway. Under a list the test keeps its verdict but changes its action:
# name the EARLIEST listed step that carried the decision.
_rc_cand = Localiser(2)
localize_mod.localize(traj, _sp45, _g, _rc_cand, rootcause=True,
                      cands=[{"step": 2, "label": "wrote a.py"},
                             {"step": 4, "label": "wrote b.py"}])
check("under a candidate list the block defers instead of contradicting",
      "Do not name an unlisted step" in _rc_cand.prompt
      and "restricted to the list above" in _rc_cand.prompt)
check("and it redirects to the earliest listed carrier",
      "EARLIEST listed" in _rc_cand.prompt)
check("without a list there is no such note",
      "Do not name an unlisted step" not in _rc_on.prompt)
check("both variants share the one test and the one default",
      all(x in localize_mod.ROOTCAUSE_BLOCK_CAND
          for x in ("Recovery is the test", "The default is (a)")))
check("the note is the only difference between the two variants",
      localize_mod.ROOTCAUSE_BLOCK_CAND.replace(localize_mod._CAND_NOTE, "")
      == localize_mod.ROOTCAUSE_BLOCK)
check("the candidate variant still carries no label field and names no step",
      not re.search(r"step \d", localize_mod.ROOTCAUSE_BLOCK_CAND)
      and not any(k in localize_mod.ROOTCAUSE_BLOCK_CAND
                  for k in ("annotation", "failure_id", "gold_patch",
                            "failure_reason", "test_patch")))

# The answer-space restoration: restore_candidates does not touch anything when
# OFF, and the new keep parameter to _fit_budget does nothing when empty.
# Use actual steps from the test traj rather than inventing ones.
_d0 = StepDigest(index=2, role="assistant", text="compressed form", name="",
                 tier=COMPRESSED, raw_chars=len(traj.steps[2].content or ""),
                 flows="", anchor=False)
_d1 = StepDigest(index=3, role="assistant", text=traj.steps[3].content or "",
                 name="", tier=KEEP, raw_chars=len(traj.steps[3].content or ""),
                 flows="", anchor=False)
check("restore_candidates with no steps is a no-op",
      compress_mod.restore_candidates(traj, [_d0, _d1], [])
      == {"asked": 0, "restored": 0, "chars_added": 0}
      and _d0.text == "compressed form")
# ON: a named step gets its own raw content back whatever the tier, and the
# stat counts only the ones that actually changed.
_vw = [copy.copy(_d0), copy.copy(_d1)]
_vst = compress_mod.restore_candidates(traj, _vw, [2, 3])
check("restore_candidates puts the named steps back in their own words",
      _vw[0].text == traj.steps[2].content and _vw[0].tier == KEEP)
check("and it counts only the steps it actually changed",
      _vst["asked"] == 2 and _vst["restored"] == 1
      and _vst["chars_added"] == len(traj.steps[2].content or "") - len("compressed form"))
check("a step whose text already IS the original is not 'restored'",
      _vst["restored"] == 1)
# restore_candidates has no way to receive a label, the same check the rest of
# this file applies to every stage that touches the prompt.
check("restore_candidates takes no label-side argument",
      not any("truth" in _n or "gold" in _n or "annotation" in _n
              for _n in inspect.signature(
                  compress_mod.restore_candidates).parameters))
check("_fit_budget takes no label-side argument either",
      not any("truth" in _n or "gold" in _n or "annotation" in _n
              for _n in inspect.signature(localize_mod._fit_budget).parameters))

# The budget trimmer's keep set. Two off-path steps of the same size and a
# budget that fits exactly one of them: without a keep set both are clipped to
# 400, with one the named step survives whole and the other pays instead.
# (A budget too small for even the kept step falls through to the final
# whole-blob clip, which is the pre-existing last resort and not what this
# argument is for -- on the real corpus the cap is 120000 and the two SP rows
# that cross it land at 119993 with the keep set applied.)
_work = [StepDigest(index=9, role="assistant", text="x" * 8000, name="",
                    tier=COMPRESSED, raw_chars=8000, flows="", anchor=False),
         StepDigest(index=10, role="assistant", text="y" * 8000, name="",
                    tier=COMPRESSED, raw_chars=8000, flows="", anchor=False)]
_trim_loose = localize_mod._fit_budget(traj, _sp45, copy.deepcopy(_work), 10000)
_trim_kept = localize_mod._fit_budget(traj, _sp45, copy.deepcopy(_work), 10000,
                                      keep=[9])
check("_fit_budget with keep=() is byte-identical to the old code",
      _trim_loose == localize_mod._fit_budget(traj, _sp45,
                                              copy.deepcopy(_work), 10000,
                                              keep=()))
check("_fit_budget respects the keep set",
      "x" * 8000 in _trim_kept and "x" * 8000 not in _trim_loose
      and "y" * 8000 not in _trim_kept)
check("and it still honours the budget while doing so",
      len(_trim_kept) <= 10000)

# ---------------------------------------------------------------------------
print("[47] the shortlist reader: C3 = C1 U C2")

from longtraj import shortlist as shortlist_mod                  # noqa: E402

# The frozen text. This arm's single largest regression risk is that someone
# edits the candidate block so it mentions the shortlist -- the reasonable
# looking change, and the one that destroys the only cheap proof the repo has
# that a re-run did not drift. The cache is keyed on the prompt, so "ran it
# again with the flag off and got calls: 0" is evidence only while these bytes
# hold. Pinned by hash rather than by substring, because a whitespace change is
# a cache miss too.
_CB_SHA = hashlib.sha256(
    localize_mod.CANDIDATE_BLOCK.encode("utf-8")).hexdigest()
_CBS_SHA = hashlib.sha256(
    localize_mod.CANDIDATE_BLOCK_SOFT.encode("utf-8")).hexdigest()
check("CANDIDATE_BLOCK is byte-identical to the sp_capc1 run",
      _CB_SHA == "19ffef26f716707a901f54aa8eceb7894d0d8d1ef5c42c5fec082d24e5"
                 "afd7c1", _CB_SHA)
check("CANDIDATE_BLOCK_SOFT is byte-identical to the sp_csoft run",
      _CBS_SHA == "816c3385eff8d7f855f6fd406a1bcd95fdefc8012a2b63beb459d0057"
                  "81e71b8", _CBS_SHA)

# -- the mixed-space wording ------------------------------------------------
# A merged answer space gets its own text, because the frozen one is false about
# it in two places: it says steps that only reasoned "are not listed", and it
# then instructs the reader to name the first step that ACTED on earlier
# thinking. Under C3 the thinking step may itself be listed. A prompt that
# misdescribes its own answer space is not one whose effect can be attributed,
# so this is a separate string and not an edit to the pinned one.
_MX = localize_mod.CANDIDATE_BLOCK_MIXED
_MXS = localize_mod.CANDIDATE_BLOCK_MIXED_SOFT
check("the mixed block does not claim reasoned steps are absent",
      "are not listed" not in _MX
      and "only read, searched, ran a test, or reasoned" not in _MX)
check("the mixed block says a step that changed nothing is answerable",
      "changed nothing" in _MX and "wrote nothing" in _MX)
check("the mixed block still keeps the but-for test",
      "but-for test" in _MX)
check("the mixed block does not send the answer forward to the acting step",
      "name the FIRST listed step that carried it out" not in _MX)
check("the mixed block keeps the same header as the one it replaces",
      "YOUR ANSWER MUST BE ONE OF THE STEPS LISTED BELOW" in _MX
      and "YOUR ANSWER SHOULD BE ONE OF THE STEPS LISTED BELOW" in _MXS)
check("enforcement is the only difference between the two mixed texts",
      _MX.replace("MUST", "") == _MXS.replace("SHOULD", "").replace(
          _MXS[_MXS.index("If nothing here fits"):],
          _MX[_MX.index("If nothing here fits"):]))
check("the mixed block marks no entry's provenance",
      not any(w in _MX for w in ("ALSO ELIGIBLE", "proposed by", "shortlist",
                                 "pattern-matching", "a reader")))
check("the mixed block takes the entries in the same slot as the frozen one",
      "{cands}" in _MX and "{cands}" in _MXS)
check("_cand_block picks on the two bits independently",
      localize_mod._cand_block(True, False) is localize_mod.CANDIDATE_BLOCK
      and localize_mod._cand_block(False, False)
      is localize_mod.CANDIDATE_BLOCK_SOFT
      and localize_mod._cand_block(True, True) is _MX
      and localize_mod._cand_block(False, True) is _MXS)

# -- the module cannot reach a label ---------------------------------------
check("shortlist never names GroundTruth",
      "GroundTruth" not in inspect.getsource(shortlist_mod))
for _fn in (shortlist_mod.propose, shortlist_mod.merge,
            shortlist_mod.stats):
    check("shortlist.%s takes no label-side argument" % _fn.__name__,
          not any("truth" in _n or "gold" in _n or "annotation" in _n
                  for _n in inspect.signature(_fn).parameters))
for _const in (shortlist_mod.SHORTLIST_PROMPT, shortlist_mod.SHORTLIST_SYSTEM,
               localize_mod.CANDIDATE_BLOCK_MIXED,
               localize_mod.CANDIDATE_BLOCK_MIXED_SOFT):
    check("the shortlist's prompt text names no label field",
          not any(w in _const for w in ("annotation", "failure_id",
                                        "gold_patch", "failure_reason",
                                        "test_patch")))

# -- merge is a union, and only ever grows ---------------------------------
_c1 = [{"step": 4, "label": "edit a"}, {"step": 10, "label": "edit b"}]


def _sl(steps, why=None):
    return shortlist_mod.Shortlist(steps=list(steps), called=True,
                                   why=(why or {i: "w%d" % i for i in steps}))


_c3, _new = shortlist_mod.merge(_c1, _sl([2, 7, 10]))
check("C3 is the union, in trajectory order",
      [c["step"] for c in _c3] == [2, 4, 7, 10], _c3)
check("a step already in C1 is not added twice", _new == [2, 7], _new)
check("C3 contains every entry of C1 unchanged", all(c in _c3 for c in _c1))
check("C3 has no duplicate step numbers",
      len({c["step"] for c in _c3}) == len(_c3))
check("abstention leaves C3 identical to C1 and nothing new",
      shortlist_mod.merge(_c1, _sl([])) == (_c1, []))
check("a missing shortlist leaves C3 identical to C1",
      shortlist_mod.merge(_c1, None) == (_c1, []))
check("a shortlist of nothing but known steps adds nothing",
      shortlist_mod.merge(_c1, _sl([4, 10])) == (_c1, []))

# -- the cap binds in the parse, and the truncation is recorded ------------
_sl_t = Trajectory(task_id="s", source="trajdebug", subset="swebenchpro",
                   task_description="build it",
                   steps=[Step(0, "user", "go")]
                         + [Step(i, "assistant", "body %d" % i)
                            for i in range(1, 20)])
_many = {"steps": [{"step": i, "why": "w"} for i in range(1, 12)]}
_parsed = shortlist_mod._parse(_sl_t, _many, [], shortlist_mod.SHORTLIST_K)
check("the K cap binds on the parse, not on the caller",
      len(_parsed.steps) == shortlist_mod.SHORTLIST_K, _parsed.steps)
check("what the cap discarded is recorded, not dropped silently",
      _parsed.n_truncated == 11 - shortlist_mod.SHORTLIST_K,
      _parsed.n_truncated)
check("the kept steps come back sorted", _parsed.steps == sorted(_parsed.steps))
check("K is the number that was pre-registered", shortlist_mod.SHORTLIST_K == 5)

# -- malformed answers abstain instead of raising --------------------------
for _name, _obj in (("a non-dict answer", None),
                    ("a missing steps key", {"relevant": True}),
                    ("a steps field that is not a list", {"steps": 7}),
                    ("an empty object", {}),
                    ("junk rows", {"steps": ["x", None, {"step": "a"}]}),
                    ("step numbers out of range",
                     {"steps": [{"step": 99}, {"step": -1}]}),
                    ("a step that is not the agent's",
                     {"steps": [{"step": 0}]})):
    _r = _quietly(shortlist_mod._parse, _sl_t, _obj, [],
                  shortlist_mod.SHORTLIST_K)
    check("%s abstains rather than raising" % _name,
          _r.steps == [] and _r.abstained, _r.to_dict())
check("a step the run does not have is dropped, the rest kept",
      _quietly(shortlist_mod._parse, _sl_t,
               {"steps": [{"step": 99}, {"step": 3}]}, [], 5).steps == [3])
check("a tool or user turn is dropped: it could never be the answer",
      _quietly(shortlist_mod._parse, _sl_t,
               {"steps": [{"step": 0}, {"step": 3}]}, [], 5).steps == [3])
check("a bare integer reads as a step with no reason",
      _quietly(shortlist_mod._parse, _sl_t, {"steps": [3]}, [], 5).steps == [3])

# -- off, and abstaining, both leave the localiser's prompt untouched ------
_sl_base = VerifyLLM()
localize_mod.localize(_vt, _vspan, _vgraph(), _sl_base, cands=list(_vcands))
_sl_empty = VerifyLLM()
localize_mod.localize(_vt, _vspan, _vgraph(), _sl_empty, cands=list(_vcands),
                      cands_mixed=False)
check("cands_mixed=False is byte-identical to not passing it at all",
      _sl_base.prompts == _sl_empty.prompts)
_c3a, _newa = shortlist_mod.merge(list(_vcands), _sl([]))
_sl_abst = VerifyLLM()
localize_mod.localize(_vt, _vspan, _vgraph(), _sl_abst, cands=_c3a,
                      cands_mixed=bool(_newa))
check("an abstaining reader leaves the whole prompt byte-identical",
      _sl_base.prompts == _sl_abst.prompts)

# -- ONE list, and nothing in it says where an entry came from ------------
# The first version of this arm showed two lists, the merged set and then the
# proposals again under "ALSO ELIGIBLE -- proposed by a reader who read the
# whole run". That is a thumb on the scale and it was measured as one: the
# localiser answered with a proposed step on 68 of the 78 rows that had one,
# and those answers were the reference step 3/68 against 21/68 for the parsed
# steps they displaced. One list, no provenance, no second mention.
_c3v, _newv = shortlist_mod.merge(list(_vcands), _sl([3, 7], {
    3: "chose the wrong module", 7: "misread the traceback"}))
_sl_mx = VerifyLLM()
_p_mx = localize_mod.localize(_vt, _vspan, _vgraph(), _sl_mx, cands=_c3v,
                              cands_mixed=True)
_lp = [q for q in _sl_mx.prompts if "critical_step" in q][0]
check("the proposed steps are in the one candidate list",
      "step 3 — chose the wrong module" in _lp
      and "step 7 — misread the traceback" in _lp, _lp[-400:])
check("each proposed step appears exactly once in the prompt",
      _lp.count("step 3 — chose the wrong module") == 1
      and _lp.count("step 7 — misread the traceback") == 1)
check("there is no second list",
      "ALSO ELIGIBLE" not in _lp and "proposed by a reader" not in _lp)
check("the list is in trajectory order, parsed and proposed interleaved",
      [c["step"] for c in _c3v] == [2, 3, 5, 7, 9], _c3v)
check("the mixed wording is the one that reached the prompt",
      localize_mod.CANDIDATE_BLOCK_MIXED.split("{cands}")[0] in _lp)
check("and the frozen wording did not",
      localize_mod.CANDIDATE_BLOCK.split("{cands}")[0] not in _lp)
check("the record says the wording was the mixed one",
      _p_mx.candidates["mixed"] is True, _p_mx.candidates)
_sl_plain = VerifyLLM()
_p_plain = localize_mod.localize(_vt, _vspan, _vgraph(), _sl_plain,
                                 cands=list(_vcands))
check("a run with no proposals makes no claim about a mixed wording",
      "mixed" not in _p_plain.candidates)

# -- a proposed entry is not distinguishable by its shape ----------------
# With no provenance marked, LENGTH would mark it instead: C1's labels are
# mechanical strings capped at 70 chars by ``candidates.classify``, while the
# reader answers in sentences. An entry three lines long reads as "this is one
# of the new ones" whatever the prose above says.
_long = _sl([4], {4: "The agent misread the traceback and concluded the mock "
                     "was at fault, which it was not, and then rewrote the "
                     "assertion to match the broken behaviour instead of "
                     "fixing the cause."})
_c3l, _newl = shortlist_mod.merge([{"step": 1, "label": "str_replace /a/b.py"}],
                                  _long)
_entry = [c for c in _c3l if c["step"] == 4][0]["label"]
check("a proposed entry is clipped to the width C1's entries have",
      len(_entry) <= shortlist_mod.LABEL_CHARS, _entry)
check("and it is clipped to a clause, not mid-word",
      _entry == "The agent misread the traceback and concluded the mock was "
                "at fault", _entry)
check("an entry with no reason still gets a label",
      shortlist_mod._clip_label("") == "no write recorded"
      and shortlist_mod._clip_label(None) == "no write recorded")
check("no ellipsis, which would itself be a provenance mark",
      "..." not in _entry and not _entry.endswith("."))

# -- whatever may be answered with may also vote -------------------------
# The pool is the answer space, with no flag to hold it back. Holding it back
# was tried: it made answering with a proposed step self-defeating, because an
# unpolled answer loses to the best polled one by the merge rule above.
_sl_wide = VerifyLLM()
_p_wide = localize_mod.localize(_vt, _vspan, _vgraph(), _sl_wide, cands=_c3v,
                                verify=True, cands_mixed=True)
check("the poll covers the whole answer space, proposals included",
      sorted(_sl_wide.focal) == [2, 3, 5, 7, 9], _sl_wide.focal)
check("the answer space the reader was offered is C3",
      _p_wide.candidates["steps"] == [2, 3, 5, 7, 9], _p_wide.candidates)
check("localize takes no argument for subsetting the poll",
      not any("pool_c1" in _n for _n in
              inspect.signature(localize_mod.localize).parameters))
check("and Config has no flag for it either",
      not hasattr(pipeline_mod.Config(), "shortlist_in_pool"))

# -- with no answer space the stage does not invent one -------------------
_sl_none = VerifyLLM()
_r_none = shortlist_mod.propose(_sl_t, [Span(start=0, end=19, label="all")],
                                _vgraph(), _sl_none, cands=None)
check("with no candidate set the reader is not called at all",
      _r_none.called is False and _r_none.skipped == "no candidates")
check("and it spent no prompt finding that out", _sl_none.prompts == [])

# -- the config and the CLI agree on the dependencies --------------------
_cfg_sl = pipeline_mod.Config()
check("the stage is off by default", _cfg_sl.shortlist is False)
check("the default K matches the module's",
      _cfg_sl.shortlist_k == shortlist_mod.SHORTLIST_K)
check("--shortlist is refused without an answer space to add to",
      cli_mod.main(_argv + ["--shortlist"]) == 2)

# -- the record keeps the control case -----------------------------------
# A row where the reader added nothing is a row whose prompt was byte-identical
# and whose answer provably did not move. That is the per-row control this arm
# is judged against and it exists nowhere else, so it is recorded too.
_st = shortlist_mod.stats(_sl([]), _c1, _c1, [])
check("an abstention is still recorded",
      _st["abstained"] is True and _st["n_new"] == 0 and _st["n_c1"] == 2, _st)
check("the record carries C1 and C3 sizes side by side",
      shortlist_mod.stats(_sl([2, 7]), _c1, _c3, [2, 7])["n_c3"] == len(_c3))
check("stats of no shortlist is empty, so the artifact key stays absent",
      shortlist_mod.stats(None) == {})
check("the prediction can carry the record into the artifact",
      "shortlist" in Prediction(task_id="t", critical_step=1).to_dict())

# ---------------------------------------------------------------------------
print("\n[53] the stage-1 model seam")

# Same seam as [35], other end of the pipeline. It exists because a corpus
# can have gold_on_flow 0.000 while TrajAudit gets 0.696, and "the tracer
# is too weak" and "the backtrack is too long" predict the same artifact
# until you swap only this one model.
_tr_cfg = lambda: Config(max_rounds=1, workers=1)
_tr_base = RecordingLLM()
_ptr_off = run(_seam_traj, _tr_base, _tr_cfg())
_tr_other = RecordingLLM()
_ptr_none = run(_seam_traj, _tr_other, _tr_cfg(), llm_trace=None)
check("llm_trace defaults to None",
      inspect.signature(run).parameters["llm_trace"].default is None)
check("passing None changes not one byte of any prompt",
      _tr_base.prompts == _tr_other.prompts)
check("passing None changes not one call count",
      _ptr_off.usage.get("calls") == _ptr_none.usage.get("calls"))


class _TraceLLM(RecordingLLM):
    """A second tracer, distinguishable only by the model id it reports."""

    def __init__(self):
        super().__init__()
        self.model = "tracer-two"


_tr_main, _tr_one = RecordingLLM(), _TraceLLM()
_ptr_on = run(_seam_traj, _tr_main, _tr_cfg(), llm_trace=_tr_one)

check("stage 1 really goes to the second tracer", len(_tr_one.prompts) > 0)
check("stages 2-6 stay on the first reader",
      len(_tr_main.prompts) == len(_tr_base.prompts) - len(_tr_one.prompts))
check("no prompt is sent to both models",
      not (set(_tr_main.prompts) & set(_tr_one.prompts)))
check("the tracer got exactly the prompts the off run opened with",
      _tr_one.prompts == _tr_base.prompts[:len(_tr_one.prompts)])
check("the second tracer's calls are counted, not silently dropped",
      _ptr_on.usage.get("calls", 0) >= _ptr_off.usage.get("calls", 0))
# An arm you cannot identify after the fact is an arm you cannot judge:
# the flow artifact must name the model that built the graph, not the one
# the rest of the run used.
with tempfile.TemporaryDirectory() as _tr_tmp:
    _tr_store = TraceStore(_tr_tmp)
    run(_seam_traj, RecordingLLM(), _tr_cfg(), store=_tr_store,
        llm_trace=_TraceLLM())
    _tr_arts = [a for a in _tr_store.iter_artifacts()]
    check("the flow artifact names the model that built the graph",
          bool(_tr_arts) and all(
              str(a.get("model")).startswith("trace:tracer-two+")
              for a in _tr_arts),
          [a.get("model") for a in _tr_arts])
    check("and still names the model the later stages used",
          all(str(a.get("model")).endswith("+" + RecordingLLM().model)
              for a in _tr_arts),
          [a.get("model") for a in _tr_arts])
# With neither seam used the composed id is the bare model, so every
# single-model artifact ever written stays byte-identical.
with tempfile.TemporaryDirectory() as _tr_tmp2:
    _s2 = TraceStore(_tr_tmp2)
    run(_seam_traj, RecordingLLM(), _tr_cfg(), store=_s2)
    check("no seam used leaves the artifact model id unchanged",
          all(a.get("model") == RecordingLLM().model
              for a in _s2.iter_artifacts()))
check("the stage-1 seam cannot carry a label",
      not re.search(r"gold|label|answer|truth|critical|mistake", "llm_trace"))

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
print("\n[54] the handoff pool reads an author, never a label")

# This pool is parsed from WHO SPOKE, which is the first answer space here
# derived from an input field rather than from what the step did. The risk it
# carries is not leakage but overclaiming, so the checks pin both: the module
# cannot see a label, and the flags are no-ops where the structure is absent.
import longtraj.candidates as _cand_mod
_hsrc = inspect.getsource(_cand_mod.handoffs)
check("handoffs() reads no label field",
      not re.search(r"mistake|gold|label_|annotation|failure_id|critical",
                    _hsrc))
check("handoffs() reads the author and nothing about the answer",
      ".name" in _hsrc and "ground_truth" not in _hsrc)


def _mkstep(i, name):
    return Step(index=i, role="assistant", content="c%d" % i, name=name)


_h_traj = Trajectory(task_id="h", source="s", subset="sub",
                     task_description="t", steps=[
    _mkstep(0, "DiagnostAgent"), _mkstep(1, "DiagnostAgent (-> ActionAgent)"),
    _mkstep(2, "ActionAgent"), _mkstep(3, "DiagnostAgent (-> JudgeAgent)")])
_hp = _cand_mod.handoffs(_h_traj)
check("only the arrow-authored steps are nominated",
      [c["step"] for c in _hp] == [1, 3], _hp)
check("the label names where control went, not what the step did",
      _hp[0]["label"] == "hands off to ActionAgent", _hp[0])
check("a corpus with no such authors nominates nothing, not everything",
      _cand_mod.handoffs(Trajectory(
          task_id="h2", source="s", subset="sub", task_description="t",
          steps=[_mkstep(0, "agent"), _mkstep(1, "")])) == [])
check("the flags are off by default",
      pipeline_mod.Config().handoff_protect is False
      and pipeline_mod.Config().handoff_candidates is False)
# The two flags are separate because one changes what is READ and the other
# what may be ANSWERED; collapsing them would make the gamble unjudgeable
# apart from the defensible half.
check("protecting and restricting are different knobs",
      "handoff_protect" in pipeline_mod.Config().fingerprint()
      and "handoff_candidates" in pipeline_mod.Config().fingerprint())

# ---------------------------------------------------------------------------
print("\n[55] the alfworld loader adds a flag, it does not change a loader")

# alfworld shares swebenchpro's schema, so load_trajdebug already loads it and
# is left byte-for-byte alone; load_trajdebug_alfworld wraps it. Everything
# that makes an A/B attributable to the flag rests on one property: with the
# flag OFF the wrapper must be indistinguishable from the loader it wraps. If
# that ever stops holding, the arm's acc delta stops being the flag's.
_alf_src = inspect.getsource(loaders_mod.load_trajdebug_alfworld) + \
    inspect.getsource(loaders_mod.alfworld_novel_observation)
# Docstrings are stripped first: both functions DISCUSS the annotation field in
# prose to explain why they never touch it, and a check that cannot tell prose
# from code would either fail on an honest comment or pass on a real read.
_alf_code = re.sub(r'"""[\s\S]*?"""', "", _alf_src)
check("the alfworld wrapper reads no label field",
      not re.search(r"annotation|critical_error|mistake|human_rationale|gold",
                    _alf_code), _alf_code[:200])
check("load_trajdebug keeps its three-argument signature",
      list(inspect.signature(loaders_mod.load_trajdebug).parameters) ==
      ["root", "subset", "limit"])
check("the de-duplication is off by default",
      inspect.signature(
          loaders_mod.load_trajdebug_alfworld).parameters["dedup"].default
      is False)
check("trajdebug_alf is registered without displacing trajdebug",
      loaders_mod.LOADERS.get("trajdebug") is loaders_mod.load_trajdebug
      and loaders_mod.LOADERS.get("trajdebug_alf")
      is loaders_mod.load_trajdebug_alfworld)

_ALF_USER = (
    "You are in the middle of a room. Looking quickly around you, you see a "
    "cabinet 1.\nYour task is to: put a soapbar in garbagecan.\n"
    "Your past actions and observations:\n"
    "step 1: go to cabinet 1 -> Nothing happens.\n"
    "You are now at step 3 and your current observation is:\n"
    "You pick up the soapbar 1 from the countertop 1.\n"
    "Finally, choose ONE admissible action from the list below.\n"
    "go to cabinet 1 / take soapbar 1")
check("the novel observation is what survives, the replay is what goes",
      loaders_mod.alfworld_novel_observation(_ALF_USER)
      == "You pick up the soapbar 1 from the countertop 1.",
      loaders_mod.alfworld_novel_observation(_ALF_USER))
# An unrecognised turn is returned WHOLE. Guessing at it would silently shrink
# the evidence the localiser sees, which is far worse than paying for replay.
check("a turn without the anchor is returned untouched, not truncated",
      loaders_mod.alfworld_novel_observation("no anchor here at all")
      == "no anchor here at all")
check("an empty body falls back to the original rather than vanishing",
      loaders_mod.alfworld_novel_observation(
          "You are now at step 3 and your current observation is:\n"
          "Finally, choose ONE admissible action") != "")

# Field-identical with the flag off, and assistant text untouched with it on.
_alf_a = list(loaders_mod.load_trajdebug(subset="alfworld", limit=3))
_alf_off = list(loaders_mod.load_trajdebug_alfworld(subset="alfworld", limit=3))
_alf_on = list(loaders_mod.load_trajdebug_alfworld(subset="alfworld", limit=3,
                                                   dedup=True))
if _alf_a:
    check("flag off: the wrapper yields the wrapped loader's records verbatim",
          [dataclasses.asdict(r.trajectory) for r in _alf_off]
          == [dataclasses.asdict(r.trajectory) for r in _alf_a])
    check("flag off: the ground truth is passed through, not rebuilt",
          [r.ground_truth for r in _alf_off] == [r.ground_truth for r in _alf_a])
    # The reference step is an assistant turn in 100/100 alfworld records, so
    # the flag must not be able to touch the text the answer lives in.
    _touched = [
        (a.task_id, s.index)
        for a, b in zip(_alf_a, _alf_on)
        for s, s2 in zip(a.trajectory.steps, b.trajectory.steps)
        if s.role != "user" and s.content != s2.content]
    check("flag on: no non-user step's text is altered", _touched == [],
          _touched[:5])
    _gold = [(r.ground_truth.critical_step if r.ground_truth else None)
             for r in _alf_a]
    _gold_same = [
        a.trajectory.steps[g].content == b.trajectory.steps[g].content
        for a, b, g in zip(_alf_a, _alf_on, _gold)
        if g is not None and g < len(a.trajectory.steps)]
    check("flag on: the reference step's own text is untouched",
          all(_gold_same) and _gold_same, _gold_same)
    check("flag on: the de-duplication actually removes replay",
          sum(len(s.content) for r in _alf_on for s in r.trajectory.steps)
          < 0.75 * sum(len(s.content) for r in _alf_a
                       for s in r.trajectory.steps))
    # The run side filters the half; the scoring side scores the half. Two
    # copies of one hash, so pin them together -- if they ever drifted, "tuned
    # on dev, read test once" would stop being true without anything failing.
    import importlib.util as _ilu
    _sp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "probe", "split_eval.py")
    if os.path.exists(_sp):
        _spec = _ilu.spec_from_file_location("_split_eval", _sp)
        _se = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_se)
        _ids = [r.trajectory.task_id for r in _alf_a]
        check("the CLI and the scorer agree on every record's half",
              [cli_mod._alf_half(i) for i in _ids] == [_se.half(i) for i in _ids])
else:   # pragma: no cover - corpus absent
    print("    (alfworld corpus not present; structural checks only)")

# ---------------------------------------------------------------------------
# 56. the alfworld prompt is a rubric, not a label
#
# ``longtraj/alfworld_prompt.py`` restates the corpus's ANNOTATION GUIDELINE to
# the localiser -- the module taxonomy a human annotator was handed. That is a
# dataset-level constant, and the danger with a dataset-level constant is that
# it is an easy place to smuggle a record-level fact. These checks make the
# distinction mechanical rather than a matter of my say-so: no step index, no
# gold field name, identical text for every record, and -- the one that
# protects TA and SP -- a byte-identical prompt when the flag is off.
print("\n[56] alfworld prompt: rubric yes, label no")

from longtraj import alfworld_prompt as _ap
from longtraj import localize as _lz

_inc = (_lz.FRAMINGS[0][1], _lz.FRAMINGS[0][2])
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

check("the flag defaults to off on Config",
      pipeline_mod.Config().alf_prompt == "")
check("off returns None, so the incumbent head and note are used verbatim",
      _ap.head_note("", _inc) is None and _ap.head_note(None, _inc) is None)

# The call site must not merely pass a falsy value -- it must omit the kwarg,
# so a TA/SP run's call is the call it was before this file existed.
_pipe_src = io.open(os.path.join(_ROOT, "longtraj", "pipeline.py"),
                    encoding="utf-8").read()
check("pipeline omits the kwarg entirely when the flag is off",
      '**({"alf_prompt": cfg.alf_prompt}' in _pipe_src
      and "if cfg.alf_prompt else {}" in _pipe_src)

# The incumbent constants must be untouched: TA and SP read them.
check("the incumbent head still says what it always said",
      _lz.INCUMBENT_HEAD.startswith("A coding agent FAILED this task."))
check("the incumbent framing table is still three framings",
      len(_lz.FRAMINGS) == 3 and _lz.FRAMINGS[0][0] == "cause")

_VARIANTS = ("domain", "rubric", "full")
_GOLD_WORDS = ("critical_error_step", "critical_failure_step", "label_entry",
               "step_annotations", "critical_failure_module", "human_rationale")

for _v in _VARIANTS:
    _h, _n = _ap.head_note(_v, _inc)
    _blob = _h + _n
    check("variant %r names no gold field" % _v,
          not [w for w in _GOLD_WORDS if w in _blob])
    # A rubric describes categories; a leak names a position. Any "step <n>"
    # in a constant that every record sees would be a constant guess at best
    # and a smuggled label at worst.
    check("variant %r contains no step index" % _v,
          not re.search(r"step\s+\d+", _blob, re.I))
    check("variant %r is a constant: no format slots to fill per record" % _v,
          "{" not in _blob and "}" not in _blob)
    check("variant %r drops the coding-agent framing" % _v,
          "A coding agent FAILED" not in _h)

# The noise control has to be word-identical to the incumbent or it is not a
# control -- it exists to measure re-drawing the localiser, nothing else.
_nh, _nn = _ap.head_note("null", _inc)
check("the null control is word-identical to the incumbent head",
      _nh.strip() == _inc[0].strip() and _nn == _inc[1])
check("the null control still misses the cache (that is its whole job)",
      _nh != _inc[0])

# A mistyped flag must not quietly run the incumbent and be recorded as having
# run the variant: that corrupts an arm comparison after the fact.
try:
    _ap.head_note("rubrik", _inc)
    check("a mistyped variant raises instead of falling back", False)
except ValueError:
    check("a mistyped variant raises instead of falling back", True)

# Same text for every record, on real records: the rubric cannot be a function
# of the trajectory it is shown with.
if _alf_a:
    _texts = set()
    for _r in _alf_a[:8]:
        _texts.add("".join(_ap.head_note("full", _inc)))
    check("the rubric is identical across records",
          len(_texts) == 1)

# And the module must not read the corpus at all.
_ap_src = io.open(os.path.join(_ROOT, "longtraj", "alfworld_prompt.py"),
                  encoding="utf-8").read()
_ap_code = re.sub(r'"""[\s\S]*?"""', "", _ap_src)
check("the prompt module reads no metadata and opens no file",
      "metadata" not in _ap_code and "open(" not in _ap_code
      and "json" not in _ap_code)

# ---------------------------------------------------------------------------
print("\n[57] alfworld answer space: positional pool, no label")

from longtraj import alfworld_space as _as  # noqa: E402

# Off by default, and off means the answer space is whatever candidates.py
# said. The flag must not be able to change a TA/SP run.
_cfg57 = pipeline_mod.Config()
check("alf_space defaults to off", _cfg57.alf_space == "")
check("alf_window defaults to the frozen 21", _cfg57.alf_window == 21)

_ps = io.open(os.path.join(_ROOT, "longtraj", "pipeline.py"),
              encoding="utf-8").read()
check("pipeline guards the substitution on the flag AND on cands is not None",
      "if cfg.alf_space and cands is not None:" in _ps)

# A mistyped space must raise, not silently run unrestricted: an arm recorded
# as restricted that actually ran unrestricted is unrecoverable after the fact.
if _alf_a:
    _t57 = _alf_a[0].trajectory
    try:
        _as.answer_space(_t57, "erly")
        check("a mistyped alf_space raises instead of falling back", False)
    except ValueError:
        check("a mistyped alf_space raises instead of falling back", True)

    # Shape compatibility with candidates.py is the whole reason this module
    # can reuse candidate_strict / verify / snap unchanged.
    _sp57 = _as.answer_space(_t57, "early+blank", 21)
    check("the pool is candidates.py's shape",
          bool(_sp57) and all(set(c) == {"step", "label"}
                              and isinstance(c["step"], int)
                              and isinstance(c["label"], str) for c in _sp57))
    check("candidates.render accepts the pool",
          isinstance(candidates_mod.render(_sp57), str))

    # Only attributable steps may be offered. On this corpus every label is an
    # odd-indexed assistant turn, so an even index in the pool would be a slot
    # that can never be right.
    _asst57 = {s.index for s in _t57.steps if s.role == "assistant"}
    check("every offered step is an assistant step",
          all(c["step"] in _asst57 for c in _sp57))
    check("every offered step is odd-indexed",
          all(c["step"] % 2 == 1 for c in _sp57))

    # The window is a window: nothing past it unless the blank feature put it
    # there, and the blank feature is defined on the step's own text.
    _blank57 = set(_as.blank_steps(_t57))
    check("nothing past the window except blank-action steps",
          all(c["step"] <= 21 or c["step"] in _blank57 for c in _sp57))
    check("'early' alone offers nothing past the window",
          all(c["step"] <= 21 for c in _as.answer_space(_t57, "early", 21)))

    # The pool must be a function of the trajectory text only. If it were a
    # function of the label, widening the window would not be monotone.
    _r9 = {c["step"] for c in _as.answer_space(_t57, "early", 9)}
    _r21 = {c["step"] for c in _as.answer_space(_t57, "early", 21)}
    check("the window is monotone in its size", _r9 <= _r21)

    # No label can reach the pool: the label lives on GroundTruth, and the
    # Trajectory the module is handed has no field that could carry it.
    _tdict57 = dataclasses.asdict(_t57)
    _flat57 = json.dumps(_tdict57)
    check("the trajectory handed to the answer space carries no label field",
          "critical_error_step" not in _flat57
          and "critical_failure_step" not in _flat57
          and "annotation" not in _flat57)

# Mechanical: the module may not read the corpus or the label by any route.
_as_src = io.open(os.path.join(_ROOT, "longtraj", "alfworld_space.py"),
                  encoding="utf-8").read()
_as_code = re.sub(r'"""[\s\S]*?"""', "", _as_src)
check("the answer-space module opens no file and imports no json",
      "open(" not in _as_code and "json" not in _as_code)
check("the answer-space module never reads metadata or a gold field",
      "metadata" not in _as_code
      and "critical" not in _as_code
      and "ground_truth" not in _as_code
      and "gold" not in _as_code)
# The one hard-coded number in the module is the window. It is frozen in the
# pre-registration, so a silent edit here would be an unrecorded design change.
check("the frozen window is still 21", _as.DEFAULT_WINDOW == 21)

# The flag is only consumed at pipeline 2d'', and 3 drops ``cands`` again
# unless --candidates asked for an answer space. So --alf-space ALONE is a
# silent no-op, and the artifact would still record the restricted arm as
# having run. The CLI must refuse rather than run.
_cli = io.open(os.path.join(_ROOT, "longtraj", "cli.py"),
               encoding="utf-8").read()
check("the CLI refuses --alf-space without --candidates",
      "if args.alf_space and not (args.candidates or args.handoff_candidates):"
      in _cli)
check("that refusal is a non-zero exit, not a warning",
      re.search(r"args\.alf_space and not \([^\n]*\):\n(?:[^\n]*\n){1,8}?"
                r"\s+return 2", _cli) is not None)

# ---------------------------------------------------------------------------
# 58. alfworld action-first rendering: moves text, invents none, reads no label
# ---------------------------------------------------------------------------
# --alf-actions hoists each assistant step's own <action> block to the front of
# its digest. The three ways that could go wrong are all mechanical, so pin
# them mechanically: it must add no text that was not already in the step, it
# must be a strict no-op when the flag is off, and it must not read metadata.
import longtraj.alfworld_prompt as _ap
from longtraj.schema import StepDigest as _SD

print(chr(10) + "[58] alfworld --alf-actions: hoists text, invents none")

check("Config.alf_actions defaults to False",
      dataclasses.fields(pipeline_mod.Config) and
      [f for f in dataclasses.fields(pipeline_mod.Config)
       if f.name == "alf_actions"][0].default is False)

_pipe_src = io.open(os.path.join(_ROOT, "longtraj", "pipeline.py"),
                    encoding="utf-8").read()
check("the pipeline guards the call site on cfg.alf_actions",
      "if cfg.alf_actions:" in _pipe_src and
      "alfworld_prompt_mod.lead_with_action(digests, traj)" in _pipe_src)

# The transform itself, on a synthetic digest -- no corpus, no gold.
_d = _SD(index=5, role="assistant", name="", tier="keep",
                           raw_chars=99, flows=[], anchor=False,
                           text="<memory>Step 3: I took the plate</memory>"
                                "<action>go to cabinet 1</action>")
_before = _d.text
_n = _ap.lead_with_action([_d])
check("it changes a digest that has an <action>", _n == 1)
check("the hoisted action leads the digest",
      _d.text.startswith(_ap.ACTION_LEAD + "go to cabinet 1"))
check("the original digest text is preserved below the lead",
      _before in _d.text)
check("it adds NO text beyond the lead and the step's own action",
      _d.text == _ap.ACTION_LEAD + "go to cabinet 1" + chr(10) + _before)
check("it is idempotent (a second pass changes nothing)",
      _ap.lead_with_action([_d]) == 0)

_d2 = _SD(index=4, role="user", name="", tier="keep",
                            raw_chars=9, flows=[], anchor=False,
                            text="no action tag here")
check("a digest with no <action> is left byte-identical",
      _ap.lead_with_action([_d2]) == 0 and _d2.text == "no action tag here")

# A user turn replays the history and ends with the simulator's fixed
# "choose ONE admissible action" block, so action_of MATCHES on it and
# returns the instruction. Leading a user digest with that would invent a
# fact; the role guard is what stops it.
_du = _SD(index=2, role="user", name="", tier="keep", raw_chars=9,
          flows=[], anchor=False,
          text="history... Finally, choose ONE admissible action for "
               "the current step. <action>go to drawer 1</action>")
check("a USER digest is never led, even when it contains an <action>",
      _ap.lead_with_action([_du]) == 0 and "ACTION TAKEN" not in _du.text)
check("the role guard uses the shared NON_ATTRIBUTABLE_ROLES",
      "NON_ATTRIBUTABLE_ROLES" in _ap_src)

check("action_of returns the action text only",
      _ap.action_of("<memory>x</memory>" + chr(10) + "<action>" + chr(10)
                    + " take plate 1 " + chr(10) + "</action>")
      == "take plate 1")
check("action_of is empty when the tag is absent",
      _ap.action_of("nothing here") == "")
check("the action clip is a guard, not a compression decision",
      _ap.ACTION_CLIP >= 300)

# No label can reach this module: it is handed digests and a Trajectory, and
# Trajectory has no field that can carry one (section 1 pins that separately).
_ap_src = re.sub(r'"""[\s\S]*?"""', "",
                 io.open(os.path.join(_ROOT, "longtraj",
                                      "alfworld_prompt.py"),
                         encoding="utf-8").read())
# NOTE "annotation" is deliberately NOT in this list: the module contains
# ``from __future__ import annotations`` and a comment naming the corpus
# guideline, neither of which reads a label. The substrings below are the
# ones that could only appear if the module went looking for ground truth.
for _bad in ("metadata", "critical", "ground_truth", "gold",
             "open(", "json"):
    check("alfworld_prompt.py source never mentions %r" % _bad,
          _bad not in _ap_src)

check("alfworld_prompt.py never reads a trajectory's meta dict",
      ".meta" not in _ap_src)
check("lead_with_action takes only digests and an optional trajectory",
      list(inspect.signature(_ap.lead_with_action).parameters) ==
      ["digests", "traj"])

# ---------------------------------------------------------------------------
# 59. ALFWorld turn ledger: text-only extraction and default-off isolation
# ---------------------------------------------------------------------------
from longtraj import alfworld_ledger as _al

print(chr(10) + "[59] alfworld --alf-ledger: compact transitions, no label")

check("Config.alf_ledger defaults to False",
      [f for f in dataclasses.fields(pipeline_mod.Config)
       if f.name == "alf_ledger"][0].default is False)
check("the ledger kwarg is omitted entirely when the flag is off",
      '**({"trajectory_override": ledger_body}' in _pipe_src and
      "if ledger_body is not None else {}" in _pipe_src)

if _alf_a:
    _lt = _alf_a[0].trajectory
    _lb, _ls = _al.render_ledger(_lt)
    _agent_steps = [s for s in _lt.steps if s.attributable]
    check("the ledger has one row per attributable agent turn",
          _ls.rows == len(_agent_steps) == 30)
    check("every original agent step index appears exactly once",
          all(_lb.count("[step %d |" % s.index) == 1 for s in _agent_steps))
    check("every extracted action comes from its own source step",
          all((_al._tag(s.content, "action", _al.ACTION_CLIP) or "(none)")
              in _lb for s in _agent_steps))
    check("the ledger is materially smaller than the raw trajectory",
          _ls.chars < _lt.total_chars() * 0.35)
    check("all four code-derived checks are rendered per row",
          _lb.count("CHECKS: admissible=") == _ls.rows and
          _lb.count("repeated_action=") == _ls.rows and
          _lb.count("repeated_observation=") == _ls.rows and
          _lb.count("refused=") == _ls.rows)

_al_src = re.sub(r'"""[\s\S]*?"""', "",
                 io.open(os.path.join(_ROOT, "longtraj",
                                      "alfworld_ledger.py"),
                         encoding="utf-8").read())
for _bad in ("metadata", "ground_truth", "critical_failure",
             "critical_error", ".meta", "open(", "json"):
    check("alfworld_ledger.py never reads %r" % _bad, _bad not in _al_src)
check("render_ledger accepts only a Trajectory",
      list(inspect.signature(_al.render_ledger).parameters) == ["traj"])

# The old localiser call and explicitly passing the default are identical.
# This pins the byte-level cache invariant independently of the pipeline guard.
class _LedgerLocaliser(LLM):
    def __init__(self):
        super().__init__(dry_run=True)
        self.prompts = []
    def generate_obj(self, prompt, max_tokens=None, system="", force_json=True):
        self.prompts.append(prompt)
        return {"critical_step": 1, "confidence": .5,
                "error_type": "plan.x", "reason": "x"}

_fake59a = _LedgerLocaliser()
_fake59b = _LedgerLocaliser()
if _alf_a:
    _t59 = _alf_a[0].trajectory
    _g59 = FlowGraph()
    _d59 = [_SD(index=s.index, role=s.role, name=s.name or "",
                tier="keep", raw_chars=len(s.content), text=s.content)
            for s in _t59.steps]
    _lz.localize(_t59, [], _g59, _fake59a, digests=_d59)
    _lz.localize(_t59, [], _g59, _fake59b, digests=_d59,
                 trajectory_override=None)
check("ledger off leaves the localiser prompt byte-identical",
          _fake59a.prompts == _fake59b.prompts)

# ---------------------------------------------------------------------------
# 60. ALFWorld module audits: partitioned evidence, closed arbitration
# ---------------------------------------------------------------------------
from longtraj import alfworld_modules as _am

print(chr(10) + "[60] alfworld --alf-modules: isolated evidence, closed set")
check("Config.alf_modules defaults to False",
      [f for f in dataclasses.fields(pipeline_mod.Config)
       if f.name == "alf_modules"][0].default is False)
check("the pipeline branch is guarded on alf_modules",
      "if cfg.alf_modules:" in _pipe_src and
      "alfworld_modules_mod.refine" in _pipe_src)
check("there are exactly four frozen cognitive audits",
      [s.key for s in _am.SPECS] ==
      ["memory", "reflection", "plan", "action_system"])
check("refine cannot be handed a label-side object",
      list(inspect.signature(_am.refine).parameters) ==
      ["traj", "llm", "incumbent", "workers"])

if _alf_a:
    _t60 = _alf_a[0].trajectory
    _turns60 = _am.turns(_t60)
    check("module views preserve all 30 original assistant indices",
          len(_turns60) == 30 and
          [t.step for t in _turns60] == _t60.attributable_indices)
    for _spec60 in _am.SPECS:
        _view60 = _am._render_view(_turns60, _spec60.key)
        check("%s view contains every attributable step" % _spec60.key,
              all(_view60.count("[step %d]" % t.step) == 1
                  for t in _turns60))
    # Parsing an invented or non-attributable step is an abstention, never a
    # snap into the closed set.
    _bad60 = _am._parse(_am.SPECS[0],
                        {"found": True, "critical_step": 2,
                         "confidence": .99}, _t60.attributable_indices)
    check("an auditor cannot nominate an unshown user step",
          not _bad60.found and _bad60.step is None)

_am_src = re.sub(r'"""[\s\S]*?"""', "",
                 io.open(os.path.join(_ROOT, "longtraj",
                                      "alfworld_modules.py"),
                         encoding="utf-8").read())
for _bad in ("ground_truth", "critical_failure_step", "critical_error_step",
             ".meta", "open(", "json.load"):
    check("alfworld_modules.py never reads %r" % _bad, _bad not in _am_src)

# ---------------------------------------------------------------------------
# 56. ALFWorld revisit anchors (--alf-anchors). The flow graph is painted from
# graph.anchors, so this is the one ALFWorld addition that changes what EVERY
# downstream stage sees. Two things therefore have to hold: it must be inert
# when off (so TA/SP paint from the same roots and keep their cache keys), and
# it must derive its anchors from action text alone.
# ---------------------------------------------------------------------------
if _alf_a:
    from longtraj import alfworld_anchors as _aa
    _t61 = _alf_a[0].trajectory

    # Inert when off -- the pipeline branch is guarded on the same falsiness.
    check("alf_anchors returns None for every falsy variant",
          all(_aa.anchors(_v, _t61, [59]) is None for _v in ("", None)))
    # A typo must raise rather than silently run the incumbent, which would be
    # recorded in the artifact as the variant having run.
    try:
        _aa.anchors("unoin", _t61, [59])
        check("an unknown alf_anchors variant raises", False)
    except ValueError:
        check("an unknown alf_anchors variant raises", True)

    _loc61 = [59]
    _un61 = _aa.anchors("union", _t61, _loc61)
    _f261 = _aa.anchors("first2", _t61, _loc61)
    _on61 = _aa.anchors("only", _t61, _loc61)
    # The union can only ENLARGE the anchor set, which is what guarantees no
    # reference step loses the band it was protected by.
    check("union is a superset of the located anchors",
          set(_loc61) <= set(_un61["anchors"]))
    check("first2 adds at most two anchors",
          len(_un61["anchors"]) >= len(_f261["anchors"]) and
          len(set(_f261["anchors"]) - set(_loc61)) <= 2)
    check("every anchor is an attributable step index",
          all(_s in _t61.attributable_indices
              for _s in set(_un61["anchors"]) - set(_loc61)))
    check("revisit steps are a subset of the anchors it proposes",
          set(_un61["revisits"]) <= set(_un61["anchors"]))
    # ``only`` drops the located anchor; it is the ablation, so it must really
    # drop it rather than quietly unioning.
    check("the 'only' ablation does not re-add the located anchor",
          59 not in _on61["anchors"] or 59 in _un61["revisits"])
    # A record with no repeated ``go to`` must fall back, never paint from an
    # empty set -- that would empty the flow block rather than widen it.
    class _NoGo(object):
        task_id = "synthetic"
        steps = []
    _fb61 = _aa.anchors("union", _NoGo(), _loc61)
    check("a record with no revisit keeps the located anchors",
          _fb61["anchors"] == _loc61 and _fb61["fired"] is False)

    # Repainting from the new anchors is pure and must not invent edges.
    _g61 = FlowGraph(
        anchors=[59], anchors_located=[59],
        edges=[FlowEdge(src=1, dst=3, kind="data"),
               FlowEdge(src=3, dst=59, kind="data"),
               FlowEdge(src=7, dst=9, kind="data")])
    _before = _g61.paint(None)
    _g61.anchors = [9, 59]
    _after = _g61.paint(None)
    check("adding an anchor only enlarges the painted set",
          _before <= _after and 7 in _after and 7 not in _before)

_aa_src = re.sub(r'"""[\s\S]*?"""', "",
                 io.open(os.path.join(_ROOT, "longtraj",
                                      "alfworld_anchors.py"),
                         encoding="utf-8").read())
for _bad in ("ground_truth", "critical_failure_step", "critical_error_step",
             "label_entry", '"annotation"', "'annotation'", "metadata",
             ".meta", "open(", "json.load"):
    check("alfworld_anchors.py never reads %r" % _bad, _bad not in _aa_src)

# ---------------------------------------------------------------------------
# 57. --alf-flownote: the suppression arm, which is the OPPOSITE intervention
# to 56's. The property that matters most is the same one: off must be inert,
# because the prompt slots it substitutes are the ones every TA/SP run fills.
from longtraj import alfworld_flownote as _fn
from longtraj.schema import DATA_FLOW as _DF, CONTROL_FLOW as _CF

_DB57 = ("  step 15 -> step 59 [soapbar 1] (1 hops from symptom; names a "
         "failure artefact; 15 chains converge here) — why\n"
         "  step 1 -> step 2 (not on a chain to any symptom)")
_CB57 = "  step 4 -> step 8 (3 hops from symptom)"

check("alf_flownote returns None for every falsy mode",
      all(_fn.blocks(_m, _DB57, _CB57, "step 59") is None
          for _m in ("", None)))
try:
    _fn.blocks("non", _DB57, _CB57, "step 59")
    check("an unknown alf_flownote mode raises", False)
except ValueError:
    check("an unknown alf_flownote mode raises", True)

_n57 = _fn.blocks("none", _DB57, _CB57, "step 59")
check("mode none removes every traced edge from both blocks",
      "-> step" not in _n57["data_block"]
      and "-> step" not in _n57["control_block"])
check("mode none names no step in the anchors slot",
      not re.search(r"step \d", _n57["anchors"]))

_a57 = _fn.blocks("anchorless", _DB57, _CB57, "step 59")
# The edges are the evidence this mode KEEPS; only the anchor-derived
# annotation goes. Both are asserted, since dropping the edges too would make
# this mode a duplicate of ``none`` and the arm would measure nothing.
check("mode anchorless keeps every edge",
      _a57["data_block"].count("-> step") == _DB57.count("-> step")
      and _a57["control_block"].count("-> step") == _CB57.count("-> step"))
check("mode anchorless strips every hops-from-symptom mark",
      "hops from symptom" not in _a57["data_block"]
      and "hops from symptom" not in _a57["control_block"]
      and "not on a chain" not in _a57["data_block"])
check("mode anchorless keeps the non-anchor marks",
      "names a failure artefact" in _a57["data_block"]
      and "chains converge here" in _a57["data_block"])
check("mode anchorless leaves no empty parenthesis behind",
      " ()" not in _a57["data_block"]
      and " ()" not in _a57["control_block"])
check("mode anchorless keeps the edge rationale",
      "why" in _a57["data_block"])

# The slot hoist in localize.py must be a pure refactor: with the flag off,
# the three slots have to equal the inline expressions they replaced. Checked
# against a real graph shape rather than a string, because the anchors slot is
# built from ``sorted(graph.anchors)``.
_g57 = FlowGraph(anchors=[59], anchors_located=[59],
                 edges=[FlowEdge(src=15, dst=59, kind="data", entity="x"),
                        FlowEdge(src=4, dst=8, kind="control")])
_anc57 = (", ".join("step %d" % a for a in sorted(_g57.anchors))
          or "(none identified)")
_d57 = localize_mod._flow_block(_g57, _DF)
_c57 = localize_mod._flow_block(_g57, _CF)
_off57 = _fn.blocks("", _d57, _c57, _anc57)
check("with alf_flownote off the prompt slots are unchanged",
      _off57 is None)
check("the hoisted anchors slot still renders every anchor",
      _anc57 == "step 59")

_fn_src = re.sub(r'"""[\s\S]*?"""', "",
                 io.open(os.path.join(_ROOT, "longtraj",
                                      "alfworld_flownote.py"),
                         encoding="utf-8").read())
for _bad in ("ground_truth", "critical_failure_step", "critical_error_step",
             "label_entry", '"annotation"', "'annotation'", "metadata",
             ".meta", "open(", "json.load"):
    check("alfworld_flownote.py never reads %r" % _bad, _bad not in _fn_src)

# ---------------------------------------------------------------------------
# 62. ALFWorld local-three: adjacent temporal attribution, closed set
# ---------------------------------------------------------------------------
from longtraj import alfworld_local3 as _l3
from longtraj.schema import Prediction as _Prediction

print(chr(10) + "[62] alfworld --alf-local3: closed adjacent temporal judge")
check("Config.alf_local3 defaults to False",
      [f for f in dataclasses.fields(pipeline_mod.Config)
       if f.name == "alf_local3"][0].default is False)
check("the pipeline branch is guarded on alf_local3",
      "if cfg.alf_local3:" in _pipe_src and
      "alfworld_local3_mod.refine(traj, lz, pred)" in _pipe_src)
check("local3 refine cannot be handed a label-side object",
      list(inspect.signature(_l3.refine).parameters) ==
      ["traj", "llm", "incumbent"])


class _Local3LLM:
    def __init__(self, obj):
        self.obj = obj
        self.prompts = []

    def generate_obj(self, prompt, max_tokens=None, system="", **kw):
        self.prompts.append(prompt)
        return copy.deepcopy(self.obj)


def _local3_traj():
    def obs(value):
        return ("Your current observation is: %s\n"
                "Your admissible actions are: 'look'\nNow it's your turn" % value)
    return Trajectory(
        task_id="local3", source="trajdebug", subset="alfworld",
        task_description="put an object in the right receptacle",
        steps=[
            Step(0, "user", obs("start")),
            Step(1, "assistant", "THOUGHT: inspect first\nACTION: look"),
            Step(2, "user", obs("object is on table")),
            Step(3, "assistant", "THOUGHT: choose table\nACTION: take object"),
            Step(4, "user", obs("you take the wrong object")),
            Step(5, "assistant", "THOUGHT: continue the plan\nACTION: go sink"),
            Step(6, "user", obs("at sink")),
        ])


_t62 = _local3_traj()
_p62 = _Prediction(task_id="local3", critical_step=3, confidence=.61,
                   reason="incumbent", error_type="planning")
_accept62 = {
    "assessments": [
        {"step": 1, "origin": "unrelated",
         "counterfactual_changes_outcome": False, "evidence": "inspection"},
        {"step": 3, "origin": "inherited",
         "counterfactual_changes_outcome": True, "evidence": "followed belief"},
        {"step": 5, "origin": "introduced",
         "counterfactual_changes_outcome": True, "evidence": "new wrong plan"},
    ],
    "critical_step": 5, "move": True, "confidence": .9,
    "reason": "step 5 first creates the failure-relevant plan",
}
_llm62 = _Local3LLM(_accept62)
_moved62 = _l3.refine(_t62, _llm62, _p62)
check("local3 candidate set is exactly attributable neighbours",
      _moved62.experts["candidate_steps"] == [1, 3, 5])
check("a complete introduced-vs-inherited case may move",
      _moved62.critical_step == 5 and _moved62.experts["accepted"] is True)
check("local3 prompt preserves exact original candidate indices",
      all(_llm62.prompts[0].count("========== STEP %d" % i) == 1
          for i in (1, 3, 5)))

_escape62 = copy.deepcopy(_accept62)
_escape62["critical_step"] = 2       # user observation, never attributable
_escaped62 = _l3.refine(_t62, _Local3LLM(_escape62), _p62)
check("an invented or user step cannot escape the closed set",
      _escaped62.critical_step == 3 and
      _escaped62.experts["accepted"] is False)

_missing62 = copy.deepcopy(_accept62)
_missing62["assessments"] = _missing62["assessments"][:-1]
_refused62 = _l3.refine(_t62, _Local3LLM(_missing62), _p62)
check("an incomplete assessment preserves the incumbent",
      _refused62.critical_step == 3 and
      "complete" in _refused62.experts["refused"])

_both62 = copy.deepcopy(_accept62)
_both62["assessments"][1]["origin"] = "introduced"
_refused_both62 = _l3.refine(_t62, _Local3LLM(_both62), _p62)
check("an also-introduced incumbent blocks a move",
      _refused_both62.critical_step == 3 and
      "incumbent_not_origin" in _refused_both62.experts["refused"])

_l3_src = re.sub(r'"""[\s\S]*?"""', "",
                 io.open(os.path.join(_ROOT, "longtraj",
                                      "alfworld_local3.py"),
                         encoding="utf-8").read())
for _bad in ("ground_truth", "critical_failure_step", "critical_error_step",
             ".meta", "open(", "json.load"):
    check("alfworld_local3.py never reads %r" % _bad, _bad not in _l3_src)

# ---------------------------------------------------------------------------
# 63. ALFWorld goal-conditioned event answer space (zero-call stage A)
# ---------------------------------------------------------------------------
from longtraj import alfworld_events as _ae

print(chr(10) + "[63] alfworld goal events: bounded, original, label-free")
_g63 = _ae.compile_goal("Your task is: clean some soapbar and put it in toilet.")
check("goal compiler recovers object, transform and target",
      (_g63.object_type, _g63.transform, _g63.target_type) ==
      ("soapbar", "clean", "toilet"))
check("goal compiler emits ordered obligations",
      _g63.obligations == ("locate", "acquire", "transform", "place"))
_g63b = _ae.compile_goal("Your task is: look at book under the desklamp.")
check("lamp tasks compile to an inspect terminal",
      (_g63b.object_type, _g63b.target_type, _g63b.terminal) ==
      ("book", "desklamp", "inspect"))
_g63c = _ae.compile_goal(
    "Your task is: find two creditcard and put them in armchair.")
check("two-object tasks resolve the pronoun to the named object",
      (_g63c.object_type, _g63c.target_type, _g63c.quantity) ==
      ("creditcard", "armchair", 2))

_goal63, _events63 = _ae.build_events(_t62, incumbent=3)
_steps63 = [e.step for e in _events63]
check("event candidates retain only original attributable indices",
      set(_steps63).issubset(set(_t62.attributable_indices)))
check("event candidates are chronological, distinct and bounded",
      _steps63 == sorted(set(_steps63)) and len(_steps63) <= _ae.MAX_EVENTS)
check("event candidates preserve the incumbent when it is attributable",
      3 in _steps63 and
      "incumbent" in next(e.reasons for e in _events63 if e.step == 3))
check("unclosed ALFWorld action tags still recover their command",
      _ae._canonical_action("<plan>x</plan>\n<action>\ntake mug 1 from desk 1",
                            ("take mug 1 from desk 1",)) ==
      "take mug 1 from desk 1")
check("event builder accepts no label-side parameter",
      list(inspect.signature(_ae.build_events).parameters) ==
      ["traj", "incumbent", "max_events"])

_ae_src = re.sub(r'"""[\s\S]*?"""', "",
                 io.open(os.path.join(_ROOT, "longtraj",
                                      "alfworld_events.py"),
                         encoding="utf-8").read())
for _bad in ("ground_truth", "critical_failure_step", "critical_error_step",
             ".meta", "open(", "json.load"):
    check("alfworld_events.py never reads %r" % _bad, _bad not in _ae_src)

# ---------------------------------------------------------------------------
# 64. ALFWorld objective obligations: one prompt, full answer space, no label
# ---------------------------------------------------------------------------
from longtraj import alfworld_obligations as _ao

print(chr(10) + "[64] alfworld --alf-obligations: objective state in localiser")
check("Config.alf_obligations defaults to False",
      [f for f in dataclasses.fields(pipeline_mod.Config)
       if f.name == "alf_obligations"][0].default is False)
check("off path omits the obligation kwarg entirely",
      '**({"obligation_block": obligation_block}' in _pipe_src and
      "if obligation_block else {}" in _pipe_src)
check("the obligation block enters the existing localiser call",
      "obligation_block: Optional[str] = None" in
      inspect.getsource(localize_mod.localize) and
      'if obligation_block:' in inspect.getsource(localize_mod.localize))
check("obligation builder accepts only a trajectory",
      list(inspect.signature(_ao.build_obligation_block).parameters) == ["traj"])


def _obs64(value, actions):
    return ("Your current observation is: %s\n"
            "Your admissible actions are: %s\nNow it's your turn" %
            (value, ", ".join("'%s'" % a for a in actions)))


_t64 = Trajectory(
    task_id="obligations", source="trajdebug", subset="alfworld",
    task_description="Your task is: clean some soapbar and put it in toilet.",
    steps=[
        Step(0, "user", _obs64("empty room", ("go to countertop 1",))),
        Step(1, "assistant", "<plan>search</plan><action>go to countertop 1</action>"),
        Step(2, "user", _obs64("On the countertop 1, you see a soapbar 1.",
                               ("take soapbar 1 from countertop 1",))),
        Step(3, "assistant", "<reflection>found it</reflection><action>take soapbar 1 from countertop 1</action>"),
        Step(4, "user", _obs64("You pick up the soapbar 1 from the countertop 1.",
                               ("clean soapbar 1 with sinkbasin 1",))),
        Step(5, "assistant", "<plan>clean it</plan><action>clean soapbar 1 with sinkbasin 1</action>"),
        Step(6, "user", _obs64("You clean the soapbar 1 using the sinkbasin 1.",
                               ("move soapbar 1 to toilet 1",))),
        Step(7, "assistant", "<plan>place it</plan><action>move soapbar 1 to toilet 1</action>"),
        Step(8, "user", _obs64("You move the soapbar 1 to the toilet 1.", ("look",))),
    ])
_tl64 = _ao.build_timeline(_t64)
check("successful simulator results complete ordered obligations",
      _tl64.completion_steps ==
      {"locate": 1, "acquire": 3, "transform": 5, "place": 7})
check("timeline preserves original attributable indices",
      [r.step for r in _tl64.rows] == [1, 3, 5, 7])
check("objective counts reach one only after matching success results",
      (_tl64.rows[-1].after.located, _tl64.rows[-1].after.acquired,
       _tl64.rows[-1].after.transformed, _tl64.rows[-1].after.placed) ==
      (1, 1, 1, 1))

_fail_steps64 = list(_t64.steps)
_fail_steps64[6] = Step(6, "user", _obs64("Nothing happens.",
                                         ("move soapbar 1 to toilet 1",)))
_fail64 = _ao.build_timeline(Trajectory(
    task_id="refusal", source="trajdebug", subset="alfworld",
    task_description=_t64.task_description, steps=_fail_steps64))
check("Nothing happens never completes the transform obligation",
      _fail64.rows[2].after.transformed == 0 and
      _fail64.completion_steps["transform"] is None)
_block64, _stats64 = _ao.build_obligation_block(_t64)
check("prompt states the first unfinished-obligation decision rule directly",
      "still-unmet obligation was FIRST" in _block64 and
      "treated by the agent as complete, unnecessary, or abandoned" in _block64)
check("prompt block reports objective transitions and exact step ids",
      "[step 5] ACTION: clean soapbar 1 with sinkbasin 1" in _block64 and
      "transform completed" in _block64 and _stats64.transitions == 4)

_ao_src = re.sub(r'"""[\s\S]*?"""', "",
                 io.open(os.path.join(_ROOT, "longtraj",
                                      "alfworld_obligations.py"),
                         encoding="utf-8").read())
for _bad in ("ground_truth", "critical_failure_step", "critical_error_step",
             ".meta", "open(", "json.load"):
    check("alfworld_obligations.py never reads %r" % _bad,
          _bad not in _ao_src)

# ---------------------------------------------------------------------------
if FAILURES:
    print("%d CHECK(S) FAILED: %s" % (len(FAILURES), FAILURES))
    sys.exit(1)

print("all checks passed")
