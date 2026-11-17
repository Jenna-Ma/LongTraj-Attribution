"""Trajectory loaders for the three corpora.

Every loader returns ``LoadedRecord(trajectory, ground_truth)``. The
trajectory object has no field that can hold a label, so the pipeline
physically cannot read the answer even by accident.

Sources
-------
longrca   : longrca-bench/data/*.parquet
            columns: question_ID, history[{step,role,content,name}],
                     mistake_agent, mistake_step, mistake_reason
trajaudit : TrajAudit-main/RootSE/<agent>/<model>/*.json
            original_traj[{thought,action,observation,response,...}],
            failure_id = reference step
trajdebug : TrajDebug-main/data/unified/<dataset>/*.json
            messages[{step,role,content,name}],
            metadata.annotation.critical_error_step
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterator, List, Optional

from .schema import GroundTruth, LoadedRecord, Step, Trajectory

# ---------------------------------------------------------------------------
# Repo layout discovery
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def find_repo_root(start: Optional[str] = None) -> str:
    """Walk up until we see the sibling corpora directories."""
    cur = os.path.abspath(start or _HERE)
    for _ in range(8):
        probe = [
            os.path.join(cur, "longrca-bench", "data"),
            os.path.join(cur, "TrajAudit-main"),
            os.path.join(cur, "TrajDebug-main"),
        ]
        if any(os.path.isdir(p) for p in probe):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return os.path.abspath(start or _HERE)


def _first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# longrca-bench (parquet)
# ---------------------------------------------------------------------------
def load_longrca(root: Optional[str] = None,
                 subset: Optional[str] = None,
                 limit: Optional[int] = None) -> Iterator[LoadedRecord]:
    """Yield LongRCA-Bench trajectories.

    ``subset`` filters on the source prefix of ``question_ID``
    (swe_bench_pro / terminal_bench_2 / travelplanner / vitabench /
    webarena_verified).
    """
    import pandas as pd

    root = root or find_repo_root()
    data_dir = _first_existing(
        os.path.join(root, "longrca-bench", "data"),
        os.path.join(root, "longrca-bench", "longrca-bench", "data"),
    )
    if not data_dir:
        return
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    n = 0
    for fp in files:
        df = pd.read_parquet(fp)
        for _, row in df.iterrows():
            qid = str(row["question_ID"])
            src = qid.split("__")[0]
            if subset and src != subset:
                continue
            history = row["history"]
            steps: List[Step] = []
            for i, h in enumerate(history):
                if not isinstance(h, dict):
                    continue
                steps.append(Step(
                    index=i,
                    role=str(h.get("role") or "assistant"),
                    content=str(h.get("content") or ""),
                    name=(str(h["name"]) if h.get("name") else None),
                ))
            if not steps:
                continue
            # step 0 carries the task statement in this corpus
            task_desc = steps[0].content[:4000] if steps else ""
            traj = Trajectory(
                task_id=qid, source="longrca", subset=src,
                task_description=task_desc, steps=steps,
                meta={"n_steps": len(steps)},
            )
            ms = row.get("mistake_step")
            gt = GroundTruth(
                task_id=qid,
                critical_step=(int(ms) if ms is not None and str(ms) != "nan" else None),
                responsible_role=(str(row["mistake_agent"])
                                  if row.get("mistake_agent") is not None else None),
                rationale=(str(row["mistake_reason"])
                           if row.get("mistake_reason") is not None else None),
            )
            yield LoadedRecord(traj, gt)
            n += 1
            if limit and n >= limit:
                return


# ---------------------------------------------------------------------------
# TrajAudit / RootSE (json, thought+action+observation triples)
# ---------------------------------------------------------------------------
def _rootse_step_text(raw: dict) -> str:
    """Flatten one RootSE step into readable text without dropping signal."""
    parts = []
    for key, label in (("thought", "Thought"), ("action", "Action"),
                       ("observation", "Observation")):
        v = raw.get(key)
        if v is None:
            continue
        v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
        if v.strip():
            parts.append("%s: %s" % (label, v))
    if not parts:
        resp = raw.get("response")
        if resp:
            parts.append("Response: %s" % resp)
    return "\n".join(parts)


def load_trajaudit(root: Optional[str] = None,
                   subset: Optional[str] = None,
                   limit: Optional[int] = None) -> Iterator[LoadedRecord]:
    """Yield RootSE trajectories.

    ``subset`` matches ``<agent>`` or ``<agent>/<model>``, e.g.
    "swe-agent" or "swe-agent/claude45".
    """
    root = root or find_repo_root()
    base = _first_existing(
        os.path.join(root, "TrajAudit-main", "TrajAudit-main", "RootSE"),
        os.path.join(root, "TrajAudit-main", "RootSE"),
    )
    if not base:
        return
    n = 0
    for agent_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(agent_dir):
            continue
        agent = os.path.basename(agent_dir)
        for model_dir in sorted(glob.glob(os.path.join(agent_dir, "*"))):
            if not os.path.isdir(model_dir):
                continue
            model = os.path.basename(model_dir)
            tag = "%s/%s" % (agent, model)
            if subset and subset not in (agent, tag):
                continue
            for fp in sorted(glob.glob(os.path.join(model_dir, "*.json"))):
                try:
                    d = json.load(open(fp, encoding="utf-8"))
                except Exception:
                    continue
                raw_steps = d.get("original_traj") or []
                steps: List[Step] = []
                for i, r in enumerate(raw_steps):
                    if isinstance(r, dict):
                        text = _rootse_step_text(r)
                    else:
                        text = str(r)
                    steps.append(Step(index=i, role="assistant", content=text))
                if not steps:
                    continue
                # The failing test is the observable symptom; keep it in meta so
                # backward tracing can anchor on it. It is NOT the label.
                failed = d.get("failed_tests") or {}
                sym = {}
                if isinstance(failed, dict) and failed:
                    first = next(iter(failed.values()))
                    if isinstance(first, dict):
                        sym = {
                            "failed_test_code": str(first.get("code") or "")[:4000],
                            "failed_test_error": str(first.get("error_description") or "")[:4000],
                            "failed_test_expected": str(first.get("expected") or "")[:2000],
                        }
                task_id = os.path.splitext(os.path.basename(fp))[0]
                traj = Trajectory(
                    task_id=task_id, source="trajaudit", subset=tag,
                    task_description=str(d.get("original_mission_prompt") or "")[:6000],
                    steps=steps,
                    meta={"repo": d.get("repo"), "language": d.get("repo_language"),
                          "n_steps": len(steps), **sym},
                )
                fid = d.get("failure_id")
                try:
                    fid = int(fid)
                except (TypeError, ValueError):
                    fid = None
                yield LoadedRecord(traj, GroundTruth(task_id=task_id, critical_step=fid))
                n += 1
                if limit and n >= limit:
                    return


# ---------------------------------------------------------------------------
# TrajDebug unified schema
# ---------------------------------------------------------------------------
def load_trajdebug(root: Optional[str] = None,
                   subset: Optional[str] = None,
                   limit: Optional[int] = None) -> Iterator[LoadedRecord]:
    """Yield TrajDebug unified trajectories. ``subset`` is the dataset dir name."""
    root = root or find_repo_root()
    base = _first_existing(
        os.path.join(root, "TrajDebug-main", "TrajDebug-main", "data", "unified"),
        os.path.join(root, "TrajDebug-main", "data", "unified"),
    )
    if not base:
        return
    n = 0
    for ds_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(ds_dir):
            continue
        ds = os.path.basename(ds_dir)
        if subset and ds != subset:
            continue
        for fp in sorted(glob.glob(os.path.join(ds_dir, "*.json"))):
            try:
                d = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            msgs = d.get("messages") or []
            steps = [
                Step(index=int(m.get("step", i)),
                     role=str(m.get("role") or "assistant"),
                     content=str(m.get("content") or ""),
                     name=(str(m["name"]) if m.get("name") else None))
                for i, m in enumerate(msgs) if isinstance(m, dict)
            ]
            if not steps:
                continue
            md = d.get("metadata") or {}
            if md.get("reward") == 1:
                continue          # successes have no critical step
            ann = md.get("annotation") or {}
            task_id = str(md.get("task_id") or os.path.basename(fp))
            traj = Trajectory(
                task_id=task_id, source="trajdebug", subset=ds,
                task_description=str(md.get("task_description") or "")[:6000],
                steps=steps,
                meta={"n_steps": len(steps),
                      "framework": (md.get("extra") or {}).get(
                          "agent_framework_description", "")[:2000]},
            )
            gt = GroundTruth(
                task_id=task_id,
                critical_step=ann.get("critical_error_step"),
                error_type=ann.get("critical_error_type"),
                rationale=ann.get("human_rationale"),
            )
            yield LoadedRecord(traj, gt)
            n += 1
            if limit and n >= limit:
                return


# ---------------------------------------------------------------------------
# longrca-bench (json export)
# ---------------------------------------------------------------------------
# The JSON export writes every tool observation as ``role: "user"`` with
# ``name: "Computer_terminal"`` -- 17900 of the 18031 user turns in the
# SWE-bench Pro split. ``NON_ATTRIBUTABLE_ROLES`` (schema.py) bars user turns
# from being nominated, which is correct for the 131 turns where a human states
# the task and wrong for the tool output: taken literally it bars 48.7% of the
# corpus from the answer space, and three reference steps along with it. So key
# the role on the SPEAKER, not on the corpus's role string, and let tool output
# land on the same ``tool`` role the other two corpora already use.
_LONGRCA_HUMAN = "human"
_LONGRCA_TOOL = "Computer_terminal"


def _longrca_role(role: object, name: object) -> str:
    """Role for one history entry, resolved from the speaker."""
    nm = str(name or "").strip()
    if nm == _LONGRCA_TOOL:
        return "tool"
    if nm == _LONGRCA_HUMAN:
        return "user"
    return str(role or "assistant")


def _load_longrca_json_export(data_dir: str, subset: Optional[str],
                              limit: Optional[int],
                              source: str) -> Iterator[LoadedRecord]:
    """Shared, label-isolated reader for one LongRCA JSON export tree."""
    n = 0
    for set_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(set_dir):
            continue
        for fp in sorted(glob.glob(os.path.join(set_dir, "*.json"))):
            try:
                d = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            qid = str(d.get("question_ID") or
                      os.path.splitext(os.path.basename(fp))[0])
            src = qid.split("__")[0]
            if subset and src != subset:
                continue
            history = d.get("history") or []
            steps: List[Step] = []
            # The corpus numbers its own turns; keep a map so a label that cites
            # a step number still resolves if numbering ever stops being dense.
            pos: dict = {}
            for i, h in enumerate(history):
                if not isinstance(h, dict):
                    continue
                try:
                    pos[int(h.get("step", i))] = len(steps)
                except (TypeError, ValueError):
                    pass
                steps.append(Step(
                    index=len(steps),
                    role=_longrca_role(h.get("role"), h.get("name")),
                    content=str(h.get("content") or ""),
                    name=(str(h["name"]) if h.get("name") else None),
                ))
            if not steps:
                continue
            # step 0 carries the task statement in this corpus
            task_desc = steps[0].content[:4000]
            traj = Trajectory(
                task_id=qid, source=source, subset=src,
                task_description=task_desc, steps=steps,
                meta={"n_steps": len(steps)},
            )
            ms = d.get("mistake_step")
            try:
                ms = pos.get(int(ms), int(ms))
            except (TypeError, ValueError):
                ms = None
            gt = GroundTruth(
                task_id=qid,
                critical_step=ms,
                responsible_role=(str(d["mistake_agent"])
                                  if d.get("mistake_agent") is not None else None),
                rationale=(str(d["mistake_reason"])
                           if d.get("mistake_reason") is not None else None),
            )
            yield LoadedRecord(traj, gt)
            n += 1
            if limit and n >= limit:
                return


def load_longrca_json(root: Optional[str] = None,
                      subset: Optional[str] = None,
                      limit: Optional[int] = None) -> Iterator[LoadedRecord]:
    """Yield LongRCA-Bench trajectories from the original JSON export.

    Same corpus and same fields as :func:`load_longrca`, which reads the
    parquet release and is left untouched. This one reads
    ``longrca-bench/data_json/<set>/*.json`` and resolves the speaker/role
    mismatch described above.

    ``subset`` filters on the source prefix of ``question_ID``
    (swe_bench_pro / terminal_bench_2 / travelplanner / vitabench /
    webarena_verified), exactly as the parquet loader does -- the directory
    name ("SWE-bench Pro") is display text and is never the filter key.
    """
    root = root or find_repo_root()
    data_dir = _first_existing(
        os.path.join(root, "longrca-bench", "data_json"),
        os.path.join(root, "longrca-bench", "longrca-bench", "data_json"),
    )
    if not data_dir:
        return
    if data_dir:
        yield from _load_longrca_json_export(
            data_dir, subset, limit, source="longrca")


def load_longrca_json_new(root: Optional[str] = None,
                          subset: Optional[str] = None,
                          limit: Optional[int] = None) -> Iterator[LoadedRecord]:
    """Yield the corrected LongRCA JSON release from ``data_json_new``.

    It is a separate source so historical runs against ``data_json`` remain
    exactly reproducible.  The trajectory object still contains no annotation;
    corrected labels are returned only in the separate ``GroundTruth`` object.
    """
    root = root or find_repo_root()
    data_dir = _first_existing(
        os.path.join(root, "longrca-bench", "data_json_new"),
        os.path.join(root, "longrca-bench", "longrca-bench", "data_json_new"),
    )
    if data_dir:
        yield from _load_longrca_json_export(
            data_dir, subset, limit, source="longrca_json_new")


# ---------------------------------------------------------------------------
# alfworld (TrajDebug unified) -- an added wrapper, not a changed loader
# ---------------------------------------------------------------------------
# alfworld uses the same schema as swebenchpro, so ``load_trajdebug`` already
# loads it correctly and is left byte-for-byte alone. What alfworld adds is one
# corpus-specific shape: every user turn REPLAYS the whole history before
# appending the single new observation. Measured over all 100 records that
# replay is 43.5% of the corpus (23.5M -> 13.3M chars), and the user side alone
# is 65% of all text while being non-attributable -- not one of those
# characters can ever be the answer, yet every window pays for them.
#
# The two anchors below bracket the novel part. They are not heuristics that
# usually work: they match 2900/2900 user turns with index > 0. A turn that
# fails to match is left EXACTLY as it was rather than guessed at, because
# dropping text on an unrecognised turn would silently shrink the evidence,
# which is a much worse failure than paying for some redundant characters.
_ALF_OBS_HEAD = re.compile(
    r"You are now at step \d+ and your current observation is:", re.I)
_ALF_OBS_TAIL = re.compile(
    r"\n*After that, you should plan the next step"
    r"|\n*Finally, choose ONE admissible action", re.I)


def alfworld_novel_observation(text: str) -> str:
    """Return the part of an alfworld user turn that is NOT replayed history.

    Returns the input unchanged whenever the leading anchor is missing.
    """
    s = str(text or "")
    head = _ALF_OBS_HEAD.search(s)
    if head is None:
        return s
    body = s[head.end():]
    tail = _ALF_OBS_TAIL.search(body)
    if tail is not None:
        body = body[:tail.start()]
    body = body.strip()
    return body or s


def load_trajdebug_alfworld(root: Optional[str] = None,
                            subset: Optional[str] = None,
                            limit: Optional[int] = None,
                            dedup: bool = False) -> Iterator[LoadedRecord]:
    """``load_trajdebug`` plus an optional alfworld-only de-duplication.

    With ``dedup=False`` this yields records that are field-for-field what
    ``load_trajdebug`` yields -- asserted in tests/test_no_leak.py SS55, which
    is what makes an A/B against the flag attributable to the flag alone.

    With ``dedup=True`` each user turn after the first keeps only its novel
    observation. Only user turns are rewritten: the reference step is an
    assistant turn in 100/100 records of this corpus, so the flag cannot touch
    the text the answer lives in. Nothing here reads ``metadata.annotation``;
    the label reaches the caller only through ``GroundTruth``, exactly as in
    the wrapped loader.
    """
    for rec in load_trajdebug(root=root, subset=subset, limit=limit):
        if not dedup:
            yield rec
            continue
        traj = rec.trajectory
        steps = [
            Step(index=s.index, role=s.role, name=s.name,
                 content=(alfworld_novel_observation(s.content)
                          if s.role == "user" and s.index > 0 else s.content))
            for s in traj.steps
        ]
        yield LoadedRecord(
            Trajectory(task_id=traj.task_id, source=traj.source,
                       subset=traj.subset,
                       task_description=traj.task_description,
                       steps=steps, meta=dict(traj.meta)),
            rec.ground_truth)


# ---------------------------------------------------------------------------
LOADERS = {
    "longrca": load_longrca,
    "longrca_json": load_longrca_json,
    "longrca_json_new": load_longrca_json_new,
    "trajaudit": load_trajaudit,
    "trajdebug": load_trajdebug,
    # alfworld rides load_trajdebug; this entry exists so the
    # de-duplication flag has a source to hang off without
    # changing what "trajdebug" means for SP.
    "trajdebug_alf": load_trajdebug_alfworld,
}


def load(source: str, root: Optional[str] = None, subset: Optional[str] = None,
         limit: Optional[int] = None) -> Iterator[LoadedRecord]:
    """Unified entry point. ``source`` in {longrca, trajaudit, trajdebug}."""
    if source not in LOADERS:
        raise ValueError("unknown source %r; expected one of %s"
                         % (source, sorted(LOADERS)))
    return LOADERS[source](root=root, subset=subset, limit=limit)


def load_all(root: Optional[str] = None,
             limit_per_source: Optional[int] = None) -> Iterator[LoadedRecord]:
    for name in ("longrca", "trajaudit", "trajdebug"):
        for rec in LOADERS[name](root=root, subset=None, limit=limit_per_source):
            yield rec
