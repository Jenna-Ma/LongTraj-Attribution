"""Stable project-integration entry point for LongTraj attribution.

Python SDK::

    from longtraj.integration import AttributionEngine

    engine = AttributionEngine(profile="sp_qwen_clean", config_path="config.json")
    result = engine.locate({
        "task_id": "run-001",
        "task_description": "Fix the failing feature",
        "steps": [
            {"step": 0, "role": "user", "content": "..."},
            {"step": 1, "role": "assistant", "name": "Planner",
             "content": "..."},
        ],
    })

JSON CLI::

    python -m longtraj.integration --input trajectory.json --output result.json

The request contract deliberately has no ground-truth field. Label-like keys
are rejected before a ``Trajectory`` is built, preserving the pipeline's
type-level answer isolation at an external project boundary.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .artifacts import TraceStore
from .llm import LLM
from .pipeline import Config, run
from .schema import Prediction, Step, Trajectory


logger = logging.getLogger(__name__)


class AttributionInputError(ValueError):
    """The external trajectory does not satisfy the integration contract."""


_FORBIDDEN_LABEL_KEYS = {
    "annotation",
    "critical_error_step",
    "critical_failure_step",
    "critical_step",
    "failure_id",
    "failure_reason",
    "gold",
    "gold_patch",
    "gold_step",
    "ground_truth",
    "label_entry",
    "mistake_reason",
    "mistake_step",
    "reference_step",
    "test_patch",
}

_ROLE_ALIASES = {
    "assistant": "assistant",
    "agent": "assistant",
    "model": "assistant",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
    "user": "user",
    "human": "user",
    "system": "system",
}


def _assert_no_labels(value: Any, path: str = "request") -> None:
    """Reject label-side keys while treating natural-language text as opaque."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.strip().lower()
            if (lowered in _FORBIDDEN_LABEL_KEYS
                    or lowered.startswith("gold_")
                    or lowered.startswith("ground_truth_")):
                raise AttributionInputError(
                    "label-side field is forbidden at %s.%s" % (path, key_text)
                )
            # Code/log text can legitimately contain strings such as
            # ``critical_step``. Never inspect inside an opaque content field.
            if lowered in {"content", "task", "task_description"}:
                continue
            _assert_no_labels(child, "%s.%s" % (path, key_text))
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            _assert_no_labels(child, "%s[%d]" % (path, i))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _normalise_role(value: Any, name: Any = None) -> str:
    # LongRCA-style exports mark terminal output as role=user. Resolve this
    # common speaker mismatch at the public boundary as the corpus loader does.
    if str(name or "").strip() == "Computer_terminal":
        return "tool"
    role = str(value or "assistant").strip().lower()
    if role not in _ROLE_ALIASES:
        raise AttributionInputError(
            "unsupported role %r; use assistant, tool, user, or system" % value
        )
    return _ROLE_ALIASES[role]


@dataclass(frozen=True)
class PreparedTrajectory:
    """Internal trajectory plus the caller's stable external step IDs."""

    trajectory: Trajectory
    external_step_ids: Tuple[Any, ...]


def prepare_trajectory(request: Mapping[str, Any]) -> PreparedTrajectory:
    """Validate and convert one external JSON-compatible trajectory.

    Pipeline indices are always dense and zero-based. The caller may supply
    arbitrary unique IDs through ``step`` or ``index``; the result then returns
    both the internal position (``critical_step``) and that external ID
    (``critical_step_id``).
    """
    if not isinstance(request, Mapping):
        raise AttributionInputError("trajectory request must be a JSON object")
    _assert_no_labels(request)

    task_id = str(request.get("task_id") or "").strip()
    if not task_id:
        raise AttributionInputError("task_id is required")

    raw_steps = request.get("steps")
    if raw_steps is None:
        raw_steps = request.get("messages")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise AttributionInputError("steps (or messages) must be a non-empty array")
    if not raw_steps:
        raise AttributionInputError("steps must not be empty")

    steps: List[Step] = []
    external_ids: List[Any] = []
    seen_ids = set()
    for position, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            raise AttributionInputError("steps[%d] must be an object" % position)
        external_id = raw.get("step", raw.get("index", position))
        try:
            id_key = json.dumps(external_id, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            id_key = repr(external_id)
        if id_key in seen_ids:
            raise AttributionInputError("duplicate external step id: %r" % external_id)
        seen_ids.add(id_key)

        name = raw.get("name")
        content = _as_text(raw.get("content", raw.get("text", "")))
        steps.append(
            Step(
                index=position,
                role=_normalise_role(raw.get("role"), name=name),
                content=content,
                name=(str(name) if name not in (None, "") else None),
            )
        )
        external_ids.append(external_id)

    if not any(step.attributable for step in steps):
        raise AttributionInputError(
            "trajectory has no attributable assistant/tool step"
        )

    task_description = _as_text(
        request.get("task_description", request.get("task", ""))
    ).strip()
    if not task_description:
        task_description = next(
            (step.content for step in steps if step.role in {"user", "system"}),
            steps[0].content,
        )

    meta = request.get("meta") or {}
    if not isinstance(meta, Mapping):
        raise AttributionInputError("meta must be an object when provided")
    source = str(request.get("source") or "external")
    subset = str(request.get("subset") or "default")
    trajectory = Trajectory(
        task_id=task_id,
        source=source,
        subset=subset,
        task_description=task_description,
        steps=steps,
        meta={**dict(meta), "n_steps": len(steps)},
    )
    return PreparedTrajectory(trajectory, tuple(external_ids))


_PROFILE_ALIASES = {
    "default": "generic",
    "rootse": "rootse_qwen_best",
    "sp": "sp_qwen_highest",
    "sp_clean": "sp_qwen_clean",
}

PROFILES = (
    "generic",
    "rootse_qwen_best",
    "sp_qwen_highest",
    "sp_qwen_clean",
)


def profile_config(profile: str = "generic", workers: int = 4) -> Config:
    """Return a fresh Config for one frozen, documented integration profile."""
    name = _PROFILE_ALIASES.get(str(profile), str(profile))
    if name == "generic":
        return Config(workers=workers)
    if name == "rootse_qwen_best":
        return Config(
            workers=workers,
            span_prior=True,
            verbatim_limit=2500,
            max_prompt_chars=700000,
        )
    if name in {"sp_qwen_highest", "sp_qwen_clean"}:
        return Config(
            workers=workers,
            span_prior=True,
            unit_spans=True,
            prior_compress=True,
            candidate_set=True,
            candidate_strict=True,
            verify=True,
            verify_outdeg=True,
            build_outline=False,
            error_anchors=False,
            full_keep=(name == "sp_qwen_clean"),
            max_prompt_chars=(700000 if name == "sp_qwen_clean" else 120000),
        )
    raise AttributionInputError(
        "unknown profile %r; choose one of %s" % (profile, ", ".join(PROFILES))
    )


def _apply_overrides(config: Config, overrides: Optional[Mapping[str, Any]]) -> Config:
    if not overrides:
        return config
    if not isinstance(overrides, Mapping):
        raise AttributionInputError("config_overrides must be an object")
    fields = {field.name for field in dataclasses.fields(Config)}
    unknown = sorted(str(key) for key in overrides if key not in fields)
    if unknown:
        raise AttributionInputError(
            "unknown Config override(s): %s" % ", ".join(unknown)
        )
    return dataclasses.replace(config, **dict(overrides))


def _diagnostics(prediction: Prediction) -> Dict[str, Any]:
    full = prediction.to_dict()
    compact_keys = {
        "task_id",
        "critical_step",
        "confidence",
        "reason",
        "error_type",
        "usage",
        "compression_ratio",
    }
    return {key: value for key, value in full.items()
            if key not in compact_keys and value is not None}


class AttributionEngine:
    """Reusable, thread-safe facade over the LongTraj main pipeline.

    Construct one engine at service startup and reuse it across requests. The
    LLM client and on-disk prompt cache are then shared rather than recreated
    for every trajectory.
    """

    def __init__(
        self,
        profile: str = "generic",
        *,
        config_path: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trace_dir: Optional[str] = None,
        reuse_flow: bool = False,
        workers: int = 4,
        dry_run: bool = False,
        config_overrides: Optional[Mapping[str, Any]] = None,
        llm: Optional[LLM] = None,
    ):
        if workers < 1:
            raise AttributionInputError("workers must be at least 1")
        if reuse_flow and not trace_dir:
            raise AttributionInputError("reuse_flow=True requires trace_dir")

        canonical = _PROFILE_ALIASES.get(str(profile), str(profile))
        base = profile_config(canonical, workers=workers)
        base = _apply_overrides(base, config_overrides)
        self.profile = canonical
        self.config = dataclasses.replace(
            base,
            workers=workers,
            trace_dir=trace_dir,
            reuse_flow=bool(reuse_flow),
        )
        self.store = TraceStore(trace_dir) if trace_dir else None
        self.llm = llm or LLM(
            model=model,
            cache_dir=cache_dir,
            dry_run=dry_run,
            config=config_path,
            config_required=not dry_run,
        )

    def locate(
        self,
        request: Mapping[str, Any],
        *,
        include_diagnostics: bool = False,
        include_step_content: bool = True,
    ) -> Dict[str, Any]:
        """Locate the critical step in one JSON-compatible trajectory."""
        prepared = prepare_trajectory(request)
        trajectory = prepared.trajectory
        prediction = run(
            trajectory,
            self.llm,
            config=self.config,
            store=self.store,
        )
        index = prediction.critical_step
        valid = isinstance(index, int) and 0 <= index < len(trajectory.steps)
        external_id = prepared.external_step_ids[index] if valid else None
        selected = None
        if valid:
            step = trajectory.steps[index]
            selected = {
                "index": index,
                "step_id": external_id,
                "role": step.role,
                "name": step.name,
            }
            if include_step_content:
                selected["content"] = step.content

        result: Dict[str, Any] = {
            "task_id": trajectory.task_id,
            "critical_step": index,
            "critical_step_id": external_id,
            "confidence": prediction.confidence,
            "reason": prediction.reason,
            "error_type": prediction.error_type,
            "selected_step": selected,
            "profile": self.profile,
            "model": self.llm.model,
            "usage": dict(prediction.usage),
            "compression_ratio": prediction.compression_ratio,
        }
        if include_diagnostics:
            result["diagnostics"] = _diagnostics(prediction)
        return result

    def locate_trajectory(
        self,
        trajectory: Trajectory,
        *,
        include_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Lower-level entry for projects already using LongTraj types."""
        prediction = run(
            trajectory,
            self.llm,
            config=self.config,
            store=self.store,
        )
        result = prediction.to_dict()
        result["profile"] = self.profile
        result["model"] = self.llm.model
        if not include_diagnostics:
            result = {
                key: result.get(key)
                for key in (
                    "task_id", "critical_step", "confidence", "reason",
                    "error_type", "usage", "compression_ratio",
                )
            } | {"profile": self.profile, "model": self.llm.model}
        return result

    def locate_batch(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        batch_workers: int = 1,
        include_diagnostics: bool = False,
        include_step_content: bool = True,
    ) -> List[Dict[str, Any]]:
        """Locate a batch while preserving input order."""
        items = list(requests)
        if any(not isinstance(item, Mapping) for item in items):
            raise AttributionInputError("every batch item must be a trajectory object")
        task_ids = [str(item.get("task_id") or "") for item in items]
        if len(set(task_ids)) != len(task_ids):
            raise AttributionInputError(
                "task_id values must be unique within a concurrent batch"
            )

        def one(item: Mapping[str, Any]) -> Dict[str, Any]:
            return self.locate(
                item,
                include_diagnostics=include_diagnostics,
                include_step_content=include_step_content,
            )

        if batch_workers <= 1:
            return [one(item) for item in items]
        with ThreadPoolExecutor(max_workers=batch_workers) as pool:
            return list(pool.map(one, items))

    def usage(self) -> Dict[str, int]:
        """Cumulative endpoint usage since this engine was constructed."""
        return dict(self.llm.snapshot())


def locate_failure(
    request: Mapping[str, Any],
    *,
    profile: str = "generic",
    config_path: Optional[str] = None,
    model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trace_dir: Optional[str] = None,
    reuse_flow: bool = False,
    workers: int = 4,
    dry_run: bool = False,
    include_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Convenience one-shot function; prefer a reused engine in a service."""
    engine = AttributionEngine(
        profile=profile,
        config_path=config_path,
        model=model,
        cache_dir=cache_dir,
        trace_dir=trace_dir,
        reuse_flow=reuse_flow,
        workers=workers,
        dry_run=dry_run,
    )
    return engine.locate(request, include_diagnostics=include_diagnostics)


def _read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if path == "-":
        sys.stdout.write(text + "\n")
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _unwrap_requests(payload: Any) -> Tuple[List[Mapping[str, Any]], bool]:
    if isinstance(payload, Mapping) and "trajectory" in payload:
        return [payload["trajectory"]], True
    if isinstance(payload, Mapping) and "trajectories" in payload:
        rows = payload["trajectories"]
        if not isinstance(rows, list):
            raise AttributionInputError("trajectories must be an array")
        return rows, False
    if isinstance(payload, Mapping):
        return [payload], True
    if isinstance(payload, list):
        return payload, False
    raise AttributionInputError("input must be one trajectory or an array")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Locate the critical step in an external failed trajectory."
    )
    parser.add_argument("--input", "-i", default="-", help="JSON path, or - for stdin")
    parser.add_argument("--output", "-o", default="-", help="JSON path, or - for stdout")
    parser.add_argument("--profile", default="generic", choices=PROFILES)
    parser.add_argument("--config", default=None, help="API config JSON")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--reuse-flow", action="store_true")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrency inside one trajectory")
    parser.add_argument("--batch-workers", type=int, default=1,
                        help="number of trajectories processed concurrently")
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--no-step-content", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="local fallback plumbing test; not a real prediction")
    parser.add_argument("--log", default="WARNING",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        payload = _read_json(args.input)
        requests, single = _unwrap_requests(payload)
        engine = AttributionEngine(
            profile=args.profile,
            config_path=args.config,
            model=args.model,
            cache_dir=args.cache_dir,
            trace_dir=args.trace_dir,
            reuse_flow=args.reuse_flow,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        results = engine.locate_batch(
            requests,
            batch_workers=args.batch_workers,
            include_diagnostics=args.include_diagnostics,
            include_step_content=not args.no_step_content,
        )
        output: Any = results[0] if single else {
            "results": results,
            "profile": engine.profile,
            "model": engine.llm.model,
            "usage": engine.usage(),
        }
        _write_json(args.output, output)
        return 0
    except (AttributionInputError, FileNotFoundError, ValueError) as exc:
        print("integration input error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

