"""Compact, label-free ALFWorld state-transition ledger.

The ALFWorld user turns replay most of the history, while assistant turns put
the executed action after long memory/reflection/plan blocks. This module
folds each assistant turn and its two neighbouring simulator turns into one
comparable record. It intentionally accepts only a ``Trajectory``: labels and
record metadata are outside its interface and are never consulted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .schema import Trajectory, clip_middle


_TAG = {
    name: re.compile(r"<%s>\s*(.*?)\s*</%s>" % (name, name), re.I | re.S)
    for name in ("memory", "reflection", "plan", "action")
}
_OBS_RE = re.compile(
    r"(?:your\s+)?current observation is:\s*(.*?)\s*"
    r"Your admissible actions(?: of the current situation)? are:",
    re.I | re.S,
)
_ADMISSIBLE_RE = re.compile(
    r"Your admissible actions(?: of the current situation)? are:\s*(.*?)"
    r"(?:\n\s*Now it's your turn|\n\s*Please begin)",
    re.I | re.S,
)
_QUOTED_RE = re.compile(r"['\"]([^'\"\r\n]+)['\"]")

OBS_CLIP = 650
THOUGHT_CLIP = 260
ACTION_CLIP = 300


@dataclass(frozen=True)
class LedgerStats:
    rows: int
    chars: int
    admissible: int
    inadmissible: int
    unknown_admissibility: int
    repeated_actions: int
    repeated_observations: int
    refused: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "ledger_rows": self.rows,
            "ledger_chars": self.chars,
            "ledger_admissible": self.admissible,
            "ledger_inadmissible": self.inadmissible,
            "ledger_unknown_admissibility": self.unknown_admissibility,
            "ledger_repeated_actions": self.repeated_actions,
            "ledger_repeated_observations": self.repeated_observations,
            "ledger_refused": self.refused,
        }


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def _clip(text: str, limit: int) -> str:
    text = _clean(text)
    return clip_middle(text, limit) if len(text) > limit else text


def _tag(text: str, name: str, limit: int = THOUGHT_CLIP) -> str:
    match = _TAG[name].search(str(text or ""))
    return _clip(match.group(1), limit) if match else ""


def _normalise_action(text: str) -> str:
    value = _clean(text).strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    return value.casefold()


def observation_of(text: str) -> str:
    """Return only the current observation, excluding replayed history."""
    match = _OBS_RE.search(str(text or ""))
    return _clip(match.group(1), OBS_CLIP) if match else ""


def admissible_of(text: str) -> Tuple[str, ...]:
    match = _ADMISSIBLE_RE.search(str(text or ""))
    if not match:
        return ()
    return tuple(_clean(v) for v in _QUOTED_RE.findall(match.group(1)) if _clean(v))


def _neighbour_user(steps: Sequence[object], pos: int, direction: int) -> str:
    j = pos + direction
    while 0 <= j < len(steps):
        step = steps[j]
        if getattr(step, "role", "") in {"user", "system"}:
            return str(getattr(step, "content", "") or "")
        if getattr(step, "attributable", False):
            break
        j += direction
    return ""


def render_ledger(traj: Trajectory) -> Tuple[str, LedgerStats]:
    """Render one compact row per attributable turn, without label access."""
    rows: List[str] = []
    seen_actions = set()
    seen_observations = set()
    n_admissible = n_inadmissible = n_unknown = 0
    n_repeat_action = n_repeat_obs = n_refused = 0

    for pos, step in enumerate(traj.steps):
        if not step.attributable:
            continue
        before_text = _neighbour_user(traj.steps, pos, -1)
        after_text = _neighbour_user(traj.steps, pos, 1)
        before = observation_of(before_text)
        after = observation_of(after_text)
        allowed = admissible_of(before_text)
        action = _tag(step.content, "action", ACTION_CLIP)
        action_key = _normalise_action(action)
        allowed_keys = {_normalise_action(a) for a in allowed}

        if not action_key or not allowed:
            admissibility = "unknown"
            n_unknown += 1
        elif action_key in allowed_keys:
            admissibility = "yes"
            n_admissible += 1
        else:
            admissibility = "NO"
            n_inadmissible += 1

        repeated_action = bool(action_key and action_key in seen_actions)
        before_key = _clean(before).casefold()
        repeated_obs = bool(before_key and before_key in seen_observations)
        refused = "nothing happens." in after.casefold()
        n_repeat_action += int(repeated_action)
        n_repeat_obs += int(repeated_obs)
        n_refused += int(refused)

        rows.append(
            "[step %d | %s]\n"
            "BEFORE: %s\n"
            "MEMORY EXCERPT: %s\n"
            "REFLECTION EXCERPT: %s\n"
            "PLAN EXCERPT: %s\n"
            "ACTION: %s\n"
            "AFTER: %s\n"
            "CHECKS: admissible=%s; repeated_action=%s; "
            "repeated_observation=%s; refused=%s"
            % (
                step.index,
                step.name or step.role,
                before or "(unparsed)",
                _tag(step.content, "memory") or "(none)",
                _tag(step.content, "reflection") or "(none)",
                _tag(step.content, "plan") or "(none)",
                action or "(none)",
                after or "(end of run or unparsed)",
                admissibility,
                str(repeated_action).lower(),
                str(repeated_obs).lower(),
                str(refused).lower(),
            )
        )
        if action_key:
            seen_actions.add(action_key)
        if before_key:
            seen_observations.add(before_key)

    header = (
        "ALFWORLD TURN LEDGER. Simulator-only user turns are folded into the "
        "adjacent agent row as BEFORE and AFTER. Original agent step indices "
        "are unchanged. Excerpts are verbatim and code-derived CHECKS are "
        "signals, not verdicts.\n\n"
    )
    body = header + "\n\n".join(rows)
    stats = LedgerStats(
        rows=len(rows), chars=len(body), admissible=n_admissible,
        inadmissible=n_inadmissible, unknown_admissibility=n_unknown,
        repeated_actions=n_repeat_action, repeated_observations=n_repeat_obs,
        refused=n_refused,
    )
    return body, stats

