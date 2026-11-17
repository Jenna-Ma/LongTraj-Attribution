"""Label-free goal events for ALFWorld failure attribution.

This module changes the unit of comparison from thirty long, homogeneous
assistant documents to a small set of code-derived causal event boundaries.
It deliberately has no pipeline seam yet: the zero-call ceiling must pass
before an LLM is allowed to consume these events.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .alfworld_ledger import (_neighbour_user, _normalise_action, _tag,
                              admissible_of, observation_of)
from .schema import Trajectory


MAX_EVENTS = 8
_REFUSAL = "nothing happens"
_ENTITY = re.compile(r"\b([a-z][a-z0-9_-]*)\s+(\d+)\b", re.I)
_TASK_PREFIX = re.compile(r"^\s*your task is(?::|\s+to:)\s*", re.I)
_TRANSFORM = re.compile(r"\b(clean|heat|cool|hot|cold)\b", re.I)
_TWO = re.compile(r"\b(?:find\s+)?two\s+([a-z][a-z0-9_-]*)\b", re.I)
_PUT = re.compile(
    r"\bput\s+(?:(?:a|an|some|the|two)\s+)?"
    r"(?:(?:clean|hot|cool|cold)\s+)?([a-z][a-z0-9_-]*)\s+"
    r"(?:in|into|on|onto)\s+(?:the\s+)?([a-z][a-z0-9_-]*)\b", re.I)
_AND_PUT = re.compile(
    r"\b(?:clean|heat|cool)\s+(?:a|an|some|the)?\s*"
    r"([a-z][a-z0-9_-]*)\s+and\s+put\s+it\s+"
    r"(?:in|into|on|onto)\s+(?:the\s+)?([a-z][a-z0-9_-]*)\b", re.I)
_LAMP = re.compile(
    r"\b(?:look at|examine)\s+(?:(?:a|an|some|the)\s+)?"
    r"([a-z][a-z0-9_-]*)\s+(?:under|with)\s+(?:the\s+)?desklamp\b", re.I)
_OPEN_ACTION = re.compile(r"<action>\s*([^\r\n<]+)", re.I)


@dataclass(frozen=True)
class GoalSpec:
    object_type: str = ""
    target_type: str = ""
    quantity: int = 1
    transform: str = ""
    terminal: str = "place"
    obligations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_type": self.object_type,
            "target_type": self.target_type,
            "quantity": self.quantity,
            "transform": self.transform,
            "terminal": self.terminal,
            "obligations": list(self.obligations),
        }


@dataclass(frozen=True)
class TurnState:
    step: int
    action: str
    action_key: str
    verb: str
    entities: Tuple[str, ...]
    fixture: str
    before: str
    after: str
    admissible: Optional[bool]
    refused: bool
    repeated_action: bool
    repeated_observation: bool
    revisit_fixture: bool
    phase: str


@dataclass
class Event:
    step: int
    reasons: List[str] = field(default_factory=list)
    priority: int = 0
    turn: Optional[TurnState] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "reasons": list(self.reasons),
            "priority": self.priority,
            "phase": self.turn.phase if self.turn else "",
            "action": self.turn.action if self.turn else "",
            "before": self.turn.before if self.turn else "",
            "after": self.turn.after if self.turn else "",
            "admissible": self.turn.admissible if self.turn else None,
            "refused": self.turn.refused if self.turn else False,
        }


def compile_goal(task_description: str) -> GoalSpec:
    text = _TASK_PREFIX.sub("", str(task_description or "").strip()).rstrip(".")
    low = text.casefold()
    quantity = 2 if _TWO.search(low) else 1
    transform_match = _TRANSFORM.search(low)
    transform = (transform_match.group(1).casefold() if transform_match else "")
    if transform == "hot":
        transform = "heat"
    elif transform in {"cold", "cool"}:
        transform = "cool"

    obj = target = ""
    two = _TWO.search(low)
    terminal = "place"
    lamp = _LAMP.search(low)
    if lamp:
        obj, target, terminal = lamp.group(1), "desklamp", "inspect"
    else:
        match = _AND_PUT.search(low) or _PUT.search(low)
        if match:
            obj, target = match.group(1), match.group(2)
            # "find two books and put them in desk": the placement regex
            # correctly finds the target but its grammatical object is a
            # pronoun. Resolve it from the preceding explicit quantity phrase.
            if obj in {"it", "them"} and two:
                obj = two.group(1)
        else:
            if two:
                obj = two.group(1)

    obligations = ["locate", "acquire"]
    if transform:
        obligations.append("transform")
    obligations.append(terminal)
    return GoalSpec(obj.casefold(), target.casefold(), quantity, transform,
                    terminal, tuple(obligations))


def _canonical_action(content: str, allowed: Sequence[str]) -> str:
    raw = _tag(content, "action", 1200)
    if not raw:
        # 183/3000 corpus turns have an opening <action> tag but no closing
        # tag. The command is still an unambiguous single line at EOF.
        match = _OPEN_ACTION.search(str(content or ""))
        raw = match.group(1).strip() if match else ""
    haystack = _normalise_action(raw)
    if not haystack:
        # Some agents omit the action tag and finish with a quoted command.
        # Only search the tail: memory and plan blocks replay old commands.
        haystack = _normalise_action(str(content or "")[-700:])
    allowed_rows = [(_normalise_action(a), " ".join(str(a).split()))
                    for a in allowed]
    for key, original in allowed_rows:
        if key and haystack == key:
            return original
    embedded = [(len(key), original) for key, original in allowed_rows
                if key and key in haystack]
    if embedded:
        return max(embedded)[1]
    return " ".join(raw.split()) if raw else ""


def _verb(action: str) -> str:
    words = _normalise_action(action).split()
    if not words:
        return ""
    if words[0] == "pick" and len(words) > 1 and words[1] == "up":
        return "take"
    return {"put": "move", "toggle": "use"}.get(words[0], words[0])


def _entities(action: str) -> Tuple[str, ...]:
    return tuple(m.group(1).casefold() for m in _ENTITY.finditer(action))


def _fixture(verb: str, entities: Tuple[str, ...]) -> str:
    if not entities:
        return ""
    if verb in {"go", "open", "close", "examine", "look", "use"}:
        return entities[-1]
    if verb in {"take", "move", "clean", "heat", "cool"} and len(entities) > 1:
        return entities[-1]
    return ""


def _phase(verb: str, goal: GoalSpec) -> str:
    if verb == "take":
        return "acquire"
    if verb in {"clean", "heat", "cool"}:
        return "transform"
    if verb == "move":
        return "place"
    if goal.terminal == "inspect" and verb in {"examine", "use"}:
        return "inspect"
    if verb in {"go", "open", "close", "look", "examine", "inventory"}:
        return "locate"
    return "other"


def turns(traj: Trajectory, goal: Optional[GoalSpec] = None) -> List[TurnState]:
    goal = goal or compile_goal(traj.task_description)
    out: List[TurnState] = []
    seen_actions = set()
    seen_observations = set()
    seen_fixtures = set()
    for pos, step in enumerate(traj.steps):
        if not step.attributable:
            continue
        before_text = _neighbour_user(traj.steps, pos, -1)
        after_text = _neighbour_user(traj.steps, pos, 1)
        before = observation_of(before_text)
        after = observation_of(after_text)
        allowed = admissible_of(before_text)
        action = _canonical_action(step.content, allowed)
        action_key = _normalise_action(action)
        allowed_keys = {_normalise_action(a) for a in allowed}
        admissible = (action_key in allowed_keys if action_key and allowed
                      else None)
        verb = _verb(action)
        entities = _entities(action)
        fixture = _fixture(verb, entities)
        before_key = " ".join(before.casefold().split())
        row = TurnState(
            step=step.index, action=action, action_key=action_key, verb=verb,
            entities=entities, fixture=fixture, before=before, after=after,
            admissible=admissible,
            refused=_REFUSAL in after.casefold(),
            repeated_action=bool(action_key and action_key in seen_actions),
            repeated_observation=bool(before_key and
                                      before_key in seen_observations),
            revisit_fixture=bool(fixture and fixture in seen_fixtures),
            phase=_phase(verb, goal),
        )
        out.append(row)
        if action_key:
            seen_actions.add(action_key)
        if before_key:
            seen_observations.add(before_key)
        if fixture:
            seen_fixtures.add(fixture)
    return out


def _add(events: Dict[int, Event], row: TurnState, reason: str,
         priority: int) -> None:
    event = events.setdefault(row.step, Event(row.step, turn=row))
    if reason not in event.reasons:
        event.reasons.append(reason)
    event.priority = max(event.priority, priority)


def _farthest(rows: Sequence[Event], chosen: Sequence[Event]) -> Event:
    if not chosen:
        return min(rows, key=lambda e: e.step)
    return max(rows, key=lambda e: (
        min(abs(e.step - c.step) for c in chosen), e.priority,
        -e.step))


def build_events(traj: Trajectory, incumbent: Optional[int] = None,
                 max_events: int = MAX_EVENTS) -> Tuple[GoalSpec, List[Event]]:
    """Return at most ``max_events`` chronological, original-step events."""
    goal = compile_goal(traj.task_description)
    rows = turns(traj, goal)
    if not rows or max_events <= 0:
        return goal, []
    by_step = {row.step: row for row in rows}
    events: Dict[int, Event] = {}

    _add(events, rows[0], "trajectory_start", 100)
    if incumbent in by_step:
        _add(events, by_step[incumbent], "incumbent", 110)

    seen_phases = set()
    seen_fixture_classes = set()
    previous: Optional[TurnState] = None
    in_refusal_run = False
    for row in rows:
        if row.phase not in {"other", "locate"} and row.phase not in seen_phases:
            _add(events, row, "obligation_entry:" + row.phase, 96)
            seen_phases.add(row.phase)
        if row.fixture and row.fixture not in seen_fixture_classes:
            _add(events, row, "search_fixture_entry:" + row.fixture, 70)
            seen_fixture_classes.add(row.fixture)
        if row.revisit_fixture:
            _add(events, row, "fixture_revisit", 76)
        if row.repeated_action:
            _add(events, row, "repeated_action", 82)
            if previous and previous.action_key == row.action_key:
                _add(events, previous, "repeat_run_start", 90)
        if row.admissible is False:
            _add(events, row, "inadmissible_action", 92)
        if row.refused:
            if not in_refusal_run:
                _add(events, row, "refusal_run_start", 91)
            in_refusal_run = True
        else:
            in_refusal_run = False
        if previous and previous.refused:
            _add(events, row, "reacts_to_refusal", 93)
        if not row.action:
            _add(events, row, "blank_action", 94)
        previous = row

    candidates = list(events.values())
    if len(candidates) <= max_events:
        return goal, sorted(candidates, key=lambda e: e.step)

    # Mandatory anchors are semantic boundaries, not scores learned from a
    # label. If they exceed the budget, their frozen priority resolves it.
    chosen: List[Event] = []
    mandatory = sorted(
        (e for e in candidates if e.priority >= 90),
        key=lambda e: (-e.priority, e.step))
    for event in mandatory:
        if len(chosen) >= max_events:
            break
        chosen.append(event)

    remaining = [e for e in candidates if e not in chosen]
    while remaining and len(chosen) < max_events:
        event = _farthest(remaining, chosen)
        chosen.append(event)
        remaining.remove(event)
    return goal, sorted(chosen, key=lambda e: e.step)
