"""Label-free objective-obligation timeline for ALFWorld localisation.

The simulator, not the model, decides whether an obligation changed state.
Short agent excerpts are shown beside that state so the existing localiser can
find the first turn where the plan diverged from an objectively unfinished
requirement.  This module intentionally accepts only ``Trajectory``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .alfworld_events import GoalSpec, TurnState, compile_goal, turns
from .alfworld_ledger import _tag
from .schema import Trajectory, clip_middle


_ENTITY = re.compile(r"\b([a-z][a-z0-9_-]*)\s+(\d+)\b", re.I)
_SUCCESS = {
    "take": re.compile(r"\byou pick up the ([a-z][a-z0-9_-]*\s+\d+)\b", re.I),
    "move": re.compile(r"\byou move the ([a-z][a-z0-9_-]*\s+\d+)\s+to\s+the\s+"
                       r"([a-z][a-z0-9_-]*\s+\d+)\b", re.I),
    "clean": re.compile(r"\byou clean the ([a-z][a-z0-9_-]*\s+\d+)\s+using\b", re.I),
    "heat": re.compile(r"\byou heat the ([a-z][a-z0-9_-]*\s+\d+)\s+using\b", re.I),
    "cool": re.compile(r"\byou cool the ([a-z][a-z0-9_-]*\s+\d+)\s+using\b", re.I),
}
_ARRIVE = re.compile(r"\byou arrive at ([a-z][a-z0-9_-]*\s+\d+)\b", re.I)
_LAMP_ON = re.compile(r"\byou turn on the (desklamp\s+\d+)\b", re.I)
_OPEN_TAG = {
    name: re.compile(r"<%s>\s*([^<]*(?:<(?!/%s>)[\s\S]*)?)" % (name, name), re.I)
    for name in ("memory", "reflection", "plan")
}
_FIXTURE_FOR = {"clean": "sinkbasin", "heat": "microwave",
                "cool": "fridge", "inspect": "desklamp"}


def _clean(value: str) -> str:
    return " ".join(str(value or "").split())


def _entity(value: str) -> str:
    return _clean(value).casefold()


def _entity_type(value: str) -> str:
    match = _ENTITY.search(_entity(value))
    return match.group(1).casefold() if match else ""


def _instances(text: str, object_type: str) -> Set[str]:
    return {_entity(m.group(0)) for m in _ENTITY.finditer(str(text or ""))
            if m.group(1).casefold() == object_type}


def _excerpt(text: str, name: str, limit: int = 190) -> str:
    closed = _tag(text, name, limit)
    if closed:
        return closed
    match = _OPEN_TAG[name].search(str(text or ""))
    if not match:
        return ""
    value = _clean(match.group(1))
    return clip_middle(value, limit) if len(value) > limit else value


@dataclass(frozen=True)
class ObjectiveState:
    located: int = 0
    acquired: int = 0
    transformed: int = 0
    placed: int = 0
    inspected: int = 0

    def count(self, obligation: str) -> int:
        return int(getattr(self, {"locate": "located", "acquire": "acquired",
                                  "transform": "transformed", "place": "placed",
                                  "inspect": "inspected"}[obligation]))


@dataclass(frozen=True)
class ObligationRow:
    step: int
    action: str
    before: ObjectiveState
    after: ObjectiveState
    next_unmet_before: str
    next_unmet_after: str
    transitions: Tuple[str, ...]
    review_flags: Tuple[str, ...]
    memory: str
    reflection: str
    plan: str
    result: str


@dataclass(frozen=True)
class ObligationTimeline:
    goal: GoalSpec
    rows: Tuple[ObligationRow, ...]
    completion_steps: Dict[str, Optional[int]]
    initial_completions: Tuple[str, ...]


@dataclass(frozen=True)
class ObligationStats:
    rows: int
    chars: int
    transitions: int
    review_rows: int
    refusals: int

    def to_dict(self) -> Dict[str, int]:
        return {"obligation_rows": self.rows,
                "obligation_chars": self.chars,
                "obligation_transitions": self.transitions,
                "obligation_review_rows": self.review_rows,
                "obligation_refusals": self.refusals}


def _required(goal: GoalSpec, obligation: str) -> int:
    return goal.quantity if obligation in {"locate", "acquire", "transform", "place"} else 1


def _next_unmet(goal: GoalSpec, state: ObjectiveState) -> str:
    for obligation in goal.obligations:
        if state.count(obligation) < _required(goal, obligation):
            return obligation
    return "none"


def _state(found: Set[str], acquired: Set[str], transformed: Set[str],
           placed: Set[str], inspected: bool) -> ObjectiveState:
    return ObjectiveState(len(found), len(acquired), len(transformed),
                          len(placed), int(inspected))


def build_timeline(traj: Trajectory) -> ObligationTimeline:
    """Build ordered objective state around every attributable ALFWorld turn."""
    goal = compile_goal(traj.task_description)
    turn_rows = turns(traj, goal)
    by_index = {step.index: step for step in traj.steps}
    found: Set[str] = set()
    acquired: Set[str] = set()
    transformed: Set[str] = set()
    placed: Set[str] = set()
    held: Set[str] = set()
    inspected = False
    current_fixture = ""
    completion_steps: Dict[str, Optional[int]] = {o: None for o in goal.obligations}
    initial: List[str] = []
    out: List[ObligationRow] = []
    refusal_actions: Set[str] = set()

    for row_pos, turn in enumerate(turn_rows):
        # The first BEFORE observation is initial simulator state. Later BEFORE
        # observations repeat the preceding AFTER and therefore add no new fact.
        newly_visible = _instances(turn.before, goal.object_type)
        was = _state(found, acquired, transformed, placed, inspected)
        found.update(newly_visible)
        before = _state(found, acquired, transformed, placed, inspected)
        if row_pos == 0 and before.located >= goal.quantity and was.located < goal.quantity:
            completion_steps["locate"] = None
            initial.append("locate")

        next_before = _next_unmet(goal, before)
        previous_fixture = current_fixture
        result = str(turn.after or "")
        transitions: List[str] = []

        take = _SUCCESS["take"].search(result)
        if take:
            obj = _entity(take.group(1))
            if _entity_type(obj) == goal.object_type:
                found.add(obj)
                acquired.add(obj)
                held.add(obj)

        move = _SUCCESS["move"].search(result)
        if move:
            obj, target = _entity(move.group(1)), _entity(move.group(2))
            held.discard(obj)
            if (_entity_type(obj) == goal.object_type and
                    _entity_type(target) == goal.target_type):
                found.add(obj)
                placed.add(obj)

        for verb in ("clean", "heat", "cool"):
            match = _SUCCESS[verb].search(result)
            if match:
                obj = _entity(match.group(1))
                if _entity_type(obj) == goal.object_type and verb == goal.transform:
                    found.add(obj)
                    transformed.add(obj)

        arrival = _ARRIVE.search(result)
        if arrival:
            current_fixture = _entity(arrival.group(1))
        if (_LAMP_ON.search(result) and goal.terminal == "inspect" and acquired):
            inspected = True

        # Result observations can reveal the goal object even when the action
        # itself was merely go/open/examine.
        found.update(_instances(result, goal.object_type))
        after = _state(found, acquired, transformed, placed, inspected)
        for obligation in goal.obligations:
            needed = _required(goal, obligation)
            if before.count(obligation) < needed <= after.count(obligation):
                transitions.append("%s completed" % obligation)
                if completion_steps[obligation] is None and obligation not in initial:
                    completion_steps[obligation] = turn.step

        next_after = _next_unmet(goal, after)
        flags: List[str] = []
        if turn.refused:
            if turn.action_key not in refusal_actions:
                flags.append("first no-progress result for this action")
            refusal_actions.add(turn.action_key)

        phase = "transform" if turn.phase == "transform" else turn.phase
        readiness = {
            "acquire": before.located - before.acquired,
            "transform": before.acquired - before.transformed,
            "place": ((before.transformed if goal.transform else before.acquired)
                      - before.placed),
            "inspect": before.acquired - before.inspected,
        }
        if phase in readiness and readiness[phase] <= 0:
            predecessor = {"acquire": "located", "transform": "acquired",
                           "place": ("transformed" if goal.transform else "acquired"),
                           "inspect": "acquired"}[phase]
            flags.append("attempts '%s' with no unconsumed %s goal object ready" %
                         (phase, predecessor))

        required_fixture = _FIXTURE_FOR.get(goal.transform or goal.terminal, "")
        if (required_fixture and next_before in {"transform", "inspect"} and
                previous_fixture.startswith(required_fixture + " ") and
                current_fixture and current_fixture != previous_fixture):
            flags.append("leaves the fixture needed for '%s' while it is unmet" %
                         next_before)

        source = by_index.get(turn.step)
        content = source.content if source is not None else ""
        memory = _excerpt(content, "memory")
        reflection = _excerpt(content, "reflection")
        plan = _excerpt(content, "plan")
        out.append(ObligationRow(
            step=turn.step, action=turn.action, before=before, after=after,
            next_unmet_before=next_before, next_unmet_after=next_after,
            transitions=tuple(transitions), review_flags=tuple(flags),
            memory=memory, reflection=reflection, plan=plan,
            result=_clean(result)))

    return ObligationTimeline(goal, tuple(out), completion_steps, tuple(initial))


def _status(goal: GoalSpec, state: ObjectiveState) -> str:
    return "; ".join("%s=%d/%d" % (o, state.count(o), _required(goal, o))
                     for o in goal.obligations)


def render_timeline(timeline: ObligationTimeline) -> str:
    goal = timeline.goal
    lines = [
        "ALFWORLD OBJECTIVE OBLIGATION TIMELINE (code-derived simulator state)",
        "Goal object=%s; target=%s; quantity=%d%s." %
        (goal.object_type, goal.target_type, goal.quantity,
         ("; required transform=" + goal.transform) if goal.transform else ""),
        "Ordered obligations:",
    ]
    for pos, obligation in enumerate(goal.obligations, 1):
        step = timeline.completion_steps.get(obligation)
        if obligation in timeline.initial_completions:
            done = "completed in the initial observation"
        elif step is not None:
            done = "completed by simulator result at step %d" % step
        else:
            done = "never objectively completed"
        lines.append("O%d %s (%d required): %s" %
                     (pos, obligation, _required(goal, obligation), done))

    lines.extend([
        "",
        "Read the rows forward. Find which still-unmet obligation was FIRST "
        "treated by the agent as complete, unnecessary, or abandoned. The "
        "objective counts come from simulator results; agent excerpts are "
        "claims to compare against them. A REVIEW flag is only a place to "
        "check, not a verdict. Later refused actions and visible failures may "
        "only execute or expose an earlier divergence.",
        "",
    ])
    for row in timeline.rows:
        transition = ", ".join(row.transitions) or "none"
        review = "; ".join(row.review_flags) or "none"
        lines.extend([
            "[step %d] ACTION: %s" % (row.step, row.action or "(unparsed)"),
            "  BEFORE: %s; next unmet=%s" %
            (_status(goal, row.before), row.next_unmet_before),
            "  RESULT: %s" % (clip_middle(row.result, 260) or "(end of run or unparsed)"),
            "  AFTER: %s; next unmet=%s; transition=%s" %
            (_status(goal, row.after), row.next_unmet_after, transition),
            "  AGENT MEMORY: %s" % (row.memory or "(none)"),
            "  AGENT REFLECTION: %s" % (row.reflection or "(none)"),
            "  AGENT PLAN: %s" % (row.plan or "(none)"),
            "  REVIEW: %s" % review,
        ])
    return "\n".join(lines).strip()


def build_obligation_block(traj: Trajectory) -> Tuple[str, ObligationStats]:
    """Return the prompt block and diagnostics, using trajectory text only."""
    timeline = build_timeline(traj)
    block = render_timeline(timeline)
    stats = ObligationStats(
        rows=len(timeline.rows), chars=len(block),
        transitions=sum(len(row.transitions) for row in timeline.rows),
        review_rows=sum(bool(row.review_flags) for row in timeline.rows),
        refusals=sum("nothing happens" in row.result.casefold()
                     for row in timeline.rows),
    )
    return block, stats
