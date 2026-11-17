"""Rule-level operation units: what each step DID, and where one run of them ends.

This module is the deterministic half of two stages that used to be entirely
model-driven, and it is deliberately a near-leaf: it imports ``candidates`` and
``schema`` and nothing else, so ``spans`` can depend on it without dragging in
``localize``.

WHAT A UNIT IS. A unit is a maximal run of consecutive attributable steps that
perform the SAME KIND of operation. The five kinds are the five that
``optype`` has always returned, renamed here to the vocabulary the chain is
read in:

    read    -> observation     the agent looked at something
    reason  -> plan            the agent thought, concluded or decided
    search  -> localization    the agent went looking for where something is
    edit    -> edit            the agent changed something
    test    -> test            the agent ran something to find out

So a trajectory renders as

    0-3 observation -> 4 plan -> 5-9 localization -> 10-11 edit -> 12 test -> ...

WHY THIS IS WORTH HAVING. ``spans.segment`` asks a model to cut a 117-step
digest into phases from scratch, which is a hard global problem answered off
one-line-per-step summaries. But the boundaries that matter are mostly already
visible in the text: an agent that stops editing and starts running tests has
changed sub-goal, and that transition is a regex away. Handing the model a
chain of ~20 typed units and asking it to GROUP them is a much smaller problem
than handing it 117 lines and asking it to cut them.

The second reason is the one that unblocks prior-conditioned compression. A
unit chain is available BEFORE any compression call has been made, so a stage
that reads it can run before compression rather than after. ``span_prior`` used
to read ``Span.summary``, which only exists once ``compress`` has written it --
that ordering is what made "let the prior tell the compressor what to keep"
circular. A rule-derived digest breaks the cycle without a second pass.

``optype`` lives here rather than in ``op_analysts`` because two callers now
need it and ``op_analysts`` sits above ``localize``. It is re-exported there so
existing imports and tests keep working.

No ground truth reaches this module: it sees ``Trajectory`` only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .candidates import _ACTION, _first_call, classify
from .schema import Trajectory

# The five operations, in the order the chain vocabulary lists them. There is
# deliberately NO "other" bucket: an earlier draft with one put 25 of 102 gold
# steps into it, a quarter of the answer space in a bin nobody owns. Every step
# lands in one of the five; the fallback is ``read``, the passive type, because
# a step that neither writes, searches, runs nor thinks is an observation.
OP_TYPES = ("edit", "reason", "read", "search", "test")

# The chain vocabulary. These are the words the segmentation prompt and the
# span labels are written in; ``OP_TYPES`` stays the internal name so nothing
# downstream of ``op_analysts`` has to change.
UNIT_ROLES = ("observation", "plan", "localization", "edit", "test")

ROLE_OF = {
    "read": "observation",
    "reason": "plan",
    "search": "localization",
    "edit": "edit",
    "test": "test",
}

_RX_SEARCH = re.compile(
    r"\bsearch_\w+|\"func_name\"\s*:\s*\"search|"
    r"\b(grep|rg|ack|find|locate|ls)\b", re.I)
_RX_TEST = re.compile(
    r"pytest|unittest|tox|\bnpm (?:run )?test\b|\bgo test\b|\bmake test\b|"
    r"run_tests|\btest\b", re.I)
_RX_READ = re.compile(
    r"\b(view|cat|head|tail|nl|less|more|read_file|open)\b", re.I)
_RX_RUN = re.compile(
    r"execute_bash|\bpython3?\b|\bnode\b|\bgo run\b|\bbash\b|\bsh\s+-c\b|"
    r"<<|&&", re.I)


def optype(text: str) -> str:
    """Which of the five operations is this step? Never returns anything else."""
    text = text or ""
    m = _ACTION.search(text)
    act = (m.group(1).strip() if m else "")
    if not act:
        # No "Action:" line. On TrajAudit that means the agent thought, planned
        # or concluded. TrajDebug, though, NEVER writes one -- it embeds the
        # call in the step body as ``str_replace_editor({...})``. Returning
        # "reason" here put 9913 of ~10000 swebenchpro steps, and all 86 gold
        # steps, in that one bucket, while classify() below would have got them
        # right: it tries the embedded call FIRST and the Action line second.
        #
        # So fall through to the embedded call. Dispatch on the tool NAME rather
        # than on the regexes, because the only action surface available here is
        # ``tool/command/path`` and matching _RX_TEST against a path misreads a
        # view of ``tests/test_x.py`` as a test run. The regexes are applied to
        # the shell command only, which is what they were measured on.
        tool, cmd, _path = _first_call(text)
        if not tool:
            return "reason"
        if classify(text)[0]:
            return "edit"
        if tool == "str_replace_editor":
            # classify() already ruled out the writing subcommands, so whatever
            # is left (view, open, missing) is an observation.
            return "read"
        if _RX_SEARCH.search(tool):
            return "search"
        act = cmd or ""
        if not act:
            return "read"
    elif classify(text)[0]:
        return "edit"
    if _RX_SEARCH.search(act):
        return "search"
    if _RX_TEST.search(act):
        return "test"
    tool, cmd, _ = _first_call(act if act[:1] in "[{" else text)
    if tool == "str_replace_editor" and cmd in (None, "", "view", "open"):
        return "read"
    if _RX_READ.search(act):
        return "read"
    if _RX_RUN.search(act):
        # Running something that is not a test suite is still execution: it
        # produces an observation the agent then acts on, and it belongs with
        # the analyst that reasons about what running things revealed.
        return "test"
    return "read"


def role_of(text: str) -> str:
    """The chain vocabulary word for this step."""
    return ROLE_OF[optype(text)]


# ---------------------------------------------------------------------------
# What the operation was ABOUT. Rule-derived, zero calls: the point of the
# chain is that it can be built before any model has been asked anything.
_PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}|\b[\w./\-]+\.(?:py|js|ts|go|rs|java"
                      r"|rb|json|ya?ml|toml|cfg|md|rst|txt|c|h|cpp)\b")
_SYM_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_]\w*)")
# TrajAudit's auto-code-rover trajectories put a serialised call list where
# other corpora put a shell line, so the prose fallbacks below would otherwise
# return a 48-character slice of raw JSON -- ``[{"func_name": "search_method_in``
# and nothing else. The digest built from these is read by the span prior now
# that it runs before compression, so a target that is half a JSON literal is
# not cosmetic: it is what the model is asked to reason over.
_JSON_CALL = re.compile(r'"(?:func_name|name|tool_name)"\s*:\s*"([^"]{1,60})"')
_JSON_ARG = re.compile(r'"(?:file_?path|path|file|class_name|method_name'
                       r'|function_name|term|query|code)"\s*:\s*"([^"]{1,80})"')


def _looks_structural(s: str) -> bool:
    """Whether ``s`` is the head of a serialised blob rather than a phrase."""
    return s.lstrip()[:2] in ('[{', '{"', '["', '[[') or s.lstrip()[:1] == '{'


def _short_path(p: str) -> str:
    """Drop the per-instance /workspace/<hash>/ prefix that carries no meaning."""
    parts = (p or "").split("/")
    if len(parts) > 2 and parts[1] == "workspace":
        parts = parts[3:]
    out = "/".join(parts)
    return out if len(out) <= 48 else "..." + out[-45:]


def target_of(text: str) -> str:
    """A short phrase naming what this step operated on. "" when unclear.

    Preference order is structural before textual: an embedded call states its
    own target, and guessing from prose is only worth doing when there is no
    call to read.
    """
    text = text or ""
    tool, cmd, path = _first_call(text)
    if path:
        return _short_path(path)
    if cmd and not _looks_structural(cmd):
        return " ".join(cmd.split())[:48]
    # A serialised call: read the call's own fields rather than slicing it.
    fn = _JSON_CALL.search(text)
    if fn:
        arg = _JSON_ARG.search(text)
        name = fn.group(1)
        if arg:
            a = arg.group(1)
            return ("%s %s" % (name, _short_path(a) if "/" in a else a))[:48]
        return name[:48]
    m = _ACTION.search(text)
    act = (m.group(1).strip() if m else "")
    if act:
        p = _PATH_RE.search(act)
        if p:
            return _short_path(p.group(0))
        if not _looks_structural(act):
            return " ".join(act.split())[:48]
    # A pure reasoning step: name the first symbol or path it talks about, so
    # the chain says "plan about urls.py" rather than a bare "plan".
    for rx in (_SYM_RE, _PATH_RE):
        m2 = rx.search(text)
        if m2:
            g = m2.group(1) if rx is _SYM_RE else m2.group(0)
            return _short_path(g)
    return ""


# ---------------------------------------------------------------------------
@dataclass
class Unit:
    """A maximal run of consecutive steps doing the same kind of operation."""

    start: int
    end: int                       # inclusive
    role: str                      # one of UNIT_ROLES
    targets: List[str]             # distinct targets seen inside, in order

    @property
    def indices(self) -> List[int]:
        return list(range(self.start, self.end + 1))

    def __len__(self) -> int:
        return self.end - self.start + 1

    def summary(self) -> str:
        """``role`` plus what it was about, the chain's payload."""
        if not self.targets:
            return self.role
        return "%s %s" % (self.role, ", ".join(self.targets[:3]))

    def to_dict(self):
        return {"start": self.start, "end": self.end, "role": self.role,
                "targets": self.targets[:6]}


#: Roles whose step is not an operation the agent chose to perform, and which
#: therefore continue the open unit instead of starting one.
#:
#: This is NOT ``Step.attributable``, and the difference is the whole reason the
#: constant exists. ``attributable`` answers "may this step be BLAMED", and on
#: TrajDebug a ``tool`` step may: the corpus labels tool results as critical
#: steps, so widening ``NON_ATTRIBUTABLE_ROLES`` to cover them would silently
#: change candidate generation and the localiser's answer space. Authorship is a
#: different question. A tool result is the ANSWER to the action above it, not a
#: new operation, and treating it as one is measurably destructive: on
#: SWE-Bench-Pro, where the corpus alternates assistant and tool, doing so
#: produced a 90-unit chain over 102 steps reading
#: ``observation -> plan -> observation -> plan -> ...`` -- the ``plan`` units
#: were every one of them tool output, and 87% of steps ended up alone in a unit.
#: Folding them gives the chain its intended shape and costs nothing: a folded
#: step keeps its index and stays inside the unit's span, so nothing downstream
#: loses the ability to cite it.
UNAUTHORED_ROLES = frozenset(("user", "system", "tool"))


def unauthored(step) -> bool:
    """Whether ``step`` continues the open unit rather than starting one."""
    return (getattr(step, "role", "") in UNAUTHORED_ROLES
            or not getattr(step, "attributable", True))


def unit_chain(traj: Trajectory, max_units: int = 0) -> List[Unit]:
    """Split ``traj`` into typed operation units covering every step.

    The cover is total and contiguous over 0..n-1, including unauthored steps,
    because ``spans`` guarantees a full cover and builds on this. An unauthored
    step -- a tool result, a user turn, the system prompt -- joins whatever unit
    is open rather than starting one: it is an observation the agent did not
    author. See ``UNAUTHORED_ROLES`` for why that is not ``attributable``.

    ``max_units`` > 0 merges the shortest adjacent pairs until at most that many
    remain, so a pathological trajectory cannot produce a 200-line chain.
    """
    n = len(traj)
    if n <= 0:
        return []
    units: List[Unit] = []
    for st in traj.steps:
        i = st.index
        if unauthored(st):
            if units:                      # fold into the open unit
                units[-1].end = i
                continue
            role, tgt = "observation", ""
        else:
            role = role_of(st.content or "")
            tgt = target_of(st.content or "")
        if units and units[-1].role == role:
            units[-1].end = i
            if tgt and tgt not in units[-1].targets:
                units[-1].targets.append(tgt)
        else:
            units.append(Unit(start=i, end=i, role=role,
                              targets=[tgt] if tgt else []))
    if not units:
        units = [Unit(start=0, end=n - 1, role="observation", targets=[])]
    units[-1].end = max(units[-1].end, n - 1)
    if max_units and len(units) > max_units:
        units = _merge_shortest(units, max_units)
    return units


def _merge_shortest(units: List[Unit], target: int) -> List[Unit]:
    """Merge the shortest unit into a neighbour until ``target`` remain.

    The survivor keeps the LONGER neighbour's role, because that is the one
    describing most of the merged steps. Ties go to the earlier unit so the
    result does not depend on iteration order.
    """
    units = [Unit(u.start, u.end, u.role, list(u.targets)) for u in units]
    while len(units) > target and len(units) > 1:
        k = min(range(len(units)), key=lambda j: (len(units[j]), j))
        left, right = k - 1, k + 1
        if left < 0:
            j = right
        elif right >= len(units):
            j = left
        else:
            j = left if len(units[left]) >= len(units[right]) else right
        a, b = (j, k) if j < k else (k, j)
        keep = units[a] if len(units[a]) >= len(units[b]) else units[b]
        merged = Unit(start=units[a].start, end=units[b].end, role=keep.role,
                      targets=(units[a].targets + [t for t in units[b].targets
                                                   if t not in units[a].targets]))
        units[a:b + 1] = [merged]
    return units


def render_chain(units: Sequence[Unit], width: int = 0) -> str:
    """The chain, one unit per line, indices first so a model can cite them."""
    out = []
    for u in units:
        rng = ("%d" % u.start) if u.start == u.end else ("%d-%d" % (u.start, u.end))
        line = "  [%s] %s" % (rng, u.summary())
        out.append(line[:width] if width else line)
    return "\n".join(out)


def render_arrow(units: Sequence[Unit], limit: int = 40) -> str:
    """The compact ``observation -> plan -> edit -> test`` form, for logs."""
    roles = [u.role for u in units[:limit]]
    tail = " -> ..." if len(units) > limit else ""
    return " -> ".join(roles) + tail


def chain_digest(units: Sequence[Unit], lo: int, hi: int,
                 max_units: int = 8) -> str:
    """A deterministic one-paragraph description of steps ``lo..hi``.

    This is what a pre-compression reader gets instead of ``Span.summary``. It
    is not a summary in the model sense -- nothing is paraphrased, it just names
    the operations in order and what they were aimed at -- which is precisely
    why it can be produced before the compressor has run.
    """
    inside = [u for u in units if u.end >= lo and u.start <= hi]
    if not inside:
        return "(no operations)"
    bits = []
    for u in inside[:max_units]:
        s, e = max(u.start, lo), min(u.end, hi)
        rng = ("%d" % s) if s == e else ("%d-%d" % (s, e))
        bits.append("%s %s" % (rng, u.summary()))
    if len(inside) > max_units:
        bits.append("... %d more" % (len(inside) - max_units))
    return " -> ".join(bits)


def role_counts(units: Sequence[Unit], lo: int, hi: int) -> dict:
    """How many STEPS of each role fall in ``lo..hi``."""
    out = {r: 0 for r in UNIT_ROLES}
    for u in units:
        s, e = max(u.start, lo), min(u.end, hi)
        if e >= s:
            out[u.role] = out.get(u.role, 0) + (e - s + 1)
    return out
