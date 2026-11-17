"""Which steps could be the critical step? The ones that CHANGED something.

On SWE-Bench-Pro our answers are almost never on a step that modified the repo,
and the reference labels almost always are:

    reference: write 0.56 / reasoning-only 0.13      (editor-only definition)
    ours:      write 0.19 / reasoning-only 0.34

and our accuracy tracks that split hard -- 0.375 when we answer an editor write
and 0.273 on a shell write, against 0.069 on a reasoning step and 0.037 on a
read. Two thirds of our answers sit on step kinds where we are essentially never
right. Restricting the answer space to the steps that committed a change is the
largest single lever measured so far.

WHAT THIS ASSUMES, STATED PLAINLY. That only ``assistant`` steps can be the
critical step is given: the corpus says so in its own
``agent_framework_description``. That the critical step must have MODIFIED
something is NOT given -- it was inferred from the label distribution, and using
it is, on its face, fitting an annotation convention.

The defensible reading, and the one this module is written to: the localiser
prompt already applies a but-for test, and a but-for test needs a counterfactual
to bite on. A step that only read a file or only stated an intention leaves no
state behind for "had this been done differently" to change; the commitment is
not made until something is written. So the restriction is a formalisation of a
criterion the prompt already claims to use, not a new one. Where that reading is
wrong -- an agent whose fatal move was a plan it never acted on -- the
restriction costs us the case, and ``evaluate`` reports that cost separately
rather than hiding it.

HOW OFTEN THAT READING IS WRONG, MEASURED. On SWE-Bench-Pro the reference step
is outside the offered list in 24 of 86 trajectories. Nineteen of those are
steps with no tool call at all -- prose that names the files and lines it is
about to change and then stops. The corpus really does label planning critical,
so roughly a fifth of this benchmark is unreachable by any write-based answer
space, and 0.78 is a ceiling on the method rather than on the implementation.
That is a limitation to report, not a bug to fix; widening the parse to reach
those steps would mean offering every reasoning step, which is the unrestricted
answer space this module exists to narrow. See probe/sp_missed.py.

DEGRADATION. If a trajectory yields fewer than ``MIN_CANDIDATES`` writes, the
parse is presumed to have missed the framework's idiom and the caller falls back
to the unrestricted answer space. An empty candidate list must never silently
become "no valid answers".

This module reads a ``Trajectory`` only. The reference label never enters it.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

MIN_CANDIDATES = 2

# --- TrajDebug idiom:  str_replace_editor({"command": "str_replace", ...}) ---
_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(\{")
_EDITOR_WRITE = {"create", "str_replace", "insert"}

# --- TrajAudit idiom:  "Action: <payload>" on its own line ------------------
_ACTION = re.compile(r"^Action:[ \t]*(.*)$", re.M)

# Shell forms that modify a file. Conservative on redirection: a bare ">" also
# redirects to /dev/null and to here-doc scratch files, so a path-looking
# target is required. Read-only tools (grep/find/cat/ls/nl/wc/head) never match.
_SHELL_WRITE = re.compile(
    r"\bsed\s+-i|"
    r"(?<![>\d])>>?\s*[\w.$~/-]*/[\w./-]+|"
    r"\bpatch\b|\bgit\s+apply\b|\bgit\s+checkout\s+--|"
    r"\btee\s+[\w./-]+|"
    r"\b(?:cp|mv)\s+[\w./-]+\s+[\w./-]+|"
    r"\brm\s+(?:-\w+\s+)*[\w./-]+")
# Framework function names that commit an edit (auto-code-rover and kin).
_FUNC_WRITE = re.compile(r"write_patch|apply_patch|edit_file|\bpatch\b|\bedit\b",
                         re.I)

# An in-line interpreter script that writes a file back out. Agents reach for
# this when the editor tool cannot express the edit -- read the file, transform
# the text in memory, write it back -- and it accounted for six of the twenty-
# four SWE-Bench-Pro reference steps the parse used to miss.
#
# The signal is a WRITE MODE, never a bare ``open(``. Every one of those six
# opens in "r" first, because the read comes before the edit, so ``open(`` marks
# file access rather than modification and would also match the read-only
# one-liners and pytest invocations that sit in the same trajectories. The
# character classes admit a backslash because these commands arrive embedded in
# JSON, where the quotes are escaped. The comma before the mode is load-bearing:
# without it open("a") matches, because a one-character filename cannot be told
# apart from the append mode. The mode is the SECOND argument, and requiring the
# comma is what separates a write from a read.
_PY_WRITE = re.compile(
    r"open\s*\([^)]*,\s*['\\\"]{1,3}[wax]\+?b?['\\\"]{1,3}|"   # open(p, "w")
    r"\.write\s*\(|\.writelines\s*\(|write_text\s*\(|"
    r"(?:json|yaml)\.dump\s*\(")

# NOT added, and deliberately: a redirect to a relative filename ("> patch.js").
# It reaches one further reference step and costs 57 extra candidates across the
# corpus, nearly all of them `cat << EOF > test_foo.py` -- throwaway scripts the
# agent writes to reproduce a bug, not repo edits. Recall bought by lengthening
# the offered list is not recall; the list is only a restriction while it is
# short. Measured in probe/sp_widen.py.


def _first_call(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(tool, command, path) from a TrajDebug-style embedded call."""
    m = _CALL.search(text)
    if not m:
        return (None, None, None)
    body = text[m.end():]
    cm = re.search(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', body)
    pm = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"', body)
    return (m.group(1), cm.group(1) if cm else None, pm.group(1) if pm else None)


def _short_path(p: Optional[str]) -> str:
    """Drop the per-instance /workspace/<instance-hash>/ prefix for display."""
    if not p:
        return ""
    parts = p.split("/")
    if len(parts) > 2 and parts[1] == "workspace":
        parts = parts[3:]
    out = "/".join(parts)
    return out if len(out) <= 70 else "..." + out[-67:]


def classify(text: str) -> Tuple[bool, str]:
    """Did this step modify anything, and what is a one-line label for it?

    Returns ``(is_write, label)``. ``label`` is rendered into the candidate list
    and so must stay short and free of anything resembling a verdict.
    """
    text = text or ""

    # 1. TrajDebug: an embedded tool call.
    tool, cmd, path = _first_call(text)
    if tool:
        if tool == "str_replace_editor":
            if cmd in _EDITOR_WRITE:
                return True, "%s %s" % (cmd, _short_path(path))
            return False, "%s %s" % (cmd or "?", _short_path(path))
        if cmd and (_SHELL_WRITE.search(cmd) or _PY_WRITE.search(cmd)):
            return True, "shell: %s" % cmd[:70]
        return False, "%s %s" % (tool, (cmd or "")[:60])

    # 2. TrajAudit: an "Action:" line.
    m = _ACTION.search(text)
    if not m:
        return False, ""
    act = m.group(1).strip()
    if not act:
        return False, ""

    # 2a. a JSON list of framework calls
    if act[:1] in "[{":
        names = re.findall(r'"func_name"\s*:\s*"([^"]+)"', act)
        if not names:
            t = re.search(r'"tool"\s*:\s*"([^"]+)"', act)
            names = [t.group(1)] if t else []
            # A tool whose NAME is neutral and whose verb lives in the payload.
            # ``str_replace_editor`` is the whole of this case: the same name
            # covers 440 reads and 285 writes on TrajAudit, so deciding from the
            # name is not merely imprecise, it is undecidable. Branch 1 already
            # reads ``command`` for exactly this tool when the call arrives in
            # TrajDebug's ``tool({...})`` form; this is the same test on the
            # ``Action: {"tool": ...}`` form, which is how TrajAudit writes it.
            #
            # Without it ``_FUNC_WRITE`` decided the question, and decided it
            # wrongly in a way nothing downstream could see: ``\bedit\b`` does
            # not match "editor" (the ``or`` ends the word), so every write in
            # this idiom read as a non-write. 43 of 102 TrajAudit rows then fell
            # under MIN_CANDIDATES and got no pool at all -- and because
            # ``pipeline`` gates the shortlist stage on a non-empty ``cands``,
            # those rows silently lost the LLM proposer too. Measured in
            # probe/ta_poolfix.py.
            if names == ["str_replace_editor"]:
                cm = re.search(r'"command"\s*:\s*"([^"]+)"', act)
                if cm:
                    pm = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"', act)
                    return (cm.group(1) in _EDITOR_WRITE,
                            "%s %s" % (cm.group(1),
                                       _short_path(pm.group(1) if pm else "")))
        if names:
            hit = [x for x in names if _FUNC_WRITE.search(x)]
            return bool(hit), ", ".join(names[:3])[:70]
        # a JSON blob we do not recognise: try the raw text for a shell write
        return bool(_SHELL_WRITE.search(act) or _PY_WRITE.search(act)), act[:70]

    # 2b. a bare command line
    head = act.split()[0]
    if head == "str_replace_editor":
        sub = act.split()[1] if len(act.split()) > 1 else ""
        cmdname = sub if sub in _EDITOR_WRITE else (
            "str_replace" if "--old_str" in act or "--new_str" in act else sub)
        if cmdname in _EDITOR_WRITE:
            fp = re.search(r"(/[\w./-]+)", act)
            return True, "%s %s" % (cmdname, _short_path(fp.group(1) if fp else ""))
        return False, act[:70]
    return bool(_SHELL_WRITE.search(act) or _PY_WRITE.search(act)), act[:70]


def candidates(traj) -> List[Dict[str, object]]:
    """Assistant steps that modified something, in trajectory order.

    Returns ``[]`` when too few were found to trust the parse -- the caller must
    read that as "do not restrict", never as "nothing is answerable".
    """
    out: List[Dict[str, object]] = []
    for s in traj.steps:
        if s.role != "assistant":
            continue
        is_write, label = classify(s.content)
        if is_write:
            out.append({"step": s.index, "label": label})
    return out if len(out) >= MIN_CANDIDATES else []


# --- Plan steps: the other half of TrajAudit's labels -----------------------
#
# A write-only answer space fits SWE-Bench-Pro (76.7% of its reference steps are
# writes) and only half-fits TrajAudit (52.0%). The gap is not a parse quality
# difference -- after the 2a fix exactly ONE TrajAudit miss is a parse failure
# and 49 are steps that genuinely wrote nothing. TrajAudit labels the step where
# the run COMMITTED to the wrong approach, and that is often the prose step that
# named the file and the method before any edit was made. Its reference steps
# also sit earlier (median position 0.39 of the assistant steps, against 0.62 on
# SWE-Bench-Pro), which is where the deciding still happens.
#
# DENSITY IS THE WHOLE QUESTION, and it is why this is not simply "offer the
# prose too". Widening the answer space has been measured four times in this
# repo and lost every time (sp_cnocand -5, sp_union -6, two arms losing the same
# 7 rows). A signal that fires on half the prose is not a restriction at all. So
# the test is deliberately narrow and its density is reported with it: it fires
# on 1.4% of assistant steps on TrajAudit (75 of 5268), a median of 0 per row.
#
# Measured in probe/ta_planpool.py, zero calls:
#   gold in pool   52/102 = 0.5098  ->  65/102 = 0.6373  (+13 rows)
#   on SWE-Bench-Pro                     66/86  -> 66/86  (+0 rows)
# The SP reading is the control: this must not move a corpus whose labels the
# write definition already fits, and it does not.
#
# THIS IS AN ANSWER-SPACE READING, NOT A SCORE. The pool getting bigger helps
# recall and hurts ranking; repo rule three (mid-layer gains do not cash out at
# the endpoint) has 15+ confirmations against it.

_PLAN_MIN_CHARS = 200

# Named code objects: a backticked identifier, a path with a source extension,
# or an explicit line reference. Requiring these is what separates "I will fix
# the bug" (no information) from "`_load_file` in application.py line 41".
_PLAN_NAMES = re.compile(
    r"`[A-Za-z_][\w.]*`|"
    r"[\w/]+\.(?:py|js|go|ts|java|rb|c|h|cpp|yml|yaml|json|tmpl|cfg)\b|"
    r"\blines?\s+\d+", re.I)
# The step states a conclusion about where the fault is, or what to do about it.
_PLAN_SUMMARY = re.compile(
    r"\b(?:root cause|the (?:issue|problem) is|here's what I|based on my"
    r"|looking at the code|I can see (?:the|that)|to summarize"
    r"|problem statement|the solution)\b", re.I)
# The step says what it is about to change. Admitted only alongside TWO named
# code objects, which is what keeps "I need to fix this" out.
_PLAN_INTENT = re.compile(
    r"\b(?:I need to|I should|I will|I'll|we need to|the (?:fix|solution)"
    r"|need to (?:modify|change|add|implement|update|fix)"
    r"|let me (?:modify|change|add|implement|update|fix))\b", re.I)


def is_plan(text: str) -> bool:
    """Did this step decide what to change, without changing it yet?

    Prose only: a step with a tool call is already judged by ``classify``, and
    letting both fire would double-count it.

    Two ways in, both from probe/ta_planpool.py's density table -- tier P1
    (an intention plus two named objects) and tier P2 (a located-the-fault
    phrasing plus one). P3, "any long Thought: step", fires on 22% of the prose
    and is NOT admitted: at that density the list stops being a restriction.
    """
    text = text or ""
    if len(text) < _PLAN_MIN_CHARS:
        return False
    tool, _, _ = _first_call(text)
    if tool:
        return False
    m = _ACTION.search(text)
    if m and m.group(1).strip():
        return False
    named = len(_PLAN_NAMES.findall(text))
    if named >= 2 and _PLAN_INTENT.search(text):
        return True
    return bool(_PLAN_SUMMARY.search(text)) and named >= 1


def candidates_with_plans(traj, cap: int = 15) -> List[Dict[str, object]]:
    """Writes UNION plan steps, in trajectory order, capped at ``cap``.

    The cap keeps the offered list short enough to still be a restriction. When
    it binds, the EARLIEST steps are kept: TrajAudit's reference step sits at
    median position 0.39, and on the 24 rows where the cap bites, keeping the
    first 15 loses zero reference steps while keeping the last 15 loses all 13
    rows this stage just gained (probe/ta_planpool.py, table three).
    """
    out: List[Dict[str, object]] = []
    for s in traj.steps:
        if s.role != "assistant":
            continue
        is_write, label = classify(s.content)
        if is_write:
            out.append({"step": s.index, "label": label})
        elif is_plan(s.content):
            out.append({"step": s.index, "label": "plan"})
    if len(out) < MIN_CANDIDATES:
        return []
    return out[:cap]


def render(cands: List[Dict[str, object]]) -> str:
    """The candidate block for the localiser prompt. Empty list -> ''."""
    if not cands:
        return ""
    lines = ["  step %d — %s" % (c["step"], c["label"] or "(edit)")
             for c in cands]
    return "\n".join(lines)


# --- Handoffs: control changing hands in a multi-agent trajectory ----------
#
# The two answer spaces above are parses of what a step DID (it wrote; it named
# a file and a method). This one is a parse of WHO IS SPEAKING, and it exists
# because one corpus makes the author carry the control structure: in
# longrca-bench's Diagnost/Action/Judge framework a step whose author is
# ``DiagnostAgent (-> ActionAgent)`` is not an ordinary turn, it is the message
# that hands the task to another agent.
#
# WHAT THIS ASSUMES, STATED AS PLAINLY AS THE WRITE RULE ABOVE. Measured on
# longrca-bench/SWE-bench Pro (128 rows, zero calls):
#
#     arrow-authored steps:  1.6% of all steps, 57.0% of the reference steps
#                            -> lift 35.4x, against 1.61x for the span prior
#     "take the first one":  0.4375 acc@0, against 0.0547 for the best constant
#
# A 35x enrichment is not a criterion the localiser prompt already applies, the
# way the write rule arguably is. It was read off the label distribution, so
# using it IS fitting an annotation convention and is reported as such. Two
# facts bound the claim, and both must travel with it:
#
#   - It is framework-specific, not general. Of the five longrca-bench subsets
#     only two have arrow authors at all; on TravelPlanner, VitaBench and
#     WebArena Verified this rule nominates NOTHING, and on Terminal Bench 2 it
#     scores 0.1429. Nothing here may be claimed as a method-level improvement.
#   - What makes it defensible as a PROTECTION set rather than an answer space
#     is separate and stronger: arrow steps are the longest class in the corpus
#     (median 2780 chars against 298 for ActionAgent), and the compressor
#     chooses by length, so it rewrites 63.2% of them against 16.3% of all
#     steps. Protecting them repairs a measured bias of our own pipeline
#     against a step kind, which is a different act from restricting the
#     answer space to where the labels are.
#
# Only ``name`` is read, which is an input field present in the loaded
# trajectory. No label is consulted anywhere in this module.

#: Authors of this shape are handing control to the agent named after the arrow.
_HANDOFF_MARK = "->"


def handoffs(traj) -> List[Dict[str, object]]:
    """Steps whose author marks a transfer of control, in trajectory order.

    Returns ``[]`` when the trajectory has none -- callers must read that as
    "this corpus has no handoff structure, change nothing", exactly as they
    read an empty ``candidates()``. There is deliberately no MIN_CANDIDATES
    floor: one handoff is a real handoff, whereas one parsed write is more
    likely a parse failure than a trajectory that edited one file once.
    """
    out: List[Dict[str, object]] = []
    for s in traj.steps:
        if _HANDOFF_MARK not in (s.name or ""):
            continue
        target = (s.name or "").split(_HANDOFF_MARK, 1)[1].strip(" )")
        out.append({"step": s.index,
                    "label": "hands off to %s" % (target or "another agent")})
    return out


def snap(cands: List[Dict[str, object]], step: Optional[int]) -> Optional[int]:
    """Force an answer onto the candidate set, nearest first, ties going early.

    Only used to repair a model that ignored the restriction. A snapped answer
    is recorded as such so the reported accuracy can be split by whether the
    localiser respected the answer space or had to be corrected into it.
    """
    if step is None or not cands:
        return step
    steps = [int(c["step"]) for c in cands]
    if step in steps:
        return step
    return min(steps, key=lambda i: (abs(i - step), i))
