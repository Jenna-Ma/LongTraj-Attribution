"""Report acc@0 on a fixed dev/test split, so a sweep cannot fit itself quietly.

Trying several variants and keeping the best-scoring one is selection ON the
reference labels even when no label ever enters a prompt. The number that comes
out of such a search is biased upward, and the bias is invisible if only the
full-corpus figure is reported.

So: split each corpus in half by a stable hash of the task id, tune on `dev`,
and read `test` once at the end. The split depends on nothing but the task id --
not on the run, the order, or the score -- so every variant is scored on exactly
the same two halves and the comparison between variants stays fair.

Usage:
    python probe/split_eval.py probe/sp_06_priorsig.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SALT = "longtraj/split/v1"


def half(task_id: str) -> str:
    """Stable, run-independent assignment of one task to dev or test."""
    h = hashlib.sha256((SALT + "|" + task_id).encode("utf-8")).hexdigest()
    return "dev" if int(h[:8], 16) % 2 == 0 else "test"


def score(cases):
    out = {}
    for name in ("all", "dev", "test"):
        rows = [c for c in cases
                if name == "all" or half(c["task_id"]) == name]
        if not rows:
            out[name] = (0.0, 0)
            continue
        hits = sum(1 for c in rows if c.get("hit@0"))
        out[name] = (hits / float(len(rows)), len(rows))
    return out


def main(argv) -> int:
    paths = argv[1:]
    if not paths:
        print(__doc__)
        return 2
    print("%-30s %-16s %-16s %-16s" % ("run", "all", "dev (tune here)",
                                       "test (read once)"))
    print("-" * 80)
    for p in paths:
        if not os.path.exists(p):
            print("%-30s  missing" % os.path.basename(p))
            continue
        try:
            blob = json.load(open(p, encoding="utf-8"))
            cases = blob["cases"]
        except Exception as exc:
            print("%-30s  unreadable (%s)" % (os.path.basename(p), exc))
            continue
        s = score(cases)
        print("%-30s %.4f (n=%-3d) %.4f (n=%-3d) %.4f (n=%-3d)"
              % (os.path.basename(p).replace(".jsonl", ""),
                 s["all"][0], s["all"][1], s["dev"][0], s["dev"][1],
                 s["test"][0], s["test"][1]))
        # A number printed without its failure count reads as a measurement.
        # Failed calls only ever push acc@0 down, so a row with failures is a
        # lower bound, not a result -- and a row from a refused account is not
        # even that. Say so next to the number, every time.
        run = blob.get("run") or {}
        llm = run.get("llm") or {}
        if run.get("account_refused") or llm.get("blocked"):
            print("%-30s  ^^ ENDPOINT REFUSED mid-run (%s calls skipped) -- "
                  "NOT a run, do not score" % ("", llm.get("blocked", "?")))
        elif llm.get("failures"):
            print("%-30s  ^^ %d failed calls -- lower bound, not a clean read"
                  % ("", llm["failures"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
