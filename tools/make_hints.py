"""Hints for STaR-style collection: for real training tasks the agent never solved, describe where the
reference fix goes (files, enclosing functions, the lines that are wrong) without giving the new code.
The hint is appended to the issue text on Kaggle and stripped again by build_sft_data.py --strip-hints."""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HINT_MARK = "\n\n---\nMaintainer note (private, do not quote it): "
MAX_LINES = 60

gold = {json.loads(l)["instance_id"]: json.loads(l) for l in open(ROOT.parent / "gemma-4-developer-agent" / "tasks.jsonl")}
test = set(open(ROOT / "factory" / "paired_test_ids.txt").read().split())
solved = {json.loads(l)["id"] for f in glob.glob(str(ROOT / "results" / "*" / "summary.jsonl"))
          for l in open(f) if json.loads(l).get("resolved")}


def changed(p):
    return [l for l in p.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def hint(patch):
    parts = []
    for block in re.split(r"(?m)^--- ", patch)[1:]:
        m = re.search(r"(?m)^\+\+\+ b/(\S+)", block)
        if not m or re.search(r"(^|/)tests?/|conftest", m.group(1)):
            continue
        funcs = []
        for h in re.findall(r"(?m)^@@[^@]*@@ ?(.*)$", block):
            name = re.search(r"(?:def|class) (\w+)", h)
            if name and name.group(1) not in funcs:
                funcs.append(name.group(1))
        removed = [l[1:].strip() for l in block.splitlines() if l.startswith("-") and not l.startswith("---") and l[1:].strip()]
        desc = f"`{m.group(1)}`"
        if funcs:
            desc += " (in " + ", ".join(f"`{f}`" for f in funcs[:4]) + ")"
        if removed:
            desc += "; lines that are wrong or incomplete there include: " + " | ".join(f"`{r[:100]}`" for r in removed[:3])
        elif block.startswith("/dev/null"):
            desc += " is a NEW file that must be created"
        else:
            desc += "; only new lines need to be added there"
        parts.append(desc)
    n = len(changed(patch))
    return (HINT_MARK + "the complete fix changes about " + str(n) + " lines in " + str(len(parts)) + " file(s): "
            + "; ".join(parts) + ". Change every one of these places. Do not mention this note.")


ids = sorted((t for t in gold if t not in test and t not in solved and len(changed(gold[t]["patch"])) <= MAX_LINES),
             key=lambda t: len(changed(gold[t]["patch"])))
hints = {t: hint(gold[t]["patch"]) for t in ids}
json.dump(hints, open(ROOT / "factory" / "hints_real.json", "w"), indent=1)
open(ROOT / "factory" / "hint_ids.txt", "w").write("\n".join(ids) + "\n")
print(len(ids), "tasks; example:\n", ids[5], hints[ids[5]])
