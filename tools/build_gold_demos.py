"""Gold-edit demonstrations for real training tasks the agent fails.

For each failed run (not a paired-test task) we keep the agent's own exploration up to its first edit
(recorded but NOT trained on), then replace that edit with the reference fix, written as the same
python heredoc the agent uses, followed by submit_patch. Only these two steps are trained on.
Every heredoc is checked against the task snapshot: applying it must give exactly the files that
`git apply <reference patch>` gives.

Output: results/gold1/{summary.jsonl, llm_calls/<task>.jsonl}, consumed by build_sft_data.py --runs gold1.
Usage: python3 tools/build_gold_demos.py [--runs collect1 --runs real3v2]
"""
import argparse
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "gemma-4-developer-agent"
OUT = ROOT / "results" / "gold1"
MAX_CHANGED = 60
MAX_COMMAND_CHARS = 7000


def changed_lines(p):
    return [l for l in p.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def parse_patch(patch):
    """[(path, is_new_file, [hunk_lines...])]"""
    files = []
    for block in re.split(r"(?m)^(?=--- )", patch):
        m = re.search(r"(?m)^\+\+\+ b/(\S+)", block)
        if not m:
            continue
        new_file = block.startswith("--- /dev/null")
        hunks = [h.splitlines()[1:] for h in re.split(r"(?m)^(?=@@ )", block) if h.startswith("@@")]
        files.append((m.group(1), new_file, hunks))
    return files


def replacements(text, hunks):
    """Turn hunks into (old, new) pairs, trimming context while `old` stays unique. None if impossible."""
    pairs = []
    for h in hunks:
        h = [l for l in h if not l.startswith("\\")]   # "\ No newline at end of file"
        old = [l[1:] for l in h if l[:1] in " -"]
        new = [l[1:] for l in h if l[:1] in " +"]
        # trim shared leading/trailing context while the old block stays unique in the file
        while len(old) > 1 and len(new) > 1 and old[0] == new[0]:
            cand = "\n".join(old[1:]) + "\n"
            if text.count(cand) != 1 or not cand.strip():
                break
            old, new = old[1:], new[1:]
        while len(old) > 1 and len(new) > 1 and old[-1] == new[-1]:
            cand = "\n".join(old[:-1]) + "\n"
            if text.count(cand) != 1 or not cand.strip():
                break
            old, new = old[:-1], new[:-1]
        o, n = "\n".join(old) + "\n", "\n".join(new) + "\n"
        if text.count(o) != 1:
            return None
        text = text.replace(o, n)
        pairs.append((o, n))
    return pairs, text


def heredoc(task_patch, snapshot):
    parts, results = [], {}
    with tarfile.open(snapshot) as tf:
        for path, new_file, hunks in parse_patch(task_patch):
            if new_file:
                content = "".join(l[1:] + "\n" for h in hunks for l in h if l.startswith("+"))
                if "\nPY\n" in "\n" + content:
                    return None
                parts.append(("new", path, content))
                results[path] = content
                continue
            try:
                text = tf.extractfile("./" + path).read().decode()
            except (KeyError, AttributeError, UnicodeDecodeError):
                return None
            r = replacements(text, hunks)
            if r is None:
                return None
            pairs, final = r
            for o, n in pairs:
                if "'''" in o + n or "\\" in o + n or "\nPY\n" in "\n" + o + n:
                    return None
            parts.append(("edit", path, pairs))
            results[path] = final
    cmds, py = [], []
    for kind, path, payload in parts:
        if kind == "new":
            cmds.append(f"cat > {path} <<'PY'\n{payload}PY")
        else:
            py.append(f'p = Path("{path}")\ns = p.read_text()')
            for o, n in payload:
                py.append(f"old = '''{o.rstrip(chr(10))}'''\nnew = '''{n.rstrip(chr(10))}'''\n"
                          f"assert s.count(old) == 1, s.count(old)\ns = s.replace(old, new)")
            py.append("p.write_text(s)")
    if py:
        cmds.insert(0, "python3 - <<'PY'\nfrom pathlib import Path\n" + "\n".join(py) + '\nprint("edited")\nPY')
    return "\n".join(cmds) + "\n", results


def check(task_patch, snapshot, command, paths):
    """Running the heredoc must give exactly the files that `git apply` of the reference patch gives."""
    if not task_patch.endswith("\n"):
        task_patch += "\n"
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tarfile.open(snapshot) as tf:
        tf.extractall(a, filter="data")
        tf.extractall(b, filter="data")
        p = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], input=task_patch, text=True, cwd=a,
                           capture_output=True)
        q = subprocess.run(["bash", "-c", command], cwd=b, capture_output=True, text=True)
        if p.returncode or q.returncode:
            return False
        return all((Path(a) / path).read_text() == (Path(b) / path).read_text() for path in paths)


def first_edit_index(calls):
    for i, c in enumerate(calls):
        msg = c["response"]["choices"][0]["message"]
        for tc in msg.get("tool_calls") or []:
            a = tc["function"].get("arguments")
            a = a if isinstance(a, str) else json.dumps(a)
            if "write_text" in a or "cat >" in a:
                return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", default=[])
    args = ap.parse_args()
    runs = args.runs or ["real3v2", "collect1"]
    gold = {json.loads(l)["instance_id"]: json.loads(l) for l in open(DATA / "tasks.jsonl")}
    test = set(open(ROOT / "factory" / "paired_test_ids.txt").read().split())
    stats = {"failed_runs": 0, "too_big": 0, "no_edit": 0, "heredoc_impossible": 0, "check_failed": 0, "built": 0}
    (OUT / "llm_calls").mkdir(parents=True, exist_ok=True)
    done = set()
    with open(OUT / "summary.jsonl", "w") as summary:
        for run in runs:
            for line in open(ROOT / "results" / run / "summary.jsonl"):
                row = json.loads(line)
                t = row["id"]
                if row.get("resolved") or t in test or t in done or t not in gold:
                    continue
                stats["failed_runs"] += 1
                if len(changed_lines(gold[t]["patch"])) > MAX_CHANGED:
                    stats["too_big"] += 1
                    continue
                f = ROOT / "results" / run / "llm_calls" / f"{t}.jsonl"
                calls = [json.loads(l) for l in open(f)] if f.exists() else []
                k = first_edit_index(calls)
                if k is None:           # never edited: keep its first 8 steps of exploration
                    if len(calls) < 3:
                        stats["no_edit"] += 1
                        continue
                    k = min(len(calls) - 1, 8)
                    stats["cut_no_edit"] = stats.get("cut_no_edit", 0) + 1
                snap = DATA / "snapshots" / f"{t}.tgz"
                h = heredoc(gold[t]["patch"], snap)
                if h is None or len(h[0]) > MAX_COMMAND_CHARS:
                    stats["heredoc_impossible"] += 1
                    continue
                command, expected = h
                if not check(gold[t]["patch"], snap, command, list(expected)):
                    stats["check_failed"] += 1
                    continue
                files = ", ".join(f"`{p}`" for p in expected)
                edit_msg = {"role": "assistant", "content": None,
                            "reasoning_content": f"I have enough context. Applying the fix in {files}.",
                            "tool_calls": [{"type": "function", "id": f"gold-edit-{t}",
                                            "function": {"name": "run_command",
                                                         "arguments": json.dumps({"command": command})}}]}
                edit_result = {"role": "tool", "tool_call_id": f"gold-edit-{t}",
                               "content": json.dumps({"status": "ok", "stdout": "edited\n" if "python3 -" in command else "",
                                                      "stderr": "", "exit_code": 0})}
                submit_msg = {"role": "assistant", "content": None,
                              "reasoning_content": "The fix is in place. Submitting.",
                              "tool_calls": [{"type": "function", "id": f"gold-submit-{t}",
                                              "function": {"name": "submit_patch", "arguments": "{}"}}]}
                explore = [dict(c, train=False) for c in calls[:k]]
                edit_call = {"messages": calls[k]["messages"], "tools": calls[k]["tools"],
                             "response": {"choices": [{"message": edit_msg}]}, "train": True}
                submit_call = {"messages": calls[k]["messages"] + [edit_msg, edit_result], "tools": calls[k]["tools"],
                               "response": {"choices": [{"message": submit_msg}]}, "train": True}
                with open(OUT / "llm_calls" / f"{t}.jsonl", "w") as out:
                    for c in explore + [edit_call, submit_call]:
                        out.write(json.dumps(c) + "\n")
                summary.write(json.dumps({"id": t, "resolved": True, "source_run": run}) + "\n")
                done.add(t)
                stats["built"] += 1
    print(stats)


if __name__ == "__main__":
    main()
