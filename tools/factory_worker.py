"""Runs INSIDE a sandbox container. Applies one mutation, runs the tests, restores the file.

stdin:  {"file": "rich/console.py", "start": int, "end": int, "new": str, "only_ids": [...]?}   (char offsets)
        or {"baseline": true}
stdout: {"failed": [nodeid, ...], "messages": {nodeid: text}, "seconds": float, "error": str|null}
"""
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

WS = Path("/workspace")
JUNIT = "/tmp/fx_junit.xml"
STABLE_IDS = "/tmp/stable_ids.txt"


_PATHS = {}


def _resolve(file_attr):
    """junit `file` may be relative to a sub-rootdir; return the path relative to /workspace."""
    if file_attr not in _PATHS:
        if (WS / file_attr).exists():
            _PATHS[file_attr] = file_attr
        else:
            hits = sorted(str(p.relative_to(WS)) for p in (WS / "tests").rglob(Path(file_attr).name)
                          if str(p).endswith(file_attr))
            _PATHS[file_attr] = hits[0] if hits else file_attr
    return _PATHS[file_attr]


def run_tests(maxfail, only_ids=None):
    # After the baseline, run ONLY tests that passed at baseline (listed in STABLE_IDS), so broken
    # test files/fixtures in this environment cannot use up --maxfail. `only_ids` runs an explicit list.
    if only_ids:
        Path("/tmp/only_ids.txt").write_text("\n".join(only_ids) + "\n")
        target = ["@/tmp/only_ids.txt"]
    elif maxfail and Path(STABLE_IDS).exists():
        target = [f"@{STABLE_IDS}"]
    else:
        target = ["tests", "--continue-on-collection-errors"]
    cmd = [sys.executable, "-m", "pytest", *target, "-q", "-p", "no:cacheprovider", "-n", "4",
           "--timeout=20", f"--junitxml={JUNIT}", "-o", "junit_family=xunit1"]
    if maxfail:
        cmd.append(f"--maxfail={maxfail}")
    Path(JUNIT).unlink(missing_ok=True)
    try:
        subprocess.run(cmd, cwd=WS, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, {}, []
    failed, messages, passed = [], {}, []
    if not Path(JUNIT).exists():
        return None, {}, []
    for case in ET.parse(JUNIT).getroot().iter("testcase"):
        file_attr = case.get("file")
        name = case.get("name")
        parts = case.get("classname", "").split(".")
        if not file_attr:
            continue
        # classname = <module path>[.<Class>]; class parts follow the file's stem
        path = _resolve(file_attr)
        # classname = <dotted module path>[.<Class>...]; strip the module part using the file path
        cls_parts = []
        for mod in (path[:-3].replace("/", "."), file_attr[:-3].replace("/", ".")):
            if case.get("classname", "") == mod or case.get("classname", "").startswith(mod + "."):
                cls_parts = parts[len(mod.split(".")):]
                break
        else:
            stem = Path(file_attr).stem
            if stem in parts:
                cls_parts = parts[len(parts) - 1 - parts[::-1].index(stem) + 1:]
        nodeid = path + "".join("::" + c for c in cls_parts) + "::" + name
        bad = case.find("failure") if case.find("failure") is not None else case.find("error")
        if bad is not None:
            failed.append(nodeid)
            messages[nodeid] = ((bad.get("message") or "") + "\n" + (bad.text or ""))[-3000:]
        elif case.find("skipped") is None:
            passed.append(nodeid)
    return failed, messages, passed


def main():
    req = json.loads(sys.stdin.read())
    t0 = time.time()
    if req.get("baseline"):
        failed, _, passed = run_tests(0)
        print(json.dumps({"passed": passed, "failed": failed, "seconds": time.time() - t0}))
        return
    path = WS / req["file"]
    original = path.read_text()
    mutated = original[:req["start"]] + req["new"] + original[req["end"]:]
    try:
        compile(mutated, str(path), "exec")
    except SyntaxError as e:
        print(json.dumps({"failed": [], "messages": {}, "seconds": 0, "error": f"syntax: {e}"}))
        return
    path.write_text(mutated)
    try:
        failed, messages, _ = run_tests(req.get("maxfail", 8), req.get("only_ids"))
    finally:
        path.write_text(original)
    out = {"failed": failed or [], "messages": messages, "seconds": time.time() - t0,
           "error": "timeout_or_crash" if failed is None else None}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
