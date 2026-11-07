"""Gold-check synthetic tasks the way Phase 2 grading does, inside the sandbox containers.

For each task in factory/synth_tasks.jsonl:
  1. apply setup_diff (bug + hidden tests)      -> the repo the agent would see
  2. apply test_patch, run the hidden tests     -> must FAIL  (bug is real)
  3. apply patch (reference fix), run the hidden tests plus every baseline-stable test in the
     same files                                  -> must PASS  (task is solvable)
  Tests that are broken only in this local environment (missing optional deps) are ignored.
  4. revert everything
Writes factory/synth_tasks_valid.jsonl.

Run only when tools/bug_factory.py is not using the same container.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACTORY = ROOT / "factory"
CONTAINER = {"rich_4075": "fx_rich", "requests_7505": "fx_requests", "fastapi_15800": "fx_fastapi"}


def sh(container, cmd, stdin=None, timeout=600):
    return subprocess.run(["docker", "exec", "-i", container, "bash", "-c", cmd], input=stdin,
                          capture_output=True, text=True, timeout=timeout)


def git_apply(container, diff, reverse=False):
    flag = "-R " if reverse else ""
    return sh(container, f"cd /workspace && git apply {flag}--whitespace=nowarn -", stdin=diff).returncode == 0


def run_ids(container, ids):
    sh(container, "cat > /tmp/val_ids.txt", stdin="\n".join(ids) + "\n")
    r = sh(container, "cd /workspace && timeout 300 python3 -m pytest @/tmp/val_ids.txt -q -p no:cacheprovider "
                      "-n 4 --timeout=30 2>&1 | tail -3; exit ${PIPESTATUS[0]}")
    return r.returncode, r.stdout.strip().splitlines()[-1:] if r.stdout.strip() else []


STABLE = {}


def stable_ids(snapshot):
    if snapshot not in STABLE:
        STABLE[snapshot] = json.loads((FACTORY / "bugs" / f"{snapshot}.baseline.json").read_text())
    return STABLE[snapshot]


def main():
    import sys
    r3 = "--round3" in sys.argv
    tasks = [json.loads(l) for l in open(FACTORY / ("synth_tasks_r3.jsonl" if r3 else "synth_tasks.jsonl"))]
    out_path = FACTORY / ("synth_tasks_r3_valid.jsonl" if r3 else "synth_tasks_valid.jsonl")
    done = {json.loads(l)["instance_id"] for l in open(out_path)} if out_path.exists() else set()
    stats = {"ok": 0, "bad_apply": 0, "bug_passes": 0, "fix_fails": 0}
    with open(out_path, "a") as out:
        for i, t in enumerate(tasks):
            if t["instance_id"] in done:
                continue
            c = CONTAINER[t["snapshot"]]
            files = sorted(set(re.findall(r"^\+\+\+ b/(\S+)", t["test_patch"], re.M)))
            applied = []
            verdict = "bad_apply"
            try:
                for diff in (t["setup_diff"], t["test_patch"]):
                    if not git_apply(c, diff):
                        raise RuntimeError
                    applied.append(diff)
                hidden = sorted(set(t["hidden_tests"]))
                same_files = sorted({i for i in stable_ids(t["snapshot"]) if i.split("::")[0] in files} | set(hidden))
                rc_bug, _ = run_ids(c, hidden)
                if rc_bug == 0:
                    verdict = "bug_passes"
                else:
                    if not git_apply(c, t["patch"]):
                        raise RuntimeError
                    applied.append(t["patch"])
                    rc_fix, tail = run_ids(c, same_files)
                    verdict = "ok" if rc_fix == 0 else "fix_fails"
            except RuntimeError:
                pass
            finally:
                for diff in reversed(applied):
                    if not git_apply(c, diff, reverse=True):
                        raise SystemExit(f"could not revert {t['instance_id']} - container workspace is dirty")
            stats[verdict] += 1
            if verdict == "ok":
                out.write(json.dumps(t) + "\n")
                out.flush()
            print(f"[{i + 1}/{len(tasks)}] {t['instance_id']} {verdict}", flush=True)
    print(stats)


if __name__ == "__main__":
    main()
