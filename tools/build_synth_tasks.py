"""Turn validated synthetic bugs (factory/bugs/*.jsonl) into competition-style tasks.

For every bug the failing tests are HIDDEN from the agent (removed from the repo it sees). At grading
time `test_patch` adds a NEW test file holding just those tests (plus the imports/fixtures/helpers of
their original module), so other tests that cannot run in the grading sandbox never affect the result.

Output: factory/synth_tasks.jsonl, one task per line:
  instance_id, repo, base_commit, snapshot (original competition snapshot to start from),
  setup_diff (apply to the snapshot to get the agent's repo: bug + hidden tests),
  patch (reference fix), test_patch, hidden_tests, issue_context (for the issue writer),
  problem_statement ("" - written later by Gemma on Kaggle), hints_text, created_at
"""
import ast
import difflib
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACTORY = ROOT / "factory"


def unified(path, before, after):
    return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                        fromfile=f"a/{path}", tofile=f"b/{path}"))


def find_test_node(module, parts):
    """parts = ['TestClass', 'test_name'] or ['test_name'] (parametrisation stripped)."""
    body, node = module.body, None
    for i, name in enumerate(parts):
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == name), None)
        if node is None:
            return None, None
        if i < len(parts) - 1:
            body = node.body
    parent_body = body
    return node, parent_body


def hide_tests(text, names):
    """Remove the given test functions (and their decorators); None if the result is not valid Python
    (e.g. a test class would become empty)."""
    module = ast.parse(text)
    lines = text.splitlines(keepends=True)
    drop = set()
    for parts in names:
        node, parent_body = find_test_node(module, parts)
        if node is None:
            return None, []
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        drop.update(range(start - 1, node.end_lineno))
    new = "".join(line for i, line in enumerate(lines) if i not in drop)
    try:
        ast.parse(new)
    except SyntaxError:
        return None, []
    return new, names


def isolate_tests(text, names):
    """Module text keeping imports/fixtures/helpers but only the given tests (others removed)."""
    module = ast.parse(text)
    lines = text.splitlines(keepends=True)
    keep_funcs = {p[0] for p in names if len(p) == 1}
    keep_methods = defaultdict(set)
    for p in names:
        if len(p) == 2:
            keep_methods[p[0]].add(p[1])
    drop = set()

    def span(node):
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        return range(start - 1, node.end_lineno)

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            if node.name not in keep_funcs:
                drop.update(span(node))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            if node.name not in keep_methods:
                drop.update(span(node))
            else:
                for m in node.body:
                    if (isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test")
                            and m.name not in keep_methods[node.name]):
                        drop.update(span(m))
    new = "".join(line for i, line in enumerate(lines) if i not in drop)
    try:
        ast.parse(new)
    except SyntaxError:
        return None
    return new


def isolated_path(path, task_id):
    p = Path(path)
    stem = p.stem[5:] if p.stem.startswith("test_") else p.stem
    tag = hashlib.sha1((task_id + path).encode()).hexdigest()[:6]
    return str(p.with_name(f"test_synth_{stem}_{tag}.py"))


def main():
    tasks_meta = {json.loads(l)["instance_id"]: json.loads(l)
                  for l in open(ROOT.parent / "gemma-4-developer-agent" / "tasks.jsonl")}
    import sys
    rnd = "r3" if "--round3" in sys.argv else "r1"
    out_path = FACTORY / ("synth_tasks.jsonl" if rnd == "r1" else "synth_tasks_r3.jsonl")
    stats = defaultdict(int)
    with open(out_path, "w") as out:
        for bugs_file in sorted((FACTORY / "bugs").glob("*.jsonl")):
            if bugs_file.name.endswith(".tried.jsonl") or (".r3." in bugs_file.name) != (rnd == "r3"):
                continue
            for line in open(bugs_file):
                bug = json.loads(line)
                snapshot = bug["snapshot"]
                tree = FACTORY / "src" / snapshot
                created = tasks_meta[snapshot]["created_at"]
                stats["bugs"] += 1
                if any("test_testing_dependencies/" in t for t in bug["failing_tests"]):
                    stats["skip_flaky_test_group"] += 1    # order-dependent fastapi tests
                    continue
                by_file = defaultdict(list)
                for nodeid in bug["failing_tests"]:
                    path, *rest = nodeid.split("::")
                    rest[-1] = rest[-1].split("[")[0]      # drop parametrisation
                    if tuple(rest) not in by_file[path]:
                        by_file[path].append(tuple(rest))
                hide_diff, test_patch, ok = "", "", True
                test_sources = []
                renamed = {}
                for path, names in by_file.items():
                    original = (tree / path).read_text(encoding="utf-8")
                    hidden, _ = hide_tests(original, names)
                    isolated = isolate_tests(original, names)
                    if hidden is None or isolated is None:
                        ok = False
                        break
                    hide_diff += unified(path, original, hidden)
                    # grading runs ONLY a new file with the hidden tests, so unrelated tests that are
                    # broken in the grading sandbox (network fixtures, flaky ordering) cannot fail the task
                    new_path = isolated_path(path, bug["id"])
                    renamed[path] = new_path
                    test_patch += "".join(difflib.unified_diff(
                        [], isolated.splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{new_path}"))
                    module = ast.parse(original)
                    for parts in names:
                        node, _ = find_test_node(module, list(parts))
                        if node is not None:
                            test_sources.append(ast.get_source_segment(original, node))
                if not ok:
                    stats["skip_hide_failed"] += 1
                    continue
                messages = "\n\n".join(f"{t}:\n{m[-1200:]}" for t, m in list(bug["messages"].items())[:2])
                fake_commit = hashlib.sha1(bug["id"].encode()).hexdigest()  # no code graph for synthetic repos
                task = {
                    "instance_id": bug["id"], "repo": bug["repo"], "base_commit": fake_commit,
                    "snapshot": snapshot, "setup_diff": bug["bug_diff"] + hide_diff,
                    "patch": bug["fix_diff"], "test_patch": test_patch,
                    "hidden_tests": [renamed[t.split("::")[0]] + "::" + t.split("::", 1)[1] for t in bug["failing_tests"]],
                    "original_tests": bug["failing_tests"], "kind": bug["kind"], "func": bug["func"],
                    "issue_context": {"tests": "\n\n".join(test_sources)[:6000], "errors": messages[:3000],
                                      "file": bug["file"], "func": bug["func"], "head": bug.get("head", "")},
                    "problem_statement": "", "hints_text": "", "created_at": created,
                }
                out.write(json.dumps(task) + "\n")
                stats["tasks"] += 1
    print(dict(stats))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
