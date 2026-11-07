"""Synthetic bug factory (SWE-smith style) for LoRA training data.

For each repo snapshot it proposes small, realistic code mutations inside library functions,
runs the repo's own test suite in the competition sandbox image (one docker container per repo,
see tools/factory_worker.py), and keeps mutations that make 1-5 previously passing tests fail.

Output: factory/bugs/<snapshot>.jsonl, one validated bug per line.

Usage:
  python3 tools/bug_factory.py --repo rich --max-valid 150 --hours 2
  (containers fx_<repo> must be running with /workspace prepared - see factory README in memory)
"""
import argparse
import ast
import difflib
import json
import random
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT.parent / "gemma-4-developer-agent" / "snapshots"
REPOS = {
    # repo key: (snapshot id, container, source package dirs, full repo name)
    "rich": ("rich_4075", "fx_rich", ["rich"], "Textualize/rich"),
    "requests": ("requests_7505", "fx_requests", ["src/requests"], "psf/requests"),
    "fastapi": ("fastapi_15800", "fx_fastapi", ["fastapi"], "fastapi/fastapi"),
}
MAX_FAILING = 5
MAX_PER_FILE = 12


# ---------------------------------------------------------------- source access
def host_tree(snapshot):
    """Extract the snapshot once on the host (read-only use: parsing and diffs)."""
    dest = ROOT / "factory" / "src" / snapshot
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(SNAPSHOTS / f"{snapshot}.tgz") as tf:
            tf.extractall(dest, filter="data")
    return dest


class Source:
    def __init__(self, text):
        self.text = text
        self.line_starts = [0]
        for line in text.splitlines(keepends=True):
            self.line_starts.append(self.line_starts[-1] + len(line))
        self.lines = text.splitlines(keepends=True)

    def offset(self, lineno, col_bytes):
        line = self.lines[lineno - 1] if lineno - 1 < len(self.lines) else ""
        col = len(line.encode("utf-8")[:col_bytes].decode("utf-8", errors="ignore"))
        return self.line_starts[lineno - 1] + col

    def span(self, node):
        return self.offset(node.lineno, node.col_offset), self.offset(node.end_lineno, node.end_col_offset)

    def seg(self, node):
        s, e = self.span(node)
        return self.text[s:e]


# ---------------------------------------------------------------- mutation operators
FLIP = {ast.Eq: "!=", ast.NotEq: "==", ast.Lt: "<=", ast.LtE: "<", ast.Gt: ">=", ast.GtE: ">",
        ast.In: "not in", ast.NotIn: "in", ast.Is: "is not", ast.IsNot: "is"}
SWAP_CALLS = {"min": "max", "max": "min", "any": "all", "all": "any"}
SIMPLE = (ast.Name, ast.Attribute, ast.Constant)


def _paren(src, node):
    text = src.seg(node)
    return f"({text})" if isinstance(node, (ast.BoolOp, ast.IfExp, ast.Lambda, ast.NamedExpr)) else text


def mutations_for_function(src, fn, qualname, relpath):
    out = []

    def add(node_or_span, new, kind):
        s, e = node_or_span if isinstance(node_or_span, tuple) else src.span(node_or_span)
        old = src.text[s:e]
        if old != new:
            out.append({"file": relpath, "start": s, "end": e, "new": new, "old": old, "kind": kind,
                        "func": qualname, "line": src.text.count("\n", 0, s) + 1})

    docstring = ast.get_docstring(fn, clean=False)
    for node in ast.walk(fn):
        if node is not fn and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue  # nested defs are handled separately
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in FLIP:
            add(node, f"{_paren(src, node.left)} {FLIP[type(node.ops[0])]} {_paren(src, node.comparators[0])}",
                "compare_flip")
        elif isinstance(node, ast.BoolOp) and len(node.values) == 2:
            op = " or " if isinstance(node.op, ast.And) else " and "
            add(node, op.join(_paren(src, v) for v in node.values), "boolop_swap")
        elif isinstance(node, ast.If):
            test = node.test
            text = src.seg(test)
            if "TYPE_CHECKING" in text or "__name__" in text:
                continue
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                add(test, src.seg(test.operand), "if_negate")
            elif isinstance(test, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
                add(test, f"not {text}", "if_negate")
            # comparisons are covered by compare_flip, boolean ops by boolop_swap
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in SWAP_CALLS:
                add(node.func, SWAP_CALLS[node.func.id], "call_swap")
            args = node.args
            if len(args) >= 2 and all(isinstance(a, SIMPLE) for a in args[:2]) and src.seg(args[0]) != src.seg(args[1]):
                s0, e0 = src.span(args[0])
                s1, e1 = src.span(args[1])
                between = src.text[e0:s1]
                add((s0, e1), src.seg(args[1]) + between + src.seg(args[0]), "arg_swap")
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                    add(kw.value, str(not kw.value.value), "kwarg_bool")
        elif isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 64:
            add(node, str(node.value + 1 if node.value in (0, 1) else node.value - 1), "const_off_by_one")
    # statement removals (missing step / missing validation)
    for node in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list) or len(stmts) < 2:
                continue
            if node is not fn and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for st in stmts:
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
                    continue  # docstring
                removable = (
                    (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call))
                    or isinstance(st, (ast.AugAssign, ast.Raise))
                    or (isinstance(st, ast.Assign) and any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in st.targets))
                )
                if removable:
                    # delete the whole line(s), like a forgotten line of code
                    s0, e0 = src.span(st)
                    line_start = src.line_starts[st.lineno - 1]
                    line_end = src.line_starts[st.end_lineno] if st.end_lineno < len(src.line_starts) else len(src.text)
                    if src.text[line_start:s0].strip() == "" and src.text[e0:line_end].strip() == "":
                        add((line_start, line_end), "", "remove_" + type(st).__name__.lower())
    # default argument bool flips
    for d in list(fn.args.defaults) + [d for d in fn.args.kw_defaults if d is not None]:
        if isinstance(d, ast.Constant) and isinstance(d.value, bool):
            add(d, str(not d.value), "default_bool")
    del docstring
    return out


def _line_span(src, first_lineno, last_lineno):
    """Char span covering whole lines first..last (inclusive, 1-based)."""
    start = src.line_starts[first_lineno - 1]
    end = src.line_starts[last_lineno] if last_lineno < len(src.line_starts) else len(src.text)
    return start, end


def round3_mutations_for_function(src, fn, qualname, relpath):
    """Harder, more realistic task types: missing keyword argument, missing branch, unimplemented feature."""
    out = []

    def add(span, new, kind, **extra):
        s, e = span
        old = src.text[s:e]
        if old != new:
            out.append({"file": relpath, "start": s, "end": e, "new": new, "old": old, "kind": kind,
                        "func": qualname, "line": src.text.count("\n", 0, s) + 1, **extra})

    body_nodes = [n for stmt in fn.body for n in ast.walk(stmt)]   # skip signature/annotations
    for node in body_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        # kwarg_drop: a call forgets one keyword argument
        if isinstance(node, ast.Call) and node.keywords:
            items = sorted(list(node.args) + list(node.keywords), key=lambda n: (n.lineno, n.col_offset))
            for i, kw in enumerate(items):
                if not isinstance(kw, ast.keyword) or kw.arg is None:
                    continue
                ks, ke = src.span(kw)
                if i > 0:
                    span = (src.span(items[i - 1])[1], ke)          # ", name=value"
                elif i + 1 < len(items):
                    span = (ks, src.span(items[i + 1])[0])          # "name=value, "
                else:
                    span = (ks, ke)                                 # only argument
                add(span, "", "kwarg_drop")
        # branch_drop: remove an `else:` block, or the last `elif` branch
        if isinstance(node, ast.If) and node.orelse:
            orelse = node.orelse
            if len(orelse) == 1 and isinstance(orelse[0], ast.If) and not orelse[0].orelse:
                elif_node = orelse[0]
                if src.lines[elif_node.lineno - 1].lstrip().startswith("elif"):
                    add(_line_span(src, elif_node.lineno, elif_node.end_lineno), "", "branch_drop")
            elif not isinstance(orelse[0], ast.If):
                else_line = orelse[0].lineno - 1
                while else_line > node.lineno and not src.lines[else_line - 1].lstrip().startswith("else"):
                    else_line -= 1
                if src.lines[else_line - 1].strip() == "else:":
                    add(_line_span(src, else_line, orelse[-1].end_lineno), "", "branch_drop")

    # body_stub: the function is not implemented yet (feature task)
    body = list(fn.body)
    doc = ast.get_docstring(fn)
    if doc is not None:
        body = body[1:]
    n_lines = (body[-1].end_lineno - body[0].lineno + 1) if body else 0
    if (len(body) >= 2 and 3 <= n_lines <= 30 and not fn.name.startswith("__")
            and not any(isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for b in body)):
        indent = src.lines[body[0].lineno - 1][: len(src.lines[body[0].lineno - 1]) - len(src.lines[body[0].lineno - 1].lstrip())]
        head = "".join(src.lines[min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1: body[0].lineno - 1])
        add(_line_span(src, body[0].lineno, body[-1].end_lineno), f"{indent}raise NotImplementedError\n",
            "body_stub", head=head)
    return out


def all_mutations(tree_dir, pkg_dirs, ops="round1"):
    muts = []
    for pkg in pkg_dirs:
        for path in sorted((tree_dir / pkg).rglob("*.py")):
            rel = str(path.relative_to(tree_dir))
            if "/tests/" in rel or "/_vendor/" in rel:
                continue
            text = path.read_text(encoding="utf-8")
            try:
                mod = ast.parse(text)
            except SyntaxError:
                continue
            src = Source(text)

            def visit(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.ClassDef):
                        visit(child, prefix + child.name + ".")
                    elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if child.end_lineno - child.lineno <= 120:
                            gen = round3_mutations_for_function if ops == "round3" else mutations_for_function
                            muts.extend(gen(src, child, prefix + child.name, rel))
                        visit(child, prefix + child.name + ".")
            visit(mod, "")
    return muts


# ---------------------------------------------------------------- running tests in the container
def worker(container, payload, timeout=900):
    res = subprocess.run(["docker", "exec", "-i", container, "python3", "/factory_worker.py"],
                         input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"error": "worker_failed: " + res.stderr[-500:], "failed": []}


def stable_baseline(container, cache):
    if cache.exists():
        return set(json.loads(cache.read_text()))
    runs = [worker(container, {"baseline": True}) for _ in range(2)]
    stable = set(runs[0]["passed"]) & set(runs[1]["passed"])
    cache.write_text(json.dumps(sorted(stable)))
    return stable


def unified(path, before, after):
    return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                        fromfile=f"a/{path}", tofile=f"b/{path}"))


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, choices=sorted(REPOS))
    ap.add_argument("--max-valid", type=int, default=150)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ops", choices=["round1", "round3"], default="round1")
    args = ap.parse_args()

    snapshot, container, pkgs, repo_name = REPOS[args.repo]
    tree = host_tree(snapshot)
    out_dir = ROOT / "factory" / "bugs"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.ops == "round1" else ".r3"
    bugs_path = out_dir / f"{snapshot}{suffix}.jsonl"
    tried_path = out_dir / f"{snapshot}{suffix}.tried.jsonl"
    id_prefix = "synth" if args.ops == "round1" else "synth3"

    stable = stable_baseline(container, out_dir / f"{snapshot}.baseline.json")
    subprocess.run(["docker", "exec", "-i", container, "bash", "-c", "cat > /tmp/stable_ids.txt"],
                   input="\n".join(sorted(stable)) + "\n", text=True, check=True)
    print(f"{args.repo}: {len(stable)} stable passing tests (only these are run per mutation)", flush=True)

    muts = all_mutations(tree, pkgs, args.ops)
    random.Random(args.seed).shuffle(muts)
    key = lambda m: f"{m['file']}:{m['start']}:{m['end']}:{m['new']}"
    tried = {json.loads(l)["key"] for l in open(tried_path)} if tried_path.exists() else set()
    valid = [json.loads(l) for l in open(bugs_path)] if bugs_path.exists() else []
    funcs_done = {(b["file"], b["func"]) for b in valid}
    per_file = {}
    for b in valid:
        per_file[b["file"]] = per_file.get(b["file"], 0) + 1
    print(f"{len(muts)} candidate mutations, {len(tried)} already tried, {len(valid)} valid so far", flush=True)

    deadline = time.time() + args.hours * 3600
    n = 0
    for m in muts:
        if len(valid) >= args.max_valid or time.time() > deadline:
            break
        k = key(m)
        if k in tried or (m["file"], m["func"]) in funcs_done or per_file.get(m["file"], 0) >= MAX_PER_FILE:
            continue
        limit = 8 if m["kind"] == "body_stub" else MAX_FAILING
        res = worker(container, {**{x: m[x] for x in ("file", "start", "end", "new")}, "maxfail": limit + 9})
        failing = sorted(set(res.get("failed") or []) & stable)
        # some repos have a few order-dependent tests that fail at random in full runs, so allow
        # extra failures here and keep only the ones confirmed below
        ok = not res.get("error") and 1 <= len(failing) <= limit + 6
        if ok:
            # confirm: re-run the failing tests on their own with the mutation; only tests that fail
            # again are caused by the bug (flaky, order-dependent tests pass in isolation)
            again = worker(container, {**{x: m[x] for x in ("file", "start", "end", "new")}, "maxfail": 0,
                                       "only_ids": failing})
            failing = sorted(set(failing) & set(again.get("failed") or []))
            ok = not again.get("error") and 1 <= len(failing) <= limit
        tried.add(k)
        with open(tried_path, "a") as f:
            f.write(json.dumps({"key": k, "kind": m["kind"], "n_fail": len(failing), "error": res.get("error"),
                                "sec": round(res.get("seconds", 0), 1)}) + "\n")
        n += 1
        if ok:
            text = (tree / m["file"]).read_text(encoding="utf-8")
            buggy = text[:m["start"]] + m["new"] + text[m["end"]:]
            bug = {
                "id": f"{id_prefix}_{args.repo}_{len(valid):04d}", "repo": repo_name, "snapshot": snapshot,
                "file": m["file"], "func": m["func"], "line": m["line"], "kind": m["kind"],
                "old": m["old"], "new": m["new"],
                "bug_diff": unified(m["file"], text, buggy),   # apply to create the buggy repo
                "fix_diff": unified(m["file"], buggy, text),   # reference solution
                "failing_tests": failing,
                "messages": {t: res["messages"].get(t, "")[-2000:] for t in failing},
                "head": m.get("head", ""),   # function signature + docstring (feature tasks)
            }
            valid.append(bug)
            funcs_done.add((m["file"], m["func"]))
            per_file[m["file"]] = per_file.get(m["file"], 0) + 1
            with open(bugs_path, "a") as f:
                f.write(json.dumps(bug) + "\n")
        if n % 10 == 0 or ok:
            print(f"[{time.strftime('%H:%M:%S')}] tried {n} | valid {len(valid)} | last: {m['kind']} "
                  f"{m['func']} fails={len(failing)} {res.get('seconds', 0):.0f}s{' VALID' if ok else ''}", flush=True)
    print(f"done: {len(valid)} valid bugs in {bugs_path}")


if __name__ == "__main__":
    main()
