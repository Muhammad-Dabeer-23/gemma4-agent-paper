"""Turn recorded LLM calls from successful runs into LoRA training sequences.

Input:  results/<run>/summary.jsonl          (which tasks were resolved)
        results/<run>/llm_calls/<task>.jsonl  (every request sent to vLLM + its response)
Output: sft_data.jsonl, one line per training sequence:
        {"task", "text", "spans": [[start, end], ...]}
        `text` is rendered with the model's own chat template exactly as vLLM renders it,
        `spans` are character ranges the model generated (loss is applied only there).

Usage:
  python3 tools/build_sft_data.py --runs collect1 [--runs collect2 ...]
  python3 tools/build_sft_data.py --selftest run4     # synthetic calls rebuilt from traces
"""
import argparse
import copy
import json
from pathlib import Path

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "model_files" / "gemma-4-31b-it-qat-w4a16-ct__chat_template.jinja"
SYSTEM_MARKER = "You are an expert Python engineer"


# ---------------------------------------------------------------- rendering
def _make_env():
    # Same settings transformers uses for chat templates.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env


_TEMPLATE = _make_env().from_string(TEMPLATE.read_text())


def render(messages, tools, add_generation_prompt):
    # vLLM serves with default_chat_template_kwargs={'enable_thinking': True}.
    return _TEMPLATE.render(messages=messages, tools=tools, bos_token="<bos>",
                            add_generation_prompt=add_generation_prompt, enable_thinking=True)


# ---------------------------------------------------------------- normalising
def _parse_args(tool_calls):
    """vLLM json-decodes tool-call arguments before applying the HF template."""
    out = []
    for tc in tool_calls or []:
        tc = copy.deepcopy(tc)
        fn = tc.get("function") or {}
        if isinstance(fn.get("arguments"), str):
            try:
                fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
            except json.JSONDecodeError:
                pass
        tc["function"] = fn
        out.append(tc)
    return out


def normalize_messages(messages):
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if v is not None}
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m["tool_calls"] = _parse_args(m["tool_calls"])
        out.append(m)
    return out


def response_message(response):
    """The assistant message the model actually generated for one call."""
    msg = response["choices"][0]["message"]
    out = {"role": "assistant", "content": msg.get("content") or ""}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        out["reasoning_content"] = reasoning
    if msg.get("tool_calls"):
        out["tool_calls"] = _parse_args(msg["tool_calls"])
    return out


def _canon(m):
    """Message identity ignoring reasoning (history may or may not carry it)."""
    calls = [(tc["function"].get("name"), json.dumps(tc["function"].get("arguments"), sort_keys=True))
             for tc in m.get("tool_calls") or []]
    content = m.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return (m.get("role"), (content or "").strip(), tuple(calls), m.get("tool_call_id"))


# ---------------------------------------------------------------- building sequences
def is_agent_call(call):
    msgs = call.get("messages") or []
    if not msgs or not call.get("tools"):
        return False
    first = msgs[0]
    content = first.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return first.get("role") in ("system", "developer") and SYSTEM_MARKER in (content or "")


def chains(calls):
    """Group consecutive calls whose history extends the previous call (breaks on compaction)."""
    groups, cur = [], []
    for c in calls:
        msgs = normalize_messages(c["messages"])
        if cur:
            prev = cur[-1]
            prev_full = [_canon(m) for m in prev["msgs"]] + [_canon(response_message(prev["response"]))]
            mine = [_canon(m) for m in msgs]
            if len(mine) >= len(prev_full) and mine[:len(prev_full)] == prev_full:
                cur.append({"msgs": msgs, "response": c["response"], "tools": c["tools"], "train": c.get("train", True)})
                continue
            groups.append(cur)
            cur = []
        cur.append({"msgs": msgs, "response": c["response"], "tools": c["tools"], "train": c.get("train", True)})
    if cur:
        groups.append(cur)
    return groups


def build_sequence(chain, stats):
    last = chain[-1]
    tools = last["tools"]
    final = last["msgs"] + [response_message(last["response"])]
    # Put the exact generated message (with its reasoning) at every target position.
    positions = []
    for c in chain:
        i = len(c["msgs"])
        final[i] = response_message(c["response"])
        if c["train"]:      # gold demos: the agent's own exploration is context only, not a target
            positions.append(i)

    text = render(final, tools, add_generation_prompt=False)
    spans = []
    for i in positions:
        prompt = render(final[:i], tools, add_generation_prompt=True)
        upto = render(final[:i + 1], tools, add_generation_prompt=False)
        if not (text.startswith(prompt) and text.startswith(upto) and len(upto) > len(prompt)):
            stats["prefix_mismatch"] += 1
            continue
        end = len(upto)
        while end > len(prompt) and text[end - 1] == "\n":   # model stops at <turn|>, not the newline
            end -= 1
        spans.append([len(prompt), end])
    return {"text": text, "spans": spans}


def _changed_lines(patch):
    return sorted(l[1:].strip() for l in (patch or "").splitlines()
                  if l[:1] in "+-" and not l.startswith(("+++", "---")) and l[1:].strip())


def _calls_before_first_edit(calls):
    last = calls[-1]
    msgs = last["messages"] + [last["response"]["choices"][0]["message"]]
    n = 0
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n += 1
            args = tc["function"].get("arguments")
            args = args if isinstance(args, str) else json.dumps(args)
            if "write_text" in args or tc["function"]["name"] in ("edit_file", "write_file"):
                return n
    return 10 ** 6


def load_run(run_dir, reference_patches=None, stats=None, max_calls_before_edit=0):
    """Successful runs: graded `resolved`, or (synthetic tasks) a patch identical to the reference fix.

    The second case matters because some synthetic test files contain tests that cannot pass in the
    grading sandbox (e.g. network fixtures), so a correct fix can still be graded as failed.
    """
    summary = {json.loads(l)["id"]: json.loads(l) for l in open(run_dir / "summary.jsonl")}
    for task, row in summary.items():
        f = run_dir / "llm_calls" / f"{task}.jsonl"
        ok = bool(row.get("resolved"))
        if not ok and reference_patches and task in reference_patches:
            ok = _changed_lines(row.get("patch")) == _changed_lines(reference_patches[task]) != []
            if ok and stats is not None:
                stats["accepted_exact_reference_fix"] += 1
        if ok and f.exists():
            calls = [json.loads(l) for l in open(f)]
            if max_calls_before_edit and _calls_before_first_edit(calls) > max_calls_before_edit:
                if stats is not None:
                    stats["skipped_slow_to_edit"] += 1
                continue   # teach decisive behaviour: skip runs that explored too long before editing
            yield task, calls


# ---------------------------------------------------------------- self-test from traces
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "run_command", "description": "Executes a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Reads a file.",
     "parameters": {"type": "object", "properties": {"filepath": {"type": "string"},
                    "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {"name": "submit_patch", "description": "Submits the patch."}},
]


def synthetic_calls(trace_path):
    steps = json.load(open(trace_path))["steps"]
    system, user = steps[0]["message"], steps[1]["message"]
    history = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls = []
    for s in steps[2:]:
        if s.get("source") != "agent":
            continue
        tcs = [{"id": tc["tool_call_id"], "type": "function",
                "function": {"name": tc["function_name"], "arguments": json.dumps(tc["arguments"])}}
               for tc in s.get("tool_calls") or []]
        msg = {"role": "assistant", "content": None if tcs else s.get("message"), "reasoning_content": s.get("message")}
        if tcs:
            msg["tool_calls"] = tcs
        calls.append({"messages": copy.deepcopy(history), "tools": TOOL_SCHEMAS,
                      "response": {"choices": [{"message": msg}]}})
        # History as ADK would resend it: no reasoning (worst case for prefix matching).
        hist_msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
        history.append(hist_msg)
        for tc in tcs:
            history.append({"role": "tool", "tool_call_id": tc["id"],
                            "content": str((s.get("observation") or {}).get("content", ""))})
    return calls


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", default=[])
    ap.add_argument("--selftest", help="run name whose traces are turned into synthetic calls")
    ap.add_argument("--out", default=str(ROOT / "sft_data.jsonl"))
    ap.add_argument("--weight", action="append", default=[],
                    help="RUN=N: repeat that run's sequences N times (e.g. upweight real tasks)")
    ap.add_argument("--max-calls-before-edit", type=int, default=0,
                    help="drop successful runs whose first edit came after this many tool calls (0 = keep all)")
    ap.add_argument("--exclude", help="file with task ids to leave out (e.g. the paired test set)")
    ap.add_argument("--tasks", action="append", default=[],
                    help="synthetic task files; runs whose patch equals the reference fix count as successes")
    args = ap.parse_args()

    stats = {"tasks": 0, "calls": 0, "agent_calls": 0, "chains": 0, "sequences": 0, "spans": 0,
             "prefix_mismatch": 0, "chars": 0, "target_chars": 0, "accepted_exact_reference_fix": 0, "skipped_slow_to_edit": 0}
    sources = []
    if args.selftest:
        run_dir = ROOT / "results" / args.selftest
        summary = {json.loads(l)["id"]: json.loads(l) for l in open(run_dir / "summary.jsonl")}
        for task, row in summary.items():
            if row.get("resolved"):
                sources.append((task, synthetic_calls(run_dir / "traces" / f"trace_{task}.json")))
    for run in args.runs:
        refs = {}
        for tf in args.tasks:
            for l in open(tf):
                t = json.loads(l)
                refs[t["instance_id"]] = t["patch"]
        repeat = {w.split("=")[0]: int(w.split("=")[1]) for w in args.weight}.get(run, 1)
        run_sources = list(load_run(ROOT / "results" / run, refs, stats, args.max_calls_before_edit))
        if args.exclude:
            excluded = set(open(args.exclude).read().split())
            stats["excluded_test_tasks"] = stats.get("excluded_test_tasks", 0) + sum(t in excluded for t, _ in run_sources)
            run_sources = [(t, c) for t, c in run_sources if t not in excluded]
        stats.setdefault("tasks_per_run", {})[run] = len(run_sources)
        sources.extend(run_sources * repeat)

    with open(args.out, "w") as out:
        for task, calls in sources:
            stats["tasks"] += 1
            stats["calls"] += len(calls)
            agent = [c for c in calls if is_agent_call(c)]
            stats["agent_calls"] += len(agent)
            for chain in chains(agent):
                stats["chains"] += 1
                seq = build_sequence(chain, stats)
                if not seq["spans"]:
                    continue
                stats["sequences"] += 1
                stats["spans"] += len(seq["spans"])
                stats["chars"] += len(seq["text"])
                stats["target_chars"] += sum(e - s for s, e in seq["spans"])
                out.write(json.dumps({"task": task, **seq}, ensure_ascii=False) + "\n")
    stats["approx_tokens"] = int(stats["chars"] / 3.5)   # measured on Kaggle: chars/4 undercounted by ~13%
    print(json.dumps(stats, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
