"""Build test-agent.ipynb from the starter notebook plus the local submission files."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['eval', 'collect', 'synth', 'paired'], default='eval',
                help='eval: the 20 held-out tasks; collect: all other training tasks (for LoRA data); '
                     'paired: the 20 held-out + first 30 other tasks (same tasks as runs v2 + real2v2)')
ap.add_argument('--run-name', default='run5')
ap.add_argument('--collect-start', type=int, default=0, help='collect mode: skip the first N non-eval tasks (already collected)')
ap.add_argument('--limit', type=int, default=0, help='only the first N tasks (quick checks)')
ap.add_argument('--sub-dir', default='submission', help='agent folder to embed (adapters are attached as a Kaggle input)')
ap.add_argument('--synth-start', type=int, default=0)
ap.add_argument('--synth-count', type=int, default=150)
ap.add_argument('--synth-expect-first', default='', help='synth mode: instance_id the first selected task must have (catches a wrong dataset)')
ap.add_argument('--stop-after-hours', type=float, default=10.5)
ap.add_argument('--ids', help='file with task ids to run (overrides the mode task list; real tasks only)')
ap.add_argument('--hints', help='JSON {task_id: hint}; each hint is appended to that task issue text (STaR-style collection)')
ap.add_argument('--adapter-sha', default='', help='first 16 hex chars of sha256(adapter_model.safetensors) the attached adapter must have')
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent.parent
starter = json.load(open(ROOT / "getting-started-gemma-4-developer-agent.ipynb"))
SUB = ROOT / args.sub_dir
FILES = sorted(str(f.relative_to(SUB)) for f in SUB.rglob("*") if f.is_file() and "adapters" not in f.relative_to(SUB).parts)
files = {f: (SUB / f).read_text() for f in FILES}

EVAL_TASK_IDS = ['fastapi_14356', 'fastapi_12942', 'fastapi_13713', 'fastapi_14964', 'fastapi_14962',
            'fastapi_14791', 'fastapi_9753', 'fastapi_14953', 'rich_3296', 'rich_3278', 'rich_4076',
            'rich_3454', 'rich_3894', 'rich_4079', 'rich_4006', 'requests_7309', 'requests_6592',
            'requests_7502', 'requests_7427', 'httpx_3672']

if args.mode == 'eval':
    TASK_IDS = EVAL_TASK_IDS[:args.limit] if args.limit else EVAL_TASK_IDS
elif args.mode == 'synth':
    TASK_IDS = []
elif args.mode == 'paired':
    all_ids = [json.loads(l)['instance_id'] for l in open(ROOT.parent / 'gemma-4-developer-agent' / 'tasks.jsonl')]
    TASK_IDS = EVAL_TASK_IDS + [i for i in all_ids if i not in set(EVAL_TASK_IDS)][:30]
else:
    all_ids = [json.loads(l)['instance_id'] for l in open(ROOT.parent / 'gemma-4-developer-agent' / 'tasks.jsonl')]
    TASK_IDS = [i for i in all_ids if i not in set(EVAL_TASK_IDS)][args.collect_start:]
if args.ids:
    TASK_IDS = open(args.ids).read().split()
HINTS = json.load(open(args.hints)) if args.hints else {}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def starter_src(i):
    return "".join(starter["cells"][i]["source"])


agent_cell = f'''import json
from swegemma.models import load_tasks

DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WORKING_DIR = Path('/kaggle/working')
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# ---- Experiment settings -------------------------------------------------
RUN_NAME = {args.run_name!r}
# Override the per-task time budget here (None = use eval_config.yaml)
MAX_TIME_MINUTES_OVERRIDE = None
# On GPUs without bfloat16 (T4), multiply the time budget by this factor
SLOW_GPU_TIME_SCALE = 2.0
TASK_IDS = {TASK_IDS!r}
# --------------------------------------------------------------------------

# Agent files under test (generated from the local submission/ folder)
AGENT_FILES = {json.dumps(files, indent=1)}

AGENT_DIR = WORKING_DIR / 'agent_under_test'
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR)
for rel, text in AGENT_FILES.items():
    p = AGENT_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')

# If agent.yaml uses LoRA adapters, copy them from an attached input (e.g. the training notebook's output).
import glob as _glob, re as _re
for _name in sorted(set(_re.findall(r'adapter:\\s*(\\w+)', ''.join(AGENT_FILES.values())))):
    _found = sorted(_glob.glob(f'/kaggle/input/**/adapters/{{_name}}/adapter_config.json', recursive=True))
    assert _found, f'agent.yaml needs adapter {{_name!r}}: attach the notebook/dataset that contains adapters/{{_name}}/'
    shutil.copytree(os.path.dirname(_found[-1]), AGENT_DIR / 'adapters' / _name)
    if {args.adapter_sha!r}:
        import hashlib as _hl
        _h = _hl.sha256(open(AGENT_DIR / 'adapters' / _name / 'adapter_model.safetensors', 'rb').read()).hexdigest()[:16]
        assert _h == {args.adapter_sha!r}, f'wrong adapter attached (sha {{_h}}, expected {args.adapter_sha}) - attach only the new adapter'
        print('adapter sha OK', _h)
    print('Adapter', _name, 'from', os.path.dirname(_found[-1]), sorted(os.listdir(AGENT_DIR / 'adapters' / _name)))

import glob as _glob
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
tasks = load_tasks(TASKS_PATH)
by_id = {{t.instance_id: t for t in tasks}}
SAMPLE_TASKS = [by_id[i] for i in TASK_IDS if i in by_id]
EVAL_TASKS_PATH = TASKS_PATH
HINTS = {HINTS!r}
if HINTS:   # STaR-style collection: append a hint from the reference fix to the issue text
    _hint_path = Path('/kaggle/working/hint_tasks.jsonl')
    with open(_hint_path, 'w') as _f:
        for _l in open(TASKS_PATH):
            _row = json.loads(_l)
            if _row['instance_id'] in HINTS:
                _row['problem_statement'] = _row['problem_statement'].rstrip() + HINTS[_row['instance_id']]
            _f.write(json.dumps(_row) + '\\n')
    EVAL_TASKS_PATH = _hint_path
    by_id = {{t.instance_id: t for t in load_tasks(_hint_path)}}
    SAMPLE_TASKS = [by_id[i] for i in TASK_IDS if i in by_id]
    print(len(HINTS), 'hints added; example:', SAMPLE_TASKS[0].problem_statement[-400:])
SNAPSHOTS_DIR = DATA_DIR / 'snapshots'
STOP_AFTER_HOURS = {args.stop_after_hours}   # end the task loop in time for packaging (Kaggle limit: 12 h)
SYNTH_ROWS = []


def prepare_snapshot(task):
    pass


def cleanup_snapshot(task):
    pass

if {args.mode == 'synth'}:
    _synth_files = sorted(_glob.glob('/kaggle/input/**/synth_tasks_valid.jsonl', recursive=True))
    assert _synth_files, 'Attach the dataset containing synth_tasks_valid.jsonl'
    SYNTH_ROWS = [json.loads(l) for l in open(_synth_files[-1])][{args.synth_start}:{args.synth_start + args.synth_count}]
    print(f'{{len(SYNTH_ROWS)}} synthetic tasks loaded from {{_synth_files[-1]}}')
    assert not {args.synth_expect_first!r} or SYNTH_ROWS[0]['instance_id'] == {args.synth_expect_first!r}, 'wrong synthetic dataset attached: ' + SYNTH_ROWS[0]['instance_id']
print(f'Agent files written to {{AGENT_DIR}}; testing {{len(SAMPLE_TASKS)}} tasks')

GRAPH_DIR = str(DATA_DIR / 'graphs')
EMBEDDINGS_DIR = str(DATA_DIR / 'embeddings')
'''

eval_cell = starter_src(10)
# Keep the starter's setup, replace its task loop with a logged one.
eval_cell = eval_cell[: eval_cell.find("# Select a small subset")]
eval_cell = eval_cell.replace(
    "max_time_minutes = float(eval_section.get('max_time_minutes', 60.0))",
    "max_time_minutes = float(MAX_TIME_MINUTES_OVERRIDE or eval_section.get('max_time_minutes', 60.0))\n"
    "# T4s are roughly 2x slower than the scoring L4s: scale time so the agent gets a similar number of turns\n"
    "if not BF16_OK and not MAX_TIME_MINUTES_OVERRIDE:\n"
    "    max_time_minutes *= SLOW_GPU_TIME_SCALE\n"
    "    timeout_seconds = int(timeout_seconds * SLOW_GPU_TIME_SCALE)",
)
eval_cell += '''limits, gen_constraints = build_submission_limits()
RESULTS_DIR = WORKING_DIR / 'results' / RUN_NAME

eval_config = EvalConfig(
    tasks_path=EVAL_TASKS_PATH,
    snapshots_dir=SNAPSHOTS_DIR,
    results_dir=RESULTS_DIR,
    submission_dir=AGENT_DIR,
    models=models,
    sandbox='subprocess',
    timeout_seconds=timeout_seconds,
{VERIFY_KW}    max_time_minutes=max_time_minutes,
    max_tool_calls=max_tool_calls,
    max_turns=max_turns,
    limits=limits,
    generation_constraints=gen_constraints,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=15,
        overlap_size=2,
        token_threshold=14336,
        event_retention_size=5,
    ),
    graph_dir=GRAPH_DIR,
    embeddings_dir=EMBEDDINGS_DIR,
    wheels_dir=DATA_DIR / 'wheels',
    verbose=False,
)
print(f'Budget: {max_time_minutes} min, {max_tool_calls} tool calls, {max_turns} turns, {timeout_seconds}s/command')

import json as _json
import time
import traceback

evaluator = Evaluator(eval_config)

# ---- Record every LLM request/response exactly as sent to vLLM (training data for LoRA) ----
import litellm
from litellm.integrations.custom_logger import CustomLogger

CURRENT_TASK = {'id': None}
CALLS_DIR = RESULTS_DIR / 'llm_calls'
CALLS_DIR.mkdir(parents=True, exist_ok=True)


def _plain(obj):
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


class CallRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def _record(self, kwargs, response_obj):
        try:
            call_id = kwargs.get('litellm_call_id')
            if call_id in self.seen or CURRENT_TASK['id'] is None:
                return
            self.seen.add(call_id)
            optional = kwargs.get('optional_params') or {}
            row = {
                'task': CURRENT_TASK['id'],
                'model': kwargs.get('model'),
                'messages': _plain(kwargs.get('messages')),
                'tools': _plain(optional.get('tools') or kwargs.get('tools')),
                'params': {k: v for k, v in optional.items() if k != 'tools'},
                'response': _plain(response_obj),
            }
            with open(CALLS_DIR / f"{CURRENT_TASK['id']}.jsonl", 'a') as f:
                f.write(_json.dumps(row, default=str) + '\\n')
        except Exception as e:  # never break the run because of logging
            print('CallRecorder error:', repr(e))

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj)


litellm.callbacks = list(litellm.callbacks or []) + [CallRecorder()]
rows = []
log_path = RESULTS_DIR / 'summary.jsonl'
log_path.parent.mkdir(parents=True, exist_ok=True)

LOOP_START = time.time()
for idx, task in enumerate(SAMPLE_TASKS, start=1):
    if (time.time() - NOTEBOOK_START) / 3600 > STOP_AFTER_HOURS:
        print(f'Stopping after {idx - 1} tasks to stay inside the Kaggle time limit')
        break
    print(f'\\n[{idx}/{len(SAMPLE_TASKS)}] {task.instance_id} ({task.repo})')
    CURRENT_TASK['id'] = task.instance_id
    t0 = time.time()
    try:
        prepare_snapshot(task)
        result = run_sync(evaluator.evaluate_task, task=task, task_index=idx, total_tasks=len(SAMPLE_TASKS))
        row = {
            'id': task.instance_id,
            'resolved': bool(result.resolved),
            'exit_code': result.test_exit_code,
            'patch_chars': len(result.agent_patch or ''),
            'tool_calls': result.tool_calls,
            'agent_seconds': round(result.duration_seconds or 0, 1),
            'wall_seconds': round(time.time() - t0, 1),
            'patch': result.agent_patch or '',
        }
        try:
            row['full_result'] = _json.loads(result.model_dump_json())
        except Exception:
            pass
    except Exception as e:
        row = {'id': task.instance_id, 'resolved': False, 'error': repr(e),
               'traceback': traceback.format_exc(), 'wall_seconds': round(time.time() - t0, 1)}
    cleanup_snapshot(task)
    rows.append(row)
    with open(log_path, 'a') as f:
        f.write(_json.dumps(row, default=str) + '\\n')
    if idx == 1:
        rec = CALLS_DIR / f"{task.instance_id}.jsonl"
        n = sum(1 for _ in open(rec)) if rec.exists() else 0
        print(f'  LLM calls recorded for first task: {n}')
        if n == 0:
            raise RuntimeError('LLM call recording failed - stopping early to save GPU quota. Check the notebook output above.')
    print(f"  -> resolved={row.get('resolved')} patch_chars={row.get('patch_chars')} "
          f"tool_calls={row.get('tool_calls')} wall={row['wall_seconds']}s")

df = pd.DataFrame(rows)
show = [c for c in ['id', 'resolved', 'patch_chars', 'tool_calls', 'agent_seconds', 'wall_seconds', 'error'] if c in df]
display(df[show])
print(f"\\nSolved {int(df['resolved'].sum())}/{len(df)} = {df['resolved'].mean():.2f}")
print(f"Empty patches: {int((df.get('patch_chars', 0) == 0).sum())}")
print(f"Mean wall time per task: {df['wall_seconds'].mean():.0f}s")
'''

synth_cell = (ROOT / 'tools' / 'nb_cells' / 'synth_cell.py').read_text()

package_cell = '''# Bundle everything (summary + harness trajectories) for download
archive = shutil.make_archive(str(WORKING_DIR / f'results_{RUN_NAME}'), 'zip', root_dir=WORKING_DIR / 'results')
print('Download this file from the Output panel:', archive)
'''

eval_cell = eval_cell.replace('{VERIFY_KW}', '    verification_timeout_seconds=verification_timeout_seconds,\n'
                              if 'verification_timeout_seconds' in starter_src(10) else '')
vllm_cell = starter_src(8)
# The harness wheelhouse gets updated: locate moved helpers instead of hard-coding module paths.
vllm_cell = vllm_cell.replace(
    "from swegemma.models.discovery import validate_single_declared_model\n",
    "import importlib.metadata as _md, pkgutil as _pkgutil, importlib as _importlib, swegemma as _swegemma\n"
    "print('swegemma', _md.version('swegemma'), '| adk-submission', _md.version('adk-submission'))\n"
    "def _find(name, pkg=_swegemma):\n"
    "    if hasattr(pkg, name):\n"
    "        return getattr(pkg, name)\n"
    "    for _m in _pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):\n"
    "        try:\n"
    "            _mod = _importlib.import_module(_m.name)\n"
    "        except Exception:\n"
    "            continue\n"
    "        if hasattr(_mod, name):\n"
    "            print(f'{name} found in {_m.name}')\n"
    "            return getattr(_mod, name)\n"
    "    return None\n"
    "import adk_submission as _adk\n"
    "validate_single_declared_model = _find('validate_single_declared_model') or _find('validate_single_declared_model', _adk)\n"
    "if validate_single_declared_model is None:\n"
    "    # removed in newer harness versions: same behaviour - the one model declared in the agent YAMLs\n"
    "    import re as _re\n"
    "    def validate_single_declared_model(agent_dir):\n"
    "        names = set()\n"
    "        for _f in Path(agent_dir).rglob('*.y*ml'):\n"
    "            names.update(m.strip().strip('\\'\"') for m in _re.findall(r'^\\s*model:\\s*(\\S+)', _f.read_text(), _re.M))\n"
    "        names.discard('')\n"
    "        assert len(names) == 1, f'expected exactly one declared model, found {names}'\n"
    "        print('declared model (local check):', next(iter(names)))\n"
    "        return next(iter(names))\n",
)
NEW_STARTER = "discover_declared_models" in vllm_cell   # Oct-2026+ starter: supports 31B/12B/E4B/E2B itself
if NEW_STARTER:
    vllm_cell = ("import re\nimport importlib.metadata as _md\nBF16_OK = True   # L4\n"
                 "print('swegemma', _md.version('swegemma'), '| adk-submission', _md.version('adk-submission'))\n" + vllm_cell)
else:
    assert "_find('validate_single_declared_model')" in vllm_cell
if not NEW_STARTER:   # old-starter patches (T4 support etc.); the new starter needs none
    # T4 (no bfloat16) support: fp16, no LoRA buffers, shorter context
    vllm_cell = vllm_cell.replace(
        "INFERENCE_API_KEY = 'EMPTY'",
        "INFERENCE_API_KEY = 'EMPTY'\n\n"
        "BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()\n"
        "MAX_MODEL_LEN = 32768 if BF16_OK else 24576\n"
        "print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'bf16:', BF16_OK)",
    )
    vllm_cell = vllm_cell.replace("import torch\n", "")
    vllm_cell = "import torch\n" + vllm_cell
    vllm_cell = vllm_cell.replace("max_model_len=32768,", "max_model_len=MAX_MODEL_LEN,")
    vllm_cell = vllm_cell.replace(
        "dtype='bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'auto',",
        "dtype='bfloat16' if BF16_OK else 'float16',",
    )
    vllm_cell = vllm_cell.replace("enable_lora=True,", "enable_lora=bool(adapters),")
if NEW_STARTER:
    assert "declared_model" in vllm_cell and "MODEL_VLLM_EXTRA_ARGS" in vllm_cell
    print('using the new starter vLLM cell (all four allowed models)')
else:
    assert "MAX_MODEL_LEN," in vllm_cell and "'float16'" in vllm_cell and "bool(adapters)" in vllm_cell

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": starter["metadata"],
    "cells": [
        md(("# Collect training trajectories\n\n" if args.mode == "collect" else "") + "# Test my agent on training tasks\n\nRuns the agent from `agent_under_test` on a fixed set of training tasks, "
           "verifies each patch, and saves a summary plus trajectories to `results_<RUN_NAME>.zip`.\n\n"
           "Use **Copy & Edit** on the starter notebook (so the competition data, wheelhouse and model inputs stay attached) "
           "and then **File → Import Notebook** with this file. Accelerator: **GPU L4 x4** if offered, otherwise **GPU T4 x2** (the notebook adapts automatically)."),
        md("## 0. Check inputs\nExpect the competition data, `metric/gemma-4-developer-agent-wheelhouse` and the `gemma-4-31b-it-qat-w4a16-ct` model."),
        code("import time\nNOTEBOOK_START = time.time()\nimport sys\nprint('Python', sys.version)\nassert sys.version_info[:2] == (3, 12), 'Wrong Kaggle environment: the competition wheels need Python 3.12. Settings -> Environment Preferences -> Pin to original environment.'\nimport subprocess\nassert subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True).stdout.count('GPU ') >= 2, 'No GPUs attached: Settings -> Accelerator -> GPU L4 x4 (it can reset after File -> Import Notebook)'\nprint(subprocess.run('find /kaggle/input -maxdepth 4 -not -path \"*/snapshots/*\" -not -path \"*/graphs/*\" -not -path \"*/embeddings/*\" -not -path \"*/wheels/*\" | head -60; nvidia-smi -L', shell=True, capture_output=True, text=True).stdout)"),
        md("## 1. Install packages (unchanged from starter)"),
        code(starter_src(2)),
        md("## 2. Agent under test and task list"),
        code(agent_cell),
        md("## 3. Start vLLM (unchanged from starter)"),
        code(vllm_cell),
        *([md("## 3b. Write issues for synthetic tasks and prepare snapshots"), code(synth_cell)] if args.mode == "synth" else []),
        md("## 4. Run and verify tasks"),
        code(eval_cell),
        md("## 5. Package results"),
        code(package_cell),
    ],
}
nb["metadata"].pop("papermill", None)
out = ROOT / ((f"test-agent-{args.run_name}.ipynb" if args.sub_dir != "submission" else "test-agent.ipynb") if args.mode in ("eval", "paired") else f"collect-{args.run_name}.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
