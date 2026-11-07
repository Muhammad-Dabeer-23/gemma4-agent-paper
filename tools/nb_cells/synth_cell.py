# Notebook cell (synth mode): write issues for synthetic tasks with Gemma, then prepare task snapshots.
# Runs after the vLLM server is up; uses SYNTH_ROWS, server_instance, DATA_DIR, WORKING_DIR, RUN_NAME, load_tasks.
import concurrent.futures, random, re, shutil, subprocess, tarfile, tempfile
import requests as _rq

BASE_URL = server_instance.base_url
SERVED = _rq.get(BASE_URL + '/models', timeout=60).json()['data'][0]['id']

ISSUE_PROMPT = (
    "You are a user of the Python library {repo}. You ran into a bug. Write the GitHub issue you would file.\n\n"
    "What you observed (from your own code, shown here as a test):\n```python\n{tests}\n```\n"
    "Error output:\n```\n{errors}\n```\n\n"
    "Rules:\n"
    "- Describe the wrong behaviour and the expected behaviour, with a short reproduction snippet a user would write.\n"
    "- Do NOT mention tests, test names, pytest, line numbers, or how to fix it. Do not guess which internal code is wrong.\n"
    "- {style}\n"
    "Output only the issue text (title on the first line)."
)
STYLES = ['Write 2-4 short paragraphs.', 'Be brief: a title plus 3-6 sentences.',
          'Write it like a terse bug report from a busy developer.']
FEATURE_PROMPT = (
    "You maintain or use the Python library {repo}. Write a GitHub feature request / issue asking for the "
    "following function to be implemented (it currently raises NotImplementedError):\n```python\n{head}```\n"
    "Expected behaviour, shown by these checks:\n```python\n{tests}\n```\n\n"
    "Rules:\n- Describe what the function should do and give a short usage example with expected output.\n"
    "- Do NOT mention tests, test names, pytest, or NotImplementedError. Do not write the implementation.\n"
    "- {style}\nOutput only the issue text (title on the first line)."
)
TITLE_PROMPT = (
    "Here is a bug in the Python library {repo}, shown as a failing check:\n```python\n{tests}\n```\n```\n{errors}\n```\n"
    "Write ONLY a one-line pull-request title that would fix it, in the style \"Fix <what is wrong> when <situation>\". "
    "No test names, no code identifiers from the check other than public API names."
)


def chat(prompt, max_tokens):
    r = _rq.post(BASE_URL + '/chat/completions', timeout=600, json={
        'model': SERVED, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.8,
        'max_tokens': max_tokens, 'chat_template_kwargs': {'enable_thinking': False}})
    return (r.json()['choices'][0]['message'].get('content') or '').strip()


def write_issue(row):
    ctx = row['issue_context']
    rnd = random.Random(row['instance_id'])
    hidden_names = {t.split('::')[-1].split('[')[0] for t in row['hidden_tests']}
    text = ''
    for attempt in range(2):
        if row.get('kind') == 'body_stub' and ctx.get('head'):
            text = chat(FEATURE_PROMPT.format(repo=row['repo'], head=ctx['head'][:2500], tests=ctx['tests'][:3500],
                                              style=rnd.choice(STYLES)), 700)
        elif attempt == 0 and rnd.random() < 0.25:   # ~25% of real tasks are just a short title
            text = chat(TITLE_PROMPT.format(repo=row['repo'], tests=ctx['tests'][:3000], errors=ctx['errors'][:1500]), 60)
            text = text.splitlines()[0] if text else ''
        else:
            text = chat(ISSUE_PROMPT.format(repo=row['repo'], tests=ctx['tests'][:4000], errors=ctx['errors'][:2000],
                                            style=rnd.choice(STYLES)), 700)
        if text and not any(n in text for n in hidden_names):
            return text
    return text


t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
    statements = list(pool.map(write_issue, SYNTH_ROWS))
print(f'wrote {len(statements)} issues in {time.time() - t0:.0f}s')
for row, text in zip(SYNTH_ROWS, statements):
    row['problem_statement'] = text
RESULTS_DIR = WORKING_DIR / 'results' / RUN_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / 'synth_issues.jsonl', 'w') as f:
    for row in SYNTH_ROWS:
        f.write(json.dumps({'id': row['instance_id'], 'problem_statement': row['problem_statement']}) + '\n')
for row in SYNTH_ROWS[:2]:
    print('-----', row['instance_id'])
    print(row['problem_statement'][:800])

# Tasks file in the harness format
EVAL_TASKS_PATH = WORKING_DIR / 'synth_tasks_run.jsonl'
KEYS = ['instance_id', 'repo', 'base_commit', 'patch', 'test_patch', 'problem_statement', 'hints_text', 'created_at']
with open(EVAL_TASKS_PATH, 'w') as f:
    for row in SYNTH_ROWS:
        f.write(json.dumps({k: row[k] for k in KEYS}) + '\n')
SAMPLE_TASKS = load_tasks(EVAL_TASKS_PATH)

# Build each task's repo just before it runs (original snapshot + bug + hidden tests)
SNAPSHOTS_DIR = Path('/tmp/synth_snapshots')
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
ROW_BY_ID = {row['instance_id']: row for row in SYNTH_ROWS}


def prepare_snapshot(task):
    row = ROW_BY_ID[task.instance_id]
    build = Path(tempfile.mkdtemp(dir='/tmp'))
    with tarfile.open(DATA_DIR / 'snapshots' / (row['snapshot'] + '.tgz')) as tf:
        tf.extractall(build, filter='data')
    p = subprocess.run(['git', 'apply', '--whitespace=nowarn', '-'], input=row['setup_diff'], text=True,
                       cwd=build, capture_output=True)
    if p.returncode:
        raise RuntimeError('setup_diff failed: ' + p.stderr[-300:])
    # fold the change into the last commit so git status/diff look clean to the agent
    subprocess.run(['git', 'add', '-A'], cwd=build, check=True)
    subprocess.run(['git', '-c', 'user.email=dev@example.com', '-c', 'user.name=dev', 'commit', '-q', '--amend',
                    '--no-edit'], cwd=build, check=True)
    with tarfile.open(SNAPSHOTS_DIR / (task.instance_id + '.tgz'), 'w:gz') as tf:
        tf.add(build, arcname='.')
    shutil.rmtree(build)


def cleanup_snapshot(task):
    (SNAPSHOTS_DIR / (task.instance_id + '.tgz')).unlink(missing_ok=True)


print(len(SAMPLE_TASKS), 'synthetic tasks ready')
