"""Build train-lora.ipynb: QLoRA fine-tuning of Gemma 4 31B on our own successful trajectories.

The training data (sft_data.jsonl from tools/build_sft_data.py) is embedded in the notebook,
so no dataset upload is needed. Inputs to attach on Kaggle: the competition wheelhouse dataset
and the gemma-4-31b-it-qat-q4_0-unquantized model (Transformers). No internet needed: LoRA is
implemented in plain PyTorch and saved in PEFT format. Works on Python 3.12 or 3.13.

Usage: python3 tools/make_train_nb.py [--data sft_data.jsonl] [--out train-lora.ipynb]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
starter = json.load(open(ROOT / "getting-started-gemma-4-developer-agent.ipynb"))

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "sft_data.jsonl"))
ap.add_argument("--out", default=str(ROOT / "train-lora.ipynb"))
args = ap.parse_args()
rows = [json.loads(l) for l in open(args.data)]
import base64, gzip
sft_b64 = base64.b64encode(gzip.compress(open(args.data, 'rb').read(), 9)).decode('ascii')
EMBED = len(sft_b64) < 850_000   # Kaggle notebooks must stay under 1 MB; otherwise read from an attached dataset
if not EMBED:
    sft_b64 = ''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


env_cell = '''import os, sys, subprocess
DIAGNOSE = False  # True = synchronous CUDA errors (exact failing op) but ~7x slower model loading
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   # avoid fragmentation OOMs
if DIAGNOSE:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print('Python', sys.version)
gpus = subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True).stdout
print(gpus or 'NO GPU')
assert gpus.count('GPU ') >= 2, 'No GPUs attached: Settings -> Accelerator -> GPU L4 x4 (it can reset after File -> Import Notebook)'
print(subprocess.run('ls /kaggle/input/models/google/gemma-4/*/', shell=True, capture_output=True, text=True).stdout)
'''

# Training needs only these wheelhouse packages; all are pure-Python or abi3 wheels, so they install
# offline on Python 3.12 or 3.13 (the vLLM wheels, which are 3.12-only, are not needed here).
install_cell = '''import glob, importlib, os, subprocess, sys
from pathlib import Path

WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
NEEDED = ['transformers', 'tokenizers', 'safetensors', 'bitsandbytes']
wheels = []
for name in NEEDED:
    found = sorted(WHEELHOUSE_DIR.glob(f'{name}-*.whl'))
    assert found, f'{name} wheel not found in {WHEELHOUSE_DIR} - attach the wheelhouse dataset'
    wheels.append(str(found[-1]))
print('Installing:', [os.path.basename(w) for w in wheels])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()

import transformers, bitsandbytes, torch
try:
    import accelerate  # needed for device_map='auto'
except ImportError as e:
    raise SystemExit('accelerate is missing in this Kaggle image - check the Kaggle environment') from e
from transformers import Gemma4ForConditionalGeneration  # fails early if this transformers lacks Gemma 4
print('transformers', transformers.__version__, '| bitsandbytes', bitsandbytes.__version__,
      '| accelerate', accelerate.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda)
'''

config_cell = f'''import glob, json, math, os, time
SFT_B64 = {sft_b64!r}  # gzip+base64 of sft_data.jsonl (Kaggle limits notebooks to 1 MB)
from pathlib import Path
import torch

# ---- Settings ---------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0     # dropout would store a copy of every layer input (memory)
# Language-model modules only. k_proj/v_proj are skipped on purpose: full-attention
# layers share K and V (attention_k_eq_v), the riskiest case for vLLM LoRA loading.
TARGET_MODULES = ['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
EPOCHS = 1
LR = 1e-4
GRAD_ACCUM = 4          # sequences per optimizer step
MAX_TOKENS = 14336      # measured on 4x L4: 12k fits, 16k OOMs; longer sequences are cut after the last target
TIME_LIMIT_HOURS = 5.2  # stop training early and still save the adapter (quota guard)
SMOKE_TEST_ONLY = False # True: run one forward/backward step and stop
ADAPTER_NAME = 'main_lora'
# -----------------------------------------------------------------------------

candidates = sorted(glob.glob('/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/*/config.json'))
assert candidates, 'Attach the gemma-4-31b-it-qat-q4_0-unquantized model (Transformers) as an input.'
MODEL_DIR = os.path.dirname(candidates[-1])
print('Base model:', MODEL_DIR)

import base64, gzip
if SFT_B64:
    SFT_ROWS = [json.loads(l) for l in gzip.decompress(base64.b64decode(SFT_B64)).decode('utf-8').splitlines() if l.strip()]
else:
    _found = sorted(glob.glob('/kaggle/input/**/sft_data.jsonl', recursive=True))
    assert _found, 'Attach the private dataset that contains sft_data.jsonl'
    SFT_ROWS = [json.loads(l) for l in open(_found[-1]) if l.strip()]
    print('training data from', _found[-1])
    assert len(SFT_ROWS) == {len(rows)}, f'expected {len(rows)} sequences, got {{len(SFT_ROWS)}} - wrong sft_data version attached?'
print(len(SFT_ROWS), 'training sequences from', len({{r['task'] for r in SFT_ROWS}}), 'tasks')
'''

tokenize_cell = '''from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
examples = []
for r in SFT_ROWS:
    enc = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    spans = r['spans']
    is_target = [False] * len(ids)
    j = 0
    for t, (a, b) in enumerate(offs):
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        if j < len(spans) and a >= spans[j][0] and b <= spans[j][1] and b > a:
            is_target[t] = True
    # cut overly long sequences at the end of the last target that fits
    if len(ids) > MAX_TOKENS:
        last = max((t for t in range(MAX_TOKENS) if is_target[t]), default=-1)
        ids, is_target = ids[:last + 1], is_target[:last + 1]
    n_t = sum(is_target)
    if n_t:
        examples.append({'task': r['task'], 'ids': ids, 'target': is_target})

assert examples and examples[0]['ids'][0] == tok.bos_token_id, 'text must start with <bos>'
lens = sorted(len(e['ids']) for e in examples)
print(f"{len(examples)} sequences | tokens: total {sum(lens)}, median {lens[len(lens)//2]}, max {lens[-1]} | "
      f"target tokens {sum(sum(e['target']) for e in examples)}")

# Sanity check: show where targets start and end in the first example.
e = examples[0]
starts = [t for t in range(1, len(e['ids'])) if e['target'][t] and not e['target'][t - 1]]
for s in starts[:2]:
    end = s
    while end < len(e['ids']) and e['target'][end]:
        end += 1
    print('--- context:', repr(tok.decode(e['ids'][max(0, s - 15):s])))
    print('    target :', repr(tok.decode(e['ids'][s:min(end, s + 40)])), '...', repr(tok.decode(e['ids'][max(s, end - 12):end])))
'''

model_cell = '''import re
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, Gemma4ForConditionalGeneration

n_gpu = torch.cuda.device_count()
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                         llm_int8_skip_modules=['vision_tower', 'audio_tower', 'embed_vision', 'embed_audio', 'lm_head'])
t0 = time.time()
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_DIR, quantization_config=bnb, dtype=torch.bfloat16, device_map='auto',
    max_memory={i: '18GiB' for i in range(n_gpu)}, attn_implementation='sdpa')
print(f'loaded in {time.time() - t0:.0f}s')
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)


class LoRALinear(nn.Module):
    """y = base(x) + B(A(dropout(x))) * alpha/r ; A, B trained in fp32, applied in the input dtype."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        dev = next(base.parameters()).device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        lora = F.linear(F.linear(self.drop(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + lora * self.scale


pattern = re.compile(r'.*language_model\\..*\\.(' + '|'.join(TARGET_MODULES) + r')$')
LORA_MODULES = {}
for name, module in list(model.named_modules()):
    if pattern.match(name) and hasattr(module, 'in_features'):
        parent_name, child = name.rsplit('.', 1)
        wrapped = LoRALinear(module, LORA_R, LORA_ALPHA, LORA_DROPOUT)
        setattr(model.get_submodule(parent_name), child, wrapped)
        LORA_MODULES[name] = wrapped
n_lora = sum(m.lora_A.numel() + m.lora_B.numel() for m in LORA_MODULES.values())
print(f'LoRA on {len(LORA_MODULES)} modules, {n_lora / 1e6:.1f}M trainable parameters; e.g. {next(iter(LORA_MODULES))}')
assert LORA_MODULES, 'no target modules found - check module names'

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.enable_input_require_grads()
text_model = model.model.language_model  # decoder only (no vision inputs needed)
lm_head = model.lm_head
SOFTCAP = model.config.text_config.final_logit_softcapping
first_device = next(text_model.parameters()).device
for i in range(n_gpu):
    print(f'GPU{i}: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB')
'''

loss_cell = '''import torch.nn.functional as F


def sequence_loss(example, chunk=1024):
    # The model is split over 4 GPUs: input ids live on the first one, the final hidden states on the
    # last one, and lm_head (tied to the embeddings) on whichever holds it. Keep every index tensor on
    # the device of the tensor it indexes.
    n = len(example['ids'])
    padded = n + (-n) % 64          # pad to a multiple of 64 (kernel alignment); pads are masked out
    ids = torch.full((1, padded), tok.pad_token_id, dtype=torch.long, device=first_device)
    ids[0, :n] = torch.tensor(example['ids'], device=first_device)
    attn = torch.zeros_like(ids)
    attn[0, :n] = 1
    hidden = text_model(input_ids=ids, attention_mask=attn).last_hidden_state[0]
    tgt = torch.zeros(padded, dtype=torch.bool, device=ids.device)
    tgt[:n] = torch.tensor(example['target'], device=ids.device)
    pos = torch.nonzero(tgt[1:]).squeeze(-1)          # position p predicts token p+1
    labels = ids[0, pos + 1].to(lm_head.weight.device)
    pos_h = pos.to(hidden.device)
    total = 0.0
    for c in range(0, len(pos_h), chunk):
        h = hidden[pos_h[c:c + chunk]].to(lm_head.weight.device)
        logits = lm_head(h).float()
        if SOFTCAP:
            logits = torch.tanh(logits / SOFTCAP) * SOFTCAP
        total = total + F.cross_entropy(logits, labels[c:c + chunk], reduction='sum')
    return total / len(pos_h)


# Smoke test: one forward/backward on the shortest sequence.
model.train()
short = min(examples, key=lambda e: len(e['ids']))
t0 = time.time()
loss = sequence_loss(short)
loss.backward()
model.zero_grad(set_to_none=True)
print(f'smoke test: {len(short["ids"])} tokens, loss {loss.item():.3f}, {time.time() - t0:.0f}s')
for i in range(n_gpu):
    print(f'GPU{i}: peak {torch.cuda.max_memory_allocated(i) / 2**30:.1f} GiB')

# Length ladder: forward+backward at increasing lengths, so a failure shows the exact limit.
longest = max(examples, key=lambda e: len(e['ids']))
for L in [8192, min(MAX_TOKENS, len(longest['ids']))]:
    L = min(L, len(longest['ids']))
    cut = {'ids': longest['ids'][:L], 'target': longest['target'][:L]}
    if not any(cut['target'][1:]):
        continue
    t0 = time.time()
    print(f'ladder {L} tokens ...', end=' ', flush=True)
    loss = sequence_loss(cut)
    loss.backward()
    for i in range(n_gpu):
        torch.cuda.synchronize(i)
    model.zero_grad(set_to_none=True)
    peaks = ', '.join(f'{torch.cuda.max_memory_allocated(i) / 2**30:.1f}' for i in range(n_gpu))
    print(f'OK loss {loss.item():.3f} in {time.time() - t0:.0f}s, peak GiB per GPU: {peaks}', flush=True)
'''

train_cell = '''import random

if SMOKE_TEST_ONLY:
    raise SystemExit('SMOKE_TEST_ONLY=True - stopping before training')

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
steps_total = math.ceil(len(examples) * EPOCHS / GRAD_ACCUM)
warmup = max(1, steps_total // 10)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))

random.seed(0)
start, step, seen = time.time(), 0, 0
log = []
stop = False
for epoch in range(EPOCHS):
    order = examples[:]
    random.shuffle(order)
    for k, ex in enumerate(order):
        try:
            loss = sequence_loss(ex) / GRAD_ACCUM
            loss.backward()
            loss_value = loss.item() * GRAD_ACCUM
        except torch.OutOfMemoryError:
            loss = None
            opt.zero_grad(set_to_none=True)   # drop partial grads of this accumulation group
            import gc; gc.collect(); torch.cuda.empty_cache()
            print(f'  OOM on {ex["task"]} ({len(ex["ids"])} tokens) - skipped', flush=True)
            continue
        del loss
        seen += 1
        log.append({'epoch': epoch, 'task': ex['task'], 'tokens': len(ex['ids']), 'loss': loss_value})
        if seen % GRAD_ACCUM == 0 or k == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            recent = sum(x['loss'] for x in log[-GRAD_ACCUM:]) / len(log[-GRAD_ACCUM:])
            print(f'epoch {epoch} step {step}/{steps_total} loss {recent:.4f} lr {sched.get_last_lr()[0]:.2e} '
                  f'elapsed {(time.time() - start) / 60:.1f} min', flush=True)
        if (time.time() - start) / 3600 > TIME_LIMIT_HOURS:
            print('time limit reached - saving what we have'); stop = True; break
    if stop:
        break
'''

save_cell = '''import shutil
from safetensors.torch import save_file

out_dir = Path('/kaggle/working/adapters') / ADAPTER_NAME
out_dir.mkdir(parents=True, exist_ok=True)
tensors = {}
for name, m in LORA_MODULES.items():
    key = 'base_model.model.' + name   # PEFT naming, e.g. base_model.model.model.language_model.layers.0.self_attn.q_proj
    tensors[key + '.lora_A.weight'] = m.lora_A.detach().to(torch.bfloat16).cpu().contiguous()
    tensors[key + '.lora_B.weight'] = m.lora_B.detach().to(torch.bfloat16).cpu().contiguous()
save_file(tensors, str(out_dir / 'adapter_model.safetensors'), metadata={'format': 'pt'})
cfg = {
    'alpha_pattern': {}, 'auto_mapping': None, 'base_model_name_or_path': 'google/gemma-4-31b-it-qat-w4a16-ct',
    'bias': 'none', 'fan_in_fan_out': False, 'inference_mode': True, 'init_lora_weights': True,
    'layer_replication': None, 'layers_pattern': None, 'layers_to_transform': None, 'loftq_config': {},
    'lora_alpha': LORA_ALPHA, 'lora_dropout': LORA_DROPOUT, 'megatron_config': None, 'megatron_core': 'megatron.core',
    'modules_to_save': None, 'peft_type': 'LORA', 'r': LORA_R, 'rank_pattern': {}, 'revision': None,
    'target_modules': TARGET_MODULES, 'task_type': 'CAUSAL_LM', 'use_dora': False, 'use_rslora': False,
}
(out_dir / 'adapter_config.json').write_text(json.dumps(cfg, indent=2))
for f in sorted(out_dir.iterdir()):
    print(f.name, f'{f.stat().st_size / 2**20:.1f} MiB')
keys = sorted(tensors)
print(len(keys), 'tensors, e.g.', keys[:2])
assert all(k.startswith('base_model.model.model.language_model.layers.') for k in keys), 'unexpected key names'
nonzero_B = sum(int(tensors[k].abs().sum() > 0) for k in keys if k.endswith('lora_B.weight'))
print(f'{nonzero_B}/{len(keys) // 2} lora_B matrices were updated by training')

(Path('/kaggle/working') / 'train_log.json').write_text(json.dumps(log))
shutil.make_archive('/kaggle/working/lora_adapter', 'zip', root_dir='/kaggle/working/adapters')
print('Download /kaggle/working/lora_adapter.zip (contains adapters/main_lora/...)')
'''

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {k: v for k, v in starter["metadata"].items() if k != "papermill"},
    "cells": [
        md("# Train a LoRA adapter on successful agent trajectories\n\n"
           "Inputs: competition wheelhouse dataset + **gemma-4-31b-it-qat-q4_0-unquantized** (Transformers). "
           "Settings: **GPU L4 x4**. Works on Python 3.12 or 3.13, no internet needed. "
           "Run with **Save Version → Save & Run All**. Output: `lora_adapter.zip`."),
        md("## 0. Environment check"), code(env_cell),
        md("## 1. Install packages"), code(install_cell),
        md("## 2. Settings and training data"), code(config_cell),
        md("## 3. Tokenize and mark target tokens"), code(tokenize_cell),
        md("## 4. Load model in 4-bit and add LoRA (plain PyTorch)"), code(model_cell),
        md("## 5. Loss and smoke test"), code(loss_cell),
        md("## 6. Train"), code(train_cell),
        md("## 7. Save adapter"), code(save_cell),
    ],
}
json.dump(nb, open(args.out, "w"), indent=1)
print("wrote", args.out, f"({len(rows)} sequences", "embedded)" if EMBED else "- NOT embedded: upload sft_data.jsonl as a private dataset and attach it)")
