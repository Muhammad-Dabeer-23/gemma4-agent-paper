"""Fit sft data into a token budget: keep every real-task sequence, then whole synthetic tasks,
shortest first (decisive runs), until the budget is used. Tokens estimated as chars / 3.5."""
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--budget", type=int, required=True)
args = ap.parse_args()
tok = lambda r: int(len(r["text"]) / 3.5)
rows = [json.loads(l) for l in open(args.src)]
is_synth = lambda t: t.startswith("synth")
real = [r for r in rows if not is_synth(r["task"])]
by_task = {}
for r in rows:
    if is_synth(r["task"]):
        by_task.setdefault(r["task"], []).append(r)
left, keep = args.budget - sum(map(tok, real)), []
for t, rs in sorted(by_task.items(), key=lambda kv: sum(map(tok, kv[1]))):
    if sum(map(tok, rs)) <= left:
        keep += rs; left -= sum(map(tok, rs))
final = real + keep
with open(args.out, "w") as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"real {len({r['task'] for r in real})} tasks / {len(real)} seqs / {sum(map(tok, real))} tok | "
      f"synth {len({r['task'] for r in keep})}/{len(by_task)} tasks / {len(keep)} seqs / {sum(map(tok, keep))} tok | "
      f"total {sum(map(tok, final))} tok, {len(final)} seqs")
