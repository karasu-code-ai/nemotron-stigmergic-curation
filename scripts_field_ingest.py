#!/usr/bin/env python3
r"""Ingest cross-source trace dumps into per-seed field sources, with the
two safety checks:

 1. SANITY: drop empty / near-empty `text` (<20 chars) and malformed rows (the signature of
 ollama thinking-channel poisoning — should be ~0 with the fixed gen_traces.py).
 2. VAL-CONTAMINATION: a problem can be in seed-A's generation pool AND seed-B's val
 (e.g. 8b12ff37 ∈ seed0-pool ∩ seed1-val). For each seed k, EXCLUDE any problem_id in
 val_seed{k}.txt before that seed can use these traces — never leak val into training.

Reads: work/field_sources/*.jsonl + work/field_sources/val_seed{0,1}.txt
Writes: work/field_sources/clean/<source>__seed{k}.jsonl (per-seed, contamination-filtered)
Reports: per source — traces in, dropped(sanity), per-seed kept(after val filter), accuracy.
These feed the cross-source field/curate (research arm C/γ) — the multi-source condition.
"""
import json, glob, os, sys

FS = "work/field_sources"
OUT = os.path.join(FS, "clean")
SEEDS = [0, 1]

def load_val(k):
 p = os.path.join(FS, f"val_seed{k}.txt")
 return set(l.strip for l in open(p)) if os.path.exists(p) else set

def main:
 os.makedirs(OUT, exist_ok=True)
 vals = {k: load_val(k) for k in SEEDS}
 for k in SEEDS:
 print(f"[ingest] val_seed{k}: {len(vals[k])} ids")

 dumps = [p for p in glob.glob(os.path.join(FS, "*.jsonl")) if "/clean/" not in p]
 if not dumps:
 sys.exit(f"[ingest] no *.jsonl in {FS}")
 REQUIRED = {"problem_id", "problem", "gold", "persona", "temperature", "text", "answer", "correct"}

 for dump in sorted(dumps):
 src = os.path.basename(dump).replace(".jsonl", "")
 rows, dropped = [], 0
 for line in open(dump):
 line = line.strip
 if not line:
 continue
 try:
 r = json.loads(line)
 except Exception:
 dropped += 1; continue
 if not REQUIRED.issubset(r) or len(str(r.get("text", "")).strip) < 20:
 dropped += 1; continue
 rows.append(r)
 persona = rows[0]["persona"] if rows else "?"
 acc = round(sum(bool(r["correct"]) for r in rows) / max(len(rows), 1), 3)
 print(f"\n[ingest] {src}: persona={persona} kept={len(rows)} dropped(sanity)={dropped} acc={acc}")
 for k in SEEDS:
 keep = [r for r in rows if r["problem_id"] not in vals[k]]
 removed = len(rows) - len(keep)
 outp = os.path.join(OUT, f"{src}__seed{k}.jsonl")
 with open(outp, "w") as f:
 for r in keep:
 f.write(json.dumps(r) + "\n")
 print(f" seed{k}: kept {len(keep)} (removed {removed} val-contaminants) -> {outp}")

if __name__ == "__main__":
 main
