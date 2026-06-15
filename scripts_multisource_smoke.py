#!/usr/bin/env python3
r"""CPU smoke for the MULTI-SOURCE ablation pool, run BEFORE committing a GPU-night.

Validates the thing most likely to silently invalidate the experiment: that the combined
(Nemotron + gemma4[+phi4]) per-seed pool is (a) actually multi-source, (b) val-contamination
free, and (c) builds sane curated corpora. The gemma4 clean files are already val-filtered by
scripts_field_ingest; the CACHED Nemotron traces (work/traces_subset.jsonl) are NOT, so this
checks whether they leak val problems into a seed's pool.

 python scripts_multisource_smoke.py --seed 0
"""
import argparse, glob, json, os
from pathlib import Path
from config import Config
from generate import Trace
from field import StigmergicField
from curate import build_all, population_success
from run_ablation import load_split

def load_jsonl_traces(path):
 return [Trace(**json.loads(l)) for l in open(path) if l.strip]

def main:
 ap = argparse.ArgumentParser
 ap.add_argument("--seed", type=int, default=0)
 args = ap.parse_args
 cfg = Config; cfg.seed = args.seed
 work = cfg.work_dir

 train, val = load_split(cfg)
 val_ids = {p["id"] if "id" in p else p["problem_id"] for p in val}
 train_ids = {p["id"] if "id" in p else p["problem_id"] for p in train}
 print(f"[smoke] seed{args.seed}: {len(train_ids)} train problems, {len(val_ids)} val (held-out)")

 # --- Nemotron (on-policy) cached traces ---
 nemo = load_jsonl_traces(work / "traces_subset.jsonl")
 nemo_val_leak = [t for t in nemo if t.problem_id in val_ids]
 print(f"[smoke] Nemotron cache: {len(nemo)} traces, {len(set(t.problem_id for t in nemo))} problems"
 f" | val-LEAK: {len(nemo_val_leak)} traces on {len(set(t.problem_id for t in nemo_val_leak))} val problems")

 # --- cross-source clean files for THIS seed ---
 clean = sorted(glob.glob(str(work / "field_sources" / "clean" / f"*__seed{args.seed}.jsonl")))
 xsrc = []
 for f in clean:
 ts = load_jsonl_traces(f)
 leak = sum(1 for t in ts if t.problem_id in val_ids)
 print(f"[smoke] {os.path.basename(f)}: {len(ts)} traces | val-LEAK: {leak}")
 xsrc += ts

 # --- map persona -> TRUE source (NOT persona; 4 Nemotron personas = ONE source) ---
 NEMO_PERSONAS = {"positional", "hypothesis_test", "pattern_diff", "decompositional"}
 def source_of(persona: str) -> str:
 if persona in NEMO_PERSONAS: return "nemotron"
 if persona.startswith("gemma"): return "gemma"
 if persona.startswith("phi"): return "phi"
 return "other:" + persona

 # --- build the per-seed multi-source pool: filter BOTH to non-val, then combine ---
 pool = [t for t in (nemo + xsrc) if t.problem_id not in val_ids]
 by_pid, by_pid_correct = {}, {}
 for t in pool:
 s = source_of(t.persona)
 by_pid.setdefault(t.problem_id, set).add(s)
 if t.correct:
 by_pid_correct.setdefault(t.problem_id, set).add(s)
 multi = {pid: s for pid, s in by_pid.items if len(s) >= 2}
 xs_correct = {pid: s for pid, s in by_pid_correct.items if len(s) >= 2}
 print(f"\n[smoke] COMBINED pool (val-filtered): {len(pool)} traces, {len(by_pid)} problems")
 print(f"[smoke] problems with >=2 distinct TRUE sources (any trace) : {len(multi)}")
 print(f"[smoke] problems with >=2 distinct TRUE sources CORRECT : {len(xs_correct)}"
 f" <-- the real cross-source consensus field (gemma/nemo/phi agree)")
 src_fam = {}
 for t in pool: s = source_of(t.persona); src_fam[s] = src_fam.get(s, 0) + 1
 print(f"[smoke] TRUE source tally: {src_fam}")

 # --- build the field + curated corpora (proxy frontier, no GPU) ---
 fld = StigmergicField(cfg); fld.deposit(pool); fld.settle(passes=3)
 pop = population_success(pool)
 model_pass_rate = {pid: (1.0 if v == 1.0 else 0.0) for pid, v in pop.items}
 corpora = build_all(pool, fld, model_pass_rate, cfg)
 print("\n[smoke] curated corpus sizes (equal-budget pre-trim):")
 for nm, c in corpora.items:
 print(f" {nm:24s} {len(c)} examples")
 print("\n[smoke] OK if: Nemotron val-LEAK handled by filter, >=2-source count is healthy, corpora non-empty.")

if __name__ == "__main__":
 main
