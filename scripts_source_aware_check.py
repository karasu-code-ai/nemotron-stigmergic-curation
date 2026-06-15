#!/usr/bin/env python3
r"""Validate the source-aware consensus fix: does it actually change WHICH traces the
consensus arm selects, away from the over-sampled (Nemotron 4-persona) source toward
cross-source-corroborated traces? CPU-only smoke before any training.

 python scripts_source_aware_check.py --seed 0
"""
import argparse, glob, json, collections
from config import Config
from generate import Trace
from field import StigmergicField, source_of
from curate import build_A, _by_problem, _correct
from run_ablation import load_split

def load_pool(cfg):
 work = cfg.work_dir
 train, val = load_split(cfg)
 val_ids = {p["id"] for p in val}
 files = [work / "traces_subset.jsonl"] + [
 __import__("pathlib").Path(f) for f in glob.glob(str(work / "field_sources" / "*.jsonl"))]
 pool = []
 for f in files:
 if not f.exists: continue
 for l in open(f):
 if l.strip:
 try:
 t = Trace(**json.loads(l))
 if t.problem_id not in val_ids:
 pool.append(t)
 except Exception: pass
 return pool

def src_dist(corpus, by_pid):
 """For each selected example, which source produced the chosen trace?"""
 c = collections.Counter
 # map (pid, completion text) -> source via the pool
 for ex in corpus:
 pid = ex["problem_id"]
 for t in by_pid.get(pid, []):
 if t.text == ex["completion"]:
 c[source_of(t.persona)] += 1
 break
 return c

def main:
 ap = argparse.ArgumentParser; ap.add_argument("--seed", type=int, default=0)
 args = ap.parse_args
 cfg = Config; cfg.seed = args.seed
 pool = load_pool(cfg)
 by_pid = _by_problem(pool)
 n_solved = sum(1 for ts in by_pid.values if _correct(ts))
 print(f"[check] pool={len(pool)} traces, {len(by_pid)} problems, {n_solved} with >=1 correct")

 fld = StigmergicField(cfg); fld.deposit(pool); fld.settle(passes=3)

 # a couple of illustrative fragments: blind weight vs distinct-source count
 print("\n[check] fragment scoring — blind weight vs DISTINCT SOURCES (a few multi-trace frags):")
 shown = 0
 for (pid, frag), srcs in fld.sources_within.items:
 ntr = fld.n_traces.get((pid, frag), 0)
 if ntr >= 3 and shown < 5:
 print(f" traces={ntr:2d} sources={len(srcs)} {sorted(srcs)} blind_w={fld.weight.get((pid,frag),0):.1f} frag='{frag[:40]}'")
 shown += 1

 # consensus selection: source distribution under blind vs source-aware
 cfg.source_aware = False
 blind = src_dist(build_A(pool, fld, cfg), by_pid)
 cfg.source_aware = True
 aware = src_dist(build_A(pool, fld, cfg), by_pid)
 print("\n[check] build_A selected-trace SOURCE distribution:")
 print(f" {'source':10} {'blind':>6} {'source-aware':>13}")
 for s in sorted(set(blind) | set(aware)):
 print(f" {s:10} {blind.get(s,0):>6} {aware.get(s,0):>13}")
 print(f" {'TOTAL':10} {sum(blind.values):>6} {sum(aware.values):>13}")
 print("\n[check] FIX WORKS if source-aware shifts selection AWAY from a single "
 "over-sampled source (nemotron's 4 personas) toward cross-source-corroborated traces.")

if __name__ == "__main__":
 main
