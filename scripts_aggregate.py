#!/usr/bin/env python3
r"""Aggregate per-seed run_ablation results into a distribution (ga-06).

Reads work/seed{S}/results.json for each seed and reports, per arm, the MEAN
and STD of val accuracy across seeds (and the lift over base). The whole point
(ga-06 "Compute Allocation in Evolutionary Search"): LLM-evolution results are
routinely reported as best-of-N with the run-to-run distribution undocumented;
a ~2% arm gap is only meaningful against its spread. So we report mean +/- std,
not a single number -- this is how we decide whether B/C/D actually beat the
baselines or whether the tiny-pass "null" (B=0.43 vs A=0.41) was just noise.

Usage:
 python scripts_aggregate.py --seeds 0 1
 python scripts_aggregate.py --seeds 0 1 2 --work work
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

ARM_ORDER = ["B0_random", "B1_shortest", "A_consensus", "B_consensus_frontier",
 "C_gamma_consensus", "D_diversity"]


def _load(work: Path, seed: int):
 f = work / f"seed{seed}" / "results.json"
 if not f.exists:
 return None
 d = json.loads(f.read_text)
 # tolerate both the old flat {arm: acc} and the new wrapped schema
 return d if "results" in d else {"results": d, "base": None,
 "corpus_sizes": {}, "seed": seed}


def main:
 ap = argparse.ArgumentParser
 ap.add_argument("--seeds", type=int, nargs="+", required=True)
 ap.add_argument("--work", default="work")
 args = ap.parse_args
 work = Path(args.work)

 runs = [(s, _load(work, s)) for s in args.seeds]
 have = [(s, d) for s, d in runs if d is not None]
 missing = [s for s, d in runs if d is None]
 if missing:
 print(f"[aggregate] WARNING: no results.json for seeds {missing} "
 f"(run not finished?)")
 if not have:
 print("[aggregate] nothing to aggregate yet."); return

 arms = ARM_ORDER + [a for s, d in have for a in d["results"]
 if a not in ARM_ORDER]
 arms = list(dict.fromkeys(arms))
 bases = [d["base"] for _, d in have if d.get("base") is not None]
 base_mean = statistics.mean(bases) if bases else None

 print(f"\n==== AGGREGATE over seeds {[s for s, _ in have]} "
 f"(n={len(have)}) ====")
 if base_mean is not None:
 bstd = statistics.pstdev(bases) if len(bases) > 1 else 0.0
 print(f" base (no SFT) : {base_mean:.3f} +/- {bstd:.3f}")

 table = {}
 for arm in arms:
 accs = [d["results"][arm] for _, d in have if arm in d["results"]]
 if not accs:
 continue
 mean = statistics.mean(accs)
 std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
 lift = (mean - base_mean) if base_mean is not None else None
 table[arm] = {"mean": mean, "std": std, "n": len(accs),
 "per_seed": accs, "lift_over_base": lift}
 lift_s = f" (lift {lift:+.3f})" if lift is not None else ""
 print(f" {arm:24s}: {mean:.3f} +/- {std:.3f} "
 f"per-seed={['%.3f' % a for a in accs]}{lift_s}")

 # headline contrasts the experiment cares about
 def gap(x, y):
 if x in table and y in table:
 return table[x]["mean"] - table[y]["mean"]
 return None
 print("\n --- key contrasts (mean gap) ---")
 for hi, lo, why in [("B_consensus_frontier", "A_consensus", "frontier weighting"),
 ("A_consensus", "B0_random", "consensus vs random (the core claim)"),
 ("C_gamma_consensus", "A_consensus", " boilerplate-discount vs plain consensus"),
 ("D_diversity", "A_consensus", "/ diversity vs argmax-1"),
 ("C_gamma_consensus", "B0_random", "best mechanism vs baseline")]:
 g = gap(hi, lo)
 if g is not None:
 print(f" {hi} - {lo}: {g:+.3f} [{why}]")
 print(" (a positive gap LARGER than the std bands above is the result; "
 "ga-06: small gaps within the spread are noise.)")

 out = {"seeds": [s for s, _ in have], "base_mean": base_mean, "arms": table}
 (work / "results_aggregate.json").write_text(json.dumps(out, indent=2))
 print(f"\n[aggregate] wrote {work/'results_aggregate.json'}")


if __name__ == "__main__":
 main
