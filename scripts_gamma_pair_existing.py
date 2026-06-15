#!/usr/bin/env python3
r"""Run Parunak's pairwise WID/CID independence test (gamma_pair) on the EXISTING Spark data —
no new collection. Three reads:
 (1) INDEPENDENT families : gemma4 / gpt-oss / phi4 on the 240-pool (genuinely distinct lineages).
 (2) NEMOTRON control : the 4 same-model personas (positional/hypothesis_test/pattern_diff/
 decompositional) — the degeneracy control (expect LOW cross-CID).
 (3) COMBINED on the 101-overlap : all 7 "families" together → the CID matrix should show the
 4 Nemotron personas as a LOW-internal-CID block vs HIGH cross-model CID.
Reuses the slice's gamma_pair.py + metric.py. Read is pre-committed in the turn log before running.
"""
import json, os, sys
sys.path.insert(0, "/opt/ml/projects/nemotron-slice/experiments/independence-slice")
import gamma_pair as G

CLEAN = "/opt/ml/projects/nemotron-stigmergy/work/field_sources/clean"
NEM = "/opt/ml/projects/nemotron-stigmergy/work/traces_subset.jsonl"
FAMMAP = {"gemma4-31b-it-q8_0": "gemma", "gpt-oss-120b": "gptoss", "phi4-reasoning-plus": "phi"}

def load(path, fam_from_persona=True):
 rows = []
 for i, l in enumerate(open(path)):
 r = json.loads(l)
 fam = FAMMAP.get(r.get("persona"), r.get("persona")) if fam_from_persona else r["family"]
 rows.append(dict(problem_id=r["problem_id"], family=fam, sample_idx=i, answer=str(r["answer"])))
 return rows

def gold_map(*paths):
 g = {}
 for p in paths:
 for l in open(p):
 r = json.loads(l); g.setdefault(r["problem_id"], str(r["gold"]))
 return g

def summarize(tag, out):
 print(f"\n===== {tag} =====")
 print(f" families={out['families']} | problems={out['n_problems']}")
 print(" CID by pair:")
 for d in sorted(out["CID"], key=lambda x: -x["CID"]):
 print(f" {d['pair']:32s} CID={d['CID']}")
 v = out.get("validation")
 if v:
 print(f" pred_within={v['pred_within']} pred_cross_unweighted={v['pred_cross_unweighted']} "
 f"pred_cross_CIDweighted={v['pred_cross_cidweighted']}")
 print(f" gamma_pred (CID-weighted)={v['gamma_pred_cidweighted']}")
 print(f" acc: best_single={v['acc_best_single']} vote_unweighted={v['acc_vote_unweighted']} "
 f"vote_CIDweighted={v['acc_vote_cidweighted']} (n_cross={v['n_cross']}, n_within={v['n_within']})")

def main:
 gem = load(f"{CLEAN}/traces_gemma4_subset__seed0.jsonl")
 gpt = load(f"{CLEAN}/traces_gptoss_subset__seed0.jsonl")
 phi = load(f"{CLEAN}/traces_phi4_subset__seed0.jsonl")
 nem = load(NEM) # personas kept as their own "families"
 gold = gold_map(f"{CLEAN}/traces_gemma4_subset__seed0.jsonl", NEM)

 indep = gem + gpt + phi
 out1, _ = G.compute(indep, gold=gold, estimator="jsd"); summarize("(1) INDEPENDENT families (240-pool)", out1)
 out2, _ = G.compute(nem, gold=gold, estimator="jsd"); summarize("(2) NEMOTRON personas — degeneracy control", out2)

 overlap = set(r["problem_id"] for r in indep) & set(r["problem_id"] for r in nem)
 comb = [r for r in (indep + nem) if r["problem_id"] in overlap]
 out3, _ = G.compute(comb, gold=gold, estimator="jsd"); summarize(f"(3) COMBINED on {len(overlap)}-overlap (all 7)", out3)

 json.dump(dict(independent=out1, nemotron_control=out2, combined_overlap=out3),
 open("/opt/ml/projects/nemotron-stigmergy/gamma_pair_existing_out.json", "w"), indent=2)
 print("\n[done] wrote gamma_pair_existing_out.json")

if __name__ == "__main__":
 main
