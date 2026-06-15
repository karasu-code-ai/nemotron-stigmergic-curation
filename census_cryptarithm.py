#!/usr/bin/env python3
r"""STAGE 1 — Determinacy census (R0), per DECISIONS #32 / docs/DETERMINACY_CENSUS_PREREG.md.

R0 = the reference solver strict rule grammar (investigators/cryptarithm_deduce.py):
bijective symbol->digit (unique mapping) + ops {add, abs_diff, mul, concat, rev_concat}.
The winner solver does FIND-FIRST (returns the most-supported answer). The census needs
ENUMERATE-AND-COUNT: the full set of distinct query answers that R0-consistent hypotheses
produce, so we can ask whether the examples *identify* gold, not just whether *a* rule exists.

We REUSE the open recipe's Solver (its self.answers Counter already aggregates every consistent
full assignment up to max_solutions) rather than reimplement the op semantics. We only change
the readout: instead of best, we take the whole Counter.

Per-problem output (the margin, not just the bin), per the pre-reg:
 bin in {D-gold, D-wrong, U, X}
 candidate_count, output_entropy, gold_among_candidates,
 gold_rank (under the count prior = #consistent assignments supporting each answer; a natural
 Occam-ish "more derivations -> simpler/more-supported" ordering),
 cap_hit / timeout flags (X reasons).
U problems also emit an ambiguity CERTIFICATE (alt consistent answers, NOT answer traces).

Pre-committed reads (writes BEFORE the numbers; see DECISIONS #32):
 (1) U + D-wrong >= 50% of residual -> #26 (identifiability gap) quantified.
 < 25% -> identifiability story WEAKER than #26 claimed.
 (3) X > 30% of residual -> "no resolution at this budget" (run R1), not a null.
"""
import json, math, os, signal, sys, time
from collections import Counter

WINNER = "/opt/ml/projects/winner_snapshot/nemotron-master"
sys.path.insert(0, os.path.join(WINNER, "investigators"))
import cryptarithm_deduce as cd # OPS, num_to_digits, is_concat, Solver

PER_PROBLEM_TIMEOUT = 25 # seconds; X-by-timeout if exceeded


def enumerate_answers(data):
 """Mirror cd.solve_problem's structure but return the FULL answer Counter under R0
 (unique bijective mapping). Returns (Counter answer->#consistent_assignments,
 meta dict with cap_hit / trivial_concat)."""
 examples = []
 for e in data["examples"]:
 inp = e["input_value"]
 out = e["output_value"]
 examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))
 q = data["question"]
 query = (q[0], q[1], q[2], q[3], q[4])

 # Trivial-concat shortcut (winner solve_problem): query op is purely concat/rev_concat.
 concat_ops, nonconcat_ops = set, set
 for ex in examples:
 (concat_ops if cd.is_concat(ex) else nonconcat_ops).add(ex[2])
 q_op = query[2]
 if q_op in concat_ops and q_op not in nonconcat_ops:
 for ex in examples:
 if ex[2] == q_op and cd.is_concat(ex):
 s0, s1, _, s3, s4, rsyms = ex
 if rsyms == (s0, s1, s3, s4):
 ans = query[0] + query[1] + query[3] + query[4]
 else:
 ans = query[3] + query[4] + query[0] + query[1]
 return Counter({ans: 1}), {"trivial_concat": True, "cap_hit": False}
 ans = query[0] + query[1] + query[3] + query[4]
 return Counter({ans: 1}), {"trivial_concat": True, "cap_hit": False}

 arith_examples = [ex for ex in examples if not cd.is_concat(ex)]
 solver = cd.Solver(arith_examples, query, unique=True)
 solver._process(0) # populate solver.answers (every consistent assignment, up to cap)
 cap_hit = len(solver.answers) >= solver.max_solutions
 return Counter(solver.answers), {"trivial_concat": False, "cap_hit": cap_hit}


def classify(pid, status, gold):
 """Return the census record for one problem (R0)."""
 path = os.path.join(WINNER, "problems", f"{pid}.jsonl")
 rec = {"id": pid, "status": status, "gold": gold}
 try:
 data = json.loads(open(path).readline)
 except Exception as ex:
 rec.update(bin="X", reason=f"load:{type(ex).__name__}")
 return rec

 timed_out = False
 try:
 signal.alarm(PER_PROBLEM_TIMEOUT)
 answers, meta = enumerate_answers(data)
 signal.alarm(0)
 except TimeoutError:
 signal.alarm(0)
 timed_out = True
 answers, meta = Counter, {"trivial_concat": False, "cap_hit": False}
 except (KeyError, IndexError, RecursionError) as ex:
 signal.alarm(0)
 rec.update(bin="X", reason=f"err:{type(ex).__name__}")
 return rec

 cands = list(answers.items) # (answer, support_count)
 cands.sort(key=lambda kv: (-kv[1], kv[0])) # count prior, tie-broken lexically
 n_distinct = len(cands)
 total = sum(answers.values)
 rec["candidate_count"] = n_distinct
 rec["cap_hit"] = meta["cap_hit"]
 rec["trivial_concat"] = meta["trivial_concat"]

 if n_distinct == 0:
 rec.update(bin="X", reason="timeout" if timed_out else "no_consistent_rule")
 return rec

 # output entropy over the support distribution (bits)
 ent = -sum((c / total) * math.log2(c / total) for _, c in cands) if total else 0.0
 rec["output_entropy"] = round(ent, 4)
 gold_in = gold in answers
 rec["gold_among_candidates"] = gold_in
 rec["gold_rank"] = ([a for a, _ in cands].index(gold) + 1) if gold_in else None
 rec["top_candidate"] = cands[0][0]

 if n_distinct == 1:
 only = cands[0][0]
 rec["bin"] = "D-gold" if only == gold else "D-wrong"
 else:
 rec["bin"] = "U"
 # ambiguity CERTIFICATE (alt consistent answers, capped at 8 for the file)
 alts = [a for a, _ in cands if a != gold][:8]
 rec["certificate"] = {
 "rule_class": "R0",
 "gold": gold,
 "gold_among_candidates": gold_in,
 "n_consistent_answers": n_distinct,
 "alt_consistent_answers": alts,
 "conclusion": f"prompt does not identify gold within R0 ({n_distinct} consistent answers"
 + (", gold present)" if gold_in else ", gold NOT even reachable in R0)"),
 }
 return rec


def main:
 signal.signal(signal.SIGALRM, lambda *a: (_ for _ in).throw(TimeoutError))
 probs = [json.loads(l) for l in open(os.path.join(WINNER, "problems.jsonl"))]
 crypt = [p for p in probs if p["category"] == "cryptarithm_deduce"]
 # gold lives in the per-problem file; problems.jsonl 'submission' is the open recipe's guess.
 strata = {"rule_unknown": [], "rule_found": [], "hypothesis_formed": []}
 for p in crypt:
 strata.setdefault(p["status"], []).append(p["id"])

 records = []
 t0 = time.time
 for status in ("rule_unknown", "rule_found", "hypothesis_formed"):
 ids = strata.get(status, [])
 print(f"[census] {status}: {len(ids)} problems", flush=True)
 for i, pid in enumerate(ids):
 data = json.loads(open(os.path.join(WINNER, "problems", f"{pid}.jsonl")).readline)
 gold = data["answer"]
 records.append(classify(pid, status, gold))
 if (i + 1) % 50 == 0:
 el = time.time - t0
 print(f" [{status} {i+1}/{len(ids)}] {el:.0f}s", flush=True)

 json.dump(records, open("census_cryptarithm.json", "w"), indent=2)
 print(f"[census] wrote census_cryptarithm.json ({len(records)} records, {time.time-t0:.0f}s)", flush=True)

 # ---- pre-committed reads (residual = rule_unknown stratum) ----
 def summarize(stratum):
 rs = [r for r in records if r["status"] == stratum]
 bins = Counter(r["bin"] for r in rs)
 n = len(rs)
 return n, bins

 print("\n[census] ===== BIN COUNTS BY STRATUM =====", flush=True)
 for st in ("rule_unknown", "rule_found", "hypothesis_formed"):
 n, bins = summarize(st)
 print(f" {st:18s} n={n}: " + " ".join(f"{b}={bins.get(b,0)}"
 for b in ("D-gold", "D-wrong", "U", "X")), flush=True)

 n, bins = summarize("rule_unknown")
 u, dw, dg, x = bins.get("U", 0), bins.get("D-wrong", 0), bins.get("D-gold", 0), bins.get("X", 0)
 ident = (u + dw) / n if n else 0
 xfrac = x / n if n else 0
 rus = [r for r in records if r["status"] == "rule_unknown"]
 gold_reachable_in_U = sum(1 for r in rus if r["bin"] == "U" and r.get("gold_among_candidates"))
 print("\n[census] ===== PRE-COMMITTED READS (residual = rule_unknown) =====", flush=True)
 print(f" bins: D-gold={dg} D-wrong={dw} U={u} X={x} (n={n})", flush=True)
 print(f" (1) identifiability fraction (U+D-wrong)/n = {ident:.3f}", flush=True)
 r1 = ("#26 QUANTIFIED (>=0.50)" if ident >= 0.50 else
 "identifiability WEAKER than #26 (<0.25)" if ident < 0.25 else
 "MID (0.25-0.50): partial support")
 print(f" READ (1): {r1}", flush=True)
 print(f" (3) X fraction = {xfrac:.3f}", flush=True)
 r3 = "X>30% -> no resolution at this budget; RUN R1" if xfrac > 0.30 else "X<=30% -> reads stand on R0"
 print(f" READ (3): {r3}", flush=True)
 print(f" of the {u} U-problems, gold reachable in R0 for {gold_reachable_in_U} "
 f"({(gold_reachable_in_U/u if u else 0):.2f}); the rest need a broader inductive bias", flush=True)

 # instrument sanity: rule_found should be overwhelmingly D-gold
 n_rf, bins_rf = summarize("rule_found")
 print(f"\n[census] instrument sanity — rule_found D-gold rate = "
 f"{bins_rf.get('D-gold',0)}/{n_rf} (expect high)", flush=True)


if __name__ == "__main__":
 main
