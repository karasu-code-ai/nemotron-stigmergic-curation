#!/usr/bin/env python3
r"""DETERMINACY CENSUS v2 — six-label discrimination mode. Supersedes census_cryptarithm.py (v1).

What v2 adds over v1:
 * FROZEN R0 manifest (committed primitive set + hash) — "R0_implementation_gap" separated from
 "dsl_miss" (true-outside-grammar) and "parser_failed" (quarantine).
 * SIX-label gold-free `identifiability_status` × orthogonal `gold_relation` (deployable where gold
 is unknown). `determined_wrong` is a DERIVED cell, not a primary bin.
 * candidate-set metrics + corpus-level `gold_output@K` and `selector_regret` — the Day-2 decisive
 numbers that say whether the wall is SELECTION vs PROPOSER-COVERAGE.
 * output-equivalence dedup (we count distinct OUTPUTS, prior = #consistent assignments = an
 Occam-ish support prior; the real MDL/naturalness selector is medium-term).

Scope: cryptarithm_deduce (the family the reference solver solver covers). R0 == the reference solver enumerator reused
verbatim (so R0_implementation_gap is expected ~0 — a validation, confirmed if huikang rule_found all
land identified). Cross-family + LLM->Python proposer + canonical execution sandbox = cloud-API.

The R0 census executes NO untrusted code (own deterministic enumerator) so it needs no sandbox; the
sandbox requirement in the prereg applies to the LLM->Python / template-miner proposers .
"""
import json, hashlib, math, os, signal, sys, time
from collections import Counter

WINNER = "/opt/ml/projects/winner_snapshot/nemotron-master"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import census_cryptarithm as v1 # reuse enumerate_answers (output-dedup'd consistent-rule enumeration)

# ---- FROZEN R0 MANIFEST (committed before the run; the registration object) ----
R0_MANIFEST = {
 "family": "cryptarithm_deduce",
 "number_construction": "2-digit positional: N(a,b)=10*a+b",
 "symbol_to_digit": "bijective (unique digits 0-9), op-symbol segregated -> operation",
 "operations": ["add", "abs_diff", "mul", "concat", "rev_concat"],
 "result_decoding": "natural-length digits (no leading-zero pad); concat/rev_concat 4-digit padded",
 "enumeration_cap": 200, # distinct outputs (Solver.max_solutions)
 "per_problem_timeout_s": v1.PER_PROBLEM_TIMEOUT,
 "prior": "support-count (number of consistent full assignments reaching each output)",
 "source": "winner_snapshot/investigators/cryptarithm_deduce.py reused verbatim",
}
R0_MANIFEST_HASH = hashlib.sha256(
 json.dumps(R0_MANIFEST, sort_keys=True).encode).hexdigest[:16]


def parse_ok(data):
 """parse_status gate: structural validity. Failures are QUARANTINED (never count toward %)."""
 try:
 exs = data["examples"]
 if not exs:
 return False
 for e in exs:
 if len(e["input_value"]) != 5 or len(e["output_value"]) < 1:
 return False
 if len(data["question"]) != 5:
 return False
 if "answer" not in data:
 return False
 return True
 except (KeyError, TypeError):
 return False


def manifest_row(pid, huikang_status, data, parse_status):
 syms = set
 if parse_status:
 for e in data["examples"]:
 syms |= set(e["input_value"]) | set(e["output_value"])
 syms |= set(data["question"]) | set(data.get("answer", ""))
 return {
 "problem_id": pid,
 "family": "cryptarithm_deduce",
 "subtype": None,
 "huikang_status": huikang_status,
 "parse_status": "ok" if parse_status else "failed",
 "R0_supported_primitives": R0_MANIFEST["operations"],
 "num_examples": len(data.get("examples", [])) if parse_status else None,
 "query_present": ("question" in data) if parse_status else False,
 "gold_available": bool(data.get("answer")) if parse_status else False,
 "n_distinct_symbols": len(syms) if parse_status else None,
 }


def classify_v2(pid, huikang_status, data):
 """Return the v2 census record: identifiability_status x gold_relation + metrics (+ certificate)."""
 rec = {"problem_id": pid, "huikang_status": huikang_status}
 if not parse_ok(data):
 rec.update(identifiability_status="parser_failed", gold_relation="gold_unknown",
 quarantined=True, reason="parse")
 return rec
 gold = data["answer"]

 timed_out = False
 try:
 signal.alarm(v1.PER_PROBLEM_TIMEOUT)
 answers, meta = v1.enumerate_answers(data) # Counter: output -> #consistent assignments
 signal.alarm(0)
 except TimeoutError:
 signal.alarm(0); timed_out = True; answers, meta = Counter, {"cap_hit": False}
 except (KeyError, IndexError, RecursionError) as ex:
 signal.alarm(0)
 rec.update(identifiability_status="parser_failed", gold_relation="gold_unknown",
 quarantined=True, reason=f"enum_err:{type(ex).__name__}")
 return rec

 outs = answers.most_common # [(output, support), ...] support-desc
 n_out = len(outs)
 rule_count = sum(answers.values) # total consistent assignments
 rec["candidate_output_count"] = n_out
 rec["candidate_rule_count"] = rule_count
 rec["cap_hit"] = meta.get("cap_hit", False)
 rec["enumeration_timeout"] = timed_out

 gold_in = gold in answers
 rec["gold_relation"] = "gold_match" if gold_in else "gold_mismatch"

 if n_out == 0:
 # no consistent rule in R0. dsl_miss UNLESS the reference solver itself solved it (=> our gap).
 if huikang_status == "rule_found":
 rec["identifiability_status"] = "R0_implementation_gap"
 else:
 rec["identifiability_status"] = "dsl_miss"
 rec["reason"] = "timeout" if timed_out else "no_consistent_rule"
 return rec

 # entropy + prior margin over the support distribution
 total = rule_count
 ent = -sum((s / total) * math.log2(s / total) for _, s in outs)
 rec["candidate_output_entropy"] = round(ent, 4)
 s1 = outs[0][1]; s2 = outs[1][1] if n_out > 1 else 0
 rec["top1_prior_margin"] = round((s1 - s2) / total, 4)
 rec["gold_rank"] = ([o for o, _ in outs].index(gold) + 1) if gold_in else None
 rec["top1_output"] = outs[0][0]

 if n_out == 1:
 rec["identifiability_status"] = "rule_identified" if rule_count == 1 else "output_identified"
 elif s1 > s2:
 rec["identifiability_status"] = "prior_selected"
 else:
 rec["identifiability_status"] = "unidentified"

 # derived cell
 rec["determined_wrong"] = rec["identifiability_status"] in ("rule_identified", "output_identified") \
 and rec["gold_relation"] == "gold_mismatch"

 if rec["identifiability_status"] in ("prior_selected", "unidentified"):
 alts = [o for o, _ in outs if o != gold][:8]
 rec["ambiguity_certificate"] = {
 "rule_class": "R0", "gold": gold, "gold_among_candidate_outputs": gold_in,
 "n_candidate_outputs": n_out, "alt_consistent_outputs": alts,
 "top1_prior_margin": rec["top1_prior_margin"],
 "conclusion": f"prompt does not identify gold output within R0 ({n_out} candidate outputs"
 + (", gold present)" if gold_in else ", gold NOT reachable in R0)"),
 }
 return rec


def main:
 signal.signal(signal.SIGALRM, lambda *a: (_ for _ in).throw(TimeoutError))
 probs = [json.loads(l) for l in open(os.path.join(WINNER, "problems.jsonl"))]
 crypt = [p for p in probs if p["category"] == "cryptarithm_deduce"]

 manifest, records = [], []
 t0 = time.time
 for i, p in enumerate(crypt):
 pid, st = p["id"], p["status"]
 try:
 data = json.loads(open(os.path.join(WINNER, "problems", f"{pid}.jsonl")).readline)
 except Exception:
 data = {}
 ps = parse_ok(data)
 manifest.append(manifest_row(pid, st, data, ps))
 records.append(classify_v2(pid, st, data))
 if (i + 1) % 100 == 0:
 print(f" [{i+1}/{len(crypt)}] {time.time-t0:.0f}s", flush=True)

 out = {"manifest_hash": R0_MANIFEST_HASH, "manifest": R0_MANIFEST,
 "n": len(records), "records": records, "problem_manifest": manifest,
 "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
 json.dump(out, open("census_v2.json", "w"), indent=2)
 print(f"\n[census-v2] wrote census_v2.json manifest={R0_MANIFEST_HASH} ({time.time-t0:.0f}s)", flush=True)

 # ---------- READS (residual = huikang rule_unknown; quarantine excluded) ----------
 def reads(stratum_filter, label):
 rs = [r for r in records if stratum_filter(r)]
 quar = [r for r in rs if r.get("quarantined")]
 live = [r for r in rs if not r.get("quarantined")]
 n = len(live)
 S = Counter(r["identifiability_status"] for r in live)
 gold_reach = [r for r in live if r.get("gold_relation") == "gold_match"]
 dw = sum(1 for r in live if r.get("determined_wrong"))
 print(f"\n[census-v2] ===== {label} (n_live={n}, quarantined={len(quar)}) =====", flush=True)
 for k in ("rule_identified", "output_identified", "prior_selected", "unidentified",
 "dsl_miss", "R0_implementation_gap"):
 print(f" {k:22s} {S.get(k,0)}", flush=True)
 print(f" determined_wrong (derived) {dw}", flush=True)
 # read 1: identifiability fraction
 if n:
 frac = (S.get("prior_selected",0) + S.get("unidentified",0) + dw) / n
 r1 = ("identifiability quantified (>=0.50)" if frac >= 0.50 else
 "WEAKER than expected (<0.25)" if frac < 0.25 else "MID (0.25-0.50)")
 print(f" READ(1) (prior_selected+unidentified+determined_wrong)/n = {frac:.3f} -> {r1}", flush=True)
 # read 3: dsl_miss as the X analogue
 xfrac = S.get("dsl_miss",0) / n
 print(f" READ(3) dsl_miss/n = {xfrac:.3f} -> {'RUN R1 (>0.30)' if xfrac>0.30 else 'reads stand on R0'}", flush=True)
 # gold_output@K + selector_regret (decisive Day-2 numbers, R0 proposer only)
 if n:
 def atK(K): return sum(1 for r in live if (r.get("gold_rank") or 10**9) <= K) / n
 print(f" gold_output@1={atK(1):.3f} @3={atK(3):.3f} @5={atK(5):.3f} "
 f"@any={len(gold_reach)/n:.3f} (R0 proposer)", flush=True)
 in_set = [r for r in live if r.get("gold_relation")=="gold_match"]
 regret = sum(1 for r in in_set if (r.get("gold_rank") or 99) > 1)
 print(f" selector_regret = {regret}/{len(in_set)} (gold in set but not top-1 by support prior)", flush=True)
 return S, n

 reads(lambda r: r["huikang_status"] == "rule_unknown", "RESIDUAL (huikang rule_unknown)")
 reads(lambda r: r["huikang_status"] == "rule_found", "CALIBRATION (huikang rule_found — expect all identified, gap=0)")
 reads(lambda r: r["huikang_status"] == "hypothesis_formed", "hypothesis_formed")


if __name__ == "__main__":
 main
