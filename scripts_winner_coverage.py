#!/usr/bin/env python3
"""Run the open recipe's deterministic reasoners (winner_snapshot) over the full parsed problem set
and measure REAL coverage: per family, how many produce a trace at all, and of those how many
produce the GOLD answer (using the open recipe's own extract_answer + compare_answer = the official
metric). Cross-checks the open recipe's gold against data/train.csv. No model, CPU-only."""
import sys, json, glob, re, collections
WIN = "/opt/ml/projects/winner_snapshot/nemotron-master"
sys.path.insert(0, WIN)
from reasoners.store_types import Problem
from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.cipher import reasoning_cipher
from reasoners.equation_numeric import reasoning_equation_numeric
from reasoners.cryptarithm import reasoning_cryptarithm
from reasoners.gravity import reasoning_gravity
from reasoners.numeral import reasoning_numeral
from reasoners.unit_conversion import reasoning_unit_conversion

GEN = {
 "gravity": reasoning_gravity, "unit_conversion": reasoning_unit_conversion,
 "cipher": reasoning_cipher, "bit_manipulation": reasoning_bit_manipulation,
 "numeral": reasoning_numeral, "equation_numeric_deduce": reasoning_equation_numeric,
 "equation_numeric_guess": reasoning_equation_numeric,
 "cryptarithm_deduce": reasoning_cryptarithm, "cryptarithm_guess": reasoning_cryptarithm,
}

def extract_answer(text: str) -> str:
 m = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text or "")
 return m[-1].strip if m else ""

def compare_answer(stored: str, predicted: str) -> bool:
 stored = (stored or "").strip; predicted = (predicted or "").strip
 if re.fullmatch(r"[01]+", stored):
 return predicted.lower == stored.lower
 try:
 sn = float(stored)
 try: pn = float(predicted)
 except ValueError: return False
 return abs(sn - pn) <= 1e-2 * max(1.0, abs(sn))
 except ValueError:
 return predicted.lower == stored.lower

# cross-check winner gold vs official train.csv
import csv
csv.field_size_limit(10**7)
train = {r["id"]: r["answer"] for r in csv.DictReader(open("data/train.csv"))}

per = collections.defaultdict(lambda: dict(total=0, trace=0, correct=0, goldmismatch=0))
files = glob.glob(f"{WIN}/problems/*.jsonl")
for f in files:
 p = json.loads(open(f).read)
 cat = p["category"]; d = per[cat]; d["total"] += 1
 if p["id"] in train and str(train[p["id"]]).strip != str(p["answer"]).strip:
 d["goldmismatch"] += 1
 fn = GEN.get(cat)
 if not fn: continue
 try:
 txt = fn(Problem.from_payload(p))
 except Exception:
 txt = None
 if txt is None: continue
 d["trace"] += 1
 if compare_answer(str(p["answer"]), extract_answer(txt)):
 d["correct"] += 1

print(f"problems scanned: {len(files)} (winner-gold vs train.csv mismatches flagged per row)")
print(f"{'category':24} {'total':>6} {'trace%':>7} {'CORRECT':>8} {'correct%':>9} {'goldmm':>7}")
tt = tc = ttr = 0
for cat in sorted(per):
 d = per[cat]; t = d["total"]; tt += t; tc += d["correct"]; ttr += d["trace"]
 print(f"{cat:24} {t:>6} {100*d['trace']/t:>6.1f}% {d['correct']:>8} {100*d['correct']/t:>8.1f}% {d['goldmismatch']:>7}")
print(f"{'TOTAL':24} {tt:>6} {100*ttr/tt:>6.1f}% {tc:>8} {100*tc/tt:>8.1f}%")
print(f"\nOVERALL solved-by-deterministic-reasoner: {tc}/{tt} = {100*tc/tt:.1f}%")
print("Residual (no correct trace) = the headroom the model must learn / we must crack.")
