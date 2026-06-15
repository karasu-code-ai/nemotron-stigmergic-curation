r"""Certified arm-vs-arm comparison for the ablation — anytime-valid, so we can add
seeds/problems and peek without inflating false positives, and we only claim
"consensus beats coverage" when it's certified above noise (the val-80 ranking
inversion that bit us — winner_corpus 0.60-val/0.85-LB vs bridge_full 0.55/0.86 — is
exactly the phantom this guards against).

Two tools:
 betting_certify — anytime-valid e-process (Waudby-Smith & Ramdas 2023, "betting"):
 a supermartingale under H0(mean<=0) so Ville's inequality bounds the
 sup-over-time false-certify rate by alpha. Use for incremental
 seed/problem accumulation.
 mcnemar_exact — fixed-sample paired binary test (the classic), for a single
 final read on a fixed val set.

Both take PAIRED per-problem correctness (same problems, two arms): a_correct[i],
b_correct[i] in {0,1}. We test whether arm A > arm B.
"""
from __future__ import annotations
import math


def betting_certify(diffs: list[int | float], alpha: float = 0.05) -> dict:
 r"""Anytime-valid one-sided certificate that mean(diffs) > 0, for diffs in [-1,1]
 (paired correctness differences a_i - b_i). Capital process

 K_t = prod_{i<=t} (1 + lam_i * d_i), lam_i >= 0 PREDICTABLE (uses only d_1..d_{i-1})

 Under H0 (true mean <= 0): E[1 + lam_i d_i | past] = 1 + lam_i*mean <= 1, so K_t is a
 non-negative supermartingale and Ville gives P(sup_t K_t >= 1/alpha) <= alpha. We
 therefore CERTIFY (reject H0) the first time K_t >= 1/alpha. lam_i is a capped Kelly-ish
 bet on the running mean (bet more when we've seen a positive edge), kept in [0, 0.9) so
 1 + lam*d > 0 for d >= -1."""
 K = 1.0
 s = 0.0 # running sum of diffs (predictable: excludes current)
 ss = 0.0 # running sum of squares
 n = 0
 peak = 1.0
 certified_at = None
 for d in diffs:
 run_mean = (s / n) if n > 0 else 0.0
 run_var = (ss / n - run_mean ** 2) if n > 0 else 0.25
 # approximate-Kelly bet lam* ~ mean/var (maximizes E[log(1+lam d)] for small edge),
 # predictable (past data only), capped to keep 1+lam*d > 0 for d >= -1.
 lam = max(0.0, min(0.9, run_mean / (run_var + 1e-6)))
 K *= (1.0 + lam * d)
 peak = max(peak, K)
 s += d; ss += d * d; n += 1
 if certified_at is None and K >= 1.0 / alpha:
 certified_at = n
 return {"certified": certified_at is not None, "certified_at_n": certified_at,
 "e_value": peak, "n": n, "mean_diff": (s / n) if n else 0.0,
 "threshold": 1.0 / alpha}


def mcnemar_exact(a_correct: list[int], b_correct: list[int]) -> dict:
 """Fixed-sample paired binary test. Discordant pairs: b01 = A wrong & B right,
 b10 = A right & B wrong. Under H0 (equal), each discordant pair is a fair coin →
 exact two-sided binomial p. Returns the one-sided p that A>B too."""
 b10 = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)
 b01 = sum(1 for a, b in zip(a_correct, b_correct) if a == 0 and b == 1)
 n = b10 + b01
 if n == 0:
 return {"b10_A_better": 0, "b01_B_better": 0, "p_two_sided": 1.0,
 "p_one_sided_A_gt_B": 1.0, "discordant": 0}
 # two-sided exact binomial(n, 0.5)
 def binom_tail_ge(k, n):
 return sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n)
 k = max(b10, b01)
 p_two = min(1.0, 2 * binom_tail_ge(k, n))
 p_one_A = binom_tail_ge(b10, n) # P(>= b10 successes) under fair coin
 return {"b10_A_better": b10, "b01_B_better": b01, "discordant": n,
 "p_two_sided": p_two, "p_one_sided_A_gt_B": p_one_A}


def certify_arms(a_correct, b_correct, alpha=0.05) -> dict:
 """Combined read for 'does arm A beat arm B' on paired per-problem correctness."""
 diffs = [a - b for a, b in zip(a_correct, b_correct)]
 acc_a = sum(a_correct) / max(len(a_correct), 1)
 acc_b = sum(b_correct) / max(len(b_correct), 1)
 return {"acc_A": acc_a, "acc_B": acc_b, "delta": acc_a - acc_b,
 "betting": betting_certify(diffs, alpha),
 "mcnemar": mcnemar_exact(a_correct, b_correct), "alpha": alpha}


# ---- self-validation + power profiling (tells us seeds/problems needed) ----
if __name__ == "__main__":
 import random
 alpha = 0.05; N = 1500
 # paired binary with a controllable concordance: rho = P(both same outcome).
 # discordant pairs carry the signal; edge = P(A right,B wrong) - P(A wrong,B right).
 def trial(base, edge, n, disc, rng, method):
 a, b = [], []
 for _ in range(n):
 if rng.random < 1 - disc: # concordant pair (no signal)
 v = 1 if rng.random < base else 0; a.append(v); b.append(v)
 else: # discordant: split by edge
 if rng.random < 0.5 + edge / (2 * disc):
 a.append(1); b.append(0)
 else:
 a.append(0); b.append(1)
 if method == "bet":
 return betting_certify([x - y for x, y in zip(a, b)], alpha)["certified"]
 return mcnemar_exact(a, b)["p_one_sided_A_gt_B"] < alpha

 rng = random.Random(0)
 print(f"[validate] alpha={alpha}, {N} trials/cell, discordant-rate=0.30")
 # H0 false-positive (edge=0)
 for meth in ("bet", "mcnemar"):
 fp = sum(trial(0.5, 0.0, 240, 0.30, rng, meth) for _ in range(N)) / N
 print(f" H0 edge=0 {meth:8} false-certify = {fp:.3f} (<= {alpha})")
 # power across effect size x n (n=240 ~ 1 seed's eval pool; 480 ~ 2 seeds)
 print(" power (certify rate) by true edge x n:")
 print(f" {'edge':>6} {'n=240 bet':>10} {'n=240 mcN':>10} {'n=480 bet':>10} {'n=480 mcN':>10}")
 for edge in (0.03, 0.05, 0.10):
 row = []
 for n in (240, 480):
 for meth in ("bet", "mcnemar"):
 row.append(sum(trial(0.5, edge, n, 0.30, rng, meth) for _ in range(N)) / N)
 print(f" {edge:>6.2f} {row[0]:>10.2f} {row[1]:>10.2f} {row[2]:>10.2f} {row[3]:>10.2f}")
 print(" => read the table: how big an edge we can DETECT at our sample size.")
