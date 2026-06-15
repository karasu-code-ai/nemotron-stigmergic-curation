# RESULT: An operational independence detector for LLM estimators (the CID result)

*Acronyms are spelled out at first use; the full glossary is in [`WRITEUP.md`](WRITEUP.md).*

## The question (independence without internals)
Given a set of estimators (here: LLM families) and their answers — **but not their internals** — can you
tell whether their *agreement* is informative (independent convergence) or hollow (correlated/shared prior),
*from the outputs alone*? This question was raised in private correspondence with H. Van Dyke Parunak. It is
the load-bearing unknown for any aggregation/consensus method.

## The result
Yes — via **pairwise cross-problem divergence (CID)** — a within/cross-problem-divergence statistic from that private correspondence, operationalized (`scripts_gamma_pair_existing.py`, selftest-validated; run on existing data, no model internals):

| pair type | mean pairwise CID (cross-problem divergence) |
|---|---|
| **same-model** Nemotron personas (4 prompt-personas of ONE model) | **0.39** |
| **genuinely different** model families (gemma4 / gpt-oss / phi4) | **0.63** |

CID cleanly separates the two (≈1.6×). A pair that **diverges across many problems but agrees on this one**
is corroborating (independent); a pair with low cross-problem divergence (same model) that agrees is just
echoing a shared prior. **You can detect the degeneracy — persona-pseudo-diversity vs real
model-independence — from the answer sets alone, no weights, no machinery.**

## Why it matters
- It is the **operational answer to the independence-without-internals question** — measured and validated on real LLM data.
- It is what makes any consensus/voting method *interpretable*: it tells you whether a given ensemble's
 agreement should be trusted before you aggregate.
- It is the operational form of this program's central **degeneracy finding** (persona diversity ≠ source
 diversity): the finding is now a *measurement*, not just an observation.
- Caveat (kept visible): in the same run, the *aggregation payoff* was null on the broad 240-pool
 (within-family agreement was the strongest correctness predictor; voting never beat best-single). The CID
 *detector* working and the *aggregation* not paying off are separate facts — the detector is the result;
 the null aggregation is consistent with coverage-saturation and the identifiability finding.

## Provenance / reproduction
- Estimator: `scripts_gamma_pair_existing.py` (Jensen–Shannon δ-divergence, leave-one-out CID).
- Data: existing 240-pool clean traces (gemma4/gpt-oss/phi4, 4 samples) + Nemotron 4-persona on-policy.
- Full numbers: `gamma_pair_existing_out.json`.
- The slice (5 distinct frontier lineages × cryptarithm-residual) is a **higher-quality replication** of
 the *detector* with a clean control.
