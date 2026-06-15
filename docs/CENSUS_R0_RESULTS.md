# Determinacy Census — Stage 1 RESULTS (R0 grammar + R1-discovery + slice retrodiction)

*Acronyms are spelled out at first use; the full glossary is in [`WRITEUP.md`](WRITEUP.md).*

*Run 2026-06-11. Instrument: `census_cryptarithm.py` (the R0 reference-solver grammar) + `census_r1_discover.py` (R1).
Reads were written BEFORE the
numbers. This file is the frozen Stage-1 record; the competition LB value of all of it = zero.*

---

## What was measured

The `cryptarithm_deduce` family, by the reference solver solver's own status field:
**rule_unknown = 559** (the residual), **rule_found = 54**, **hypothesis_formed = 46**.
R0 = the reference solver strict grammar (bijective symbol→digit; ops {add, abs_diff, mul, concat,
rev_concat}). The census change vs the reference solver solver: **enumerate-and-count** the full set of
R0-consistent query answers per problem (the reference solver does find-first), so we can ask whether the
examples *identify* gold — not just whether *a* rule exists.

## R0 bins (the headline table)

| stratum | n | D-gold | D-wrong | U | X |
|---|---|---|---|---|---|
| **rule_unknown** (residual) | 559 | 0 | 16 | 14 | **529 (94.6%)** |
| rule_found | 54 | **54 (100%)** | 0 | 0 | 0 |
| hypothesis_formed | 46 | 38 | 0 | 2 | 6 |

**Instrument validity:** rule_found → 54/54 D-gold (unique, gold, rank 1). When R0 *can* express
the rule it pins it uniquely and correctly. So the residual's X is not an instrument failure.

## Pre-committed reads

- **(1) identifiability fraction (U+D-wrong)/n = 0.054.** Mechanically this is "<0.25 → weaker
 than." **But that reading is an artifact of X-dominance, not an adjudication of.** R0
 cannot express 94.6% of the residual, so it is not in a position to say whether the *expressible*
 part is identifiable. The honest statement: **the identifiability-gap hypothesis is neither confirmed nor weakened by R0** — it
 is *unresolved within R0*, which is exactly what read (3) routes on.
- **(3) X fraction = 0.946 ≫ 0.30 → RUN R1.** Fired hard. R0 is the wrong hypothesis class for
 the residual.

## R1 discovery (the contingency the pre-reg demands when X>40%)

`census_r1_discover.py`: a finite, structurally-named op library (16 binary ops — add, absdiff,
mul, sub, fdiv, mod, ±mod100, max, min, gcd, left, right, +concat/rev_concat — op-per-operator-
symbol). Defined structurally, never "whatever yields gold."

| R1 setting | examples-consistent | gold reachable |
|---|---|---|
| bijective, natural length | 2/40 | 0/40 |
| bijective + leading-zero pad | 1/40 | 0/40 |
| **non-bijective + pad** | 26/40 (65%) | **4/40 (10%)** |

**Read:** broadening the grammar lets *spurious* rules fit the examples (65% under the loosest
setting) but **almost never reaches gold (10%)**. So a natural R1 broadening does NOT recover the
residual: it fits the examples with rules that disagree with gold on the query. Distinguishing
"identifiability gap" (examples underdetermine gold) from "grammar gap" (the generator's rule
family is outside any simple class) **requires the generator's actual grammar (not public) or a
program-synthesis search**. That search (R2) is out of scope for this study.

**Refined finding (sharper than the pre-reg's statement):** the residual is not merely
"underdetermined within the reference solver grammar." The reference solver grammar provably covers the *solved*
strata (rule_found 54/54) and a reasonable broadening fails to recover the residual — so the
residual escapes both R0 and a natural R1. The operational claim the data supports is
**grammar-coverage**, not underdetermination; the underdetermination question is only
decidable against the generator's grammar, which the competition does not publish.

## Slice retrodiction — pre-committed read (2)

Classify the 30 independence-slice ids by census bin; M1-solved = the 5 frontier-family solves
(`2a25de27, 8c4f70b6, adcc6292, b9b5a2c1, d9575f79`).

- slice bins: X=22, U=5, D-wrong=1, D-gold=2.
- **derivable(M1-solved) = 2/5 (40%)** — both D-gold rank-1 (`b9b5a2c1`, `d9575f79`, both
 hypothesis_formed problems the reference solver punted on).
- **derivable(unsolved) = 2/25 (8%)**.
- **READ = MIXED.** The *direction* supports the mechanism (5× enrichment: frontier-solved problems
 are far more R0-derivable), but 2/5 < the ≥3/5 confirmation bar → "**not established, no
 smoothing**" (frozen read). Note 3 of the 5 frontier solves are R0-non-derivable → on those, the
 frontier families used knowledge beyond R0's grammar (the "generator-prior" the read flags).

## Stage 1.5 poison probe — input set (GPU, queued)

7 clean U-bin candidates where gold is among the R0-consistent answers (≥2 answers, gold present):
`26a2a1b8, 491b8ea5, 844f826c, c43b5a13, e3b06854, f7828fc1, fc7e4f9b`. These are the genuine
underdetermination-within-R0 cases — the right substrate for the probe (derive gold AND a non-gold
consistent candidate; test whether the LLM rationalizes both). Ambiguity certificates emitted in
`census_cryptarithm.json`.

---

## v2 census — discrimination mode (the v2 six-label schema; `census_v2.py`, manifest `6c3d46f1`)

v2 re-runs the same R0 enumerator under the v2 six-label schema (gold-free
`identifiability_status` × orthogonal `gold_relation`, frozen primitive manifest). It reconciles
exactly with v1 and adds the decisive Day-2 numbers.

**Residual (huikang rule_unknown, n_live=559, quarantined=0):**

| label | n | ↔ v1 |
|---|---|---|
| rule_identified | 15 | (D-wrong: unique rule, ≠gold) |
| output_identified | 1 | (D-wrong: ≥2 rules, 1 output, ≠gold) |
| **determined_wrong** (derived) | **16** | = v1 D-wrong; **all 16 identified cells are non-gold** |
| prior_selected | 7 | (U, prior breaks it) |
| unidentified | 7 | (U, top-2 tie) |
| **dsl_miss** | **529 (94.6%)** | = v1 X |
| R0_implementation_gap | **0** | (manifest clean) |
| **identified × gold_match (factory-eligible)** | **0** | factory eligible set EMPTY under R0 |

**Decisive Day-2 numbers (R0 proposer only):** `gold_output@1 = 0.000`, `@3 = 0.005`, `@5 = 0.009`,
`@any = 0.013`; `selector_regret = 7/7` (whenever gold is reachable it is never top-1 under the
support prior). **K-coverage 1.3% ≪ 15% → COVERAGE-WALL for the symbolic proposer** (the v2 §4
threshold), pre-answering the symbolic half of the fork.

**Calibration (rule_found, n=54):** 54/54 `rule_identified`, `gold_output@1 = 1.000`,
`R0_implementation_gap = 0` — the instrument has **zero blind spots**; dsl_miss on the residual is
genuine, not a missing primitive.

**Side finding — `hypothesis_formed` (n=46):** 35 rule_identified + 3 output_identified,
`gold_output@1 = 0.848`, dsl_miss=6 — a **near-solvable stratum** (the reference solver formed a hypothesis but
didn't commit; the census recovers gold for 84.8%). Candidate factory material the reference solver left on the
table.

**Two structural reads for the paper, locked before the LLM proposer runs:**
1. `determined_wrong = 16` is **convention-dependence evidence** — R0 finds a *unique* consistent
 answer that is *not* gold on 16 problems → gold relies on a generator convention outside R0 even
 where R0 is itself unambiguous.
2. **The factory's fate is now a separate experiment** — R0 yields 0 eligible traces, so whether
 the trace factory exists at all depends on the LLM→Python proposer lifting `gold_output@K` above the
 wall on the 529 `dsl_miss`.

## Artifacts
- `census_cryptarithm.py`, `census_cryptarithm.json` (v1, 659 records: bins, margins, certificates)
- `census_v2.py`, `census_v2.json` (v2 discrimination mode: six labels + manifest + gold_output@K)
- `census_r1_discover.py` (R1 discovery diagnostic)
- `census_r0.log` (run log)
