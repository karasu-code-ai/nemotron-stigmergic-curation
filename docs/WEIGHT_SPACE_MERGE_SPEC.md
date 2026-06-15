# WEIGHT-SPACE MERGE — spec + single-specialist pre-check (the gating experiment)

*Acronyms are spelled out at first use; the full glossary is in [`WRITEUP.md`](WRITEUP.md).*

*Same treatment as the independence slice: the experiment, the pre-committed reads, the design, the
schedule. This is the weight-space limb from the bigger arc (specialist LoRAs → merged rank-≤32 adapter
= "swarm compressed into a bounded substrate").*

## 0. Prior results that shape the design

Naive averaging-class merging of our two independently-trained adapters was **catastrophic**
(0.86 → 0.78 at 70/30 → **0.15** at 50/50): independently-initialized LoRAs live in **different loss
basins** with a barrier between them. Two design consequences, both mandatory:

1. **Basin alignment: specialists must be WARM-STARTED from one shared parent adapter**
 (`cfg.init_adapter` continue-SFT (Supervised Fine-Tuning), already supported in the bridge trainer) — never fresh-init.
 Merging stays within one basin; interpolation between a parent and its fine-tuned children is the
 benign regime (model-soup conditions), unlike the cross-basin merge we just falsified.
2. **The merge op is TIES (Trim-Elect-sign-Merge)-style, not plain averaging** (the voting-not-averaging analog in weight
 space): per-parameter sign election across specialists → drop disagreeing/low-magnitude entries →
 merge survivors → SVD (Singular Value Decomposition) re-factorize to rank ≤32. Small interventions only (the dose-response says
 this model punishes aggressive weight surgery).

## 1. The gate: single-specialist pre-check (run BEFORE building any merge)

**Question:** does ONE family-specialist beat the 0.86-generalist *on its own family*? If specialists
can't even specialize, the merge limb is dead regardless of merge quality.

**Design:**
- **Specialist:** warm-start from `bridge_correct_only` (the freshest clean parent once @600 lands);
 continue-SFT on a **bit_manipulation-only corpus** = the GF(2) (binary-field arithmetic) traces (987) + the reference solver's correct bit_manip
 (472 not covered by GF(2)), MINUS a held-out family slice (below). Short run: ~150 steps @ eff-batch 16
 (~2 epochs of ~1,450; the family corpus is small — do NOT 600-step it, that's 6+ epochs = memorization).
 ~4–6h on the box.
- **Held-out family slice: 200 bit_manip problems** (seeded random) excluded from the specialist's corpus;
 these are the eval set. (Was 150; bumped after a power simulation — see the table below.) The generalist
 saw them in training (unavoidable — it trained on everything), which BIASES THE TEST AGAINST THE
 SPECIALIST; flag this in the read (a specialist win is therefore conservative/strong evidence).
- **Eval:** local one-engine vLLM (`evaluate_adapters_vllm`, 0.75 util config), greedy, official metric:
 specialist vs generalist on the 200 held-out bit_manip problems. Paired per-problem → McNemar (a paired significance test) via
 `stats_certify.py`. Report disc + b10/b01 split with every p, never a bare Δ.

**Power calibration (simulated 2026-06-09, mcnemar_exact, base acc 0.87, 2000 trials/cell):** at n=150 a
TRUE 5pt edge has only 0.35–0.55 power (disagreement-rate 0.20–0.10); n=200 gives 0.41–0.66; an 8pt edge
is well-powered (0.79–0.99 at n=200). So a raw Δ-threshold read would be ambiguous half the time at
realistic effect sizes — the n=80 ablation lesson. Hence sign+split framing below, not point-estimate.

**Pre-committed reads (locked 2026-06-09, before any run — sign+split, NOT raw Δ):**
- **GO:** McNemar p<0.05 with a b10:b01 split ≥2:1 → specialists genuinely specialize → build the TIES
 merge (specialist + parent [+ a second specialist if time]) → rank-check ≤32 → submit as a lottery
 ticket and log the result.
- **DEAD:** powered null — disc ≥ 20 with near-even or flipped split → the limb is dead, consistent with
 saturation ("a generalist at convergence can't be locally beaten on a family it already learned").
 Otherwise the limb is closed.
- **AMBIGUOUS (do not call either way):** non-sig with disc < 20, or sig with a thin split → one more
 specialist (different family, e.g. equation_numeric) before any build decision. A single marginal read
 never gates a build (the +0.100 lesson). Also note: the result is conditional on THIS 200-draw — the
 seed-variance caveat from the ablation applies to the slice composition.
