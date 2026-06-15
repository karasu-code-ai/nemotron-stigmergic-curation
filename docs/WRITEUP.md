# Stigmergic Multi-Source Curation for Reasoning Distillation — a pre-registered data-method study

*Writeup for the NVIDIA Nemotron Model Reasoning Challenge (submitted to the Data / Synthetic-Data Method track).
Submission account: kyrenic. Model: LoRA r=32 SFT of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
This document is the merit case + reproducibility record. v2 (2026-06-11) — rebuilt around the
falsification record; the v1 "win-claim" skeleton it replaces is preserved in git history.*

---

## In plain terms

Different AI models, given the same hard reasoning puzzles, tend to make *different* mistakes. We tested a
simple idea: instead of training a new model on one AI's answers, pool answers from **several independent**
AIs and keep the reasoning they *agree on* — betting that independent agreement is a better training signal
than any single source. We ran it as a strict, honest experiment: for every idea we wrote down the
success criterion **before** seeing the result, so we couldn't fool ourselves.

**The honest headline: the clever curation did *not* beat simply training on the whole messy pile of data —
and working out *why* is the real contribution.** Three things came out of it: (1) a record of ideas that
*sounded* good but provably didn't help (and what that teaches us); (2) one genuinely useful, reusable tool —
a way to measure whether a group of AIs is *actually* independent (so you know whether their agreement means
anything), using only their answers, with no access to the models' internals; and (3) a precise diagnosis of
*why* the hardest puzzles are hard — the few worked examples simply don't contain enough information to pin
down the intended rule, so no amount of clever searching reliably recovers it.

## Quick glossary (acronyms and terms used below)

- **LoRA** — *Low-Rank Adaptation*: a lightweight fine-tuning method that trains a small add-on instead of the whole model.
- **SFT** — *Supervised Fine-Tuning*: training a model on input→output examples. (We never retrain from scratch.)
- **Stigmergic curation** — choosing training data by what *independent* sources converge on; named after how
  ants coordinate indirectly through traces left in a shared environment.
- **"Pheromone field"** — the metaphor for that shared pool of independent reasoners' traces.
- **CID** — *cross-problem divergence*, our detector: do two models *disagree across many problems* (genuinely
  independent) or *echo each other* (a shared prior)? Measured from answers alone.
- **Consensus / voting vs. averaging** — combining answers by agreement (voting) vs. by blending (averaging);
  a recurring theme is that voting carries the signal while averaging washes it out.
- **Determinacy census / "R0"** — re-running the reference solver's own rule set to *count* how many hard
  problems actually have a single determinable answer.
- **Identifiability / grammar-coverage gap** — when the given examples don't pin down one rule, so even a
  correct-*looking* guess can disagree with the intended answer.
- **TIES** — *Trim, Elect-sign, Merge*; **SVD** — *Singular Value Decomposition*: two ways to combine the
  *weights* of separately-trained models.
- **GF(2) / bit-manipulation** — *GF(2)* is arithmetic over the two binary values (XOR/AND); a puzzle family
  solved by such binary rules.
- **Gold** — the competition's official correct answer for a problem.
- **The (hard) residual** — the puzzles the reference solver could not crack; where the remaining difficulty concentrates.
- **Frontier models** — the strongest proprietary models (e.g. Claude, GPT, Gemini).
- **MoE / "A3B"** — *Mixture-of-Experts*; the Nemotron model has 30B parameters but activates only ~3B per token.
- **LB** — *leaderboard*: the Kaggle public score (0–1).

---

## Abstract — what this submission actually is

We set out to test a specific synthetic-data method: **stigmergic multi-source curation** — treat a
population of independent reasoners' traces as an offline "pheromone field," read which reasoning the
*independent* sources converge on, and distill that curated agreement back by SFT as a better training
target than a single trace or a length/random heuristic.

We ran it as a **pre-registered program**: every proposed leaderboard lever was gated behind a diagnostic
and, where possible, a second-seed confirmation, with the read written *before* the number. **The headline
honest result is that the naive curation/coverage levers do not beat the full-corpus baseline on this task —
and *why* they don't is itself the contribution.** Mapping that boundary produced (a) a falsification record
of pre-registered nulls that reshape the problem, (b) one clearly positive, transferable tool — an
operational *independence detector* for LLM estimators that separates same-model personas (mean CID 0.39)
from genuinely independent model families (0.63) from outputs alone, no weights — and (c) a sharp diagnosis
of *where* the task's difficulty actually lives: a **grammar-coverage gap** (the worked examples are not
expressible in the reference hypothesis class) with a **constructively-demonstrated identifiability
component**, confirmed by four independent instruments. The deliverable is **negative
results that reshape the problem, plus a validated measurement method** — offered honestly as such, not as a
points-beating curation trick.

Anchor submission: the full-corpus adapter at **0.86** public LB.

---

## 1. The method and the recipe (reproducibility)

**Substrate.** We forked an open-source recipe (Tong Hui Kang's public Nemotron pipeline) for reproducibility: LoRA (Low-Rank Adaptation) r=32, α=32, sequence length 8192, completion-only loss, 9 target
modules, effective batch 16, learning rate 2e-4, bf16 (bfloat16) precision, on the official chat
template (`enable_thinking` + boxed-answer suffix). Trained on a single DGX Spark (GB10, unified memory),
the Hugging Face peft + transformers libraries, **no trl / bitsandbytes / 4-bit quantization** (a constraint of
the Blackwell GPU). One LoRA at a time (the machine runs out of memory with two concurrent 30B models).
Evaluation is greedy decoding, single-sample, using the official metric verbatim.

**The curation method under test.** Generate traces from multiple *independent* model families
(gemma-4-31B / gpt-oss-120B / phi-4 / Nemotron on-policy), normalize formatting and remove near-duplicates (via MinHash) from the traces,
and score candidate training fragments by **cross-source** agreement (counting distinct *sources* that
converge, not distinct traces) — with a γ-discount (gamma-discount) that down-weights agreement explained by within-problem
boilerplate. The curated corpus is the Supervised Fine-Tuning (SFT) target. Source-aware curation, frontier-weighting, and a
diversity arm were compared at equal budget against coverage/random baselines.

Full data-source disclosure (all public / no-cost, license-clean) in `DATA_SOURCES.md`; the submission code
path is open-sourced under CC BY 4.0: **https://github.com/karasu-code-ai/nemotron-stigmergic-curation**

---

## 2. The falsification record (the spine)

Every lever below was pre-registered with a GO/HALT read *before* the number, and reported as-is. The
discipline: **measure before you build; a null is a finding.**

**(a) The degeneracy finding — the run-independent floor.** *Persona diversity is not source diversity.*
Agreement among samples from one model family reflects a shared prior, not independent convergence — a
consensus signal over correlated estimators degenerates. This held in the Part-9 single-source null and is
the conceptual floor of the whole program. **We then made it *operationally measurable* — see §3.**

**(b) Consensus-beats-coverage — HALTED on the second seed.** Seed-0 showed the consensus+frontier bundle at
Δ=+0.100, p=0.019 over a coverage/random baseline. The pre-committed rule required seed-1 confirmation;
**seed-1 did not reproduce** (Δ=−0.037, p=0.97, discordant split sign-flipped 1/4; the baseline itself rose
0.375→0.487 — *on the n=80 subset accuracy, not the 0–1 LB scale* — to meet the arms). At n=80 the between-arm variance swamps the effect — seed-0 was a favorable
draw. Reported as a null. (This is the +0.100-trap the program is built to catch.)

**(c) Coverage vs corroboration — and its refinement.** Independent *same-class* sources (open-weights, ~30B
tier) **corroborate** the solver (agree where it's already right) but do **not extend** it: a 2nd source
added +9 problems, a 3rd added **+0** — coverage saturates at the strongest single source.
**Refinement (the independence slice):** five *genuinely independent frontier families*
(Claude/GPT/Gemini/DeepSeek/Qwen) on the hard residual **did extend** — k=5/30 new solves, one-per-problem
(4 families, GPT=0) pure complementarity — where same-class sources could not. So extension requires *genuine* lineage
diversity, not same-class diversity. **But the same families almost never *agreed*** (no 3-way coalition
formed on any of 30 problems): **independence buys *coverage*, not *agreement*.** Consensus-based curation
has nothing to grip where sources don't converge.

**(d) Corpus surgery is monotonically harmful — a dose-response.** Holding the recipe fixed and varying only
the corpus: full messy corpus **0.86** > curate-down to the open recipe's curated 7,830 **0.85** > correctness-filtered
(drop all metric-wrong traces) **0.82** > GF(2)-solver bit_manip swap **0.70**. Every "improvement" hurt.
The −4pt from correctness-filtering is the load-bearing datum: **the messy/wrong traces are load-bearing for
format learning** — consistent with a *quasispecies / stochastic-resonance* prediction — that a tuned
fraction of noisy examples is actually load-bearing, so there is a sweet spot in the middle: remove the noise
and you fall off the good side of the curve, which the ladder above traces out.

**(e) Weight-space: voting beats averaging, but can't exceed the parents.** Merging two independently-trained
adapters by naive (averaging-class) SVD weight-fusion was catastrophic (0.15 at 50/50); a **TIES sign-election**
("voting") merge of the *same* two adapters at the *same* mix recovered to **0.68/0.70** (+0.53). So *voting beats averaging* in weight space — but both stay **below the parents** (0.86/0.85): independently-trained
adapters settle in different *loss basins* (different valleys of the error landscape), and post-hoc merging is lossy regardless of method. **We then tested
the obvious rescue — warm-starting for basin alignment — and it failed too.** A bit_manip specialist
warm-started from the correct-only parent (so *same basin*) genuinely learned the skill in isolation: it
solved 18% (36/200) of held-out GF(2)-residual bit_manip that the 0.86 generalist got **0%** of (a paired McNemar significance test, p≈3×10⁻¹¹ — overwhelmingly unlikely by chance). **But TIES-merging that specialist back into its parent scored 0.67 LB — *below* the
0.82 parent and no better than the cross-basin merges (0.68/0.70).** So warm-start/basin-alignment is
necessary but **not sufficient**: it makes specialization *real* but not *mergeable*. The composition holds
for the **standalone specialist**, not for the **merged artifact** — post-hoc weight merging sheds general
competence regardless of basin alignment. (This also explains the gf2-0.70 down-move: the generalist
genuinely cannot do GF(2)-residual bit_manip; a *narrow* specialist can, but it tanks the other families —
whether you reach it by a mixed retrain or by merging.) The durable weight-space results are therefore
*voting > averaging* and the *standalone-specialist* effect — **not** merge > parents, which never holds.

**(f) The generation walls, and the difficulty diagnosis (grammar-coverage, with an identifiability component).** We tried to *generate* the missing hard
traces. Solver-augmented cryptarithm (letter-arithmetic puzzles) was already baked into the 0.86 corpus (a newer public solver added
only +32, in a family that doesn't generalize). On-policy self-generation (the model generating its own training traces) could not produce correct
hard-family traces (the model's own capability ceiling). Bounded "act-don't-deliberate" prompting beat
runaway reasoning on solvable problems but got **0/12** on the hard residual; a 6-strategy prompt vocabulary
got **0/12**; a **tool-equipped code agent** (sandboxed Python execution) that genuinely ran search code got
**0/6**. Reasoning, prompt-diversity, *and* tools all bottom out → the hard residual is an **identifiability
gap, not a capability gap**: the few worked examples don't uniquely determine the rule (any consistent rule
found ≠ gold), which is exactly why the reference solver flagged these `rule_unknown`. **This unifies the
nulls** — agreement is uninformative and coverage can't extend precisely *because* the residual is not
reachable by search of any kind we tried (for most of its mass a grammar-coverage gap; with a constructively
demonstrated identifiability component, §3). **We then made this precise with a pre-registered determinacy census**
(`census_cryptarithm.py`): re-running the winning solver's own rule grammar in *enumerate-and-count*
mode (not find-first) over the 559 `cryptarithm_deduce` problems it marked `rule_unknown`, the grammar
that solves the easy strata (rule_found **54/54**, uniquely and correctly) leaves **94.6% (529/559) of
the residual with no consistent rule at all**, and a deliberately broadened 16-operator search recovers
gold for only **10%** even when it fits the examples. So the sharpened, honest statement is that the
residual is a **grammar-coverage gap** — the worked examples aren't expressible in the solver's
hypothesis class, and a natural broadening fits them only with rules that disagree with gold on the
query. Whether the *expressible* fraction is *additionally* underdetermined is decidable only against
the generator's (unpublished) grammar; we therefore report grammar-coverage as what the data supports
and leave program-synthesis recovery as registered future work. (Census record:
`docs/CENSUS_R0_RESULTS.md`.)

*The generative-proposer wall.* The strongest constructive test of the diagnosis:
instead of asking the model to solve blind, we ran an LLM→Python program-synthesis search (the model writes candidate Python rules — three independent
model families, K=4 candidates each, with two counterexample-guided (CEGIS) repair attempts — accepted only
if the program exactly reproduces every worked example) over the full 559-residual. It reached gold on **9/559** (frame gold@K = 0.016) vs the
fixed enumerator's **7/559** (0.013) — and a *perfect* selector over its candidate sets tops out at **1.6%**
(gold in-set on 9, top-1 on 2), far under the 15% threshold, so this is a **coverage wall, not a selection
wall**. Controls read **1.000/0.848** (the proposer nails determined problems), so the residual null is the
substrate, not the method; and because the read was committed into the scorer *before* the number, the
"just need a stronger proposer" escape is structurally foreclosed. This is the **fourth** independent
instrument convicting the residual as **not-solvable-by-search** (after the slice's empty coalitions, the CID
*saturation on cryptarithm* — a distinct facet from the §3 positive that CID *detection* works — and the
tool-agent's 0/6) — and the first from the constructive side. Search of every kind we tried (consensus,
coverage, tools, generative proposal) bottoms out; that the residual is *underdetermined* rather than merely
beyond our hypothesis classes is the interpretation §2f bounds (grammar-coverage is what the data proves;
full underdetermination is decidable only against the generator's grammar). The structural foreclosure here —
*pre-registration compiled into the scorer, not merely written down* — is itself a reusable practice we
recommend for replay-validated rule-induction and red-teaming benchmarks. (Record: `hs_score.json`, commit
f0a0a66.)

**(g) The leaderboard itself has ~±1pt run-to-run variance.** An accidental double-submission of the
identical adapter scored 0.70 then 0.71 — vLLM greedy non-determinism flips borderline problems per
generation run. So *every* ±1pt delta in this study (including re-submissions of the same file) sits at the
noise floor; the analysis is built on signs and pre-registered reads, never point estimates.

**(h) The expansion arm — does *adding* diverse correct traces help? (No — on both variants.)** The corpus-surgery
ladder in (d) only ever curated *down*; every removal or swap lost points. Its untested mirror is
*expansion*. Holding the recipe fixed, we added **2,146 verified-correct reasoning traces from three
independent open-weight families** (gpt-oss-120B / gemma-4-31B / phi-4) over 549 problems to the full 0.86
corpus (→ 20,109 examples) and retrained — the real-scale version of the multi-source curation thesis, all
generation ours and submission-license-clean. **Pre-registered read (written before the number):** **>0.86**
completes the diversity symmetry — both over-curation *and* the right addition move the needle, i.e. messy
*diversity* (not merely messy *volume*) is load-bearing; **=0.86** = multi-source corroboration is
leaderboard-neutral (consistent with the coverage-saturation of (c)); **<0.86** = the 549-problem
reweighting cost outweighs the added diversity. The census (f) constrains expectations up front: the *hard*
families are a grammar gap traces cannot reach, so any lift must come from generalization on the *solvable*
families, not from cracking the residual. **RESULT (Jun 15): expansion does NOT help — a clean
pre-registered negative on both variants.** The from-scratch expansion scored **0.85, then 0.86 on a
re-roll** — i.e. *within the ±1pt noise band of the 0.86 baseline (g), statistically indistinguishable → no
lift*; a warm-started continue-SFT variant that *preserves the 0.86 basin*
and adds an even broader cross-source set (1,318 problems) scored **0.83** — a clear degradation. So the
`<0.86` branch fires: the reweighting/quality cost of the added problems outweighs the diversity, and
continued-SFT expansion actively drifts the basin downward. **This resolves the diversity symmetry: messy
*volume* is load-bearing (correctness-filtering *down* costs −4pt, (d)), but adding diverse correct traces
*up* does not extend the lever — coverage saturates ((c)) and the hard residual is a grammar gap traces
can't reach ((f)). The full corpus is locally optimal across BOTH curation directions** — the cleanest
statement of the central null. (Reads were pre-committed before each number.)

---

## 3. The positive results — a method that *works*

**An operational independence detector for LLM estimators (the core positive).** The open question
underneath all consensus methods — *is a set of estimators independent enough that their agreement is
informative, without knowing their internals?* — is the question of independence-without-internals, raised in private correspondence with H. Van Dyke Parunak. We operationalized a within/cross-problem-divergence statistic and validated it on real LLM data:
**pairwise cross-problem divergence (CID) cleanly separates same-model personas (mean CID 0.39) from
genuinely different model families (0.63)** — i.e., you can detect persona-pseudo-diversity vs real
model-independence *from the outputs alone, no weights, no machinery.* This is the degeneracy finding turned
from observation into measurement, and it is task-transferable.

**Independence extends the frontier — when it is genuine.** The slice's k=5/30 one-per-family complementarity
(§2c) is, against the corroboration-saturation prior, the program's strongest pro-independence datum.

**Bounded commitment beats runaway deliberation.** On *solvable* problems, an "infer-rule, apply, answer"
prompt solved cases that the same model's unbounded chain-of-thought (step-by-step) reasoning never stopped to give a boxed answer — a fast-intuitive-vs-runaway-deliberative effect (System 1 vs System 2), and a hint that the right unit is a fast bounded agent, not a deliberator.

**The propose-verify-repair loop reaches gold on determined problems** — the keeper (a standout case we retained as evidence) demonstrates this twice
under the final committed instrument, producing a ~200-char string-level program outside R0's arithmetic
grammar that regenerates all examples and matches gold. This does two jobs simultaneously: it proves the
rules are reachable by a generative proposer (the wall is non-enumerability, not unreachability —
grammar-escape is possible), and it serves as the determined-substrate positive control that makes the
residual null attributable to the problems, not the method. The census 2×2 makes this precise: the keeper
occupies the determined-correct cell (identified × gold-match); our 16 occupy the determined-wrong cell
(identified × gold-mismatch) — adjacent cells, opposite gold outcomes, identical "identified" status. The
only thing separating them is whether the examples contained enough information to rank gold first. That is
the identifiability wall, stated constructively: the loop can reach the rule, but the examples can't tell it
which rule to keep.

---

## 4. Honest limitations

- **Single-pass.** We never closed the generate→curate→train→regenerate loop; a one-pass design cannot show
 the compounding selection effect. The iterated, diverse-strategy loop is named as future work, not claimed.
- **LB variance** (§2g) caps the resolution of every comparison at ~±1pt; the final standing is on the
 private split.
- **Infrastructure honesty.** Single DGX Spark, one LoRA at a time, greedy single-sample; the cross-family
 collection ran against four API providers and cost ~\$223 with non-trivial transport firefighting (logged
 as method, not hidden).
- **The aggregation (consensus) operator is, for this task, the wrong one** — coverage/voting, not averaging-
 class consensus, is where any signal lives, and on the underdetermined residual even those have nothing to
 grip.

## 5. One thread across three substrates (for the discussion)

The same *convex-hull* principle — averaging can only produce a result *between* its inputs, whereas voting
can land *outside* them (the structural reason voting can beat averaging) —
appears in **forecast space** (Parunak's pairwise voting, +8% over linear pooling), in **weight space** (our
TIES sign-election ≫ naive averaging, §2e), and was *probed* in **trace space** (the slice — though
consensus never formed on the underdetermined residual, so it's vacuous there). Two of three substrates
confirm; the third is undefined for an interpretable reason. That cross-substrate convergence — anchored to
the method's originating literature — is the durable scientific yield this competition produced.

---

## 6. Related work — where this sits in the literature

We name the theoretical anchors rather than re-derive them; the contribution is their *convergence* on one
reasoning benchmark.

- **Identifiability / underdetermination — Rashomon sets, predictive multiplicity, teaching dimension.**
 That many rules fit the worked examples yet disagree with gold on the query is a *Rashomon* phenomenon at
 the rule level (Breiman 2001): a set of near-equally-consistent hypotheses the data cannot separate. Their
 disagreement on the held-out query is *predictive multiplicity* (Marx, Calmon & Ustun, ICML 2020). The
 root cause is a *teaching-dimension* shortfall (Goldman & Kearns 1995) — the worked examples lie below the
 count needed to single out the rule within the hypothesis class — which is exactly what the determinacy
 census (§2f) quantifies and what the reference solver's `rule_unknown` label names. This
 Rashomon/teaching-dimension framing characterizes the *identifiability component* specifically — the slice
 where a consistent rule exists but the examples can't single out gold; for the majority of the residual the
 census shows the simpler **grammar-coverage** gap (no consistent rule in the hypothesis class at all).
- **Diversity, voting, and the independence detector — ensemble ambiguity, interpreted-vs-generated
 signals.** The Krogh–Vedelsby ambiguity decomposition (1995) — ensemble error = mean individual error −
 ensemble diversity — is the formal reason coverage/voting, not averaging, carry the signal, and it casts
 the CID statistic (§3) as an output-only estimate of the diversity term. The convex-hull/voting principle
 recurring across our three substrates (§5) is Hong & Page's interpreted-vs-generated distinction (2009)
 and Parunak et al.'s pairwise diversity-weighted aggregation (AAMAS 2013) — the originating literature.
- **Self-consistency, and where it degenerates.** Within-model agreement as a correctness signal is the
 self-consistency default (Wang et al. 2022); the degeneracy finding (§2a, §3) is precisely that
 self-consistency over samples of *one* model is agreement among correlated estimators — informative on
 solvable tasks, vacuous on the underdetermined residual.

---

## Attribution
NVIDIA Research (Competition Data, CC BY 4.0) · Tong Hui Kang (open baseline recipe — the reproducibility
substrate) · bankoglu (hard-families GF(2) CoT data) · Google (Gemma) / Microsoft (Phi) / OpenAI (gpt-oss)
open weights · H. Van Dyke Parunak & Sven Brueckner (polyagent / stigmergy lineage; the interpreted-vs-
generated and convex-hull/voting framing). Full external-data/tools disclosure: `DATA_SOURCES.md`.

**Code (CC BY 4.0):** https://github.com/karasu-code-ai/nemotron-stigmergic-curation
