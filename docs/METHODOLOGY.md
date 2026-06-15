# Methodology & Thesis: Stigmergic Curation of a Multi-Source Reasoning Field

*Acronyms are spelled out at first use; the full glossary is in [`WRITEUP.md`](WRITEUP.md).*

> A deeper methodological companion to the main writeup (`WRITEUP.md`). Written to stand on its own
> and to be reusable beyond this

---

## 1. Thesis

**A population's curated agreement is a learnable training signal.** If many semi-independent
reasoners attack the same problem and we can read off *which fragments of their reasoning they
converge on* — discounting boilerplate, weighting by learning-progress, preserving diversity —
then that converged signal, distilled back into one model by SFT (Supervised Fine-Tuning), is a better training target than
any single trace or a random/length heuristic.

This reframes stigmergy. The 2025–26 stigmergic-LLM literature uses a shared field for *online
coordination* (agents read/write a blackboard at inference). **We use the field as an offline
curation / distillation signal** — the trace population is the pheromone field, and curation is
the act of reading it. To our knowledge that specific use is novel.

The intellectual lineage is the Parunak/Brueckner **polyagent** program: a heavyweight cognitive
agent (here, an LLM) shadowed by a swarm of lightweight *ghost* agents (here, sampled reasoning
traces) whose aggregate behavior over a shared substrate is mined for signal. Symmetric stigmergy
 names our exact loop: environment → agent-state → environment, i.e.
**generate → curate → train → generate**.

## 2. The field and the three curation signals

Each problem accumulates a set of reasoning traces. We fragment traces and treat fragments as
pheromone **deposits** over a per-problem field. Three operators read the field:

1. **Consensus.** Fragments deposited by ≥2 traces of the same problem reinforce. Naive consensus
 over-counts generic boilerplate ("let me think step by step", "the answer is"), so we use a
 **γ-discounted, cluster-weighted** aggregation: a fragment's weight ∝ within-problem
 support / (γ + cross-problem support). High cross-problem frequency ⇒ boilerplate ⇒ discounted.

2. **Frontier (learning progress / zone of proximal development).** We weight problems by
 `pop_success · (1 − greedy_pass_rate)` — the population can solve it, the base model greedily
 cannot *yet*. This is a **real** signal (a dedicated greedy base-eval), not the
 population-always-solves proxy that conflates "easy" with "known." Grounded in the adaptive-walk
 result: a learner's asymptote is `p01/(p01+p10)`; curating at the edge of learnability
 maximizes the climb.

3. **Diversity preservation.** Argmax-1 selection collapses the field toward a single mode —
 collective cognitive convergence, which occurs *even under random mixing* and is only
 resisted by active variation (reward-the-rare, speciation, bridging subpopulations;). So an
 arm keeps the top-k *distinct* correct traces per problem rather than the single best.

**Arms (equal budget — the experimental control):**
B0 random · B1 shortest · A consensus · B consensus+frontier · C γ-discounted consensus · D
diversity-archive. Equal example count per arm isolates *which traces* from *how many*.

## 3. The central finding: single-source fields are degenerate

Our first real-scale ablation (Part-9: subset 120, k=6, 2 seeds, equal budget) put all six arms in
a 0.41–0.46 band — within the ±~0.05 noise floor at n=80/seed. **B (consensus+frontier) was the
top arm in both independent seeds (0.4625, zero cross-seed variance) and beat random in both** — a
consistent directional edge, but small.

The diagnosis is the methodological core of this work. The Part-9 "agents" were **four personas of
one base model**. Personas change the *prompt*, not the *weights* — so their agreements are
**correlated by shared parameters.** The within/cross consensus ratio presupposes
*independent* sources; when sources share weights, agreement reflects the shared prior, not
convergent truth, and the signal degenerates. This is precisely the cluster-weighted /
correlated-estimator failure the polyagent literature predicts.

**The fix is multi-source.** We rebuilt the field from genuinely independent model *families*:

| Agent | Family | License | Role |
|---|---|---|---|
| Nemotron-3-Nano-30B | NVIDIA (the competition model) | Nemotron OML | on-policy |
| gemma4:31b | Google | Apache-2.0 | independent; **86.8% solve rate** on our pool |
| phi4-reasoning-plus | Microsoft | MIT | independent reasoner |

Each generates traces on the *same* fixed problem pool (replicated from the evaluator's per-seed
split so the field is built over exactly the problems the ablation reads). Cross-source agreement
now reflects convergent reasoning across independent priors — the condition actually requires.
A val-contamination filter excludes any problem in a seed's held-out val before that seed can use
cross-source traces.

**Framing (regime transition vs. transport).** The single→multi-source move has a precise analogue
in the self-revising-discovery formalism (Self-Revising Discovery Systems for Science,
arXiv:2606.01444): re-sampling one model at temperature is *optimization within a fixed
representational regime* — every sample is a transport of the same prior. An independent model
*family* is, in their terms, an **"isolated new type" that receives empty transport** — it cannot be
reached by reinterpreting the existing regime; it injects genuinely new representational content.
Cross-source consensus is therefore a *regime-expanding* operation, not a fixed-regime one — which is
why it escapes the correlated-agreement ceiling that re-sampling cannot.

## 4. The bridge: a competitive substrate ("their data, our stack")

Curation is a *second-order* lever — it can only help if the underlying corpus and training are
strong. The competition's first-order lever (publicly established) is rule-correct synthetic volume:
per-family procedural CoT (chain-of-thought) generators → augmented corpus → LoRA (Low-Rank Adaptation) SFT. That recipe is open. We
reproduced it **on our own stack** (peft + transformers Trainer, DGX Spark / GB10, chat-template
completion-only loss, r=32 with `lm_head` targets, dynamic per-batch padding) rather than the
the open recipe's training harness. This gives a strong, license-clean baseline onto which the stigmergic
curator drops in as the corpus policy — and a real leaderboard number to anchor the method.

## 5. Results and calibration

Greedy single-sample, official metric (vLLM, boxed extraction). Our internal val-80 *undercounts*
the official LB (smaller, possibly-harder slice + a truncated generation budget vs. the official
7,680 tokens):

- base Nemotron: ~0.34 val · official demo LoRA: 0.53 LB
- **bridge (~0.3 epoch, our stack): 0.5375 val → 0.77 LB** — calibration gap **+0.23**.
- bridge_full (~1.3 epoch): eval pending (submission).
- top-10% award cutoff ~0.85; the public open-source recipe ~0.86.

0.77 from ~0.3 epoch establishes the trajectory toward the top-10% band. The stigmergic ablation's
contribution is measured *on top of* this substrate (cross-source field; in progress).

**Statistical discipline (ga-06):** we report 2-seed mean±std distributions, not best-of-N, with a
paired per-problem analysis, because the predicted curation effect (~single-digit %) sits near the
marginal-accuracy noise floor. A swarm's output is a distribution, not a point (/).

## 6. Limitations and the simulated-environment argument

- **Second-order effect.** We do not claim curation alone wins the leaderboard; we claim a *novel,
 measurable* data-curation method on a competitive substrate.
- **No literal shared environment.** Rule-induction puzzles have no real shared substrate, so we
 *simulate* the stigmergic field over traces. We argue this is the **hardest** case, not a weakness:
 in domains with a *literal* field — RF time-frequency maps, network connection graphs
 (PCAP/Zeek), the multi-sensor environment of a physical space — the substrate is real and shared,
 and the mechanism should work *better*. The competition is the proof-of-mechanism in the least
 favorable setting.

## 7. Attribution

NVIDIA Research (Competition Data, CC BY 4.0); Tong Hui Kang (open baseline recipe); Google (Gemma),
Microsoft (Phi) open weights; H. Van Dyke Parunak & Sven Brueckner (the polyagent/stigmergy lineage
this method is built on). Code: this repo, CC BY 4.0. See `DATA_SOURCES.md`.
