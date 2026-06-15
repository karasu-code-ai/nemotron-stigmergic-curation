# Data & code sources — disclosure + attribution

For competition compliance (rules §2.4 CC BY 4.0 attribution; §2.6 External Data;
§3.6 public-code sharing) and the required solution writeup (§2.8.c). Everything below is
**publicly available and equally accessible to all participants at no cost**, satisfying
§2.6.

## Competition data
- **NVIDIA Nemotron Model Reasoning Challenge** train/test data — © NVIDIA Research,
 **CC BY 4.0**. *Attribution: NVIDIA Research team.* Kept on our own machines only;
 not redistributed (§2.4.b).
- **Model:** `nemotron-3-nano-30b-a3b-bf16` (NVIDIA), via Kaggle/HF.

## External data (public Kaggle / HF) — used to enrich the multi-source field
- `huikang/huikang-nemotron-repository-snapshot` — open pipeline (reasoners/,
 augmenters/, corpus.py) + augmented corpus. Public Kaggle dataset → deemed OSI-licensed
 under §3.6.b. **Bridge baseline; attribute Tong Hui Kang.**
- `huikang/nemotron-base-model-generation` — base-model sampled traces.
- `kishanvavdara/nemotron-reasoning-traj` — 30B reasoning trajectories.
- `kienngx/nemotron-30b-competition-trainingdata-cot-labels` — CoT + labels.
- `bankoglu/hard-families-cot` — **987 bit_manipulation solver-CoT (GF(2)-affine derivations) on the
 competition problems; cracks 95 of the reference solver bit_manip residual = +1.0pt coverage.** Public
 Kaggle, §2.6-clean. *Attribute bankoglu.* (vetted: CoT-boxed==gold 200/200, genuine derivation not gold-leakage.)
- `konbu17/bit-manipulation-(synthetic-)cot` — family-specific CoT (the base=0.0 family).
- `ritwikakancharla/nemotron-math-v2-filtered-high` — volume — orthogonal-domain math volume (superseded by the mainstream version below).
- `mohamedamr992/replay-math` — `nemotron_math_1gb.jsonl` (~1GB). **The public "Replay_Data 0.86" recipe's
 orthogonal-domain corpus: ~2M math answer-tokens mixed as a REPLAY BUFFER into the open-recipe corpus**.
 The corpus-EXPANSION lever (we only tested curation-down/swap before). **Evaluated but DEFERRED: we held the 0.86 recipe fixed to isolate the DATA lever, so replay-math is NOT in the submitted corpus.**
 (Recipe tweaks that would ride with it — lr 3.5e-4, Cut Cross Entropy, shared-expert MoE weight-tying — are
 RECIPE, tracked separately so data-vs-recipe stays attributable.)
 > Each is an independent "agent" contributing traces to the stigmergic field.

## Models we run as additional field "agents" (open weights, local via ollama)
LICENSE-VERIFIED (2026-06-03). The question: do a model's OUTPUTS, used to TRAIN a submitted model,
carry restrictions that conflict with the competition's required clean CC BY 4.0 submission license
(§2.5/§3.6.c)? The open-weight families used for trace generation are all license-clean for
output→train→redistribute:
- **gpt-oss-120b** — OpenAI, **Apache 2.0**.
- **gemma-4-31b-it** — Google, **Apache 2.0** (confirmed on the model card).
- **phi-4 / phi4-reasoning** — Microsoft, **MIT** (fine-tune/distillation permitted without restriction).


## multi-source EXPANSION corpus (2026-06-12) — generated traces, submission-clean
The real-scale retrain (`bridge_corpus_multisource_broad.json`, 20,109 rows) = open-recipe base
(17,963) **+ 2,146 verified-correct reasoning traces WE generated** from three independent,
license-clean open-weight families over 549 problems. All generation is ours (local ollama); the open competition problems (`train.csv`, CC BY 4.0) never left our machines.
- **gpt-oss-120b** — OpenAI, **Apache 2.0** — 1,331 traces (broad full-corpus run). *Attribute OpenAI.*
- **gemma4-31b-it** — Google, **Apache 2.0** (confirmed) — 629 traces. *Attribute Google.*
- **phi-4** — Microsoft, **MIT** — 186 traces. *Attribute Microsoft.*
Build = `scripts_build_multisource_broad.py` (reproducible
from the raw trace files in `work/field_sources/`). Merge actions applied: val-drop
`8b12ff37`, correct-only filter (drops gravitational `\text{}`-transport + bit no-box rows).
Frontier-slice traces (Claude/GPT/Gemini/DeepSeek/Qwen) are **excluded** from (5 verified-correct,
census MIXED; repo carries analysis only per §2.4.b — confirmed).

## Generation & measurement APIs (research / analysis ONLY — not in the submitted corpus)
These external services generated traces used for **measurement and method development only**; none of their
outputs are in the submitted training corpus (frontier-slice traces + candidate programs stay local per
§2.4.b). Disclosed for §2.6 transparency; the submission corpus remains open-weights + competition data (above).
- **Independence slice — 5 frontier families (~$223):** Anthropic (`claude-opus-4-8`), OpenAI
 (`gpt-5.5`), Google AI Studio (`gemini-3.1-pro-preview` → swapped to `gemini-3.5-flash`), and OpenRouter
 (`deepseek/deepseek-v4`, `qwen/qwen3.7-max`). Used as independent *estimators* for the cross-family
 agreement measurement — research, never training data.
- **Generative proposer:** OpenRouter (`google/gemini-3.5-flash`, `qwen/qwen3-coder-next`,
 `mistralai/codestral-2508`) — verifier-gated candidate programs over the residual; candidates stay local (§2.4.b).
- **NVIDIA NIM** (`integrate.api.nvidia.com`, Nemotron) — on-policy self-generation probes; free preview tier.
(Frontier APIs above were used strictly as **measurement estimators** for the independence slice and method development — never as submitted training data, per §2.4.b.)

## Intellectual / literature sources
The method's lineage and the writeup's framing draw on a reviewed reading corpus, not training data:
- **Method lineage:** the Parunak/Brueckner polyagent–stigmergy program (interpreted-vs-generated
 signals, diversity-weighted voting / CID, symmetric stigmergy) — published anchors cited below.
- **Writeup §6 citations:** Breiman 2001 (Rashomon) · Marx, Calmon & Ustun 2020 (predictive multiplicity) ·
 Krogh & Vedelsby 1995 (ambiguity decomposition) · Goldman & Kearns 1995 (teaching dimension) · Hong & Page
 2009 (interpreted/generated signals) · Parunak et al. AAMAS 2013 (aggregating agent estimates) · Wang et al.
 2022 (self-consistency).
- **Full bibliography:** see the §6 citations in `WRITEUP.md`.

## Code provenance
- Our pipeline (generate/curate/field/train/evaluate, the stigmergic curator) — ours.
- The open recipe's pipeline borrowed for the bridge — attributed above; our submission's
 open-source code is OSI-licensed (CC BY 4.0, §3.6.c).
