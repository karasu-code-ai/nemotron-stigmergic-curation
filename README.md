# Stigmergic Multi-Source Curation for Reasoning Distillation

A pre-registered **data-method** study for the NVIDIA Nemotron Model Reasoning Challenge
(Best Data / Synthetic-Data Method). We test *stigmergic multi-source curation* — treating a
population of independent reasoners' traces (gpt-oss-120B, gemma-4-31B, phi-4, Nemotron on-policy)
as an offline "pheromone field" and distilling cross-source consensus as the SFT target for a
LoRA (r=32) on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

**Headline (honest):** the naive curation/coverage levers do **not** beat the full-corpus baseline
(0.86 public LB) — and *why* they don't is the contribution: a falsification record of
pre-registered nulls (the full corpus is locally optimal in both curation directions), an
operational **LLM-independence (CID) detector** that separates same-model personas (mean CID 0.39)
from genuinely independent families (0.63) from outputs alone, and an **output-identifiability
census** locating the task's hard residual as a grammar-coverage gap with a constructive
identifiability component.

## Read this first
- **`docs/WRITEUP.md`** — the full writeup (merit case + reproducibility record).
- **`docs/METHODOLOGY.md`** — method detail. **`DATA_SOURCES.md`** — full data/model disclosure (all public/no-cost).
- **`docs/RESULT_CID_independence_detector.md`** — the transferable positive.
- **`docs/CENSUS_R0_RESULTS.md`** + `census_*.py` — the determinacy census.

## Recipe
LoRA r=32 / α=32, seq 8192, completion-only loss, 9 target modules, eff-batch 16, lr 2e-4, bf16;
peft + transformers (no trl/bitsandbytes/4-bit); single DGX Spark (GB10). See `train.py`, `config.py`, `curate.py`, `generate.py`, `metric.py`.

## License
CC BY 4.0 — see `LICENSE`. Attribution: NVIDIA Research (Competition Data) · Tong Hui Kang (open recipe) ·
bankoglu (GF(2) data) · Google/Microsoft/OpenAI (open weights) · Parunak & Brueckner (method lineage).
