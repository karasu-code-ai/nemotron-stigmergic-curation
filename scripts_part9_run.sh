#!/usr/bin/env bash
# PART 9 long run (corpus-driven redesign). 6 arms x 2 seeds, equal budget,
# real frontier signal, + arm C gamma-consensus + arm D diversity
# (/). Reports mean+/-std (ga-06), not best-of-N.
#
# TWO PHASES per seed (different venvs, one 62GB model on one GPU, sequential):
# A. GENERATE via vLLM (venv-vllm) -> caches traces to work/seed{N}/
# B. ALL via transformers (venv-ml) -> loads the cache (NO regen), then
# baseline -> curate(+greedy frontier) -> train 6 adapters -> eval 6 adapters
# Then aggregate to work/results_aggregate.json.
#
# Resumable: trace cache, greedy_pass, and adapters are cached per seed, so
# re-running skips finished work. Launch detached:
# nohup bash scripts_part9_run.sh > /tmp/part9.log 2>&1 &
set -u
cd /opt/ml/projects/nemotron-stigmergy

PYVLLM=/opt/ml/venv-vllm/bin/python
PYML=/opt/ml/venv-ml/bin/python
SEEDS=(0 1)

export HF_HOME=/opt/ml/hf-cache
export NEMOTRON_MODEL=/opt/ml/models/nemotron-3-nano-30b-a3b-bf16
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say{ echo "[part9 $(date +%m-%d_%H:%M:%S)] $*"; }

say "START seeds=${SEEDS[*]} (A: vLLM generate | B: transformers all)"
$PYML -c 'from config import Config as C;c=C;print("[cfg] subset=%d k=%d temps=%s budget=%d maxtok=%d seq=%d real_frontier=%s gamma_zeta=%s div=%d"%(c.subset_size,c.k_samples,c.temperatures,c.budget_examples,c.gen_max_new_tokens,c.max_seq_len,c.real_frontier_signal,c.gamma_zeta,c.diversity_per_problem))'

# ---------- PER SEED: A (vLLM generate) then B (transformers all) ----------
# Outer loop is per-seed so SEED 0's full results.json lands before SEED 1
# generation even starts -- you wake up to at least one complete seed.
for s in "${SEEDS[@]}"; do
 cache="work/seed${s}/traces_subset.jsonl"

 # --- Phase A: vLLM generation (fast path) ---
 if [ -s "$cache" ]; then
 say "SEED $s PHASE A: trace cache exists -- skipping generation"
 else
 say "===== SEED $s PHASE A: vLLM generate ====="
 ( export PATH="/opt/ml/venv-vllm/bin:/usr/local/cuda/bin:$PATH"; \
 export CUDA_HOME=/usr/local/cuda; \
 $PYVLLM -u run_ablation.py --stage generate --backend vllm --seed "$s")
 rc=$?
 if [ $rc -ne 0 ] || [ ! -s "$cache" ]; then
 say "SEED $s PHASE A FAILED (rc=$rc). If vLLM can't load Nemotron-H, fallback:"
 say " $PYML run_ablation.py --stage generate --backend transformers --seed $s"
 else
 say "SEED $s PHASE A done: $(wc -l < "$cache") traces cached"
 fi
 fi

 # --- Phase B: transformers curate/train/eval (loads cache, no regen) ---
 if [ ! -s "$cache" ]; then
 say "SEED $s PHASE B SKIPPED: no trace cache."
 continue
 fi
 say "===== SEED $s PHASE B: transformers all ====="
 $PYML -u run_ablation.py --stage all --backend transformers --seed "$s"
 rc=$?
 [ $rc -ne 0 ] && say "SEED $s PHASE B rc=$rc (continuing)" || say "SEED $s PHASE B done"
 # aggregate after EACH seed so partial results are always available
 $PYML -u scripts_aggregate.py --seeds "${SEEDS[@]}" 2>/dev/null || true
done

# ---------- AGGREGATE (distribution, not best-of-N) ----------
say "===== AGGREGATE ====="
$PYML -u scripts_aggregate.py --seeds "${SEEDS[@]}"
say "END"
