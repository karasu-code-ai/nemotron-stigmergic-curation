#!/usr/bin/env bash
# PART 9b RESCUE (2026-06-03, overnight). WHY: the original Phase B ran
# baseline+eval+frontier through the TRANSFORMERS backend, which generates
# CACHE-LESS on NemotronH ("requires an initialized NemotronHHybridDynamicCache;
# none provided") -> ~3.3 tok/s -> the 80-problem baseline alone was ~14h and ran
# BEFORE any training. The night would produce zero adapters.
#
# FIX (the flagged vLLM-eval architecture): route ALL generation/eval through
# vLLM (~25 tok/s, batched); use transformers ONLY for LoRA training (forward/
# backward, not autoregressive, so the no-cache bug doesn't bite). Per seed:
# 1 generate (vLLM, skip if cached)
# 2 curate (vLLM) -> greedy frontier greedy_pass.json + corpora [FAST]
# 3 train (transformers) -> 6 LoRA adapters (frontier+traces cached) [GPU work]
# 4 baseline(vLLM) -> baseline.json
# 5 eval (vLLM LoRA) -> results.json
# 6 aggregate
# All steps are tested run_ablation stages. Resumable (traces/greedy_pass/adapters
# /baseline all cached). Original launcher (scripts_part9_run.sh) is superseded;
# generation traces from its Phase A are reused, nothing regenerated.
set -u
cd /opt/ml/projects/nemotron-stigmergy
PYVLLM=/opt/ml/venv-vllm/bin/python
PYML=/opt/ml/venv-ml/bin/python
SEEDS=(0 1)
export HF_HOME=/opt/ml/hf-cache
export NEMOTRON_MODEL=/opt/ml/models/nemotron-3-nano-30b-a3b-bf16
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say{ echo "[part9b $(date +%m-%d_%H:%M:%S)] $*"; }
vllm_env{ export PATH="/opt/ml/venv-vllm/bin:/usr/local/cuda/bin:$PATH"; export CUDA_HOME=/usr/local/cuda; }

say "RESCUE START seeds=${SEEDS[*]} (vLLM gen/curate/baseline/eval | transformers train)"
$PYML -c 'from config import Config as C;c=C;print("[cfg] subset=%d k=%d budget=%d maxtok=%d real_frontier=%s"%(c.subset_size,c.k_samples,c.budget_examples,c.gen_max_new_tokens,c.real_frontier_signal))'

for s in "${SEEDS[@]}"; do
 cache="work/seed$s/traces_subset.jsonl"

 # 1. GENERATE (vLLM) -- skip if traces cached
 if [ ! -s "$cache" ]; then
 say "SEED $s GEN (vLLM)"
 ( vllm_env; $PYVLLM -u run_ablation.py --stage generate --backend vllm --seed "$s") \
 || { say "SEED $s GEN failed"; continue; }
 else
 say "SEED $s traces cached ($(wc -l <"$cache"))"
 fi

 # 2. CURATE+FRONTIER (vLLM) -- writes greedy_pass.json (fast) + corpora
 say "SEED $s CURATE+FRONTIER (vLLM)"
 ( vllm_env; $PYVLLM -u run_ablation.py --stage curate --backend vllm --seed "$s") \
 || say "SEED $s curate rc=$? (continuing)"

 # 3. TRAIN 6 adapters (transformers; frontier+traces cached -> no slow gen)
 say "SEED $s TRAIN (transformers x6 arms)"
 $PYML -u run_ablation.py --stage train --backend transformers --seed "$s" \
 || say "SEED $s train rc=$? (continuing)"

 # 4. BASELINE (vLLM)
 say "SEED $s BASELINE (vLLM)"
 ( vllm_env; $PYVLLM -u run_ablation.py --stage baseline --backend vllm --seed "$s") \
 || say "SEED $s baseline rc=$? (continuing)"

 # 5. EVAL 6 adapters (vLLM LoRA serving; falls back to transformers if unsupported)
 say "SEED $s EVAL (vLLM LoRA)"
 ( vllm_env; $PYVLLM -u run_ablation.py --stage eval --backend vllm --seed "$s") \
 || say "SEED $s eval rc=$? (continuing)"

 # 6. aggregate after each seed so partial results are always available
 $PYML -u scripts_aggregate.py --seeds "${SEEDS[@]}" 2>/dev/null || true
 say "SEED $s DONE"
done

say "===== AGGREGATE ====="
$PYML -u scripts_aggregate.py --seeds "${SEEDS[@]}"
say "RESCUE END"
