r"""Standalone eval of ALL trained adapters from ONE vLLM engine.

Why standalone: run_ablation --stage eval runs curate (loads 2880 traces + builds
the 20k-key field) in the SAME process, then vLLM tries to reserve its share of
the unified 121GB pool -> OOM. And the per-adapter LLM reload OOM'd too. Here:
NO curate (arm names come from the corpus_*.json filenames already on disk), ONE
engine for all 12 adapters (6 arms x 2 seeds) via LoRARequest, with unified-memory-
safe knobs: small max_num_seqs (caps vLLM's init profiling dummy batch -- the real
OOM source), enforce_eager (no CUDA-graph memory), conservative util.

Run from venv-vllm: PATH=...venv-vllm/bin python -u scripts_eval_seed.py
Writes work/seed{0,1}/results.json. Greedy, boxed, official metric. FIXED val split.
"""
import json, glob, os
from pathlib import Path
from config import Config
from metric import extract_answer, is_correct
from generate import build_prompt
from run_ablation import load_split

_SYS = "Solve the problem. Give the final answer as \\boxed{...}."

def main:
 cfg = Config
 val = load_split(cfg)[1] # val = rows[:val_size], seed-independent (FIXED)
 prompts = [build_prompt(p["problem"], _SYS) for p in val]
 print(f"[eval] val={len(val)} (fixed split)", flush=True)

 from vllm import LLM, SamplingParams
 from vllm.lora.request import LoRARequest
 llm = LLM(model=cfg.model_id, enable_lora=True, max_lora_rank=cfg.lora_r,
 max_model_len=8192, gpu_memory_utilization=0.80, max_num_seqs=16,
 enforce_eager=True, trust_remote_code=True)
 sp = SamplingParams(n=1, temperature=0.0, max_tokens=cfg.gen_max_new_tokens)

 lid = 0
 for seed in (0, 1):
 adir = Path("adapters") / f"seed{seed}"
 arms = sorted(os.path.basename(p)[len("corpus_"):-len(".json")]
 for p in glob.glob(f"work/seed{seed}/corpus_*.json"))
 if not arms or not adir.exists:
 print(f"[eval] seed{seed}: no corpora/adapters, skip"); continue
 results = {}
 for nm in arms:
 ap = adir / nm
 if not (ap / "adapter_model.safetensors").exists:
 print(f"[eval] seed{seed} {nm}: adapter missing, skip"); continue
 lid += 1
 res = llm.generate(prompts, sp, lora_request=LoRARequest(nm, lid, str(ap)))
 correct = sum(int(is_correct(extract_answer(r.outputs[0].text), str(p["answer"])))
 for p, r in zip(val, res))
 results[nm] = round(correct / max(len(val), 1), 4)
 print(f"[eval] seed{seed} {nm:24s}: {results[nm]:.3f}", flush=True)
 base = None
 bj = Path(f"work/seed{seed}/baseline.json")
 if bj.exists:
 base = json.loads(bj.read_text).get("val_base_acc")
 sizes = {nm: len(json.load(open(f"work/seed{seed}/corpus_{nm}.json"))) for nm in arms}
 Path(f"work/seed{seed}/results.json").write_text(json.dumps(
 {"results": results, "corpus_sizes": sizes, "base": base,
 "seed": seed, "budget": cfg.budget_examples}, indent=2))
 print(f"[eval] seed{seed} -> results.json (base={base})", flush=True)
 print("[eval] DONE", flush=True)

if __name__ == "__main__":
 main
