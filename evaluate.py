r"""Evaluation against a FIXED held-out split. Greedy single-sample generation,
boxed extraction, competition metric. Used twice:
 (1) Saturday-AM baseline reproduction GATE (base model, no SFT)
 (2) final four-way comparison (each trained adapter)
"""
from __future__ import annotations
from pathlib import Path

from metric import extract_answer, is_correct
from generate import build_prompt
from config import Config

_EVAL_SYS = "Solve the problem. Give the final answer as \\boxed{...}."


def evaluate_backend(problems: list[dict], backend, cfg: Config) -> float:
 """Generic eval over any GenerationBackend (n=1, temp=0). Use this with the
 base-model backend for the Saturday gate.

 On-ruler: if the backend exposes chat_prompt (TransformersBackend), use the
 OFFICIAL prompt format (chat template + enable_thinking + boxed suffix) so the
 baseline tracks Kaggle. Other backends (Dummy) fall back to build_prompt."""
 if hasattr(backend, "chat_prompt"):
 prompts = [backend.chat_prompt(p["problem"]) for p in problems]
 else:
 prompts = [build_prompt(p["problem"], _EVAL_SYS) for p in problems]
 batch = backend.generate(prompts, n=1, temperature=0.0,
 max_new_tokens=cfg.gen_max_new_tokens)
 correct = sum(
 int(is_correct(extract_answer(comps[0]) if comps else None, str(p["answer"])))
 for p, comps in zip(problems, batch))
 return correct / max(len(problems), 1)


def evaluate_adapters_vllm(problems: list[dict], adapter_dir: Path,
 names: list[str], cfg: Config) -> dict:
 """Eval MANY LoRA adapters from ONE vLLM engine via LoRARequest.

 Why one engine: on the GB10's unified 121GB pool, building a fresh LLM per
 adapter (the old call-site) OOMs -- the prior engine's ~110GB isn't released
 before the next inits. Load the 62GB base ONCE with enable_lora, then swap the
 tiny adapters in. Memory kwargs (0.85 util, max_model_len 8192) match the
 proven submission-eval / harness config so vLLM's reservation + the model
 weights both fit the unified pool. Greedy, boxed, official metric."""
 from vllm import LLM, SamplingParams
 from vllm.lora.request import LoRARequest
 llm = LLM(model=cfg.model_id, enable_lora=True, max_lora_rank=cfg.lora_r,
 max_model_len=8192, gpu_memory_utilization=0.85,
 trust_remote_code=True)
 sp = SamplingParams(n=1, temperature=0.0, max_tokens=cfg.gen_max_new_tokens)
 prompts = [build_prompt(p["problem"], _EVAL_SYS) for p in problems]
 out: dict = {}
 for i, nm in enumerate(names):
 req = LoRARequest(nm, i + 1, str(adapter_dir / nm))
 res = llm.generate(prompts, sp, lora_request=req)
 correct = sum(int(is_correct(extract_answer(r.outputs[0].text), str(p["answer"])))
 for p, r in zip(problems, res))
 out[nm] = correct / max(len(problems), 1)
 print(f"[eval] {nm:24s}: {out[nm]:.3f}", flush=True)
 return out


def evaluate_adapter(problems: list[dict], adapter_path: Path, cfg: Config) -> float:
 """Eval a trained LoRA adapter. Tries vLLM LoRA serving; falls back to
 transformers+peft. Confirm LoRA-serving support for Nemotron in your vLLM."""
 try:
 from vllm import LLM, SamplingParams
 from vllm.lora.request import LoRARequest
 llm = LLM(model=cfg.model_id, enable_lora=True, max_lora_rank=cfg.lora_r,
 trust_remote_code=True)
 sp = SamplingParams(n=1, temperature=0.0, max_tokens=cfg.gen_max_new_tokens)
 prompts = [build_prompt(p["problem"], _EVAL_SYS) for p in problems]
 req = LoRARequest("adapter", 1, str(adapter_path))
 results = llm.generate(prompts, sp, lora_request=req)
 correct = sum(int(is_correct(extract_answer(r.outputs[0].text), str(p["answer"])))
 for p, r in zip(problems, results))
 return correct / max(len(problems), 1)
 except Exception as e:
 print(f"[eval] vLLM-LoRA path unavailable ({e}); using transformers+peft.")
 return _evaluate_adapter_hf(problems, adapter_path, cfg)


def _evaluate_adapter_hf(problems: list[dict], adapter_path: Path, cfg: Config) -> float:
 import torch
 import nemotron_compat
 nemotron_compat.apply # inject pure-torch rmsnorm_fn; no mamba-ssm build
 from transformers import AutoModelForCausalLM, AutoTokenizer
 from peft import PeftModel
 tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
 base = AutoModelForCausalLM.from_pretrained(
 cfg.model_id, trust_remote_code=True,
 torch_dtype=torch.bfloat16, device_map="cuda:0") # NOT "auto": auto OOMs
 # via CPU-offload on the GB10 (unified mem reads N/A to accelerate)
 model = PeftModel.from_pretrained(base, str(adapter_path)).eval
 correct = 0
 for p in problems:
 msg = [{"role": "user", "content": p["problem"] +
 "\n\nSolve step by step. Give the final answer as \\boxed{...}."}]
 try:
 prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
 except Exception:
 prompt = build_prompt(p["problem"], _EVAL_SYS)
 ids = tok(prompt, return_tensors="pt").to(model.device)
 with torch.no_grad:
 out = model.generate(**ids, max_new_tokens=cfg.gen_max_new_tokens, do_sample=False)
 text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
 correct += int(is_correct(extract_answer(text), str(p["answer"])))
 return correct / max(len(problems), 1)
