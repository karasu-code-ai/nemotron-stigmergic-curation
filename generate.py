r"""Population generator ('polyagent' layer) + pluggable model backends. v2.

v2 adds TransformersBackend (HF .generate, the no-vLLM fallback that works on
ARM+Blackwell) and OllamaBackend (Apple-Silicon field-tuning while the Spark is
offline). DummyBackend (CPU logic check) and VLLMBackend (fast path) unchanged.

Each problem is attacked by a population = personas x temperatures x k_samples,
all from ONE base model. Output is a flat list of Trace records tagged correct/
incorrect against the gold answer. Generations are cached so the expensive step
runs ONCE. Swap backends WITHOUT touching field/curate/ablation logic.
"""
from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol, Sequence

from metric import extract_answer, is_correct
from config import Config


@dataclass
class Trace:
 problem_id: str
 problem: str
 gold: str
 persona: str
 temperature: float
 text: str
 answer: str | None
 correct: bool


class GenerationBackend(Protocol):
 def generate(self, prompts: Sequence[str], n: int, temperature: float,
 max_new_tokens: int) -> list[list[str]]:
 """Return, for each prompt, a list of n completions."""
 ...


# ---------------------------------------------------------------------------
class DummyBackend:
 """No GPU, no model. Deterministic pseudo-reasoning with SHARED 'consensus'
 fragments + idiosyncratic noise; sometimes 'correct' so corpora are
 non-empty. Validates the ENTIRE field/curate/ablation pipeline CPU-side."""
 def __init__(self, gold_lookup: dict[str, str], hit_rate: float = 0.5):
 self.gold_lookup = gold_lookup
 self.hit_rate = hit_rate

 def generate(self, prompts, n, temperature, max_new_tokens):
 out = []
 for p in prompts:
 gold = next((v for k, v in self.gold_lookup.items if k in p), None)
 comps = []
 for i in range(n):
 seed = int(hashlib.md5(f"{p}{temperature}{i}".encode).hexdigest, 16)
 hit = ((seed % 100) / 100.0 < self.hit_rate) and gold is not None
 lines = ["First, identify the relevant quantities.",
 "Set up the governing relation."]
 if seed % 3 == 0:
 lines.append("Consider the boundary case separately.")
 if seed % 2 == 0:
 lines.append("Simplify the expression carefully.")
 lines.append(f"Idiosyncratic step number {seed % 7}.")
 ans = gold if hit else str(seed % 999)
 lines.append(f"Therefore the answer is \\boxed{{{ans}}}.")
 comps.append("\n".join(lines))
 out.append(comps)
 return out


class OllamaBackend:
 """Apple-Silicon / no-CUDA generation via local ollama (OpenAI-compatible
 at :11434). NOT competition-valid -- gemma != Nemotron. Use ONLY to
 tune field/curation on realistic-ish traces while the Spark is offline.
 ollama can't train, so --stage train still needs the Spark."""
 def __init__(self, model="gemma3:1b", host="http://localhost:11434/v1"):
 import requests
 self._r, self.model, self.host = requests, model, host

 def generate(self, prompts, n, temperature, max_new_tokens):
 out = []
 for p in prompts:
 comps = []
 for _ in range(n): # ollama shim ignores n>1; loop
 r = self._r.post(f"{self.host}/chat/completions", json={
 "model": self.model, "temperature": temperature,
 "max_tokens": max_new_tokens,
 "messages": [{"role": "user", "content": p}]}, timeout=180)
 comps.append(r.json["choices"][0]["message"]["content"])
 out.append(comps)
 return out


class TransformersBackend:
 """No-vLLM generation via HF transformers in BF16. The reliable path on the
 Spark (ARM+Blackwell) when vLLM lacks Nemotron-3 support. Slower than vLLM
 (no continuous batching) -- keep the subset small. Loads the model ONCE.
 Raw-prompt completion (matches VLLMBackend) so the ablation stays consistent;
 applying the chat template is a later quality lever, not needed for B-vs-B0."""
 def __init__(self, model_id: str, batch_size: int = 8, device_map: str = "cuda:0"):
 import torch
 import nemotron_compat
 nemotron_compat.apply # inject pure-torch rmsnorm_fn (no-op if kernels built)
 from transformers import AutoModelForCausalLM, AutoTokenizer
 self.torch = torch
 self.batch_size = batch_size
 self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
 if self.tok.pad_token is None:
 self.tok.pad_token = self.tok.eos_token
 self.tok.padding_side = "left" # required for batched decode
 # device_map="cuda:0" (single device), NOT "auto". On the GB10 Spark
 # nvidia-smi reports unified memory as N/A, so accelerate's "auto" wrongly
 # decides it must offload layers to CPU -> meta-device params + OOM (SIGKILL
 # 137) during generation. The ~62GB BF16 model fits the GPU directly with
 # ~56GB headroom on this 122GB unified box, so pin it to the GPU.
 self.model = AutoModelForCausalLM.from_pretrained(
 model_id, trust_remote_code=True,
 torch_dtype=torch.bfloat16, device_map=device_map).eval

 def chat_prompt(self, problem: str) -> str:
 r"""Render a problem EXACTLY as the official scorer does: official boxed
 suffix + the tokenizer chat template with enable_thinking=True. Used by
 the on-ruler baseline/eval path. Falls back to build_prompt if the
 template doesn't accept enable_thinking."""
 content = official_user_content(problem)
 msg = [{"role": "user", "content": content}]
 try:
 return self.tok.apply_chat_template(
 msg, tokenize=False, add_generation_prompt=True,
 enable_thinking=True)
 except TypeError:
 # older tokenizers: no enable_thinking kwarg
 return self.tok.apply_chat_template(
 msg, tokenize=False, add_generation_prompt=True)
 except Exception:
 return build_prompt(problem, "")

 def generate(self, prompts, n, temperature, max_new_tokens):
 import time
 torch = self.torch
 do_sample = temperature > 0.0
 n_eff = n if do_sample else 1 # greedy can't yield distinct samples
 out: list[list[str]] = [[] for _ in prompts]
 nb = (len(prompts) + self.batch_size - 1) // self.batch_size
 for bi, start in enumerate(range(0, len(prompts), self.batch_size), 1):
 batch = list(prompts[start:start + self.batch_size])
 enc = self.tok(batch, return_tensors="pt", padding=True,
 truncation=True, max_length=4096).to(self.model.device)
 gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
 num_return_sequences=n_eff,
 pad_token_id=self.tok.pad_token_id)
 if do_sample:
 gen_kwargs["temperature"] = temperature
 t0 = time.time
 with torch.no_grad:
 gen = self.model.generate(**enc, **gen_kwargs)
 new = gen[:, enc.input_ids.shape[1]:] # strip the (left-padded) prompt
 dt = time.time - t0
 ntok = new.shape[0] * new.shape[1]
 # per-batch progress so long runs aren't a black box (RUNBOOK Part 5)
 print(f"[gen] batch {bi}/{nb} ({len(batch)} prompts x{n_eff}) "
 f"{new.shape[1]} new tok in {dt:.0f}s "
 f"= {ntok / dt:.1f} tok/s", flush=True)
 texts = self.tok.batch_decode(new, skip_special_tokens=True)
 for i in range(len(batch)):
 comps = texts[i * n_eff:(i + 1) * n_eff]
 if not do_sample and n > 1:
 comps = comps * n # replicate greedy if n requested
 out[start + i] = comps
 return out


class VLLMBackend:
 """Fast path. Confirm Nemotron-3 (hybrid Mamba-Transformer MoE) is supported
 by your installed vLLM; if it isn't, use TransformersBackend instead."""
 def __init__(self, model_id: str, dtype: str = "auto",
 tensor_parallel_size: int = 1, **kw):
 from vllm import LLM
 self.llm = LLM(model=model_id, dtype=dtype, trust_remote_code=True,
 tensor_parallel_size=tensor_parallel_size, **kw)

 def generate(self, prompts, n, temperature, max_new_tokens):
 from vllm import SamplingParams
 sp = SamplingParams(n=n, temperature=temperature, max_tokens=max_new_tokens)
 results = self.llm.generate(list(prompts), sp)
 return [[o.text for o in r.outputs] for r in results]


# ---------------------------------------------------------------------------
def build_prompt(problem: str, system_prompt: str) -> str:
 r"""Plain-text prompt embedding the persona as system text. Raw completion
 (no chat template) keeps all backends consistent for the ablation; switching
 to the tokenizer chat template later is a quality improvement."""
 return (f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
 f"Problem:\n{problem}\n\nSolve step by step. "
 f"Give the final answer as \\boxed{{...}}.\n\nSolution:\n")


# The OFFICIAL prompt suffix, copied verbatim from the competition metric kernel
# (metric/nvidia-nemotron-metric, generate_predictions). The official scorer
# appends exactly this to each problem, then renders it through the tokenizer's
# chat template with enable_thinking=True. Matching it keeps our baseline/eval
# numbers on the same ruler as Kaggle (RUNBOOK Part 0).
OFFICIAL_BOXED_SUFFIX = (
 "\nPlease put your final answer inside `\\boxed{}`. "
 "For example: `\\boxed{your answer}`")


def official_user_content(problem: str) -> str:
 return problem + OFFICIAL_BOXED_SUFFIX


def _cache_path(cfg: Config, tag: str) -> Path:
 return cfg.work_dir / f"traces_{tag}.jsonl"


def generate_population(problems: list[dict], backend: GenerationBackend,
 cfg: Config, tag: str = "subset",
 use_cache: bool = True) -> list[Trace]:
 """problems: list of {id, problem, answer}. Cached by `tag` so the GPU-heavy
 generation runs once; later stages reload it for free."""
 cfg.ensure_dirs
 cache = _cache_path(cfg, tag)
 if use_cache and cache.exists:
 traces = [Trace(**json.loads(l)) for l in cache.read_text.splitlines]
 print(f"[generate] loaded {len(traces)} cached traces from {cache}")
 return traces

 traces: list[Trace] = []
 for persona_name, sys_prompt in cfg.personas.items:
 prompts = [build_prompt(p["problem"], sys_prompt) for p in problems]
 for temp in cfg.temperatures:
 print(f"[generate] persona={persona_name} temp={temp} "
 f"x{cfg.k_samples} over {len(problems)} problems")
 batch = backend.generate(prompts, n=cfg.k_samples, temperature=temp,
 max_new_tokens=cfg.gen_max_new_tokens)
 for prob, comps in zip(problems, batch):
 for text in comps:
 ans = extract_answer(text)
 traces.append(Trace(
 problem_id=str(prob["id"]), problem=prob["problem"],
 gold=str(prob["answer"]), persona=persona_name,
 temperature=temp, text=text, answer=ans,
 correct=is_correct(ans, str(prob["answer"])),))
 with cache.open("w") as f:
 for t in traces:
 f.write(json.dumps(asdict(t)) + "\n")
 n_ok = sum(t.correct for t in traces)
 print(f"[generate] {len(traces)} traces, {n_ok} correct "
 f"({n_ok / max(len(traces), 1):.1%}). cached -> {cache}")
 return traces
