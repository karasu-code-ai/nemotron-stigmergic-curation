#!/usr/bin/env python3
r"""Overnight-safe GPU job: pre-compute the REAL frontier signal (greedy BASE-model
pass-rate/F1) over the whole multi-source problem pool, and seed BOTH per-seed
caches. (1 - greedy_pass = 'still to learn' = the frontier weight's second term.)
Source-independent, so it CANNOT be wrong by a curation bug.

Uses the PROVEN vLLM-direct path (mirrors scripts_bridge_eval.py, which ran cleanly
rather than the untested VLLMBackend wrapper -- reliability matters for an
unattended run. No LoRA (base model), greedy, chat-template + boxed suffix to match
how we eval.

 PATH=/opt/ml/venv-vllm/bin:$PATH python scripts_frontier_precompute.py
"""
import glob, json
from pathlib import Path
from config import Config
from generate import Trace
from metric import extract_answer, is_correct

SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
 "For example: `\\boxed{your answer}`")

def main:
 cfg = Config
 work = cfg.work_dir
 base_cache = work / "greedy_pass.json"

 # union problem set from every source's traces (problem_id -> problem/gold)
 files = [work / "traces_subset.jsonl"] + \
 [Path(f) for f in glob.glob(str(work / "field_sources" / "clean" / "*.jsonl"))]
 uni = {}
 for f in files:
 if not f.exists:
 continue
 for l in open(f):
 if not l.strip:
 continue
 t = Trace(**json.loads(l))
 uni.setdefault(t.problem_id, {"id": t.problem_id, "problem": t.problem, "gold": t.gold})
 problems = list(uni.values)
 print(f"[frontier-pre] {len(problems)} union problems from {len(files)} files", flush=True)

 if not base_cache.exists:
 from vllm import LLM, SamplingParams
 llm = LLM(model=cfg.model_id, max_model_len=8192, gpu_memory_utilization=0.80,
 max_num_seqs=16, enforce_eager=True, trust_remote_code=True) # no LoRA = base model
 sp = SamplingParams(n=1, temperature=0.0, max_tokens=4096)
 convs = [[{"role": "user", "content": p["problem"] + SUFFIX}] for p in problems]
 try:
 res = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": True})
 except TypeError:
 res = llm.chat(convs, sp)
 rate = {}
 for p, r in zip(problems, res):
 ok = is_correct(extract_answer(r.outputs[0].text), str(p["gold"]))
 rate[str(p["id"])] = 1.0 if ok else 0.0
 base_cache.write_text(json.dumps(rate))
 print(f"[frontier-pre] base greedy solved {int(sum(rate.values))}/{len(rate)}", flush=True)
 else:
 rate = json.loads(base_cache.read_text)
 print(f"[frontier-pre] reused existing cache ({len(rate)} entries)", flush=True)

 # fan out to each per-seed work dir so the seeded ablation loads it for free
 for s in (0, 1):
 d = work / f"seed{s}"; d.mkdir(parents=True, exist_ok=True)
 (d / "greedy_pass.json").write_text(json.dumps(rate))
 print(f"[frontier-pre] seeded {d}/greedy_pass.json", flush=True)
 solved = sum(rate.values)
 print(f"[frontier-pre] DONE: base solves {int(solved)}/{len(rate)} "
 f"({solved/max(len(rate),1):.1%}); frontier weight = 1 - this", flush=True)

if __name__ == "__main__":
 main
