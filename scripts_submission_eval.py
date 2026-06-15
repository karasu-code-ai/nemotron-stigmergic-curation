#!/usr/bin/env python3
r"""Local mirror of the OFFICIAL Kaggle submission eval -- predict the leaderboard
score for an adapter BEFORE spending a submission.

Matches the published harness parameters EXACTLY:
 engine vLLM, base = Nemotron-3-Nano-30B + LoRA
 sampling temperature=0.0 (greedy), top_p=1.0, max_tokens=7680
 model max_model_len=8192, max_lora_rank=32
 prompt official chat template + enable_thinking=True + the official
 boxed suffix (see metric.py)
 scoring metric.extract_final_answer + metric.verify (rel_tol=1e-2)

We score on the FIXED val split (it has gold answers; test.csv does not). Absolute
numbers will track -- but not perfectly equal -- the leaderboard (different problems),
so use this to RANK arms and sanity-check before uploading, not as the literal LB.

!! GPU NOTE: loads the 62GB model in vLLM. Do NOT run while generation/training is
 using the GPU. One vLLM at a time on the GB10.

Examples:
 # predict score for the best existing adapter
 python scripts_submission_eval.py --adapter adapters/B1_shortest
 # base model (no adapter) reference
 python scripts_submission_eval.py --base-only
 # rank several Part-9 arms in one model load
 python scripts_submission_eval.py --seed 0 --arms B0_random B_consensus_frontier C_gamma_consensus D_diversity
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from config import Config
from metric import extract_final_answer, verify

# The official suffix + thinking mode (verbatim intent from metric.py docstring).
OFFICIAL_SUFFIX = ("Please put your final answer inside `\\boxed{}`. "
 "For example: `\\boxed{your answer}`")
HARNESS_MAX_TOKENS = 7680
HARNESS_MAX_MODEL_LEN = 8192
HARNESS_MAX_LORA_RANK = 32


def load_val(cfg: Config) -> list[dict]:
 """Reproduce run_ablation.load_split's val slice (same seed/shuffle -> same
 held-out problems the arms were scored on)."""
 import random
 df = pd.read_csv(cfg.data_dir / cfg.train_csv).rename(
 columns={cfg.problem_col: "problem", cfg.answer_col: "answer", cfg.id_col: "id"})
 rows = df[["id", "problem", "answer"]].to_dict("records")
 random.Random(cfg.seed).shuffle(rows)
 return rows[:cfg.val_size]


def build_prompts(tok, problems: list[dict]) -> list[str]:
 out = []
 for p in problems:
 msg = [{"role": "user", "content": f"{p['problem']}\n\n{OFFICIAL_SUFFIX}"}]
 try:
 txt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
 enable_thinking=True)
 except TypeError:
 txt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
 out.append(txt)
 return out


def score(texts: list[str], problems: list[dict]) -> tuple[float, list[int]]:
 hits = [int(verify(str(p["answer"]), extract_final_answer(t)))
 for p, t in zip(problems, texts)]
 return sum(hits) / max(len(hits), 1), hits


def main -> None:
 ap = argparse.ArgumentParser(description="Predict Kaggle score for adapter(s) locally")
 ap.add_argument("--adapter", type=Path, help="single adapter dir")
 ap.add_argument("--seed", type=int, default=None, help="adapters/seed{N}/ for --arms")
 ap.add_argument("--arms", nargs="+", help="arm names to rank (under adapters[/seedN])")
 ap.add_argument("--base-only", action="store_true", help="eval base model, no adapter")
 ap.add_argument("--max-tokens", type=int, default=HARNESS_MAX_TOKENS)
 ap.add_argument("--val-size", type=int, default=None, help="override cfg.val_size")
 ap.add_argument("--limit", type=int, default=None, help="only first N val problems (quick)")
 ap.add_argument("--out", type=Path, default=Path("work/submission_eval.json"))
 args = ap.parse_args

 cfg = Config
 if args.val_size is not None:
 cfg.val_size = args.val_size
 problems = load_val(cfg)
 if args.limit:
 problems = problems[:args.limit]

 # resolve adapters to eval
 targets: list[tuple[str, Path | None]] = []
 if args.base_only:
 targets.append(("base", None))
 if args.adapter:
 targets.append((args.adapter.name, args.adapter))
 if args.arms:
 root = Path("adapters") / (f"seed{args.seed}" if args.seed is not None else "")
 for a in args.arms:
 targets.append((a, root / a))
 if not targets:
 raise SystemExit("specify --adapter, --arms, or --base-only")

 from vllm import LLM, SamplingParams
 from vllm.lora.request import LoRARequest
 from transformers import AutoTokenizer

 tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
 prompts = build_prompts(tok, problems)
 sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

 need_lora = any(p is not None for _, p in targets)
 llm = LLM(model=cfg.model_id, trust_remote_code=True,
 enable_lora=need_lora,
 max_lora_rank=HARNESS_MAX_LORA_RANK,
 max_model_len=HARNESS_MAX_MODEL_LEN,
 gpu_memory_utilization=0.85)

 print(f"\n[subm-eval] {len(problems)} val problems | greedy | max_tokens={args.max_tokens} "
 f"| official prompt+thinking\n")
 results = {}
 for i, (name, path) in enumerate(targets):
 if path is None:
 outs = llm.generate(prompts, sp)
 else:
 req = LoRARequest(name, i + 1, str(path))
 outs = llm.generate(prompts, sp, lora_request=req)
 texts = [o.outputs[0].text for o in outs]
 acc, hits = score(texts, problems)
 # how often the model actually emitted a boxed answer (extraction hygiene)
 boxed = sum(int("\\boxed{" in t) for t in texts) / max(len(texts), 1)
 results[name] = {"acc": acc, "n": len(problems), "boxed_rate": boxed}
 print(f" {name:24s} acc={acc:.3f} boxed_rate={boxed:.2f} (n={len(problems)})")

 args.out.parent.mkdir(parents=True, exist_ok=True)
 args.out.write_text(json.dumps(results, indent=2))
 print(f"\n[subm-eval] wrote {args.out}")
 print("[subm-eval] pick the top arm, package it with scripts_package_submission.py, upload.")


if __name__ == "__main__":
 main
