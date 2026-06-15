r"""Orchestrator v2. Run the full bench or individual stages, so you can
checkpoint across the weekend and manage compute.

 python run_ablation.py --stage generate --backend transformers
 python run_ablation.py --stage baseline --backend transformers # THE GATE
 python run_ablation.py --stage curate --backend transformers
 python run_ablation.py --stage train --backend transformers
 python run_ablation.py --stage eval --backend transformers
 python run_ablation.py --stage all --backend transformers

 --backend dummy CPU logic check, no model (validate plumbing)
 --backend ollama Apple-Silicon field tuning (local models; can't train)
 --backend transformers HF .generate, BF16 (safe Spark default)
 --backend vllm fast path (only if it loads)
 (--dummy is kept as an alias for --backend dummy)

 train/eval are skipped automatically for dummy/ollama (they can't fine-tune).
"""
from __future__ import annotations
import argparse
import json
import random

import pandas as pd

from config import Config
from generate import (generate_population, build_prompt, DummyBackend,
 OllamaBackend, TransformersBackend, VLLMBackend)
from field import StigmergicField
from curate import build_all, population_success
from evaluate import evaluate_backend, evaluate_adapter
from metric import extract_answer, is_correct


def greedy_pass_rate(train: list[dict], backend, cfg: Config) -> dict[str, float]:
 """F1: a REAL per-problem model_pass_rate = does the BASE model solve
 this problem under GREEDY decoding? This is the frontier signal's second
 term (1 - model_pass_rate = 'still to learn'); the old proxy ('population
 always solves it') confounded B with population difficulty. Cached.

 Returns {problem_id: 1.0 if greedy answer correct else 0.0}. Neutral prompt
 (no persona) so it reflects the base model's own default behavior."""
 cache = cfg.work_dir / "greedy_pass.json"
 if cache.exists:
 d = json.loads(cache.read_text)
 print(f"[frontier] loaded {len(d)} cached greedy pass-rates")
 return d
 prompts = [build_prompt(p["problem"], "") for p in train]
 # n=1, temperature=0 => greedy single sample per problem
 outs = backend.generate(prompts, n=1, temperature=0.0,
 max_new_tokens=cfg.gen_max_new_tokens)
 rate: dict[str, float] = {}
 for p, comps in zip(train, outs):
 text = comps[0] if comps else ""
 rate[str(p["id"])] = 1.0 if is_correct(extract_answer(text), str(p["answer"])) else 0.0
 solved = sum(rate.values)
 print(f"[frontier] greedy base-model solves {int(solved)}/{len(rate)} "
 f"train problems ({solved/max(len(rate),1):.1%})")
 cache.write_text(json.dumps(rate))
 return rate


def load_split(cfg: Config):
 df = pd.read_csv(cfg.data_dir / cfg.train_csv)
 df = df.rename(columns={cfg.problem_col: "problem",
 cfg.answer_col: "answer", cfg.id_col: "id"})
 rows = df[["id", "problem", "answer"]].to_dict("records")
 rng = random.Random(cfg.seed)
 rng.shuffle(rows)
 val = rows[:cfg.val_size]
 train = rows[cfg.val_size:cfg.val_size + cfg.subset_size]
 return train, val


def resolve_backend_name(cfg: Config, args) -> str:
 if args.dummy:
 return "dummy"
 return args.backend or cfg.backend


def get_backend(cfg: Config, train, name: str):
 if name == "dummy":
 gold = {r["problem"]: str(r["answer"]) for r in train}
 return DummyBackend(gold_lookup=gold, hit_rate=0.5)
 if name == "ollama":
 return OllamaBackend
 if name == "transformers":
 return TransformersBackend(cfg.model_id, batch_size=cfg.gen_batch_size)
 if name == "vllm":
 return VLLMBackend(cfg.model_id)
 raise ValueError(f"unknown backend: {name}")


def main:
 ap = argparse.ArgumentParser
 ap.add_argument("--stage", default="all",
 choices=["generate", "baseline", "curate", "train", "eval", "all"])
 ap.add_argument("--backend", default=None,
 choices=["dummy", "ollama", "transformers", "vllm"],
 help="overrides config.backend")
 ap.add_argument("--dummy", action="store_true", help="alias for --backend dummy")
 ap.add_argument("--seed", type=int, default=None,
 help="override config.seed; isolates work/adapter dirs per seed "
 "so multiple seeds don't clobber each other (ga-06: report "
 "the run distribution, not best-of-N)")
 args = ap.parse_args

 cfg = Config
 if args.seed is not None:
 cfg.seed = args.seed
 # isolate ALL per-run artifacts (trace cache, corpora, adapters,
 # baseline, results) under a seed-specific subdir.
 cfg.work_dir = cfg.work_dir / f"seed{cfg.seed}"
 cfg.adapter_dir = cfg.adapter_dir / f"seed{cfg.seed}"
 cfg.ensure_dirs
 name = resolve_backend_name(cfg, args)
 can_train = name not in ("dummy", "ollama")
 train, val = load_split(cfg)
 print(f"[data] train-subset={len(train)} val={len(val)} backend={name} "
 f"(val is FIXED, never trained)")

 # Backends that load a 62GB model are expensive to construct, so build lazily
 # and reuse one instance across whatever stages need generation this run.
 _be = {"obj": None}
 def backend:
 if _be["obj"] is None:
 _be["obj"] = get_backend(cfg, train, name)
 return _be["obj"]

 if args.stage in ("generate", "all"):
 generate_population(train, backend, cfg, tag="subset", use_cache=True)

 if args.stage in ("baseline", "all"):
 base_acc = evaluate_backend(val, backend, cfg)
 (cfg.work_dir / "baseline.json").write_text(json.dumps({"val_base_acc": base_acc}))
 print(f"[baseline] base model val accuracy = {base_acc:.3f} "
 f"<-- must look sane before trusting anything downstream")

 if args.stage in ("curate", "train", "eval", "all"):
 traces = generate_population(train, backend, cfg, tag="subset", use_cache=True)

 fld = StigmergicField(cfg)
 fld.deposit(traces)
 fld.settle(passes=3)
 print(fld.summary)

 # Frontier signal's second term (1 - model_pass_rate).
 # F1: use a REAL greedy base-model eval if enabled (kills the p10
 # confound); else fall back to the old population-solves proxy.
 if getattr(cfg, "real_frontier_signal", False):
 model_pass_rate = greedy_pass_rate(train, backend, cfg)
 else:
 pop = population_success(traces)
 model_pass_rate = {pid: (1.0 if v == 1.0 else 0.0) for pid, v in pop.items}

 corpora = build_all(traces, fld, model_pass_rate, cfg)
 for nm, c in corpora.items:
 (cfg.work_dir / f"corpus_{nm}.json").write_text(json.dumps(c))

 if args.stage in ("train", "all") and can_train:
 from train import train_lora
 for nm, c in corpora.items:
 train_lora(c, cfg, run_name=nm)
 elif args.stage in ("train", "all"):
 print(f"[train] skipped: backend '{name}' cannot fine-tune. "
 f"Use --backend transformers (or vllm) on the Spark.")

 if args.stage in ("eval", "all") and can_train:
 base = None
 bj = cfg.work_dir / "baseline.json"
 if bj.exists:
 base = json.loads(bj.read_text)["val_base_acc"]
 from evaluate import evaluate_adapters_vllm
 results = evaluate_adapters_vllm(val, cfg.adapter_dir, list(corpora), cfg)
 print(f"\n==== RESULT: val accuracy at equal budget "
 f"({cfg.budget_examples} ex) ====")
 if base is not None:
 print(f" base (no SFT) : {base:.3f}")
 for nm, acc in results.items:
 print(f" {nm:24s} : {acc:.3f} (corpus n={len(corpora[nm])})")
 print("\nHypotheses (equal budget): mechanism arms (B/C/D) > "
 "consensus (A) > baselines (B1~=B0).")
 print(" C_gamma_consensus tests (discount boilerplate consensus)")
 print(" D_diversity tests/ (keep heterogeneity vs argmax-1)")
 print(" NOTE: trust the 2-seed mean+/-std (results_aggregate.json), "
 "not any single run (ga-06).")
 (cfg.work_dir / "results.json").write_text(json.dumps(
 {"results": results,
 "corpus_sizes": {nm: len(c) for nm, c in corpora.items},
 "base": base, "seed": cfg.seed,
 "budget": cfg.budget_examples}))
 elif args.stage in ("eval", "all"):
 print(f"[eval] skipped: backend '{name}' cannot train/eval adapters.")


if __name__ == "__main__":
 main
