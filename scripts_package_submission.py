#!/usr/bin/env python3
r"""Package a trained LoRA adapter into a Kaggle-ready `submission.zip`.

The competition wants the ADAPTER ITSELF (rank <= 32, with adapter_config.json),
zipped; Kaggle loads Nemotron-3-Nano-30B + your adapter with vLLM on their side
and runs greedy eval. (This supersedes RUNBOOK Part 10's old "write submission.csv"
model -- that was a wrong guess; see COMPETITION.md.)

Zip layout (matches the known-good SebAustin reference): adapter files at the ZIP
ROOT (adapter_config.json + adapter_model.safetensors[+ README/chat_template]),
tokenizer files skipped by default.

Resolve which adapter three ways:
 --adapter-dir PATH explicit
 --seed N --arm NAME -> adapters/seed{N}/{NAME}
 --best [--seed N] auto-pick the highest-val-acc arm from
 results_aggregate.json / seed{N}/results.json
 / results.json (prior run)

IMPORTANT (base path): our adapter_config.json records base_model_name_or_path as a
LOCAL Spark path, which Kaggle can't load. Most PEFT harnesses override the base and
just attach the adapter, but to be safe pass --base-model-id <hf_id> to rewrite it in
the zipped copy (the on-disk adapter is left untouched). Verify the exact id from the
official submission demo before trusting it.

Examples:
 python scripts_package_submission.py --best # auto, prior run
 python scripts_package_submission.py --best --seed 0 # auto, seed0 (Part-9)
 python scripts_package_submission.py --seed 0 --arm C_gamma_consensus
 python scripts_package_submission.py --adapter-dir adapters/B1_shortest \
 --base-model-id nvidia/Nemotron-3-Nano-30B-A3B --output submission.zip
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Tokenizer files are redundant with the base model the harness already loads.
SKIP_NAMES = {"tokenizer.json", "tokenizer_config.json",
 "special_tokens_map.json", "tokenizer.model"}
# Strictly-required PEFT files (used by --minimal).
REQUIRED_ONLY = {"adapter_config.json", "adapter_model.safetensors",
 "adapter_model.bin"}

ARM_ORDER = ["B0_random", "B1_shortest", "A_consensus", "B_consensus_frontier",
 "C_gamma_consensus", "D_diversity"]


def _read_json(p: Path):
 try:
 return json.loads(p.read_text)
 except Exception:
 return None


def _best_from_results(work: Path, seed: int | None) -> tuple[str | None, dict]:
 """Return (best_arm_name, {arm: acc}) from the most authoritative results file
 available, in priority order: aggregate (2-seed mean) -> seed{N} -> prior flat."""
 # 1) 2-seed aggregate
 agg = _read_json(work / "results_aggregate.json")
 if agg and agg.get("arms"):
 accs = {a: d["mean"] for a, d in agg["arms"].items}
 if accs:
 return max(accs, key=accs.get), accs
 # 2) per-seed wrapped results
 for s in ([seed] if seed is not None else [0, 1]):
 r = _read_json(work / f"seed{s}" / "results.json")
 if r:
 accs = r.get("results", r)
 accs = {k: v for k, v in accs.items if isinstance(v, (int, float))}
 if accs:
 return max(accs, key=accs.get), accs
 # 3) prior-run flat results.json ({base, arm: acc, ...})
 r = _read_json(work / "results.json")
 if r:
 accs = {k: v for k, v in r.items if k != "base" and isinstance(v, (int, float))}
 if accs:
 return max(accs, key=accs.get), accs
 return None, {}


def resolve_adapter(args) -> tuple[Path, str | None, dict]:
 work = Path(args.work)
 if args.adapter_dir:
 return Path(args.adapter_dir), None, {}
 if args.best:
 arm, accs = _best_from_results(work, args.seed)
 if not arm:
 raise SystemExit("[package] --best: no results file found to pick from. "
 "Pass --adapter-dir or --arm explicitly.")
 base = Path("adapters")
 cand = (base / f"seed{args.seed}" / arm) if args.seed is not None else base / arm
 if not cand.is_dir and args.seed is None:
 # aggregate/seeded results but adapters live under seed0
 for s in (0, 1):
 if (base / f"seed{s}" / arm).is_dir:
 cand = base / f"seed{s}" / arm
 break
 return cand, arm, accs
 if args.arm:
 base = Path("adapters")
 cand = (base / f"seed{args.seed}" / args.arm) if args.seed is not None else base / args.arm
 return cand, args.arm, {}
 raise SystemExit("[package] specify one of --adapter-dir / --best / --arm")


def main -> None:
 ap = argparse.ArgumentParser(description="Build Kaggle submission.zip from a LoRA adapter")
 g = ap.add_argument_group("adapter selection")
 g.add_argument("--adapter-dir", type=Path, help="explicit adapter directory")
 g.add_argument("--best", action="store_true", help="auto-pick highest-val-acc arm")
 g.add_argument("--arm", help="arm name, e.g. C_gamma_consensus")
 g.add_argument("--seed", type=int, default=None, help="seed subdir under adapters/")
 ap.add_argument("--work", default="work", help="results dir for --best (default work)")
 ap.add_argument("--output", type=Path, default=Path("submission.zip"))
 ap.add_argument("--base-model-id", default=None,
 help="rewrite adapter_config.json base_model_name_or_path in the ZIP "
 "(leaves the on-disk adapter untouched). Use the official HF id.")
 ap.add_argument("--include-tokenizer", action="store_true",
 help="also include tokenizer files (default: skip)")
 ap.add_argument("--minimal", action="store_true",
 help="zip ONLY adapter_config.json + weights (drop README/chat_template)")
 args = ap.parse_args

 adapter, arm, accs = resolve_adapter(args)
 if not adapter.is_dir:
 raise SystemExit(f"[package] adapter dir not found: {adapter}")

 cfg_path = adapter / "adapter_config.json"
 if not cfg_path.is_file:
 raise SystemExit(f"[package] missing {cfg_path}")
 config = json.loads(cfg_path.read_text)

 # ---- validation gates ----
 r = config.get("r")
 if r is None:
 raise SystemExit("[package] adapter_config.json missing 'r' (rank)")
 if r > 32:
 raise SystemExit(f"[package] LoRA rank {r} exceeds the max of 32 -- REJECTED")
 weights = [f for f in os.listdir(adapter)
 if f.endswith(".safetensors") or f.endswith(".bin")]
 if not weights:
 raise SystemExit("[package] no adapter weights (.safetensors/.bin) found")

 base_id = config.get("base_model_name_or_path", "NOT SET")
 local_base = isinstance(base_id, str) and (base_id.startswith("/") or "/opt/" in base_id)

 print(f"[package] adapter : {adapter}")
 if arm:
 acc = accs.get(arm)
 print(f"[package] arm : {arm}" + (f" (local val acc {acc:.3f})" if isinstance(acc, float) else ""))
 print(f"[package] rank : {r} (<= 32 OK) alpha={config.get('lora_alpha')}")
 print(f"[package] targets : {config.get('target_modules')}")
 print(f"[package] base : {base_id}")
 if local_base and not args.base_model_id:
 print("[package] *** WARNING: base_model_name_or_path is a LOCAL path. Kaggle "
 "cannot load it. Most harnesses override the base & just attach the\n"
 "[package] adapter, but if the submission errors, re-run with "
 "--base-model-id <official_hf_id> (verify the id from the demo).")

 # ---- choose files ----
 if args.minimal:
 keep = lambda n: n in REQUIRED_ONLY
 else:
 skip = set if args.include_tokenizer else set(SKIP_NAMES)
 keep = lambda n: n not in skip

 # ---- optionally rewrite base id into a temp copy of the config ----
 tmp_cfg = None
 if args.base_model_id:
 patched = dict(config)
 patched["base_model_name_or_path"] = args.base_model_id
 tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
 json.dump(patched, tmp); tmp.close
 tmp_cfg = Path(tmp.name)
 print(f"[package] rewrote base_model_name_or_path -> {args.base_model_id} (in zip only)")

 out = args.output
 if out.exists:
 out.unlink
 n = 0
 with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
 for root, _dirs, files in os.walk(adapter):
 for name in sorted(files):
 if not keep(name):
 continue
 src = Path(root) / name
 arc = src.relative_to(adapter)
 if name == "adapter_config.json" and tmp_cfg is not None:
 zf.write(tmp_cfg, str(arc))
 else:
 zf.write(src, str(arc))
 n += 1
 if tmp_cfg is not None:
 tmp_cfg.unlink(missing_ok=True)

 with zipfile.ZipFile(out, "r") as zf:
 infos = zf.infolist
 names = [i.filename for i in infos]
 print(f"\n[package] {out} ({out.stat.st_size:,} bytes, {n} files):")
 for i in infos:
 print(f" {i.filename} ({i.file_size:,} B)")

 # final sanity
 problems = []
 if "adapter_config.json" not in names:
 problems.append("adapter_config.json missing from zip")
 if not any(x.endswith((".safetensors", ".bin")) for x in names):
 problems.append("no weights in zip")
 if problems:
 raise SystemExit("[package] FAILED sanity: " + "; ".join(problems))
 print(f"\n[package] OK -> upload {out} to Kaggle. "
 f"(Predict the score first with scripts_submission_eval.py.)")


if __name__ == "__main__":
 main
