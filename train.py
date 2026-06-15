r"""LoRA SFT wrapper — trl-FREE (peft + transformers Trainer).

Why no trl: on this Spark the model needs transformers==4.53.3 (the NemotronH
remote code breaks on 4.54+'s cache refactor), but modern trl (1.5.x) imports
`is_trackio_available` which only exists in transformers>=4.56 -> hard ImportError,
and older trl drags accelerate into an unresolvable pin war. peft 0.19.1 +
transformers.Trainer import cleanly under 4.53.3, so we drive SFT directly. We
pre-tokenize and mask the prompt so loss is computed on the COMPLETION only
(matches rejection-sampling SFT / STaR).

Model specifics (verified against the local weights + official submission demo):
 - device_map="cuda:0" (NOT "auto": auto offloads to CPU on the GB10 unified-mem
 box -> OOM, same trap fixed in generate.py/evaluate.py).
 - LoRA target_modules = in_proj/out_proj (Mamba) + up_proj/down_proj (MoE experts),
 the official set; these names exist in the checkpoint.
 - import nemotron_compat first (no-op now that real mamba-ssm kernels are built).
"""
from __future__ import annotations
from pathlib import Path

from config import Config

# Official LoRA targets for NemotronH (Mamba in/out_proj + MoE up/down_proj).
LORA_TARGET_MODULES = ["in_proj", "out_proj", "up_proj", "down_proj"]


def _build_prompt_and_full(ex: dict, tok) -> tuple[str, str]:
 r"""Return (prompt_text, full_text) using the OFFICIAL chat format so train-time
 matches generation-time (official boxed suffix + chat template w/ thinking).
 prompt_text = everything up to the assistant turn (for loss masking);
 full_text = prompt + the curated completion."""
 suffix = ("\nPlease put your final answer inside `\\boxed{}`. "
 "For example: `\\boxed{your answer}`")
 user = ex["problem"] + suffix
 try:
 prompt_text = tok.apply_chat_template(
 [{"role": "user", "content": user}],
 tokenize=False, add_generation_prompt=True, enable_thinking=True)
 except Exception:
 prompt_text = f"User: {user}\nAssistant: "
 full_text = prompt_text + ex["completion"] + (tok.eos_token or "")
 return prompt_text, full_text


def train_lora(corpus: list[dict], cfg: Config, run_name: str) -> Path:
 """Fine-tune a LoRA adapter on `corpus`; save under adapter_dir/run_name."""
 import torch
 import nemotron_compat
 nemotron_compat.apply # no-op once real mamba-ssm kernels are present
 from datasets import Dataset
 from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
 TrainingArguments, default_data_collator)
 from peft import LoraConfig, get_peft_model

 cfg.ensure_dirs
 out_dir = cfg.adapter_dir / run_name

 tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
 if tok.pad_token is None:
 tok.pad_token = tok.eos_token

 # --- pre-tokenize with completion-only loss masking ---
 max_len = cfg.max_seq_len

 def encode(ex):
 prompt_text, full_text = _build_prompt_and_full(ex, tok)
 full = tok(full_text, truncation=True, max_length=max_len,
 add_special_tokens=False)
 plen = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
 ids = full["input_ids"]
 labels = list(ids)
 for i in range(min(plen, len(labels))):
 labels[i] = -100 # mask the prompt; train on the completion only
 full["labels"] = labels
 return full

 rows = [encode(ex) for ex in corpus]
 # pad to the longest in the (tiny) corpus; keeps it simple and correct
 maxlen = max(len(r["input_ids"]) for r in rows)
 pad_id = tok.pad_token_id
 for r in rows:
 n = maxlen - len(r["input_ids"])
 r["input_ids"] = r["input_ids"] + [pad_id] * n
 r["attention_mask"] = r["attention_mask"] + [0] * n
 r["labels"] = r["labels"] + [-100] * n
 ds = Dataset.from_list(rows)

 model = AutoModelForCausalLM.from_pretrained(
 cfg.model_id, trust_remote_code=True,
 torch_dtype=torch.bfloat16, device_map="cuda:0", # NOT auto (OOM trap))
 model.config.use_cache = False

 peft_cfg = LoraConfig(
 r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
 bias="none", task_type="CAUSAL_LM",
 target_modules=LORA_TARGET_MODULES,)
 model = get_peft_model(model, peft_cfg)
 model.print_trainable_parameters

 args = TrainingArguments(
 output_dir=str(out_dir),
 per_device_train_batch_size=cfg.batch_size,
 gradient_accumulation_steps=cfg.grad_accum,
 learning_rate=cfg.learning_rate,
 num_train_epochs=cfg.epochs,
 logging_steps=5, save_strategy="no",
 bf16=True, report_to=[], seed=cfg.seed,
 dataloader_pin_memory=False,)

 trainer = Trainer(
 model=model, args=args, train_dataset=ds,
 data_collator=default_data_collator,)
 trainer.train
 model.save_pretrained(str(out_dir)) # saves the LoRA adapter only
 tok.save_pretrained(str(out_dir))
 print(f"[train] {run_name}: adapter saved -> {out_dir}", flush=True)
 return out_dir
