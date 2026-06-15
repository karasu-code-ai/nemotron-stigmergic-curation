r"""Central configuration v2. One place to tune the whole experiment.

v2 changes vs v1:
 - problem_col -> "prompt" (VERIFIED competition schema: id, prompt, answer)
 - model_id read from $NEMOTRON_MODEL so the SAME file works on Spark + Mac
 - added `backend` (dummy | ollama | transformers | vllm)
 - personas broadened off pure-math (task = logic puzzles: bit ops, algebra, etc.)
 - defaults are deliberately TINY (plumbing pass). Scale up per RUNBOOK Part 9.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


# Personas drive the "polyagent" generator: different reasoning STYLES from the
# SAME base model. Task is logical-reasoning puzzles (bit manipulation, algebra,
# etc.), so styles favor explicit symbol-pushing, concrete simulation, and
# verification -- all strong for deterministic-answer puzzles.
# Task = RULE INDUCTION: infer a hidden transformation from input->output
# examples, then apply it to a query. Personas are different STYLES of rule
# discovery -- all strong for this, and genuinely distinct so independent traces
# explore different hypotheses (which is what within-problem consensus rewards).
PERSONAS: dict[str, str] = {
 "positional": (
 "You discover transformation rules by analyzing each output position as "
 "a function of input positions. For bit/string tasks, determine how each "
 "output element depends on specific input elements. Verify your rule "
 "against ALL given examples before applying it. Put the final answer in "
 "\\boxed{}."),
 "hypothesis_test": (
 "You propose candidate rules (a shift, rotation, XOR mask, substitution, "
 "arithmetic relation, etc.), then test each against every example, "
 "discarding any that fail. Once a rule fits all examples, apply it. End "
 "with the answer in \\boxed{}."),
 "pattern_diff": (
 "You compare inputs to outputs to find what changed: which bits flip, "
 "how symbols map, what shifts. Build the rule from these differences, "
 "confirm it on all examples, then apply it to the query. End with "
 "\\boxed{}."),
 "decompositional": (
 "You break the transformation into independent parts (e.g. per-bit, "
 "per-character, or per-term), solve each mapping separately, then combine "
 "them. Show the decomposition, verify on examples, and end with "
 "\\boxed{}."),
}


@dataclass
class Config:
 # ---- paths
 data_dir: Path = Path("data")
 work_dir: Path = Path("work")
 adapter_dir: Path = Path("adapters")
 train_csv: str = "train.csv"
 test_csv: str = "test.csv"

 # ---- columns (VERIFIED: train=id,prompt,answer ; test=id,prompt)
 # answers are STRUCTURED STRINGS (8-bit binary, decrypted text), not
 # just numbers -> metric.py compares bitstrings/text as exact strings.
 problem_col: str = "prompt"
 answer_col: str = "answer"
 id_col: str = "id"

 # ---- model: overridable per-machine via env (Spark path is the default)
 model_id: str = os.environ.get(
 "NEMOTRON_MODEL", "/opt/ml/models/nemotron-3-nano-30b-a3b-bf16")
 max_seq_len: int = 4096 # PART 9 (was 2048): rule derivation is long

 # ---- backend: transformers is the safe default on ARM+Blackwell.
 # switch to vllm only if it actually loads; dummy/ollama for no-real-model runs.
 backend: str = "transformers"
 gen_batch_size: int = 8 # TransformersBackend batching; lower if OOM

 # ---- generation (the swarm) -- PART 9, TRIMMED to fit overnight on the GB10
 # (measured vLLM throughput ~25 tok/s aggregate => full 200x8x2temps x3072tok
 # is ~3 DAYS for 2 seeds; infeasible). Trimmed so SEED 0's full pipeline lands
 # by morning; seed 1 continues (resumable). Diversity still ample: 4 personas
 # x k=6 = 24 traces/problem for within-problem consensus. Re-raise for a final
 # maximal run once results justify the compute.
 k_samples: int = 6 # PART 9 trimmed (full was 8)
 temperatures: tuple[float, ...] = (0.8,) # PART 9 trimmed (full was (0.6,0.9))
 gen_max_new_tokens: int = 2048 # PART 9 trimmed (full 3072); eval allows 7680.
 personas: dict[str, str] = field(default_factory=lambda: dict(PERSONAS))

 # ---- stigmergic field
 # "per_problem" = consensus WITHIN each puzzle (right for unique-rule
 # puzzles; default). "cross_problem" = v1 global fragment consensus
 # (captures procedural/method recurrence). See field.py.
 consensus_scope: str = "per_problem"
 evaporation_rate: float = 0.15
 deposit_amount: float = 1.0
 min_fragment_weight: float = 0.05
 min_fragment_chars: int = 12
 # Arm C (gamma / cluster-weighted consensus): boilerplate-discount
 # constant. gamma(frag) = w_within / (gamma_zeta + w_cross). 0.1 per.
 gamma_zeta: float = 0.1
 # Arm D (diversity-preserving curation/): max DISTINCT correct
 # traces kept per problem before flattening to the equal budget.
 diversity_per_problem: int = 3

 # ---- frontier / learning-progress coupling
 drop_model_solved: bool = True
 drop_population_failed: bool = True
 # F1: use a REAL greedy base-model pass rate as model_pass_rate (the
 # frontier signal's 2nd term) instead of the population-solves proxy. When
 # true, run_ablation runs a cached greedy eval over the train subset.
 real_frontier_signal: bool = True

 # ---- equal-budget comparison (THE control variable). PART 9 (trimmed).
 budget_examples: int = 80 # PART 9 trimmed; keep EQUAL across all arms
 # (low enough that B, after drop_*, can fill it)

 # ---- LoRA / SFT
 lora_r: int = 16 # submission requires rank <= 32 (32 is the cap)
 lora_alpha: int = 32
 lora_dropout: float = 0.05
 learning_rate: float = 2e-4
 epochs: int = 2
 batch_size: int = 1
 grad_accum: int = 16

 # ---- experiment hygiene
 seed: int = 0 # overridable via run_ablation --seed (multi-seed run)
 val_size: int = 80 # FIXED held-out slice, NEVER trained on
 # (PART 9 trimmed from 100: eval is transformers
 # .generate x6 arms x2 seeds -- the bottleneck)
 subset_size: int = 120 # PART 9 trimmed (full 200): generation pool

 def ensure_dirs(self) -> None:
 for d in (self.work_dir, self.adapter_dir):
 d.mkdir(parents=True, exist_ok=True)
