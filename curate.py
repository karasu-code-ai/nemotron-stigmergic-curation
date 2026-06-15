r"""Corpus construction = the ablation. Four arms, IDENTICAL budget.

 B0 random correct trace per problem (vanilla rejection-sampling SFT / STaR)
 B1 best correct trace per problem (shortest) (confidence-proxy baseline)
 A highest field-traffic correct trace (stigmergic consensus)
 B A + frontier (learning-progress) weighting (full mechanism)

HYPOTHESIS: B > A > B1 ~= B0 at equal budget.
If B clears the baselines at EQUAL budget, the curation signal carries real
information -- that's the result, independent of the leaderboard.

The equal-budget control is the whole experiment. If you change budget_examples,
change it for all four arms.
"""
from __future__ import annotations
import random
from collections import defaultdict

from generate import Trace
from field import StigmergicField
from config import Config


SFTExample = dict # {"problem_id","problem","answer","completion"}


def _by_problem(traces: list[Trace]) -> dict[str, list[Trace]]:
 d: dict[str, list[Trace]] = defaultdict(list)
 for t in traces:
 d[t.problem_id].append(t)
 return d


def _correct(traces: list[Trace]) -> list[Trace]:
 return [t for t in traces if t.correct]


def _example(t: Trace) -> SFTExample:
 return {"problem_id": t.problem_id, "problem": t.problem,
 "answer": t.gold, "completion": t.text}


# ---- frontier / learning-progress signal ----------------------------------
def population_success(traces: list[Trace]) -> dict[str, float]:
 return {pid: sum(t.correct for t in ts) / len(ts)
 for pid, ts in _by_problem(traces).items}


def frontier_weights(traces: list[Trace], model_pass_rate: dict[str, float],
 cfg: Config) -> dict[str, float]:
 """w(problem) = pop_success * (1 - model_pass_rate). Zone of proximal
 development: population CAN solve it, model CANNOT yet. Drop the rest."""
 pop = population_success(traces)
 w: dict[str, float] = {}
 for pid, ps in pop.items:
 mp = model_pass_rate.get(pid, 0.0)
 if cfg.drop_population_failed and ps == 0.0:
 continue # unlearnable now
 if cfg.drop_model_solved and mp >= 1.0:
 continue # nothing left to learn
 w[pid] = ps * (1.0 - mp)
 return w


# ---- four corpus builders --------------------------------------------------
def build_B0(traces: list[Trace], cfg: Config) -> list[SFTExample]:
 rng = random.Random(cfg.seed)
 out = []
 for ts in _by_problem(traces).values:
 ok = _correct(ts)
 if ok:
 out.append(_example(rng.choice(ok)))
 return out


def build_B1(traces: list[Trace], cfg: Config) -> list[SFTExample]:
 out = []
 for ts in _by_problem(traces).values:
 ok = _correct(ts)
 if ok:
 out.append(_example(min(ok, key=lambda t: len(t.text))))
 return out


def build_A(traces: list[Trace], fld: StigmergicField, cfg: Config) -> list[SFTExample]:
 out = []
 for ts in _by_problem(traces).values:
 ok = _correct(ts)
 if ok:
 out.append(_example(max(ok, key=fld.trace_score)))
 return out


def build_B(traces: list[Trace], fld: StigmergicField,
 model_pass_rate: dict[str, float], cfg: Config) -> list[SFTExample]:
 fw = frontier_weights(traces, model_pass_rate, cfg)
 scored: list[tuple[float, SFTExample]] = []
 for pid, ts in _by_problem(traces).items:
 if pid not in fw: # dropped: solved or unlearnable
 continue
 ok = _correct(ts)
 if not ok:
 continue
 scored.append((fw[pid], _example(max(ok, key=fld.trace_score))))
 # frontier weight orders WHICH problems survive the fixed budget
 scored.sort(key=lambda x: -x[0])
 return [ex for _, ex in scored]


def build_C(traces: list[Trace], fld: StigmergicField, cfg: Config) -> list[SFTExample]:
 """Arm C = cluster-weighted consensus. Identical to A (one correct
 trace per problem) but selects by the BOILERPLATE-DISCOUNTED gamma score
 (within/cross fragment ratio) instead of raw field traffic. Tests whether
 discounting generic agreement -- the failure v2's own comment worries about --
 beats plain consensus (A) at equal budget."""
 out = []
 for ts in _by_problem(traces).values:
 ok = _correct(ts)
 if ok:
 out.append(_example(max(ok, key=fld.trace_score_gamma)))
 return out


def _frag_set(t: Trace, cfg: Config) -> set:
 from field import fragment_trace
 return set(fragment_trace(t.text, cfg.min_fragment_chars))


def _diverse_pick(ok: list[Trace], m: int, fld: StigmergicField, cfg: Config) -> list[Trace]:
 """Greedy max-diversity selection of up to m correct traces for ONE problem.
 Seed with the gamma-best trace, then repeatedly add the correct trace whose
 fragment set is LEAST similar (min Jaccard) to those already chosen. Keeps
 heterogeneous reasoning in the corpus instead of collapsing to one mode
 ( anti-collapse / speciation)."""
 if len(ok) <= 1:
 return list(ok)
 chosen = [max(ok, key=fld.trace_score_gamma)]
 fsets = {id(t): _frag_set(t, cfg) for t in ok}
 while len(chosen) < m and len(chosen) < len(ok):
 best, best_sim = None, 2.0
 for t in ok:
 if t in chosen:
 continue
 ft = fsets[id(t)]
 # similarity to the closest already-chosen trace (Jaccard)
 sim = 0.0
 for c in chosen:
 fc = fsets[id(c)]
 union = len(ft | fc) or 1
 sim = max(sim, len(ft & fc) / union)
 if sim < best_sim:
 best, best_sim = t, sim
 if best is None:
 break
 chosen.append(best)
 return chosen


def build_D(traces: list[Trace], fld: StigmergicField, cfg: Config) -> list[SFTExample]:
 """Arm D = diversity-preserving curation (/). Instead of one
 consensus trace per problem, keep up to cfg.diversity_per_problem DISTINCT
 correct traces per problem (greedy max-diversity), then flatten and truncate
 to the SAME budget. Tests whether retaining within-problem heterogeneity in
 the training corpus beats argmax-1 (i.e. whether consensus collapse, not
 signal quality, is what caps A/B)."""
 m = getattr(cfg, "diversity_per_problem", 3)
 rng = random.Random(cfg.seed)
 # order problems by their best gamma trace (so budget truncation keeps the
 # strongest problems), then within each emit up to m diverse traces.
 per_problem = []
 for pid, ts in _by_problem(traces).items:
 ok = _correct(ts)
 if not ok:
 continue
 picks = _diverse_pick(ok, m, fld, cfg)
 best = max(fld.trace_score_gamma(t) for t in picks)
 per_problem.append((best, [_example(t) for t in picks]))
 per_problem.sort(key=lambda x: -x[0])
 out: list[SFTExample] = []
 for _, exs in per_problem:
 out.extend(exs)
 return out


def enforce_budget(corpus: list[SFTExample], budget: int) -> list[SFTExample]:
 """Truncate to identical example count -- the control variable. Caller is
 responsible for ordering (B is pre-sorted by frontier weight; baselines are
 shuffled in build_all so truncation isn't biased by problem order)."""
 return corpus[:budget]


def build_all(traces: list[Trace], fld: StigmergicField,
 model_pass_rate: dict[str, float], cfg: Config) -> dict[str, list[SFTExample]]:
 rng = random.Random(cfg.seed)
 b0 = build_B0(traces, cfg); rng.shuffle(b0)
 b1 = build_B1(traces, cfg); rng.shuffle(b1)
 a = build_A(traces, fld, cfg); rng.shuffle(a)
 b = build_B(traces, fld, model_pass_rate, cfg) # keep frontier order
 c = build_C(traces, fld, cfg); rng.shuffle(c) # gamma-consensus
 d = build_D(traces, fld, cfg) # keep diversity order

 raw = {"B0_random": b0, "B1_shortest": b1, "A_consensus": a,
 "B_consensus_frontier": b, "C_gamma_consensus": c, "D_diversity": d}

 # EQUAL BUDGET is THE control variable. B drops model-solved/unlearnable
 # problems (esp. with the real frontier signal), so it can have fewer
 # eligible examples than the others. Pad-free fix: set the effective budget
 # to the SMALLEST arm (capped by the target), and truncate EVERY arm to it,
 # so all arms train on an identical example count. Each arm keeps its own
 # ordering (B=frontier, D=diversity, others shuffled) so truncation drops
 # each arm's *lowest-priority* examples, not arbitrary ones.
 sizes = {nm: len(v) for nm, v in raw.items}
 eff = min(cfg.budget_examples, min(sizes.values))
 corpora = {nm: enforce_budget(v, eff) for nm, v in raw.items}

 print(f"[curate] target budget={cfg.budget_examples} EFFECTIVE budget={eff} "
 f"(= min arm size; pre-truncation sizes {sizes})")
 for name, c in corpora.items:
 print(f"[curate] {name:24s}: {len(c)} examples")
 if eff < cfg.budget_examples:
 limiter = min(sizes, key=sizes.get)
 print(f"[curate] NOTE: effective budget capped by '{limiter}' "
 f"({sizes[limiter]} eligible). All arms truncated to {eff} to keep "
 f"the equal-budget control. If {limiter} is starved, lower "
 f"budget_examples or relax drop_* / generate more.")
 if len(set(len(c) for c in corpora.values)) > 1:
 print("[curate] WARNING: arms STILL differ in size -- investigate.")
 return corpora
