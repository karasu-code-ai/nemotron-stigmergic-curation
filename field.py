r"""The stigmergic trace field. THE novel component. v2 (per-problem consensus).

Idea (Grasse 1959, applied to reasoning): the work already done is the stimulus
for the next work. Decompose SUCCESSFUL traces into fragments; let a field over
fragments accumulate weight by CONSENSUS across INDEPENDENT successful traces,
with EVAPORATION removing idiosyncratic paths. Then prefer traces that traverse
high-consensus reasoning.

*** v2 CRITICAL CHANGE for THIS competition ***
Every puzzle has a UNIQUE hidden rule. So cross-problem fragment consensus (v1)
mostly rewards generic boilerplate ("let me look at the examples"), which is
noise. The meaningful unit here is WITHIN-PROBLEM consensus: independent traces
solving the SAME puzzle that converge on the same intermediate finding (e.g.
"output bit 0 = input bit 3 XOR input bit 7"). Agreement among independent
solvers about a hidden truth is exactly what stigmergic convergence should
detect -- arguably a cleaner test of the idea than v1.

 cfg.consensus_scope = "per_problem" (default) -> weights keyed (problem_id, fragment)
 cfg.consensus_scope = "cross_problem" -> v1 behavior (global fragment weights)

A second signal also survives across problems: PROCEDURAL consensus (method
fragments that recur even when rules differ). cross_problem mode captures that;
keep it available for comparison.

Keep fragmentation CRUDE. Line/step splitting is fine -- let the FIELD carry the
structure, not the parser.
"""
from __future__ import annotations
import re
from collections import defaultdict
from dataclasses import dataclass, field as dfield

from generate import Trace
from config import Config


_STEP_MARKERS = re.compile(r"(?:^|\n)\s*(?:step\s*\d+[:.)]|\d+[.)]\s)", re.I)


def normalize_fragment(s: str) -> str:
 """Light normalization so near-identical steps collide, but distinct steps
 stay distinct. Keeps digits/letters (bit positions, indices matter), drops
 punctuation and case. Moderate on purpose."""
 s = s.strip.lower
 s = re.sub(r"\s+", " ", s)
 s = re.sub(r"[^\w\s]", "", s)
 return s.strip


def fragment_trace(text: str, min_chars: int = 12) -> list[str]:
 r"""Crude fragmentation: drop the \boxed answer, split on newlines and step
 markers, normalize, drop trivia. De-duplicate within a trace so one trace
 casts at most one vote per fragment."""
 text = re.sub(r"\\boxed\{[^}]*\}", "", text)
 frags: list[str] = []
 for part in re.split(r"\n+", text):
 for sub in _STEP_MARKERS.split(part):
 f = normalize_fragment(sub)
 if len(f) >= min_chars:
 frags.append(f)
 return list(dict.fromkeys(frags))


# The 4 Nemotron ablation personas are ONE source (shared weights → correlated
# agreement). Map every persona to its true model family so consensus counts
# INDEPENDENT sources, not samples. Extend as new families are added.
NEMOTRON_PERSONAS = {"positional", "hypothesis_test", "pattern_diff", "decompositional"}

def source_of(persona: str) -> str:
 """Map a trace's persona to its true independent model source. Persona diversity
 is NOT source diversity (the spine finding): 4 Nemotron personas share weights, so
 their agreement is correlated, not corroborating. Counting sources (not personas/
 traces) is what makes cross-source consensus a real signal."""
 p = (persona or "").lower
 if persona in NEMOTRON_PERSONAS or p.startswith("nemotron"):
 return "nemotron"
 if p.startswith("gemma"):
 return "gemma"
 if p.startswith("phi"):
 return "phi"
 if p.startswith("gpt") or "oss" in p:
 return "gptoss"
 return p or "unknown" # unknown persona → its own source (conservative)


def _scope(cfg: Config) -> str:
 return getattr(cfg, "consensus_scope", "per_problem")


@dataclass
class StigmergicField:
 cfg: Config
 # in per_problem mode keys are (problem_id, fragment); in cross_problem mode
 # keys are fragment strings. Same dict, different key type.
 weight: dict = dfield(default_factory=lambda: defaultdict(float))
 n_traces: dict = dfield(default_factory=lambda: defaultdict(int))
 problems_seen: dict = dfield(default_factory=lambda: defaultdict(set))
 # --- always-maintained dual weights for the gamma (cluster-weighted) score,
 # regardless of consensus_scope. w_within keyed (problem_id, fragment);
 # w_cross keyed fragment. Lets build_C discount boilerplate (fragments common
 # across MANY problems) via a within/cross ratio ( cluster-weighted
 # aggregation; #62 congruence != correlation) without disturbing `weight`,
 # which A/B still use.
 w_within: dict = dfield(default_factory=lambda: defaultdict(float))
 w_cross: dict = dfield(default_factory=lambda: defaultdict(float))
 # --- source-aware consensus (multi-source field). Which DISTINCT model sources
 # reached each fragment. sources_within keyed (problem_id, fragment); sources_cross
 # keyed fragment. Counting SOURCES (not traces) is the fix for the degenerate
 # single-source field: 4 Nemotron personas reaching a fragment = 1 source, not 4
 # votes ( correlated-estimator failure). Set-membership, so it does NOT
 # evaporate — "source X corroborated this" is a fact, not a decaying deposit.
 sources_within: dict = dfield(default_factory=lambda: defaultdict(set))
 sources_cross: dict = dfield(default_factory=lambda: defaultdict(set))

 def _key(self, problem_id: str, frag: str):
 if _scope(self.cfg) == "per_problem":
 return (problem_id, frag)
 return frag

 def deposit(self, traces: list[Trace]) -> None:
 """Only SUCCESSFUL traces deposit. Each distinct successful trace is one
 independent vote per fragment. In per_problem mode a vote counts only
 within its own puzzle."""
 for t in traces:
 if not t.correct:
 continue
 src = source_of(t.persona)
 for frag in fragment_trace(t.text, self.cfg.min_fragment_chars):
 k = self._key(t.problem_id, frag)
 self.weight[k] += self.cfg.deposit_amount
 self.n_traces[k] += 1
 self.problems_seen[frag].add(t.problem_id) # for diagnostics
 # dual weights for the gamma score (kept independent of scope)
 self.w_within[(t.problem_id, frag)] += self.cfg.deposit_amount
 self.w_cross[frag] += self.cfg.deposit_amount
 # source-aware: record WHICH independent source corroborated this
 self.sources_within[(t.problem_id, frag)].add(src)
 self.sources_cross[frag].add(src)

 def evaporate(self) -> None:
 """Decay all weights; prune the faint. Evaporation is the regularizer
 against premature lock-in. Tune the rate: too slow -> lock onto a
 mediocre attractor; too fast -> nothing accrues."""
 r = self.cfg.evaporation_rate
 for k in list(self.weight.keys):
 self.weight[k] *= (1.0 - r)
 if self.weight[k] < self.cfg.min_fragment_weight:
 del self.weight[k]
 self.n_traces.pop(k, None)
 # decay the dual gamma weights in lock-step (same regularizer)
 for d in (self.w_within, self.w_cross):
 for k in list(d.keys):
 d[k] *= (1.0 - r)
 if d[k] < self.cfg.min_fragment_weight:
 del d[k]

 def settle(self, passes: int = 3) -> "StigmergicField":
 for _ in range(passes):
 self.evaporate
 return self

 def trace_score(self, t: Trace) -> float:
 """Mean field weight over a trace's fragments = how much CONSENSUS the
 trace traverses. In per_problem mode this is consensus about THIS
 puzzle's rule among independent solvers of the same puzzle. MEAN (not
 sum) so we reward shared reasoning, not length.

 When cfg.source_aware is set (multi-source field), delegate to the
 source-counting score so one over-sampled source can't fake consensus."""
 if getattr(self.cfg, "source_aware", False):
 return self.trace_score_xsrc(t)
 frags = fragment_trace(t.text, self.cfg.min_fragment_chars)
 if not frags:
 return 0.0
 total = sum(self.weight.get(self._key(t.problem_id, f), 0.0) for f in frags)
 return total / len(frags)

 def trace_score_xsrc(self, t: Trace) -> float:
 """Source-aware consensus: mean over the trace's fragments of the number of
 DISTINCT independent sources (model families) that reached each fragment on
 THIS problem. Counting sources, not traces, is the fix for the multi-source
 field — 4 Nemotron personas converging is 1 source of agreement, not 4
 (the correlated-estimator failure that made the single-source field
 degenerate). A fragment corroborated by gemma + gpt-oss + nemotron scores 3."""
 frags = fragment_trace(t.text, self.cfg.min_fragment_chars)
 if not frags:
 return 0.0
 total = sum(len(self.sources_within.get((t.problem_id, f),)) for f in frags)
 return total / len(frags)

 def trace_score_xsrc_gamma(self, t: Trace) -> float:
 """Source-aware AND boilerplate-discounted. Numerator = distinct independent
 sources that reached the fragment on THIS problem (cross-source corroboration,); denominator = the fragment's cross-PROBLEM frequency weight (boilerplate
 reaches every puzzle → discounted). The principled multi-source curator:
 reward fragments that independent families agree on AND that are specific to
 this puzzle."""
 frags = fragment_trace(t.text, self.cfg.min_fragment_chars)
 if not frags:
 return 0.0
 zeta = getattr(self.cfg, "gamma_zeta", 0.1)
 total = 0.0
 for f in frags:
 ns = len(self.sources_within.get((t.problem_id, f),))
 total += ns / (zeta + self.w_cross.get(f, 0.0))
 return total / len(frags)

 def trace_score_gamma(self, t: Trace) -> float:
 """Cluster-weighted / diversity-discounted consensus score (#62).

 v2's per_problem score rewards any fragment many same-puzzle traces share
 -- but that INCLUDES generic boilerplate ("look at the examples"), which
 is *correlated* agreement, not *informative* agreement. The fix, straight
 from cluster-weighted forecast aggregation: weight a fragment's
 within-problem consensus by how SPECIFIC it is to this problem, i.e.
 discount fragments that are common across MANY problems.

 gamma(frag) = w_within[(pid,frag)] / (zeta + w_cross[frag])

 A fragment reached by k independent solvers of THIS puzzle but rare
 elsewhere scores high; boilerplate reached by everyone everywhere scores
 ~1 (w_within ~ w_cross). zeta avoids singularities ( used 0.1). MEAN
 over the trace's fragments, like trace_score, so we reward shared
 problem-SPECIFIC reasoning, not length.

 When cfg.source_aware is set, delegate to the source-counted variant so the
 within-problem numerator counts independent SOURCES, not correlated samples."""
 if getattr(self.cfg, "source_aware", False):
 return self.trace_score_xsrc_gamma(t)
 frags = fragment_trace(t.text, self.cfg.min_fragment_chars)
 if not frags:
 return 0.0
 zeta = getattr(self.cfg, "gamma_zeta", 0.1)
 total = 0.0
 for f in frags:
 wi = self.w_within.get((t.problem_id, f), 0.0)
 wc = self.w_cross.get(f, 0.0)
 total += wi / (zeta + wc)
 return total / len(frags)

 def summary(self) -> str:
 if not self.weight:
 return "[field] empty -- no successful traces deposited?"
 scope = _scope(self.cfg)
 top = sorted(self.weight.items, key=lambda x: -x[1])[:8]
 if scope == "per_problem":
 # how many DISTINCT puzzles have at least one consensus fragment
 # (weight from >=2 traces): the signal we actually rely on here.
 multi = sum(1 for k, c in self.n_traces.items if c >= 2)
 body = " | ".join(f"{w:.1f}:{k[1][:28]}" for k, w in top)
 return (f"[field/per_problem] {len(self.weight)} (problem,frag) keys; "
 f"{multi} reached by >=2 traces (within-problem consensus).\n"
 f" top: {body}")
 else:
 cross = sum(1 for ps in self.problems_seen.values if len(ps) > 1)
 body = " | ".join(f"{w:.1f}:{str(k)[:28]}" for k, w in top)
 return (f"[field/cross_problem] {len(self.weight)} fragments, "
 f"{cross} span >1 problem.\n top: {body}")
