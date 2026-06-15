#!/usr/bin/env python3
r"""R1 DISCOVERY (DECISIONS #32, R1 contingency): R0 yields X=94.6% on the rule_unknown
residual -> the hidden transformation is outside the open recipe's {add,abs_diff,mul,concat,
rev_concat} grammar. Before binning under R1 we must DISCOVER R1's grammar empirically.

Method (anti-overfit): define a FINITE, structurally-named op library (NOT 'whatever yields
gold'). For each problem, backtrack over a bijective symbol->digit map with op-per-operator-
symbol, trying the whole library; record which ops make ALL examples consistent and what the
query answer set is. We read which ops UNLOCK the residual (consistency on examples), then —
separately — whether the unlocked rule pins the query (determinacy). Gold is used only to
label D-gold/D-wrong AFTER the fact, never to select the rule.

This script is a DIAGNOSTIC over a sample; if a small op-set explains most of the residual,
that set becomes the registered R1 instrument.
"""
import json, math, os, signal, sys, time
from collections import Counter, defaultdict

WINNER = "/opt/ml/projects/winner_snapshot/nemotron-master"

# ---- structural op library (each: (name, fn) on two 0..99 ints; None = undefined) ----
def _ops:
 L = []
 L += [("add", lambda a, b: a + b)]
 L += [("absdiff", lambda a, b: abs(a - b))]
 L += [("mul", lambda a, b: a * b)]
 L += [("sub", lambda a, b: a - b if a >= b else None)]
 L += [("fdiv", lambda a, b: a // b if b else None)]
 L += [("mod", lambda a, b: a % b if b else None)]
 L += [("addmod100", lambda a, b: (a + b) % 100)]
 L += [("mulmod100", lambda a, b: (a * b) % 100)]
 L += [("submod100", lambda a, b: (a - b) % 100)]
 L += [("max", lambda a, b: max(a, b))]
 L += [("min", lambda a, b: min(a, b))]
 L += [("gcd", lambda a, b: math.gcd(a, b))]
 L += [("left", lambda a, b: a)]
 L += [("right", lambda a, b: b)]
 L += [("sumsq", lambda a, b: a + b)] # placeholder kept distinct? drop dup -> remove
 L = [(n, f) for n, f in L if n != "sumsq"]
 return L

OPS = _ops
# concat handled specially (4-digit padded), added as pseudo-ops by index sentinel
CONCAT = [("concat", None), ("revconcat", None)]


def digits(n):
 if n is None or n < 0:
 return None
 if n == 0:
 return (0,)
 d = []
 while n > 0:
 d.append(n % 10); n //= 10
 return tuple(reversed(d))


def concat_digits(a, b, rev):
 v = (b * 100 + a) if rev else (a * 100 + b)
 if v >= 10000:
 return None
 return (v // 1000, (v // 100) % 10, (v // 10) % 10, v % 10)


def parse(data):
 ex = []
 for e in data["examples"]:
 i = e["input_value"]; o = e["output_value"]
 ex.append((i[0], i[1], i[2], i[3], i[4], tuple(o)))
 q = data["question"]
 return ex, (q[0], q[1], q[2], q[3], q[4])


class Disc:
 """Backtracking: bijective symbol->digit + op-per-operator-symbol over the full library.
 Collects the set of query answers consistent with ALL examples."""
 def __init__(self, examples, query, cap=300):
 self.ex = examples; self.q = query; self.cap = cap
 self.m = {}; self.used = set; self.opass = {}
 self.answers = Counter; self.op_for_query = Counter

 def go(self):
 self._rec(0); return self.answers

 def _try_assign(self, sym, d):
 if sym in self.m:
 return 0 if self.m[sym] == d else None
 if d in self.used:
 return None
 self.m[sym] = d; self.used.add(d); return 1

 def _undo(self, sym, fl):
 if fl == 1:
 self.used.discard(self.m[sym]); del self.m[sym]

 def _vals(self, sym):
 return (self.m[sym],) if sym in self.m else tuple(d for d in range(10) if d not in self.used)

 def _rec(self, idx):
 if len(self.answers) >= self.cap:
 return
 if idx == len(self.ex):
 self._query; return
 s0, s1, op, s3, s4, r = self.ex[idx]
 rlen = len(r)
 for d0 in self._vals(s0):
 f0 = self._try_assign(s0, d0)
 if f0 is None: continue
 for d1 in self._vals(s1):
 f1 = self._try_assign(s1, d1)
 if f1 is None: continue
 lv = d0 * 10 + d1
 for d3 in self._vals(s3):
 f3 = self._try_assign(s3, d3)
 if f3 is None: continue
 for d4 in self._vals(s4):
 f4 = self._try_assign(s4, d4)
 if f4 is None: continue
 rv = d3 * 10 + d4
 cand_ops = [self.opass[op]] if op in self.opass else \
 list(range(len(OPS))) + ["concat", "revconcat"]
 for oid in cand_ops:
 if oid == "concat":
 rd = concat_digits(lv, rv, False)
 elif oid == "revconcat":
 rd = concat_digits(lv, rv, True)
 else:
 rd = digits(OPS[oid][1](lv, rv))
 if rd is None or len(rd) != rlen:
 continue
 assigns = []; ok = True
 for rs, rdig in zip(r, rd):
 fa = self._try_assign(rs, rdig)
 if fa is None: ok = False; break
 assigns.append((rs, fa))
 if ok:
 new = op not in self.opass
 if new: self.opass[op] = oid
 self._rec(idx + 1)
 if new: del self.opass[op]
 for rs, fa in reversed(assigns):
 self._undo(rs, fa)
 if len(self.answers) >= self.cap:
 self._undo(s4, f4); self._undo(s3, f3)
 self._undo(s1, f1); self._undo(s0, f0); return
 self._undo(s4, f4)
 self._undo(s3, f3)
 self._undo(s1, f1)
 self._undo(s0, f0)

 def _query(self):
 q0, q1, qop, q3, q4 = self.q
 for s in (q0, q1, q3, q4):
 if s not in self.m: return
 ql = self.m[q0] * 10 + self.m[q1]; qr = self.m[q3] * 10 + self.m[q4]
 cand = [self.opass[qop]] if qop in self.opass else list(range(len(OPS))) + ["concat", "revconcat"]
 d2s = {}
 for s, d in self.m.items:
 d2s.setdefault(d, s)
 for oid in cand:
 if oid == "concat": rd = concat_digits(ql, qr, False)
 elif oid == "revconcat": rd = concat_digits(ql, qr, True)
 else: rd = digits(OPS[oid][1](ql, qr))
 if rd is None: continue
 parts = []; ok = True
 for d in rd:
 if d not in d2s: ok = False; break
 parts.append(d2s[d])
 if not ok: continue
 ans = "".join(parts)
 self.answers[ans] += 1
 opname = OPS[oid][0] if isinstance(oid, int) else oid
 self.op_for_query[opname] += 1


def main:
 signal.signal(signal.SIGALRM, lambda *a: (_ for _ in).throw(TimeoutError))
 n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
 to = int(sys.argv[2]) if len(sys.argv) > 2 else 15
 probs = [json.loads(l) for l in open(os.path.join(WINNER, "problems.jsonl"))]
 ru = [p["id"] for p in probs if p["category"] == "cryptarithm_deduce" and p["status"] == "rule_unknown"][:n]

 unlocked = 0; reached_gold = 0; timeouts = 0; op_usage = Counter
 binc = Counter
 t0 = time.time
 for i, pid in enumerate(ru):
 data = json.loads(open(os.path.join(WINNER, "problems", f"{pid}.jsonl")).readline)
 gold = data["answer"]; ex, q = parse(data)
 try:
 signal.alarm(to)
 ans = Disc(ex, q).go
 signal.alarm(0)
 except TimeoutError:
 signal.alarm(0); timeouts += 1; binc["X-timeout"] += 1; continue
 nd = len(ans)
 if nd == 0:
 binc["X"] += 1; continue
 unlocked += 1
 gin = gold in ans
 if gin: reached_gold += 1
 if nd == 1:
 binc["D-gold" if gold in ans else "D-wrong"] += 1
 else:
 binc["U-goldin" if gin else "U-nogold"] += 1
 if (i + 1) % 20 == 0:
 print(f" [{i+1}/{len(ru)}] {time.time-t0:.0f}s unlocked={unlocked} gold={reached_gold}", flush=True)

 print("\n[R1-discover] sample =", len(ru), f" ({time.time-t0:.0f}s, timeouts={timeouts})")
 print(" bins:", dict(binc))
 print(f" examples-consistent (unlocked, >=1 answer): {unlocked}/{len(ru)} = {unlocked/len(ru):.2f}")
 print(f" gold reachable under R1 library: {reached_gold}/{len(ru)} = {reached_gold/len(ru):.2f}")


if __name__ == "__main__":
 main
