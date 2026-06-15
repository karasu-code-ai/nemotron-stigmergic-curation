r"""Competition metric — RECONCILED against the OFFICIAL scorer (RUNBOOK Part 0).

Part 0 is DONE. The two functions `extract_final_answer` and `verify` below are
copied VERBATIM from the official Kaggle metric kernel
`metric/nvidia-nemotron-metric` (cell "Metric for NVIDIA (129716)", pulled
2026-05-30 via the Kaggle API). They ARE the ruler. Everything else in this file
is a thin backward-compat shim so the rest of the repo (generate.py, evaluate.py)
keeps its `extract_answer(text)` / `is_correct(pred, gold)` API.

What changed vs our old v2 reimplementation (all to MATCH the official):
 - String comparison is `.lower` + outer `.strip` ONLY. We previously
 collapsed *internal* whitespace ("queen discovers" vs "queen discovers") —
 the official does NOT, so that was too lenient. Removed.
 - No-box fallback now follows the official chain: "final answer is:" patterns
 -> last number -> last non-empty line. We previously did last-number only,
 which under-extracted.
 - Numeric tolerance is `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`. We previously
 had no abs_tol and a hand-rolled gold==0 branch.
 - Bitstrings are still string-compared (official guards `re.fullmatch('[01]+')`),
 so an off-by-one-bit answer is still correctly REJECTED. Our old "bitstring
 trap" intuition matched the official here.
 - Extraction returns 'NOT_FOUND' (a string), never None.

Official scoring context (from the kernel's `score` defaults; we don't run vLLM
here but these define the ruler): rel_tol=1e-2, abs_tol=1e-5; submission is graded
by `verify(str(ground_truth), str(extracted_answer))` per row, accuracy = mean.
The official prompt suffix appended to each problem is:
 "Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`"
and generation uses the chat template with enable_thinking=True.
"""
from __future__ import annotations
import math
import re
from typing import Optional

# Exposed for any importer; values match the official kernel exactly.
REL_TOL = 1e-2
ABS_TOL = 1e-5


# ---------------------------------------------------------------------------
# OFFICIAL metric functions — VERBATIM from metric/nvidia-nemotron-metric.
# Do not "improve" these; divergence from them makes B-vs-B0 untrustworthy.
# ---------------------------------------------------------------------------
def extract_final_answer(text: str | None) -> str:
 r"""Extracts the final answer from the model response.

 Prioritizes extracting answers inside `\boxed{}`.
 If no `\boxed{}` format is found, attempts to extract numbers from other formats.

 Examples:
 >>> extract_final_answer(r"The answer is \boxed{42}")
 '42'
 >>> extract_final_answer("The final answer is: 3.14")
 '3.14'
 >>> extract_final_answer("Just a number 100 in text")
 '100'
 >>> extract_final_answer(None)
 'NOT_FOUND'
 """
 if text is None:
 return 'NOT_FOUND'

 # Search for boxed answer. For each \boxed{ occurrence, take everything up
 # to the last } before the next \boxed{ (or end of text). This handles
 # answers that themselves contain '}' (the model writes them literally,
 # producing e.g. \boxed{}52} for the answer "}52") as well as nested LaTeX
 # like \boxed{\frac{1}{2}}.
 boxed_starts = list(re.finditer(r'\\boxed\{', text))
 matches = []
 for i, m in enumerate(boxed_starts):
 start = m.end
 end = boxed_starts[i + 1].start if i + 1 < len(boxed_starts) else len(text)
 segment = text[start:end]
 last_brace = segment.rfind('}')
 matches.append(segment[:last_brace] if last_brace != -1 else segment)
 if matches:
 non_empty = [m.strip for m in matches if m.strip]
 if non_empty:
 return non_empty[-1]
 return matches[-1].strip

 # Other common formats if \boxed{} is not found
 patterns = [
 r'The final answer is:\s*([^\n]+)',
 r'Final answer is:\s*([^\n]+)',
 r'Final answer\s*[:：]\s*([^\n]+)',
 r'final answer\s*[:：]\s*([^\n]+)',
 ]
 for pattern in patterns:
 matches = re.findall(pattern, text, re.IGNORECASE)
 if matches:
 return matches[-1].strip

 # If no structured format is found, extract the last valid number in the text
 matches = re.findall(r'-?\d+(?:\.\d+)?', text)
 if matches:
 return matches[-1]

 # If no numeric answer is found, return the last line of text as a fallback
 lines = [line.strip for line in text.splitlines if line.strip]
 return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer: str, predicted: str) -> bool:
 """Verify if the answer matches.

 For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
 otherwise, compare strictly as strings (case-insensitive).

 Examples:
 >>> verify("10011000", "10011000")
 True
 >>> verify("10011000", "10011001")
 False
 >>> verify("24.64", "24.6401")
 True
 >>> verify("XLVII", "xlvii")
 True
 >>> verify("11011", "00011011")
 False
 """
 # Clean up strings
 stored_answer = stored_answer.strip
 predicted = predicted.strip

 # If the answer is a binary string, compare strictly as strings
 if re.fullmatch(r'[01]+', stored_answer):
 return predicted.lower == stored_answer.lower

 try:
 # Try to convert the answers to floating point numbers
 stored_num = float(stored_answer)
 predicted_num = float(predicted)
 # Use a small absolute tolerance for numbers near zero
 return math.isclose(stored_num, predicted_num, rel_tol=REL_TOL, abs_tol=ABS_TOL)
 except Exception:
 # Fallback to case-insensitive string comparison
 return predicted.lower == stored_answer.lower


# ---------------------------------------------------------------------------
# Backward-compatible shim used by generate.py / evaluate.py.
# ---------------------------------------------------------------------------
def extract_answer(text: Optional[str]) -> str:
 r"""Official extraction. Returns the answer string (or 'NOT_FOUND')."""
 return extract_final_answer(text)


def is_correct(pred: Optional[str], gold: str, rel_tol: float = REL_TOL) -> bool:
 """Official comparison. NOTE official arg order is verify(stored=gold, predicted=pred).

 `rel_tol` is accepted for backward compatibility; the official metric is fixed
 at REL_TOL/ABS_TOL, so a non-default rel_tol is ignored to keep us on-ruler.
 """
 if pred is None:
 return False
 return verify(str(gold), str(pred))


def score_predictions(preds: list[str], golds: list[str], rel_tol: float = REL_TOL) -> float:
 """Accuracy over raw model outputs vs gold answer strings, official-style."""
 assert len(preds) == len(golds), "preds/golds length mismatch"
 if not preds:
 return 0.0
 n = sum(int(verify(str(g), str(extract_final_answer(p)))) for p, g in zip(preds, golds))
 return n / len(preds)


if __name__ == "__main__":
 # --- official docstring examples (the ruler's own truth table) ---
 assert extract_final_answer(r"The answer is \boxed{42}") == "42"
 assert extract_final_answer("The final answer is: 3.14") == "3.14"
 assert extract_final_answer("Just a number 100 in text") == "100"
 assert extract_final_answer(None) == "NOT_FOUND"
 assert verify("10011000", "10011000") is True # bitstring exact
 assert verify("10011000", "10011001") is False # off-by-one bit -> REJECT
 assert verify("24.64", "24.6401") is True # within rel tol
 assert verify("XLVII", "xlvii") is True # case-insensitive string
 assert verify("11011", "00011011") is False # padded bitstring -> REJECT

 # --- shim maps correctly: is_correct(pred, gold) == verify(gold, pred) ---
 assert is_correct("11011101", "11011101") # bitstring exact
 assert not is_correct("11011100", "11011101") # off-by-one bit
 assert is_correct("42.005", "42") # 0.012% within 1% rel
 assert not is_correct("43", "42") # 2.4% rel -> reject
 assert extract_answer(r"...therefore \boxed{11011101}.") == "11011101"
 assert extract_answer(r"\boxed{queen discovers near valley}") == "queen discovers near valley"
 # internal-whitespace mismatch is REJECTED by the official ruler (was wrongly
 # accepted by our old reimplementation):
 assert not is_correct("queen discovers near valley", "queen discovers near valley")
 assert is_correct("queen discovers near valley", "Queen discovers near valley") # case only
 print("metric.py self-check passed (reconciled with official metric)")
