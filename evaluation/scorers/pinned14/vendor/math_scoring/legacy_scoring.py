"""Score explicit final answers only; intermediate correct numbers cannot earn credit."""
import re
from functools import lru_cache
from math_verify import parse, verify, LatexExtractionConfig


def final_answer(text):
    locations = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
    if locations:
        start = locations[-1].end()
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                return text[start:end].strip()
        return None
    markers = list(re.finditer(r'(?:####|(?:final\s+)?answer\s*(?:is|:))\s*([^\n]+)', text, re.I))
    if markers:
        return markers[-1].group(1).strip().rstrip('.').strip('$ ')
    single = list(re.finditer(r'\\boxed\s+([^\s$])', text))
    return single[-1].group(1) if single else None


@lru_cache(maxsize=30000)
def parsed_answer(answer):
    return parse('\\boxed{' + answer + '}', extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
                 extraction_mode='first_match', parsing_timeout=2)


def score_answer(text, gold, domain=None):
    answer = final_answer(text)
    if answer is None:
        return 0
    try:
        return int(verify(parsed_answer(str(gold)), parsed_answer(answer), timeout_seconds=2))
    except Exception:
        return 0
