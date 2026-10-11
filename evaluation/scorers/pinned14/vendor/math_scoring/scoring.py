"""Dual scorers: historical rule and last complete boxed sensitivity rule."""
import re
from .legacy_scoring import final_answer as legacy_answer, score_answer as legacy_score
from .legacy_scoring import parsed_answer, verify


def complete_boxes(text):
    boxes = []
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        depth = 1
        for end in range(match.end(), len(text)):
            # Escaped literal braces are not TeX grouping braces.
            slashes, pos = 0, end - 1
            while pos >= 0 and text[pos] == '\\':
                slashes, pos = slashes + 1, pos - 1
            if slashes % 2 == 0:
                depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                boxes.append(dict(start=match.start(), end=end + 1, answer=text[match.end():end].strip()))
                break
    return boxes


def robust_answer(text):
    boxes = complete_boxes(text)
    if boxes:
        return boxes[-1]["answer"]
    # Preserve historical non-boxed answer markers. Unclosed boxes remain invalid.
    return legacy_answer(text)


def correct(answer, gold):
    if answer is None:
        return 0
    try:
        return int(verify(parsed_answer(str(gold)), parsed_answer(answer), timeout_seconds=2))
    except Exception:
        return 0


def scores(text, gold):
    boxes = complete_boxes(text)
    old, new = legacy_answer(text), robust_answer(text)
    return dict(legacy_correct=correct(old, gold), robust_correct=correct(new, gold),
                legacy_answer=old, robust_answer=new,
                legacy_parse_success=old is not None, robust_parse_success=new is not None,
                first_complete_boxed_position=boxes[0]["start"] if boxes else None,
                last_complete_boxed_position=boxes[-1]["start"] if boxes else None,
                boxed_position_unit="zero-based character offset in decoded text")
