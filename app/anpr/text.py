"""Turning OCR output into a plate number, or rejecting it.

OCR on a small, angled, poorly-lit plate produces near misses far more often
than it produces nothing. Most of them are the same handful of shape
confusions - O for 0, I for 1, S for 5 - and they are correctable because an
Indian plate has a known shape:

    MH 12 AB 1234       state, district, series, number

Position tells you which character class to expect, so an `O` where a digit
belongs is a 0. A read that still does not fit the format is discarded rather
than reported: a wrong plate number in an evidence log is worse than none.
"""

from __future__ import annotations

import re

# Two letters, one or two digits, up to three letters, four digits.
# Covers MH12AB1234, DL8CAF5030 and the older MH12A1234.
PLATE_PATTERN = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$")

# Bharat series plates: 22BH1234AA
BH_PATTERN = re.compile(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$")

VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB",
}

# Measured, not guessed. 400 annotated Indian plates were read and every
# mistake counted (tools/plate_benchmark.py); these are the substitutions that
# actually occurred, most frequent first: O->0 (22), A->4 (7), J->1, U->0.
#
# Only confusions that cross the letter/digit line are here, because those are
# the ones position can settle. A 4 read as a 1 is two digits, and no amount of
# knowing "a digit belongs here" tells them apart - that needs a better
# recogniser, and it is the larger half of the remaining error.
_TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "U": "0",
                           "I": "1", "L": "1", "J": "1",
                           "Z": "2", "A": "4", "S": "5", "G": "6",
                           "T": "7", "B": "8"})
_TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "4": "A",
                            "5": "S", "6": "G", "7": "T", "8": "B"})


def clean(raw: str) -> str:
    """Strip everything OCR added that a plate cannot contain."""
    return re.sub(r"[^A-Z0-9]", "", (raw or "").upper())


def normalise(raw: str) -> str | None:
    """Coerce an OCR read into a valid plate number, or None if it cannot be.

    Returning None is the important half: an evidence log with a confidently
    wrong number is worse than one that admits it could not read the plate.
    """
    text = clean(raw)
    if not 8 <= len(text) <= 11:
        return None

    for candidate in (text, _repair_indian(text), _repair_bh(text)):
        if not candidate:
            continue
        if _is_valid(candidate) and _close_enough(text, candidate):
            return candidate
        # The body parsed but the state code did not; try to rescue it.
        match = PLATE_PATTERN.match(candidate)
        if match:
            fixed = _repair_state(match.group(1))
            if fixed:
                rescued = fixed + candidate[2:]
                if _close_enough(text, rescued):
                    return rescued
    return None


# How many characters repair may change before it stops being a repair.
#
# Measured: extending the confusion tables raised exact reads from 30.5% to
# 35.2% of 400 annotated plates, but also let more *wrong* numbers through the
# format check - and a wrong number is not a harmless miss. It reaches the
# vehicle registry and flags an innocent driver.
#
# So repair is allowed to fix a few shape confusions, not to rewrite a read
# until something fits. A read needing five corrections was never close to this
# plate; accepting it is inventing a number, not recovering one.
MAX_REPAIRS = 3


def _close_enough(read: str, repaired: str) -> bool:
    """Did repair correct a read, or manufacture a different number?"""
    if len(read) != len(repaired):
        # A length change means characters were inserted or dropped, which the
        # position-based repair cannot justify. Only the shortest such change
        # is credible.
        return abs(len(read) - len(repaired)) <= 1
    changed = sum(1 for a, b in zip(read, repaired) if a != b)
    return changed <= MAX_REPAIRS


# Letters that OCR routinely swaps for one another on a plate.
# Used only to repair a state code, where a single unambiguous substitution is
# safe because the set of valid codes is small and known. M/H/K/A are here
# because they are what the measurement found: MH read as HH, AH or KH was the
# single most common state error in 400 plates, and Maharashtra is the most
# represented state on Indian roads.
_LOOKALIKE = {
    "O": "DQ", "D": "O", "Q": "O", "I": "TL", "T": "I", "L": "I",
    "S": "5", "B": "8", "G": "C", "C": "G", "U": "V", "V": "U",
    "M": "NHKA", "N": "M", "R": "P", "P": "R", "K": "XM", "X": "K",
    "H": "MA", "A": "HM",
}


def _repair_state(code: str) -> str | None:
    """Fix a state code OCR got slightly wrong - but only if it is unambiguous.

    OL is not a state; DL is. One substitution away is a safe correction. Two
    candidates being equally plausible is not, so that read is thrown away.
    """
    if code in VALID_STATE_CODES:
        return code
    matches = set()
    for i, char in enumerate(code):
        for replacement in _LOOKALIKE.get(char, ""):
            if not replacement.isalpha():
                continue
            candidate = code[:i] + replacement + code[i + 1:]
            if candidate in VALID_STATE_CODES:
                matches.add(candidate)
    return matches.pop() if len(matches) == 1 else None


def _is_valid(text: str) -> bool:
    match = PLATE_PATTERN.match(text)
    if match:
        return match.group(1) in VALID_STATE_CODES
    return bool(BH_PATTERN.match(text))


def _repair_indian(text: str) -> str | None:
    """Fix per-position character-class confusions for a standard plate."""
    if len(text) < 9:
        return None
    # Last four are always digits; first two are always letters.
    state = text[:2].translate(_TO_LETTER)
    number = text[-4:].translate(_TO_DIGIT)
    middle = text[2:-4]
    if not middle:
        return None

    # The middle is digits then letters: district code, then series.
    digits, letters = "", ""
    for char in middle:
        if not letters and char.translate(_TO_DIGIT).isdigit() and len(digits) < 2:
            digits += char.translate(_TO_DIGIT)
        else:
            letters += char.translate(_TO_LETTER)
    if not digits:
        return None
    return f"{state}{digits}{letters}{number}"


def _repair_bh(text: str) -> str | None:
    if len(text) < 9:
        return None
    year = text[:2].translate(_TO_DIGIT)
    series = text[2:4].translate(_TO_LETTER)
    number = text[4:8].translate(_TO_DIGIT)
    suffix = text[8:].translate(_TO_LETTER)
    return f"{year}{series}{number}{suffix}" if series == "BH" else None


def state_of(plate: str) -> str | None:
    """The issuing state code, when the plate carries one."""
    match = PLATE_PATTERN.match(plate or "")
    if match and match.group(1) in VALID_STATE_CODES:
        return match.group(1)
    return None


def format_display(plate: str) -> str:
    """Group a plate the way it is printed: MH 12 AB 1234."""
    match = PLATE_PATTERN.match(plate or "")
    if not match:
        return plate
    return " ".join(part for part in match.groups() if part)
