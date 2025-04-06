"""
number_unit_normalization.py

This module performs early normalization of numeric and unit expressions.
It includes:
  1) Constant replacement (e.g., "pi" or "π" → "3.14")
  2) Converting spelled-out numbers into digits (e.g., "fifty" → "50")
  3) Tokenizing the text (with separation between digits and letters)
  4) Merging multi-word unit phrases (e.g., "meters per second squared" → one token)
  5) Standardizing spelled-out units into shorter forms (e.g., "centimeters" → "cm")

These steps produce a standardized text that can be further processed
by later modules (such as unit_conversion.py) that apply numeric conversions.
"""

import re
from typing import List

import unicodedata
from word2number import w2n

###############################################################################
# 0) Single-word map for spelled-out unit standardization
###############################################################################
single_word_map = {
    "centimeters": "cm", "centimetres": "cm", "centimeter": "cm",
    "meters": "m", "metres": "m", "meter": "m",
    "kilometers": "km", "kilometres": "km", "kilometer": "km",
    "degrees": "deg", "degree": "deg",
    "radians": "rad", "radian": "rad",
    "cm": "cm", "m": "m", "km": "km", "deg": "deg", "rad": "rad",
    # Speed
    "cm/s": "cm/s", "m/s": "m/s", "km/h": "km/h",
    "meters per second": "m/s", "metres per second": "m/s",
    "centimeters per second": "cm/s", "centimetres per second": "cm/s",
    "kilometers per hour": "km/h", "kilometres per hour": "km/h",
    # Acceleration
    "m/s^2": "m/s^2", "m/s²": "m/s^2",
    "meters per second squared": "m/s^2", "metres per second squared": "m/s^2",
    "cm/s^2": "cm/s^2", "cm/s²": "cm/s^2",
    "centimeters per second squared": "cm/s^2", "centimetres per second squared": "cm/s^2",
    # Time
    "seconds": "s"
}


###############################################################################
# HELPER FUNCTION: REJOIN TOKENS
###############################################################################
def rejoin_tokens(tokens: List[str]) -> str:
    """
    Rejoin tokens with spaces, then remove extra space before punctuation.
    E.g., ["Hello", "world", ","] -> "Hello world,"
    """
    raw = " ".join(tokens)
    # Remove extra spaces before punctuation
    return re.sub(r"\s+([^\w\s])", r"\1", raw)


###############################################################################
# STEP 1: CONSTANT REPLACEMENT (EARLY)
###############################################################################
def normalize_constants_early(text: str) -> str:
    """
    Replace known constants (like π, tau, e, phi) with their numeric approximations.
    This is done before spelled-out number parsing to avoid conflicts.
    """
    constants_map = {
        "pi": "3.14",
        "π": "3.14",
        "tau": "6.28",
        "τ": "6.28",
        "e": "2.71",
        "phi": "1.62"
    }
    tokens = text.split()
    result_tokens = []
    for tok in tokens:
        # Separate trailing punctuation
        core = tok.rstrip(".,;!?")
        trailing = tok[len(core):]
        norm_core = unicodedata.normalize("NFC", core.lower())
        if norm_core in constants_map:
            result_tokens.append(constants_map[norm_core] + trailing)
        else:
            result_tokens.append(tok)
    return " ".join(result_tokens)


###############################################################################
# STEP 2: CONVERT SPELLED-OUT NUMBERS
###############################################################################
VALID_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "trillion", "and"
}


def is_all_spelled_number_words(token: str) -> bool:
    """
    Return True if all parts of a token (split on '-') are recognized spelled-out number words.
    E.g., "fifty-five" returns True, while "meters" returns False.
    """
    parts = token.lower().split('-')
    return all(part in VALID_NUMBER_WORDS for part in parts)


def convert_spelled_numbers_phrases(text: str) -> str:
    """
    Convert spelled-out English numbers (e.g., "one hundred and twenty-five") into digits.
    Non-numeric tokens are left as-is.

    If a series of tokens are all recognized as spelled-out number words, combine & convert.

    If a token is "and" and the buffer is empty, output it directly (separator).
    """
    tokens = text.split()
    result_tokens = []
    buffer = []

    def flush_buffer():
        if not buffer:
            return
        phrase = " ".join(buffer)
        try:
            number_value = w2n.word_to_num(phrase.lower())
            result_tokens.append(str(number_value))
        except ValueError:
            # If w2n can't parse, revert them
            result_tokens.extend(buffer)
        buffer.clear()

    for tok in tokens:
        if tok.lower() == "and" and not buffer:
            result_tokens.append(tok)
        elif re.match(r'^[a-zA-Z-]+$', tok) and is_all_spelled_number_words(tok):
            buffer.append(tok)
        else:
            flush_buffer()
            result_tokens.append(tok)
    flush_buffer()

    return " ".join(result_tokens)


###############################################################################
# STEP 3: SEPARATE NUMBER AND LETTER IF MISSING
###############################################################################
def separate_number_and_letter(text: str) -> str:
    """
    Insert a space between a digit and a letter if missing. e.g., "2.5m" -> "2.5 m".
    """
    return re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)


###############################################################################
# STEP 4: TOKENIZE & MERGE MULTI-WORD UNITS
###############################################################################
def tokenize_for_normalization(text: str) -> List[str]:
    """
    Tokenize the input text and merges multi-word unit phrases (e.g., "meters per second squared")
    into single tokens. Also transforms slash-based expressions like "m / s" -> "m/s".
    No numeric conversion here; just textual merging.
    """
    # Insert space between digits & letters
    text = separate_number_and_letter(text)
    rough_tokens = text.split()

    merged_tokens = []
    i = 0
    while i < len(rough_tokens):
        token = rough_tokens[i]
        lower = token.lower()

        # (1) Merge "meters per second squared"
        if lower in ["meters", "metres", "meter"] and (i + 3 < len(rough_tokens)):
            maybe_per = rough_tokens[i + 1].lower().strip(".,;!?")
            maybe_second = rough_tokens[i + 2].lower().strip(".,;!?")
            maybe_squared = rough_tokens[i + 3].lower().strip(".,;!?")
            if maybe_per == "per" and maybe_second == "second" and maybe_squared == "squared":
                trailing = ""
                m = re.search(r"([.,;!?]+)$", rough_tokens[i + 3])
                if m:
                    trailing = m.group(1)
                merged_tokens.append("meters per second squared" + trailing)
                i += 4
                continue

        # (2) Merge "kilometers per hour"
        if lower in ["kilometers", "kilometres", "kilometer"] and (i + 2 < len(rough_tokens)):
            maybe_per = rough_tokens[i + 1].lower().strip(".,;!?")
            maybe_hour = rough_tokens[i + 2].lower().strip(".,;!?")
            if maybe_per == "per" and maybe_hour == "hour":
                trailing = ""
                m = re.search(r"([.,;!?]+)$", rough_tokens[i + 2])
                if m:
                    trailing = m.group(1)
                merged_tokens.append("kilometers per hour" + trailing)
                i += 3
                continue

        # (3) Merge "centimeters per second squared"
        if lower in ["centimeters", "centimetres", "centimeter"] and (i + 3 < len(rough_tokens)):
            maybe_per = rough_tokens[i + 1].lower().strip(".,;!?")
            maybe_second = rough_tokens[i + 2].lower().strip(".,;!?")
            maybe_squared = rough_tokens[i + 3].lower().strip(".,;!?")
            if maybe_per == "per" and maybe_second == "second" and maybe_squared == "squared":
                trailing = ""
                m = re.search(r"([.,;!?]+)$", rough_tokens[i + 3])
                if m:
                    trailing = m.group(1)
                merged_tokens.append("centimeters per second squared" + trailing)
                i += 4
                continue

        # (4) "m / s ^ 2" -> "m/s^2"
        if lower == "m" and i + 4 < len(rough_tokens):
            if (rough_tokens[i + 1] == "/" and
                    rough_tokens[i + 2].lower() == "s" and
                    rough_tokens[i + 3] == "^" and
                    rough_tokens[i + 4] in ["2", "²"]):
                merged_tokens.append("m/s^2")
                i += 5
                continue

        # (5) "m / s" -> "m/s"
        if lower == "m" and i + 2 < len(rough_tokens):
            if rough_tokens[i + 1] == "/" and rough_tokens[i + 2].lower() == "s":
                merged_tokens.append("m/s")
                i += 3
                continue

        # (6) "cm / s" -> "cm/s"
        if lower in ["cm", "centimeter", "centimetre"] and (i + 2 < len(rough_tokens)):
            if rough_tokens[i + 1] == "/" and rough_tokens[i + 2].lower() == "s":
                merged_tokens.append("cm/s")
                i += 3
                continue

        # Otherwise, add the token as-is
        merged_tokens.append(token)
        i += 1

    return merged_tokens


###############################################################################
# STEP 5: STANDARDIZE SPELLED-OUT UNITS (NO NUMERIC CONVERSION)
###############################################################################
def standardize_spelled_out_units(tokens: List[str]) -> List[str]:
    """
    Replace any spelled-out or multi-word units from the single_word_map with
    their short form (e.g. "centimeters" -> "cm", "kilometers per hour" -> "km/h")
    if they appear as single tokens at this stage.

    This step does not convert numeric values, only textual units.
    """
    normalized_tokens = []
    for tok in tokens:
        # Strip punctuation for matching; reattach after
        core = tok.rstrip(".,;!?")
        trailing = tok[len(core):]

        # Lowercase & remove punctuation from the core to match in single_word_map
        check = re.sub(r"[.,;!?]+", "", core.lower()).strip()

        if check in single_word_map:
            normalized_tokens.append(single_word_map[check] + trailing)
        else:
            normalized_tokens.append(tok)
    return normalized_tokens


###############################################################################
# STEP 6: FULL EARLY NORMALIZATION PIPELINE
###############################################################################
def normalize_numbers_units(text: str) -> str:
    """
    Complete early normalization pipeline:
      1) Replace constants (π, tau, etc.) with numeric approximations.
      2) Convert spelled-out numbers to digits.
      3) Tokenize the text and merge multi-word unit phrases (like "meters per second squared").
      4) Standardize spelled-out unit tokens (e.g. "centimeters" -> "cm").
      5) Rejoin tokens into a final normalized string.
    """
    # (1) Constants
    text = normalize_constants_early(text)
    # (2) Spelled-out numbers
    text = convert_spelled_numbers_phrases(text)
    # (3) Merge multi-word unit phrases
    tokens = tokenize_for_normalization(text)
    # (4) Standardize spelled-out units (no numeric conversion)
    tokens = standardize_spelled_out_units(tokens)
    # (5) Rejoin
    return rejoin_tokens(tokens)


# Test if run directly
if __name__ == "__main__":
    sample_texts = [
        "Move fifty centimeters forward.",
        "Go ahead 418 centimeters.",
        "Walk one hundred and twenty meters.",
        "Run two thousand five hundred kilometres!",
        "Accelerate at ten meters per second squared.",
        "Rotate π now!",
        "Turn 30 degrees, then move 1.23m.",
        "I have fifty apples and one hundred and twenty-five oranges.",
        "Please move two kilometers per hour"
    ]
    for text in sample_texts:
        normalized = normalize_numbers_units(text)
        print("Original:  ", text)
        print("Normalized:", normalized)
        print()
