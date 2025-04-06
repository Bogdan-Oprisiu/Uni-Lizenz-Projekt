"""
number_unit_normalization.py

This module performs early normalization of numeric and unit expressions.
It includes:
  - Constant replacement (e.g., "pi" or "π" → "3.14")
  - Converting spelled-out numbers into digits (e.g., "fifty" → "50")
  - Tokenizing the text (with separation between digits and letters)
  - Merging multi-word unit phrases (e.g., "meters per second squared" → a single token)

These early steps produce a standardized text that can be further processed
by later unit conversion modules.
"""

import re
from typing import List

import unicodedata
from word2number import w2n


###############################################################################
# HELPER FUNCTION: REJOIN TOKENS
###############################################################################
def rejoin_tokens(tokens: List[str]) -> str:
    """
    Rejoin tokens with spaces, then remove extra space before punctuation.
    E.g., ["Hello", "world", ","] -> "Hello world,"
    """
    raw = " ".join(tokens)
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
        # Separate trailing punctuation for preservation.
        core = tok.rstrip(".,;!?")
        trailing = tok[len(core):]
        # Normalize the token (using Unicode normalization and lowercasing).
        norm_core = unicodedata.normalize("NFC", core.lower())
        if norm_core in constants_map:
            result_tokens.append(constants_map[norm_core] + trailing)
        else:
            result_tokens.append(tok)
    return " ".join(result_tokens)


###############################################################################
# STEP 2: CONVERT SPELLED-OUT NUMBERS
###############################################################################
# Define a set of valid number words (can be extended if needed)
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
    Convert spelled-out English numbers (e.g., "one hundred and twenty-five")
    into digits. Non-numeric tokens are left as-is.

    If a series of tokens are all recognized as number words, they are combined,
    converted, and replaced with the digit representation.

    Modification: If a token is "and" and the buffer is empty,
    it is output immediately (so that "and" used as a separator in a sentence is preserved).
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
            # "and" that is not part of a number phrase should be output directly.
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
    Insert a space between a digit and a letter if missing.
    E.g., "2.5m" becomes "2.5 m".
    """
    return re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)


###############################################################################
# STEP 4: TOKENIZE & MERGE MULTI-WORD UNITS
###############################################################################
def tokenize_for_normalization(text: str) -> List[str]:
    """
    Tokenize the input text and merge multi-word unit phrases into single tokens.
    This includes handling phrases like:
      - "meters per second squared" → merged into one token.
      - "kilometers per hour" → merged into one token.
    """
    # First, ensure there is space between digits and letters.
    text = separate_number_and_letter(text)
    rough_tokens = text.split()

    merged_tokens = []
    i = 0
    while i < len(rough_tokens):
        token = rough_tokens[i]
        lower = token.lower()

        # Example 1: Merge "meters per second squared"
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

        # Example 2: Merge "kilometers per hour"
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

        # Example 3: Merge "centimeters per second squared"
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

        # Example 4: Merge slash-based tokens (e.g., "m / s ^ 2" -> "m/s^2")
        if lower == "m" and i + 4 < len(rough_tokens):
            if (rough_tokens[i + 1] == "/" and
                    rough_tokens[i + 2].lower() == "s" and
                    rough_tokens[i + 3] == "^" and
                    rough_tokens[i + 4] in ["2", "²"]):
                merged_tokens.append("m/s^2")
                i += 5
                continue

        # Example 5: Merge "m / s" -> "m/s"
        if lower == "m" and i + 2 < len(rough_tokens):
            if rough_tokens[i + 1] == "/" and rough_tokens[i + 2].lower() == "s":
                merged_tokens.append("m/s")
                i += 3
                continue

        # Example 6: Merge "cm / s" -> "cm/s"
        if lower in ["cm", "centimeter", "centimetre"] and (i + 2 < len(rough_tokens)):
            if rough_tokens[i + 1] == "/" and rough_tokens[i + 2].lower() == "s":
                merged_tokens.append("cm/s")
                i += 3
                continue

        # If no merge is applicable, add the token as-is.
        merged_tokens.append(token)
        i += 1

    return merged_tokens


###############################################################################
# STEP 5: FULL EARLY NORMALIZATION PIPELINE
###############################################################################
def normalize_numbers_units(text: str) -> str:
    """
    Complete early normalization pipeline:
      1) Replace constants (π, tau, etc.) with numeric approximations.
      2) Convert spelled-out numbers to digits.
      3) Tokenize the text and merge multi-word unit phrases.
      4) Rejoin tokens into a final normalized string.
    """
    text = normalize_constants_early(text)
    text = convert_spelled_numbers_phrases(text)
    tokens = tokenize_for_normalization(text)
    return rejoin_tokens(tokens)


# For testing purposes, run this module directly.
if __name__ == "__main__":
    sample_texts = [
        "Move fifty centimeters forward.",
        "Go ahead 418 centimeters.",
        "Walk one hundred and twenty meters.",
        "Run two thousand five hundred kilometres!",
        "Accelerate at ten meters per second squared.",
        "Rotate π now!",
        "Turn 30 degrees, then move 1.23m.",
        "I have fifty apples and one hundred and twenty-five oranges."
    ]
    for text in sample_texts:
        normalized = normalize_numbers_units(text)
        print("Original:  ", text)
        print("Normalized:", normalized)
        print()
