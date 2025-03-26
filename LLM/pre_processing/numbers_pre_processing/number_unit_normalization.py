# number_unit_normalization.py

import re
from typing import List

import unicodedata
from word2number import w2n


###############################################################################
# HELPER FUNCTION: REJOIN TOKENS
###############################################################################
def rejoin_tokens(tokens: List[str]) -> str:
    """
    Rejoin tokens with spaces, then remove spaces before punctuation.
    E.g. ["Hello","world",","] -> "Hello world,"
    """
    raw = " ".join(tokens)
    return re.sub(r"\s+([^\w\s])", r"\1", raw)


###############################################################################
# STEP 1: CONSTANT REPLACEMENT (EARLY)
###############################################################################
def normalize_constants_early(text: str) -> str:
    """
    First pass: Replace known constants (like π, τ, e, phi)
    with their numeric approximations, *before* spelled-out number parsing.
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
# A set of valid spelled-out number words recognized by w2n
# (You can expand this set if needed.)
VALID_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "trillion", "and"
}


def is_all_spelled_number_words(token: str) -> bool:
    """
    Returns True if *all parts* of a token (split on '-') are recognized
    spelled-out number words, e.g. "fifty-five" => True,
    "move" => False, "meters" => False, "fifty-thousand" => True, etc.
    """
    parts = token.lower().split('-')
    return all(part in VALID_NUMBER_WORDS for part in parts)


def convert_spelled_numbers_phrases(text: str) -> str:
    """
    Convert spelled-out English numbers (e.g., "one hundred and twenty-five")
    into digits. Non-numeric tokens are left as-is. We only add a token
    to the buffer if each sub-part is recognized as a spelled-out number word.
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
        # If token is purely alpha/hyphens and each part is a recognized number word
        if re.match(r'^[a-zA-Z-]+$', tok) and is_all_spelled_number_words(tok):
            # Add to buffer
            buffer.append(tok)
        else:
            # We hit something that isn't a spelled-out number
            # => flush buffer, keep this token
            flush_buffer()
            result_tokens.append(tok)
    # End: flush any leftover
    flush_buffer()

    return " ".join(result_tokens)


###############################################################################
# STEP 3: INSERT SPACE BETWEEN DIGIT AND LETTER IF MISSING
###############################################################################
def separate_number_and_letter(text: str) -> str:
    """Insert a space between a digit and a letter if missing, e.g. '2.5m' -> '2.5 m'."""
    return re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)


###############################################################################
# STEP 4: TOKENIZE & MERGE MULTI-WORD UNITS
###############################################################################
def tokenize_for_normalization(text: str) -> List[str]:
    """
    Splits text on whitespace (keeping punctuation) and merges multi-word units
    like 'meters per second squared' into a single token.
    """
    text = separate_number_and_letter(text)
    rough_tokens = text.split()

    merged_tokens = []
    i = 0
    while i < len(rough_tokens):
        token = rough_tokens[i]
        lower = token.lower()

        # 4a) Merge "meters per second squared"
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

        # 4b) Merge "kilometers per hour"
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

        # 4c) Merge "centimeters per second squared"
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

        # 4d) Merge slash-based tokens, e.g. "m / s ^ 2" => "m/s^2"
        if lower == "m" and i + 4 < len(rough_tokens):
            if (rough_tokens[i + 1] == "/" and
                    rough_tokens[i + 2].lower() == "s" and
                    rough_tokens[i + 3] == "^" and
                    rough_tokens[i + 4] in ["2", "²"]):
                merged_tokens.append("m/s^2")
                i += 5
                continue

        # 4e) Merge "m / s" => "m/s"
        if lower == "m" and i + 2 < len(rough_tokens):
            if rough_tokens[i + 1] == "/" and rough_tokens[i + 2].lower() == "s":
                merged_tokens.append("m/s")
                i += 3
                continue

        # 4f) Merge "cm / s" => "cm/s"
        if lower in ["cm", "centimeter", "centimetre"] and (i + 2 < len(rough_tokens)):
            if rough_tokens[i + 1] == "/" and rough_tokens[i + 2].lower() == "s":
                merged_tokens.append("cm/s")
                i += 3
                continue

        # If none match, keep the token
        merged_tokens.append(token)
        i += 1

    return merged_tokens


###############################################################################
# STEP 5: CONVERT LEFTOVER UNIT TOKENS & REMOVE QUOTES
###############################################################################
def token_based_normalize_units(tokens: List[str]) -> List[str]:
    """
    Convert textual unit expressions to short forms (e.g. 'centimeters' -> 'cm').
    Also remove leftover quotes on numbers (e.g. "30'" -> "30").
    """
    new_tokens = []
    for token in tokens:
        core = token.rstrip(".,;!?")
        trailing = token[len(core):]

        # e.g. "30'" => "30"
        m_ticks = re.match(r"^(\d+(?:\.\d+)?)(['’]{1,2})$", core)
        if m_ticks:
            new_tokens.append(m_ticks.group(1) + trailing)
            continue

        lower = core.lower()

        # Acceleration
        if lower in ["centimeters per second squared", "centimetres per second squared", "cm/s^2", "cms2"]:
            new_tokens.append("m/s^2" + trailing)
        elif lower in ["meters per second squared", "metres per second squared", "m/s^2"]:
            new_tokens.append("m/s^2" + trailing)

        # Speed
        elif lower in ["centimeters per second", "centimetres per second", "cm/s"]:
            new_tokens.append("cm/s" + trailing)
        elif lower in ["meters per second", "metres per second", "m/s"]:
            new_tokens.append("m/s" + trailing)
        elif lower in ["kilometers per hour", "kilometres per hour", "km/h"]:
            new_tokens.append("km/h" + trailing)

        # Angles
        elif re.match(r"^\d+(?:\.\d+)?°$", core):
            new_tokens.append(core.replace("°", " deg") + trailing)
        elif lower in ["degree", "degrees"]:
            new_tokens.append("deg" + trailing)
        elif core == "°":
            new_tokens.append("deg" + trailing)

        # Distances
        elif lower in ["centimeters", "centimetres", "cms", "centimeter"]:
            new_tokens.append("cm" + trailing)
        elif lower in ["meters", "metres", "meter"]:
            new_tokens.append("m" + trailing)
        elif lower in ["kilometers", "kilometres", "kilometer"]:
            new_tokens.append("km" + trailing)
        else:
            new_tokens.append(token)
    return new_tokens


###############################################################################
# STEP 6: INFER DEFAULT UNITS
###############################################################################
def is_bare_number(tok: str) -> bool:
    return bool(re.match(r"^\d+(?:\.\d+)?$", tok))


def infer_default_units(tokens: List[str]) -> List[str]:
    """
    E.g. "move 50" => "move 50 cm", "accelerate 10" => "accelerate 10 m/s^2".
    """
    new_toks = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        lower = token.lower()

        # turn/rotate => deg
        if lower in ["turn", "rotate"]:
            new_toks.append(token)
            if i + 1 < len(tokens) and is_bare_number(tokens[i + 1]):
                new_toks.append(tokens[i + 1])
                next_unit_candidate = ""
                if i + 2 < len(tokens):
                    next_unit_candidate = tokens[i + 2].rstrip(".,;!?").lower()
                if next_unit_candidate not in ["deg", "rad"]:
                    new_toks.append("deg")
                i += 2
            else:
                i += 1

        # accelerate => m/s^2
        elif lower.startswith("accelerat"):
            new_toks.append(token)
            if i + 1 < len(tokens) and is_bare_number(tokens[i + 1]):
                new_toks.append(tokens[i + 1])
                next_unit_candidate = ""
                if i + 2 < len(tokens):
                    next_unit_candidate = tokens[i + 2].rstrip(".,;!?").lower()
                if next_unit_candidate not in ["m/s^2"]:
                    new_toks.append("m/s^2")
                i += 2
            else:
                i += 1

        # drive/speed => cm/s
        elif lower in ["drive", "speed"]:
            new_toks.append(token)
            if i + 1 < len(tokens) and is_bare_number(tokens[i + 1]):
                new_toks.append(tokens[i + 1])
                next_unit_candidate = ""
                if i + 2 < len(tokens):
                    next_unit_candidate = tokens[i + 2].rstrip(".,;!?").lower()
                if next_unit_candidate not in ["cm/s", "m/s", "km/h"]:
                    new_toks.append("cm/s")
                i += 2
            else:
                i += 1

        # move/go/strafe => cm
        elif lower in ["move", "go", "strafe"]:
            new_toks.append(token)
            if i + 1 < len(tokens) and tokens[i + 1].lower() == "ahead":
                new_toks.append(tokens[i + 1])
                i += 1
            if i + 1 < len(tokens) and is_bare_number(tokens[i + 1]):
                new_toks.append(tokens[i + 1])
                next_unit_candidate = ""
                if i + 2 < len(tokens):
                    next_unit_candidate = tokens[i + 2].rstrip(".,;!?").lower()
                if next_unit_candidate not in ["cm", "m", "km"]:
                    new_toks.append("cm")
                i += 2
            else:
                i += 1

        else:
            new_toks.append(token)
            i += 1
    return new_toks


###############################################################################
# STEP 7: COMPLETE PIPELINE
###############################################################################
def normalize_numbers_units(text: str) -> str:
    """
    1) Replace constants first (so "π" isn't lost).
    2) Convert spelled-out numbers to digits (some tokens might become numeric).
    3) Tokenize & unify multi-word units.
    4) Convert leftover unit tokens (e.g. "centimeters" -> "cm").
    5) Infer default units in context (e.g., "move 50" => "move 50 cm").
    6) Rejoin tokens into final string.
    """
    # 1) constants
    text = normalize_constants_early(text)
    # 2) spelled-out => digits
    text = convert_spelled_numbers_phrases(text)
    # 3) tokenize & merge multi-word
    tokens = tokenize_for_normalization(text)
    # 4) convert leftover units
    tokens = token_based_normalize_units(tokens)
    # 5) infer default units
    tokens = infer_default_units(tokens)
    # 6) rejoin
    return rejoin_tokens(tokens)


###############################################################################
# TESTS
###############################################################################
def run_tests():
    test_cases = [
        ("Move fifty centimeters forward.", "Move 50 cm forward."),
        ("Go ahead 418 centimeters.", "Go ahead 418 cm."),
        ("Turn right ninety degrees.", "Turn right 90 deg."),
        ("Walk one hundred and twenty meters.", "Walk 120 m."),
        ("Run two thousand five hundred kilometres!", "Run 2500 km!"),
        ("Accelerate at ten meters per second squared.", "Accelerate at 10 m/s^2."),
        ("Drive at 60 kilometers per hour.", "Drive at 60 km/h."),
        ("Rotate 180 degrees.", "Rotate 180 deg."),
        ("Halt immediately!", "Halt immediately!"),
        # Default
        ("Move 50 with no unit.", "Move 50 cm with no unit."),
        ("Accelerate 10 with no unit.", "Accelerate 10 m/s^2 with no unit."),
        ("Turn 45 with no unit.", "Turn 45 deg with no unit."),
        ("Drive 20 with no unit.", "Drive 20 cm/s with no unit."),
        # Constants
        ("Rotate pi.", "Rotate 3.14."),
        ("Rotate π.", "Rotate 3.14."),
        ("Rotate tau.", "Rotate 6.28."),
        ("Turn 45° to face north.", "Turn 45 deg to face north."),
        ("Turn 30' to face east.", "Turn 30 deg to face east."),
        ("Turn 15'' for precision.", "Turn 15 deg for precision."),
    ]
    for inp, exp in test_cases:
        out = normalize_numbers_units(inp)
        assert out == exp, f"\nInput:    {inp}\nExpected: {exp}\nGot:      {out}"
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()

    # Manual samples
    lines = [
        "Move fifty centimeters forward.",
        "Go ahead 418 centimeters.",
        "Rotate π now!",
        "Compute e times phi.",
    ]
    for line in lines:
        print("\nOriginal:", line)
        print("Normalized:", normalize_numbers_units(line))
