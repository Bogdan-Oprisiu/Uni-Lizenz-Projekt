# number_unit_normalization.py

import re
from typing import List
from phrase_to_number import convert_spelled_numbers_phrases

###############################################################################
# Tokenization
###############################################################################

def tokenize_for_normalization(text: str) -> List[str]:
    """
    Tokenize the text on whitespace and punctuation, but also
    merges multi-word units ("meters per second squared") into a single token
    so we can replace them with short forms (m/s^2).

    Example:
        "accelerate at ten meters per second squared" ->
        ["accelerate","at","ten","meters per second squared"]
    """
    # First split on whitespace/punctuation
    # We'll do a simple approach: split on whitespace, keep punctuation attached for now
    rough_tokens = text.split()

    merged_tokens = []
    i = 0
    while i < len(rough_tokens):
        token = rough_tokens[i]

        # Attempt to detect multi-word units by peeking ahead
        lower = token.lower()

        # We'll look for patterns like:
        #  "meters per second squared"
        #  "kilometers per hour"
        #  "feet per minute" etc. (if relevant)
        #  "per second squared" can be 3 tokens: "per","second","squared"

        if lower in ["meters", "metres", "meter"] and (i+3 <= len(rough_tokens)):
            # Look ahead to see if it's "meters per second squared"
            maybe_per = rough_tokens[i+1].lower()
            maybe_second = rough_tokens[i+2].lower()
            maybe_squared = rough_tokens[i+3].lower()
            if maybe_per == "per" and maybe_second == "second" and maybe_squared == "squared":
                # Merge them all into one token
                merged_tokens.append("meters per second squared")
                i += 4
                continue

        if lower in ["kilometers", "kilometres", "kilometer"] and (i+2 <= len(rough_tokens)):
            # Look for "kilometers per hour"
            maybe_per = rough_tokens[i+1].lower()
            if maybe_per == "per" and (i+2 < len(rough_tokens)):
                maybe_hour = rough_tokens[i+2].lower()
                if maybe_hour == "hour":
                    merged_tokens.append("kilometers per hour")
                    i += 3
                    continue

        if lower in ["centimeters", "centimetres", "centimeter"] and (i+3 <= len(rough_tokens)):
            # "centimeters per second squared"
            maybe_per = rough_tokens[i+1].lower()
            maybe_second = rough_tokens[i+2].lower()
            maybe_squared = rough_tokens[i+3].lower()
            if maybe_per == "per" and maybe_second == "second" and maybe_squared == "squared":
                merged_tokens.append("centimeters per second squared")
                i += 4
                continue

        # Otherwise, just keep the single token
        merged_tokens.append(token)
        i += 1

    return merged_tokens


def rejoin_tokens(tokens: List[str]) -> str:
    """
    Rejoin tokens with spaces, then remove spaces before punctuation
    (e.g. ["hello","world",","] -> "hello world,").
    """
    raw = " ".join(tokens)
    # remove space before punctuation
    return re.sub(r"\s+([^\w\s])", r"\1", raw)


###############################################################################
# 2) Unit Normalization & removing minutes/seconds ticks
###############################################################################

def token_based_normalize_units(tokens: List[str]) -> List[str]:
    """
    Convert textual unit expressions (e.g. "meters per second squared") -> "m/s^2"
    on a per-token basis. Also remove leftover single/double quote marks
    after numeric tokens (e.g. "30'" -> "30").
    """
    new_tokens = []
    for token in tokens:
        # 1) If token ends with `'` or `''` and starts with a numeric,
        #    we remove the quotes. E.g. "30'" -> "30", "15''" -> "15".
        #    If you prefer to interpret them as "0.5 deg", do so here.
        m_ticks = re.match(r"^(\d+(?:\.\d+)?)(['’]{1,2})$", token)
        if m_ticks:
            # just remove the quotes
            numeric_part = m_ticks.group(1)
            new_tokens.append(numeric_part)
            continue

        lower = token.lower()

        # 2) Acceleration
        if lower in ["centimeters per second squared", "centimetres per second squared", "cm/s^2", "cms2"]:
            new_tokens.append("cm/s^2")
        elif lower in ["meters per second squared", "metres per second squared", "m/s^2"]:
            new_tokens.append("m/s^2")

        # 3) Speed
        elif lower in ["centimeters per second", "centimetres per second", "cm/s"]:
            new_tokens.append("cm/s")
        elif lower in ["meters per second", "metres per second", "m/s"]:
            new_tokens.append("m/s")
        elif lower in ["kilometers per hour", "kilometres per hour", "km/h"]:
            new_tokens.append("km/h")

        # 4) Rotation
        elif re.match(r"^\d+(?:\.\d+)?°$", token):
            # e.g. "45°" => "45 deg"
            deg_val = re.sub(r"°$", " deg", token)
            new_tokens.append(deg_val)
        elif lower in ["degree", "degrees"]:
            new_tokens.append("deg")
        elif token == "°":
            new_tokens.append("deg")

        # 5) Distance
        elif lower in ["centimeters","centimetres","cms","centimeter"]:
            new_tokens.append("cm")
        elif lower in ["meters","metres","meter"]:
            new_tokens.append("m")
        elif lower in ["kilometers","kilometres","kilometer"]:
            new_tokens.append("km")
        else:
            new_tokens.append(token)

    return new_tokens


###############################################################################
# 3) Infer Default Units (context-aware, token-based)
###############################################################################

def is_bare_number(tok: str) -> bool:
    return bool(re.match(r"^\d+(?:\.\d+)?$", tok))


def infer_default_units(tokens: List[str]) -> List[str]:
    """
    If we see "turn" or "rotate", the next bare number => "deg"
    If we see "accelerate", the next bare number => "cm/s^2"
    If we see "drive" or "speed", the next bare number => "cm/s"
    If we see "move", "go", or "strafe", the next bare number => "cm"
    """
    new_toks = []
    i = 0
    n = len(tokens)

    while i < n:
        token = tokens[i]
        lower = token.lower()

        if lower in ["turn", "rotate"]:
            new_toks.append(token)
            # next token is numeric => append "deg" if not followed by angle unit
            if i+1 < n and is_bare_number(tokens[i+1]):
                numeric_tok = tokens[i+1]
                new_toks.append(numeric_tok)
                # see if i+2 is deg/rad => skip if yes
                if i+2 < n:
                    next_unit = tokens[i+2].lower()
                    if next_unit not in ["deg", "rad"]:
                        new_toks.append("deg")
                else:
                    new_toks.append("deg")
                i += 2
            i += 1

        elif lower.startswith("accelerat"):
            new_toks.append(token)
            if i+1 < n and is_bare_number(tokens[i+1]):
                numeric_tok = tokens[i+1]
                new_toks.append(numeric_tok)
                # see if i+2 is "cm/s^2" or "m/s^2"
                if i+2 < n:
                    maybe_unit = tokens[i+2].lower()
                    if maybe_unit not in ["cm/s^2","m/s^2"]:
                        new_toks.append("cm/s^2")
                else:
                    new_toks.append("cm/s^2")
                i += 2
            i += 1

        elif lower in ["drive","speed"]:
            new_toks.append(token)
            if i+1 < n and is_bare_number(tokens[i+1]):
                numeric_tok = tokens[i+1]
                new_toks.append(numeric_tok)
                if i+2 < n:
                    maybe_unit = tokens[i+2].lower()
                    if maybe_unit not in ["cm/s","m/s","km/h"]:
                        new_toks.append("cm/s")
                else:
                    new_toks.append("cm/s")
                i += 2
            i += 1

        elif lower in ["move","go","strafe"]:
            new_toks.append(token)
            if i+1 < n and tokens[i+1].lower() == "ahead":
                new_toks.append(tokens[i+1])  # keep "ahead"
                i += 1
            # next numeric => "cm"
            if i+1 < n and is_bare_number(tokens[i+1]):
                numeric_tok = tokens[i+1]
                new_toks.append(numeric_tok)
                if i+2 < n:
                    maybe_dist = tokens[i+2].lower()
                    if maybe_dist not in ["cm","m","km"]:
                        new_toks.append("cm")
                else:
                    new_toks.append("cm")
                i += 2
            i += 1

        else:
            new_toks.append(token)
            i += 1

    return new_toks


###############################################################################
# 4) Constants
###############################################################################

def normalize_constants(tokens: List[str]) -> List[str]:
    """
    Replace 'pi'/'π' with '3.14', and 'tau' with '6.28' in a token-based manner.
    """
    new = []
    for t in tokens:
        lower = t.lower()
        if lower == "pi":
            new.append("3.14")
        elif "π" in t:   # if the token has a π we might just replace with 3.14
            new.append(t.replace("π","3.14"))
        elif lower == "tau":
            new.append("6.28")
        else:
            new.append(t)
    return new


###############################################################################
# 5) The main function
###############################################################################

def normalize_numbers_units(text: str) -> str:
    """
    1) Convert spelled-out numbers to digits (via phrase_to_number).
    2) Tokenize & unify multi-word units to single tokens (like "meters per second squared").
    3) Convert leftover units in each token (like "centimeters" -> "cm").
    4) Remove minute/second ticks from numeric tokens ("30'" -> "30").
    5) Infer default units in a context-aware manner
    6) Replace constants (pi, tau)
    7) Rejoin
    """
    # 1) spelled-out => digits
    text = convert_spelled_numbers_phrases(text)

    # 2) Merge multi-word units into single tokens
    tokens = tokenize_for_normalization(text)

    # 3) Convert leftover unit tokens (token-based)
    tokens = token_based_normalize_units(tokens)

    # 4) Infer default units
    tokens = infer_default_units(tokens)

    # 5) Replace constants (pi,tau)
    tokens = normalize_constants(tokens)

    # 6) Rejoin
    final_text = rejoin_tokens(tokens)
    return final_text


###############################################################################
# Quick tests
###############################################################################

if __name__ == "__main__":
    sample_texts = [
        "Move fifty centimeters forward.",
        "Go ahead 418 centimeters.",
        "Turn right ninety degrees.",
        "Walk one hundred and twenty meters.",
        "Run two thousand five hundred kilometres!",
        "Accelerate at ten meters per second squared.",
        "Drive at 60 kilometers per hour.",
        "Rotate 180 degrees.",
        "Halt immediately!",

        "Move 50 with no unit.",  # → "50 cm"
        "Accelerate 10 with no unit.",  # → "10 cm/s^2"
        "Turn 45 with no unit.",  # → "45 deg"
        "Drive 20 with no unit.",  # → "20 cm/s"

        "Rotate pi.",
        "Rotate π.",
        "Rotate tau.",

        "Turn 45° to face north.",
        "Turn 30' to face east.",
        "Turn 15'' for precision."
    ]

    for text in sample_texts:
        out = normalize_numbers_units(text)
        print(f"Original:   {text}")
        print(f"Normalized: {out}\n")
