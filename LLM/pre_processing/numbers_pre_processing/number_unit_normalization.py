# number_unit_normalization.py

from phrase_to_number import convert_spelled_numbers_phrases


def normalize_units(text: str) -> str:
    """
    Standardize textual units into short forms:
      "centimeters" -> "cm",
      "meters per second" -> "m/s",
      "kilometers per hour" -> "km/h",
      etc.
    """
    # Normalize acceleration
    text = re.sub(
        r"\b(centimeters per second squared|centimetres per second squared|cm/s\^?2|cms2)\b",
        "cm/s^2", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(meters per second squared|metres per second squared|m/s\^?2)\b",
        "m/s^2", text, flags=re.IGNORECASE
    )

    # Normalize speed
    text = re.sub(
        r"\b(centimeters per second|centimetres per second|cm/s)\b",
        "cm/s", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(meters per second|metres per second|m/s)\b",
        "m/s", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(kilometers per hour|kilometres per hour|km/h)\b",
        "km/h", text, flags=re.IGNORECASE
    )

    # Normalize rotation units
    # First, handle degree symbols attached to digits.
    # Convert "NN°" => "NN deg" (handles multiple digits, decimals)
    text = re.sub(r"(\d+(?:\.\d+)?)°", r"\1 deg", text)
    # Then textual 'degrees?' => 'deg'
    text = re.sub(r"\b(degrees?|degree)\b", "deg", text, flags=re.IGNORECASE)
    # Finally, replace any standalone degree symbols (if any remain).
    text = re.sub(r"°", "deg", text, flags=re.IGNORECASE)

    # Infer minutes and seconds to be 0.5 degrees.
    # This replaces an optional number with one or two quotes with "0.5 degrees".
    text = re.sub(
        r"(?<!\w)(?:\d+(?:\.\d+)?\s*)?(?:'{1,2})(?!\w)",
        "0.5 degrees", text, flags=re.IGNORECASE
    )

    # Normalize distance
    text = re.sub(
        r"\b(centimeters|centimetres|cms|centimeter)\b",
        "cm", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(meters|metres|meter)\b",
        "m", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(kilometers|kilometres|kilometer)\b",
        "km", text, flags=re.IGNORECASE
    )

    return text


import re


def infer_default_units(text: str) -> str:
    """
    A more context-aware approach:
      - If we see "turn" or "rotate", we find the first number after
        those words (optionally skipping "left"/"right" or punctuation),
        and if that number has no angle unit, we append 'deg'.
      - If we see "accelerate" or "acceleration", find the next number and append "cm/s^2".
      - If we see "drive" or "speed", find the next number and append "cm/s".
      - If we see "move", "go", or "strafe", find the next number and append "cm".

    This no longer applies to every number in the text, only the number
    that follows each command word.
    """
    # 1) Rotation: ("turn|rotate") [possible directions] + number => "deg"
    #    E.g. "turn right 45" -> "turn right 45 deg"
    #    We'll skip if we already see deg/rad, etc. after the number.
    pattern_rotate = re.compile(
        r"(?:turn|rotate)\s+(?:left|right\s+)?(\d+(?:\.\d+)?)(?!\s*(?:deg|rad|°|['’]{1,2}))",
        flags=re.IGNORECASE
    )
    text = pattern_rotate.sub(r"\1 deg", text)

    # 2) Acceleration: ("accelerate" or "acceleration") + next number => "cm/s^2"
    pattern_accel = re.compile(
        r"(?:accelerat\w*)\s+(?:at\s+)?(\d+(?:\.\d+)?)(?!\s*(?:cm/s\^?2|m/s\^?2))",
        flags=re.IGNORECASE
    )
    text = pattern_accel.sub(r"\1 cm/s^2", text)

    # 3) Speed: ("drive" or "speed") + next number => "cm/s"
    pattern_speed = re.compile(
        r"(?:\bspeed\b|\bdrive\b)\s+(?:at\s+)?(\d+(?:\.\d+)?)(?!\s*(?:cm/s|m/s|km/h))",
        flags=re.IGNORECASE
    )
    text = pattern_speed.sub(r"\1 cm/s", text)

    # 4) Distance: ("move"|"go"|"strafe") + next number => "cm"
    #    e.g. "move 50" => "move 50 cm"
    pattern_dist = re.compile(
        r"(?:\bmove\b|\bgo\b|\bstrafe\b)\s+(?:ahead\s+)?(\d+(?:\.\d+)?)(?!\s*(?:cm|m|km))",
        flags=re.IGNORECASE
    )
    text = pattern_dist.sub(r"\1 cm", text)

    return text


def normalize_constants(text: str) -> str:
    """
    Replace 'pi'/'π' with '3.14', and 'tau' with '6.28'.
    """
    text = re.sub(r'\bpi\b', '3.14', text, flags=re.IGNORECASE)
    text = text.replace('π', '3.14')
    text = re.sub(r'\btau\b', '6.28', text, flags=re.IGNORECASE)
    return text


def normalize_numbers_units(text: str) -> str:
    """
    Pipeline:
      1) Convert multi-word spelled-out numbers to digits.
      2) Normalize textual unit expressions to short forms (cm, m, deg, etc.).
      3) Infer default units if not present.
      4) Replace math constants (pi, tau).

    This is typically followed by a final numeric conversion step (unit_conversion.py) that ensures
    units like [cm, cm/s, cm/s^2, rad].
    """
    # 1) Convert spelled-out numbers (e.g. "fifty") to digits.
    text = convert_spelled_numbers_phrases(text)

    # 2) Standardize textual units to short forms.
    text = normalize_units(text)

    # 3) Append default units where appropriate.
    text = infer_default_units(text)

    # 4) Replace math constants (pi, τ).
    text = normalize_constants(text)

    return text


if __name__ == "__main__":
    # Test examples
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
        "Rotate pi.",  # → "3.14"
        "Rotate π.",  # → "3.14"
        "Rotate tau.",  # → "6.28"
        "Turn 45° to face north.",  # → "Turn 45 deg to face north."
        "Turn 30' to face east.",  # → "Turn 0.5 degrees to face east."
        "Turn 15'' for precision."  # → "Turn 0.5 degrees for precision."
    ]

    for text in sample_texts:
        normalized = normalize_numbers_units(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}\n")
