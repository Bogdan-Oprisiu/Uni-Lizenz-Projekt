"""
number_unit_normalization.py

This module provides functions for normalizing text by converting number words
to digits and standardizing unit representations.

Functions:
- normalize_number_words(text): Converts written numbers (e.g., "fifty") into digits.
- normalize_units(text): Standardizes unit expressions (e.g., "centimeters" → "cm"),
  including units for distance, speed, acceleration, and rotation.
- infer_default_units(text): Appends default units (cm, cm/s, cm/s^2, deg) when none are provided,
  based on context keywords.
- normalize_numbers_units(text): Applies the full normalization pipeline.
"""

import re
from word2number import w2n


def normalize_number_words(text: str) -> str:
    """
    Convert number words in the text to numeric digits.

    This function splits the text into words and attempts to convert each word using the w2n library.
    If the conversion fails (ValueError), the original word is kept.
    """
    words = text.split()
    new_words = []
    for word in words:
        # Remove surrounding punctuation (we'll add them back later if needed)
        stripped = word.strip(".,:;!?")
        try:
            # Attempt to convert the stripped word to a number.
            num = w2n.word_to_num(stripped.lower())
            # Replace the original token with the numeric string.
            new_word = word.replace(stripped, str(num))
            new_words.append(new_word)
        except ValueError:
            new_words.append(word)
    return " ".join(new_words)


def normalize_units(text: str) -> str:
    """
    Standardize common unit representations in the text.

    The function normalizes units for:
      - Acceleration (e.g., meters per second squared → m/s^2)
      - Speed (e.g., meters per second → m/s, kilometers per hour → km/h)
      - Rotation (e.g., degrees → deg, radians → rad)
      - Distance (e.g., centimeters → cm, meters → m, kilometers → km)
    """
    # Normalize acceleration units (handle both "cm/s^2" and "m/s^2")
    text = re.sub(
        r"\b(centimeters per second squared|centimetres per second squared|cm/s\^?2|cms2)\b",
        "cm/s^2", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(meters per second squared|metres per second squared|m/s\^?2)\b",
        "m/s^2", text, flags=re.IGNORECASE)

    # Normalize speed units
    text = re.sub(
        r"\b(centimeters per second|centimetres per second|cm/s)\b",
        "cm/s", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(meters per second|metres per second|m/s)\b",
        "m/s", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(kilometers per hour|kilometres per hour|km/h)\b",
        "km/h", text, flags=re.IGNORECASE)

    # Normalize rotation units
    text = re.sub(r"\b(degrees|degree)\b", "deg", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(radians|radian)\b", "rad", text, flags=re.IGNORECASE)

    # Normalize distance units
    text = re.sub(r"\b(centimeters|centimetres|cms|centimeter)\b", "cm", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(meters|metres|meter)\b", "m", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(kilometers|kilometres|kilometer)\b", "km", text, flags=re.IGNORECASE)

    return text


def infer_default_units(text: str) -> str:
    """
    Infer default units for numeric values that are not explicitly provided,
    based on contextual keywords in the text.

    Defaults:
      - If the text indicates a rotation command ("rotate" or "turn"), append "deg".
      - If the text indicates an acceleration command ("accelerate" or "acceleration"), append "cm/s^2".
      - If the text indicates a speed command ("drive" or "speed"), append "cm/s".
      - If the text indicates a distance command ("move", "go", or "strafe"), append "cm".

    Note: This is a heuristic implementation and may need further tuning for edge cases.
    """
    lower_text = text.lower()

    # For rotation commands: ensure numbers are followed by a unit (deg or rad)
    if "rotate" in lower_text or "turn" in lower_text:
        # Look for standalone numbers and append " deg" if not already followed by "deg" or "rad"
        text = re.sub(r"(\d+)(?!\s*(deg|rad))", r"\1 deg", text)

    # For acceleration commands: append " cm/s^2" to numbers not followed by it.
    elif "accelerat" in lower_text:
        text = re.sub(r"(\d+)(?!\s*(cm/s\^?2|m/s\^?2))", r"\1 cm/s^2", text)

    # For speed commands: append " cm/s" to numbers not followed by recognized speed units.
    elif "drive" in lower_text or "speed" in lower_text:
        text = re.sub(r"(\d+)(?!\s*(cm/s|m/s|km/h))", r"\1 cm/s", text)

    # For distance commands: append " cm" to numbers not followed by a distance unit.
    elif "move" in lower_text or "go" in lower_text or "strafe" in lower_text:
        text = re.sub(r"(\d+)(?!\s*(cm|m|km))", r"\1 cm", text)

    return text


def normalize_numbers_units(text: str) -> str:
    """
    Apply the full normalization pipeline:
      1. Convert written numbers to digits.
      2. Standardize unit representations.
      3. Infer and append default units for numbers without explicit units.

    Parameters:
      text (str): The input text.

    Returns:
      str: The normalized text.
    """
    text = normalize_number_words(text)
    text = normalize_units(text)
    text = infer_default_units(text)
    return text


# For testing purposes, run this module directly.
if __name__ == "__main__":
    sample_texts = [
        "Move fifty centimeters forward.",
        "Go ahead 418 centimeters.",
        "Turn right 90 degrees.",
        "Walk one hundred and twenty meters.",
        "Run 2 kilometres.",
        "Accelerate at ten meters per second squared.",
        "Drive at 60 kilometers per hour.",
        "Rotate 180 degrees.",
        "Halt immediately!",
        "Move 50 with no unit.",              # Should infer distance default → "50 cm"
        "Accelerate 10 with no unit.",         # Should infer acceleration default → "10 cm/s^2"
        "Turn 45 with no unit.",               # Should infer rotation default → "45 deg"
        "Drive 20 with no unit."               # Should infer speed default → "20 cm/s"
    ]

    for text in sample_texts:
        normalized = normalize_numbers_units(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}\n")
