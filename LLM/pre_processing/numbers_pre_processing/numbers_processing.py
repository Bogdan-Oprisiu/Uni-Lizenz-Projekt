"""
numbers_processing.py

This module combines all number-related processing into a single pipeline.
It uses:
  - Early normalization from number_unit_normalization.py, which:
      • Replaces constants (e.g., "pi" → "3.14")
      • Converts spelled-out numbers into digits
      • Tokenizes the text and merges multi-word unit phrases
      • Standardizes spelled-out units into short forms
  - Unit inference from infer_units.py, which inserts default units for bare numbers
    (e.g., if "move 50" is found, it becomes "move 50 cm")
  - Unit conversion from unit_conversion.py, which converts normalized numeric expressions
    to the final standards:
      • Distance: centimeters (cm)
      • Speed: centimeters per second (cm/s)
      • Acceleration: centimeters per second squared (cm/s^2)
      • Angle: radians (rad)

Due to a bug where merged multi-word tokens (like "meters per second squared")
get split when rejoining, this pipeline will first protect such tokens by replacing
internal spaces with underscores, then later restore them if needed.

Usage:
    combined_numbers_pipeline(text: str) -> str
"""

import re

from pre_processing.numbers_pre_processing.infer_units import infer_default_units
from pre_processing.numbers_pre_processing.number_unit_normalization import normalize_numbers_units

# List of known multi-word unit phrases that should remain merged.
# These are in their fully merged, standardized form.
# Adjust as needed.
KNOWN_MERGED_UNITS = {
    "meters per second squared": "m/s^2",
    "kilometers per hour": "km/h",
    "centimeters per second squared": "cm/s^2"
}


def protect_merged_units(text: str) -> str:
    """
    Replace spaces in known multi-word unit phrases with underscores so that
    they are not split when tokenizing by whitespace.
    E.g., "meters per second squared" -> "meters_per_second_squared"
    """
    for phrase, std in KNOWN_MERGED_UNITS.items():
        protected = phrase.replace(" ", "_")
        # Use word boundaries to replace the exact phrase
        text = re.sub(rf"\b{phrase}\b", protected, text, flags=re.IGNORECASE)
    return text


def restore_protected_units(text: str) -> str:
    """
    Optionally, restore underscores to spaces if desired.
    In our pipeline, the unit_conversion module may rely on the short forms (like "m/s^2")
    so we might not need to restore. If you want to restore the original phrase, you could.
    For now, we'll leave them as is.
    """
    # In this example, we assume the unit_conversion module knows that "meters_per_second_squared"
    # should be treated as "m/s^2" (via normalize_unit_names).
    return text


def combined_numbers_pipeline(text: str) -> str:
    """
    Full pipeline for numeric processing:
      1) Early normalization: constants replacement, spelled-out numbers conversion,
         tokenization, merging of multi-word unit phrases, and standardizing spelled-out units.
      2) Protect merged multi-word unit tokens so they are not split.
      3) Unit inference: Insert default units for bare numbers.
      4) Rejoin tokens into a string.
      5) Optionally, restore the protected tokens.
      6) Unit conversion: Convert numeric expressions into final standards.

    Returns the fully normalized text.
    """
    # Step 1: Early normalization.
    early_norm = normalize_numbers_units(text)
    # Step 2: Protect known merged unit phrases.
    protected_text = protect_merged_units(early_norm)
    # Step 3: Tokenize the protected text and perform unit inference.
    tokens = protected_text.split()
    inferred_tokens = infer_default_units(tokens)
    inferred_text = " ".join(inferred_tokens)
    # # Step 4: Final unit conversion.
    # final_text = normalize_all_units(restored_text)

    final_text = inferred_text
    return final_text


if __name__ == "__main__":
    test_sentences = [
        "Move fifty centimeters forward.",
        "I moved 2 km and accelerate 9.81 m/s^2, then turned 45 deg!",
        "Distance: 3 m, angle = 180 deg.",
        "Walk one hundred and twenty meters, then drive 60 km/h.",
        "Run two thousand five hundred kilometres!",
        "Accelerate at ten meters per second squared.",
        "move 50 and accelerate 10",
        "turn 30 and then move 100",
        "Drive at 60 kilometers per hour."  # Should be handled by early normalization.
    ]
    for sentence in test_sentences:
        print("Original:", sentence)
        print("Processed:", combined_numbers_pipeline(sentence))
        print()
