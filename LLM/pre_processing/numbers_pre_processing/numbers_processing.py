"""
numbers_processing.py

This file centralizes all number-related processing steps by calling
functions already defined in:
  - phrase_to_number.py
  - number_unit_normalization.py
  - unit_conversion.py

We combine them so we can test everything together in one file,
and export a single function `combined_numbers_pipeline(text)` for use elsewhere.
"""

# 1) Import the existing functions
from phrase_to_number import convert_spelled_numbers_phrases
from number_unit_normalization import (
    normalize_numbers_units  # calls convert_spelled_numbers_phrases internally
)
from unit_conversion import normalize_all_units


def combined_numbers_pipeline(text: str) -> str:
    """
    Applies the full pipeline of:
      1) Spell-out number conversion & textual unit normalization
         (via `normalize_numbers_units`, which also calls `convert_spelled_numbers_phrases`)
      2) Final numeric conversion to [cm, cm/s, cm/s^2, rad]
         (via `normalize_all_units`).

    Returns the fully normalized text.
    """
    # Step 1: Convert spelled-out numbers, standardize textual units, infer defaults, replace constants
    #         (Note that `normalize_numbers_units` already calls `convert_spelled_numbers_phrases`.)
    intermediate = normalize_numbers_units(text)

    # Step 2: Convert short-form units to final numeric units (cm, cm/s, cm/s^2, rad)
    final = normalize_all_units(intermediate)

    return final


if __name__ == "__main__":
    # A COMBINED TEST SUITE for angles, distances, speeds, accelerations,
    # spelled-out numbers, decimals, and default-unit inference.

    test_sentences = [
        # Spelled-out + distance
        "Move fifty centimeters forward!!!",
        "Go ahead four hundred eighteen centimeters.",
        "Walk one hundred and twenty meters.",
        "Run two thousand five hundred kilometres!",
        "Move 50 with no unit.",      # default: "50 cm"
        "Go 10.0 with no unit?",      # default: "10.0 cm"

        # Angles + spelled-out
        "Turn right ninety degrees.",
        "Rotate 180 degrees!",
        "Turn 45 with no unit.",      # default: "45 deg"
        "Rotate pi.",                 # => "3.14"
        "Rotate τ.",                  # => "3.14" if typed as 'π' or "tau" => "6.28"

        # Speed
        "Drive 20 with no unit.",     # default: "20 cm/s"
        "Drive at 60 kilometers per hour.",
        "Run at 10 m/s.",
        "Speed is 2.5 m/s??",
        "Drive 100 cm/s or 0.5 km/h??",

        # Acceleration
        "Accelerate 10 with no unit.",   # default: "10 cm/s^2"
        "Accelerate at ten meters per second squared.",
        "Accelerate at 9.81 m/s^2.",
        "Acceleration is 0.05 m/s²",
        "We used 1 cm/s^2 in that experiment.",

        # Mixed
        "Accelerate at 9.81 m/s^2, then rotate 180 deg and move 2.5m.",
        "He turned 45 deg, walked 3 kilometers, then drove 10 kilometers per hour.",
    ]

    for idx, sentence in enumerate(test_sentences, 1):
        processed = combined_numbers_pipeline(sentence)
        print(f"{idx}) Original:   {sentence}")
        print(f"   Processed:  {processed}\n")
