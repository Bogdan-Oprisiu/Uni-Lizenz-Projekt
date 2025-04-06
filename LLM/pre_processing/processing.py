"""
processing.py

This module runs all preprocessing steps in a single pipeline, in the following order:
  1. Lowercase normalization and noise removal (normalization_and_noise_removal.py)
  2. Spell checking (spell_checker.py)
  3. Early normalization of numbers and units (number_unit_normalization.py)
  4. Unit inference (infer_units.py)
  5. Unit conversion (unit_conversion.py)

Usage:
    full_text_processing(text: str) -> str
"""

from pre_processing.normalization_and_noise_removal import preprocess_text
from pre_processing.numbers_pre_processing.infer_units import infer_default_units
from pre_processing.numbers_pre_processing.number_unit_normalization import normalize_numbers_units
from pre_processing.numbers_pre_processing.unit_conversion import normalize_all_units
from pre_processing.spell_checker import basic_spell_checker


def full_text_processing(text: str) -> str:
    # Step 1: Normalize text (lowercase, remove noise)
    text = preprocess_text(text)
    # Step 2: Spell checking
    text = basic_spell_checker(text)
    # Step 3: Early normalization (convert spelled-out numbers, merge multi-word unit phrases, standardize units)
    text = normalize_numbers_units(text)
    # Step 4: Unit inference – insert default units for bare numbers.
    tokens = text.split()
    tokens = infer_default_units(tokens)
    text = " ".join(tokens)
    # Step 5: Final unit conversion (convert numeric expressions to final standards)
    text = normalize_all_units(text)
    return text


if __name__ == "__main__":
    test_sentences = [
        "Move Fifty centimeters forward!!!",
        "I moved 2 km and accelerate 9.81 m/s^2, then turned 45 deg!",
        "Walk one hundred and twenty meters, then drive 60 km/h.",
        "Accelerate at ten meters per second squared.",
        "move 50 and accelerate 10",
        "turn 30 and then move 100"
    ]
    for sentence in test_sentences:
        print("Original:", sentence)
        print("Processed:", full_text_processing(sentence))
        print()
