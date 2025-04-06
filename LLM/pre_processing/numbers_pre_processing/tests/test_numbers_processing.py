"""
test_numbers_processing.py

Extensive tests for the combined numbers processing pipeline defined in numbers_processing.py.
The pipeline performs:
  1) Early normalization (constants replacement, spelled-out numbers to digits,
     tokenization and merging of multi-word unit phrases, and standardizing spelled-out units).
  2) Unit inference (inserting default units for bare numbers).
  3) (Optionally) Unit conversion (if applied as the final step; here we assume the pipeline
     stops after unit inference).

These tests ensure that:
  - Recognized units (even with trailing punctuation) prevent default unit insertion.
  - Default units are added only when a bare number has no recognized unit immediately following.
  - Merged multi-word unit phrases remain intact.
  - The pipeline handles various edge cases.
"""

import re

from pre_processing.numbers_pre_processing.numbers_processing import combined_numbers_pipeline


def normalize_space(text: str) -> str:
    """Helper to collapse multiple spaces and trim."""
    return re.sub(r"\s+", " ", text).strip()


# Test cases:

def test_simple_sentence():
    text = "Move fifty centimeters forward."
    # Early normalization should convert "fifty" -> "50" and "centimeters" -> "cm"
    expected = "Move 50 cm forward."
    result = combined_numbers_pipeline(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_no_inference_when_unit_exists():
    # When a recognized unit is already present (with punctuation), no default unit is added.
    text = "I moved 2 km and accelerate 9.81 m/s^2, then turned 45 deg!"
    # Expect that the token "9.81" is followed by "m/s^2," (without additional unit insertion),
    # and "2 km" and "45 deg!" remain unchanged.
    expected = "I moved 2 km and accelerate 9.81 m/s^2, then turned 45 deg!"
    result = combined_numbers_pipeline(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_default_inference():
    # When a bare number has no following unit, a default should be inferred.
    text = "move 50 and accelerate 10"
    # "move" defaults to "cm", "accelerate" defaults to "cm/s^2"
    expected = "move 50 cm and accelerate 10 cm/s^2"
    result = combined_numbers_pipeline(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_multiple_commands():
    text = "turn 30 and then move 100"
    # "turn" -> default "deg", "move" -> default "cm"
    expected = "turn 30 deg and then move 100 cm"
    result = combined_numbers_pipeline(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_merged_unit_phrase():
    # This test ensures that a multi-word unit phrase is merged and standardized.
    text = "Accelerate at ten meters per second squared."
    # Early normalization should merge "meters per second squared" into "m/s^2"
    expected = "Accelerate at 10 m/s^2."
    result = combined_numbers_pipeline(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_merged_unit_with_trailing_punctuation():
    text = "Accelerate at fifteen meters per second squared, then stop."
    expected = "Accelerate at 15 m/s^2, then stop."
    result = combined_numbers_pipeline(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_single_word_unit_standardization():
    # Single-word spelled-out units should be standardized.
    text = "Walk one hundred and twenty meters."
    # "meters" should be replaced by "m" based on your SINGLE_WORD_UNIT_MAP.
    expected = "Walk 120 m."
    result = combined_numbers_pipeline(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_mixed_sentence():
    text = "Drive at 60 kilometers per hour."
    # Early normalization should convert "kilometers per hour" to "km/h"
    expected = "Drive at 60 km/h."
    result = combined_numbers_pipeline(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_no_command_context():
    # Bare numbers with no preceding command should remain unchanged.
    text = "The temperature is 42."
    expected = "The temperature is 42."
    result = combined_numbers_pipeline(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def run_all_tests():
    test_simple_sentence()
    test_no_inference_when_unit_exists()
    test_default_inference()
    test_multiple_commands()
    test_merged_unit_phrase()
    test_merged_unit_with_trailing_punctuation()
    test_single_word_unit_standardization()
    test_mixed_sentence()
    test_no_command_context()
    print("All numbers_processing tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
