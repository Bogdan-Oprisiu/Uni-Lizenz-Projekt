"""
test_processing.py

Extensive tests for the full text processing pipeline defined in processing.py.
The pipeline runs:
  1. Lowercase normalization and noise removal.
  2. Spell checking.
  3. Early normalization of numbers and units (constants replacement, spelled-out numbers to digits,
     tokenization, merging of multi-word unit phrases, and unit standardization).
  4. Unit inference (inserting default units for bare numbers).
  5. Final unit conversion (converting numeric expressions to final standards:
         Distance → cm, Speed → cm/s, Acceleration → cm/s^2, Angle → rad).

These tests ensure that:
  - Spelled-out numbers and units are converted and standardized.
  - Recognized units are preserved and final conversion is applied.
  - Default units are inserted only when needed.
  - The final numeric values are formatted as expected.
"""

import re

from pre_processing.numbers_pre_processing.tests.test_numbers_processing import test_no_inference_when_unit_exists
from pre_processing.processing import full_text_processing


def normalize_space(text: str) -> str:
    """Collapse multiple spaces and trim."""
    return re.sub(r"\s+", " ", text).strip()


def test_simple_sentence():
    text = "Move Fifty centimeters forward!!!"
    # "Fifty" → 50; "centimeters" → "cm" then final conversion: 50 becomes 50.00 cm.
    expected = "move 50.00 cm forward."
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_units_already_present():
    text = "I moved 2 km and accelerate 9.81 m/s^2, then turned 45 deg!"
    # "2 km" remains unchanged (final conversion doesn't alter km here since it's not converted),
    # "9.81 m/s^2" converts to 981.00 cm/s^2,
    # "45 deg" converts to 45*(pi/180)=0.7854 rad.
    expected = "i moved 2 km and accelerate 981.00 cm/s^2, then turned 0.7854 rad!"
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_default_inference():
    text = "move 50 and accelerate 10"
    # "move" → default unit "cm": "50" becomes "50.00 cm"
    # "accelerate" → default unit "cm/s^2": "10" becomes "10.00 cm/s^2"
    expected = "move 50.00 cm and accelerate 10.00 cm/s^2"
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_multiple_commands():
    text = "turn 30 and then move 100"
    # "turn" → default "deg": "30" becomes 30 deg → 30*(pi/180)=0.5236 rad,
    # "move" → default "cm": "100" becomes "100.00 cm"
    expected = "turn 0.5236 rad and then move 100.00 cm"
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_merged_unit_phrase():
    text = "Accelerate at ten meters per second squared."
    # "ten" becomes 10, and "meters per second squared" should be merged to "m/s^2" then converted:
    # 10 m/s^2 becomes 10*100 = 1000.00 cm/s^2.
    expected = "accelerate at 1000.00 cm/s^2."
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_merged_unit_with_punctuation():
    text = "Accelerate at fifteen meters per second squared, then stop."
    # "fifteen" becomes 15, merged unit becomes "m/s^2", conversion: 15*100=1500.00 cm/s^2,
    expected = "accelerate at 1500.00 cm/s^2, then stop."
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_single_word_unit_standardization():
    text = "Walk one hundred and twenty meters."
    # "one hundred and twenty" becomes "120" and "meters" → "m", then conversion: 120*100=12000.00 cm.
    expected = "walk 12000.00 cm."
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_mixed_sentence():
    text = "Drive at 60 kilometers per hour."
    # "kilometers per hour" becomes "km/h", then conversion: 60 km/h -> 60*100000/3600=1666.67 cm/s.
    expected = "drive at 1666.67 cm/s."
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_no_command_context():
    text = "The temperature is 42."
    # With no command context, bare numbers remain unchanged and no default unit is inferred.
    expected = "the temperature is 42."
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def test_complex_sentence():
    text = ("Turn 30 and then move 100. "
            "After that, accelerate 9.81 m/s^2, but if you jump 50, "
            "nothing should be inferred.")
    # "Turn 30" → "30" gets default "deg" → 30 deg becomes 0.5236 rad,
    # "move 100" → default "cm" becomes 100.00 cm,
    # "accelerate 9.81 m/s^2" is recognized and converted to 981.00 cm/s^2,
    # "jump 50" remains unchanged.
    expected = ("turn 0.5236 rad and then move 100. "
                "after that, accelerate 981.00 cm/s^2, but if you jump 50, "
                "nothing should be inferred.")
    result = full_text_processing(text)
    assert normalize_space(result) == expected, f"Expected: {expected}\nGot: {result}"


def run_all_tests():
    test_simple_sentence()
    test_no_inference_when_unit_exists()
    test_default_inference()
    test_multiple_commands()
    test_merged_unit_phrase()
    test_merged_unit_with_punctuation()
    test_single_word_unit_standardization()
    test_mixed_sentence()
    test_no_command_context()
    test_complex_sentence()
    print("All processing tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
