"""
test_infer_units.py

This file contains extensive tests for the infer_default_units() function defined in infer_units.py.
It tests several scenarios:
  - Simple command inference (e.g., "turn 30" becomes ["turn", "30", "deg"])
  - Cases where a recognized unit is already present, including with trailing punctuation.
  - Bare numbers with no preceding command remain unchanged.
  - Multiple commands in one token list are handled correctly.
  - Case-insensitivity of command and unit tokens.
  - Multiple numbers following the same command.
  - No inference for unrecognized command words.
"""

from pre_processing.numbers_pre_processing.infer_units import infer_default_units


def test_simple_turn():
    tokens = ["turn", "30"]
    expected = ["turn", "30", "deg"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test simple turn failed: expected {expected}, got {result}"


def test_move_inference():
    tokens = ["move", "50"]
    expected = ["move", "50", "cm"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test move inference failed: expected {expected}, got {result}"


def test_bare_number_no_command():
    tokens = ["42"]
    expected = ["42"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test bare number no command failed: expected {expected}, got {result}"


def test_unit_already_present():
    tokens = ["go", "100", "m"]
    expected = ["go", "100", "m"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test unit already present failed: expected {expected}, got {result}"


def test_multiple_commands():
    tokens = ["move", "50", "forward", "and", "accelerate", "10"]
    expected = ["move", "50", "cm", "forward", "and", "accelerate", "10", "cm/s^2"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test multiple commands failed: expected {expected}, got {result}"


def test_recognized_unit_with_punctuation():
    # The unit "m/s^2," should be recognized as "m/s^2" so no default is inserted.
    tokens = ["accelerate", "9.81", "m/s^2,"]
    expected = ["accelerate", "9.81", "m/s^2,"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test recognized unit with punctuation failed: expected {expected}, got {result}"


def test_case_insensitivity():
    tokens = ["Turn", "45", "Deg"]
    expected = ["Turn", "45", "Deg"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test case insensitivity failed: expected {expected}, got {result}"


def test_multiple_numbers_same_command():
    tokens = ["move", "50", "and", "60"]
    expected = ["move", "50", "cm", "and", "60", "cm"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test multiple numbers same command failed: expected {expected}, got {result}"


def test_no_inference_for_non_command():
    tokens = ["jump", "100"]
    expected = ["jump", "100"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test no inference for non-command failed: expected {expected}, got {result}"


def run_all_tests():
    test_simple_turn()
    test_move_inference()
    test_bare_number_no_command()
    test_unit_already_present()
    test_multiple_commands()
    test_recognized_unit_with_punctuation()
    test_case_insensitivity()
    test_multiple_numbers_same_command()
    test_no_inference_for_non_command()
    print("All infer_default_units tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
