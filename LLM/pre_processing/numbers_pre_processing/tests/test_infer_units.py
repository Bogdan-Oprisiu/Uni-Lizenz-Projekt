"""
test_infer_units.py

This file contains extensive tests for the infer_default_units() function defined in infer_units.py.
It verifies that:
  - When a bare number is immediately followed by a recognized unit, no default unit is added.
  - Default units are inserted only when no recognized unit follows.
  - The function is case-insensitive.
  - Command context is maintained correctly.
  - Bare numbers without a preceding command remain unchanged.
  - Multiple commands in one token list are handled correctly.
"""

from pre_processing.numbers_pre_processing.infer_units import infer_default_units


def test_simple_inference():
    # Simple case: "turn 30" -> should insert "deg"
    tokens = ["turn", "30"]
    expected = ["turn", "30", "deg"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test simple inference failed: expected {expected}, got {result}"


def test_already_specified_unit():
    # If a recognized unit already follows, do not insert a default unit.
    tokens = ["move", "50", "cm"]
    expected = ["move", "50", "cm"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test already specified unit failed: expected {expected}, got {result}"


def test_number_without_command():
    # Bare number with no preceding command should remain unchanged.
    tokens = ["42"]
    expected = ["42"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test bare number without command failed: expected {expected}, got {result}"


def test_unrecognized_command():
    # If the command word is not in DEFAULT_COMMAND_UNITS, no default is added.
    tokens = ["jump", "100"]
    expected = ["jump", "100"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test unrecognized command failed: expected {expected}, got {result}"


def test_multiple_commands():
    # When multiple commands are present, each bare number should be checked with the latest command.
    tokens = ["move", "50", "forward", "and", "accelerate", "10"]
    # "move" gives default "cm", "accelerate" gives default "cm/s^2"
    expected = ["move", "50", "cm", "forward", "and", "accelerate", "10", "cm/s^2"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test multiple commands failed: expected {expected}, got {result}"


def test_case_insensitivity():
    # Test that the function handles mixed case properly.
    tokens = ["Turn", "45", "DEG"]
    # "Turn" is recognized (case-insensitive) and DEG is already present.
    expected = ["Turn", "45", "DEG"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test case insensitivity failed: expected {expected}, got {result}"


def test_default_inference_with_extra_tokens():
    # In a longer sentence, only numbers with no following recognized unit get a default.
    tokens = ["move", "50", "and", "then", "accelerate", "10", "m/s^2"]
    # For "move 50", default "cm" should be inserted.
    # For "accelerate 10", since the next token "m/s^2" is recognized, no default should be added.
    expected = ["move", "50", "cm", "and", "then", "accelerate", "10", "m/s^2"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test default inference with extra tokens failed: expected {expected}, got {result}"


def test_multiple_numbers_with_same_command():
    # When multiple bare numbers follow the same command, each should get the default.
    tokens = ["move", "50", "and", "60"]
    # Assuming context persists (the last command is "move") both numbers get "cm"
    expected = ["move", "50", "cm", "and", "60", "cm"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test multiple numbers with same command failed: expected {expected}, got {result}"


def run_all_tests():
    test_simple_inference()
    test_already_specified_unit()
    test_number_without_command()
    test_unrecognized_command()
    test_multiple_commands()
    test_case_insensitivity()
    test_default_inference_with_extra_tokens()
    test_multiple_numbers_with_same_command()
    print("All infer_default_units extensive tests passed!")


if __name__ == "__main__":
    run_all_tests()
