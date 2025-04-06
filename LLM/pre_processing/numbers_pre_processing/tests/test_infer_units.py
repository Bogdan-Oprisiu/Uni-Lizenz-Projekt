"""
test_infer_units.py

This file contains tests for the infer_default_units() function defined in infer_units.py.
It tests several scenarios:
  - Simple command inference (e.g., "turn 30" becomes ["turn", "30", "deg"])
  - Cases where a unit is already specified.
  - Bare numbers with no command context.
  - Multiple commands in a single token list.
  - Unrecognized command words (no default unit inferred).
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


def test_already_specified_unit():
    tokens = ["go", "100", "m"]
    expected = ["go", "100", "m"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test already specified unit failed: expected {expected}, got {result}"


def test_multiple_commands():
    tokens = ["move", "50", "forward", "and", "accelerate", "10"]
    expected = ["move", "50", "cm", "forward", "and", "accelerate", "10", "cm/s^2"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test multiple commands failed: expected {expected}, got {result}"


def test_unit_already_present():
    tokens = ["turn", "45", "deg"]
    expected = ["turn", "45", "deg"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test unit already present failed: expected {expected}, got {result}"


def test_unrecognized_command():
    tokens = ["jump", "100"]
    expected = ["jump", "100"]
    result = infer_default_units(tokens)
    assert result == expected, f"Test unrecognized command failed: expected {expected}, got {result}"


def run_all_tests():
    test_simple_turn()
    test_move_inference()
    test_bare_number_no_command()
    test_already_specified_unit()
    test_multiple_commands()
    test_unit_already_present()
    test_unrecognized_command()
    print("All infer_default_units tests passed!")


if __name__ == "__main__":
    run_all_tests()
