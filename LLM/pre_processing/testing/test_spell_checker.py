"""
test_spell_checker.py

This file contains tests for the basic_spell_checker function defined in spell_checker.py.
It verifies that:
  - Common misspellings are corrected.
  - Non-alphabetic tokens (numbers, punctuation) remain unchanged.
  - Mixed alphanumeric tokens are left untouched.
  - Unit tokens (e.g., "km", "m/s^2") are preserved and not modified.
"""

from pre_processing.spell_checker import basic_spell_checker


def test_common_misspellings():
    input_text = "This is an exampel of a sentense with errors."
    expected_output = "This is an example of a sentence with errors."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_no_alphabetic_changes():
    input_text = "The year is 2025, and cost is $100."
    expected_output = "The year is 2025, and cost is $100."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_partial_misspellings():
    input_text = "User123 posted a mesage about AI2025."
    expected_output = "User123 posted a message about AI2025."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_capitalization_handling():
    input_text = "Becaus It's Rainning"
    # "Becaus" should be corrected to "because" and "Rainning" to "raining"
    expected_output = "because It's raining"
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_unrecognized_words():
    input_text = "Xyzztty is a magical incantation."
    expected_output = "Xyzztty is a magical incantation."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_unit_protection():
    """
    Ensure that unit tokens are preserved without modification.
    For example, "km", "m/s^2" should remain unchanged.
    """
    input_text = "Drive at 60 km and accelerate at 9.81 m/s^2."
    expected_output = "Drive at 60 km and accelerate at 9.81 m/s^2."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_mixed_units_and_text():
    """
    Verify that in a sentence mixing text and unit tokens, the units are not altered.
    """
    input_text = "The distance is 100 km and the speed is 50 m/s."
    expected_output = "The distance is 100 km and the speed is 50 m/s."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def run_all_tests():
    test_common_misspellings()
    test_no_alphabetic_changes()
    test_partial_misspellings()
    test_capitalization_handling()
    test_unrecognized_words()
    test_unit_protection()
    test_mixed_units_and_text()
    print("All spell checker tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
