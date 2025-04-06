"""
test_number_unit_normalization.py

Tests for the early normalization pipeline in number_unit_normalization.py.
These tests cover:
  - Token rejoining (rejoin_tokens)
  - Constant replacement (normalize_constants_early)
  - Spelled-out number conversion (convert_spelled_numbers_phrases)
  - Inserting spaces between numbers and letters (separate_number_and_letter)
  - Tokenization and merging of multi-word unit phrases (tokenize_for_normalization)
  - Standardization of spelled-out units via the single_word_map in this module
  - Full early normalization pipeline (normalize_numbers_units)

Additional tests are included for acceleration units.
"""

import re

from pre_processing.numbers_pre_processing.number_unit_normalization import (
    rejoin_tokens,
    normalize_constants_early,
    convert_spelled_numbers_phrases,
    separate_number_and_letter,
    tokenize_for_normalization,
    normalize_numbers_units,
)


# --- Test rejoin_tokens ---

def test_rejoin_tokens():
    tokens = ["Hello", "world", ","]
    expected = "Hello world,"
    result = rejoin_tokens(tokens)
    assert result == expected, f"Expected '{expected}', got '{result}'"

    tokens = ["This", "is", "a", "test", ":", "yes", "!"]
    expected = "This is a test: yes!"
    result = rejoin_tokens(tokens)
    assert result == expected, f"Expected '{expected}', got '{result}'"


# --- Test normalize_constants_early ---

def test_normalize_constants_early():
    input_text = "Rotate pi, then turn τ!"
    expected = "Rotate 3.14, then turn 6.28!"
    result = normalize_constants_early(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"

    input_text = "phi and e are interesting."
    expected = "1.62 and 2.71 are interesting."
    result = normalize_constants_early(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"


# --- Test convert_spelled_numbers_phrases ---

def test_convert_spelled_numbers_phrases():
    # Simple conversion
    input_text = "fifty"
    expected = "50"
    result = convert_spelled_numbers_phrases(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"

    # Multi-word phrase conversion with "and" as separator
    input_text = "one hundred and twenty-five"
    expected = "125"  # Mathematically, one hundred and twenty-five is 125.
    result = convert_spelled_numbers_phrases(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"

    # Sentence with numbers and conjunctions
    input_text = "I have fifty apples and one hundred and twenty-five oranges."
    expected = "I have 50 apples and 125 oranges."
    result = convert_spelled_numbers_phrases(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"

    # When buffer fails to parse, it should leave original words
    input_text = "This is not a number phrase"
    expected = "This is not a number phrase"
    result = convert_spelled_numbers_phrases(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"


# --- Test separate_number_and_letter ---

def test_separate_number_and_letter():
    input_text = "2.5m and 123abc"
    expected = "2.5 m and 123 abc"
    result = separate_number_and_letter(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"

    input_text = "4 m is fine"
    expected = "4 m is fine"
    result = separate_number_and_letter(input_text)
    assert result == expected, f"Expected '{expected}', got '{result}'"


# --- Test tokenize_for_normalization ---

def test_tokenize_for_normalization():
    # Test merging "meters per second squared"
    input_text = "He ran at 10 meters per second squared, then stopped."
    tokens = tokenize_for_normalization(input_text)
    expected_token = "meters per second squared,"
    assert expected_token in tokens, f"Expected token '{expected_token}' in tokens: {tokens}"

    # Test merging "kilometers per hour"
    input_text2 = "Speed was measured in kilometers per hour."
    tokens2 = tokenize_for_normalization(input_text2)
    expected_token2 = "kilometers per hour."
    assert expected_token2 in tokens2, f"Expected token '{expected_token2}' in tokens: {tokens2}"

    # Test merging slash-based tokens: "m / s ^ 2" -> "m/s^2"
    input_text3 = "The acceleration was m / s ^ 2 during the test."
    tokens3 = tokenize_for_normalization(input_text3)
    expected_token3 = "m/s^2"
    assert expected_token3 in tokens3, f"Expected token '{expected_token3}' in tokens: {tokens3}"

    # Test merging "m / s" -> "m/s"
    input_text4 = "Speed is m / s in this experiment."
    tokens4 = tokenize_for_normalization(input_text4)
    expected_token4 = "m/s"
    assert expected_token4 in tokens4, f"Expected token '{expected_token4}' in tokens: {tokens4}"

    # Test merging "cm / s" -> "cm/s"
    input_text5 = "It measured cm / s exactly."
    tokens5 = tokenize_for_normalization(input_text5)
    expected_token5 = "cm/s"
    assert expected_token5 in tokens5, f"Expected token '{expected_token5}' in tokens: {tokens5}"

    # Test preservation of punctuation when no merging is needed.
    input_text6 = "I ran 5 times."
    tokens6 = tokenize_for_normalization(input_text6)
    assert tokens6[-1] == "times.", f"Expected last token to be 'times.', got {tokens6[-1]}"


# --- Additional Tests: Standardize Spelled-Out Units for Acceleration ---
def test_acceleration_standardization():
    # Test that various ways of writing acceleration are standardized to "m/s^2" form in the tokens.
    # We assume that the early normalization pipeline should standardize acceleration units.

    # Example: "meters per second squared" should merge and then be standardized.
    input_text = "Accelerate at ten meters per second squared."
    # After converting spelled-out numbers, "ten" becomes "10".
    # And the unit should be standardized to the abbreviation in single_word_map.
    expected = "Accelerate at 10 m/s^2."
    output = normalize_numbers_units(input_text)
    # Compare normalized result (removing extra spaces if any)
    result_clean = re.sub(r"\s+", " ", output).strip()
    expected_clean = re.sub(r"\s+", " ", expected).strip()
    assert result_clean == expected_clean, f"\nInput: {input_text}\nExpected: {expected_clean}\nGot: {result_clean}"

    # Another variant: "centimeters per second squared" should be standardized.
    input_text = "The acceleration is  five centimeters per second squared."
    # "five" should become "5" and "centimeters per second squared" should become "cm/s^2"
    expected = "The acceleration is 5 cm/s^2."
    output = normalize_numbers_units(input_text)
    result_clean = re.sub(r"\s+", " ", output).strip()
    expected_clean = re.sub(r"\s+", " ", expected).strip()
    assert result_clean == expected_clean, f"\nInput: {input_text}\nExpected: {expected_clean}\nGot: {result_clean}"

    # Test with trailing punctuation:
    input_text = "Accelerate at fifteen meters per second squared, then stop."
    expected = "Accelerate at 15 m/s^2, then stop."
    output = normalize_numbers_units(input_text)
    result_clean = re.sub(r"\s+", " ", output).strip()
    expected_clean = re.sub(r"\s+", " ", expected).strip()
    assert result_clean == expected_clean, f"\nInput: {input_text}\nExpected: {expected_clean}\nGot: {result_clean}"


# --- Test full early normalization: normalize_numbers_units ---

def test_normalize_numbers_units():
    cases = [
        ("Move fifty centimeters forward.", "Move 50 cm forward."),
        ("Go ahead 418 centimeters.", "Go ahead 418 cm."),  # Note: standardization may change "centimeters" to "cm"
        ("Walk one hundred and twenty meters.", "Walk 120 m."),
        ("Run two thousand five hundred kilometres!", "Run 2500 km!"),
        ("Rotate π now!", "Rotate 3.14 now!"),
        ("Turn 30 degrees, then move 1.23m.", "Turn 30 deg, then move 1.23 m."),
        # Edge case: numbers with punctuation & extra spaces
        ("I have  fifty   apples and   one hundred and twenty-five oranges.",
         "I have 50 apples and 125 oranges."),
    ]
    for inp, expected in cases:
        output = normalize_numbers_units(inp)
        result_clean = re.sub(r"\s+", " ", output).strip()
        expected_clean = re.sub(r"\s+", " ", expected).strip()
        assert result_clean == expected_clean, f"\nInput:    {inp}\nExpected: {expected_clean}\nGot:      {result_clean}"


def run_all_tests():
    test_rejoin_tokens()
    test_normalize_constants_early()
    test_convert_spelled_numbers_phrases()
    test_separate_number_and_letter()
    test_tokenize_for_normalization()
    test_acceleration_standardization()
    test_normalize_numbers_units()
    print("All tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
