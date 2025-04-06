"""
test_number_unit_normalization.py

Tests for the early normalization pipeline in number_unit_normalization.py.
These tests cover:
  - Token rejoining (rejoin_tokens)
  - Constant replacement (normalize_constants_early)
  - Spelled-out number conversion (convert_spelled_numbers_phrases)
  - Inserting spaces between numbers and letters (separate_number_and_letter)
  - Tokenization and merging of multi-word unit phrases (tokenize_for_normalization)
  - Full early normalization pipeline (normalize_numbers_units)
"""

from pre_processing.numbers_pre_processing.number_unit_normalization import (
    rejoin_tokens,
    normalize_constants_early,
    convert_spelled_numbers_phrases,
    separate_number_and_letter,
    tokenize_for_normalization,
    normalize_numbers_units,
)


def test_rejoin_tokens():
    tokens = ["Hello", "world", ","]
    expected = "Hello world,"
    assert rejoin_tokens(tokens) == expected, f"Expected '{expected}', got '{rejoin_tokens(tokens)}'"


def test_normalize_constants_early():
    input_text = "Rotate pi and turn τ!"
    expected = "Rotate 3.14 and turn 6.28!"
    output = normalize_constants_early(input_text)
    assert output == expected, f"Expected '{expected}', got '{output}'"


def test_convert_spelled_numbers_phrases():
    input_text = "I have fifty apples and one hundred and twenty-five oranges."
    # Expect "fifty" -> "50" and "one hundred and twenty-five" -> "125"
    expected = "I have 50 apples and 125 oranges."
    output = convert_spelled_numbers_phrases(input_text)
    assert output == expected, f"Expected '{expected}', got '{output}'"


def test_separate_number_and_letter():
    input_text = "2.5m and 123abc"
    expected = "2.5 m and 123 abc"
    output = separate_number_and_letter(input_text)
    assert output == expected, f"Expected '{expected}', got '{output}'"


def test_tokenize_for_normalization():
    # Test merging multi-word unit: "meters per second squared"
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


def test_normalize_numbers_units():
    test_cases = [
        # Spelled-out numbers get converted and constants replaced.
        ("Move fifty centimeters forward.", "Move 50 centimeters forward."),
        ("Go ahead 418 centimeters.", "Go ahead 418 centimeters."),
        ("Walk one hundred and twenty meters.", "Walk 120 meters."),
        ("Run two thousand five hundred kilometres!", "Run 2500 kilometres!"),
        ("Rotate π now!", "Rotate 3.14 now!"),
        ("Turn 30 degrees, then move 1.23m.", "Turn 30 degrees, then move 1.23 m."),
    ]
    for inp, expected in test_cases:
        output = normalize_numbers_units(inp)
        assert output == expected, f"\nInput:    {inp}\nExpected: {expected}\nGot:      {output}"


def run_all_tests():
    test_rejoin_tokens()
    test_normalize_constants_early()
    test_convert_spelled_numbers_phrases()
    test_separate_number_and_letter()
    test_tokenize_for_normalization()
    test_normalize_numbers_units()
    print("All tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
