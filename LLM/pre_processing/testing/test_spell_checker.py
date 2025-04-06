"""
test_spell_checker.py

This file contains tests for the basic_spell_checker function defined in spell_checker.py.
"""
from pre_processing.spell_checker import basic_spell_checker


def test_common_misspellings():
    """
    Test that basic_spell_checker fixes some common English misspellings.
    """
    input_text = "This is an exampel of a sentense with errors."
    expected_output = "This is an example of a sentence with errors."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_no_alphabetic_changes():
    """
    Non-alphabetic tokens (numbers, punctuation) should remain untouched.
    """
    input_text = "The year is 2025, and cost is $100."
    # Spell-check should not alter digits, dollar sign, punctuation, or presumably correct words.
    expected_output = "The year is 2025, and cost is $100."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_partial_misspellings():
    """
    Ensure that partial words or words with numeric parts are left alone
    if they're not purely alphabetic.
    """
    input_text = "User123 posted a mesage about AI2025."
    # 'mesage' is alphabetic, so it should become 'message',
    # but 'User123' and 'AI2025' are alphanumeric and should remain unchanged.
    expected_output = "User123 posted a message about AI2025."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_capitalization_handling():
    """
    The basic_spell_checker lowercases everything by default if it's purely alphabetic,
    because SpellChecker by default doesn't preserve original capitalization.
    We just verify it's at least spelled correctly. For example, "Becaus" => "because".
    """
    input_text = "Becaus It's Rainning"
    # The SpellChecker library typically lowercases the checked word ("Becaus" => "because").
    # "It's" is not purely alphabetic (apostrophe), so it's left alone.
    # "Rainning" => "raining" or "raining"? SpellChecker likely picks "raining".
    expected_output = "because It's raining"
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


def test_unrecognized_words():
    """
    If the word is not recognized at all and SpellChecker has no suggestions,
    it should remain as is.
    """
    input_text = "Xyzztty is a magical incantation."
    # "Xyzzy" might not be in the dictionary, so it should remain "Xyzzy".
    expected_output = "Xyzztty is a magical incantation."
    corrected = basic_spell_checker(input_text)
    assert corrected == expected_output, f"Expected '{expected_output}', got '{corrected}'"


if __name__ == "__main__":
    # Run each test. An AssertionError will be raised if any test fails.
    test_common_misspellings()
    test_no_alphabetic_changes()
    test_partial_misspellings()
    test_capitalization_handling()
    test_unrecognized_words()

    print("All tests passed successfully!")
