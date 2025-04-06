"""
test_normalization_and_noise_removal.py

This file contains tests for the text preprocessing pipeline defined in your main file.
We cover both individual function tests and end-to-end pipeline tests,
matching the new simplified code that always preserves math symbols.
"""

import sys
import os
import re

from pre_processing.normalization_and_noise_removal import (
    lowercase_normalization,
    noise_removal,
    preprocess_text
)


def test_lowercase_normalization():
    """
    Ensure lowercase_normalization converts all characters to lowercase.
    """
    input_text = "This Is A TEST"
    expected_output = "this is a test"
    output = lowercase_normalization(input_text)
    assert output == expected_output, f"Expected '{expected_output}', got '{output}'"


def test_noise_removal_exclamations_questions():
    """
    Multiple exclamation or question marks should convert to a single period.
    """
    input_text = "Hey!!! Are you sure??"
    # '!!!' -> '.', '??' -> '.'
    # Result => "Hey. Are you sure."
    expected_output = "hey. are you sure."
    # We'll handle case in separate step, so let's just
    # test noise_removal alone (no lowercase here).
    # Actually, your pipeline does lowercase first, but let's
    # just confirm the punctuation handling:
    # We'll feed it with "Hey!!! Are you sure??" and see if the
    # output is "Hey. Are you sure."
    output = noise_removal(input_text)
    assert output == "Hey. Are you sure.", f"Expected 'Hey. Are you sure.', got '{output}'"

def test_noise_removal_punctuation_collapse():
    """
    Multiple consecutive periods or commas should collapse to a single one.
    """
    input_text = "Wow.... So many,,, commas!!!"
    # "!!!" -> "."
    # multiple '.' -> single '.'
    # multiple ',' -> single ','
    # => "Wow. So many, commas."
    # Not testing lowercase, only punctuation.
    expected_output = "Wow. So many, commas."
    output = noise_removal(input_text)
    assert output == expected_output, f"Expected '{expected_output}', got '{output}'"

def test_noise_removal_vertical_bar_slash():
    """
    Vertical bars and backslashes should convert to forward slashes.
    """
    input_text = "What|about\\this???"
    # '???' -> '.'
    # '|' and '\' -> '/'
    # => "What/about/this."
    expected_output = "What/about/this."
    output = noise_removal(input_text)
    assert output == expected_output, f"Expected '{expected_output}', got '{output}'"

def test_noise_removal_unallowed_chars():
    """
    Characters not in the allowed set should be removed (e.g., '@#$^&=+~')
    but we now preserve + - * = ^ % ( ) no matter what.
    So let's confirm that random symbols like '@#' vanish,
    but things like '+' remain.
    """
    input_text = "Hello @#$^&*( )=+~ End"
    # Explanation:
    #   - '@#$', '~' => removed
    #   - '^', '&', '*', '(', ')' => kept (since ^, *, (, ) are in our new whitelist)
    #   - '=' also kept
    #   - '+' is preserved
    #
    # So let's see what remains:
    # "Hello ^&*( )=+ End"
    # But actually, in your code, the ampersand '&' is NOT whitelisted. So it should get removed.
    # So the result should be => "Hello ^*( )=+ End"
    # Then we collapse multiple spaces if any appear.
    # Final => "Hello ^*( )=+ End"
    expected_output = "Hello ^*( )=+ End"
    output = noise_removal(input_text)
    assert output == expected_output, f"Expected '{expected_output}', got '{output}'"

def test_noise_removal_fraction_symbols():
    """
    Ensure fraction symbols like ½, ⅓, etc. are preserved,
    while unknown symbols are removed.
    """
    input_text = "1½ + 1⅓ ???"
    # '???' -> '.'
    # We keep ½ and ⅓, we keep '+'
    # => "1½ + 1⅓."
    expected_output = "1½ + 1⅓ ."
    output = noise_removal(input_text)
    assert output == expected_output, f"Expected '{expected_output}', got '{output}'"

def test_noise_removal_spaces():
    """
    Verify multiple spaces collapse to one, leading/trailing spaces are stripped.
    """
    input_text = "   multiple    spaces   here   !!!  "
    # "!!!" -> "."
    # => "multiple spaces here ."
    # Then we see if there's an extra space before the period. Actually
    # the code doesn't remove space before the period, so let's see:
    # Steps:
    #   => "multiple spaces here ."
    # The code doesn't force removal of the space before the period. But we do collapse multiple spaces:
    # So final might be => "multiple spaces here ."
    # We'll accept that.
    expected_output = "multiple spaces here ."
    output = noise_removal(input_text)
    assert output == expected_output, f"Expected '{expected_output}', got '{output}'"

def test_preprocess_text_end_to_end():
    """
    End-to-end pipeline: lowercase + noise removal.
    We expect everything to be lowercase and punctuation to be normalized.
    """
    input_text = "   Move LEFT  50 centimeters!!!  ???"
    # Lowercase => "   move left  50 centimeters!!!  ???" '!!!' -> '.' '???' -> '.' => "move left 50 centimeters. ."
    # Then multiple spaces collapse => "move left 50 centimeters. ." Actually we get => "move left 50 centimeters."
    # Because your code doesn't specifically remove the space between the two '.' we might see "move left 50
    # centimeters. ." Let's check the code: There's no logic that merges consecutive periods. We do "Collapse
    # multiple consecutive periods" into a single period: So "!!! ???" => ". ." => " .." => which might collapse to
    # "." So final => "move left 50 centimeters."
    expected_output = "move left 50 centimeters."
    output = preprocess_text(input_text)
    assert output == expected_output, f"Expected '{expected_output}', got '{output}'"

def test_preprocess_text_equation():
    """
    We keep math symbols in any context.
    """
    input_text = "Calculate: 3 * 4 = 12???"
    # Lowercase => "calculate: 3 * 4 = 12???"
    # '???' -> '.'
    # The code doesn't specifically remove ':', let's see. Actually we do not list ':' in the whitelisted set.
    # So ':' is removed. So "calculate 3 * 4 = 12."
    # multiple spaces collapse -> "calculate 3 * 4 = 12."
    expected_output = "calculate 3 * 4 = 12."
    output = preprocess_text(input_text)
    assert output == expected_output, f"Expected '{expected_output}', got '{output}'"

def test_parentheses_are_preserved():
    """
    Ensure parentheses remain in the final output.
    """
    input_text = "Use (x^2 - y^2) = (x-y)(x+y)? Maybe???"
    # Lowercase => "use (x^2 - y^2) = (x-y)(x+y)? maybe???"
    # '???' -> '.'
    # => "use (x^2 - y^2) = (x-y)(x+y). maybe."
    # Then whitelisting:
    #  - we keep '(', ')', '^', '-', '='
    # => "use (x^2 - y^2) = (x-y)(x+y). maybe."
    # multiple spaces collapse if needed.
    expected_output = "use (x^2 - y^2) = (x-y)(x+y). maybe."
    output = preprocess_text(input_text)
    assert output == expected_output, f"Expected '{expected_output}', got '{output}'"


if __name__ == "__main__":
    # Run each test manually
    test_lowercase_normalization()
    test_noise_removal_exclamations_questions()
    test_noise_removal_punctuation_collapse()
    test_noise_removal_vertical_bar_slash()
    test_noise_removal_unallowed_chars()
    test_noise_removal_fraction_symbols()
    test_noise_removal_spaces()
    test_preprocess_text_end_to_end()
    test_preprocess_text_equation()
    test_parentheses_are_preserved()

    print("All tests passed successfully!")
