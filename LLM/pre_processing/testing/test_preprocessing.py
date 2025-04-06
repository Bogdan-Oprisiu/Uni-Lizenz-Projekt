"""
test_preprocessing.py

This file contains tests for the text preprocessing pipeline defined in preprocess_text.py.
We will cover a variety of edge cases to ensure our pipeline works as intended.
"""

import sys
import os
import re

# If needed, adjust the path so we can import from the same directory or a parent directory.
# For example, if "preprocess_text.py" is one level above, do:
# sys.path.append(os.path.abspath(".."))

# Import the functions from the module you provided.
# Assuming it's named "preprocess_text.py" and the functions are as defined:
from preprocess_text import preprocess_text, lowercase_normalization, noise_removal

def test_lowercase_normalization():
    """Verify that the text is converted to lowercase."""
    input_text = "This Is A TEST"
    expected_output = "this is a test"
    assert lowercase_normalization(input_text) == expected_output, \
        f"Expected '{expected_output}' but got '{lowercase_normalization(input_text)}'"

def test_noise_removal_exclamations_questions():
    """
    Convert multiple exclamation marks and question marks into a single period.
    """
    input_text = "Hey!!! Are you sure??"
    # Should become: "Hey. Are you sure."
    # Then we'll see if it collapses punctuation as intended
    # but doesn't remove valid spaces or letters.
    expected_output = "Hey. Are you sure."
    processed = noise_removal(input_text)
    assert processed == expected_output, \
        f"Expected '{expected_output}' but got '{processed}'"

def test_noise_removal_multiple_periods_commas():
    """
    Collapse consecutive periods or commas down to one.
    """
    input_text = "Ellipsis...... and,,, commas!"
    # Step by step:
    #   - "!" -> "."
    #   - multiple periods -> one period
    #   - multiple commas -> one comma
    # So it should become something like: "Ellipsis. and, commas."
    expected_output = "Ellipsis. and, commas."
    processed = noise_removal(input_text)
    assert processed == expected_output, \
        f"Expected '{expected_output}' but got '{processed}'"

def test_noise_removal_vertical_bar():
    """
    Vertical bars should be converted to forward slashes.
    """
    input_text = "This|is|a\\test"
    # Both '|' and '\' should become '/', giving "This/is/a/test".
    expected_output = "This/is/a/test"
    processed = noise_removal(input_text)
    assert processed == expected_output, \
        f"Expected '{expected_output}' but got '{processed}'"

def test_noise_removal_unallowed_chars():
    """
    Ensure that characters not in the allowed set are removed.
    """
    input_text = "Hello @#$%^&()=+~ World"
    # None of @#$%^&()=+~ are in the allowed set, so they should vanish.
    # The result should be "Hello World"
    expected_output = "Hello World"
    processed = noise_removal(input_text)
    assert processed == expected_output, \
        f"Expected '{expected_output}' but got '{processed}'"

def test_noise_removal_multiplication_symbol():
    """
    Check that the multiplication symbol '×' is preserved,
    but unknown symbols like 'ß' or '©' are removed.
    """
    input_text = "3 × 4 = 12 ß©"
    # We preserve '×' but remove ß©
    # Also, '=' is not in the allowed list so it should be removed.
    # The result should be: "3 × 4 12"
    expected_output = "3 × 4 12"
    processed = noise_removal(input_text)
    assert processed == expected_output, \
        f"Expected '{expected_output}' but got '{processed}'"

def test_noise_removal_fraction_symbols():
    """
    Ensure common fraction symbols like ½, ⅓, ¼ are preserved.
    """
    input_text = "1½ + 1⅓ = 2??"
    # We'll see how it handles the fraction symbols.
    # The '??' -> '.' as per exclamation/question rule.
    # So final expected: "1½ + 1⅓ = 2."
    # The '=' is not in the allowed set, so it should be removed too.
    # So final should be: "1½ + 1⅓ 2."
    expected_output = "1½ + 1⅓ 2."
    processed = noise_removal(input_text)
    assert processed == expected_output, \
        f"Expected '{expected_output}' but got '{processed}'"

def test_noise_removal_spaces():
    """
    Check if multiple spaces are collapsed into one space
    and leading/trailing spaces are trimmed.
    """
    input_text = "   Hello      World   "
    expected_output = "Hello World"
    processed = noise_removal(input_text)
    assert processed == expected_output, \
        f"Expected '{expected_output}' but got '{processed}'"

def test_preprocess_text_end_to_end():
    """
    Test the full pipeline: lowercase + noise removal on a complex example.
    """
    input_text = "   Move LEFT  50 centimeters!!!  ???"
    # First, to lowercase: "   move left  50 centimeters!!!  ???"
    # Then noise removal:
    #   - "!!!" -> "."
    #   - "???" -> "."
    #   - multiple spaces collapsed
    # Should end up: "move left 50 centimeters."
    expected_output = "move left 50 centimeters."
    processed = preprocess_text(input_text)
    assert processed == expected_output, \
        f"Expected '{expected_output}' but got '{processed}'"


def test_preprocess_text_equation_symbols():
    """
    Test the pipeline with some equation-like text, checking
    star/asterisk and forward slash are preserved.
    """
    input_text = "CALCULATE: 3 * 4 ??? 3/4 ???"
    # Lowercase => "calculate: 3 * 4 ??? 3/4 ???"
    # '???' -> '.'
    # ':' is not in the allowed set (but you might want to keep it, see your pattern),
    #   so it should be removed unless you specifically add it to the noise removal.
    # By default, we do allow colons in the code snippet, so let's check:
    # Actually we do have ':' in the special punctuation set, so it might remain
    # if we add it to the set. But in your code, the comment says "Remove any characters
    # not alphanumeric, whitespace, commas, periods, forward slashes, asterisks, the multiplication symbol (×)..."
    # So colon will be removed.
    # Final: "calculate 3 * 4 . 3/4 ."
    expected_output = "calculate 3 * 4 . 3/4 ."
    processed = preprocess_text(input_text)
    assert processed == expected_output, \
        f"Expected '{expected_output}' but got '{processed}'"


# If you want to run the tests via command line, you can do:
# python test_preprocessing.py
# The asserts will raise an AssertionError if any test fails.
if __name__ == "__main__":
    test_lowercase_normalization()
    test_noise_removal_exclamations_questions()
    test_noise_removal_multiple_periods_commas()
    test_noise_removal_vertical_bar()
    test_noise_removal_unallowed_chars()
    test_noise_removal_multiplication_symbol()
    test_noise_removal_fraction_symbols()
    test_noise_removal_spaces()
    test_preprocess_text_end_to_end()
    test_preprocess_text_equation_symbols()

    print("All tests passed successfully!")
