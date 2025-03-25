"""
A basic spell checker module that uses the pyspellchecker library.
This module provides a function to correct spelling errors in a given text.
"""

from spellchecker import SpellChecker

# Initialize the spell checker once.
# You may customize this by loading domain-specific words into the dictionary.
spell = SpellChecker()


def basic_spell_checker(text: str) -> str:
    """
    Check each word in the text and replace it with its most likely correction.
    This function only corrects alphabetic words; numbers and punctuation are left untouched.

    Parameters:
        text (str): The input text to be spell-checked.

    Returns:
        str: The text with spelling corrections applied.
    """
    # Simple whitespace tokenization
    words = text.split()
    corrected_words = []

    for word in words:
        # We only correct words that are fully alphabetic.
        # This avoids changing numbers, special symbols, or mixed strings.
        if word.isalpha():
            correction = spell.correction(word)
            corrected_words.append(correction)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


# For testing purposes, this block will run if the module is executed directly.
if __name__ == '__main__':
    sample_texts = [
        "This is an exampel of a sentense with erors.",
        "The quik brown fox jumpd over the lazy dog!",
        "Spellng mistaks are common in humann input."
    ]

    for text in sample_texts:
        corrected = basic_spell_checker(text)
        print("Original: ", text)
        print("Corrected:", corrected)
        print()
