"""
A basic spell checker module that uses the pyspellchecker library.
This module provides a function to correct spelling errors in a given text,
but it will not modify words that represent units (like "km", "m", etc.).
"""

from spellchecker import SpellChecker

# Initialize the spell checker once.
# You may customize this by loading domain-specific words into the dictionary.
spell = SpellChecker()
spell.distance = 2

# Define a set of unit abbreviations and other tokens that should be left unchanged.
PROTECTED_WORDS = {
    "km", "m", "cm", "deg", "rad",
    "km/h", "m/s", "cm/s", "m/s^2", "cm/s^2"
}


def basic_spell_checker(text: str) -> str:
    """
    Check each word in the text and replace it with its most likely correction.
    This function only corrects purely alphabetic words that are not in the protected set.
    Numbers and punctuation are left untouched.
    """
    words = text.split()
    corrected_words = []
    for word in words:
        # Check if the word (in lowercase) is in the protected set.
        if word.lower() in PROTECTED_WORDS:
            corrected_words.append(word)
        # Only correct words that are fully alphabetic.
        elif word.isalpha():
            correction = spell.correction(word)
            # Fallback to the original word if no correction is found.
            if correction is None:
                correction = word
            corrected_words.append(correction)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)


if __name__ == '__main__':
    sample_texts = [
        "This is an exampel of a sentense with erors.",
        "The quik brown fox jumpd over the lazy dog!",
        "Spellng mistaks are common in humann input.",
        "Move 50 km and then go 10 m/s.",
        "Accelerate 9.81 m/s^2 when needed."
    ]

    for text in sample_texts:
        corrected = basic_spell_checker(text)
        print("Original: ", text)
        print("Corrected:", corrected)
        print()
