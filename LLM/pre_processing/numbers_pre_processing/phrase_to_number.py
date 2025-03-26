import re
from typing import List

from word2number import w2n

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "trillion", "and"
}


def tokenize_with_punctuation(text: str) -> List[str]:
    """
    Splits the text into tokens, ensuring:
      - decimals remain intact (e.g. '9.81')
      - known unit strings (e.g. 'm/s^2', 'km/h') remain single tokens
      - spelled-out words, including hyphenated (e.g. 'fifty-six') remain tokens
      - punctuation remains separate

    The pattern now captures:
      1) recognized short unit strings, e.g. "m/s^2", "km/h"
      2) decimal numbers, e.g. "3.14"
      3) spelled-out alpha with optional internal hyphens, e.g. "fifty-six"
      4) single punctuation
    """
    pattern = (
        r"(?:m/s\^?2|m/s²|cm/s\^?2|cm/s²|km/h|m/s|cm/s)"  # recognized short unit strings
        r"|(?:\d+(?:\.\d+)?)"  # decimal numbers
        r"|[a-zA-Z]+(?:-[a-zA-Z]+)*"  # spelled words, possibly hyphenated (fifty-six)
        r"|[^\w\s]"  # single punctuation, e.g. commas, !
    )
    return re.findall(pattern, text)


def is_spelled_number_word(token: str) -> bool:
    """
    Return True if *all parts* of a possibly-hyphenated token are in NUMBER_WORDS.
    e.g. 'fifty-six' => split -> ['fifty','six'], each must be in NUMBER_WORDS.
    """
    parts = token.lower().split('-')
    return all(part in NUMBER_WORDS for part in parts)


def convert_spelled_numbers_phrases(text: str) -> str:
    """
    Convert multi-word spelled-out numbers (e.g., "one hundred and twenty")
    or hyphenated numbers ("fifty-six") into digits.

    - Leaves decimal numbers alone.
    - Keeps recognized short-unit strings (m/s^2, km/h, etc.) as single tokens.
    """
    tokens = tokenize_with_punctuation(text)
    result_tokens = []
    buffer = []

    def flush_buffer():
        """Try to parse the buffered spelled-out words with word2number."""
        if not buffer:
            return
        phrase = " ".join(buffer)
        try:
            number_value = w2n.word_to_num(phrase.lower())
            result_tokens.append(str(number_value))
        except ValueError:
            # If it fails, just keep them as-is
            result_tokens.extend(buffer)
        buffer.clear()

    for token in tokens:
        # If the token is a spelled-out number (including possible hyphens),
        # accumulate in buffer so we can parse multi-word phrases like:
        # "three hundred and fifty-six" => ["three","hundred","and","fifty-six"]
        if is_spelled_number_word(token):
            # remove hyphens -> spaces for w2n compatibility
            # e.g. "fifty-six" -> "fifty six"
            cleaned = token.replace("-", " ")
            buffer.append(cleaned)
        else:
            # Not spelled-out => flush the buffer first
            flush_buffer()
            # Then just add this token to the result
            result_tokens.append(token)

    # End: flush leftover buffer
    flush_buffer()

    # Join tokens back; remove spaces before punctuation
    raw_text = " ".join(result_tokens)
    final_text = re.sub(r"\s+([^\w\s])", r"\1", raw_text)
    return final_text


if __name__ == "__main__":
    examples = [
        "I have one hundred and twenty apples.",
        "She ran two thousand five hundred meters.",
        "There are three hundred and fifty-six steps.",
        "One million and thirty-one is an example.",
        "Move Fifty centimeters forward!",
        "Go ahead ten and five centimeters.",
        "Ninety-nine bottles of beer on the wall.",
        "Four hundred and forty-four thousand boxes.",
    ]
    for ex in examples:
        converted = convert_spelled_numbers_phrases(ex)
        print(f"Original:  {ex}")
        print(f"Converted: {converted}\n")
