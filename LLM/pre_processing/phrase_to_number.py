# phrase_to_number.py

import re
from typing import List

from word2number import w2n

# A small set of recognized spelled-out number words.
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "trillion", "and"
}


def tokenize_with_punctuation(text: str) -> List[str]:
    """
    Splits the text into tokens, separating punctuation from words.
    E.g. "one hundred and twenty." -> ["one", "hundred", "and", "twenty", "."]
    """
    pattern = r"[a-zA-Z]+|\d+|[^\w\s]"
    return re.findall(pattern, text)


def is_spelled_number_word(token: str) -> bool:
    """
    Return True if token is a recognized spelled-out number word
    (including 'and' for w2n compatibility).
    """
    return token.lower() in NUMBER_WORDS


def convert_spelled_numbers_phrases(text: str) -> str:
    """
    Convert multi-word spelled-out numbers (e.g., "one hundred and twenty") into digits ("120").
    """
    tokens = tokenize_with_punctuation(text)
    result_tokens = []
    buffer = []

    def flush_buffer():
        """Attempt to parse accumulated spelled-out words with word2number."""
        if not buffer:
            return
        phrase = " ".join(buffer)
        try:
            number_value = w2n.word_to_num(phrase.lower())
            result_tokens.append(str(number_value))
        except ValueError:
            # Parsing failed; put them back as original words
            result_tokens.extend(buffer)
        buffer.clear()

    for token in tokens:
        if is_spelled_number_word(token):
            # Accumulate numbery words
            buffer.append(token)
        else:
            # Flush any number phrase we've built so far
            flush_buffer()
            # Non-number token goes straight to result
            result_tokens.append(token)

    # End of loop: flush remaining buffer
    flush_buffer()

    # Rejoin tokens; remove spaces before punctuation
    raw_text = " ".join(result_tokens)
    final_text = re.sub(r"\s+([^\w\s])", r"\1", raw_text)
    return final_text


if __name__ == "__main__":
    examples = [
        "I have one hundred and twenty apples.",
        "She ran two thousand five hundred meters.",
        "There are three hundred and fifty-six steps.",
        "One million and thirty one is an example.",
        "Move Fifty centimeters forward!",
        "Go ahead ten and five centimeters."
    ]
    for ex in examples:
        converted = convert_spelled_numbers_phrases(ex)
        print(f"Original:  {ex}")
        print(f"Converted: {converted}\n")
