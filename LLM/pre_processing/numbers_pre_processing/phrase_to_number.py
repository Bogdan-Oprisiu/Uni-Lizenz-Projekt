# phrase_to_number.py

import re
from typing import List
from word2number import w2n

NUMBER_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen","twenty","thirty","forty","fifty",
    "sixty","seventy","eighty","ninety","hundred","thousand","million",
    "billion","trillion","and"
}


def tokenize_with_punctuation(text: str) -> List[str]:
    """
    Splits the text into tokens, ensuring:
      - decimals remain intact (e.g. '9.81')
      - known unit strings (e.g. 'm/s^2', 'km/h') remain single tokens
      - spelled-out words remain tokens
      - punctuation remains separate
    """
    pattern = (
        r"(?:m/s\^?2|m/s²|cm/s\^?2|cm/s²|km/h|m/s|cm/s)"  # recognized short unit strings
        r"|(?:\d+(?:\.\d+)?)"                           # decimal numbers
        r"|[a-zA-Z]+"                                   # pure alphabetic
        r"|[^\w\s]"                                     # single punctuation (commas, etc.)
    )
    return re.findall(pattern, text)


def is_spelled_number_word(token: str) -> bool:
    """Return True if token is in the spelled-out number set (e.g. 'twenty', 'hundred')."""
    return token.lower() in NUMBER_WORDS


def convert_spelled_numbers_phrases(text: str) -> str:
    """
    Convert multi-word spelled-out numbers (like "one hundred and twenty")
    into digits ("120"). Leaves decimal numbers alone, and keeps recognized
    units (like "m/s^2") as one token.
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
        # If the token is a spelled-out number word, accumulate in buffer
        if is_spelled_number_word(token):
            buffer.append(token)
        else:
            # Flush any number-phrase before adding non-number token
            flush_buffer()
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
        "One million and thirty one is an example.",
        "Move Fifty centimeters forward!",
        "Go ahead ten and five centimeters."
    ]
    for ex in examples:
        converted = convert_spelled_numbers_phrases(ex)
        print(f"Original:  {ex}")
        print(f"Converted: {converted}\n")
