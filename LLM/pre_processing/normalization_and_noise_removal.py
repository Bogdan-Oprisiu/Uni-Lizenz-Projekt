import re


def lowercase_normalization(text: str) -> str:
    """
    Convert all text to lowercase.
    """
    return text.lower()


def noise_removal(text: str) -> str:
    """
    Remove extraneous special characters while preserving basic punctuation and
    mathematical symbols used in equations. Specifically, we:
      - Convert one or more exclamation/question marks to a single period.
      - Collapse multiple consecutive periods or commas.
      - Convert vertical bars (|) to forward slashes (/).
      - Remove any characters that are not alphanumeric, whitespace, commas, periods,
        forward slashes, asterisks, the multiplication symbol (×), or common fraction symbols.
    """
    # Convert exclamation and question marks to a period
    text = re.sub(r"[!?]+", ".", text)
    # Collapse multiple periods into one
    text = re.sub(r'\.{2,}', '.', text)
    # Collapse multiple commas into one
    text = re.sub(r',{2,}', ',', text)
    # Convert vertical bars to forward slashes
    text = re.sub(r"[\\|]+", "/", text)

    # Allowed characters:
    # - \w (alphanumeric plus underscore)
    # - \s (whitespace)
    # - Comma, period, forward slash, asterisk
    # - Multiplication symbol: ×
    # - Common Unicode fraction symbols: ½, ⅓, ⅔, ¼, ¾, ⅛, ⅜, ⅝, ⅞
    allowed_chars = r"[^\w\s,.\/*×½⅓⅔¼¾⅛⅜⅝⅞]"
    cleaned = re.sub(allowed_chars, "", text)

    # Collapse multiple spaces and strip leading/trailing spaces.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def preprocess_text(text: str) -> str:
    """
    Run the preprocessing pipeline: lowercase normalization followed by noise removal.
    """
    text = lowercase_normalization(text)
    text = noise_removal(text)
    return text


# For testing purposes, run this module directly.
if __name__ == "__main__":
    sample_texts = [
        "   Stop.   ",
        "Move LEFT  50 centimeters!!!  ",
        "Calculate 3 * 4 = 12?",
        "What is 1\\2 plus 3/4?",
        "The measurement is 1½ meters.",
        "Strafe right: 217 cm, with acceleration   9???"
    ]

    for text in sample_texts:
        processed = preprocess_text(text)
        print(f"Original: {text}\nProcessed: {processed}\n")
