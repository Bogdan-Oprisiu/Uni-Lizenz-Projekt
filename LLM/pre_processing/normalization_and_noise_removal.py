import re

def lowercase_normalization(text: str) -> str:
    """
    Convert all text to lowercase.
    """
    return text.lower()

def noise_removal(text: str) -> str:
    """
    Remove any special characters from the text, allowing only alphanumeric
    characters, whitespace, commas, and periods. Exclamation and question marks
    are converted to a period so that sentence endings are preserved.
    """
    # Convert exclamation and question marks to a period and collapse multiple punctuation.
    text = re.sub(r"[!?]+", ".", text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    # Remove any character that is not a word character, whitespace, comma, or period.
    cleaned = re.sub(r"[^\w\s,.]", "", text)
    # Collapse multiple spaces into a single space and strip leading/trailing spaces.
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
        "Strafe right: 217 cm, with acceleration   9???",
        "Go   ahead 418 centimeters; please.",
        "Move BACKWARD 356 cm with acceleration ."
    ]

    for text in sample_texts:
        processed = preprocess_text(text)
        print(f"Original: {text}\nProcessed: {processed}\n")
