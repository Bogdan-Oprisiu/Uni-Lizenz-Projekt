import re


def lowercase_normalization(text: str) -> str:
    """
    Convert all text to lowercase.
    """
    return text.lower()


def preserve_asterisks_in_context(text: str) -> str:
    """
    Identify valid usage of '*' (e.g., "3 * 4", "cm*s^-2") and replace the star
    with a placeholder [STAR_PRESERVE]. All other stars remain as-is.

    Example recognized contexts:
      - "3 * 4" (with spaces or no spaces)
      - "cm*s^-2" (units without spaces)
    """
    preserve_pattern = r'(\w+)(\s*)\*(\s*)(\w+(\^-?\d+)?)'

    def preserve_match(m):
        left_word = m.group(1)
        left_spaces = m.group(2)
        right_spaces = m.group(3)
        right_word = m.group(4)
        return f"{left_word}{left_spaces}[STAR_PRESERVE]{right_spaces}{right_word}"

    text = re.sub(preserve_pattern, preserve_match, text)
    return text


def noise_removal(text: str) -> str:
    """
    Remove extraneous characters while preserving:
      - letters, digits, underscores
      - whitespace
      - basic punctuation (. , /)
      - star '*'
      - parentheses ( )
      - math operators: + - = ^ %
      - multiplication symbol ×
      - fraction symbols (½, ⅓, etc.)
      - placeholders [STAR_PRESERVE]

    Steps:
        1) Convert !/? sequences to '.'
        2) Collapse multiple periods/commas
        3) Convert vertical bars/backslashes to '/'
        4) Whitelist allowed characters
        5) Collapse multiple spaces & strip
        6) Replace [STAR_PRESERVE] with ' * '
        7) Collapse spaces again
        8) Merge consecutive empty sentences ('. .') into a single period
    """

    # 1) Convert multiple '!' or '?' => single '.'
    text = re.sub(r"[!?]+", ".", text)

    # 2) Collapse multiple consecutive periods or commas => single
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)

    # 3) Convert vertical bars/backslashes => '/'
    text = re.sub(r"[\\|]+", "/", text)

    # 4) Whitelist approach
    allowed_chars = (
        r"[^\w\s\.,/\*\(\)\+\-\=\^\%\×½⅓⅔¼¾⅛⅜⅝⅞"
        r"\[\]STAR_PRESERVE]"
    )
    cleaned = re.sub(allowed_chars, "", text)

    # 5) Collapse multiple spaces, strip
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 6) Replace placeholders with ' * '
    cleaned = cleaned.replace("[STAR_PRESERVE]", " * ")

    # 7) Collapse spaces again
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 8) Merge consecutive empty sentences like ". ." into a single "."
    #    This handles the case "some text. ." => "some text."
    cleaned = re.sub(r"\.\s+\.", ".", cleaned)

    return cleaned


def preprocess_text(text: str) -> str:
    """
    Full pipeline:
      1) lowercase
      2) preserve recognized asterisks
      3) remove noise / unrecognized symbols
    """
    text = lowercase_normalization(text)
    text = preserve_asterisks_in_context(text)
    text = noise_removal(text)
    return text


if __name__ == "__main__":
    sample_texts = [
        # This first one yields "move left 50 centimeters. ."
        # without the new Step 8. Now it should unify into "move left 50 centimeters."
        "Move LEFT  50 centimeters!!! ???",

        "3 * 4 plus some stray * asterisks *** and cm*s^-2.",
        "Hello @#$%^&*( )=+~ End",
        "What is 1\\2 plus 3/4? Some *random* bullet points *like this*?",
        "Use (x^2 - y^2) = (x-y)(x+y) if you want? Maybe???"
    ]

    for text in sample_texts:
        processed = preprocess_text(text)
        print(f"Original: {repr(text)}\nProcessed: {repr(processed)}\n")
