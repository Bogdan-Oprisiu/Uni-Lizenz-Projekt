# processing.py

from preprocessing import preprocess_text
from basic_spell_checker import basic_spell_checker
from number_unit_normalization import normalize_numbers_units
from unit_conversion import normalize_all_units


def full_text_processing(text: str) -> str:
    """
    Run the entire pipeline on the raw user input text, returning the final normalized text.
    """
    # 1. Preprocessing (lowercase, noise removal, etc.)
    text = preprocess_text(text)

    # 2. Spell checking
    text = basic_spell_checker(text)

    # 3. Convert spelled-out numbers ("fifty" -> "50"), standardize units ("meters" -> "m"),
    #    infer missing units, convert common constants (pi -> 3.14).
    text = normalize_numbers_units(text)

    # 4. Finally do numeric conversions to get everything into [cm, cm/s, cm/s^2, rad].
    text = normalize_all_units(text)

    return text


if __name__ == "__main__":
    tests = [
        "Move Fifty centimeters forward!!!",
        "Accelerate 10 with no unit.",
        "Turn 45 with no unit.",
        "Drive 20 with no unit.",
        "Rotate pi.",
        "Rotate 180 degrees.",
        "Accelerate at 9.81 m/s^2."
    ]

    for t in tests:
        out = full_text_processing(t)
        print(f"Original:   {t}")
        print(f"Processed:  {out}\n")
