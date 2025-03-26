# processing.py

from normalization_and_noise_removal import preprocess_text
from spell_checker import basic_spell_checker
from pre_processing.numbers_pre_processing.number_unit_normalization import normalize_numbers_units
from pre_processing.numbers_pre_processing.unit_conversion import normalize_all_units


def full_text_processing(text: str) -> str:
    """
    Run the entire pipeline on the raw user input text, returning the final normalized text:
      1. Lowercase normalization & noise removal
      2. Spell checking
      3. Converting spelled-out numbers, standardizing textual units, inferring defaults, replacing constants
      4. Final numeric conversions to [cm, cm/s, cm/s^2, rad]
    """
    # 1. Preprocessing (lowercase, noise removal, etc.)
    text = preprocess_text(text)

    # 2. Spell checking
    text = basic_spell_checker(text)

    # 3. Convert spelled-out numbers ("fifty" -> "50"), standardize units ("meters" -> "m"),
    #    infer missing units, replace constants (pi -> 3.14), etc.
    text = normalize_numbers_units(text)

    # 4. Finally, convert short-form units to final forms [cm, cm/s, cm/s^2, rad].
    text = normalize_all_units(text)

    return text


# For testing
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
