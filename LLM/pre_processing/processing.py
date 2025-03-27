# processing.py
from pre_processing.normalization_and_noise_removal import preprocess_text
from pre_processing.numbers_pre_processing.number_unit_normalization import normalize_numbers_units
from pre_processing.numbers_pre_processing.unit_conversion import normalize_all_units
from pre_processing.spell_checker import basic_spell_checker


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


if __name__ == "__main__":
    # A broader set of test lines, checking spelled-out numbers, default units, angles, speed, acceleration, etc.
    tests = [
        # Basic spelled-out + distance
        "Move Fifty centimeters forward!!!",
        "Go ahead four hundred eighteen centimeters.",
        "Walk one hundred and twenty meters.",
        "Run two thousand five hundred kilometres!",
        "Move 50 with no unit.",
        "Go 10.0 with no unit?",

        # Angles + spelled-out
        "Turn right ninety degrees.",
        "Rotate 180 degrees!",
        "Turn 45 with no unit.",
        "Rotate pi.",
        "Rotate 180 deg.",
        "Rotate τ.",  # if typed as 'π' or "tau" => "6.28"

        # Speed
        "Drive 20 with no unit.",
        "Drive at 60 kilometers per hour.",
        "Run at 10 m/s.",
        "Speed is 2.5 m/s??",
        "Drive 100 cm/s or 0.5 km/h??",

        # Acceleration
        "Accelerate 10 with no unit.",
        "Accelerate at ten meters per second squared.",
        "Accelerate at 9.81 m/s^2.",
        "We had 1 cm/s^2 motion.",
        "What if it's 0.1 m/s^2 or 0.05 cm/s^2?",
        "Acceleration is 3.33 m/s² for test.",

        # Mixed scenario
        "Accelerate at 9.81 m/s^2, then rotate 180 deg and move 2.5m!",
        "He turned 45 deg, walked 3 kilometers, then drove 10 kilometers per hour.",

        # Checking minutes/seconds ticks in angles
        "Turn 45° to face north.",
        "Turn 30' to face east.",
        "Turn 15'' for precision.",
    ]

    for t in tests:
        out = full_text_processing(t)
        print(f"Original:   {t}")
        print(f"Processed:  {out}\n")
