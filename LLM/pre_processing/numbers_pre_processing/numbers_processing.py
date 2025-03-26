# numbers_processing.py

from number_unit_normalization import normalize_numbers_units
from unit_conversion import normalize_all_units  # your existing conversion code


def combined_numbers_pipeline(text: str) -> str:
    """
    Full pipeline:
      1) Normalize spelled-out numbers and textual unit expressions.
      2) Convert normalized short-form units to final numeric values.
    """
    intermediate = normalize_numbers_units(text)
    final = normalize_all_units(intermediate)
    return final


if __name__ == "__main__":
    test_sentences = [
        "Move fifty centimeters forward!!!",
        "Go ahead four hundred eighteen centimeters.",
        "Walk one hundred and twenty meters.",
        "Run two thousand five hundred kilometres!",
        "Move 50 with no unit.",
        "Go 10.0 with no unit?",
        "Turn right ninety degrees.",
        "Rotate 180 degrees!",
        "Turn 45 with no unit.",
        "Rotate pi.",
        "Rotate τ.",
        "Drive 20 with no unit.",
        "Drive at 60 kilometers per hour.",
        "Run at 10 m/s.",
        "Speed is 2.5 m/s??",
        "Drive 100 cm/s or 0.5 km/h??",
        "Accelerate 10 with no unit.",
        "Accelerate at ten meters per second squared.",
        "Accelerate at 9.81 m/s^2.",
        "Acceleration is 0.05 m/s²",
        "We used 1 cm/s^2 in that experiment.",
        "Accelerate at 9.81 m/s^2, then rotate 180 deg and move 2.5m",
        "He turned 45 deg, walked 3 kilometers, then drove 10 kilometers per hour.",
    ]
    for idx, sentence in enumerate(test_sentences, 1):
        processed = combined_numbers_pipeline(sentence)
        print(f"{idx}) Original:   {sentence}")
        print(f"   Processed:  {processed}\n")
