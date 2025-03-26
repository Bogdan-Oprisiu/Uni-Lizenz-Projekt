# unit_conversion.py

import math
import re


#
# 1. ANGLES
#

def convert_angle(value: float, is_degrees: bool) -> str:
    """Convert numeric degrees to radians, or reformat radians to 4 decimals."""
    if is_degrees:
        return f"{(value * math.pi / 180.0):.4f} rad"
    else:
        return f"{value:.4f} rad"


def normalize_angles(text: str) -> str:
    """
    Convert things like "45 deg" -> "0.7854 rad",
    "45 degrees" -> "0.7854 rad",
    "1.57 rad" -> "1.5700 rad", etc.
    """
    pattern = re.compile(
        r"(?P<val>\d+(?:\.\d+)?)(?:\s*°|\s*(?:degrees|degree|deg))(?P<deg>\b)?"
        r"|(?P<val2>\d+(?:\.\d+)?)(?:\s*(?:radians|radian|rad))(?P<rad>\b)?",
        flags=re.IGNORECASE
    )

    def _repl(m):
        if m.group("val") is not None:
            val = float(m.group("val"))
            return convert_angle(val, is_degrees=True)
        else:
            val = float(m.group("val2"))
            return convert_angle(val, is_degrees=False)

    return pattern.sub(_repl, text)


#
# 2. DISTANCE
#

def convert_distance_to_cm(value: float, unit: str) -> float:
    """km -> *100000, m -> *100, cm -> *1."""
    unit = unit.lower()
    if unit in ["km", "kilometers", "kilometres", "kilometer"]:
        return value * 100000.0
    elif unit in ["m", "meters", "metres", "meter"]:
        return value * 100.0
    return value  # already cm or unknown


def normalize_distance(text: str) -> str:
    """
    Convert recognized distance to cm, e.g. "10 m" -> "1000.00 cm".

    IMPORTANT FIX:
      - We skip matching 'm' if it's actually 'm/s' or 'm/s^2' (i.e. negative lookahead).
    """
    pattern = re.compile(
        # 1) Capture a decimal number in group(1).
        # 2) Then look for space+distance-unit, but:
        #    - "m(?!/)" means 'm' not followed by '/'
        #    - same for 'meter(?!/)', 'km(?!/)', etc.
        r"(\d+(?:\.\d+)?)(?:\s*(?:km(?!/)|kilometers(?!/)|kilometres(?!/)|kilometer(?!/)"
        r"|m(?!/)|meters(?!/)|metres(?!/)|meter(?!/)|cm(?!/)|centimeters(?!/)|centimetres(?!/)|centimeter(?!/))"
        r"|(?=km(?!/)|m(?!/)|cm(?!/)))",
        flags=re.IGNORECASE
    )

    def _repl(m):
        full_str = m.group(0)
        numeric_part = re.findall(r"\d+(?:\.\d+)?", full_str)[0]
        # Keep only letters (no slash)
        unit_part = re.sub(r"[^a-zA-Z]+", "", full_str)

        val = float(numeric_part)
        val_cm = convert_distance_to_cm(val, unit_part)
        return f"{val_cm:.2f} cm"

    return pattern.sub(_repl, text)


#
# 3. SPEED
#

def convert_speed_to_cms(value: float, unit: str) -> float:
    """km/h -> factor ~27.78, m/s -> *100, cm/s -> *1."""
    unit = unit.lower()
    if unit in ["km/h", "kilometers per hour", "kilometres per hour"]:
        return value * 100000.0 / 3600.0
    elif unit in ["m/s", "meters per second", "metres per second"]:
        return value * 100.0
    return value  # already cm/s or unknown


def normalize_speed(text: str) -> str:
    """
    Convert recognized speed to cm/s.

    IMPORTANT FIX:
      - skip 'm/s^2' by negative lookahead (?!\^), so we don't mistakenly treat "m/s^2" as speed.
    """
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)(?:\s*(?:km/h|kilometers per hour|kilometres per hour"
        r"|m/s(?!\^)|meters per second(?! squared)|metres per second(?! squared)"
        r"|cm/s|centimeters per second|centimetres per second)"
        r"|(?=km/h|m/s(?!\^)|cm/s))",
        flags=re.IGNORECASE
    )

    def _repl(m):
        full_str = m.group(0)
        numeric_part = re.findall(r"\d+(?:\.\d+)?", full_str)[0]
        unit_part = re.sub(r"[^a-zA-Z/]+", "", full_str)

        val = float(numeric_part)
        val_cms = convert_speed_to_cms(val, unit_part)
        return f"{val_cms:.2f} cm/s"

    return pattern.sub(_repl, text)


#
# 4. ACCELERATION
#

def convert_accel_to_cms2(value: float, unit: str) -> float:
    """m/s^2 -> *100, cm/s^2 -> *1."""
    unit = unit.lower()
    # unify them all
    if unit in ["m/s^2", "m/s2", "m/s²", "meters per second squared", "metres per second squared"]:
        return value * 100.0
    return value  # already cm/s^2 or unknown


def normalize_acceleration(text: str) -> str:
    """
    Convert recognized acceleration to cm/s^2.
    E.g. "2 m/s^2" -> "200.00 cm/s^2".
    """
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)(?:\s*(?:m/s\^?2|m/s²|meters per second squared|metres per second squared"
        r"|cm/s\^?2|cm/s²|centimeters per second squared|centimetres per second squared)"
        r"|(?=m/s\^?2|cm/s\^?2))",
        flags=re.IGNORECASE
    )

    def _repl(m):
        full_str = m.group(0)
        numeric_part = re.findall(r"\d+(?:\.\d+)?", full_str)[0]
        # keep letters, '/', '^', and '²' to detect exact variant
        unit_part = re.sub(r"[^a-zA-Z/\^²]+", "", full_str)

        val = float(numeric_part)
        val_cms2 = convert_accel_to_cms2(val, unit_part)
        return f"{val_cms2:.2f} cm/s^2"

    return pattern.sub(_repl, text)


#
# 5. FULL PIPELINE
#

def normalize_all_units(text: str) -> str:
    """
    1) angles -> rad
    2) distance -> cm
    3) speed -> cm/s
    4) acceleration -> cm/s^2
    """
    text = normalize_angles(text)
    text = normalize_distance(text)
    text = normalize_speed(text)
    text = normalize_acceleration(text)
    return text


# 6. TESTING
if __name__ == "__main__":
    sample_texts = [
        # Angles
        "Turn right 90 degrees.",
        "Turn right 90°.",
        "Rotate left 1.57 radians.",
        "Rotate 180 deg.",
        "Rotate 2 rad.",
        "Turn 360 deg!",
        "Turn 45° to face north.",

        # Distances
        "Move 5 m forward.",
        "Move 5m forward.",
        "Move 1.5 m forward.",
        "Jump 10.0m high.",
        "Run 0.33 m forward.",
        "Go 100 cm. Then 2.5 km ahead!",

        # Speed
        "Run at 10 m/s.",
        "Run at 10.0m/s.",
        "Drive 2 km/h?",
        "Drive 60.5 km/h now.",
        "Run at 99.9 cm/s, or 1.2 m/s maybe.",

        # Acceleration
        "Accelerate at 2 m/s^2.",
        "Accelerate at 2m/s^2!",
        "Acceleration is 3.33 m/s² for test.",
        "We had 1 cm/s^2 motion.",
        "Testing 9.81 m/s^2 exactly.",
        "What if it's 0.1 m/s^2 or 0.05 cm/s^2?",

        # Combined
        "Accelerate at 9.81 m/s^2, then turn 90 degrees and move 2.5m!",
        "He turned 45 deg, walked 3 km, then drove 10 kilometers per hour."
    ]

    for text in sample_texts:
        out = normalize_all_units(text)
        print(f"Original:   {text}")
        print(f"Normalized: {out}\n")
