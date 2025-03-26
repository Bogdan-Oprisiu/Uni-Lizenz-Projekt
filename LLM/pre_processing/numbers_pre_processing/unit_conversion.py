import math
import re


#
# 1. ANGLES (unchanged)
#

def convert_angle(value: float, is_degrees: bool) -> str:
    """Convert numeric degrees to radians, or reformat radians to 4 decimals."""
    if is_degrees:
        return f"{(value * math.pi / 180.0):.4f} rad"
    else:
        return f"{value:.4f} rad"


def normalize_angles(text: str) -> str:
    pattern = re.compile(
        r"(?P<val>\d+(?:\.\d+)?)(?:\s*°|\s*(?:degrees|degree|deg))"
        r"|(?P<val2>\d+(?:\.\d+)?)(?:\s*(?:radians|radian|rad))",
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
    return value


def normalize_distance(text: str) -> str:
    """
    Convert recognized distance to cm, e.g. "10 m" -> "1000.00 cm".

    The pattern uses a negative lookahead (?!/) so that if the unit is part of
    a compound expression (like "m/s^2"), it will not be captured here.
    """
    pattern = re.compile(
        r"""
        (\d+(?:\.\d+)?)
        (?:\s*
          (?:km(?!/)|kilometers(?!/)|kilometres(?!/)|kilometer(?!/)|
              m(?!/)|meters(?!/)|metres(?!/)|meter(?!/)|
              cm(?!/)|centimeters(?!/)|centimetres(?!/)|centimeter(?!/))
        )
        (?=[\s,;.!?]|$)
        """,
        flags=re.IGNORECASE | re.VERBOSE
    )

    def _repl(m):
        full_str = m.group(0)
        numeric_part = re.findall(r"\d+(?:\.\d+)?", full_str)[0]
        # Remove non-letter characters from unit portion.
        unit_part = re.sub(r"[^a-zA-Z]+", "", full_str[len(numeric_part):].strip())
        val = float(numeric_part)
        val_cm = convert_distance_to_cm(val, unit_part)
        return f"{val_cm:.2f} cm"

    return pattern.sub(_repl, text)


#
# 3. SPEED (unchanged)
#

def convert_speed_to_cms(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ["km/h", "kilometers per hour", "kilometres per hour"]:
        return value * 100000.0 / 3600.0
    elif unit in ["m/s", "meters per second", "metres per second"]:
        return value * 100.0
    return value


def normalize_speed(text: str) -> str:
    pattern = re.compile(
        r"""
        (\d+(?:\.\d+)?)
        (?:\s*
          (?:km/h|kilometers\ per\ hour|kilometres\ per\ hour|
             m/s|meters\ per\ second|metres\ per\ second|
             cm/s|centimeters\ per\ second|centimetres\ per\ second)
        )
        (?=[\s,;.!?]|$)
        """,
        flags=re.IGNORECASE | re.VERBOSE
    )

    def _repl(m):
        full_str = m.group(0)
        numeric_part = re.findall(r"\d+(?:\.\d+)?", full_str)[0]
        unit_part = re.sub(r"[^a-zA-Z/]+", "", full_str[len(numeric_part):].strip())
        val = float(numeric_part)
        val_cms = convert_speed_to_cms(val, unit_part)
        return f"{val_cms:.2f} cm/s"

    return pattern.sub(_repl, text)


#
# 4. ACCELERATION
#

def convert_accel_to_cms2(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ["m/s^2", "m/s2", "m/s²", "meters per second squared", "metres per second squared"]:
        return value * 100.0
    return value


def normalize_acceleration(text: str) -> str:
    """
    Convert recognized acceleration to cm/s^2.
    For example, "2 m/s^2" should become "200.00 cm/s^2".
    """
    pattern = re.compile(
        r"""
        (\d+(?:\.\d+)?)
        (?:\s*
           (?:m/s\^?2|m/s²
             |meters\ per\ second\ squared
             |metres\ per\ second\ squared
             |cm/s\^?2|cm/s²|centimeters\ per\ second\ squared|centimetres\ per\ second\ squared)
        )
        (?=[\s,;.!?]|$)
        """,
        flags=re.IGNORECASE | re.VERBOSE
    )

    def _repl(m):
        full_str = m.group(0)
        numeric_part = m.group(1)
        # IMPORTANT: Allow digits in the unit portion so "m/s^2" remains intact.
        leftover = full_str[len(numeric_part):].strip()
        unit_part = re.sub(r"[^a-zA-Z0-9/\^²]+", "", leftover)
        val = float(numeric_part)
        val_cms2 = convert_accel_to_cms2(val, unit_part)
        return f"{val_cms2:.2f} cm/s^2"

    return pattern.sub(_repl, text)


#
# 5. FULL PIPELINE
#

def normalize_all_units(text: str) -> str:
    """
    1) Convert angles -> rad
    2) Convert distances -> cm
    3) Convert speeds -> cm/s
    4) Convert accelerations -> cm/s^2
    """
    text = normalize_angles(text)
    text = normalize_distance(text)
    text = normalize_speed(text)
    text = normalize_acceleration(text)
    return text


#
# 6. TESTING (manual)
#

if __name__ == "__main__":
    sample_texts = [
        # Acceleration tests
        "We had 1 cm/s^2 motion.",
        "What if it's 0.1 m/s^2 or 0.05 cm/s^2?",
        "Accelerate at 2 m/s^2.",
        "Accelerate at 2m/s^2!",
        "Acceleration is 3.33 m/s² for test.",
        "Testing 9.81 m/s^2 exactly.",
        "What if it's 0.1 m/s^2 or 0.05 cm/s^2?",
        "Accelerate at 9.81 m/s^2, then turn 90 degrees and move 2.5m!",

        # Additional tests:
        "2 m/s^2",  # should become "200.00 cm/s^2"
    ]

    for text in sample_texts:
        out = normalize_all_units(text)
        print(f"Original:   {text}")
        print(f"Normalized: {out}\n")
