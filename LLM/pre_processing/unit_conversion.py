# unit_conversion.py

import math
import re

#
# 1. ANGLES
#

def convert_angle(value: float, is_degrees: bool) -> str:
    """
    Convert a numeric angle value to radians if it's degrees;
    if already radians, just format to 4 decimals.
    """
    if is_degrees:
        # Convert degrees -> radians
        value_rad = value * math.pi / 180.0
        return f"{value_rad:.4f} rad"
    else:
        # Already in radians
        return f"{value:.4f} rad"


def normalize_angles(text: str) -> str:
    """
    Finds occurrences of a number followed by an angle unit and converts:
      - degrees → radians
      - existing radians → rad (just ensuring uniform format)

    We handle patterns like:
       "45 degrees", "45 deg", "45°"
       "1.57 radians", "1.57 rad", etc.
    """
    pattern = re.compile(
        # We have two alternations:
        #  (1)  (?P<val>\d+(?:\.\d+)?)(?:\s*°|\s*(?:degrees|degree|deg))(?P<deg>\b)?
        #       This captures a number + "degrees/deg/°"
        #  (2)  (?P<val2>\d+(?:\.\d+)?)(?:\s*(?:radians|radian|rad))(?P<rad>\b)?
        #       This captures a number + "radians/rad"
        r"(?P<val>\d+(?:\.\d+)?)(?:\s*°|\s*(?:degrees|degree|deg))(?P<deg>\b)?"
        r"|(?P<val2>\d+(?:\.\d+)?)(?:\s*(?:radians|radian|rad))(?P<rad>\b)?",
        flags=re.IGNORECASE
    )

    def _repl(m):
        if m.group("val") is not None:
            # This means we matched degrees or °
            val_str = m.group("val")
            val = float(val_str)
            return convert_angle(val, is_degrees=True)
        else:
            # This means we matched radians
            val_str = m.group("val2")
            val = float(val_str)
            return convert_angle(val, is_degrees=False)

    return pattern.sub(_repl, text)


#
# 2. DISTANCE
#

def convert_distance_to_cm(value: float, unit: str) -> float:
    """
    Convert various distance units to centimeters.
    """
    unit = unit.lower()
    if unit in ["km", "kilometers", "kilometres", "kilometer"]:
        return value * 100000.0
    elif unit in ["m", "meters", "metres", "meter"]:
        return value * 100.0
    # If already cm or unknown, just return as cm
    return value

def normalize_distance(text: str) -> str:
    """
    Convert recognized distance values to cm, output as e.g. "123.45 cm".
    Handles cases like "10 km", "10km", "5 meters", "5m", etc.
    """
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)(?:\s*(?:km|kilometers|kilometres|kilometer|m|meters|metres|meter|cm|centimeters|centimetres|centimeter)|(?=km|m|cm))",
        flags=re.IGNORECASE
    )

    def _repl(m):
        full_str = m.group(0)
        # separate the numeric part from the letters
        numeric_part = re.findall(r"\d+(?:\.\d+)?", full_str)[0]
        unit_part = re.sub(r"[^a-zA-Z]+", "", full_str)  # get the letters only

        val = float(numeric_part)
        val_cm = convert_distance_to_cm(val, unit_part)
        return f"{val_cm:.2f} cm"

    return pattern.sub(_repl, text)


#
# 3. SPEED
#

def convert_speed_to_cms(value: float, unit: str) -> float:
    """
    Convert speed to cm/s.
    """
    unit = unit.lower()
    if unit in ["km/h", "kilometers per hour", "kilometres per hour"]:
        # multiply by 100000/3600 ≈ 27.7778
        return value * 100000.0 / 3600.0
    elif unit in ["m/s", "meters per second", "metres per second"]:
        return value * 100.0
    # else if already cm/s or unknown
    return value

def normalize_speed(text: str) -> str:
    """
    Convert recognized speed values to cm/s, e.g. "10 m/s" -> "1000.00 cm/s".
    Also handles no-space: "10m/s".
    """
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)(?:\s*(?:km/h|kilometers per hour|kilometres per hour|m/s|meters per second|metres per second|cm/s|centimeters per second|centimetres per second)|(?=km/h|m/s|cm/s))",
        flags=re.IGNORECASE
    )

    def _repl(m):
        full_str = m.group(0)
        numeric_part = re.findall(r"\d+(?:\.\d+)?", full_str)[0]
        # keep letters and '/' only to identify the unit
        unit_part = re.sub(r"[^a-zA-Z/]+", "", full_str)

        val = float(numeric_part)
        val_cms = convert_speed_to_cms(val, unit_part)
        return f"{val_cms:.2f} cm/s"

    return pattern.sub(_repl, text)


#
# 4. ACCELERATION
#

def convert_accel_to_cms2(value: float, unit: str) -> float:
    """
    Convert acceleration to cm/s^2.
    """
    unit = unit.lower()
    # Common forms: "m/s^2", "m/s²", "m/s2", "meters per second squared"
    # We'll unify them all to numeric conversion
    if unit in [
        "m/s^2", "m/s2", "m/s²",
        "meters per second squared", "metres per second squared"
    ]:
        return value * 100.0
    # if already cm/s^2 or unknown, keep it
    return value

def normalize_acceleration(text: str) -> str:
    """
    Convert recognized acceleration units to cm/s^2,
    e.g. "2 m/s^2" -> "200.00 cm/s^2".
    Handles no-space scenario: "2m/s^2".
    """
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)(?:\s*(?:m/s\^?2|m/s²|meters per second squared|metres per second squared|cm/s\^?2|cm/s²|centimeters per second squared|centimetres per second squared)|(?=m/s|cm/s))",
        flags=re.IGNORECASE
    )

    def _repl(m):
        full_str = m.group(0)
        numeric_part = re.findall(r"\d+(?:\.\d+)?", full_str)[0]
        # keep letters, '/', '^', and '²'
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
# 6. TESTING
#

if __name__ == "__main__":
    sample_texts = [
        "Turn right 90 degrees.",
        "Turn right 90°.",
        "Rotate left 1.57 radians.",
        "Move 5 meters forward.",
        "Jump 10.0m high.",      # no space
        "Run at 10 m/s.",
        "Accelerate at 2m/s^2",  # no space
        "Drive 2 kilometers at 60km/h.",
        "Move 1.5km ahead.",
        "Turn 45° to face north.",
        "Accelerate at 9.81 m/s²."
    ]

    for text in sample_texts:
        normalized = normalize_all_units(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}\n")
