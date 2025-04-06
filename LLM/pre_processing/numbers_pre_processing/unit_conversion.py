"""
unit_conversion.py

This file converts short-form or spelled-out units to:
  - Distance: cm
  - Speed: cm/s
  - Acceleration: cm/s^2
  - Angles: rad

It provides:
  1) A function to normalize spelled-out units to short abbreviations
     (e.g., "centimeters" -> "cm").
  2) Individual conversion functions for distance, speed, acceleration, and angle.
  3) A 'normalize_all_units' function that scans a string for numeric+unit patterns,
     applies the conversion, and returns the fully normalized result.

Example:
  Input:  "9.81 m/s^2, 10 m/s, 2 km/h."
  Output: "981.00 cm/s^2, 1000.00 cm/s, 55.56 cm/s."
"""

import math
import re


############################
# 1) Spell-out to abbreviations
############################

def normalize_unit_names(unit_str: str) -> str:
    """
    Convert spelled-out or variant unit into a short abbreviation.
    E.g., "centimeters" -> "cm", "meters per second squared" -> "m/s^2", etc.

    If the unit is unknown, return it unchanged.
    """
    # Remove punctuation and convert to lowercase for matching
    unit_str_clean = re.sub(r"[.,;!?]+", "", unit_str).lower().strip()

    single_word_map = {
        "centimeters": "cm", "centimetres": "cm", "centimeter": "cm",
        "meters": "m", "metres": "m", "meter": "m",
        "kilometers": "km", "kilometres": "km", "kilometer": "km",
        "degrees": "deg", "degree": "deg",
        "radians": "rad", "radian": "rad",
        "cm": "cm", "m": "m", "km": "km", "deg": "deg", "rad": "rad",
        # Speed
        "cm/s": "cm/s", "m/s": "m/s", "km/h": "km/h",
        "meters per second": "m/s", "metres per second": "m/s",
        "centimeters per second": "cm/s", "centimetres per second": "cm/s",
        "kilometers per hour": "km/h", "kilometres per hour": "km/h",
        # Acceleration
        "m/s^2": "m/s^2", "m/s²": "m/s^2",
        "meters per second squared": "m/s^2", "metres per second squared": "m/s^2",
        "cm/s^2": "cm/s^2", "cm/s²": "cm/s^2",
        "centimeters per second squared": "cm/s^2", "centimetres per second squared": "cm/s^2",
        # Time
        "seconds": "s"
    }

    if unit_str_clean in single_word_map:
        return single_word_map[unit_str_clean]
    return unit_str  # Unknown or already short form


############################
# 2) Individual Conversion Functions
############################

def convert_distance_to_cm(value: float, unit: str) -> float:
    """Convert distance in (km, m, cm) to cm."""
    u = unit.lower()
    if u == "km":
        return value * 100000.0
    elif u == "m":
        return value * 100.0
    return value


def convert_speed_to_cms(value: float, unit: str) -> float:
    """Convert speed in (km/h, m/s, cm/s) to cm/s."""
    u = unit.lower()
    if u == "km/h":
        return value * 100000.0 / 3600.0  # 1 km/h = 27.7778 cm/s
    elif u == "m/s":
        return value * 100.0
    return value


def convert_accel_to_cms2(value: float, unit: str) -> float:
    """Convert acceleration in (m/s^2, cm/s^2) to cm/s^2."""
    u = unit.lower()
    if u in ["m/s^2", "m/s²"]:
        return value * 100.0
    return value


def convert_angle(value: float, unit: str) -> float:
    """Convert angle in deg to rad, else leave if already rad."""
    u = unit.lower()
    if u == "deg":
        return value * math.pi / 180.0
    return value


############################
# 3) Full Normalization Pipeline
############################

def normalize_all_units(text: str) -> str:
    """
    1) Find numeric expressions with a single token recognized unit (including spelled-out).
    2) Normalize spelled-out units -> short form.
    3) Convert to a final standard (distance=cm, speed=cm/s, accel=cm/s^2, angle=rad).
    4) Format numeric: 2 decimals for distance/speed/accel, or 4 for angles.
    5) Retain trailing punctuation after the expression.
    """

    # This pattern captures exactly one token for the unit (no multi-word):
    #  group(1): numeric value
    #  group(2): unit token (stops at next whitespace or punctuation)
    #  group(3): optional punctuation
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)(?:\s+)([^\s,.;!?]+)([.,;!?])?"
    )

    def replacer(m):
        value_str = m.group(1)  # e.g. "9.81"
        raw_unit = m.group(2)  # e.g. "m/s^2"
        trailing = m.group(3) if m.group(3) else ""

        short_unit = normalize_unit_names(raw_unit)
        val = float(value_str)

        # Identify the converter
        if short_unit in ["cm", "m", "km"]:
            cm_value = convert_distance_to_cm(val, short_unit)
            return f"{cm_value:.2f} cm{trailing}"

        if short_unit in ["cm/s", "m/s", "km/h"]:
            cms_value = convert_speed_to_cms(val, short_unit)
            return f"{cms_value:.2f} cm/s{trailing}"

        if short_unit in ["cm/s^2", "m/s^2"]:
            cms2_value = convert_accel_to_cms2(val, short_unit)
            return f"{cms2_value:.2f} cm/s^2{trailing}"

        if short_unit in ["deg", "rad"]:
            rad_val = convert_angle(val, short_unit)
            return f"{rad_val:.4f} rad{trailing}"

        # If unknown or spelled out multi-words not handled, return original
        return m.group(0)

    return pattern.sub(replacer, text)


if __name__ == "__main__":
    lines = [
        "9.81 m/s^2, 10 m/s, 2 km/h.",
        "Move 2 km and accelerate 9.81 m/s^2, then turn 45 deg!",
        "Distance: 3 m, angle = 180 deg.",
        "Just 42 cm/s^2??"
    ]
    for line in lines:
        converted = normalize_all_units(line)
        print("Original: ", line)
        print("Converted:", converted)
        print()
