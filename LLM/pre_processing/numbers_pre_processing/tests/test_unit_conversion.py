"""
test_unit_conversion.py

Tests for the unit conversion functions defined in unit_conversion.py.
These tests verify that spelled-out units are first normalized to their abbreviations,
and that the conversion formulas for distance, speed, acceleration, and angles work as expected.
"""

import math
import re

from pre_processing.numbers_pre_processing.unit_conversion import (
    normalize_unit_names,
    convert_distance_to_cm,
    convert_speed_to_cms,
    convert_accel_to_cms2,
    convert_angle,
    normalize_all_units
)


###########################################
# Tests for Spelled-Out Unit Normalization
###########################################
def test_normalize_unit_names():
    # Test converting spelled-out distance units.
    assert normalize_unit_names("centimeters") == "cm"
    assert normalize_unit_names("Centimetres") == "cm"
    assert normalize_unit_names("meter") == "m"
    assert normalize_unit_names("Meters") == "m"
    assert normalize_unit_names("kilometers") == "km"
    # Test angle and time units.
    assert normalize_unit_names("degrees") == "deg"
    assert normalize_unit_names("seconds") == "s"


###########################################
# Tests for Distance Conversion
###########################################
def test_convert_distance_to_cm():
    # 1 km -> 100000 cm, 1 m -> 100 cm, 1 cm -> 1 cm.
    assert math.isclose(convert_distance_to_cm(1, "km"), 100000.0, rel_tol=1e-6)
    assert math.isclose(convert_distance_to_cm(1, "m"), 100.0, rel_tol=1e-6)
    assert math.isclose(convert_distance_to_cm(1, "cm"), 1.0, rel_tol=1e-6)


###########################################
# Tests for Speed Conversion
###########################################
def test_convert_speed_to_cms():
    # 1 km/h -> 27.77778 cm/s, 1 m/s -> 100 cm/s, 1 cm/s remains.
    assert math.isclose(convert_speed_to_cms(1, "km/h"), 100000.0 / 3600.0, rel_tol=1e-6)
    assert math.isclose(convert_speed_to_cms(1, "m/s"), 100.0, rel_tol=1e-6)
    assert math.isclose(convert_speed_to_cms(1, "cm/s"), 1.0, rel_tol=1e-6)


###########################################
# Tests for Acceleration Conversion
###########################################
def test_convert_accel_to_cms2():
    # 1 m/s^2 -> 100 cm/s^2; 1 cm/s^2 -> 1 cm/s^2.
    assert math.isclose(convert_accel_to_cms2(1, "m/s^2"), 100.0, rel_tol=1e-6)
    assert math.isclose(convert_accel_to_cms2(1, "m/s²"), 100.0, rel_tol=1e-6)
    assert math.isclose(convert_accel_to_cms2(1, "cm/s^2"), 1.0, rel_tol=1e-6)


###########################################
# Tests for Angle Conversion
###########################################
def test_convert_angle():
    # 45 deg -> 0.7854 rad; 90 deg -> 1.5708 rad; rad unchanged.
    assert math.isclose(convert_angle(45, "deg"), 45 * math.pi / 180.0, rel_tol=1e-4)
    assert math.isclose(convert_angle(90, "deg"), 90 * math.pi / 180.0, rel_tol=1e-4)
    assert math.isclose(convert_angle(1.57, "rad"), 1.57, rel_tol=1e-4)


###########################################
# End-to-End Tests for Normalization Pipeline
###########################################
def test_normalize_all_units():
    # For each test, the input uses abbreviated units.
    cases = [
        # Distances: "2 km" becomes "200000.00 cm", "3 m" becomes "300.00 cm", "42 cm" remains.
        ("2 km", "200000.00 cm"),
        ("3 m", "300.00 cm"),
        ("42 cm", "42.00 cm"),
        # Speed: "60 km/h" -> 60 * 100000/3600 = 1666.67 cm/s, "10 m/s" -> 1000.00 cm/s.
        ("60 km/h", f"{60 * 100000 / 3600:.2f} cm/s"),
        ("10 m/s", "1000.00 cm/s"),
        ("99.9 cm/s", "99.90 cm/s"),
        # Acceleration: "9.81 m/s^2" -> 981.00 cm/s^2, "1 cm/s^2" remains.
        ("9.81 m/s^2", "981.00 cm/s^2"),
        ("1 cm/s^2", "1.00 cm/s^2"),
        # Angles: "45 deg" -> approx 0.7854 rad, "90 deg" -> approx 1.5708 rad.
        ("45 deg", f"{45 * math.pi / 180.0:.4f} rad"),
        ("90 deg", f"{90 * math.pi / 180.0:.4f} rad"),
        # Compound sentence
        (
            "Move 2 km and accelerate 9.81 m/s^2, then turn 45 deg.",
            f"Move 200000.00 cm and accelerate 981.00 cm/s^2, then turn {45 * math.pi / 180.0:.4f} rad."
        )
    ]
    for inp, expected in cases:
        result = normalize_all_units(inp)
        # Remove extra spaces for comparison.
        result_clean = re.sub(r"\s+", " ", result).strip()
        expected_clean = re.sub(r"\s+", " ", expected).strip()
        assert result_clean == expected_clean, f"\nInput: {inp}\nExpected: {expected_clean}\nGot: {result_clean}"


def run_all_tests():
    test_normalize_unit_names()
    test_convert_distance_to_cm()
    test_convert_speed_to_cms()
    test_convert_accel_to_cms2()
    test_convert_angle()
    test_normalize_all_units()
    print("All unit conversion tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
