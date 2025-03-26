# test_unit_conversions.py

import math

from numbers_processing import combined_numbers_pipeline
from unit_conversion import (
    convert_distance_to_cm, convert_speed_to_cms, convert_accel_to_cms2,
    convert_angle, normalize_distance, normalize_speed, normalize_acceleration,
    normalize_angles
)


def test_convert_distance_to_cm():
    test_cases = [
        (1, "km", 100000.0),
        (0.5, "km", 50000.0),
        (2, "m", 200.0),
        (2.5, "m", 250.0),
        (100, "cm", 100.0),
        (42, "unknown", 42.0)
    ]
    for val, unit, expected in test_cases:
        result = convert_distance_to_cm(val, unit)
        assert math.isclose(result, expected, rel_tol=1e-6), \
            f"Expected {val} {unit} => {expected} cm, got {result} cm"
    print("test_convert_distance_to_cm() passed.")


def test_convert_speed_to_cms():
    test_cases = [
        (1, "km/h", 27.7778),
        (60, "km/h", 1666.6667),
        (1, "m/s", 100.0),
        (2.5, "m/s", 250.0),
        (99.9, "cm/s", 99.9),
        (42, "whatever", 42.0)
    ]
    for val, unit, expected in test_cases:
        result = convert_speed_to_cms(val, unit)
        assert math.isclose(result, expected, rel_tol=1e-3), \
            f"Expected {val} {unit} => {expected} cm/s, got {result} cm/s"
    print("test_convert_speed_to_cms() passed.")


def test_convert_accel_to_cms2():
    test_cases = [
        (1, "m/s^2", 100.0),
        (2.5, "m/s²", 250.0),
        (9.81, "m/s^2", 981.0),
        (1, "cm/s^2", 1.0),
        (0.05, "m/s2", 5.0),
        (42, "unknown", 42.0)
    ]
    for val, unit, expected in test_cases:
        result = convert_accel_to_cms2(val, unit)
        assert math.isclose(result, expected, rel_tol=1e-6), \
            f"Expected {val} {unit} => {expected} cm/s^2, got {result} cm/s^2"
    print("test_convert_accel_to_cms2() passed.")


def test_convert_angle():
    test_cases = [
        (0, True, 0.0),
        (180, True, math.pi),
        (45, True, math.pi / 4),
        (1.57, False, 1.57)
    ]
    for val, is_deg, expected in test_cases:
        result_str = convert_angle(val, is_deg)
        fval = float(result_str.split()[0])
        assert math.isclose(fval, expected, rel_tol=1e-4), \
            f"For {val} deg={is_deg}, expected ~{expected}, got {result_str}"
    print("test_convert_angle() passed.")


def test_normalize_distance():
    cases = [
        ("10 m", "1000.00 cm"),
        ("2.5 km", "250000.00 cm"),
        ("42 cm", "42.00 cm"),
        ("1.23m", "123.00 cm")
    ]
    for inp, expected in cases:
        result = normalize_distance(inp)
        assert result == expected, f"For '{inp}', expected '{expected}', got '{result}'"
    print("test_normalize_distance() passed.")


def test_normalize_speed():
    cases = [
        ("10 m/s", "1000.00 cm/s"),
        ("10.0m/s", "1000.00 cm/s"),
        ("2 km/h", "55.56 cm/s"),
        ("99.9 cm/s", "99.90 cm/s"),
    ]
    for inp, expected in cases:
        result = normalize_speed(inp)
        assert result == expected, f"For '{inp}', expected '{expected}', got '{result}'"
    print("test_normalize_speed() passed.")


def test_normalize_acceleration():
    cases = [
        ("2 m/s^2", "200.00 cm/s^2"),
        ("2m/s^2", "200.00 cm/s^2"),
        ("1 cm/s^2", "1.00 cm/s^2"),
        ("0.05 m/s^2", "5.00 cm/s^2"),
        ("9.81 m/s²", "981.00 cm/s^2"),
    ]
    for inp, expected in cases:
        result = normalize_acceleration(inp)
        assert result == expected, f"For '{inp}', expected '{expected}', got '{result}'"
    print("test_normalize_acceleration() passed.")


def test_normalize_angles():
    cases = [
        ("45 deg", "0.7854 rad"),
        ("180 deg", "3.1416 rad"),
        ("1.57 rad", "1.5700 rad"),
        ("90 degrees", "1.5708 rad"),
        ("90°", "1.5708 rad"),
    ]
    for inp, expected in cases:
        result = normalize_angles(inp)
        got_num = float(result.split()[0])
        want_num = float(expected.split()[0])
        assert math.isclose(got_num, want_num, rel_tol=1e-4), \
            f"For '{inp}', expected ~'{expected}', got '{result}'"
    print("test_normalize_angles() passed.")


def test_multiple_units_in_one_line():
    cases = [
        (
            "Accelerate at 9.81 m/s^2, then rotate 180 deg and move 2.5m.",
            "Accelerate at 981.00 cm/s^2, then rotate 3.1416 rad and move 250.00 cm."
        ),
        (
            "He turned 45 deg, walked 3 kilometers, then drove 10 kilometers per hour.",
            "He turned 0.7854 rad, walked 300000.00 cm, then drove 277.78 cm/s."
        ),
    ]
    for inp, expected in cases:
        result = combined_numbers_pipeline(inp)
        assert result == expected, f"\nInput: {inp}\nExpected: {expected}\nGot: {result}"
    print("test_multiple_units_in_one_line() passed.")


def run_all_tests():
    test_convert_distance_to_cm()
    test_convert_speed_to_cms()
    test_convert_accel_to_cms2()
    test_convert_angle()
    test_normalize_distance()
    test_normalize_speed()
    test_normalize_acceleration()
    test_normalize_angles()
    test_multiple_units_in_one_line()
    print("All unit conversion tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
