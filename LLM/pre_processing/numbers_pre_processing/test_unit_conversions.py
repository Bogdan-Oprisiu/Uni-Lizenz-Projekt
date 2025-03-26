# test_unit_conversions.py

import math
from unit_conversion import (
    convert_distance_to_cm, convert_speed_to_cms, convert_accel_to_cms2,
    convert_angle, normalize_distance, normalize_speed, normalize_acceleration,
    normalize_angles
)

def test_convert_distance_to_cm():
    # (value, unit, expected_cm)
    test_cases = [
        (1, "km", 100000.0),
        (0.5, "km", 50000.0),
        (2, "m", 200.0),
        (2.5, "m", 250.0),
        (100, "cm", 100.0),   # already cm => no change
        (42, "unknown", 42.0) # not recognized => no change
    ]
    for val, unit, expected in test_cases:
        result = convert_distance_to_cm(val, unit)
        assert math.isclose(result, expected, rel_tol=1e-6), \
            f"Expected {val} {unit} => {expected} cm, got {result} cm"
    print("test_convert_distance_to_cm() passed.")


def test_convert_speed_to_cms():
    test_cases = [
        (1, "km/h", 27.7778),       # approx
        (60, "km/h", 1666.6667),
        (1, "m/s", 100.0),
        (2.5, "m/s", 250.0),
        (99.9, "cm/s", 99.9),       # already cm/s
        (42, "whatever", 42.0)      # unknown => no change
    ]
    for val, unit, expected in test_cases:
        result = convert_speed_to_cms(val, unit)
        # We'll allow ~1e-3 tolerance for floating differences
        assert math.isclose(result, expected, rel_tol=1e-3), \
            f"Expected {val} {unit} => {expected} cm/s, got {result} cm/s"
    print("test_convert_speed_to_cms() passed.")


def test_convert_accel_to_cms2():
    test_cases = [
        (1, "m/s^2", 100.0),
        (2.5, "m/s²", 250.0),
        (9.81, "m/s^2", 981.0),
        (1, "cm/s^2", 1.0),     # already cm/s^2
        (0.05, "m/s2", 5.0),
        (42, "unknown", 42.0)
    ]
    for val, unit, expected in test_cases:
        result = convert_accel_to_cms2(val, unit)
        assert math.isclose(result, expected, rel_tol=1e-6), \
            f"Expected {val} {unit} => {expected} cm/s^2, got {result} cm/s^2"
    print("test_convert_accel_to_cms2() passed.")


def test_convert_angle():
    # convert_angle(value, is_degrees)
    # degrees -> radians
    # else just format to 4 decimals
    test_cases = [
        (0, True, 0.0),
        (180, True, math.pi),               # ~3.14159
        (45, True, math.pi/4),              # ~0.785398
        (1.57, False, 1.57)                 # rad as-is, but 4 decimals
    ]
    for val, is_deg, expected in test_cases:
        result_str = convert_angle(val, is_deg)
        # parse out the float
        fval = float(result_str.split()[0])
        assert math.isclose(fval, expected, rel_tol=1e-4), \
            f"For {val} deg={is_deg}, expected ~{expected}, got {result_str}"
    print("test_convert_angle() passed.")


###############################################################################
# 2) Quiet Tests for the Regex-Based Normalizers
#    - We test them on small strings that only contain the relevant pattern
###############################################################################

def test_normalize_distance():
    # We'll feed in small strings like "10 m", "2.5 km" and check the result
    cases = [
        ("10 m", "1000.00 cm"),
        ("2.5 km", "250000.00 cm"),
        ("42 cm", "42.00 cm"),
        ("1.23m", "123.00 cm"),  # no space
    ]
    for inp, expected in cases:
        result = normalize_distance(inp)
        # Because the function returns e.g. "123.00 cm"
        # We can assert exact string match:
        assert result == expected, f"For '{inp}', expected '{expected}', got '{result}'"
    print("test_normalize_distance() passed.")


def test_normalize_speed():
    cases = [
        ("10 m/s", "1000.00 cm/s"),
        ("10.0m/s", "1000.00 cm/s"),
        ("2 km/h", "55.56 cm/s"),     # approx
        ("99.9 cm/s", "99.90 cm/s"),  # already cm/s => just formatted
    ]
    for inp, expected in cases:
        result = normalize_speed(inp)
        # We'll do a substring check for approximate values in some cases
        # or if your code is exactly "55.56 cm/s" for 2 km/h:
        if "55.56" in expected:
            assert expected in result, f"For '{inp}', expected '{expected}', got '{result}'"
        else:
            assert (result == expected), f"For '{inp}', expected '{expected}', got '{result}'"
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
        # We'll do a partial check if we have 4 decimals
        # e.g. "1.5708 rad" might become "1.5708 rad" or "1.5708 rad"
        # We'll just check the prefix:
        if " rad" in expected:
            # get numeric from result
            got_num = float(result.split()[0])
            want_num = float(expected.split()[0])
            assert math.isclose(got_num, want_num, rel_tol=1e-4), \
                f"For '{inp}', expected ~'{expected}', got '{result}'"
        else:
            assert result == expected
    print("test_normalize_angles() passed.")


def run_all_tests():
    test_convert_distance_to_cm()
    test_convert_speed_to_cms()
    test_convert_accel_to_cms2()
    test_convert_angle()

    test_normalize_distance()
    test_normalize_speed()
    test_normalize_acceleration()
    test_normalize_angles()
    print("All unit conversion tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
