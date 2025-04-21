from itertools import chain
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. CORE UNITS
# --------------------------------------------------------------------------- #
DISTANCE = ["mm", "millimeter", "cm", "centimeter", "m", "meter",
            "km", "kilometer", "in", "inch", "ft", "foot", "feet"]

ANGLE = ["deg", "degree", "rad", "radian", "grad"]

LINEAR_VELOCITY = ["mm/s", "cm/s", "m/s", "km/h", "in/s", "ft/s"]
ANGULAR_VELOCITY = ["deg/s", "rad/s"]

ACCEL_LINEAR = ["mm/s^2", "cm/s^2", "m/s^2", "m/s^2"]
ACCEL_ANGULAR = ["deg/s^2", "deg/s^2", "rad/s^2", "rad/s^2"]

MISC = ["%", "percent"]


# --------------------------------------------------------------------------- #
# 2. NUMERIC LEXICON
# --------------------------------------------------------------------------- #
def generate_integers(start: int, stop: int, step: int = 1):
    """Inclusive range → strings."""
    return [str(i) for i in range(start, stop + 1, step)]


def generate_decimals(int_range, decimals=(0.25, 0.5, 0.75)):
    """Combine each integer with fractional suffixes → '10.25', …"""
    return [f"{i + frac:.2f}".rstrip("0").rstrip(".")
            for i in int_range
            for frac in decimals]


BASE_INTS = generate_integers(0, 360)  # full circle
BIG_INTS = generate_integers(400, 1000, 100)  # coarser tail
DECIMALS = generate_decimals(range(0, 51))  # 0.25 … 50.75
NEGATIVES = [f"-{n}" for n in generate_integers(1, 50)]

# --------------------------------------------------------------------------- #
# 3. FINAL ASSEMBLY
# --------------------------------------------------------------------------- #
units_and_numbers = list(chain(
    DISTANCE, ANGLE, LINEAR_VELOCITY, ANGULAR_VELOCITY,
    ACCEL_LINEAR, ACCEL_ANGULAR, MISC,
    BASE_INTS, BIG_INTS, DECIMALS, NEGATIVES
))

# Optional de‑dup & sort (tokenizer ignores order, but the file looks cleaner):
units_and_numbers = sorted(set(units_and_numbers), key=lambda x: (len(x), x))

# --------------------------------------------------------------------------- #
# 4. SAVE
# --------------------------------------------------------------------------- #
OUTPUT = Path("units_and_numbers.txt")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(units_and_numbers))
print(f"Wrote {len(units_and_numbers)} lines ➜ {OUTPUT}")
