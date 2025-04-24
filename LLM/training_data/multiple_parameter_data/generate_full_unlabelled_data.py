import math  # Added import for math.pi
import random
import re

# Assuming pre_processing.processing.full_text_processing exists and works as intended
try:
    from pre_processing.processing import full_text_processing
except ImportError:
    print("Warning: pre_processing.processing not found. Using basic text processing.")


    def full_text_processing(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# =============================================================================
# 1) Command Template Definitions (Same as Before, Possibly Reused)
# =============================================================================

FORWARD_SUBGROUPS = [
    [
        "{verb} {direction} quickly {distance}{unit}",
        "please {verb} {direction} briefly {distance}{unit}",
        "quickly {verb} {distance}{unit} {direction}"
    ],
    [
        "{verb} {direction} steadily {distance}{unit}",
        "proceed {direction} at a steady pace for {distance}{unit}",
        "{verb} {distance}{unit} {direction} in a controlled manner"
    ],
]

BACK_SUBGROUPS = [
    [
        "{verb} {direction} fast {distance}{unit}",
        "please {verb} {direction} swiftly {distance}{unit}",
        "quickly {verb} {distance}{unit} {direction}"
    ],
    [
        "{verb} {direction} slowly {distance}{unit}",
        "carefully {verb} {direction} for {distance}{unit}",
        "{verb} {distance}{unit} {direction} with caution"
    ],
]

SIDE_SUBGROUPS = [
    [
        "{verb} {direction} quickly {distance}{unit}",
        "please {verb} {direction} in a swift manner for {distance}{unit}",
        "quickly {verb} {distance}{unit} to the {direction}"
    ],
    [
        "{verb} {direction} steadily {distance}{unit}",
        "strafe {distance}{unit} {direction} at a consistent pace",
        "{verb} {distance}{unit} {direction} calmly"
    ],
]

ROTATE_SUBGROUPS = [
    [
        "{verb} {direction} quickly {angle}{unit}",
        "please {verb} a small angle {angle}{unit} to the {direction}",
        "spin {direction} briefly by {angle}{unit}"
    ],
    [
        "{verb} {direction} steadily by {angle}{unit}",
        "turn {angle}{unit} toward the {direction} in a controlled manner",
        "{verb} {direction} a larger angle of {angle}{unit}"
    ],
]

STOP_SUBGROUPS = [
    ["stop", "halt", "cease movement"],
    ["please stop", "bring it to a halt", "stop moving"],
]

# Verbs and Nouns (Same as labeled script)
FORWARD_VERBS = ["move", "go", "proceed", "advance", "head"]
FORWARD_NOUNS = ["forward", "ahead", "straight"]
BACK_VERBS = ["move", "go", "retreat", "reverse", "step"]
BACK_NOUNS = ["back", "backward", "reverse", "recede"]
SIDE_VERBS = ["move", "slide", "shift", "strafe"]
ROTATE_VERBS = ["rotate", "turn", "spin", "pivot"]
ROTATE_DIRECTIONS = ["left", "right", "port", "starboard"]

# Units
DISTANCE_UNIT = " cm"


# ANGLE_UNIT defined within rotate function

# =============================================================================
# 2) Generation Functions for Each Command Type (Unlabeled - Text Only)
# =============================================================================

def generate_forward_cmd():
    distance = round(random.uniform(1.0, 500.0), 2)
    use_accel = random.random() < 0.3
    accel_val = None
    accel_unit_text = ""

    template_base = random.choice(random.choice(FORWARD_SUBGROUPS))
    if use_accel:
        accel_val = round(random.uniform(0.1, 50.0), 2)  # cm/s^2
        accel_unit_text = " cm/s^2"
        template = template_base + " with acceleration {accel_val}{accel_unit_text}"
    else:
        template = template_base

    text_cmd = template.format(
        verb=random.choice(FORWARD_VERBS),
        direction=random.choice(FORWARD_NOUNS),
        distance=distance,
        unit=DISTANCE_UNIT,
        accel_val=accel_val if use_accel else None,
        accel_unit_text=accel_unit_text if use_accel else ""
    )
    return full_text_processing(text_cmd) + "."


def generate_back_cmd():
    distance = round(random.uniform(1.0, 500.0), 2)
    use_accel = random.random() < 0.3
    accel_val = None
    accel_unit_text = ""

    template_base = random.choice(random.choice(BACK_SUBGROUPS))
    if use_accel:
        accel_val = round(random.uniform(0.1, 50.0), 2)  # cm/s^2
        accel_unit_text = " cm/s^2"
        template = template_base + " with acceleration {accel_val}{accel_unit_text}"
    else:
        template = template_base

    text_cmd = template.format(
        verb=random.choice(BACK_VERBS),
        direction=random.choice(BACK_NOUNS),
        distance=distance,
        unit=DISTANCE_UNIT,
        accel_val=accel_val if use_accel else None,
        accel_unit_text=accel_unit_text if use_accel else ""
    )
    return full_text_processing(text_cmd) + "."


def generate_side_cmd(side):  # side is "left" or "right"
    distance = round(random.uniform(1.0, 500.0), 2)
    use_accel = random.random() < 0.3
    accel_val = None
    accel_unit_text = ""

    if side == "left":
        direction_choices = ["left", "port"]
    else:  # side == "right"
        direction_choices = ["right", "starboard"]

    template_base = random.choice(random.choice(SIDE_SUBGROUPS))
    if use_accel:
        accel_val = round(random.uniform(0.1, 50.0), 2)  # cm/s^2
        accel_unit_text = " cm/s^2"
        template = template_base + " with acceleration {accel_val}{accel_unit_text}"
    else:
        template = template_base

    text_direction = random.choice(direction_choices)
    text_cmd = template.format(
        verb=random.choice(SIDE_VERBS),
        direction=text_direction,
        distance=distance,
        unit=DISTANCE_UNIT,
        accel_val=accel_val if use_accel else None,
        accel_unit_text=accel_unit_text if use_accel else ""
    )
    # Add direction hint sometimes if strafing
    if random.random() < 0.1 and "strafe" not in text_cmd:
        text_cmd = f"strafing to the {text_direction} " + text_cmd

    return full_text_processing(text_cmd) + "."


def generate_rotate_cmd():
    # --- Angle (Use Radians Consistently) ---
    angle_rad = round(random.uniform(-math.pi, math.pi), 4)  # Generate and keep angle in radians

    # --- Direction ---
    direction = random.choice(ROTATE_DIRECTIONS)  # Direction needed for formatting

    # --- Acceleration (Standardized to rad/s^2) ---
    use_accel = random.random() < 0.3
    accel_val = None
    accel_unit_text = ""

    template_base = random.choice(random.choice(ROTATE_SUBGROUPS))

    if use_accel:
        # Always use rad/s^2 now
        accel_val = round(random.uniform(0.01, 3.0), 4)  # Generate rad/s^2 value
        accel_unit_text = " rad/s^2"  # Standard unit text
        template = template_base + " with angular acceleration {accel_val}{accel_unit_text}"
    else:
        template = template_base

    # --- Text Generation (Use Radians) ---
    # Use radians for angle in text
    angle_val_text = angle_rad
    angle_unit_text = " rad"

    # Format the text command
    text_cmd = template.format(
        verb=random.choice(ROTATE_VERBS),
        direction=direction,
        angle=angle_val_text,  # Use radian value
        unit=angle_unit_text,  # Use " rad" unit
        accel_val=accel_val if use_accel else None,
        accel_unit_text=accel_unit_text if use_accel else ""
    )
    return full_text_processing(text_cmd) + "."


def generate_stop_cmd():
    phrase = random.choice(random.choice(STOP_SUBGROUPS))
    return full_text_processing(phrase) + "."


def generate_unlabeled_command(cmd_type):
    if cmd_type == "forward":
        return generate_forward_cmd()
    elif cmd_type == "back":
        return generate_back_cmd()
    elif cmd_type in ["left", "right"]:
        return generate_side_cmd(cmd_type)
    elif cmd_type == "rotate":
        return generate_rotate_cmd()
    elif cmd_type == "stop":
        return generate_stop_cmd()
    else:
        # Default to stop if type is unrecognized
        return generate_stop_cmd()


# =============================================================================
# 4) Main Script - Unlabeled Data Generation
# =============================================================================

if __name__ == "__main__":
    num_samples = 100_000  # Adjust the number of samples as needed
    valid_command_types = ["forward", "back", "left", "right", "rotate", "stop"]

    all_commands = []
    print(f"Generating {num_samples} unlabeled samples...")
    for i in range(num_samples):
        cmd_t = random.choice(valid_command_types)
        text_cmd = generate_unlabeled_command(cmd_t)
        # Add line number prefix for potential tracking (optional)
        # all_commands.append(f"{i+1}: {text_cmd}")
        all_commands.append(text_cmd)  # Without line numbers
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")

    # Define output filename
    output_filename = "synthetic_unlabeled_robot_commands_with_accel.txt"

    print(f"Writing unlabeled commands to {output_filename}...")
    # Write them to a single text file
    with open(output_filename, "w", encoding="utf-8") as f_out:
        for text_cmd in all_commands:
            f_out.write(text_cmd + "\n")

    print("Data generation complete.")
