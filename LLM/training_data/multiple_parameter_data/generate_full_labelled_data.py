import json
import math  # Added import for math.pi and math.degrees
import random
import re

# Assuming pre_processing.processing.full_text_processing exists and works as intended
# If not, replace with desired text normalization logic (e.g., lowercasing, removing extra spaces)
try:
    from pre_processing.processing import full_text_processing
except ImportError:
    print("Warning: pre_processing.processing not found. Using basic text processing.")


    def full_text_processing(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# =============================================================================
# 1) Command Template Definitions (with optional acceleration placeholders)
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

# Verbs and Nouns
FORWARD_VERBS = ["move", "go", "proceed", "advance", "head"]
FORWARD_NOUNS = ["forward", "ahead", "straight"]

BACK_VERBS = ["move", "go", "retreat", "reverse", "step"]
BACK_NOUNS = ["back", "backward", "reverse", "recede"]

SIDE_VERBS = ["move", "slide", "shift", "strafe"]
# Directions handled by argument

ROTATE_VERBS = ["rotate", "turn", "spin", "pivot"]
ROTATE_DIRECTIONS = ["left", "right", "port", "starboard"]  # Port=Left, Starboard=Right

# Units
DISTANCE_UNIT = " cm"


# ANGLE_UNIT defined within rotate function

# =============================================================================
# 2) Generation Functions for Each Command Type (Labeled)
# =============================================================================

def generate_forward_labeled():
    # --- Parameters ---
    distance = round(random.uniform(1.0, 500.0), 2)
    use_accel = random.random() < 0.3  # 30% chance to include acceleration
    accel_val = None
    accel_unit_text = ""

    json_params = {"distance": distance}
    json_command = "forward"

    if use_accel:
        accel_val = round(random.uniform(0.1, 50.0), 2)  # cm/s^2
        accel_unit_text = " cm/s^2"
        json_params["acceleration"] = accel_val

    # --- Text Generation ---
    template_base = random.choice(random.choice(FORWARD_SUBGROUPS))
    if use_accel:
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
    text_cmd = full_text_processing(text_cmd) + "."

    # --- JSON Generation ---
    json_data = {"command": json_command, "parameters": json_params}
    json_string = json.dumps(json_data)

    return text_cmd, json_string


def generate_back_labeled():
    # --- Parameters ---
    distance = round(random.uniform(1.0, 500.0), 2)
    use_accel = random.random() < 0.3
    accel_val = None
    accel_unit_text = ""

    json_params = {"distance": distance}
    json_command = "back"

    if use_accel:
        accel_val = round(random.uniform(0.1, 50.0), 2)  # cm/s^2
        accel_unit_text = " cm/s^2"
        json_params["acceleration"] = accel_val

    # --- Text Generation ---
    template_base = random.choice(random.choice(BACK_SUBGROUPS))
    if use_accel:
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
    text_cmd = full_text_processing(text_cmd) + "."

    # --- JSON Generation ---
    json_data = {"command": json_command, "parameters": json_params}
    json_string = json.dumps(json_data)

    return text_cmd, json_string


def generate_side_labeled(side):  # side is "left" or "right"
    # --- Parameters ---
    distance = round(random.uniform(1.0, 500.0), 2)
    use_accel = random.random() < 0.3
    accel_val = None
    accel_unit_text = ""

    json_params = {"distance": distance}
    # Determine direction based on side for strafing etc.
    if side == "left":
        direction_choices = ["left", "port"]
        json_command = "left"
        if random.random() < 0.2:  # Occasionally specify direction explicitly for strafe
            json_params["direction"] = random.choice(direction_choices)
            json_command = "strafe"
    else:  # side == "right"
        direction_choices = ["right", "starboard"]
        json_command = "right"
        if random.random() < 0.2:
            json_params["direction"] = random.choice(direction_choices)
            json_command = "strafe"

    if use_accel:
        accel_val = round(random.uniform(0.1, 50.0), 2)  # cm/s^2
        accel_unit_text = " cm/s^2"
        json_params["acceleration"] = accel_val

    # --- Text Generation ---
    template_base = random.choice(random.choice(SIDE_SUBGROUPS))
    if use_accel:
        template = template_base + " with acceleration {accel_val}{accel_unit_text}"
    else:
        template = template_base

    # Direction text might differ slightly from json_command
    text_direction = random.choice(direction_choices)

    text_cmd = template.format(
        verb=random.choice(SIDE_VERBS),
        direction=text_direction,
        distance=distance,
        unit=DISTANCE_UNIT,
        accel_val=accel_val if use_accel else None,
        accel_unit_text=accel_unit_text if use_accel else ""
    )
    # Sometimes add direction explicitly for strafing text
    if json_command == "strafe" and "direction" in json_params:
        if random.random() < 0.5 and "strafe" not in text_cmd:  # Add direction hint if not obvious
            text_cmd = f"strafing to the {json_params['direction']} " + text_cmd

    text_cmd = full_text_processing(text_cmd) + "."

    # --- JSON Generation ---
    json_data = {"command": json_command, "parameters": json_params}
    json_string = json.dumps(json_data)

    return text_cmd, json_string


def generate_rotate_labeled():
    # --- Angle (Use Radians Consistently) ---
    angle_rad = round(random.uniform(-math.pi, math.pi), 4)  # Generate and keep angle in radians

    # --- Direction ---
    direction = random.choice(ROTATE_DIRECTIONS)

    # --- JSON Base (Use Radians) ---
    # Use radians for angle in JSON
    json_params = {"angle": angle_rad, "direction": direction}
    json_command = "rotate"

    # --- Acceleration (Standardized to rad/s^2) ---
    use_accel = random.random() < 0.3
    accel_val = None
    accel_unit_text = ""

    if use_accel:
        # Always use rad/s^2 now
        accel_val = round(random.uniform(0.01, 3.0), 4)  # Generate rad/s^2 value
        accel_unit_text = " rad/s^2"  # Standard unit text
        # *** FIX: Always add acceleration (as rad/s^2 value) to JSON if use_accel is True ***
        json_params["acceleration"] = accel_val

    # --- Text Generation (Use Radians) ---
    template_base = random.choice(random.choice(ROTATE_SUBGROUPS))

    # Use radians for angle in text
    angle_val_text = angle_rad
    angle_unit_text = " rad"

    # Add acceleration part to template if needed
    if use_accel:
        template = template_base + " with angular acceleration {accel_val}{accel_unit_text}"
    else:
        template = template_base

    # Format the text command
    text_cmd = template.format(
        verb=random.choice(ROTATE_VERBS),
        direction=direction,
        angle=angle_val_text,  # Use radian value
        unit=angle_unit_text,  # Use " rad" unit
        accel_val=accel_val if use_accel else None,
        accel_unit_text=accel_unit_text if use_accel else ""
    )
    text_cmd = full_text_processing(text_cmd) + "."

    # --- JSON Final ---
    json_data = {"command": json_command, "parameters": json_params}
    json_string = json.dumps(json_data)

    return text_cmd, json_string


def generate_stop_labeled():
    # --- Text Generation ---
    text_cmd = random.choice(random.choice(STOP_SUBGROUPS))
    text_cmd = full_text_processing(text_cmd) + "."

    # --- JSON Generation ---
    json_params = {}
    json_command = "stop"
    json_data = {"command": json_command, "parameters": json_params}
    json_string = json.dumps(json_data)

    return text_cmd, json_string


# =============================================================================
# 3) Master Function to Select Command Type
# =============================================================================

def generate_labeled_command(command_type):
    if command_type == "forward":
        return generate_forward_labeled()
    elif command_type == "back":
        return generate_back_labeled()
    elif command_type in ["left", "right"]:
        return generate_side_labeled(command_type)
    elif command_type == "rotate":
        return generate_rotate_labeled()
    elif command_type == "stop":
        return generate_stop_labeled()
    else:
        # Default to stop command if type is unrecognized
        return generate_stop_labeled()


# =============================================================================
# 4) Main Script - Labeled Data Generation
# =============================================================================

if __name__ == "__main__":
    num_samples = 10_000  # Adjust number of samples as needed
    valid_command_types = ["forward", "back", "left", "right", "rotate", "stop"]

    commands_text = []
    commands_json = []

    print(f"Generating {num_samples} labeled samples...")
    for i in range(num_samples):
        ctype = random.choice(valid_command_types)
        txt, js = generate_labeled_command(ctype)
        commands_text.append(txt)
        commands_json.append(js)
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")

    # Define output filenames
    txt_filename = "synthetic_labeled_robot_commands_with_accel.txt"
    json_filename = "synthetic_labeled_robot_commands_with_accel_json.txt"

    print(f"Writing text commands to {txt_filename}...")
    with open(txt_filename, "w", encoding="utf-8") as f_txt:
        for text_cmd in commands_text:
            f_txt.write(text_cmd + "\n")

    print(f"Writing JSON commands to {json_filename}...")
    with open(json_filename, "w", encoding="utf-8") as f_js:
        for json_cmd in commands_json:
            f_js.write(json_cmd + "\n")

    print("Data generation complete.")
