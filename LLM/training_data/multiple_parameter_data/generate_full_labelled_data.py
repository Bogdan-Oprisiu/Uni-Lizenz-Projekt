import json
import random
import re

from pre_processing.processing import full_text_processing

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
        "move {direction} calmly {distance}{unit}",
        "{verb} {distance}{unit} {direction} at a consistent pace"
    ],
]

ROTATE_SUBGROUPS = [
    [
        "{verb} {direction} briefly by {angle}{unit}",
        "spin {direction} quickly {angle}{unit}",
        "please {verb} a small angle {angle}{unit} to the {direction}"
    ],
    [
        "turn {direction} steadily by {angle}{unit}",
        "rotate {direction} a larger angle of {angle}{unit}",
        "{verb} {angle}{unit} toward the {direction} in a controlled manner"
    ],
]

STOP_SUBGROUPS = [
    [
        "stop",
        "halt"
    ],
    [
        "please stop",
        "cease movement"
    ],
]

# =============================================================================
# 2) Synonyms, Helper Functions, and Acceleration Handling
# =============================================================================

directions_forward = ["forward", "ahead", "advance", "proceed", "go ahead"]
directions_back = ["back", "reverse", "backward", "retreat", "recede"]
side_directions = ["left", "right", "port", "starboard"]
rotate_directions = ["left", "right", "port", "starboard"]

verbs_forward = ["move", "go", "advance", "proceed", "head"]
verbs_back = ["move", "go", "reverse", "retreat", "step", "recede"]
verbs_side = ["strafe", "move", "slide", "shift"]
verbs_rotate = ["rotate", "turn", "spin", "pivot"]


def random_distance():
    """Generate a distance as a float between 10 and 500, formatted with one decimal."""
    return f"{random.uniform(10, 500):.1f}"


def random_angle_degrees():
    """
    Generate a random angle in degrees (instead of radians, since the JSON schema uses degrees).
    Range: -180 to 180, formatted with 1 decimal.
    """
    angle_deg = random.uniform(-180, 180)
    return f"{angle_deg:.1f}"


def random_acceleration_linear():
    """Generate a random linear acceleration in cm/s^2, 0 to 50 range."""
    return f"{random.uniform(0, 50):.2f}"


def random_acceleration_angular():
    """Generate a random angular acceleration in deg/s^2, 0 to 60 range."""
    return f"{random.uniform(0, 60):.2f}"


def fill_template(template, components):
    """Replace placeholders in the template with values from components."""
    return template.format(**components)


def choose_template(subgroup_list):
    """Randomly choose a subgroup, then pick one template from that subgroup."""
    subgroup = random.choice(subgroup_list)
    return random.choice(subgroup)


def maybe_add_acceleration_phrase(cmd_text, is_linear=True):
    """
    With 50% chance, append an acceleration phrase to the textual command.
    If it's linear, e.g. "with acceleration 12.34 cm/s^2".
    If it's angular, e.g. "with angular acceleration 25.67 deg/s^2".
    """
    if random.random() < 0.5:  # 50% chance to include acceleration phrase
        if is_linear:
            accel_val = random_acceleration_linear()
            return cmd_text + f" with acceleration {accel_val} cm/s^2"
        else:
            accel_val = random_acceleration_angular()
            return cmd_text + f" with angular acceleration {accel_val} deg/s^2"
    return cmd_text


def extract_acceleration(cmd_text, is_linear=True):
    """
    Extract the acceleration value from the command text using regex.
    Returns a float if found, otherwise None.
    """
    if is_linear:
        match = re.search(r"acceleration\s+(\d+(\.\d+)?)\s*cm/s\^?2", cmd_text)
    else:
        match = re.search(r"angular acceleration\s+(\d+(\.\d+)?)\s*deg/s\^?2", cmd_text)
    if match:
        return float(match.group(1))
    return None


# =============================================================================
# 3) Labeled Generation Functions (Returning Text + JSON)
# =============================================================================

def generate_forward_labeled():
    distance = random_distance()
    template = choose_template(FORWARD_SUBGROUPS)
    components = {
        "verb": random.choice(verbs_forward),
        "direction": random.choice(directions_forward),
        "distance": distance,
        "unit": "cm"
    }
    command_text = fill_template(template, components)
    command_text = maybe_add_acceleration_phrase(command_text, is_linear=True)
    command_text = full_text_processing(command_text) + "."

    accel_val = extract_acceleration(command_text, is_linear=True)

    expected_json = {
        "command": "forward",
        "parameters": {
            "distance": float(distance)
        }
    }
    if accel_val is not None:
        expected_json["parameters"]["acceleration"] = accel_val

    return command_text, json.dumps(expected_json)


def generate_back_labeled():
    distance = random_distance()
    template = choose_template(BACK_SUBGROUPS)
    components = {
        "verb": random.choice(verbs_back),
        "direction": random.choice(directions_back),
        "distance": distance,
        "unit": "cm"
    }
    command_text = fill_template(template, components)
    command_text = maybe_add_acceleration_phrase(command_text, is_linear=True)
    command_text = full_text_processing(command_text) + "."

    accel_val = extract_acceleration(command_text, is_linear=True)

    expected_json = {
        "command": "back",
        "parameters": {
            "distance": float(distance)
        }
    }
    if accel_val is not None:
        expected_json["parameters"]["acceleration"] = accel_val

    return command_text, json.dumps(expected_json)


def generate_side_labeled(direction):
    distance = random_distance()
    template = choose_template(SIDE_SUBGROUPS)
    components = {
        "verb": random.choice(verbs_side),
        "direction": direction,  # "left" or "right"
        "distance": distance,
        "unit": "cm"
    }
    command_text = fill_template(template, components)
    command_text = maybe_add_acceleration_phrase(command_text, is_linear=True)
    command_text = full_text_processing(command_text) + "."

    accel_val = extract_acceleration(command_text, is_linear=True)

    expected_json = {
        "command": direction,  # either "left" or "right"
        "parameters": {
            "distance": float(distance)
        }
    }
    if accel_val is not None:
        expected_json["parameters"]["acceleration"] = accel_val

    return command_text, json.dumps(expected_json)


def generate_rotate_labeled():
    angle_str = random_angle_degrees()
    template = choose_template(ROTATE_SUBGROUPS)
    chosen_direction = random.choice(["left", "right"])  # ensure valid schema direction
    components = {
        "verb": random.choice(verbs_rotate),
        "direction": chosen_direction,
        "angle": angle_str,
        "unit": "deg"
    }
    command_text = fill_template(template, components)
    command_text = maybe_add_acceleration_phrase(command_text, is_linear=False)
    command_text = full_text_processing(command_text) + "."

    accel_val = extract_acceleration(command_text, is_linear=False)

    expected_json = {
        "command": "rotate",
        "parameters": {
            "angle": float(angle_str),
            "direction": chosen_direction
        }
    }
    if accel_val is not None:
        expected_json["parameters"]["acceleration"] = accel_val

    return command_text, json.dumps(expected_json)


def generate_stop_labeled():
    template_group = random.choice(STOP_SUBGROUPS)
    phrase = random.choice(template_group)
    command_text = full_text_processing(phrase) + "."

    expected_json = {
        "command": "stop",
        "parameters": {}
    }
    return command_text, json.dumps(expected_json)


def generate_labeled_command(command_type):
    """Dispatch function to produce text + JSON for a single command."""
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
        return generate_stop_labeled()


# =============================================================================
# 4) Main Script - Labeled Data Generation
# =============================================================================

if __name__ == "__main__":
    num_samples = 10_000
    valid_command_types = ["forward", "back", "left", "right", "rotate", "stop"]

    commands_text = []
    commands_json = []

    for _ in range(num_samples):
        ctype = random.choice(valid_command_types)
        txt, js = generate_labeled_command(ctype)
        commands_text.append(txt)
        commands_json.append(js)

    txt_filename = "synthetic_labeled_robot_commands_with_accel.txt"
    json_filename = "synthetic_labeled_robot_commands_with_accel_json.txt"

    with open(txt_filename, "w", encoding="utf-8") as f_txt, \
            open(json_filename, "w", encoding="utf-8") as f_js:
        for text_cmd, json_cmd in zip(commands_text, commands_json):
            f_txt.write(text_cmd + "\n")
            f_js.write(json_cmd + "\n")

    print(f"Labeled data with acceleration generated:\n"
          f" - {txt_filename}\n"
          f" - {json_filename}")
