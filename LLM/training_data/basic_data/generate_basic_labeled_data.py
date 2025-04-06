import json
import math
import random

from pre_processing.processing import full_text_processing

# =========================================================================================
#    1. Subgroup Template Definitions (same as before)
# =========================================================================================

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

STRAFING_SUBGROUPS = [
    [
        "{verb} {direction} swiftly {distance}{unit}",
        "strafing to the {direction} quickly {distance}{unit}",
        "please {verb} {distance}{unit} in the {direction} direction fast"
    ],
    [
        "{verb} {direction} steadily {distance}{unit}",
        "strafing to the {direction} carefully {distance}{unit}",
        "{verb} {distance}{unit} {direction} at a consistent rate"
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

# =========================================================================================
#    2. Synonyms and Helper Functions (same as before)
# =========================================================================================

directions_forward = ["forward", "ahead", "advance", "proceed", "go ahead"]
directions_back = ["back", "reverse", "backward", "retreat", "recede"]
rotate_directions = ["left", "right", "port", "starboard"]
side_directions = ["left", "right", "port", "starboard"]

verbs_forward = ["move", "go", "advance", "proceed", "head"]
verbs_back = ["move", "go", "reverse", "retreat", "step", "recede"]
verbs_side = ["strafe", "move", "slide", "shift"]
verbs_rotate = ["rotate", "turn", "spin", "pivot"]


def random_distance():
    """Generate a distance as a float between 10 and 500, formatted with one decimal."""
    return f"{random.uniform(10, 500):.1f}"


def random_angle():
    """Generate a random angle in radians between -pi/2 and pi/2, formatted to 4 decimals."""
    angle_rad = random.uniform(-math.pi / 2, math.pi / 2)
    return f"{angle_rad:.4f}"


def fill_template(template, components):
    """Replace placeholders in the template with values from components."""
    return template.format(**components)


def choose_template(subgroup_list):
    """Randomly choose a subgroup, then a template from that subgroup."""
    subgroup = random.choice(subgroup_list)
    return random.choice(subgroup)


# =========================================================================================
#    3. Labeled Generation Functions (Updated to return both command text and expected JSON)
# =========================================================================================

def generate_forward_labeled():
    distance = random_distance()
    components = {
        "verb": random.choice(verbs_forward),
        "direction": random.choice(directions_forward),
        "distance": distance,
        "unit": "cm"
    }
    command_text = full_text_processing(fill_template(choose_template(FORWARD_SUBGROUPS), components)) + "."
    expected_json = {
        "command": "forward",
        "parameters": {
            "distance": float(distance)
        }
    }
    return command_text, json.dumps(expected_json)


def generate_back_labeled():
    distance = random_distance()
    components = {
        "verb": random.choice(verbs_back),
        "direction": random.choice(directions_back),
        "distance": distance,
        "unit": "cm"
    }
    command_text = full_text_processing(fill_template(choose_template(BACK_SUBGROUPS), components)) + "."
    expected_json = {
        "command": "back",
        "parameters": {
            "distance": float(distance)
        }
    }
    return command_text, json.dumps(expected_json)


def generate_side_labeled(command_type):
    distance = random_distance()
    direction = command_type if command_type in ["left", "right"] else random.choice(side_directions)
    components = {
        "verb": random.choice(verbs_side),
        "direction": direction,
        "distance": distance,
        "unit": "cm"
    }
    command_text = full_text_processing(fill_template(choose_template(SIDE_SUBGROUPS), components)) + "."
    expected_json = {
        "command": direction,  # either "left" or "right"
        "parameters": {
            "distance": float(distance)
        }
    }
    return command_text, json.dumps(expected_json)


def generate_rotate_labeled():
    angle_str = random_angle()
    components = {
        "verb": random.choice(verbs_rotate),
        "direction": random.choice(rotate_directions),
        "angle": angle_str,
        "unit": "rad"
    }
    command_text = full_text_processing(fill_template(choose_template(ROTATE_SUBGROUPS), components)) + "."
    expected_json = {
        "command": "rotate",
        "parameters": {
            "angle": float(angle_str),
            "direction": components["direction"]
        }
    }
    return command_text, json.dumps(expected_json)


def generate_strafing_labeled():
    distance = random_distance()
    direction = random.choice(side_directions)
    components = {
        "verb": random.choice(verbs_side),
        "direction": direction,
        "distance": distance,
        "unit": "cm"
    }
    command_text = full_text_processing(fill_template(choose_template(STRAFING_SUBGROUPS), components)) + "."
    expected_json = {
        "command": "strafe",
        "parameters": {
            "distance": float(distance),
            "direction": direction
        }
    }
    return command_text, json.dumps(expected_json)


def generate_stop_labeled():
    phrase = random.choice(random.choice(STOP_SUBGROUPS))
    command_text = full_text_processing(phrase) + "."
    expected_json = {
        "command": "stop",
        "parameters": {}
    }
    return command_text, json.dumps(expected_json)


def generate_improved_labeled_command(command_type):
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
    elif command_type == "strafing":
        return generate_strafing_labeled()
    else:
        return generate_stop_labeled()


# =========================================================================================
#    4. High-Level Dispatch and Data Generation
# =========================================================================================

if __name__ == "__main__":
    num_labeled = 10_000
    valid_command_types = ["forward", "back", "left", "right", "rotate", "stop", "strafing"]

    labeled_commands = []
    labeled_jsons = []

    for _ in range(num_labeled):
        cmd_type = random.choice(valid_command_types)
        command_text, expected_json = generate_improved_labeled_command(cmd_type)
        labeled_commands.append(command_text)
        labeled_jsons.append(expected_json)

    with open("synthetic_basic_labeled_robot_commands.txt", "w", encoding="utf-8") as f_cmd, \
            open("synthetic_basic_labeled_robot_commands_json.txt", "w", encoding="utf-8") as f_json:
        for cmd, js in zip(labeled_commands, labeled_jsons):
            f_cmd.write(cmd + "\n")
            f_json.write(js + "\n")

    print("Labeled command data generated and saved as:")
    print(" - synthetic_labeled_robot_commands.txt (command text)")
    print(" - synthetic_labeled_robot_commands_json.txt (expected JSON outputs)")
