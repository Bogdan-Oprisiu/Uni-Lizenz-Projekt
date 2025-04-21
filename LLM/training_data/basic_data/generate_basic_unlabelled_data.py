import math
import random

from pre_processing.processing import full_text_processing

# =========================================================================================
#    1. Subgroup Template Definitions
# =========================================================================================

# Subgroups for "forward" command
FORWARD_SUBGROUPS = [
    # Quick movement templates
    [
        "{verb} {direction} quickly {distance}{unit}",
        "please {verb} {direction} briefly {distance}{unit}",
        "quickly {verb} {distance}{unit} {direction}"
    ],
    # Steady movement templates
    [
        "{verb} {direction} steadily {distance}{unit}",
        "proceed {direction} at a steady pace for {distance}{unit}",
        "{verb} {distance}{unit} {direction} in a controlled manner"
    ],
]

# Subgroups for "back" command
BACK_SUBGROUPS = [
    # Quick reversal templates
    [
        "{verb} {direction} fast {distance}{unit}",
        "please {verb} {direction} swiftly {distance}{unit}",
        "quickly {verb} {distance}{unit} {direction}"
    ],
    # Slow or careful reversal templates
    [
        "{verb} {direction} slowly {distance}{unit}",
        "carefully {verb} {direction} for {distance}{unit}",
        "{verb} {distance}{unit} {direction} with caution"
    ],
]

# Subgroups for side (left/right) commands
SIDE_SUBGROUPS = [
    # Quick side movements
    [
        "{verb} {direction} quickly {distance}{unit}",
        "please {verb} {direction} in a swift manner for {distance}{unit}",
        "quickly {verb} {distance}{unit} to the {direction}"
    ],
    # Steady side movements
    [
        "{verb} {direction} steadily {distance}{unit}",
        "move {direction} calmly {distance}{unit}",
        "{verb} {distance}{unit} {direction} at a consistent pace"
    ],
]

# Subgroups for rotate command
ROTATE_SUBGROUPS = [
    # Short rotation (small angles, quick spin)
    [
        "{verb} {direction} briefly by {angle}{unit}",
        "spin {direction} quickly {angle}{unit}",
        "please {verb} a small angle {angle}{unit} to the {direction}"
    ],
    # Longer rotation (bigger angle, steady turn)
    [
        "turn {direction} steadily by {angle}{unit}",
        "rotate {direction} a larger angle of {angle}{unit}",
        "{verb} {angle}{unit} toward the {direction} in a controlled manner"
    ],
]

# Subgroups for strafing command
STRAFING_SUBGROUPS = [
    # Quick strafing
    [
        "{verb} {direction} swiftly {distance}{unit}",
        "strafing to the {direction} quickly {distance}{unit}",
        "please {verb} {distance}{unit} in the {direction} direction fast"
    ],
    # Steady strafing
    [
        "{verb} {direction} steadily {distance}{unit}",
        "strafing to the {direction} carefully {distance}{unit}",
        "{verb} {distance}{unit} {direction} at a consistent rate"
    ],
]

# Subgroups for stop command (short vs. long)
STOP_SUBGROUPS = [
    # Short
    [
        "stop",
        "halt"
    ],
    # Slightly longer
    [
        "please stop",
        "cease movement"
    ],
]

# =========================================================================================
#    2. Synonyms and Helper Functions
# =========================================================================================

# Expanded synonym lists for better diversity (kept separate to ensure cross-validation).
directions_forward = ["forward", "ahead", "advance", "proceed", "go ahead"]
directions_back = ["back", "reverse", "backward", "retreat", "recede"]
rotate_directions = ["left", "right", "port", "starboard"]
side_directions = ["left", "right", "port", "starboard"]

# Synonym lists for verbs per command type.
verbs_forward = ["move", "go", "advance", "proceed", "head"]
verbs_back = ["move", "go", "reverse", "retreat", "step", "recede"]
verbs_side = ["strafe", "move", "slide", "shift"]
verbs_rotate = ["rotate", "turn", "spin", "pivot"]


def random_distance():
    """
    Generate a distance as a float between 10 and 500, formatted with one decimal.
    """
    return f"{random.uniform(10, 500):.1f}"


def random_angle():
    """
    Generate a random angle in radians between -pi/2 and pi/2, formatted to 4 decimals.
    """
    angle_rad = random.uniform(-math.pi / 2, math.pi / 2)
    return f"{angle_rad:.4f}"


def fill_template(template, components):
    """
    Replace placeholders in the template with values from components.
    Placeholders: {verb}, {direction}, {distance}, {unit}, {angle}
    """
    return template.format(**components)


def choose_template(subgroup_list):
    """
    Randomly choose one subgroup from the list of subgroups,
    then randomly choose one template from that subgroup.
    """
    subgroup = random.choice(subgroup_list)
    return random.choice(subgroup)


# =========================================================================================
#    3. Dedicated Generation Functions with Subgroup Selection
# =========================================================================================

def generate_forward_command():
    distance = random_distance()
    components = {
        "verb": random.choice(verbs_forward),
        "direction": random.choice(directions_forward),
        "distance": distance,
        "unit": "cm"
    }
    template = choose_template(FORWARD_SUBGROUPS)
    return full_text_processing(fill_template(template, components)) + "."


def generate_back_command():
    distance = random_distance()
    components = {
        "verb": random.choice(verbs_back),
        "direction": random.choice(directions_back),
        "distance": distance,
        "unit": "cm"
    }
    template = choose_template(BACK_SUBGROUPS)
    return full_text_processing(fill_template(template, components)) + "."


def generate_side_command(command_type):
    """
    command_type is either "left" or "right".
    We’ll still respect that so, e.g., if a user specifically wants "left," we fix 'direction' to "left."
    But we can also choose a random template from the side subgroups.
    """
    distance = random_distance()
    # If command_type is specifically left/right, override direction,
    # otherwise pick a random one from side_directions.
    if command_type in ["left", "right"]:
        direction = command_type
    else:
        direction = random.choice(side_directions)

    components = {
        "verb": random.choice(verbs_side),
        "direction": direction,
        "distance": distance,
        "unit": "cm"
    }
    template = choose_template(SIDE_SUBGROUPS)
    return full_text_processing(fill_template(template, components)) + "."


def generate_rotate_command():
    angle_str = random_angle()
    components = {
        "verb": random.choice(verbs_rotate),
        "direction": random.choice(rotate_directions),
        "angle": angle_str,
        "unit": "rad"
    }
    template = choose_template(ROTATE_SUBGROUPS)
    return full_text_processing(fill_template(template, components)) + "."


def generate_strafing_command():
    distance = random_distance()
    direction = random.choice(side_directions)
    components = {
        "verb": random.choice(verbs_side),
        "direction": direction,
        "distance": distance,
        "unit": "cm"
    }
    template = choose_template(STRAFING_SUBGROUPS)
    return full_text_processing(fill_template(template, components)) + "."


def generate_stop_command():
    """
    We now have short vs. longer subgroups for 'stop'.
    """
    chosen_subgroup = random.choice(STOP_SUBGROUPS)
    # Each subgroup for stop is basically short phrases, but we can pick one:
    phrase = random.choice(chosen_subgroup)
    return full_text_processing(phrase) + "."


# =========================================================================================
#    4. High-Level Dispatch
# =========================================================================================

def generate_improved_command_text(command_type):
    """
    Generate a basic command string (without acceleration) for the given command type
    using refined subgroups and synonyms.
    """
    if command_type == "forward":
        return generate_forward_command()
    elif command_type == "back":
        return generate_back_command()
    elif command_type in ["left", "right"]:
        return generate_side_command(command_type)
    elif command_type == "rotate":
        return generate_rotate_command()
    elif command_type == "stop":
        return generate_stop_command()
    elif command_type == "strafing":
        return generate_strafing_command()
    else:
        # Default fallback if an unknown command type is given.
        return "stop."


# =========================================================================================
#    5. Generate Data
# =========================================================================================

if __name__ == "__main__":
    # Set the number of basic unlabeled commands to generate.
    num_unlabeled = 100_000
    valid_command_types = ["forward", "back", "left", "right", "rotate", "stop", "strafing"]

    # Generate the basic unlabeled commands using the improved methods.
    basic_unlabeled_commands = [
        generate_improved_command_text(random.choice(valid_command_types))
        for _ in range(num_unlabeled)
    ]

    # Save the commands to a plain text file (one command per line),
    # prefixing each command with its index to show that we did it.
    with open("synthetic_basic_unlabeled_robot_commands.txt", "w", encoding="utf-8") as f:
        for idx, command in enumerate(basic_unlabeled_commands, start=1):
            f.write(f"{idx}: {command}\n")

    print("Basic unlabeled command data generated and saved as synthetic_basic_unlabeled_robot_commands.txt")
