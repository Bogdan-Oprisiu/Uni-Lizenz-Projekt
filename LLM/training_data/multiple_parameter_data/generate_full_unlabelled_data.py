import random

from pre_processing.processing import full_text_processing

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
# 2) Synonyms, Helper Functions, and Acceleration
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
    return f"{random.uniform(10, 500):.1f}"


def random_angle_degrees():
    angle_deg = random.uniform(-180, 180)
    return f"{angle_deg:.1f}"


def random_acceleration_linear():
    return f"{random.uniform(0, 50):.2f}"


def random_acceleration_angular():
    return f"{random.uniform(0, 60):.2f}"


def fill_template(template, components):
    return template.format(**components)


def choose_template(subgroup_list):
    subgroup = random.choice(subgroup_list)
    return random.choice(subgroup)


def maybe_add_acceleration_phrase(cmd_text, is_linear=True):
    if random.random() < 0.5:
        if is_linear:
            accel_val = random_acceleration_linear()
            return cmd_text + f" with acceleration {accel_val} cm/s^2"
        else:
            accel_val = random_acceleration_angular()
            return cmd_text + f" with angular acceleration {accel_val} deg/s^2"
    return cmd_text


# =============================================================================
# 3) Unlabeled Generation Functions
# =============================================================================

def generate_forward_cmd():
    distance = random_distance()
    template = choose_template(FORWARD_SUBGROUPS)
    components = {
        "verb": random.choice(verbs_forward),
        "direction": random.choice(directions_forward),
        "distance": distance,
        "unit": "cm"
    }
    cmd = fill_template(template, components)
    cmd = maybe_add_acceleration_phrase(cmd, is_linear=True)
    return full_text_processing(cmd) + "."


def generate_back_cmd():
    distance = random_distance()
    template = choose_template(BACK_SUBGROUPS)
    components = {
        "verb": random.choice(verbs_back),
        "direction": random.choice(directions_back),
        "distance": distance,
        "unit": "cm"
    }
    cmd = fill_template(template, components)
    cmd = maybe_add_acceleration_phrase(cmd, is_linear=True)
    return full_text_processing(cmd) + "."


def generate_side_cmd(direction):
    distance = random_distance()
    template = choose_template(SIDE_SUBGROUPS)
    components = {
        "verb": random.choice(verbs_side),
        "direction": direction,
        "distance": distance,
        "unit": "cm"
    }
    cmd = fill_template(template, components)
    cmd = maybe_add_acceleration_phrase(cmd, is_linear=True)
    return full_text_processing(cmd) + "."


def generate_rotate_cmd():
    angle_deg = random_angle_degrees()
    template = choose_template(ROTATE_SUBGROUPS)
    chosen_dir = random.choice(rotate_directions)
    components = {
        "verb": random.choice(verbs_rotate),
        "direction": chosen_dir,
        "angle": angle_deg,
        "unit": "deg"
    }
    cmd = fill_template(template, components)
    cmd = maybe_add_acceleration_phrase(cmd, is_linear=False)
    return full_text_processing(cmd) + "."


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
        return generate_stop_cmd()


# =============================================================================
# 4) Main Script - Unlabeled Data Generation
# =============================================================================

if __name__ == "__main__":
    num_samples = 10_000
    valid_command_types = ["forward", "back", "left", "right", "rotate", "stop"]

    all_commands = []
    for _ in range(num_samples):
        cmd_t = random.choice(valid_command_types)
        text_cmd = generate_unlabeled_command(cmd_t)
        all_commands.append(text_cmd)

    # Write them to a single text file
    with open("synthetic_unlabeled_robot_commands_with_accel.txt", "w", encoding="utf-8") as f:
        for idx, cmd in enumerate(all_commands, start=1):
            f.write(f"{idx}: {cmd}\n")

    print("Unlabeled commands with optional acceleration saved in:")
    print("synthetic_unlabeled_robot_commands_with_accel.txt")
