import math
import random

from pre_processing.processing import full_text_processing

# Define synonyms and simpler templates for basic commands (without acceleration)
directions_forward = ["forward", "ahead", "advance"]
directions_back = ["back", "reverse", "backward"]
rotate_directions = ["left", "right"]

basic_templates = {
    "forward": [
        "Move {direction} {distance} cm.",
        "Go {direction} {distance} centimeters."
    ],
    "back": [
        "Move {direction} {distance} cm.",
        "Go {direction} {distance} centimeters."
    ],
    "left": [
        "Strafe left {distance} cm.",
        "Move left {distance} cm."
    ],
    "right": [
        "Strafe right {distance} cm.",
        "Move right {distance} cm."
    ],
    "rotate": [
        "Rotate {direction} by {angle} rad.",
        "Turn {direction} {angle} rad."
    ],
    "stop": [
        "Stop.",
        "Halt.",
        "Cease movement."
    ],
    "strafing": [
        "Strafe {direction} {distance} cm.",
        "Slide {direction} {distance} cm."
    ]
}


def generate_basic_command_text(command_type):
    """
    Generate a basic command string (without acceleration) for the given command type.
    """
    if command_type in ["forward", "back"]:
        direction = random.choice(directions_forward if command_type == "forward" else directions_back)
        distance = random.randint(10, 500)
        template = random.choice(basic_templates[command_type])
        command_str = template.format(direction=direction, distance=distance)
        return " ".join(full_text_processing(command_str).split())

    elif command_type in ["left", "right"]:
        distance = random.randint(10, 500)
        template = random.choice(basic_templates[command_type])
        command_str = template.format(distance=distance)
        return " ".join(full_text_processing(command_str).split())

    elif command_type == "rotate":
        direction = random.choice(rotate_directions)
        angle_deg = random.randint(10, 180)
        angle_rad = math.radians(angle_deg)
        template = random.choice(basic_templates["rotate"])
        command_str = template.format(direction=direction, angle=angle_rad)
        return " ".join(full_text_processing(command_str).split())

    elif command_type == "stop":
        template = random.choice(basic_templates["stop"])
        command_str = template
        return " ".join(full_text_processing(command_str).split())

    elif command_type == "strafing":
        direction = random.choice(["left", "right"])
        distance = random.randint(10, 500)
        template = random.choice(basic_templates["strafing"])
        command_str = template.format(direction=direction, distance=distance)
        return " ".join(full_text_processing(command_str).split())


# Set the number of basic unlabeled commands to generate
num_unlabeled = 10_000
valid_command_types = ["forward", "back", "left", "right", "rotate", "stop", "strafing"]

# Generate the basic unlabeled commands
basic_unlabeled_commands = [
    generate_basic_command_text(random.choice(valid_command_types))
    for _ in range(num_unlabeled)
]

# Save the commands to a plain text file (one command per line)
with open("synthetic_basic_unlabeled_robot_commands.txt", "w", encoding="utf-8") as f:
    for command in basic_unlabeled_commands:
        f.write(command + "\n")

print("Basic unlabeled command data generated and saved as synthetic_basic_unlabeled_robot_commands.txt")
