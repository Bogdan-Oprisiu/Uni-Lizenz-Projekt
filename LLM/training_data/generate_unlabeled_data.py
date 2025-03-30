import random

import math
from pre_processing.processing import full_text_processing

# Define synonyms and templates for valid commands
directions_forward = ["forward", "ahead", "advance"]
directions_back = ["back", "reverse", "backward"]
rotate_directions = ["left", "right"]

# Updated templates for valid commands, now including strafing commands
templates = {
    "forward": [
        "Move {direction} {distance} cm with acceleration {acceleration}.",
        "Go {direction} {distance} centimeters."
    ],
    "back": [
        "Move {direction} {distance} cm with acceleration {acceleration}.",
        "Go {direction} {distance} centimeters."
    ],
    "left": [
        "Strafe left {distance} cm with acceleration {acceleration}.",
        "Move left {distance} centimeters."
    ],
    "right": [
        "Strafe right {distance} cm with acceleration {acceleration}.",
        "Move right {distance} centimeters."
    ],
    "rotate": [
        "Rotate {direction} by {angle} rad with acceleration {acceleration}.",
        "Turn {direction} {angle} rad."
    ],
    "stop": [
        "Stop.",
        "Halt.",
        "Cease movement."
    ],
    "strafing": [
        "Strafe {direction} {distance} cm with acceleration {acceleration}.",
        "Slide {direction} {distance} centimeters."
    ]
}


def generate_command_text(command_type):
    """Generate a single unlabeled command string based on the command_type."""
    if command_type in ["forward", "back"]:
        direction = random.choice(directions_forward if command_type == "forward" else directions_back)
        distance = random.randint(10, 500)
        acceleration = random.choice([None, random.randint(5, 20)])  # Occasionally omit acceleration
        template = random.choice(templates[command_type])
        command_str = template.format(
            direction=direction,
            distance=distance,
            acceleration=acceleration if acceleration is not None else ""
        )
        return " ".join(full_text_processing(command_str).split())

    elif command_type in ["left", "right"]:
        distance = random.randint(10, 500)
        acceleration = random.choice([None, random.randint(5, 20)])
        template = random.choice(templates[command_type])
        command_str = template.format(
            distance=distance,
            acceleration=acceleration if acceleration is not None else ""
        )
        return " ".join(full_text_processing(command_str).split())

    elif command_type == "rotate":
        direction = random.choice(rotate_directions)
        angle_deg = random.randint(10, 180)
        angle_rad = math.radians(angle_deg)

        acceleration = random.choice([None, round(random.uniform(20, 40), 2)])
        template = random.choice(templates["rotate"])
        command_str = template.format(
            direction=direction,
            angle=angle_rad,
            acceleration=acceleration if acceleration is not None else ""
        )
        return " ".join(full_text_processing(command_str).split())

    elif command_type == "stop":
        template = random.choice(templates["stop"])
        return " ".join(full_text_processing(template).split())

    elif command_type == "strafing":
        # For strafing, randomly choose a lateral direction (left or right)
        direction = random.choice(["left", "right"])
        distance = random.randint(10, 500)
        acceleration = random.choice([None, random.randint(5, 20)])
        template = random.choice(templates["strafing"])
        command_str = template.format(
            direction=direction,
            distance=distance,
            acceleration=acceleration if acceleration is not None else ""
        )
        return " ".join(full_text_processing(command_str).split())


def generate_invalid_command_text():
    """Generate an error command string that is truly nonsensical or uses an invalid value."""
    error_type = random.choice(["nonsense", "invalid_value"])
    if error_type == "nonsense":
        # Completely nonsensical string
        nonsense_str = "asdfghjkl 1234 zxcvbnm"
        return " ".join(full_text_processing(nonsense_str).split())
    elif error_type == "invalid_value":
        # Use a non-numeric value where a number is expected
        invalid_value_str = "turn left by banana"
        return " ".join(full_text_processing(invalid_value_str).split())


# Number of unlabeled commands to generate
num_unlabeled = 10_000
error_probability = 0.001

# List to store command strings
unlabeled_commands = []

valid_command_types = ["forward", "back", "left", "right", "rotate", "stop", "strafing"]

# Generate commands (mixing valid and error cases)
for _ in range(num_unlabeled):
    if random.random() < error_probability:
        # Generate an error command
        command = generate_invalid_command_text()
    else:
        # Generate a valid command by randomly picking a type
        command_type = random.choice(valid_command_types)
        command = generate_command_text(command_type)
    unlabeled_commands.append(command)

# Save the unlabeled commands to a plain text file (one command per line)
with open("synthetic_unlabeled_robot_commands.txt", "w", encoding="utf-8") as f:
    for command in unlabeled_commands:
        f.write(command + "\n")

print("Unlabeled command data generated and saved as synthetic_unlabeled_robot_commands.txt")
