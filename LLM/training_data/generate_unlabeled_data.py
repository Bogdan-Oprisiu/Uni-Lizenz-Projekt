import random

# Define synonyms and templates for valid commands
directions_forward = ["forward", "ahead", "advance"]
directions_back = ["back", "reverse", "backward"]
rotate_directions = ["left", "right"]

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
        "Rotate {direction} by {angle} degrees with acceleration {acceleration}.",
        "Turn {direction} {angle} degrees."
    ],
    "stop": [
        "Stop.",
        "Halt.",
        "Cease movement."
    ]
}


def generate_command_text(command_type):
    """Generate a single unlabeled command string based on the command_type."""
    if command_type in ["forward", "back"]:
        direction = random.choice(directions_forward if command_type == "forward" else directions_back)
        distance = random.randint(10, 500)
        # Occasionally omit acceleration to simulate default usage
        acceleration = random.choice([None, random.randint(5, 20)])
        template = random.choice(templates[command_type])
        # When acceleration is None, we simply leave it blank in the text.
        command_str = template.format(
            direction=direction,
            distance=distance,
            acceleration=acceleration if acceleration is not None else ""
        )
        return " ".join(command_str.split())

    elif command_type in ["left", "right"]:
        distance = random.randint(10, 500)
        acceleration = random.choice([None, random.randint(5, 20)])
        template = random.choice(templates[command_type])
        command_str = template.format(
            distance=distance,
            acceleration=acceleration if acceleration is not None else ""
        )
        return " ".join(command_str.split())

    elif command_type == "rotate":
        direction = random.choice(rotate_directions)
        angle = random.randint(10, 180)  # in degrees
        acceleration = random.choice([None, round(random.uniform(20, 40), 2)])
        template = random.choice(templates["rotate"])
        command_str = template.format(
            direction=direction,
            angle=angle,
            acceleration=acceleration if acceleration is not None else ""
        )
        return " ".join(command_str.split())

    elif command_type == "stop":
        template = random.choice(templates["stop"])
        return template


# Number of unlabeled commands to generate
num_unlabeled = 3_000_000  # For example, 3x as many as our labeled data

# List to store command strings
unlabeled_commands = []

# Generate a command for each iteration; each line is just the raw command text.
for _ in range(num_unlabeled):
    command_type = random.choice(["forward", "back", "left", "right", "rotate", "stop"])
    command = generate_command_text(command_type)
    unlabeled_commands.append(command)

# Save the unlabeled commands to a plain text file (one command per line)
with open("unlabeled_robot_commands.txt", "w", encoding="utf-8") as f:
    for command in unlabeled_commands:
        f.write(command + "\n")

print("Unlabeled command data generated and saved as unlabeled_robot_commands.txt")
