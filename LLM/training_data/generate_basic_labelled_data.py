import json
import math
import random

from pre_processing.processing import full_text_processing

# Define synonyms and simpler templates for basic labeled commands (without acceleration)
directions_forward = ["forward", "ahead", "advance"]
directions_back = ["back", "reverse", "backward"]
rotate_directions = ["left", "right"]

# Basic templates without acceleration
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


def generate_basic_labelled_command(command_type):
    """
    Generate a single basic labelled command string (without acceleration)
    along with its expected output.
    """
    if command_type in ["forward", "back"]:
        direction = random.choice(directions_forward if command_type == "forward" else directions_back)
        distance = random.randint(10, 500)
        template = random.choice(basic_templates[command_type])
        command_str = template.format(direction=direction, distance=distance)
        command_str = " ".join(full_text_processing(command_str).split())
        return {
            "input_text": command_str,
            "expected_output": {
                "action": command_type,
                "parameters": {
                    "distance": distance
                }
            }
        }
    elif command_type in ["left", "right"]:
        distance = random.randint(10, 500)
        template = random.choice(basic_templates[command_type])
        command_str = template.format(distance=distance)
        command_str = " ".join(full_text_processing(command_str).split())
        return {
            "input_text": command_str,
            "expected_output": {
                "action": command_type,
                "parameters": {
                    "distance": distance
                }
            }
        }
    elif command_type == "rotate":
        direction = random.choice(rotate_directions)
        angle_deg = random.randint(10, 180)
        angle_rad = math.radians(angle_deg)
        template = random.choice(basic_templates["rotate"])
        command_str = template.format(direction=direction, angle=angle_rad)
        command_str = " ".join(full_text_processing(command_str).split())
        return {
            "input_text": command_str,
            "expected_output": {
                "action": "rotate",
                "parameters": {
                    "direction": direction,
                    "angle": angle_rad
                }
            }
        }
    elif command_type == "stop":
        template = random.choice(basic_templates["stop"])
        command_str = template
        command_str = " ".join(full_text_processing(command_str).split())
        return {
            "input_text": command_str,
            "expected_output": {
                "action": "stop",
                "parameters": {}
            }
        }
    elif command_type == "strafing":
        # For strafing, choose left or right as the lateral direction.
        direction = random.choice(["left", "right"])
        distance = random.randint(10, 500)
        template = random.choice(basic_templates["strafing"])
        command_str = template.format(direction=direction, distance=distance)
        command_str = " ".join(full_text_processing(command_str).split())
        return {
            "input_text": command_str,
            "expected_output": {
                "action": "strafing",
                "parameters": {
                    "direction": direction,
                    "distance": distance
                }
            }
        }


# Number of basic labeled commands to generate
num_basic_labelled = 10_000
valid_command_types = ["forward", "back", "left", "right", "rotate", "stop", "strafing"]

basic_labelled_commands = [
    generate_basic_labelled_command(random.choice(valid_command_types))
    for _ in range(num_basic_labelled)
]

# Save to JSON file
with open("synthetic_basic_labeled_robot_commands.json", "w", encoding="utf-8") as f:
    json.dump(basic_labelled_commands, f, indent=2)

print("Basic labelled command data generated and saved as synthetic_basic_labeled_robot_commands.json")
