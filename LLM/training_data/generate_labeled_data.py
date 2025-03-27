import json
import random

from pre_processing.processing import full_text_processing

# Define parameter ranges and synonyms
directions_forward = ["forward", "ahead", "advance"]
directions_back = ["back", "reverse", "backward"]

# For lateral movement, left/right (strafe)
rotate_directions = ["left", "right"]

# Updated templates for valid commands
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


def generate_valid_command(command_type):
    if command_type in ["forward", "back"]:
        direction = random.choice(directions_forward if command_type == "forward" else directions_back)
        distance = random.randint(10, 500)
        acceleration = random.choice([None, random.randint(5, 20)])  # Occasionally trigger default use
        template = random.choice(templates[command_type])
        command_str = template.format(
            direction=direction,
            distance=distance,
            acceleration=acceleration if acceleration is not None else ""
        )
        # Clean up extra spaces and then preprocess the command string
        command_str = " ".join(command_str.split())
        processed_str = full_text_processing(command_str)
        return {
            "input_text": processed_str,
            "expected_output": {
                "action": command_type,
                "parameters": {
                    "distance": distance,
                    "acceleration": acceleration if acceleration is not None else "default"
                }
            }
        }
    elif command_type in ["left", "right"]:
        distance = random.randint(10, 500)
        acceleration = random.choice([None, random.randint(5, 20)])
        template = random.choice(templates[command_type])
        command_str = template.format(
            distance=distance,
            acceleration=acceleration if acceleration is not None else ""
        )
        command_str = " ".join(command_str.split())
        processed_str = full_text_processing(command_str)
        return {
            "input_text": processed_str,
            "expected_output": {
                "action": command_type,
                "parameters": {
                    "distance": distance,
                    "acceleration": acceleration if acceleration is not None else "default"
                }
            }
        }
    elif command_type == "rotate":
        direction = random.choice(rotate_directions)
        angle = random.randint(10, 180)  # degrees
        acceleration = random.choice([None, round(random.uniform(20, 40), 2)])
        template = random.choice(templates["rotate"])
        command_str = template.format(
            direction=direction,
            angle=angle,
            acceleration=acceleration if acceleration is not None else ""
        )
        command_str = " ".join(command_str.split())
        processed_str = full_text_processing(command_str)
        return {
            "input_text": processed_str,
            "expected_output": {
                "action": "rotate",
                "parameters": {
                    "angle": angle,
                    "direction": direction,
                    "acceleration": acceleration if acceleration is not None else "default"
                }
            }
        }
    elif command_type == "stop":
        template = random.choice(templates["stop"])
        processed_str = full_text_processing(template)
        return {
            "input_text": processed_str,
            "expected_output": {
                "action": "stop",
                "parameters": {}
            }
        }


# Updated error cases: Instead of just minor grammatical issues, these errors now
# generate commands that are truly nonsensical or lack any coherent structure.
def generate_invalid_command():
    error_type = random.choice(["nonsense", "invalid_value"])
    if error_type == "nonsense":
        # Generate a completely nonsensical command string.
        nonsense_str = "asdfghjkl 1234 zxcvbnm"
        processed_str = full_text_processing(nonsense_str)
        return {
            "input_text": processed_str,
            "expected_output": {
                "errors": [
                    {
                        "rawCommand": nonsense_str,
                        "code": "INVALID_COMMAND",
                        "description": "The command is nonsensical and does not match any known pattern."
                    }
                ]
            }
        }
    elif error_type == "invalid_value":
        # Use an invalid, non-numeric value for a parameter.
        invalid_value_str = "turn left by banana"
        processed_str = full_text_processing(invalid_value_str)
        return {
            "input_text": processed_str,
            "expected_output": {
                "errors": [
                    {
                        "rawCommand": invalid_value_str,
                        "code": "INVALID_PARAMETER_TYPE",
                        "description": "Expected a numeric value for parameter, but received a non-numeric string."
                    }
                ]
            }
        }


# Generate a set of valid examples
valid_examples = [
    generate_valid_command(random.choice(["forward", "back", "left", "right", "rotate", "stop"]))
    for _ in range(1_000_000)
]

# Generate a smaller set of invalid examples
invalid_examples = [generate_invalid_command() for _ in range(20)]

# Combine and shuffle the dataset
dataset = valid_examples + invalid_examples
random.shuffle(dataset)

# Save as JSON file
with open("synthetic_labeled_robot_commands.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Synthetic data generated and saved as synthetic_labeled_robot_commands.json")
