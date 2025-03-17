import json
import random

# Define parameter ranges and synonyms
directions_forward = ["forward", "ahead", "advance"]
directions_back = ["back", "reverse", "backward"]
turns_left = ["left", "to the left", "counterclockwise"]
turns_right = ["right", "to the right", "clockwise"]

# Templates for valid commands
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
        "Turn {direction} by {angle} radians with acceleration {acceleration}.",
        "Rotate {direction} {angle} radians."
    ],
    "right": [
        "Turn {direction} by {angle} radians with acceleration {acceleration}.",
        "Rotate {direction} {angle} radians."
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
        acceleration = random.choice([None, random.randint(5, 20)])  # Sometimes None to trigger default use
        template = random.choice(templates[command_type])
        command_str = template.format(direction=direction, distance=distance,
                                      acceleration=acceleration if acceleration else "")
        # Clean up extra spaces if acceleration is missing
        command_str = " ".join(command_str.split())
        return {
            "input_text": command_str,
            "expected_output": {
                "action": command_type,
                "parameters": {
                    "distance": distance,
                    "acceleration": acceleration if acceleration is not None else "default"
                }
            }
        }
    elif command_type in ["left", "right"]:
        direction = random.choice(turns_left if command_type == "left" else turns_right)
        angle = round(random.uniform(0.1, 3.14), 2)
        acceleration = random.choice([None, round(random.uniform(0.1, 1.0), 2)])
        template = random.choice(templates[command_type])
        command_str = template.format(direction=direction, angle=angle,
                                      acceleration=acceleration if acceleration else "")
        command_str = " ".join(command_str.split())
        return {
            "input_text": command_str,
            "expected_output": {
                "action": command_type,
                "parameters": {
                    "angle": angle,
                    "acceleration": acceleration if acceleration is not None else "default"
                }
            }
        }
    elif command_type == "stop":
        template = random.choice(templates["stop"])
        return {
            "input_text": template,
            "expected_output": {
                "action": "stop",
                "parameters": {}
            }
        }


# Generate a set of valid examples
valid_examples = [generate_valid_command(random.choice(["forward", "back", "left", "right", "stop"])) for _ in
                  range(100000)]


# Optionally: Generate error cases (similar approach, but inject mistakes)
def generate_invalid_command():
    # Example: misspelled command or missing value
    error_type = random.choice(["misspelling", "missing_parameter", "invalid_value"])
    if error_type == "misspelling":
        # For instance, "florward" instead of "forward"
        return {
            "input_text": "florward 100",
            "expected_output": {
                "errors": [
                    {
                        "rawCommand": "florward 100",
                        "code": "INVALID_COMMAND",
                        "description": "Unknown command 'florward'. Did you mean 'forward'?"
                    }
                ]
            }
        }
    elif error_type == "missing_parameter":
        return {
            "input_text": "move forward",
            "expected_output": {
                "errors": [
                    {
                        "rawCommand": "move forward",
                        "code": "MISSING_PARAMETER",
                        "description": "The 'distance' parameter is missing."
                    }
                ]
            }
        }
    elif error_type == "invalid_value":
        return {
            "input_text": "turn left by fast",
            "expected_output": {
                "errors": [
                    {
                        "rawCommand": "turn left by fast",
                        "code": "INVALID_PARAMETER_TYPE",
                        "description": "Expected a numeric value for 'angle', but received a string."
                    }
                ]
            }
        }


invalid_examples = [generate_invalid_command() for _ in range(20)]

# Combine and shuffle
dataset = valid_examples + invalid_examples
random.shuffle(dataset)

# Save as JSON file
with open("synthetic_robot_commands.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Synthetic data generated and saved as synthetic_robot_commands.json")
