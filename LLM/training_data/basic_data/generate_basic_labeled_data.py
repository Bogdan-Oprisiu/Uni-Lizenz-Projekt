import json
import math
import random
import re  # Make sure re is imported

# =============================================================================
# START: Import Classes (Assuming they are in utils/command_processor.py)
# =============================================================================
try:
    # Adjust the path if your file/classes have different names
    from utils.generate_json_command import CommandProcessor
    print("Successfully imported CommandProcessor from utils.")
except ImportError as e:
    print(f"FATAL: Could not import CommandProcessor from utils: {e}")
    print("Please ensure CommandProcessor and UnitConverter classes are defined")
    print("in a file accessible via the 'utils' package (e.g., utils/command_processor.py)")
    exit(1)
# =============================================================================
# END: Import Classes
# =============================================================================


# Assuming pre_processing.processing.full_text_processing exists and works as intended
try:
    from pre_processing.processing import full_text_processing
except ImportError:
    print("Warning: pre_processing.processing not found. Using basic text processing.")


    def full_text_processing(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# =========================================================================================
#    1. Subgroup Template Definitions (No changes needed here)
# =========================================================================================
# NOTE: Ellipses (...) represent the full original template lists for brevity.
# Copy the full lists from the previous version of the script.
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
#    2. Synonyms and Helper Functions (No changes needed here)
# =========================================================================================
directions_forward = ["forward", "ahead", "advance", "proceed", "go ahead"]
directions_back = ["back", "reverse", "backward", "retreat", "recede"]
rotate_directions = ["left", "right", "port", "starboard"]
side_directions = ["left", "right", "port", "starboard"]
verbs_forward = ["move", "go", "advance", "proceed", "head"]
verbs_back = ["move", "go", "reverse", "retreat", "step", "recede"]
verbs_side = ["strafe", "move", "slide", "shift"]
verbs_rotate = ["rotate", "turn", "spin", "pivot"]


def random_distance(unit='cm'):
    """Generate distance and return value and unit."""
    val = round(random.uniform(10, 500), 1)
    return val, unit  # Return value and unit separately


def random_angle(unit='rad'):
    """Generate angle and return value and unit."""
    if unit == 'rad':
        val = round(random.uniform(-math.pi / 2, math.pi / 2), 4)
    elif unit == 'deg':
        val = round(random.uniform(-90, 90), 1)
    else:  # Default to radians
        val = round(random.uniform(-math.pi / 2, math.pi / 2), 4)
        unit = 'rad'
    return val, unit  # Return value and unit separately


def fill_template(template, components):
    """Replace placeholders in the template with values from components."""
    # Format numeric values for the text string
    formatted_components = components.copy()
    if 'distance' in formatted_components:
        formatted_components['distance'] = f"{formatted_components['distance']:.1f}"  # Format distance for text
    if 'angle' in formatted_components:
        # Format angle based on its unit for text
        if formatted_components.get('unit') == 'deg':
            formatted_components['angle'] = f"{formatted_components['angle']:.1f}"
        else:  # Assume rad
            formatted_components['angle'] = f"{formatted_components['angle']:.4f}"

    return template.format(**formatted_components)


def choose_template(subgroup_list):
    """Randomly choose a subgroup, then a template from that subgroup."""
    subgroup = random.choice(subgroup_list)
    return random.choice(subgroup)


# =========================================================================================
#    3. Labeled Generation Functions (Updated to return command text and abstract map)
# =========================================================================================

def generate_forward_labeled():
    distance_val, distance_unit = random_distance(unit='cm')  # Assume cm for basic generation
    components = {
        "verb": random.choice(verbs_forward),
        "direction": random.choice(directions_forward),
        "distance": distance_val,
        "unit": distance_unit
    }
    command_text = full_text_processing(fill_template(choose_template(FORWARD_SUBGROUPS), components)) + "."
    # Abstract map expected from LLM
    expected_map = {
        "command": "forward",
        "distance": distance_val,
        "distance_unit": distance_unit  # Include unit if LLM might extract it
    }
    return command_text, expected_map


def generate_back_labeled():
    distance_val, distance_unit = random_distance(unit='cm')
    components = {
        "verb": random.choice(verbs_back),
        "direction": random.choice(directions_back),
        "distance": distance_val,
        "unit": distance_unit
    }
    command_text = full_text_processing(fill_template(choose_template(BACK_SUBGROUPS), components)) + "."
    expected_map = {
        "command": "back",
        "distance": distance_val,
        "distance_unit": distance_unit
    }
    return command_text, expected_map


def generate_side_labeled(command_type):
    distance_val, distance_unit = random_distance(unit='cm')
    # Use the specific command ('left' or 'right') as the primary direction
    direction = command_type if command_type in ["left", "right"] else random.choice(side_directions)
    # Allow text to sometimes use port/starboard even if command is left/right
    text_direction = random.choice([d for d in side_directions if (d in ['left', 'port'] and direction == 'left') or (
                d in ['right', 'starboard'] and direction == 'right')])

    components = {
        "verb": random.choice(verbs_side),
        "direction": text_direction,  # Use potentially different text direction
        "distance": distance_val,
        "unit": distance_unit
    }
    command_text = full_text_processing(fill_template(choose_template(SIDE_SUBGROUPS), components)) + "."
    expected_map = {
        "command": direction,  # Command is 'left' or 'right'
        "distance": distance_val,
        "distance_unit": distance_unit
    }
    return command_text, expected_map


def generate_rotate_labeled():
    # Randomly choose if the text uses radians or degrees
    use_deg_in_text = random.random() < 0.3  # 30% chance to use degrees in text
    angle_unit_text = 'deg' if use_deg_in_text else 'rad'
    angle_val, angle_unit_map = random_angle(unit=angle_unit_text)  # Generate value in the chosen unit

    chosen_direction = random.choice(rotate_directions)
    components = {
        "verb": random.choice(verbs_rotate),
        "direction": chosen_direction,
        "angle": angle_val,  # Value matches the unit chosen for text
        "unit": angle_unit_text  # Unit matches the value for text
    }
    command_text = full_text_processing(fill_template(choose_template(ROTATE_SUBGROUPS), components)) + "."
    # Map should contain the value AND the unit as extracted from text
    expected_map = {
        "command": "rotate",
        "angle": angle_val,
        "angle_unit": angle_unit_map,  # Unit extracted from text
        "direction": chosen_direction
    }
    return command_text, expected_map


def generate_strafing_labeled():
    distance_val, distance_unit = random_distance(unit='cm')
    chosen_direction = random.choice(side_directions)
    components = {
        "verb": random.choice(verbs_side),
        "direction": chosen_direction,
        "distance": distance_val,
        "unit": distance_unit
    }
    command_text = full_text_processing(fill_template(choose_template(STRAFING_SUBGROUPS), components)) + "."
    # Map reflects the 'strafe' command intent
    # IMPORTANT: Ensure 'strafe' command is defined in possible_commands.json
    # or modify this map generation if the LLM should output left/right instead.
    expected_map = {
        "command": "strafe",
        "distance": distance_val,
        "distance_unit": distance_unit,
        "direction": chosen_direction
    }
    return command_text, expected_map


def generate_stop_labeled():
    phrase = random.choice(random.choice(STOP_SUBGROUPS))
    command_text = full_text_processing(phrase) + "."
    expected_map = {
        "command": "stop"
        # No parameters needed in the map
    }
    return command_text, expected_map


def generate_labeled_command_and_map(command_type):
    """Dispatches to the correct generator function."""
    if command_type == "forward":
        return generate_forward_labeled()
    elif command_type == "back":
        return generate_back_labeled()
    elif command_type in ["left", "right"]:
        # This generates 'left' or 'right' command maps
        return generate_side_labeled(command_type)
    elif command_type == "rotate":
        return generate_rotate_labeled()
    elif command_type == "stop":
        return generate_stop_labeled()
    elif command_type == "strafing":
        # This generates 'strafe' command maps
        return generate_strafing_labeled()
    else:
        # Default to stop if type is unknown
        print(f"Warning: Unknown command type '{command_type}' requested.")
        return generate_stop_labeled()


# =========================================================================================
#    4. High-Level Dispatch and Data Generation (3 Files)
# =========================================================================================

if __name__ == "__main__":
    num_samples = 10_000  # Number of samples to generate
    valid_command_types = ["forward", "back", "left", "right", "rotate", "stop", "strafing"]

    # Output file names
    text_output_file = "synthetic_basic_labeled_commands.txt"
    map_output_file = "synthetic_basic_labeled_commands_map.jsonl"  # JSON Lines format
    final_json_output_file = "synthetic_basic_labeled_commands_final.jsonl"  # JSON Lines format

    # Instantiate the processor ONCE
    try:
        # Assumes possible_commands.json is located correctly relative to
        # where CommandProcessor class calculates its default path.
        processor = CommandProcessor()
    except Exception as e:
        print(f"FATAL: Could not initialize CommandProcessor: {e}")
        exit(1)  # Stop if processor cannot be initialized

    print(f"Generating {num_samples} samples...")

    # Use 'with' statements for automatic file closing
    try:
        with open(text_output_file, "w", encoding="utf-8") as f_text, \
                open(map_output_file, "w", encoding="utf-8") as f_map, \
                open(final_json_output_file, "w", encoding="utf-8") as f_final:

            processed_count = 0
            error_count = 0
            generation_errors = 0  # Track errors during map generation itself

            for i in range(num_samples):
                try:
                    cmd_type = random.choice(valid_command_types)
                    command_text, expected_map = generate_labeled_command_and_map(cmd_type)

                    # Process the map to get the final JSON
                    final_json_str, error_msg = processor.process_command(expected_map)

                    # Write the command text
                    f_text.write(command_text + "\n")

                    # Write the expected map (as a JSON line)
                    f_map.write(json.dumps(expected_map) + "\n")

                    # Write the final JSON or an error object
                    if error_msg:
                        error_count += 1
                        # Write an error object instead of the command JSON
                        error_obj = {"error": error_msg, "input_map": expected_map}
                        f_final.write(json.dumps(error_obj) + "\n")
                        if error_count < 10:  # Print first few processing errors
                            print(f"Error processing sample {i + 1}: {error_msg} for map: {expected_map}")
                    else:
                        f_final.write(final_json_str + "\n")
                        processed_count += 1

                except Exception as gen_e:
                    # Catch errors during the generation step itself
                    generation_errors += 1
                    print(f"ERROR generating sample {i + 1} (type: {cmd_type}): {gen_e}")
                    # Optionally write placeholders or skip lines in output files
                    try:
                        f_text.write(f"GENERATION_ERROR: {gen_e}\n")
                        f_map.write(json.dumps({"error": "Generation failed", "details": str(gen_e)}) + "\n")
                        f_final.write(json.dumps({"error": "Generation failed", "details": str(gen_e)}) + "\n")
                    except Exception as write_e:
                        print(f"ERROR writing generation error to file: {write_e}")

                if (i + 1) % 1000 == 0:
                    print(f"  ... processed {i + 1}/{num_samples} samples.")

            print("\nData generation complete.")
            print(f"Successfully generated text/map pairs for {num_samples - generation_errors} samples.")
            print(f"Successfully processed map to final JSON for {processed_count} samples.")
            print(f"Encountered generation errors in {generation_errors} samples.")
            print(f"Encountered processing errors in {error_count} samples (see {final_json_output_file} for details).")
            print("Output files:")
            print(f" - Command Text: {text_output_file}")
            print(f" - Expected LLM Map (JSON Lines): {map_output_file}")
            print(f" - Final Validated JSON / Errors (JSON Lines): {final_json_output_file}")

    except IOError as e:
        print(f"FATAL: An error occurred writing to output files: {e}")
    except Exception as e:
        print(f"FATAL: An unexpected error occurred during generation loop: {e}")
