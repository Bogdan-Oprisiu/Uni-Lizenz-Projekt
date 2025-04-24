import json
import math
import random
import re  # Make sure re is imported

# =============================================================================
# START: Import Classes (Assuming they are in utils/command_processor.py)
# =============================================================================
try:
    # Adjust the path if your file/classes have different names
    # This assumes CommandProcessor is in utils/generate_json_command.py as per user's last script
    from utils.generate_json_command import CommandProcessor

    print("Successfully imported CommandProcessor from utils.")
except ImportError as e:
    print(f"FATAL: Could not import CommandProcessor from utils: {e}")
    print("Please ensure CommandProcessor and UnitConverter classes are defined")
    print("in a file accessible via the 'utils' package (e.g., utils/generate_json_command.py)")
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
        "move {direction} calmly {distance}{unit}",  # Using move instead of strafe here
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
# Removed STRAFING_SUBGROUPS as per unification
STOP_SUBGROUPS = [
    [
        "stop",
        "halt"
    ],
    [
        "please stop",
        "cease movement",
        "stop moving",
        "bring it to a halt"
    ],
]

# =========================================================================================
#    2. Synonyms and Helper Functions (No changes needed here)
# =========================================================================================
directions_forward = ["forward", "ahead", "advance", "proceed", "go ahead", "straight"]
directions_back = ["back", "reverse", "backward", "retreat", "recede"]
rotate_directions = ["left", "right", "port", "starboard"]
side_directions = ["left", "right", "port", "starboard"]  # Used for text generation in side commands
verbs_forward = ["move", "go", "advance", "proceed", "head"]
verbs_back = ["move", "go", "reverse", "retreat", "step", "recede"]
verbs_side = ["move", "slide", "shift", "strafe"]  # Keep 'strafe' as a possible verb in text
verbs_rotate = ["rotate", "turn", "spin", "pivot"]

# Define standard units used IN TEXT generation primarily
DISTANCE_UNIT_TEXT = "cm"
LINEAR_ACCEL_UNIT_TEXT = "cm/s^2"


# Angular units can vary in text generation

def random_distance(unit=DISTANCE_UNIT_TEXT):
    """Generate distance and return value and unit."""
    val = round(random.uniform(10, 500), 1)
    return val, unit


def random_angle(unit='rad'):
    """Generate angle and return value and unit."""
    if unit == 'rad':
        val = round(random.uniform(-math.pi, math.pi), 4)  # Full circle range
    elif unit == 'deg':
        val = round(random.uniform(-180, 180), 1)  # Full circle range
    else:  # Default to radians
        val = round(random.uniform(-math.pi, math.pi), 4)
        unit = 'rad'
    return val, unit


def random_linear_accel(unit=LINEAR_ACCEL_UNIT_TEXT):
    """Generate linear acceleration value and unit."""
    val = round(random.uniform(0.1, 50.0), 2)
    # Could add logic here to sometimes generate m/s^2 if needed
    return val, unit


def random_angular_accel(unit='rad/s^2'):
    """Generate angular acceleration value and unit."""
    if unit == 'rad/s^2':
        val = round(random.uniform(0.01, 3.0), 4)
    elif unit == 'deg/s^2':
        val = round(random.uniform(1.0, 180.0), 2)  # Approx range equivalent to 3 rad/s^2
    else:  # Default to rad/s^2
        val = round(random.uniform(0.01, 3.0), 4)
        unit = 'rad/s^2'
    return val, unit


def fill_template(template, components):
    """Replace placeholders in the template with values from components."""
    formatted_components = components.copy()
    # Format numeric values for the text string representation
    if 'distance' in formatted_components:
        formatted_components['distance'] = f"{formatted_components['distance']:.1f}"
    if 'angle' in formatted_components:
        if formatted_components.get('unit') == 'deg':
            formatted_components['angle'] = f"{formatted_components['angle']:.1f}"
        else:  # Assume rad
            formatted_components['angle'] = f"{formatted_components['angle']:.4f}"
    if 'accel_val' in formatted_components and formatted_components['accel_val'] is not None:
        # Format acceleration based on its type/unit for text
        accel_unit = formatted_components.get('accel_unit_text', '')
        if 'deg' in accel_unit:
            formatted_components['accel_val'] = f"{formatted_components['accel_val']:.2f}"
        elif 'rad' in accel_unit:
            formatted_components['accel_val'] = f"{formatted_components['accel_val']:.4f}"
        else:  # Assume linear cm/s^2
            formatted_components['accel_val'] = f"{formatted_components['accel_val']:.2f}"
    else:
        # Ensure keys exist even if None to avoid KeyError in format
        formatted_components['accel_val'] = ''
        formatted_components['accel_unit_text'] = ''

    return template.format(**formatted_components)


def choose_template(subgroup_list):
    """Randomly choose a subgroup, then a template from that subgroup."""
    subgroup = random.choice(subgroup_list)
    return random.choice(subgroup)


# =========================================================================================
#    3. Labeled Generation Functions (Returning text and map, including acceleration)
# =========================================================================================

def generate_forward_labeled():
    distance_val, distance_unit = random_distance()
    use_accel = random.random() < 0.3
    accel_val, accel_unit = (None, None)
    accel_text_suffix = ""
    expected_map = {"command": "forward", "distance": distance_val, "distance_unit": distance_unit}

    if use_accel:
        accel_val, accel_unit = random_linear_accel()
        accel_text_suffix = f" with acceleration {{accel_val}}{accel_unit}"  # Use accel_unit directly
        expected_map["acceleration"] = accel_val
        expected_map["acceleration_unit"] = accel_unit

    components = {
        "verb": random.choice(verbs_forward),
        "direction": random.choice(directions_forward),
        "distance": distance_val,
        "unit": distance_unit,
        "accel_val": accel_val,
        "accel_unit_text": accel_unit  # Pass unit directly for formatting
    }
    template = choose_template(FORWARD_SUBGROUPS) + accel_text_suffix
    command_text = full_text_processing(fill_template(template, components)) + "."

    return command_text, expected_map


def generate_back_labeled():
    distance_val, distance_unit = random_distance()
    use_accel = random.random() < 0.3
    accel_val, accel_unit = (None, None)
    accel_text_suffix = ""
    expected_map = {"command": "back", "distance": distance_val, "distance_unit": distance_unit}

    if use_accel:
        accel_val, accel_unit = random_linear_accel()
        accel_text_suffix = f" with acceleration {{accel_val}}{accel_unit}"
        expected_map["acceleration"] = accel_val
        expected_map["acceleration_unit"] = accel_unit

    components = {
        "verb": random.choice(verbs_back),
        "direction": random.choice(directions_back),
        "distance": distance_val,
        "unit": distance_unit,
        "accel_val": accel_val,
        "accel_unit_text": accel_unit
    }
    template = choose_template(BACK_SUBGROUPS) + accel_text_suffix
    command_text = full_text_processing(fill_template(template, components)) + "."

    return command_text, expected_map


def generate_side_labeled(command_type):  # command_type is 'left' or 'right'
    distance_val, distance_unit = random_distance()
    use_accel = random.random() < 0.3
    accel_val, accel_unit = (None, None)
    accel_text_suffix = ""
    # Map uses the canonical 'left' or 'right' command
    expected_map = {"command": command_type, "distance": distance_val, "distance_unit": distance_unit}

    if use_accel:
        accel_val, accel_unit = random_linear_accel()
        accel_text_suffix = f" with acceleration {{accel_val}}{accel_unit}"
        expected_map["acceleration"] = accel_val
        expected_map["acceleration_unit"] = accel_unit

    # Choose text direction (can include port/starboard)
    text_direction = random.choice([d for d in side_directions if
                                    (d in ['left', 'port'] and command_type == 'left') or (
                                            d in ['right', 'starboard'] and command_type == 'right')])

    components = {
        "verb": random.choice(verbs_side),
        "direction": text_direction,
        "distance": distance_val,
        "unit": distance_unit,
        "accel_val": accel_val,
        "accel_unit_text": accel_unit
    }
    template = choose_template(SIDE_SUBGROUPS) + accel_text_suffix
    command_text = full_text_processing(fill_template(template, components)) + "."

    return command_text, expected_map


# Fragment relevant din data_gen_3_files_accel_imported
def generate_rotate_labeled():
    # 1. Generează unghiul și unitatea sa PENTRU TEXT
    use_deg_in_text = random.random() < 0.3
    angle_unit_text = 'deg' if use_deg_in_text else 'rad'
    # random_angle returnează valoarea ȘI unitatea CORECTĂ a acelei valori
    angle_val, angle_unit_map = random_angle(unit=angle_unit_text)

    chosen_direction = random.choice(rotate_directions)
    use_accel = random.random() < 0.3
    accel_val, accel_unit = (None, None)  # Inițializare
    accel_text_suffix = ""
    # 2. Creează harta INIȚIALĂ doar cu unghiul
    expected_map = {
        "command": "rotate",
        "angle": angle_val,  # Valoarea generată
        "angle_unit": angle_unit_map,  # Unitatea CORECTĂ a valorii generate
        "direction": chosen_direction
    }

    if use_accel:
        use_deg_accel_in_text = random.random() < 0.3
        accel_unit_text = 'deg/s^2' if use_deg_accel_in_text else 'rad/s^2'

        accel_val, accel_unit_map = random_angular_accel(unit=accel_unit_text)
        accel_text_suffix = f" with angular acceleration {{accel_val}}{accel_unit_text}"

        expected_map["acceleration"] = accel_val
        expected_map["acceleration_unit"] = accel_unit_map

    components = {
        "verb": random.choice(verbs_rotate),
        "direction": chosen_direction,
        "angle": angle_val,
        "unit": angle_unit_text,
        "accel_val": accel_val,
        "accel_unit_text": accel_unit_text if use_accel else ""
    }
    template = choose_template(ROTATE_SUBGROUPS) + accel_text_suffix
    command_text = full_text_processing(fill_template(template, components)) + "."

    return command_text, expected_map


def generate_stop_labeled():
    phrase = random.choice(random.choice(STOP_SUBGROUPS))
    command_text = full_text_processing(phrase) + "."
    expected_map = {"command": "stop"}
    return command_text, expected_map


def generate_labeled_command_and_map(command_type):
    """Dispatches to the correct generator function."""
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
    # Removed 'strafing' case
    else:
        print(f"Warning: Unknown command type '{command_type}' requested.")
        return generate_stop_labeled()  # Default to stop


# =========================================================================================
#    4. High-Level Dispatch and Data Generation (3 Files)
# =========================================================================================

if __name__ == "__main__":
    num_samples = 10_000  # Number of samples to generate
    # Adjusted valid types - removed 'strafing'
    valid_command_types = ["forward", "back", "left", "right", "rotate", "stop"]

    # Output file names
    text_output_file = "synthetic_accel_labeled_commands.txt"
    map_output_file = "synthetic_accel_labeled_commands_map.jsonl"  # JSON Lines format
    final_json_output_file = "synthetic_accel_labeled_commands_final.jsonl"  # JSON Lines format

    # Instantiate the processor ONCE
    try:
        processor = CommandProcessor()
    except Exception as e:
        print(f"FATAL: Could not initialize CommandProcessor: {e}")
        exit(1)

    print(f"Generating {num_samples} samples with acceleration...")

    # Use 'with' statements for automatic file closing
    try:
        with open(text_output_file, "w", encoding="utf-8") as f_text, \
                open(map_output_file, "w", encoding="utf-8") as f_map, \
                open(final_json_output_file, "w", encoding="utf-8") as f_final:

            processed_count = 0
            error_count = 0
            generation_errors = 0

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
                        error_obj = {"error": error_msg, "input_map": expected_map}
                        f_final.write(json.dumps(error_obj) + "\n")
                        if error_count < 10:
                            print(f"Error processing sample {i + 1}: {error_msg} for map: {expected_map}")
                    else:
                        f_final.write(final_json_str + "\n")
                        processed_count += 1

                except Exception as gen_e:
                    generation_errors += 1
                    print(f"ERROR generating sample {i + 1} (type: {cmd_type}): {gen_e}")
                    try:  # Try to log generation error to files
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
