import json

import os

from utils.unit_converter import UnitConverter

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_definitions_path = os.path.join(script_dir, '..', 'possible_commands.json')
    default_definitions_path = os.path.normpath(default_definitions_path)
except NameError:
    # Fallback if __file__ is not defined (e.g., running in an interactive environment)
    print("Warning: __file__ not defined. Assuming 'possible_commands.json' is in parent directory.")
    default_definitions_path = os.path.normpath(os.path.join('.', '..', 'possible_commands.json'))


class CommandProcessor:
    """
    Validates input parameters against command definitions and generates
    a standardized JSON command string for a robot, using UnitConverter.
    """

    def __init__(self, definitions_path=default_definitions_path):
        """
        Initializes the CommandProcessor by loading command definitions
        and creating a UnitConverter instance.

        Args:
            definitions_path (str): The path to the JSON file containing
                                     command definitions. Defaults to a path
                                     calculated relative to this script file.
        """
        print(f"Attempting to load definitions from: {definitions_path}")  # Debug print
        if not os.path.exists(definitions_path):
            # Try one more common location: current working directory
            alt_path = "possible_commands.json"
            if os.path.exists(alt_path):
                print(f"Definitions not found at {definitions_path}, using {alt_path} instead.")
                definitions_path = alt_path
            else:
                raise FileNotFoundError(f"Command definitions file not found at '{definitions_path}' or '{alt_path}'")

        try:
            with open(definitions_path, 'r', encoding='utf-8') as f:
                self.definitions = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {definitions_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Could not load definitions from {definitions_path}: {e}")

        # Pre-process definitions
        self._commands = {cmd['name']: cmd for cmd in self.definitions.get('commandLanguage', {}).get('commands', [])}
        self._errors = {err['code']: err['description'] for err in
                        self.definitions.get('errors', {}).get('definitions', [])}

        # Add default error descriptions if missing from JSON
        # These codes are returned by UnitConverter
        if "UNKNOWN_UNIT" not in self._errors:
            self._errors["UNKNOWN_UNIT"] = "An unknown unit was provided."
        if "UNIT_MISMATCH" not in self._errors:
            self._errors["UNIT_MISMATCH"] = "Cannot convert between incompatible unit types (e.g., length to angle)."
        # These might be defined in JSON but ensure they exist
        if "INTERNAL_ERROR" not in self._errors:
            self._errors["INTERNAL_ERROR"] = "An internal error occurred during processing."
        if "INVALID_PARAMETER_TYPE" not in self._errors:
            self._errors["INVALID_PARAMETER_TYPE"] = "The parameter value type is incorrect."

        if not self._commands:
            raise ValueError("No commands found in the definitions file.")

        # Create an instance of the UnitConverter
        self.converter = UnitConverter()

    def _get_error_description(self, code, param_name=None, cmd_name=None):
        """Helper to format error messages using defined codes."""
        base_desc = self._errors.get(code, f"Unknown error code: {code}")
        details = []
        if cmd_name:
            details.append(f"Command: '{cmd_name}'")
        if param_name:
            details.append(f"Parameter: '{param_name}'")
        return f"{code}: {base_desc}" + (f" ({', '.join(details)})" if details else "")

    def process_command(self, input_params):
        """
        Validates input parameters and generates the final JSON command.

        Args:
            input_params (dict): A dictionary representing the abstract command
                                 extracted by the LLM. Expected keys include
                                 'command' (str) and keys matching parameter
                                 names (e.g., 'angle', 'distance'). Optional
                                 unit keys (e.g., 'angle_unit') can be included.

        Returns:
            tuple: A tuple containing:
                   - str: The validated and standardized JSON command string, or None if validation fails.
                   - str: An error message string, or None if validation succeeds.
        """
        if not isinstance(input_params, dict):
            return None, self._get_error_description("INVALID_PARAMETER_TYPE", "input_params (must be dict)")

        command_name = input_params.get("command")

        # 1. Validate Command Name
        if not command_name or command_name not in self._commands:
            return None, self._get_error_description("INVALID_COMMAND", cmd_name=command_name or "None")

        command_def = self._commands[command_name]
        defined_params = command_def.get("parameters", {})
        final_json_params = {}

        # 2. Validate Parameters
        for param_name, param_def in defined_params.items():
            is_required = param_def.get("required", False)
            expected_type = param_def.get("type")
            standard_unit = param_def.get("unit")  # The unit expected in the final JSON
            default_value = param_def.get("default")
            allowed_values = param_def.get("enum")

            input_value = input_params.get(param_name)
            input_unit_key = f"{param_name}_unit"
            input_unit = input_params.get(input_unit_key)  # Unit provided by LLM

            # 2a. Missing Required Parameters
            if is_required and input_value is None:
                if default_value is not None:
                    # Use default value, ensuring it matches expected type
                    if expected_type == "number":
                        try:
                            final_json_params[param_name] = float(default_value)
                        except (ValueError, TypeError):
                            print(f"Warning: Default value '{default_value}' for '{param_name}' is not a valid number.")
                            # Decide how to handle: skip, error out, or use None? Let's skip for now.
                            continue
                    elif expected_type == "string":
                        final_json_params[param_name] = str(default_value)
                    else:
                        final_json_params[param_name] = default_value  # Use as is for other types

                    continue  # Skip further checks for this param, use default
                else:
                    # Required, no input, no default -> Error
                    return None, self._get_error_description("MISSING_PARAMETER", param_name, command_name)

            # 2b. Optional Parameters without Input
            elif not is_required and input_value is None:
                if default_value is not None:
                    # Use default value, ensuring it matches expected type
                    if expected_type == "number":
                        try:
                            final_json_params[param_name] = float(default_value)
                        except (ValueError, TypeError):
                            print(f"Warning: Default value '{default_value}' for '{param_name}' is not a valid number.")
                            continue  # Skip this optional param if default is bad
                    elif expected_type == "string":
                        final_json_params[param_name] = str(default_value)
                    else:
                        final_json_params[param_name] = default_value
                # Else: Optional, no input, no default -> just skip the parameter
                continue  # Parameter is optional and not provided, skip validation

            # --- Parameter is present, proceed with validation ---
            validated_value = input_value

            # 2c. Type Validation (Initial check before conversion)
            if expected_type == "number":
                try:
                    # Check if it *can* be a float, but don't convert yet
                    float(input_value)
                except (ValueError, TypeError):
                    return None, self._get_error_description("INVALID_PARAMETER_TYPE", param_name, command_name)

            elif expected_type == "string":
                if not isinstance(input_value, str):
                    return None, self._get_error_description("INVALID_PARAMETER_TYPE", param_name, command_name)
                validated_value = str(input_value)  # Ensure string type
                # Enum validation for strings
                if allowed_values and validated_value not in allowed_values:
                    return None, self._get_error_description("INVALID_PARAMETER_VALUE", param_name, command_name)

            # --- Add other type checks ---

            # 2d. Unit Conversion (using UnitConverter)
            if expected_type == "number" and standard_unit:
                # If LLM provided a unit and it's different from the standard
                if input_unit and self.converter._normalize_unit(input_unit) != self.converter._normalize_unit(
                        standard_unit):
                    converted_value, conv_error = self.converter.convert(
                        value=input_value,
                        from_unit=input_unit,
                        to_unit=standard_unit
                    )
                    if conv_error:
                        # Use the error code returned by the converter
                        return None, self._get_error_description(conv_error, param_name, command_name)
                    validated_value = converted_value
                else:
                    # No input unit OR input unit matches standard, just ensure type is float
                    try:
                        validated_value = float(input_value)
                    except (ValueError, TypeError):
                        # Should have been caught earlier, but double-check
                        return None, self._get_error_description("INVALID_PARAMETER_TYPE", param_name, command_name)

            # 2e. Store Validated Value
            if isinstance(validated_value, float):
                # Round numbers to avoid excessive precision from conversions
                final_json_params[param_name] = round(validated_value, 4)
            else:
                final_json_params[param_name] = validated_value  # Store strings etc.

        # 3. Check for Extraneous Input Parameters
        for input_key in input_params:
            if input_key not in defined_params and input_key != 'command' and not input_key.endswith('_unit'):
                print(f"Warning: Extraneous parameter '{input_key}' provided for command '{command_name}'.")

        # 4. Generate Final JSON
        final_command_obj = {
            "command": command_name,
            "parameters": final_json_params
        }

        try:
            final_json_string = json.dumps(final_command_obj)
            return final_json_string, None
        except TypeError as e:
            return None, self._get_error_description("INTERNAL_ERROR",
                                                     cmd_name=command_name) + f" (JSON serialization failed: {e})"


# =============================================================================
# Example Usage (for CommandProcessor)
# =============================================================================
if __name__ == "__main__":
    # Instantiate WITHOUT argument to use the calculated default path
    processor = CommandProcessor()

    print("--- CommandProcessor Tests (with UnitConverter) ---")

    # Test 1: Valid Rotate (deg -> rad) - Same as before
    input1 = {"command": "rotate", "angle": 90, "angle_unit": "deg", "direction": "right"}
    json_out1, error1 = processor.process_command(input1)
    print(f"Input 1: {input1}\nOutput 1 JSON: {json_out1}\nOutput 1 Error: {error1}\n")
    # Expected: {"command": "rotate", "parameters": {"angle": 1.5708, "direction": "right", "acceleration": 0.5}}

    # Test 2: Valid Forward (cm -> cm) - Same as before
    input2 = {"command": "forward", "distance": 150.5, "distance_unit": "cm", "acceleration": 5.0}
    json_out2, error2 = processor.process_command(input2)
    print(f"Input 2: {input2}\nOutput 2 JSON: {json_out2}\nOutput 2 Error: {error2}\n")
    # Expected: {"command": "forward", "parameters": {"distance": 150.5, "acceleration": 5.0}}

    # Test 3: Valid Stop - Same as before
    input3 = {"command": "stop"}
    json_out3, error3 = processor.process_command(input3)
    print(f"Input 3: {input3}\nOutput 3 JSON: {json_out3}\nOutput 3 Error: {error3}\n")
    # Expected: {"command": "stop", "parameters": {}}

    # Test 4: Missing Required (distance) - Same as before
    input4 = {"command": "forward", "acceleration": 10}
    json_out4, error4 = processor.process_command(input4)
    print(f"Input 4: {input4}\nOutput 4 JSON: {json_out4}\nOutput 4 Error: {error4}\n")
    # Expected: Error MISSING_PARAMETER

    # Test 5: Invalid Command - Same as before
    input5 = {"command": "fly", "height": 100}
    json_out5, error5 = processor.process_command(input5)
    print(f"Input 5: {input5}\nOutput 5 JSON: {json_out5}\nOutput 5 Error: {error5}\n")
    # Expected: Error INVALID_COMMAND

    # Test 6: Invalid Type (angle) - Same as before
    input6 = {"command": "rotate", "angle": "ninety", "angle_unit": "deg", "direction": "left"}
    json_out6, error6 = processor.process_command(input6)
    print(f"Input 6: {input6}\nOutput 6 JSON: {json_out6}\nOutput 6 Error: {error6}\n")
    # Expected: Error INVALID_PARAMETER_TYPE

    # Test 7: Invalid Enum (direction) - Same as before
    input7 = {"command": "rotate", "angle": 1.0, "direction": "up"}
    json_out7, error7 = processor.process_command(input7)
    print(f"Input 7: {input7}\nOutput 7 JSON: {json_out7}\nOutput 7 Error: {error7}\n")
    # Expected: Error INVALID_PARAMETER_VALUE

    # --- New/Updated Tests using UnitConverter ---

    # Test 8: Unit Conversion (m -> cm) - Should now work
    input8 = {"command": "forward", "distance": 1.5, "distance_unit": "m"}
    json_out8, error8 = processor.process_command(input8)
    print(f"Input 8: {input8}\nOutput 8 JSON: {json_out8}\nOutput 8 Error: {error8}\n")
    # Expected: {"command": "forward", "parameters": {"distance": 150.0, "acceleration": 10.0}} (with default accel)

    # Test 9: Extraneous Parameter - Same as before
    input9 = {"command": "forward", "distance": 50, "color": "red"}
    json_out9, error9 = processor.process_command(input9)
    print(f"Input 9: {input9}\nOutput 9 JSON: {json_out9}\nOutput 9 Error: {error9}\n")
    # Expected: Warning printed, JSON: {"command": "forward", "parameters": {"distance": 50.0, "acceleration": 10.0}}

    # Test 10: Imperial Unit (inch -> cm)
    input10 = {"command": "forward", "distance": 10, "distance_unit": "inch"}
    json_out10, error10 = processor.process_command(input10)
    print(f"Input 10: {input10}\nOutput 10 JSON: {json_out10}\nOutput 10 Error: {error10}\n")
    # Expected: {"command": "forward", "parameters": {"distance": 25.4, "acceleration": 10.0}}

    # Test 11: Imperial Unit (feet -> cm)
    input11 = {"command": "back", "distance": 2, "distance_unit": "ft"}
    json_out11, error11 = processor.process_command(input11)
    print(f"Input 11: {input11}\nOutput 11 JSON: {json_out11}\nOutput 11 Error: {error11}\n")
    # Expected: {"command": "back", "parameters": {"distance": 60.96, "acceleration": 10.0}}

    # Test 12: Unit Mismatch (cm -> rad)
    input12 = {"command": "rotate", "angle": 100, "angle_unit": "cm", "direction": "left"}
    json_out12, error12 = processor.process_command(input12)
    print(f"Input 12: {input12}\nOutput 12 JSON: {json_out12}\nOutput 12 Error: {error12}\n")
    # Expected: Error UNIT_MISMATCH

    # Test 13: Unknown Unit
    input13 = {"command": "forward", "distance": 5, "distance_unit": "cubit"}
    json_out13, error13 = processor.process_command(input13)
    print(f"Input 13: {input13}\nOutput 13 JSON: {json_out13}\nOutput 13 Error: {error13}\n")
    # Expected: Error UNKNOWN_UNIT

    # Test 14: Acceleration Unit Conversion (deg/s^2 -> rad/s^2)
    input14 = {"command": "rotate", "angle": 1.0, "direction": "left", "acceleration": 57.2958,
               "acceleration_unit": "deg/s^2"}
    json_out14, error14 = processor.process_command(input14)
    print(f"Input 14: {input14}\nOutput 14 JSON: {json_out14}\nOutput 14 Error: {error14}\n")
    # Expected: {"command": "rotate", "parameters": {"angle": 1.0, "direction": "left", "acceleration": 1.0}} (approx)

    # Test 15: Linear Acceleration Unit Conversion (m/s^2 -> cm/s^2)
    input15 = {"command": "forward", "distance": 200, "acceleration": 0.5, "acceleration_unit": "m/s^2"}
    json_out15, error15 = processor.process_command(input15)
    print(f"Input 15: {input15}\nOutput 15 JSON: {json_out15}\nOutput 15 Error: {error15}\n")
    # Expected: {"command": "forward", "parameters": {"distance": 200.0, "acceleration": 50.0}}

    # Test 16: Input unit matches standard unit (rad)
    input16 = {"command": "rotate", "angle": 0.7854, "angle_unit": "rad", "direction": "port"}
    json_out16, error16 = processor.process_command(input16)
    print(f"Input 16: {input16}\nOutput 16 JSON: {json_out16}\nOutput 16 Error: {error16}\n")
    # Expected: {"command": "rotate", "parameters": {"angle": 0.7854, "direction": "port", "acceleration": 0.5}}

    # Test 17: No unit provided for number, assume standard (cm for distance)
    input17 = {"command": "forward", "distance": 75}
    json_out17, error17 = processor.process_command(input17)
    print(f"Input 17: {input17}\nOutput 17 JSON: {json_out17}\nOutput 17 Error: {error17}\n")
    # Expected: {"command": "forward", "parameters": {"distance": 75.0, "acceleration": 10.0}}

    # Test 18: No unit provided for angle, assume standard (rad)
    input18 = {"command": "rotate", "angle": -0.5, "direction": "right"}
    json_out18, error18 = processor.process_command(input18)
    print(f"Input 18: {input18}\nOutput 18 JSON: {json_out18}\nOutput 18 Error: {error18}\n")
    # Expected: {"command": "rotate", "parameters": {"angle": -0.5, "direction": "right", "acceleration": 0.5}}
