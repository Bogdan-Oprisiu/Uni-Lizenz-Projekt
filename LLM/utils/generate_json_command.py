import json
import os

# Import the necessary UnitConverter class
try:
    from utils.unit_converter import UnitConverter

    print("Successfully imported UnitConverter from utils.")
except ImportError as e:
    print(f"FATAL: Could not import UnitConverter from utils: {e}")
    print("Please ensure UnitConverter class is defined in utils/unit_converter.py")
    exit(1)

# Path calculation logic (using a robust method)
try:
    # Assumes this script file is where the CommandProcessor class definition resides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path relative to this script file: go up one level, find the JSON
    default_definitions_path = os.path.join(script_dir, '..', 'possible_commands.json')
    default_definitions_path = os.path.normpath(default_definitions_path)
except NameError:
    # Fallback if __file__ is not defined (e.g., running in an interactive environment like a REPL)
    print("Warning: __file__ not defined. Assuming 'possible_commands.json' is in parent directory relative to CWD.")
    # Relative to current working directory in this fallback case
    default_definitions_path = os.path.normpath(os.path.join('.', '..', 'possible_commands.json'))


class CommandProcessor:
    """
    Validates input parameters against command definitions (including aliases)
    and generates a standardized JSON command string for a robot, using UnitConverter.
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
        print(f"CommandProcessor: Attempting to load definitions from: {definitions_path}")  # Debug print
        if not os.path.exists(definitions_path):
            # Try one more common location: current working directory if relative path failed
            alt_path = "possible_commands.json"
            if os.path.exists(alt_path):
                print(f"Definitions not found at '{definitions_path}', using '{alt_path}' from CWD instead.")
                definitions_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Command definitions file not found at '{definitions_path}' or '{alt_path}' (relative to CWD).")

        try:
            with open(definitions_path, 'r', encoding='utf-8') as f:
                self.definitions = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {definitions_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Could not load definitions from {definitions_path}: {e}")

        # Pre-process definitions for easier lookup, including aliases
        self._commands = {}
        self._command_aliases = {}
        for cmd_def in self.definitions.get('commandLanguage', {}).get('commands', []):
            cmd_name = cmd_def.get('name')
            if cmd_name:
                self._commands[cmd_name] = cmd_def
                # Map aliases back to the canonical command name
                for alias in cmd_def.get('aliases', []):
                    # Optional: Check for alias conflicts here if needed
                    self._command_aliases[alias.lower()] = cmd_name  # Store aliases lowercase

        self._errors = {err['code']: err['description'] for err in
                        self.definitions.get('errors', {}).get('definitions', [])}

        # Add default error descriptions if missing from JSON
        if "UNKNOWN_UNIT" not in self._errors: self._errors["UNKNOWN_UNIT"] = "An unknown unit was provided."
        if "UNIT_MISMATCH" not in self._errors: self._errors["UNIT_MISMATCH"] = "Incompatible unit types."
        if "INTERNAL_ERROR" not in self._errors: self._errors["INTERNAL_ERROR"] = "Internal processing error."
        if "INVALID_PARAMETER_TYPE" not in self._errors: self._errors[
            "INVALID_PARAMETER_TYPE"] = "Incorrect parameter type."
        if "INVALID_COMMAND" not in self._errors: self._errors["INVALID_COMMAND"] = "Unknown command specified."
        if "MISSING_PARAMETER" not in self._errors: self._errors["MISSING_PARAMETER"] = "Required parameter missing."
        if "INVALID_PARAMETER_VALUE" not in self._errors: self._errors[
            "INVALID_PARAMETER_VALUE"] = "Invalid value for parameter."

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

    def _resolve_command_name(self, input_command_name):
        """Finds the canonical command name, checking aliases."""
        if not input_command_name:
            return None
        # Check direct name first (case-sensitive)
        if input_command_name in self._commands:
            return input_command_name
        # Check aliases (case-insensitive)
        return self._command_aliases.get(input_command_name.lower())

    def process_command(self, input_params):
        """
        Validates input parameters and generates the final JSON command.
        Handles command aliases and uses UnitConverter for unit conversions.

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

        input_command_name = input_params.get("command")

        # 1. Resolve and Validate Command Name (Handles Aliases)
        command_name = self._resolve_command_name(input_command_name)
        if not command_name:
            # Pass the original input name to the error message for clarity
            return None, self._get_error_description("INVALID_COMMAND", cmd_name=input_command_name or "None")

        command_def = self._commands[command_name]  # Use canonical name to get definition
        defined_params = command_def.get("parameters", {})
        final_json_params = {}

        # 2. Validate Parameters (Loop remains largely the same)
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
                            continue
                    elif expected_type == "string":
                        final_json_params[param_name] = str(default_value)
                    else:
                        final_json_params[param_name] = default_value
                    continue
                else:
                    return None, self._get_error_description("MISSING_PARAMETER", param_name, command_name)

            # 2b. Optional Parameters without Input
            elif not is_required and input_value is None:
                if default_value is not None:
                    if expected_type == "number":
                        try:
                            final_json_params[param_name] = float(default_value)
                        except (ValueError, TypeError):
                            print(f"Warning: Default value '{default_value}' for '{param_name}' is not a valid number.")
                            continue
                    elif expected_type == "string":
                        final_json_params[param_name] = str(default_value)
                    else:
                        final_json_params[param_name] = default_value
                continue

            # --- Parameter is present ---
            validated_value = input_value

            # 2c. Type Validation (Initial check)
            if expected_type == "number":
                try:
                    float(input_value)
                except (ValueError, TypeError):
                    return None, self._get_error_description("INVALID_PARAMETER_TYPE", param_name, command_name)
            elif expected_type == "string":
                if not isinstance(input_value, str): return None, self._get_error_description("INVALID_PARAMETER_TYPE",
                                                                                              param_name, command_name)
                validated_value = str(input_value)
                if allowed_values and validated_value not in allowed_values: return None, self._get_error_description(
                    "INVALID_PARAMETER_VALUE", param_name, command_name)

            # --- Add other type checks ---

            # 2d. Unit Conversion (using UnitConverter)
            if expected_type == "number" and standard_unit:
                if input_unit and self.converter._normalize_unit(input_unit) != self.converter._normalize_unit(
                        standard_unit):
                    converted_value, conv_error = self.converter.convert(input_value, input_unit, standard_unit)
                    if conv_error: return None, self._get_error_description(conv_error, param_name, command_name)
                    validated_value = converted_value
                else:
                    try:
                        validated_value = float(input_value)
                    except (ValueError, TypeError):
                        return None, self._get_error_description("INVALID_PARAMETER_TYPE", param_name, command_name)

            # 2e. Store Validated Value
            if isinstance(validated_value, float):
                final_json_params[param_name] = round(validated_value, 4)
            else:
                final_json_params[param_name] = validated_value

        # 3. Check for Extraneous Input Parameters
        for input_key in input_params:
            if input_key not in defined_params and input_key != 'command' and not input_key.endswith('_unit'):
                print(f"Warning: Extraneous parameter '{input_key}' provided for command '{command_name}'.")

        # 4. Generate Final JSON using the CANONICAL command name
        final_command_obj = {
            "command": command_name,  # Use the resolved canonical name
            "parameters": final_json_params
        }

        try:
            final_json_string = json.dumps(final_command_obj)
            return final_json_string, None
        except TypeError as e:
            return None, self._get_error_description("INTERNAL_ERROR",
                                                     cmd_name=command_name) + f" (JSON serialization failed: {e})"


# =============================================================================
# Example Usage (for CommandProcessor) - Should still work
# =============================================================================
if __name__ == "__main__":
    try:
        # Instantiate WITHOUT argument to use the calculated default path
        processor = CommandProcessor()

        print("--- CommandProcessor Tests (with UnitConverter & Aliases) ---")

        # Test 1: Valid Rotate (deg -> rad)
        input1 = {"command": "rotate", "angle": 90, "angle_unit": "deg", "direction": "right"}
        json_out1, error1 = processor.process_command(input1)
        print(f"Input 1: {input1}\nOutput 1 JSON: {json_out1}\nOutput 1 Error: {error1}\n")

        # Test 8: Unit Conversion (m -> cm)
        input8 = {"command": "forward", "distance": 1.5, "distance_unit": "m"}
        json_out8, error8 = processor.process_command(input8)
        print(f"Input 8: {input8}\nOutput 8 JSON: {json_out8}\nOutput 8 Error: {error8}\n")

        # Test 10: Imperial Unit (inch -> cm)
        input10 = {"command": "forward", "distance": 10, "distance_unit": "inch"}
        json_out10, error10 = processor.process_command(input10)
        print(f"Input 10: {input10}\nOutput 10 JSON: {json_out10}\nOutput 10 Error: {error10}\n")

        # Test 12: Unit Mismatch (cm -> rad)
        input12 = {"command": "rotate", "angle": 100, "angle_unit": "cm", "direction": "left"}
        json_out12, error12 = processor.process_command(input12)
        print(f"Input 12: {input12}\nOutput 12 JSON: {json_out12}\nOutput 12 Error: {error12}\n")

        # Test 13: Unknown Unit
        input13 = {"command": "forward", "distance": 5, "distance_unit": "cubit"}
        json_out13, error13 = processor.process_command(input13)
        print(f"Input 13: {input13}\nOutput 13 JSON: {json_out13}\nOutput 13 Error: {error13}\n")

        # Test 14: Acceleration Unit Conversion (deg/s^2 -> rad/s^2)
        input14 = {"command": "rotate", "angle": 1.0, "direction": "left", "acceleration": 57.2958,
                   "acceleration_unit": "deg/s^2"}
        json_out14, error14 = processor.process_command(input14)
        print(f"Input 14: {input14}\nOutput 14 JSON: {json_out14}\nOutput 14 Error: {error14}\n")

        # --- New Tests for Aliases and Unified Lateral ---

        # Test 19: Using an alias for command ('turn' -> 'rotate')
        input19 = {"command": "turn", "angle": -0.5, "angle_unit": "rad", "direction": "starboard"}
        json_out19, error19 = processor.process_command(input19)
        print(f"Input 19: {input19}\nOutput 19 JSON: {json_out19}\nOutput 19 Error: {error19}\n")
        # Expected: {"command": "rotate", "parameters": {"angle": -0.5, "direction": "starboard", "acceleration": 0.5}}

        # Test 20: Using 'strafe left' alias -> 'left' command
        input20 = {"command": "strafe left", "distance": 30, "distance_unit": "cm"}
        json_out20, error20 = processor.process_command(input20)
        print(f"Input 20: {input20}\nOutput 20 JSON: {json_out20}\nOutput 20 Error: {error20}\n")
        # Expected: {"command": "left", "parameters": {"distance": 30.0, "acceleration": 10.0}}

        # Test 21: Using 'strafe' directly (should fail as it's not a command or alias)
        input21 = {"command": "strafe", "distance": 20, "direction": "right"}
        json_out21, error21 = processor.process_command(input21)
        print(f"Input 21: {input21}\nOutput 21 JSON: {json_out21}\nOutput 21 Error: {error21}\n")
        # Expected: Error INVALID_COMMAND


    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error loading definitions: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
