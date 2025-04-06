import re


def infer_default_units(tokens):
    """
    Insert default units after a bare number if the number follows a command word
    and does not already have a recognized unit.

    E.g.:
      - "turn 30" => ["turn", "30", "deg"]
      - "move 50" => ["move", "50", "cm"]
      - "accelerate 10" => ["accelerate", "10", "cm/s^2"]

    Assumptions:
      1) 'tokens' is a list of normalized tokens (lowercased commands, spelled-out
         numbers converted, multi-word units merged, etc.).
      2) If the next token after a number is not in recognized_units, we insert
         the default unit of the last command word we saw.
      3) If we haven't seen a command word yet, or the command isn't known,
         do nothing for that number.

    Returns a modified list of tokens.
    """

    # Map command words to their default units.
    DEFAULT_COMMAND_UNITS = {
        "turn": "deg",
        "rotate": "deg",
        "move": "cm",
        "go": "cm",
        "strafe": "cm",
        "drive": "cm/s",
        "speed": "cm/s",
        "accelerate": "cm/s^2",
    }

    # Set of units considered already specified.
    recognized_units = {
        "cm", "m", "km", "deg", "rad", "cm/s", "m/s", "km/h", "cm/s^2", "m/s^2"
    }

    def is_bare_number(tok: str) -> bool:
        return bool(re.match(r"^\d+(\.\d+)?$", tok))

    new_tokens = []
    last_command = None  # Keep track of the most recent command word.
    i = 0

    while i < len(tokens):
        token_lower = tokens[i].lower()

        # If token is a command word, update context.
        if token_lower in DEFAULT_COMMAND_UNITS:
            last_command = token_lower
            new_tokens.append(tokens[i])
            i += 1

        # If token is a bare number.
        elif is_bare_number(tokens[i]):
            # Look ahead to check if the next token is a recognized unit.
            next_unit = ""
            if i + 1 < len(tokens):
                next_unit = tokens[i + 1].lower()

            if next_unit in recognized_units:
                new_tokens.append(tokens[i])
                i += 1
            else:
                # If we have a command context, insert the default unit.
                if last_command is not None:
                    default_unit = DEFAULT_COMMAND_UNITS.get(last_command)
                    new_tokens.append(tokens[i])
                    new_tokens.append(default_unit)
                    i += 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1

        else:
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens


#######################################
# Main block: Print Examples
#######################################
if __name__ == "__main__":
    # Example token lists (assume tokens are already normalized from previous steps)
    examples = [
        # Case: Command with no unit following a bare number.
        (["turn", "30"], ["turn", "30", "deg"]),
        (["move", "50"], ["move", "50", "cm"]),
        (["accelerate", "10"], ["accelerate", "10", "cm/s^2"]),
        # Case: Already specified unit: no insertion.
        (["go", "100", "m"], ["go", "100", "m"]),
        (["drive", "60", "km/h"], ["drive", "60", "km/h"]),
        # Case: No command context.
        (["42"], ["42"]),
        # Case: Multiple commands.
        (["move", "50", "forward", "and", "accelerate", "10"],
         ["move", "50", "cm", "forward", "and", "accelerate", "10", "cm/s^2"]),
        # Case: Command with unit already specified.
        (["turn", "45", "deg"], ["turn", "45", "deg"]),
        # Case: Command with non-recognized command word; no inference.
        (["jump", "100"], ["jump", "100"])
    ]

    print("Examples:")
    for tokens, expected in examples:
        inferred = infer_default_units(tokens)
        print("Input tokens:    ", tokens)
        print("Inferred tokens: ", inferred)
        print("Expected tokens: ", expected)
        print("-" * 40)


    #######################################
    # Test Cases for infer_default_units
    #######################################
    def run_tests():
        # 1. Simple command inference.
        tokens = ["turn", "30"]
        expected = ["turn", "30", "deg"]
        assert infer_default_units(tokens) == expected, f"Test 1 Failed: Expected {expected}"

        # 2. Command inference with "move".
        tokens = ["move", "50"]
        expected = ["move", "50", "cm"]
        assert infer_default_units(tokens) == expected, f"Test 2 Failed: Expected {expected}"

        # 3. Bare number without command remains unchanged.
        tokens = ["42"]
        expected = ["42"]
        assert infer_default_units(tokens) == expected, f"Test 3 Failed: Expected {expected}"

        # 4. Already specified unit.
        tokens = ["go", "100", "m"]
        expected = ["go", "100", "m"]
        assert infer_default_units(tokens) == expected, f"Test 4 Failed: Expected {expected}"

        # 5. Multiple commands in sequence.
        tokens = ["move", "50", "forward", "and", "accelerate", "10"]
        expected = ["move", "50", "cm", "forward", "and", "accelerate", "10", "cm/s^2"]
        assert infer_default_units(tokens) == expected, f"Test 5 Failed: Expected {expected}"

        # 6. Command with already specified unit.
        tokens = ["turn", "45", "deg"]
        expected = ["turn", "45", "deg"]
        assert infer_default_units(tokens) == expected, f"Test 6 Failed: Expected {expected}"

        # 7. Unrecognized command: no inference.
        tokens = ["jump", "100"]
        expected = ["jump", "100"]
        assert infer_default_units(tokens) == expected, f"Test 7 Failed: Expected {expected}"

        print("All infer_default_units tests passed!")


    run_tests()
