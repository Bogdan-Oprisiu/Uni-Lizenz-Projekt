import re


def infer_default_units(tokens):
    """
    Insert default units after a bare number if the number follows a command word
    and does not already have a recognized unit (or a recognized unit with trailing punctuation).

    E.g.:
      - "turn 30" => ["turn", "30", "deg"]
      - "move 50" => ["move", "50", "cm"]
      - "accelerate 10" => ["accelerate", "10", "cm/s^2"]

    Assumptions:
      1) 'tokens' is a list of normalized tokens (lowercased commands, spelled-out
         numbers converted, multi-word unit phrases replaced, etc.).
      2) If the next token after a number is not in recognized_units (accounting for punctuation),
         we insert the default unit of the last command word we saw.
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

        # 1) If token is a command word, remember it
        if token_lower in DEFAULT_COMMAND_UNITS:
            last_command = token_lower
            new_tokens.append(tokens[i])
            i += 1
            continue

        # 2) If token is a bare number, check the next token
        if is_bare_number(tokens[i]):
            next_unit = ""
            # If there's a next token, strip trailing punctuation before checking
            if i + 1 < len(tokens):
                # e.g., "m/s^2," -> "m/s^2"
                stripped = re.sub(r"[.,;!?]+$", "", tokens[i + 1])
                next_unit = stripped.lower()

            # If next_unit is recognized, do nothing
            if next_unit in recognized_units:
                new_tokens.append(tokens[i])
                i += 1
            else:
                # If there's a valid command context, add default unit
                if last_command is not None:
                    default_unit = DEFAULT_COMMAND_UNITS.get(last_command)
                    new_tokens.append(tokens[i])
                    new_tokens.append(default_unit)
                    i += 1
                else:
                    # No known command -> no default
                    new_tokens.append(tokens[i])
                    i += 1
            continue

        # 3) Otherwise it's just a normal token
        new_tokens.append(tokens[i])
        i += 1

    return new_tokens
