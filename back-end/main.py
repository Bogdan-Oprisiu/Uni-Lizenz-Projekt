import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI()

# 1. Load the command schema and error definitions from possible_commands.json
try:
    with open("../possible_commands.json", "r", encoding="utf-8") as f:
        commands_schema = json.load(f)
except FileNotFoundError:
    commands_schema = {}
    print("Warning: possible_commands.json not found. Validation may not work.")

# A quick helper to retrieve valid command definitions
valid_commands = commands_schema.get("commandLanguage", {}).get("commands", [])
error_defs = commands_schema.get("errors", {}).get("definitions", [])

# 2. Define a Pydantic model for incoming user commands
class UserCommand(BaseModel):
    action: str = Field(..., description="Name of the command (e.g. forward, back, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the command")
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC timestamp of when the command was received"
    )

# 3. Endpoint to view the loaded schema (optional, for debugging/documentation)
@app.get("/schema")
def get_schema():
    return commands_schema

# 4. Endpoint to store user data, validating against the schema
@app.post("/user-data")
def store_user_data(cmd: UserCommand):
    """
    Validates the user command against possible_commands.json,
    then stores it as a separate JSON file in 'human_data' folder.
    """

    # Validate the command's action against the known commands
    matching_cmd = next((c for c in valid_commands if c["name"] == cmd.action), None)
    if not matching_cmd:
        # If action not found, raise an error referencing the schema
        error_msg = _get_error_description("INVALID_COMMAND") or "Invalid command action."
        raise HTTPException(status_code=400, detail=error_msg)

    # Check required parameters
    missing_params = []
    for param_name, param_info in matching_cmd.get("parameters", {}).items():
        if param_info.get("required") and param_name not in cmd.parameters:
            missing_params.append(param_name)

    if missing_params:
        error_msg = _get_error_description("MISSING_PARAMETER") or "Required parameter missing."
        raise HTTPException(
            status_code=400,
            detail=f"{error_msg} Missing: {', '.join(missing_params)}"
        )

    # Optionally, fill in default values for any parameters
    for param_name, param_info in matching_cmd.get("parameters", {}).items():
        if param_info.get("required") is False and param_name not in cmd.parameters:
            default_val = param_info.get("default")
            if default_val is not None:
                cmd.parameters[param_name] = default_val

    # Everything looks good, let's store the command in a separate JSON file
    os.makedirs("human_data", exist_ok=True)

    time_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    filename = f"{time_str}_{short_uuid}.json"
    filepath = os.path.join("human_data", filename)

    # Convert to dict and save
    cmd_dict = cmd.dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cmd_dict, f, indent=2)

    return {
        "message": "User data stored successfully",
        "filename": filename,
        "data": cmd_dict
    }

# 5. Endpoint to list all stored command files
@app.get("/list-files")
def list_stored_files():
    """
    Returns a list of all JSON files in the 'human_data' folder.
    """
    if not os.path.exists("human_data"):
        return {"files": []}
    files = [f for f in os.listdir("human_data") if f.endswith(".json")]
    return {"files": files}

# 6. (Optional) Helper function to retrieve error descriptions from the schema
def _get_error_description(code: str) -> Optional[str]:
    for err in error_defs:
        if err.get("code") == code:
            return err.get("description")
    return None
