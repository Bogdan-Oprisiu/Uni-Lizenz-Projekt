from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from game import Game

app = FastAPI()

# Create a global game instance.
game = Game()

# Define a request model for commands.
# Commands support an optional numeric parameter.
# Examples:
#   "forward 50"      => Move forward with 50× the base acceleration.
#   "rotate_left 45"  => Rotate left by 45 degrees.
#   "backward"        => Move backward with default magnitude (1.0).
class Command(BaseModel):
    command: str

@app.get("/state")
def get_state():
    """Return the current game state, including sensor data and score."""
    return game.get_state()

@app.post("/command")
def send_command(cmd: Command):
    """
    Process a command to update the game state.

    Supported commands (with optional numeric parameter):
      - "forward [magnitude]": Move forward (e.g. "forward 50")
      - "backward [magnitude]": Move backward (e.g. "backward 25")
      - "left [magnitude]": Strafe left (e.g. "left 10")
      - "right [magnitude]": Strafe right (e.g. "right 10")
      - "rotate_left [angle]": Rotate left (e.g. "rotate_left 45")
      - "rotate_right [angle]": Rotate right (e.g. "rotate_right 45")

    If no numeric parameter is provided, a default value of 1.0 is used.
    Returns the updated game state, including the current score.
    """
    if game.game_over:
        raise HTTPException(status_code=400, detail="Game Over. Please reset the game.")
    game.update(cmd.command)
    return game.get_state()

@app.post("/reset")
def reset_game():
    """Reset the game state."""
    game.reset()
    return {"message": "Game reset", "state": game.get_state()}
