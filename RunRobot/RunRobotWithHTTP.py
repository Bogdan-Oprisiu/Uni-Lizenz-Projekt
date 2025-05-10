import sys
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymata4 import pymata4
import uvicorn

# ––– pin definitions & maps (unchanged) ––––––––––––––––––––––––––––––––––
MOTORS    = {1:(2,3),2:(4,5),3:(6,7),4:(8,9)}
LIMIT_PINS= [14,15,16,17]
INITIAL_DELAY = 5.0

DIR_MAP = {
  'forward':  {1:(1,0),2:(1,0),3:(1,0),4:(1,0)},
  'backward': {1:(0,1),2:(0,1),3:(0,1),4:(0,1)},
  'left':     {1:(0,1),2:(1,0),3:(1,0),4:(0,1)},
  'right':    {1:(1,0),2:(0,1),3:(0,1),4:(1,0)},
}

board = None
_program_start = time.time()

def stop_all_motors():
    for in1,in2 in MOTORS.values():
        board.digital_write(in1,0)
        board.digital_write(in2,0)
    print("🛑 All motors stopped!")

def limit_switch_cb(data):
    _, pin, val, event_time = data
    print(f"[{time.strftime('%H:%M:%S')}] Limit pin {pin} = {val}")
    if event_time < _program_start + INITIAL_DELAY:
        return
    if val == 0:
        print(f"⚠️ Limit on {pin}! Stopping.")
        stop_all_motors()

def drive(direction: str):
    cfg = DIR_MAP.get(direction)
    if not cfg:
        raise ValueError(f"Invalid direction '{direction}'")
    for m,(in1,in2) in MOTORS.items():
        v1, v2 = cfg[m]
        board.digital_write(in1, v1)
        board.digital_write(in2, v2)
    print(f"▶️ Driving {direction}")

# ––– FastAPI app –––––––––––––––––––––––––––––––––––––––––––––––––––––––––
app = FastAPI(title="Robot API")

class Cmd(BaseModel):
    direction: Literal['forward','backward','left','right','stop','exit']

@app.on_event("startup")
def connect_board():
    global board
    print("Opening COM6…")
    board = pymata4.Pymata4(com_port='COM6', baud_rate=57600)
    # setup pins & callbacks once
    for in1,in2 in MOTORS.values():
        board.set_pin_mode_digital_output(in1); board.digital_write(in1,0)
        board.set_pin_mode_digital_output(in2); board.digital_write(in2,0)
    for p in LIMIT_PINS:
        board.set_pin_mode_digital_input(p, callback=limit_switch_cb)
    print("Board ready.")

@app.on_event("shutdown")
def disconnect_board():
    stop_all_motors()
    board.shutdown()
    print("COM6 closed.")

@app.post("/drive")
def drive_endpoint(cmd: Cmd):
    d = cmd.direction
    if d == 'stop':
        stop_all_motors(); return {"status":"stopped"}
    if d == 'exit':
        stop_all_motors(); return {"status":"exiting"}
    try:
        drive(d)
        return {"status":f"driving {d}"}
    except ValueError as e:
        raise HTTPException(400, str(e))

if __name__ == "__main__":
    uvicorn.run("RunRobotWithHTTP:app", host="0.0.0.0", port=8000, reload=False)
