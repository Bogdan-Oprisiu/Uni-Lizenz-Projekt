import sys
import time
from pymata4 import pymata4

# —–– pin assignments —–––––––––––––––––––––––––––
# Each motor has two direction pins: (IN1, IN2)
MOTORS = {
    1: (2,  3),   # Front-Left
    2: (4,  5),   # Front-Right
    3: (6,  7),   # Back-Left
    4: (8,  9),   # Back-Right
}

# Four limit switches on analog pins A0–A3 (digital 14–17)
LIMIT_PINS = [14, 15, 16, 17]

INITIAL_DELAY = 5.0
_program_start = time.time()

# —–– mapping of movement → (IN1,IN2) for each motor —–––––––––––––––––
DIR_MAP = {
    'forward':  {1:(1,0), 2:(1,0), 3:(1,0), 4:(1,0)},
    'backward': {1:(0,1), 2:(0,1), 3:(0,1), 4:(0,1)},
    'left':     {1:(0,1), 2:(1,0), 3:(1,0), 4:(0,1)},  
    'right':    {1:(1,0), 2:(0,1), 3:(0,1), 4:(1,0)},  
}

# —–– helper to stop all motors instantly —––––––
def stop_all_motors():
    for in1, in2 in MOTORS.values():
        board.digital_write(in1, 0)
        board.digital_write(in2, 0)
    print("🛑 All motors stopped!")

# —–– callback fired on any limit switch change —––
def limit_switch_cb(data):
    # data = [pin_type, pin_number, value, timestamp]
    _, pin, val, event_time = data
    print(f"[{time.strftime('%H:%M:%S')}] Limit pin {pin} = {val}")

    # ignore during initial startup delay
    if event_time < _program_start + INITIAL_DELAY:
        return

    if val == 0:  # assuming pull-up, pressed = 0
        print(f"⚠️  Limit switch on pin {pin}! Stopping motors.")
        stop_all_motors()

# —–– helper to drive in a given direction —––––
def drive(direction: str):
    cfg = DIR_MAP.get(direction)
    if not cfg:
        print(f"Unknown direction '{direction}'. Valid: {list(DIR_MAP)}")
        return
    for motor, (in1, in2) in MOTORS.items():
        v1, v2 = cfg[motor]
        board.digital_write(in1, v1)
        board.digital_write(in2, v2)
    print(f"▶️  Driving {direction}")

# —–– setup —––––––––––––––––––––––––––––––––––––
board = pymata4.Pymata4(com_port='COM6', baud_rate=57600)

# configure motor pins as outputs and ensure they're off
for in1, in2 in MOTORS.values():
    board.set_pin_mode_digital_output(in1); board.digital_write(in1, 0)
    board.set_pin_mode_digital_output(in2); board.digital_write(in2, 0)

# configure each limit switch as an input with callback
for pin in LIMIT_PINS:
    board.set_pin_mode_digital_input(pin, callback=limit_switch_cb)

print("Ready. Enter a direction (forward, backward, left, right), 'stop', or 'exit'.")

# —–– main loop: read user commands —––––––––––––––––––––
while True:
    cmd = input("> ").strip().lower()
    if cmd in ('exit', 'quit'):
        print("Exiting program.")
        break
    elif cmd == 'stop':
        stop_all_motors()
    else:
        drive(cmd)
    # small delay to debounce input
    time.sleep(0.1)

# cleanup
stop_all_motors()
board.shutdown()
sys.exit(0)
