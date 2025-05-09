import sys, time
from pymata4 import pymata4

# —–– pin definitions —––––––––––––––
# motor control pins
IN1, IN2, IN3, IN4 = 2, 3, 4, 5   
# HC-SR04 pins
STOP_PIN = 10

INITIAL_DELAY = 5.0
_program_start = time.time()

# —–– helper to kill the motors —––––––
def stop_motors():
    for pin in (IN1, IN2, IN3, IN4):
        board.digital_write(pin, 0)

# —–– callback invoked on pin 10 changes —––––  
def estop_cb(data):
    # data = [pin_type, pin_number, value, timestamp]
    _, pin, val, event_time = data

    # always print the pin output
    print(f"[{time.strftime('%H:%M:%S')}] Pin {pin} = {val}")

    # ignore during initial startup delay
    if event_time < _program_start + INITIAL_DELAY:
        return

    if pin == STOP_PIN and val == 0:
        print("🔴 Emergency STOP!")
        stop_motors()

# —–– program start —––––––––––––––––
board = pymata4.Pymata4(com_port='COM6', baud_rate=57600)

# configure motor pins as outputs
for p in (IN1, IN2, IN3, IN4):
    board.set_pin_mode_digital_output(p)
    board.digital_write(p, 0)

# configure the stop button pin *with* callback
board.set_pin_mode_digital_input(STOP_PIN, callback=estop_cb)
# :contentReference[oaicite:0]{index=0}

# fire up your motors here
board.digital_write(IN1, 1)
board.digital_write(IN2, 0)
board.digital_write(IN3, 1)
board.digital_write(IN4, 0)

try:
    # main loop can do whatever; the callback will interrupt it
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    stop_motors()
    board.shutdown()
    sys.exit(0)
