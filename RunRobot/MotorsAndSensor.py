import sys
import time
from pymata4 import pymata4

# motor control pins
PIN_IN1 = 2
PIN_IN2 = 3
PIN_IN3 = 4
PIN_IN4 = 5

# HC-SR04 pins
TRIGGER_PIN = 7
ECHO_PIN    = 8

# connect to your board on COM6 (StandardFirmata @57600 bps)
board = pymata4.Pymata4(
    com_port='COM6',
    baud_rate=57600
)

# --- setup motor pins as digital outputs and default low ---
for pin in (PIN_IN1, PIN_IN2, PIN_IN3, PIN_IN4):
    board.set_pin_mode_digital_output(pin)
    board.digital_write(pin, 0)

# --- setup the ultrasonic sensor ---
board.sonar_config(TRIGGER_PIN, ECHO_PIN)
# give it a moment to initialize
time.sleep(0.1)

try:
    while True:
        # spin one way
        board.digital_write(PIN_IN1, 1)
        board.digital_write(PIN_IN2, 0)
        board.digital_write(PIN_IN3, 1)
        board.digital_write(PIN_IN4, 0)
        time.sleep(4)

        # read & print distance
        distance, timestamp = board.sonar_read(TRIGGER_PIN)
        print(f"[{time.strftime('%H:%M:%S')}] Distance: {distance} cm")

        # reverse direction
        board.digital_write(PIN_IN1, 0)
        board.digital_write(PIN_IN2, 1)
        board.digital_write(PIN_IN3, 0)
        board.digital_write(PIN_IN4, 1)
        time.sleep(4)

        # read again
        distance, timestamp = board.sonar_read(TRIGGER_PIN)
        print(f"[{time.strftime('%H:%M:%S')}] Distance: {distance} cm")

except KeyboardInterrupt:
    # clean up on Ctrl+C
    board.shutdown()
    sys.exit(0)
