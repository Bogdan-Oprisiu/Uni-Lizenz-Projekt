import sys
import time
from pymata4 import pymata4

# HC-SR04 pins
TRIGGER_PIN = 7
ECHO_PIN    = 8

# connect to your board on COM6 (StandardFirmata @57600 bps)
board = pymata4.Pymata4(
    com_port='COM6',
    baud_rate=57600
)

# --- setup the ultrasonic sensor ---
board.sonar_config(TRIGGER_PIN, ECHO_PIN)
time.sleep(0.1)

try:
    while True:
        # read & print distance
        distance, timestamp = board.sonar_read(TRIGGER_PIN)
        print(f"[{time.strftime('%H:%M:%S')}] Distance: {distance} cm")
        time.sleep(0.5)

except KeyboardInterrupt:
    # clean up on Ctrl+C
    board.shutdown()
    sys.exit(0)
