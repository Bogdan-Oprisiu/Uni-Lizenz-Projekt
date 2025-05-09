from pymata4 import pymata4
import sys, time

DIGITAL_PIN = 6  # arduino pin number

def blink(my_board, pin):
    my_board.set_pin_mode_digital_output(pin)
    for _ in range(4):
        print("ON")
        my_board.digital_write(pin, 1)
        time.sleep(1)
        print("OFF")
        my_board.digital_write(pin, 0)
        time.sleep(1)
    my_board.shutdown()

board = pymata4.Pymata4(
    com_port='COM6',
    baud_rate=57600 
)

try:
    blink(board, DIGITAL_PIN)
except KeyboardInterrupt:
    board.shutdown()
    sys.exit(0)
