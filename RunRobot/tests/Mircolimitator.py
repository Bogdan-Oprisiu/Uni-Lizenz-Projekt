import sys
import time

from pymata4 import pymata4

board = pymata4.Pymata4(
    com_port='COM6',
    baud_rate=57600 
)

board.set_pin_mode_digital_input(10)

while True:
    value = board.digital_read(10)
    print(value)
    time.sleep(1)

