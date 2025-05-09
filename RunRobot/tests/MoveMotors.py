"""
 Copyright (c) 2020 Alan Yorinks All rights reserved.

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU AFFERO GENERAL PUBLIC LICENSE
 Version 3 as published by the Free Software Foundation; either
 or (at your option) any later version.
 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.

 You should have received a copy of the GNU AFFERO GENERAL PUBLIC LICENSE
 along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
"""


import sys
import time

from pymata4 import pymata4

board = pymata4.Pymata4(
    com_port='COM6',
    baud_rate=57600 
)
board.digital_write(2, 0)
board.digital_write(3, 0)
board.digital_write(4, 0)
board.digital_write(5, 0)
"""
Setup a digital pin for input pullup and monitor its changes.
"""
while True:
    board.digital_write(2, 1)
    board.digital_write(3, 0)
    board.digital_write(4, 1)
    board.digital_write(5, 0)
    time.sleep(4)
    board.digital_write(2, 0)
    board.digital_write(3, 1)
    board.digital_write(4, 0)
    board.digital_write(5, 1)
    time.sleep(4)