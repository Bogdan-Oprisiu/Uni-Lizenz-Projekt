import sys
import time
from pymata4 import pymata4
import tkinter as tk


parent = tk.Tk()
parent.title("Analog IN")
parent.geometry('800x600')
label = tk.Label(parent, text="Click the button and check the label text:")
label.pack()
board=pymata4.Pymata4()

busy = 0

board.set_pin_mode_analog_input(0, callback=None, differential=1)
board.set_pin_mode_digital_output(2)
board.set_pin_mode_digital_output(3)
board.set_pin_mode_digital_output(4)
board.set_pin_mode_pwm_output(5)

def action(x):
    global action
    action = x
    print(action)

def start():
    global busy
    if busy == 0:
        busy = 1
        x = board.analog_read(0)
        label.config(text=x[0])
        board.digital_pin_write(2, 1)
        time.sleep(1)
        board.digital_pin_write(2, 0)
        time.sleep(1)
        busy = 0

def sens1():
    global busy
    if busy == 0:
        busy = 1
        board.pwm_write(5, 0)
        time.sleep(0.1)
        board.digital_pin_write(4,0)
        board.digital_pin_write(3,1)
        time.sleep(0.5)
        board.pwm_write(5, 100)
        time.sleep(0.5)
        board.pwm_write(5, 150)
        time.sleep(0.5)
        board.pwm_write(5, 200)
        time.sleep(0.5)
        board.digital_pin_write(4,0)
        board.digital_pin_write(3,0)
        board.pwm_write(5, 0)
        busy = 0

def sens2():
    global busy
    if busy == 0:
        busy = 1
        board.pwm_write(5, 100)
        board.digital_pin_write(4,1)
        board.digital_pin_write(3,1)
        time.sleep(0.5)
        board.digital_pin_write(4,1)
        board.digital_pin_write(3,0)
        time.sleep(2)
        board.digital_pin_write(4,0)
        board.digital_pin_write(3,0)
        board.pwm_write(5, 0)
        busy = 0

button = tk.Button(parent, text="Releu", command=start)
button.pack()
button1 = tk.Button(parent, text='Sens 1', command=sens1)
button1.pack()
button2 = tk.Button(parent, text="Sens 2", command=sens2)
button2.pack()

parent.mainloop()
