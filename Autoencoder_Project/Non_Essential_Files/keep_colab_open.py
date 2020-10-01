# pip install pyautogui
import pyautogui  # https://pyautogui.readthedocs.io/en/latest/mouse.html
import time
import random


"""
Simple script to keep colab open while building a model.  Google colab will timeout after a certain amount of time has 
passed.  Running this constant loop with the mouse hovered over the colab webpage will keep you "interacting" with the
colab session and keep it open.  This works best for running a script overnight since you will have to leave your 
computer alone while the clicking happens.
"""

# wait 15 seconds for you to position your mouse over the
time.sleep(10)

while True:
    print("clicking now")
    pyautogui.click()
    time.sleep(random.randint(5, 10))  # click at random intervals, but honestly they probably know...
