# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 22:18:07 2021

@author: kylei
"""

'''Goal is to actually have the game classes in this file'''
'''Essentially the main of the snake game'''

from game_class import GameClass
from tensorflow import keras
from tkinter import *

if __name__ == "__main__":
    model = keras.models.load_model("ga_model")
    game1 = GameClass(model_inp = model)
    game1.play_game(mode="ANN")