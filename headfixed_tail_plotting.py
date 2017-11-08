import numpy as np
import matplotlib.pyplot as plt
import analysis
import scipy.stats
import os
import json

def getStimParams ():
    stimType = input("Please type a single number to select the stimulus.\
1 for moving dot\
2 for loom\
3 for OMR")

csv_path = "../testing-nov6B/Oct.11.17_1B_prey_tail_angles.csv"

tail_angle_array = analysis.load_tail_angles(csv_path)

tail_end_angles = analysis.calculate_tail_end_angles(tail_angle_array, num_to_average=3, plot=True)

print(tail_end_angles.shape)
