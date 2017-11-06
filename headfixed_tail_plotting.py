import numpy as np
import matplotlib.pyplot as plt
import analysis
import scipy.stats
import os
import json

csv_path = "../testing-nov6B/OMR behaviour 350fps to 30fps_tail_angles.csv"

tail_angle_array = analysis.load_tail_angles(csv_path)

tail_end_angles = analysis.calculate_tail_end_angles(tail_angle_array, num_to_average=3, plot=True)

print(tail_end_angles.shape)
