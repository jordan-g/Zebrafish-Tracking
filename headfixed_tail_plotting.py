import numpy as np
import matplotlib.pyplot as plt
import analysis
import scipy.stats
import os
import json

csv_path = "testing/May.19.17 day2fish deep lv30 0.3 escape 3_tail_angles.csv"

tail_angle_array = analysis.load_tail_angles(csv_path)

tail_end_angles = analysis.calculate_tail_end_angles(tail_angle_array, num_to_average=5, plot=True)