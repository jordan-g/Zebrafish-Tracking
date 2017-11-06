import numpy as np
import matplotlib.pyplot as plt
import analysis
import scipy.stats
import os
import json

csv_path = "testing_Nov6/May.19.17 day2Fish loom lv30 0.3 at 10s_tail_angles.csv"

tail_angle_array = analysis.load_tail_angles(csv_path)

tail_end_angles = analysis.calculate_tail_end_angles(tail_angle_array, num_to_average=5, plot=True)