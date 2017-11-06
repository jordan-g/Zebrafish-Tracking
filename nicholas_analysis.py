import analysis as an
import tracking
import numpy as np
import matplotlib.pyplot as plt
import sys
from ggplot import *
import pandas

tracking_path = sys.argv[1]
video_path = sys.argv[2]

fps, n_frames = tracking.get_video_info(video_path)

print("FPS: {}, # frames: {}.".format(fps, n_frames))

(tail_coords_array, spline_coords_array,
 heading_angle_array, body_position_array,
 eye_coords_array, tracking_params) = an.open_saved_data(tracking_path)

heading_angle_array = an.fix_heading_angles(heading_angle_array)

tail_angle_array = an.get_freeswimming_tail_angles(tail_coords_array, heading_angle_array, body_position_array)

tail_end_angle_array = an.get_tail_end_angles(tail_angle_array, num_to_average=1)[0]

plt.plot(tail_end_angle_array)
plt.plot(heading_angle_array[0])
plt.show()
