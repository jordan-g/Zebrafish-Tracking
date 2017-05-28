import analysis as an
import tracking
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# --- PLOTTING TAIL ANGLES --- #

# # Set the base filename here
# fname = "may.19.17 fish3 loom lv30 0.3 10s"

# # Set the paths to the tracking data & the fish video
# tracking_path = "Michael's Videos/{}_tracking.npz".format(fname)
# video_path = "Michael's Videos/{}.avi".format(fname)

tracking_path = sys.argv[1]

# # Get info about fish video
# fps, n_frames = tracking.get_video_info(video_path)
# print("FPS: {}, # frames: {}.".format(fps, n_frames))

# Load up tracking data
(tail_coords_array, spline_coords_array,
 heading_angle_array, body_position_array,
 eye_coords_array, tracking_params) = an.open_saved_data(tracking_path)

# Get tail angle array
tail_angle_array = an.get_headfixed_tail_angles(tail_coords_array, tail_angle=tracking_params['tail_angle'], tail_direction=tracking_params['tail_direction'])
tail_end_angle_array = an.get_tail_end_angles(tail_angle_array, num_to_average=3)[0]

# Smooth out the trace
x = range(tail_end_angle_array.shape[0])
x_new = np.linspace(0, tail_end_angle_array.shape[0]-1, 9000)
tail_end_angle_array = smooth(tail_end_angle_array, 3)
f = interp1d(x, tail_end_angle_array, kind='quadratic')
tail_end_angle_array = f(x_new)

# Plot & save the tail angle array
plt.figure(figsize=(20, 4))
plt.plot(tail_end_angle_array, 'b', lw=0.5)
plt.show()
# plt.savefig("Michael's Videos/{}_tail_angle.png".format(os.path.splitext(os.path.basename(video_path))[0]), dpi=200)
# plt.savefig("Michael's Videos/{}_tail_angle.svg".format(os.path.splitext(os.path.basename(video_path))[0]), dpi=200)
np.savetxt("{}_tail_angle.csv".format(os.path.splitext(os.path.basename(video_path))[0]), tail_end_angle_array, delimiter=",")

# --- CONVERTING BEHAVIOR TIMES TO CORRESPONDING TIMES IN SCOPE VIDEO --- #
# OIR Files - 0.67 s / frame

# # Set scope frames per second
# scope_video_fps = 1.0/0.67

# # List of frame #(s) where behavior event(s) begin
# behavior_times = [4160]

# # Convert these to corresponding frame #s in the scope video
# behavior_times_converted = [ ((behavior_time/fps)*scope_video_fps) for behavior_time in behavior_times ]

print(behavior_times_converted)