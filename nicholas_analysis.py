import analysis as an
import tracking
import numpy as np
import matplotlib.pyplot as plt

tracking_path = "test/Test1_tracking.npz"
video_path = "Test1.avi"

fps, n_frames = tracking.get_video_info(video_path)

print("FPS: {}, # frames: {}.".format(fps, n_frames))

(tail_coords_array, spline_coords_array,
 heading_angle_array, body_position_array,
 eye_coords_array, tracking_params) = an.open_saved_data(tracking_path)

heading_angle_array = an.fix_heading_angles(heading_angle_array)

tail_angle_array = an.get_freeswimming_tail_angles(tail_coords_array, heading_angle_array, body_position_array)

tail_end_angle_array = an.get_tail_end_angles(tail_angle_array, num_to_average=3)[0]

plt.plot(tail_end_angle_array)
plt.show()