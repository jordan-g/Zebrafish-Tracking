import analysis as an
import tracking
import numpy as np
import matplotlib.pyplot as plt

tracking_path = "test/may.19.17 fish3 loom lv30 0.3 10s_tracking.npz"
video_path = "may.19.17 fish3 loom lv30 0.3 10s.avi"
behavior_times = [4160]

scope_video_n_frames = 25

fps, n_frames = tracking.get_video_info(video_path)

print("FPS: {}, # frames: {}.".format(fps, n_frames))

(tail_coords_array, spline_coords_array,
 heading_angle_array, body_position_array,
 eye_coords_array, tracking_params) = an.open_saved_data(tracking_path)

# calculate tail angles
tail_angle_array = an.get_headfixed_tail_angles(tail_coords_array, tail_angle=tracking_params['tail_angle'], tail_direction=tracking_params['tail_direction'])

tail_end_angle_array = an.get_tail_end_angles(tail_angle_array, num_to_average=3)[0]

plt.plot(tail_end_angle_array)
plt.show()

behavior_times_converted = [ scope_video_n_frames*behavior_time/tail_end_angle_array.shape[-1] for behavior_time in behavior_times ]

print(behavior_times_converted)