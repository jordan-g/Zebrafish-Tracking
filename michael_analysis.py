import analysis as an
import tracking
import numpy as np
import matplotlib.pyplot as plt
import os

# OIR Files - 0.67 s / frame

tracking_path = "Michael's Videos/may.19.17 fish3 loom lv30 0.3 10s_tracking.npz"
# tracking_path = "test/May.19.17 day2Fish loom lv30 0.3 at 10s_tracking.npz"
# tracking_path = "test/May.19.17 day2fish deep lv30 0.3 escape 3_tracking.npz"
video_path = "Michael's Videos/may.19.17 fish3 loom lv30 0.3 10s.avi"
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

plt.figure(figsize=(8, 4))
plt.plot(tail_end_angle_array, 'b', lw=0.5)
plt.savefig("{}_tail_angle.png".format(os.path.splitext(os.path.basename(video_path))[0]), dpi=200)
plt.savefig("{}_tail_angle.svg".format(os.path.splitext(os.path.basename(video_path))[0]), dpi=200)

# plt.show()

behavior_times_converted = [ scope_video_n_frames*behavior_time/tail_end_angle_array.shape[-1] for behavior_time in behavior_times ]

print(behavior_times_converted)