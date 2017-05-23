import numpy as np # Imports numpy as np.
import matplotlib.pyplot as plt # Imports matplotlib.pyplot as plt.
import matplotlib.animation as an # Imports matplotlib.animation as an.
import open_media
import analysis
import sys

# Set the number of frames to load. If 0, all frames are loaded.
n_frames = 0

tracking_path = sys.argv[1]
video_path    = sys.argv[2]

if not (tracking_path.endswith('npz') and video_path.endswith(('.avi', '.mov', '.mp4'))):
    raise ValueError('Invalid arguments provided. The first argument provided needs to be the .npz tracking data file, the second should be the video.')

# Get heading & tail angle arrays
(tail_coords_array, spline_coords_array,
 heading_angle_array, body_position_array,
 eye_coords_array, tracking_params) = analysis.open_saved_data(tracking_path)

heading_angle_array = analysis.fix_heading_angles(heading_angle_array)
tail_angle_array = analysis.get_freeswimming_tail_angles(tail_coords_array, heading_angle_array, body_position_array)
heading_angle_array = heading_angle_array[0, :, 0]
tail_end_angle_array = analysis.get_tail_end_angles(tail_angle_array, num_to_average=1)[0]

# Get info about the video
fps, n_frames_total = open_media.get_video_info(video_path)
print("FPS: {}, # frames: {}.".format(fps, n_frames_total))

# Update number of frames to load
if n_frames == 0:
    n_frames = n_frames_total

ffmpeg_writer = an.writers["ffmpeg"] # Creates an ffmpeg writer.
metadata = dict(title = "Animation for Assignment 10", author = "Nicholas Guilbeault") # Creates metadata for the visual animation.
writer = ffmpeg_writer(fps = 60, metadata = metadata, bitrate = 1600) # Specifies features of the animation. The FPS is set to 15 and the bitrate is set to 1600.

# Load frames
print("Loading {} frames...".format(n_frames))
frames = open_media.open_video(video_path, range(n_frames))
print("Done.")

# Create the figure & subplots
figure_plot = plt.figure(figsize=(8, 3))
subplot_1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
subplot_2 = plt.subplot2grid((2, 2), (0, 1))
subplot_3 = plt.subplot2grid((2, 2), (1, 1))

# Set initial min & max xlim
xlim_start = -100
xlim_end = 100

subplot_1.set_axis_off()

subplot_2.set_xlim(xlim_start, xlim_end) # Sets the x axis limits.
subplot_2.set_ylim(-3, 3) # Sets the y axis limits.
subplot_2.set_xlabel("Frame", fontsize = 5) # Sets the x axis label.
subplot_2.set_ylabel("Tail Angle", fontsize = 5) # Sets the y axis label.
subplot_2.minorticks_off() # Sets the minor tick marks.
subplot_2.tick_params(axis = "both", labelsize = 5) # Sets the size of the tick labels.
plt.tight_layout() # Sets the plot layout.
subplot_2.plot(range(n_frames), tail_end_angle_array[:n_frames], 'b', lw=0.5)

subplot_3.set_xlim(xlim_start, xlim_end) # Sets the x axis limits.
subplot_3.set_xlabel("Frame", fontsize = 5) # Sets the x axis label.
subplot_3.set_ylabel("Heading Angle", fontsize = 5) # Sets the y axis label.
subplot_3.minorticks_off() # Sets the minor tick marks.
subplot_3.tick_params(axis = "both", labelsize = 5) # Sets the size of the tick labels.
plt.tight_layout() # Sets the plot layout.
subplot_3.plot(range(n_frames), heading_angle_array[:n_frames], 'r', lw=0.5)

# Create the operators for plotting the animation.
subplot_frame_operator = subplot_1.imshow(frames[0], vmin=0, vmax=255, animated=True, cmap='gray')
subplot_tail_operator, = subplot_2.plot([], [], c = "b", marker = "o", ms = 3)
subplot_heading_operator, = subplot_3.plot([], [], c = "r", marker = "o", ms = 3)

# Create the animation and save it into a video file.
print("Creating the animation and saving it into a video file...\n") # Prints a message
with writer.saving(figure_plot, "Animation.mp4", 200): # Facilitates the writing of the animation. Plots the animation onto the figure and saves the animation into a file called NG_Assignment10_Animation.mp4.
    for i in range(n_frames): # Sets the number of iterations.
        print("Processing frame {} of {}.".format(i+1, n_frames))

        # Sets the value for each variable at each time step.
        frame = frames[i]
        xlim_start += 1
        xlim_end   += 1

        subplot_frame_operator.set_array(frame)

        subplot_2.set_xlim(xlim_start, xlim_end)
        subplot_tail_operator.set_data(i, tail_end_angle_array[i])

        subplot_3.set_xlim(xlim_start, xlim_end)
        subplot_heading_operator.set_data(i, heading_angle_array[i])

        writer.grab_frame() # Grabs the figure information and writes the data into a video frame.

print("Finished.\n") # Prints the last message.
