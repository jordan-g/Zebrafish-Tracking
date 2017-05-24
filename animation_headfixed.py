import numpy as np # Imports numpy as np.
import matplotlib.pyplot as plt # Imports matplotlib.pyplot as plt.
import matplotlib.animation as an # Imports matplotlib.animation as an.
import open_media
import analysis
import sys
import utilities
import psutil
import cv2
import os

plt.close('all')

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

tail_angle_array = analysis.get_headfixed_tail_angles(tail_coords_array, tail_angle=tracking_params['tail_angle'], tail_direction=tracking_params['tail_direction'])
# heading_angle_array = heading_angle_array[0, :, 0]
tail_end_angle_array = analysis.get_tail_end_angles(tail_angle_array, num_to_average=1)[0]

# Get info about the video
fps, n_frames_total = open_media.get_video_info(video_path)
print("FPS: {}, # frames: {}.".format(fps, n_frames_total))

# Update number of frames to load
if n_frames == 0:
    n_frames = n_frames_total

    if n_frames > tail_end_angle_array.shape[0]:
        n_frames = tail_end_angle_array.shape[0]

# Create a video capture object that we can re-use
try:
    capture = cv2.VideoCapture(video_path)
except:
    print("Error: Could not open video.")

# Calculate the amount of frames to keep in memory at a time -- enough to fill 1/2 of the available memory
mem = psutil.virtual_memory()
mem_to_use = 0.5*mem.available
frame = open_media.open_video(video_path, [0], capture=capture, greyscale=False)[0]
frame_size = frame.shape[0]*frame.shape[1]
big_chunk_size = int(mem_to_use / frame_size)

# Split frame numbers into big chunks - we keep only one big chunk of frames in memory at a time
big_split_frame_nums = utilities.split_list_into_chunks(range(n_frames), big_chunk_size)

ffmpeg_writer = an.writers["ffmpeg"] # Creates an ffmpeg writer.
metadata = dict(title = "Animation for Assignment 10", author = "Nicholas Guilbeault") # Creates metadata for the visual animation.
writer = ffmpeg_writer(fps = 60, metadata = metadata, bitrate = 3200) # Specifies features of the animation. The FPS is set to 15 and the bitrate is set to 1600.

# # Load frames
# print("Loading {} frames...".format(n_frames))
# frames = open_media.open_video(video_path, range(n_frames))
# print("Done.")

# Create the figure & subplots
figure_plot = plt.figure(figsize=(10, 4))
subplot_1 = plt.subplot2grid((1, 3), (0, 0))
subplot_2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
# subplot_3 = plt.subplot2grid((2, 2), (1, 1))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

# Set initial min & max xlim
xlim_start = -100
xlim_end = 100

subplot_1.set_axis_off()
plt.tight_layout()
subplot_1.autoscale_view('tight')

subplot_2.set_xlim(xlim_start, xlim_end) # Sets the x axis limits.
subplot_2.set_ylim(-3, 3) # Sets the y axis limits.
subplot_2.set_xlabel("Frame", fontsize = 5) # Sets the x axis label.
subplot_2.set_ylabel("Tail Angle", fontsize = 5) # Sets the y axis label.
subplot_2.minorticks_off() # Sets the minor tick marks.
subplot_2.tick_params(axis = "both", labelsize = 5) # Sets the size of the tick labels.
plt.tight_layout() # Sets the plot layout.
subplot_2.plot(range(n_frames), tail_end_angle_array[:n_frames], 'b', lw=0.5)

# subplot_3.set_xlim(xlim_start, xlim_end) # Sets the x axis limits.
# subplot_3.set_xlabel("Frame", fontsize = 5) # Sets the x axis label.
# subplot_3.set_ylabel("Heading Angle", fontsize = 5) # Sets the y axis label.
# subplot_3.minorticks_off() # Sets the minor tick marks.
# subplot_3.tick_params(axis = "both", labelsize = 5) # Sets the size of the tick labels.
# plt.tight_layout() # Sets the plot layout.
# subplot_3.plot(range(n_frames), heading_angle_array[:n_frames], 'r', lw=0.5)

# Create the operators for plotting the animation.
if frame.ndim == 3:
    subplot_frame_operator = subplot_1.imshow(frame, vmin=0, vmax=255, animated=True, interpolation='none')
else:
    subplot_frame_operator = subplot_1.imshow(frame, vmin=0, vmax=255, animated=True, cmap='gray', interpolation='none')
subplot_tail_operator, = subplot_2.plot([], [], c = "b", marker = "o", ms = 3)
# subplot_heading_operator, = subplot_3.plot([], [], c = "r", marker = "o", ms = 3)

frame_counter = 0

# Create the animation and save it into a video file.
print("Creating the animation and saving it into a video file...\n") # Prints a message
with writer.saving(figure_plot, "{}_animation.mp4".format(os.path.splitext(os.path.basename(video_path))[0]), 200): # Facilitates the writing of the animation. Plots the animation onto the figure and saves the animation into a file called NG_Assignment10_Animation.mp4.
    for i in range(len(big_split_frame_nums)):
        print("Processing frames {} to {}...".format(big_split_frame_nums[i][0], big_split_frame_nums[i][-1]))

        # Get the frame numbers to process
        frame_nums = big_split_frame_nums[i]

        # Boolean indicating whether to have the capture object seek to the starting frame
        # This only needs to be done at the beginning to seek to frame 0
        seek_to_starting_frame = i == 0

        print("Opening frames...")

        # Load this big chunk of frames
        frames = open_media.open_video(video_path, frame_nums, capture=capture, seek_to_starting_frame=seek_to_starting_frame, greyscale=False)

        for i in range(len(frame_nums)): # Sets the number of iterations.
            print("Processing frame {} of {}.".format(i+1, len(frame_nums)))

            # Sets the value for each variable at each time step.
            frame = frames[i]
            xlim_start += 1
            xlim_end   += 1

            subplot_frame_operator.set_array(frame)

            subplot_2.set_xlim(xlim_start, xlim_end)
            subplot_tail_operator.set_data(frame_counter, tail_end_angle_array[frame_counter])

            # subplot_3.set_xlim(xlim_start, xlim_end)
            # subplot_heading_operator.set_data(frame_counter, heading_angle_array[frame_counter])

            writer.grab_frame() # Grabs the figure information and writes the data into a video frame.

            frame_counter += 1

print("Finished.\n") # Prints the last message.
