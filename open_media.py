import cv2
import os
import numpy as np
from moviepy.video.io.ffmpeg_reader import *
from helper_functions import *
import traceback
import pdb

def open_image(image_path):
    try:
        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except:
        print("Error: Could not open image.")
        return None

    # convert to greyscale
    if len(frame.shape) >= 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame

def open_folder(folder_path, frame_nums=None, frame_filenames=None, return_frames=True, calc_background=False, progress_signal=None):
    if frame_filenames == None:
        # get filenames of all frame images in the folder
        frame_filenames = get_frame_filenames_from_folder(folder_path)

        if len(frame_filenames) == 0:
            print("Error: No frame images found in the folder {}.".format(folder_path))
            return None

    # no frame numbers given; use all frames
    if frame_nums == None:
        frame_nums = range(len(frame_filenames))

    n_frames = len(frame_nums) # number of frames to use for the background

    # get the rest of the frames
    for i in range(n_frames):
        frame_num = frame_nums[i]

        frame = open_image(os.path.join(folder_path, frame_filenames[frame_num]))

        # stop if the frame couldn't be opened
        if frame == None:
            return None

        if i == 0:
            if return_frames:
                # initialize array to hold all frames
                frames = np.zeros((n_frames, frame.shape[0], frame.shape[1]))
            if calc_background:
                # initialize background array
                background = np.zeros(frame.shape)

        if return_frames:
            frames[i] = frame

        if calc_background:
            # update background array
            mask = np.less(background, frame)
            background[mask] = frame[mask]

        if progress_signal:
            # send an update signal to the GUI every 10% of progress
            percent_complete = int(100*frame_num/n_frames)
            progress_signal.emit(percent_complete)

    print("{} frame{} opened.".format(n_frames, "s"*(n_frames > 1)))

    results = []
    if return_frames:
        results.append(frames.astype(np.uint8))
    if calc_background:
        results.append(background.astype(np.uint8))

    if len(results) == 1:
        return results[0]
    else:
        return results

def open_video(video_path, frame_nums=None, return_frames=True, calc_background=False, progress_signal=None):
    # open the video
    try:
        capture = FFMPEG_VideoReader(video_path)
    except:
        print("Error: Could not open video.")
        return None

    # get video info
    fps, n_frames_total = get_video_info(video_path)

    # no frame numbers given; use all frames
    if frame_nums == None:
        frame_nums = range(n_frames_total)

    n_frames = len(frame_nums) # number of frames to use for the background

    # check whether frame numbers are sequential (ie. [0, 1, 2, ...])
    # or not (ie. [20, 30, 40, ...])
    frame_nums_are_sequential = list(frame_nums) == list(range(frame_nums[0], frame_nums[-1]+1))

    frame_count = 0
    frames      = None
    background  = None

    for i in range(n_frames_total):
        frame = capture.read_frame()

        # stop if the frame couldn't be read
        if frame == None:
            pass
        else:
            # convert to greyscale
            if len(frame.shape) >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i == 0:
                if return_frames:
                    # initialize array to hold all frames
                    frames = np.zeros((n_frames, frame.shape[0], frame.shape[1]))

                if calc_background:
                    # initialize background array
                    background = frame.copy()

            if i in frame_nums:
                if return_frames:
                    frames[frame_count] = frame

                frame_count += 1

                if progress_signal:
                    # send an update signal to the GUI every 10% of progress
                    percent_complete = int(100.0*float(frame_count)/n_frames)
                    progress_signal.emit(percent_complete)

            if calc_background:
                # update background array
                mask = np.less(background, frame)
                background[mask] = frame[mask]

    # close the capture object
    capture.close()

    print("{} frame{} opened.".format(n_frames, "s"*(n_frames > 1)))

    results = []
    if return_frames:
        results.append(frames.astype(np.uint8))
    if calc_background:
        results.append(background.astype(np.uint8))

    if len(results) == 1:
        return results[0]
    else:
        return results

# --- Helper functions --- #

def get_frame_filenames_from_folder(folder_path):
    frame_filenames = [] # list of frame filenames

    for filename in sort_nicely(os.listdir(folder_path)):
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            frame_filenames.append(filename)

    return frame_filenames

def get_video_info(video_path):
    # get video info
    fps      = ffmpeg_parse_infos(video_path)["video_fps"]
    n_frames = ffmpeg_parse_infos(video_path)["video_nframes"]

    return fps, n_frames