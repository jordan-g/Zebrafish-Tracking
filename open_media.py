import cv2
import os
import numpy as np
from moviepy.video.io.ffmpeg_reader import *
from helper_functions import *
import traceback
import pdb
import scipy.ndimage as ndi
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
# import tracking

def open_image(image_path):
    # read the frame
    try:
        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except:
        print("Error: Could not open image.")
        return None

    # convert to greyscale
    if len(frame.shape) >= 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame

def open_video(video_path, frame_nums=None, return_frames=True, calc_background=False, progress_signal=None, capture=None, seek_to_starting_frame=True, dark_background=False, greyscale=True, thread=None):
    # create a capture object if it's not provided
    if capture is None:
        new_capture = True

        try:
            capture = cv2.VideoCapture(video_path)
        except:
            print("Error: Could not open video.")
            if return_frames and calc_background:
                return None, None
            else:
                return None
    else:
        new_capture = False

    # get video info
    fps, n_frames_total = get_video_info(video_path)

    # no frame numbers given; read all frames
    if frame_nums is None:
        frame_nums = range(n_frames_total)

    n_frames = len(frame_nums)

    # check whether frame numbers are sequential (ie. [0, 1, 2, ...])
    # or not (ie. [20, 35, 401, ...])
    frame_nums_are_sequential = list(frame_nums) == list(range(frame_nums[0], frame_nums[-1]+1))

    if not frame_nums_are_sequential:
        n_frames_range = frame_nums[-1] - frame_nums[0] + 1

    frames      = None
    background  = None

    # optionally seek to the first frame
    # if the first frame is frame 0, we don't need to do this
    if seek_to_starting_frame or frame_nums[0] != 0:
        try:
            capture.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_nums[0]-1)
        except:
            capture.set(1, frame_nums[0]-1)

    try:
        # get the first frame
        frame = capture.read()[1][..., 0]

        if calc_background:
            # initialize background array
            background = frame.copy()
    except:
        print("Error: Could not load any frames from the video.")
        if return_frames and calc_background:
            return None, None
        else:
            return None

    frame_count = 1

    if return_frames:
        # initialize list to hold all frames
        frames = []

        frames.append(frame)

    if not frame_nums_are_sequential:
        while frame_count <= n_frames_range-1:
            _ = capture.grab()

            if frame_nums[0] + frame_count in frame_nums:
                frame = capture.retrieve()[1][..., 0]

                if return_frames:
                    frames.append(frame)

                if calc_background and frame is not None:
                    # update background array
                    if dark_background:
                        mask_2 = np.greater(background, frame)
                    else:
                        mask_2 = np.less(background, frame)
                    background[mask_2] = frame[mask_2]

            if progress_signal:
                # send an update signal to the GUI
                percent_complete = int(100.0*float(frame_count)/n_frames_range)
                progress_signal.emit(percent_complete, video_path)

            frame_count += 1

            if thread is not None and thread.running == False:
                if return_frames and calc_background:
                    return None, None
                else:
                    return None
    else:
        while frame_count <= n_frames-1:
            frame = capture.read()[1][..., 0]

            if return_frames:
                frames.append(frame)

            if progress_signal:
                # send an update signal to the GUI
                percent_complete = int(100.0*float(frame_count)/n_frames)
                progress_signal.emit(percent_complete, video_path)

            if calc_background and frame is not None:
                # update background array
                if dark_background:
                    mask_2 = np.greater(background, frame)
                else:
                    mask_2 = np.less(background, frame)
                background[mask_2] = frame[mask_2]

            frame_count += 1

            if thread is not None and thread.running == False:
                if return_frames and calc_background:
                    return None, None
                else:
                    return None

    if new_capture:
        # close the capture object
        capture.release()

    if return_frames:
        n_frames = len(frames)
        print("{} frame{} opened.".format(n_frames, "s"*(n_frames > 1)))

    if return_frames and frames is None:
        print("Error: Could not get any frames from the video.")
        return None

    if return_frames and calc_background:
        return np.array(frames), background
    elif calc_background:
        return background
    else:
        return np.array(frames)

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
