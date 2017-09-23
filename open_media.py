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
import tracking

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

def open_video(video_path, frame_nums=None, return_frames=True, calc_background=False, progress_signal=None, capture=None, seek_to_starting_frame=True, invert=False, greyscale=True, thread=None):
    mask_points = np.array([(371, 6), (271, 79), (163, 206), (110, 315), (77, 513), (98, 665), (161, 803), (254, 917), (409, 1021), (886, 1021), (1057, 896), (1174, 709), (1213, 496), (1178, 311), (1090, 152), (917, 6)])
    mask = np.zeros((1024, 1280)).astype(np.uint8)
    cv2.fillConvexPoly(mask, mask_points, 1)
    mask = mask.astype(bool)

    def process_frame(frame, invert=False, greyscale=False, mask=None):
        # optionally invert the frame
        if invert:
            frame = 255 - frame

        # convert to greyscale
        if len(frame.shape) >= 3 and greyscale:
            frame = frame[..., 0]

        # frame = denoise_tv_chambolle(frame, weight=0.01, multichannel=False)
        # frame = ndi.median_filter(frame, 3) #Added

        frame[mask == False] = 255

        return frame

    # create a capture object if it's not provided
    if capture is None:
        new_capture = True

        try:
            capture = cv2.VideoCapture(video_path)
        except:
            print("Error: Could not open video.")
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

    frames      = None
    background  = None

    # optionally seek to the first frame
    # if the first frame is frame 0, we don't need to do this
    if seek_to_starting_frame or frame_nums[0] != 0:
        try:
            capture.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_nums[0]-1)
        except:
            capture.set(1, frame_nums[0]-1)

    # get the first frame
    _, frame = capture.read()

    frame = process_frame(frame, invert=invert, greyscale=greyscale, mask=mask)

    if calc_background:
        # initialize background array
        background = frame.copy().astype(np.uint8)

    if return_frames:
        # initialize array to hold all frames
        if greyscale:
            frames = np.zeros((n_frames, frame.shape[0], frame.shape[1])).astype(np.uint8)
        else:
            frames = np.zeros((n_frames, frame.shape[0], frame.shape[1], frame.shape[2])).astype(np.uint8)

        frames[0] = frame

    frame_count = 1

    if not frame_nums_are_sequential:
        for i in range(frame_nums[0]+1, frame_nums[-1]+1):
            _ = capture.grab()

            if i in frame_nums:
                _, frame = capture.retrieve()
                # frame = process_frame(frame, invert=invert, greyscale=greyscale, mask=mask)

                if return_frames:
                    frames[frame_count] = process_frame(capture.retrieve()[1][..., 0], invert=invert, greyscale=greyscale, mask=mask)
                else:
                    frame = process_frame(capture.retrieve()[1][..., 0], invert=invert, greyscale=greyscale, mask=mask)

                if progress_signal:
                    # send an update signal to the GUI
                    percent_complete = int(100.0*float(frame_count)/n_frames)
                    progress_signal.emit(percent_complete, video_path)

                if calc_background:
                    # update background array
                    mask_2 = np.less(background, frame)
                    background[mask_2] = frame[mask_2]

                frame_count += 1

            if thread is not None and thread.running == False:
                return None
    else:
        while frame_count < n_frames:
            _, frame = capture.read()

            frame = process_frame(frame, invert=invert, greyscale=greyscale, mask=mask)

            if return_frames:
                frames[frame_count] = frame

            if progress_signal:
                # send an update signal to the GUI
                percent_complete = int(100.0*float(frame_count)/n_frames)
                progress_signal.emit(percent_complete, video_path)

            if calc_background:
                # update background array
                mask_2 = np.less(background, frame)
                background[mask_2] = frame[mask_2]

            frame_count += 1

            if thread is not None and thread.running == False:
                return None

    if new_capture:
        # close the capture object
        capture.release()

    print("{} frame{} opened.".format(n_frames, "s"*(n_frames > 1)))

    if return_frames and frames is None:
        print("Error: Could not get any frames from the video.")
        return None
    # if return_frames:
    #     frames = tracking.remove_noise(frames)

    if return_frames and calc_background:
        return frames, background
    elif calc_background:
        return background
    else:
        return frames

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
