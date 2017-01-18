from __future__ import division
import numpy as np
import cv2
from scipy import interpolate
import pylab
import scipy.ndimage
import scipy.stats

import os
import re
import itertools
import time

import pdb
import time
import multiprocessing
from multiprocessing import sharedctypes
from functools import partial
from itertools import chain

from moviepy.video.io.ffmpeg_reader import *
from skimage.morphology import skeletonize
import bwmorph_thin
# import thinning
import skeleton
# import mahotas as mh
from collections import deque
import skfmm

default_crop_params = { 'offset': [0, 0],      # crop offset
                        'crop': None,          # crop size
                        'tail_threshold': 200, # pixel brightness to use for thresholding to find the tail (0-255)
                        'head_threshold': 50   # pixel brightness to use for thresholding to find the eyes (0-255)
                      }

try:
    xrange
except:
    xrange = range

# headfixed tail tracking global variables
fitted_tail           = []
tail_funcs            = None
tail_brightness       = None
background_brightness = None
tail_length           = None
n_frames_tracked      = 0

cv2.setNumThreads(0) # avoids crashes when using multiprocessing with opencv

# --- Loading --- #

def load_frame_from_image(image_path, background=None):
    try:
        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except:
        print("Error: Could not open image.")
        return None

    # convert to greyscale
    if len(frame.shape) >= 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if background != None:
        # subtract background from frame
        bg_sub_frame = subtract_background_from_frame(frame, background)

        return frame, bg_sub_frame
    else:
        return frame

def load_frames_from_folder(folder_path, frame_filenames, frame_nums, background=None):
    print("Loading frames from {}...".format(folder_path))
    if frame_filenames == None and folder_path != None:
        # get filenames of all frame images in the folder
        frame_filenames = get_frame_filenames_from_folder(folder_path)

        if len(frame_filenames) == 0:
            # no frames found in the folder; end here
            if background != None:
                return [None]*2
            else:
                return None

    n_frames_total = len(frame_filenames) # total number of frames in the folder

    # no frame numbers given; use all frames
    if frame_nums == None:
        frame_nums = range(n_frames_total)

    n_frames = len(frame_nums) # number of frames to use for the background

    # initialize list of frames
    frames = []

    if background != None:
        # initialize list of background subtracted frames
        bg_sub_frames = []

    for frame_number in frame_nums:
        # load the frame
        if background != None:
            frame, bg_sub_frame = load_frame_from_image(os.path.join(folder_path, frame_filenames[frame_number]), background)

            # add to background subtracted frames list
            bg_sub_frames.append(bg_sub_frame)
        else:
            frame = load_frame_from_image(os.path.join(folder_path, frame_filenames[frame_number]), None)

        # add to frames list
        frames.append(frame)

    print("{} frames loaded.".format(n_frames_total))

    if background != None:
        return frames, bg_sub_frames
    else:
        return frames

def load_frames_from_video(video_path, cap, frame_nums, background=None, batch_offset=None):
    if batch_offset != None:
        transform = np.float32([[1, 0, -batch_offset[0]], [0, 1, -batch_offset[1]]])

    if cap == None and video_path != None:
        new_cap = True # creating a new capture object; we will release it at the end

        # open the video
        try:
            cap = cv2.VideoCapture(video_path)
        except:
            print("Error: Could not open video.")
            if background != None:
                return [None]*2
            else:
                return None
    else:
        new_cap = False # reusing an existinc capture object

    # get video info
    fps, n_frames_total = get_video_info(video_path)

    # no frame numbers given; use all frames
    if frame_nums == None:
        frame_nums = range(n_frames_total)

    # print(n_frames_total)

    n_frames = len(frame_nums) # number of frames to use for the background

    # initialize list of frames
    frames = []

    if background != None:
        # initialize list of background subtracted frames
        bg_sub_frames = []

    # check whether frame numbers are sequential (ie. [1, 2, 3, ...])
    # or not (ie. [20, 30, 40, ...])
    frame_nums_are_sequential = frame_nums == list(range(frame_nums[0], frame_nums[-1]+1))

    if frame_nums_are_sequential:
        # frame numbers are sequential - just set frame position once
        try:
            cap.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_nums[0]-1)
        except:
            cap.set(1, frame_nums[0]-1)

    for frame_num in frame_nums:
        if not frame_nums_are_sequential:  
            # frame numbers are not sequential - set frame position
            try:
                cap.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_num-1)
            except:
                cap.set(1, frame_num-1)
            _, frame = cap.read()

        if frame != None:
            # convert to greyscale
            if len(frame.shape) >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if batch_offset != None:
                offset_frame = cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]), borderValue=255)
            else:
                offset_frame = frame

            # add to frames list
            frames.append(offset_frame)

            if background != None:
                # subtract background from frame
                bg_sub_frame = subtract_background_from_frame(offset_frame, background)

                if batch_offset != None:
                    offset_bg_sub_frame = cv2.warpAffine(bg_sub_frame, transform, (bg_sub_frame.shape[1], bg_sub_frame.shape[0]), borderValue=255)
                else:
                    offset_bg_sub_frame = bg_sub_frame

                # add to frames list
                bg_sub_frames.append(offset_bg_sub_frame)

    if new_cap:
        # release the capture object
        cap.release()

    if background != None:
        return frames, bg_sub_frames
    else:
        return frames

# --- Background subtraction --- #

def get_background_from_folder(folder_path, frame_filenames=None, frame_nums=None, save_frames=False, progress_signal=None):
    if frame_filenames == None and folder_path != None:
        # get filenames of all frame images in the folder
        frame_filenames = get_frame_filenames_from_folder(folder_path)

        if len(frame_filenames)  == 0:
            # no frames found in the folder; end here
            if save_frames:
                return [None]*2
            else:
                return None

    n_frames_total = len(frame_filenames) # total number of frames in the folder

    # no frame numbers given; use all frames
    if frame_nums == None:
        frame_nums = range(n_frames_total)

    n_frames = len(frame_nums) # number of frames to use for the background

    # initialize background
    background = None

    if save_frames:
        # initialize list of frames
        frames = []

    for frame_number in frame_nums:
        # send an update signal to the GUI every 10% of progress
        percent_complete = int(100*frame_number/n_frames)
        if progress_signal and percent_complete % 10 == 0:
            progress_signal.emit(percent_complete)

        # get frame
        frame = load_frame_from_image(os.path.join(folder_path, frame_filenames[frame_number]))

        # convert to greyscale
        if len(frame.shape) >= 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if save_frames:
            # add to frames list
            frames.append(frame)

        # update background
        if background == None:
            background = frame
        else:
            mask = np.less(background, frame)
            background[mask] = frame[mask]

    if save_frames:
        return background, frames
    else:
        return background

def get_background_from_video(video_path, cap, frame_nums=None, save_frames=False, progress_signal=None, batch_offset=None):
    if batch_offset != None:
        transform = np.float32([[1, 0, -batch_offset[0]], [0, 1, -batch_offset[1]]])

    if cap == None and video_path != None:
        new_cap = True # creating a new capture object; we will release it at the end

        # open the video
        try:
            cap = cv2.VideoCapture(video_path)
        except:
            print("Error: Could not open video.")
            if save_frames:
                return [None]*2
            else:
                return None
    else:
        new_cap = False # reusing an existing capture object

    # get video info
    fps, n_frames_total = get_video_info(video_path)

    # no frame numbers given; use all frames
    if frame_nums == None:
        frame_nums = range(n_frames_total)

    n_frames = len(frame_nums) # number of frames to use for the background
    
    # initialize background
    background = None

    if save_frames:
        # initialize list of frames
        frames = []

    for frame_number in frame_nums:
        # send an update signal to the GUI every 10% of progress
        percent_complete = int(100*frame_number/n_frames)
        if progress_signal and percent_complete % 10 == 0:
            progress_signal.emit(percent_complete)

        # # get frame
        # try:
        #     cap.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_number)
        # except:
        #     cap.set(1, frame_number)
        _, frame = cap.read()

        print(frame_number)

        if frame != None:
            # convert to greyscale
            if len(frame.shape) >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if batch_offset != None:
                offset_frame = cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]), borderValue=255)
            else:
                offset_frame = frame

            if save_frames:
                # add to frames list
                frames.append(offset_frame)

            # update background
            if background == None:
                background = offset_frame
            else:
                mask = np.less(background, offset_frame)
                background[mask] = offset_frame[mask]

    if new_cap:
        # release the capture object
        cap.release()

    if save_frames:
        return background, frames
    else:
        return background

def subtract_background_from_frames(frames, background):
    # initialize list of background subtracted frames
    bg_sub_frames = []

    for frame in frames:
        # subtract background from frame
        bg_sub_frame = subtract_background_from_frame(frame, background)
        bg_sub_frames.append(bg_sub_frame)

    return bg_sub_frames

def subtract_background_from_frame(frame, background):
    # subtract background from frame
    bg_sub_frame = frame - background
    bg_sub_frame[bg_sub_frame < 10] = 255

    return bg_sub_frame

# --- Tracking --- #

def open_and_track_image(params, tracking_dir, progress_signal=None):
    image_path          = params['media_paths'][0]
    subtract_background = params['subtract_background']
    background          = params['background']
    shrink_factor       = crop_params['shrink_factor']

    # load the frame
    if background != None:
        frame, bg_sub_frame = load_frame_from_image(image_path, background)

        frame = bg_sub_frame
    else:
        frame = load_frame_from_image(image_path, None)

    if params['invert']:
        # invert the frame
        frame = (255 - frame)

    for k in range(len(crop_params)):
        # get crop & offset
        crop   = crop_params[k]['crop']
        offset = crop_params[k]['offset']

        # shrink & crop the frame
        shrunken_frame = shrink_frame(frame, shrink_factor)
        cropped_frame  = crop_frame(shrunken_frame, offset, crop)

        # track the frame
        coords = track_cropped_frame(frame, params, crop_params[k])

        # add tracked points to the frame
        tracked_frame = add_tracking_to_frame(cropped_frame, coords)

        # save the new frame
        cv2.imwrite(os.path.join(tracking_dir, "crop_{}.png".format(k)), tracked_frame)

def track_frames(params, progress_signal, n_frames_tracked, n_frames_total, media_number, frames):
    crop_params   = params['crop_params']
    tracking_type = params['type']
    n_tail_points = params['n_tail_points']
    shrink_factor = params['shrink_factor']

    # get number of frames & number of crops
    n_frames = len(frames)
    n_crops  = len(crop_params)

    # initialize tracking data arrays
    tail_coords_array    = np.zeros((n_crops, n_frames, 2, n_tail_points)) + np.nan
    spline_coords_array  = np.zeros((n_crops, n_frames, 2, n_tail_points)) + np.nan
    heading_angle_array  = np.zeros((n_crops, n_frames, 1)) + np.nan
    body_position_array  = np.zeros((n_crops, n_frames, 2)) + np.nan
    eye_coords_array     = np.zeros((n_crops, n_frames, 2, 2)) + np.nan

    # set booleans for head & tail tracking
    track_head = tracking_type == "freeswimming"
    track_tail = tracking_type == "headfixed" or params['track_tail'] == True

    for frame_number in range(n_frames):
        # get the frame
        frame = frames[frame_number]

        for k in range(n_crops):
            # get crop & offset
            crop   = crop_params[k]['crop']
            offset = crop_params[k]['offset']

            # shrink & crop the frame
            cropped_frame          = crop_frame(frame, offset, crop)
            shrunken_cropped_frame = shrink_frame(cropped_frame, shrink_factor)

            # track the frame
            coords = track_cropped_frame(shrunken_cropped_frame, params, crop_params[k])

            # add coords to coord arrays
            if coords[0] != None:
                tail_coords_array[k, frame_number, :, :coords[0].shape[1]]    = coords[0]
                spline_coords_array[k, frame_number, :, :coords[1].shape[1]]  = coords[1]

            heading_angle_array[k, frame_number, :] = coords[2]
            body_position_array[k, frame_number, :] = coords[3]
            eye_coords_array[k, frame_number, :, :] = coords[4]

        if progress_signal and (frame_number + 1) % 50 == 0:
            # send an update signal to the controller
            percent_complete = int(100.0*(n_frames_tracked.value + frame_number)/n_frames_total)
            progress_signal.emit(params['media_type'], media_number, percent_complete)

            n_frames_tracked.value += 50

    if progress_signal:
        # send an update signal to the controller
        percent_complete = int(100*(n_frames_tracked.value + n_frames)/n_frames_total)
        progress_signal.emit(params['media_type'], media_number, percent_complete)

    return tail_coords_array, spline_coords_array, heading_angle_array, body_position_array, eye_coords_array

def track_some_frames(frames, params, frame_nums):
    return track_frames(params, None, None, None, None, frames[frame_nums])

def open_and_track_folder(params, tracking_dir, progress_signal=None): # todo: add video creation from tracking data
    folder_path         = params['media_paths'][0]
    subtract_background = params['subtract_background']
    background          = params['background']
    crop_params         = params['crop_params']
    n_tail_points       = params['n_tail_points']
    save_video          = params['save_video']
    saved_video_fps     = params['saved_video_fps']
    use_multiprocessing = params['use_multiprocessing']

    global n_frames_tracked
    n_frames_tracked = 0

    # start timer
    start_time = time.time()

    # get frame filenames
    frame_filenames = get_frame_filenames_from_folder(folder_path)

    n_frames_total = len(frame_filenames) # total number of frames in the folder

    # get number of crops
    n_crops  = len(crop_params)

    if subtract_background and background == None:
        # get background
        background = get_background_from_folder(folder_path, frame_filenames)

    # initialize tracking data arrays
    tail_coords_array    = np.zeros((n_crops, n_frames_total, 2, n_tail_points)) + np.nan
    spline_coords_array  = np.zeros((n_crops, n_frames_total, 2, n_tail_points)) + np.nan
    heading_angle_array  = np.zeros((n_crops, n_frames_total, 1)) + np.nan
    body_position_array  = np.zeros((n_crops, n_frames_total, 2)) + np.nan
    eye_coords_array     = np.zeros((n_crops, n_frames_total, 2, 2)) + np.nan

    # split frame numbers into big chunks - we keep only one big chunk of frames in memory at a time
    big_split_frame_nums = split_list_into_chunks(range(n_frames_total), 3000)

    if use_multiprocessing:
        # create a pool of workers
        pool = multiprocessing.Pool(None)

    if progress_signal:
        # send an update signal to the controller
        progress_signal.emit("folder", 0, 0)

    for frame_nums in big_split_frame_nums:
        # load this big chunk of frames
        if subtract_background:
            frames, bg_sub_frames = load_frames_from_folder(folder_path, frame_filenames, frame_nums, background)

            frames = bg_sub_frames
        else:
            frames = load_frames_from_folder(folder_path, frame_filenames, frame_nums, None)

        if params['invert']:
            # invert the frames
            frames = [255 - frames[i] for i in range(len(frames))]

        if use_multiprocessing:
            # split frames into small chunks - we let multiple processes deal with a chunk at a time
            split_frames = yield_chunks_from_list(frames, 10)

            # have workers process each chunk
            result_list = pool.map_async(partial(track_frames, params, None, None, None, None), split_frames).get()

            n_frames_tracked += len(frame_nums)

            if progress_signal:
                # send an update signal to the controller
                percent_complete = int(100*n_frames_tracked/n_frames_total)
                progress_signal.emit("folder", 0, percent_complete)

            n_chunks = len(result_list)

            # add results to tracking data arrays
            tail_coords_array[:, frame_nums, :, :]   = np.concatenate([result_list[i][0] for i in range(n_chunks)], axis=1)
            spline_coords_array[:, frame_nums, :, :] = np.concatenate([result_list[i][1] for i in range(n_chunks)], axis=1)
            heading_angle_array[:, frame_nums, :]    = np.concatenate([result_list[i][2] for i in range(n_chunks)], axis=1)
            body_position_array[:, frame_nums, :]    = np.concatenate([result_list[i][3] for i in range(n_chunks)], axis=1)
            eye_coords_array[:, frame_nums, :, :]    = np.concatenate([result_list[i][4] for i in range(n_chunks)], axis=1)
        else:
            # track this big chunk of frames and add results to tracking data arrays
            (tail_coords_small_array, spline_coords_small_array,
             heading_angle_small_array, body_position_small_array, eye_coords_small_array) = track_frames(params, progress_signal, n_frames_tracked, n_frames_total, 0, frames)

            n_frames_tracked += len(frame_nums)

            tail_coords_array[:, frame_nums, :, :]    = tail_coords_small_array
            spline_coords_array[:, frame_nums, :, :]  = spline_coords_small_array
            heading_angle_array[:, frame_nums, :]     = heading_angle_small_array
            body_position_array[:, frame_nums, :]     = body_position_small_array
            eye_coords_array[:, frame_nums, :, :]     = eye_coords_small_array

    if use_multiprocessing:
        pool.close()
        pool.join()
    
    # set directory for saving tracking data
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)

    # save tracking data
    np.savez(os.path.join(tracking_dir, "{}_tracking.npz".format(os.path.splitext(os.path.basename(folder_path))[0])),
                          tail_coords=tail_coords_array, spline_coords=spline_coords_array,
                          heading_angle=heading_angle_array, body_position=body_position_array,
                          eye_coords=eye_coords_array, params=params)

    # stop timer
    end_time = time.time()

    # print total tracking time
    print("Finished tracking. Total time: {}s.".format(end_time - start_time))

def open_and_track_video(video_path, params, tracking_dir, video_number=0, progress_signal=None): # todo: add video creation from tracking data
    subtract_background = params['subtract_background']
    background          = params['backgrounds'][video_number]
    crop_params         = params['crop_params']
    n_tail_points       = params['n_tail_points']
    save_video          = params['save_video']
    saved_video_fps     = params['saved_video_fps']
    use_multiprocessing = params['use_multiprocessing']
    n_crops             = len(params['crop_params'])
    batch_offset        = params['batch_offsets'][video_number]

    global n_frames_tracked
    n_frames_tracked = 0

    # start timer
    start_time = time.time()

    # open the video
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print("Error: Could not open video.")
        return

    # get video info
    fps, n_frames_total = get_video_info(video_path)

    if subtract_background and background == None:
        # get background
        background = get_background_from_video(video_path, cap, batch_offset=batch_offset)

    # initialize tracking data arrays
    tail_coords_array    = np.zeros((n_crops, n_frames_total, 2, n_tail_points)) + np.nan
    spline_coords_array  = np.zeros((n_crops, n_frames_total, 2, n_tail_points)) + np.nan
    heading_angle_array  = np.zeros((n_crops, n_frames_total, 1)) + np.nan
    body_position_array  = np.zeros((n_crops, n_frames_total, 2)) + np.nan
    eye_coords_array     = np.zeros((n_crops, n_frames_total, 2, 2)) + np.nan

    # split frame numbers into big chunks - we keep only one big chunk of frames in memory at a time
    big_split_frame_nums = split_list_into_chunks(range(n_frames_total), 3000)

    if use_multiprocessing:
        # create a pool of workers
        pool = multiprocessing.Pool(None)

    if progress_signal:
        # send an update signal to the controller
        progress_signal.emit("video", video_number, 0)

    for i in range(len(big_split_frame_nums)):
        frame_nums = big_split_frame_nums[i]

        # load this big chunk of frames
        if subtract_background:
            frames, bg_sub_frames = load_frames_from_video(video_path, cap, frame_nums, background, batch_offset)

            frames = bg_sub_frames
        else:
            frames = load_frames_from_video(video_path, cap, frame_nums, None, batch_offset)

        if len(frames) != len(frame_nums):
            frame_nums = frame_nums[:len(frames)]

        if params['invert']:
            # invert the frames
            frames = [255 - frames[i] for i in range(len(frames))]

        if i == 0 and params['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')

            writer = cv2.VideoWriter("{}_video.mov".format(os.path.splitext(os.path.basename(video_path))[0]), fourcc, params['saved_video_fps'],
                (frames[0].shape[1], frames[0].shape[0]), True)

        if use_multiprocessing:
            # split frames into small chunks - we let multiple processes deal with a chunk at a time
            split_frames = yield_chunks_from_list(frames, 10)

            # have workers process each chunk
            result_list = pool.map_async(partial(track_frames, params, None, None, None, None), split_frames).get()

            n_frames_tracked += len(frame_nums)

            if progress_signal:
                # send an update signal to the controller
                percent_complete = int(100*n_frames_tracked/n_frames_total)
                progress_signal.emit("video", video_number, percent_complete)

            n_chunks = len(result_list)

            # add results to tracking data arrays
            print(len(frames))
            tail_coords_array[:, frame_nums, :, :]   = np.concatenate([result_list[i][0] for i in range(n_chunks)], axis=1)
            spline_coords_array[:, frame_nums, :, :] = np.concatenate([result_list[i][1] for i in range(n_chunks)], axis=1)
            heading_angle_array[:, frame_nums, :]    = np.concatenate([result_list[i][2] for i in range(n_chunks)], axis=1)
            body_position_array[:, frame_nums, :]    = np.concatenate([result_list[i][3] for i in range(n_chunks)], axis=1)
            eye_coords_array[:, frame_nums, :, :]    = np.concatenate([result_list[i][4] for i in range(n_chunks)], axis=1)
        else:
            # track this big chunk of frames and add results to tracking data arrays
            (tail_coords_small_array, spline_coords_small_array,
             heading_angle_small_array, body_position_small_array, eye_coords_small_array) = track_frames(params, progress_signal, n_frames_tracked, n_frames_total, 0, frames)

            tail_coords_array[:, frame_nums, :, :]   = tail_coords_small_array
            spline_coords_array[:, frame_nums, :, :] = spline_coords_small_array
            heading_angle_array[:, frame_nums, :]    = heading_angle_small_array
            body_position_array[:, frame_nums, :]    = body_position_small_array
            eye_coords_array[:, frame_nums, :, :]    = eye_coords_small_array

        for k in range(n_crops):
            tail_coords_array[k]   = get_absolute_coords(tail_coords_array[k], params['crop_params'][k]['offset'], params['shrink_factor'])
            spline_coords_array[k] = get_absolute_coords(spline_coords_array[k], params['crop_params'][k]['offset'], params['shrink_factor'])
            body_position_array[k] = get_absolute_coords(body_position_array[k], params['crop_params'][k]['offset'], params['shrink_factor'])
            eye_coords_array[k]    = get_absolute_coords(eye_coords_array[k], params['crop_params'][k]['offset'], params['shrink_factor'])

        if params['save_video']:

            frames = load_frames_from_video(video_path, cap, frame_nums, None, batch_offset)
            if params['invert']:
                # invert the frames
                frames = [255 - frames[i] for i in range(len(frames))]

            for k in range(len(frames)):
                frame = frames[k]
                frame_num = frame_nums[k]
                # print(frame_num)
                coords = [tail_coords_array[:, frame_num, :, :], spline_coords_array[:, frame_num, :, :],
                          heading_angle_array[:, frame_num, :], body_position_array[:, frame_num, :], eye_coords_array[:, frame_num, :, :]]
                tracked_frame = add_tracking_to_frame(frame, coords, n_crops=n_crops)

                writer.write(tracked_frame)

    writer.release()

    if use_multiprocessing:
        pool.close()
        pool.join()
    
    # set directory for saving tracking data
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)

    # save tracking data
    np.savez(os.path.join(tracking_dir, "{}_tracking.npz".format(os.path.splitext(os.path.basename(video_path))[0])),
                          tail_coords=tail_coords_array, spline_coords=spline_coords_array,
                          heading_angle=heading_angle_array, body_position=body_position_array,
                          eye_coords=eye_coords_array, params=params)

    # stop timer
    end_time = time.time()

    # print total tracking time
    print("Finished tracking. Total time: {}s.".format(end_time - start_time))

def open_and_track_video_batch(params, tracking_dir, progress_signal=None):
    video_paths         = params['media_paths']

    # track each video with the same parameters
    for i in range(len(video_paths)):
        open_and_track_video(video_paths[i], params, tracking_dir, i, progress_signal)

def get_video_batch_align_offsets(params):
    video_paths = params['media_paths']

    # # open the first video
    # try:
    #     cap = cv2.VideoCapture(video_paths[0])
    # except:
    #     print("Error: Could not open video.")
    #     return

    # store first frame of the first video
    source_frame = load_frames_from_video(video_paths[0], None, [1], None)[0]

    batch_offsets = [None]

    for k in range(1, len(video_paths)):
        first_frame = load_frames_from_video(video_paths[k], None, [1], None)[0]

        transform = cv2.estimateRigidTransform(source_frame, first_frame, False)

        # print(transform)

        offset = transform[:, 2]

        batch_offsets.append(offset)

    return batch_offsets

def apply_align_offset_to_frame(frame, batch_offset):
    transform = np.float32([[1, 0, -batch_offset[0]], [0, 1, -batch_offset[1]]])
    return cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]), borderValue=255)

def track_cropped_frame(frame, params, crop_params):
    tracking_type = params['type']

    if tracking_type == "freeswimming":
        track_tail = params['track_tail']

        # track head
        heading_angle, body_position, eye_coords = track_head(frame, params, crop_params)

        if track_tail and body_position != None:
            # set tail start to coordinates of the body midpoint position
            tail_start_coords = body_position

            # track tail
            tail_coords, spline_coords = track_freeswimming_tail(frame, params, crop_params, tail_start_coords)
        else:
            tail_coords, spline_coords = [None]*2
    elif tracking_type == "headfixed":
        # set head coords to None since we aren't interested in them
        heading_angle, body_position, eye_coords = [None]*3

        # track tail
        tail_coords, spline_coords = track_headfixed_tail(frame, params, crop_params)

    return tail_coords, spline_coords, heading_angle, body_position, eye_coords

# --- Head tracking --- #

def track_head(frame, params, crop_params):
    adjust_thresholds = params['adjust_thresholds']
    eye_resize_factor = params['eye_resize_factor']
    interpolation     = translate_interpolation(params['interpolation'])
    body_threshold    = crop_params['body_threshold']
    eye_threshold     = crop_params['eye_threshold']
    track_eyes        = params['track_eyes']

    if eye_resize_factor != 1:
        orig_frame = frame.copy()
        frame = cv2.resize(frame, (0, 0), fx=eye_resize_factor, fy=eye_resize_factor, interpolation=interpolation)

    # create body threshold frame
    body_threshold_frame = simplify_body_threshold_frame(get_threshold_frame(frame, body_threshold))

    # get heading angle & body position
    heading_angle, body_position = get_heading_angle_and_position(body_threshold_frame, eye_resize_factor)

    if track_eyes:
        # create eye threshold frame
        eye_threshold_frame  = get_threshold_frame(frame, eye_threshold)

        # get eye coordinates
        eye_coords = get_eye_coords(eye_threshold_frame, eye_resize_factor)
    else:
        eye_coords = None

    if track_eyes and eye_coords == None and adjust_thresholds: # eyes not found; adjust the threshold & try again
        # initialize counter
        i = 0
        
        # create a list of head thresholds to go through
        eye_thresholds = list(range(eye_threshold-1, eye_threshold-5, -1)) + list(range(eye_threshold+1, eye_threshold+5))
        
        while eye_coords == None and i < 8:
            # create a thresholded frame using new threshold
            eye_threshold_frame = get_threshold_frame(frame, eye_thresholds[i])

            # get eye coordinates
            eye_coords = get_eye_coords(eye_threshold_frame, eye_resize_factor)

            # increase counter
            i += 1

    return heading_angle, body_position, eye_coords

def get_eye_coords(eye_threshold_image, eye_resize_factor, min_intereye_dist=3, max_intereye_dist=10): # todo: make intereye dist variables user settable
    # get eye centroids
    centroid_coords = get_centroids(eye_threshold_image, eye_resize_factor)

    if centroid_coords == None:
        # no centroids found; end here.
        return None

    # get the number of found eye centroids
    n_centroids = centroid_coords.shape[1]

    # get all permutations of pairs of centroid indices
    perms = itertools.permutations(np.arange(n_centroids), r=2)

    for p in list(perms):
        # set eye coordinates
        eye_coords = np.array([[centroid_coords[0, p[0]], centroid_coords[0, p[1]]],
                               [centroid_coords[1, p[0]], centroid_coords[1, p[1]]]])

        # find distance between eyes
        intereye_dist = np.sqrt((eye_coords[0, 1] - eye_coords[0, 0])**2 + (eye_coords[1, 1] - eye_coords[1, 0])**2)

        if not (intereye_dist < min_intereye_dist or intereye_dist > max_intereye_dist):
            # eye coords fit the criteria of min & max distance; stop looking.
            break

    return eye_coords

def get_heading_angle_and_position(head_threshold_image, eye_resize_factor=1):
    # find contours
    try:
        image, contours, _ = cv2.findContours(head_threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, _ = cv2.findContours(head_threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    try:
        # combine contours into one
        contour = np.concatenate(contours, axis=0)

        # fit an ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)

        # get center position
        position = np.array([y, x])/eye_resize_factor

        if position[0] < 0 or position[1] < 0:
            return [None]*2

        return angle, position
    except:
        return [None]*2

def get_centroids(eye_threshold_image, eye_resize_factor=1, prev_eye_coords=None): # todo: rewrite
    # find contours
    try:
        image, contours, _ = cv2.findContours(eye_threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, _ = cv2.findContours(eye_threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contour_areas = [ cv2.contourArea(contour) for contour in contours]

    if len(contour_areas) < 2:
        return None
    
    max_areas = np.argpartition(contour_areas, -2)[-2:]

    contours = [contours[i] for i in max_areas]

    # get moments
    moments = [cv2.moments(contour) for contour in contours]

    # initialize x & y coord lists
    centroid_x_coords = []
    centroid_y_coords = []

    # get coordinates
    for m in moments:
        if m['m00'] != 0.0:
            centroid_y_coords.append(m['m01']/m['m00'])
            centroid_x_coords.append(m['m10']/m['m00'])

    # put x & y coord lists into an array
    centroid_coords = np.array([centroid_y_coords, centroid_x_coords])/eye_resize_factor

    n_centroids = centroid_coords.shape[1]

    if n_centroids < 2:
        # too few centroids found -- we need at least 2 (one for each eye)
        return None

    return centroid_coords

# --- Freeswimming tail tracking --- #

def track_freeswimming_tail(frame, params, crop_params, body_position):
    adjust_thresholds  = params['adjust_thresholds']
    tail_crop          = params['tail_crop']
    min_tail_body_dist = params['min_tail_body_dist']
    max_tail_body_dist = params['max_tail_body_dist']
    n_tail_points      = params['n_tail_points']
    tail_threshold     = crop_params['tail_threshold']

    # create array of tail crop coords
    if tail_crop == None:
        tail_crop_coords = np.array([[0, frame.shape[0]], [0, frame.shape[1]]])
    else:
        tail_crop_coords = np.array([[np.maximum(0, body_position[0]-tail_crop[0]), np.minimum(frame.shape[0], body_position[0]+tail_crop[0])],
                                     [np.maximum(0, body_position[1]-tail_crop[1]), np.minimum(frame.shape[1], body_position[1]+tail_crop[1])]])
    
    # get body center position relative to the tail crop
    rel_body_position = (body_position.T - tail_crop_coords[:, 0]).T

    # crop the frame to the tail
    tail_crop_frame = frame[tail_crop_coords[0, 0]:tail_crop_coords[0, 1], tail_crop_coords[1, 0]:tail_crop_coords[1, 1]]

    # create a thresholded frame
    tail_threshold_frame = get_threshold_frame(tail_crop_frame, tail_threshold)

    # get tail coordinates
    tail_coords, spline_coords = get_freeswimming_tail_coords(tail_threshold_frame, rel_body_position,
                                                              min_tail_body_dist, max_tail_body_dist,
                                                              n_tail_points)

    if adjust_thresholds and tail_coords == None:
        # initialize counter
        i = 0
        
        # create a list of tail thresholds to go through
        tail_thresholds = list(range(tail_threshold-1, tail_threshold-5, -1)) + list(range(tail_threshold+1, tail_threshold+5))
        
        while tail_coords == None and i < 8:
            #  create a thresholded frame using new threshold
            tail_threshold_frame = get_threshold_frame(tail_crop_frame, tail_thresholds[i])

            # get tail coordinates
            tail_coords, spline_coords = get_freeswimming_tail_coords(tail_threshold_frame, rel_body_position,
                                                                      min_tail_body_dist, max_tail_body_dist,
                                                                      n_tail_points)

            # increase counter
            i += 1

    if tail_coords != None:
        # convert tail coords to be relative to initial frame
        tail_coords   += tail_crop_coords[:, 0][:, np.newaxis].astype(int)
        spline_coords += tail_crop_coords[:, 0][:, np.newaxis].astype(int)

    return tail_coords, spline_coords

def get_freeswimming_tail_coords(tail_threshold_frame, body_position, min_tail_body_dist, max_tail_body_dist, n_tail_points, max_r=4, smoothing_factor=3): # todo: make max radius & smoothing factor user settable
    # get tail skeleton matrix
    skeleton_matrix = get_tail_skeleton_frame(tail_threshold_frame)

    # get coordinates of nonzero points of thresholded image
    nonzeros = np.nonzero(skeleton_matrix)

    # zero out pixels that are close to body
    for (r, c) in zip(nonzeros[0], nonzeros[1]):
        if np.sqrt((r - body_position[0])**2 + (c - body_position[1])**2) < min_tail_body_dist:
            skeleton_matrix[r, c] = 0

    try:
        image, contours, _ = cv2.findContours(skeleton_matrix, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, _ = cv2.findContours(skeleton_matrix, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) > 0:
        max_length_index = np.argmax([ len(contour) for contour in contours ])

        tail_coords = contours[max_length_index]

        tail_coords = [ (i[0][1], i[0][0]) for i in tail_coords ]

        min_body_distance = np.sqrt((tail_coords[0][0] - body_position[0])**2 + (tail_coords[0][1] - body_position[1])**2)
        startpoint_index = 0

        for i in range(1, len(tail_coords)-1):
            if tail_coords[i-1] == tail_coords[i+1]:
                r = tail_coords[i][0]
                c = tail_coords[i][1]
                body_distance = np.sqrt((r - body_position[0])**2 + (c - body_position[1])**2)
                if body_distance < min_body_distance:
                    min_body_distance = body_distance
                    startpoint_index = i

        if startpoint_index != 0:
            items = deque(tail_coords)
            items.rotate(-startpoint_index)
            tail_coords = list(items)

        min_diff = 10000
        endpoint_index = None

        for i in range(1, len(tail_coords)-1):
            if tail_coords[i-1] == tail_coords[i+1]:
                dist_1 = i
                dist_2 = len(tail_coords)-i
                diff = abs(dist_2 - dist_1)
                if diff < min_diff:
                    min_diff = diff
                    endpoint_index = i

        tail_coords = tail_coords[:endpoint_index]

        tail_coords = np.array(tail_coords).T
    else:
        tail_coords = None

    if tail_coords == None:
        # couldn't get tail coordinates; end here.
        return [None]*2

    # get number of tail coordinates
    n_tail_coords = tail_coords.shape[1]

    # get size of thresholded image
    y_size = tail_threshold_frame.shape[0]
    x_size = tail_threshold_frame.shape[1]

    if tail_coords != None:
        # convert tail coordinates to floats
        tail_coords = tail_coords.astype(float)

    # modify tail skeleton coordinates (Huang et al., 2013)
    for i in range(n_tail_coords):
        y = tail_coords[0, i]
        x = tail_coords[1, i]

        pixel_sum = 0
        y_sum     = 0
        x_sum     = 0

        for k in range(-1, 2):
            for l in range(-1, 2):
                pixel_sum += tail_threshold_frame[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]
                y_sum     += k*tail_threshold_frame[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]
                x_sum     += l*tail_threshold_frame[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]

        if pixel_sum != 0:
            y_sum /= float(pixel_sum)
            x_sum /= float(pixel_sum)

            tail_coords[0, i] = y + y_sum
            tail_coords[1, i] = x + x_sum

    try:
        # make ascending spiral in 3D space
        t = np.zeros(n_tail_coords)
        t[1:] = np.sqrt((tail_coords[1, 1:] - tail_coords[1, :-1])**2 + (tail_coords[0, 1:] - tail_coords[0, :-1])**2)
        t = np.cumsum(t)
        t /= t[-1]

        nt = np.linspace(0, 1, 100)

        # calculate cubic spline
        spline_y_coords = interpolate.UnivariateSpline(t, tail_coords[0, :], k=3, s=smoothing_factor)(nt)
        spline_x_coords = interpolate.UnivariateSpline(t, tail_coords[1, :], k=3, s=smoothing_factor)(nt)

        spline_coords = np.array([spline_y_coords, spline_x_coords])
    except:
        print("Error: Could not calculate tail spline.")
        return [None]*2

    # get number of spline coordinates
    n_spline_coords = spline_coords.shape[1]

    r = lambda m, n: [i*n//m + n//(2*m) for i in range(m)] # generates a list of m evenly spaced numbers from 0 to n

    if n_tail_coords > n_tail_points:
        # get evenly spaced tail indices
        tail_nums = [0] + r(n_tail_points-2, tail_coords.shape[1]) + [tail_coords.shape[1]-1]

        tail_coords = tail_coords[:, tail_nums]

    if n_spline_coords > n_tail_points:
        # get evenly spaced spline indices
        spline_nums = [0] + r(n_tail_points-2, spline_coords.shape[1]) + [spline_coords.shape[1]-1]

        spline_coords = spline_coords[:, spline_nums]

    return tail_coords, spline_coords

def get_ordered_tail_coords(skeleton_matrix, max_r, body_position, min_tail_body_dist, max_tail_body_dist, min_n_tail_points):
    # get size of matrix
    y_size = skeleton_matrix.shape[0]
    x_size = skeleton_matrix.shape[1]

    # get coordinates of nonzero points of skeleton matrix
    nonzeros = np.nonzero(skeleton_matrix)

    # initialize tail starting point coordinates
    tail_start_coords = None

    # initialize variable for storing the closest distance to the center of the body
    closest_body_distance = None

    # loop through all nonzero points of the tail skeleton
    for (r, c) in zip(nonzeros[0], nonzeros[1]):
        # get distance of this point to the body center
        body_distance = np.sqrt((r - body_position[0])**2 + (c - body_position[1])**2)

        # look for an endpoint near the body
        if body_distance < max_tail_body_dist:
            if (closest_body_distance == None) or (closest_body_distance != None and body_distance < closest_body_distance):
                # either no point has been found yet or this is a closer point than the previous best

                # get nonzero elements in 3x3 neigbourhood around the point
                nonzero_neighborhood = skeleton_matrix[r-1:r+2, c-1:c+2] != 0

                # if the number of non-zero points in the neighborhood is at least 2
                # (ie. there's at least one direction to move in along the tail),
                # set this to our tail starting point.
                if np.sum(nonzero_neighborhood) == 2:
                    tail_start_coords = np.array([r, c])
                    closest_body_distance = body_distance

    if tail_start_coords == None:
        # still could not find start of the tail; end here.
        return None

    # walk along the tail
    found_coords = walk_along_tail(tail_start_coords, max_r, skeleton_matrix)

    # if len(found_coords) < min_n_tail_points:
    #     print("try again")
    #     # we didn't manage to get the full tail; try moving along the tail in reverse
    #     found_coords = walk_along_tail(found_coords[-1], max_r, skeleton_matrix)

    # zero out pixels that are close to body and eyes
    for i in range(len(found_coords)-1, -1, -1):
        if np.sqrt((found_coords[i][0] - body_position[0])**2 + (found_coords[i][1] - body_position[1])**2) < min_tail_body_dist:
            del found_coords[i]

    if len(found_coords) < min_n_tail_points:
        # we still didn't get enough tail points; give up here.
        return None

    # convert to an array
    found_coords = np.array(found_coords).T

    return found_coords

def walk_along_tail(tail_start_coords, max_r, skeleton_matrix):
    found_coords = [tuple(tail_start_coords)]

    # initialize radius (half the side length of square area to look in for the next point)
    r = 1

    # set maximum tail points to avoid crazy things happening
    max_tail_points = 200

    while len(found_coords) < max_tail_points and r <= max_r:
        # find coordinates of the next point
        next_coords = find_next_tail_coords_in_neighborhood(found_coords, r, skeleton_matrix)

        if next_coords != None:
            # add coords to found coords list
            found_coords.append(tuple(next_coords))
        else:
            r += 1

    return found_coords

def find_next_tail_coords_in_neighborhood(found_coords, r, skeleton_matrix):
    # pad the skeleton matrix with zeros
    padded_matrix = np.zeros((skeleton_matrix.shape[0] + 2*r, skeleton_matrix.shape[1] + 2*r))
    padded_matrix[r:-r, r:-r] = skeleton_matrix

    # get x, y of last coordinates
    last_coords = found_coords[-1]
    y = last_coords[0]
    x = last_coords[1]

    # get neighborhood around the current point
    neighborhood = padded_matrix[y:y+2*r+1, x:x+2*r+1]

    # get coordinates of nonzero elements in the neighborhood
    nonzero_y_coords, nonzero_x_coords = np.nonzero(neighborhood)

    # translate these coordinates to non-padded image coordinates
    diff_x = r - x
    diff_y = r - y
    nonzero_y_coords -= diff_y
    nonzero_x_coords -= diff_x

    # convert to a list of coordinates
    nonzeros = [tuple(a) for a in list(np.vstack([nonzero_y_coords, nonzero_x_coords]).T)]

    # find the next point(s) we can traverse
    unique_coords = find_unique_coords(nonzeros, found_coords)

    if unique_coords == None or len(unique_coords) == 0:
        # all of the nonzero points have been traversed already; end here
        return None

    if r > 1:
        # find the closest point to the last found coordinate
        distances = [ np.sqrt((unique_coords[i][0] - found_coords[-1][0])**2 + (unique_coords[i][1] - found_coords[-1][1])**2) for i in xrange(len(unique_coords))]
        closest_index = np.argmin(distances)

        next_coords = unique_coords[closest_index]
    else:
        # just pick the first unique point that was found
        next_coords = unique_coords[0]

    return next_coords

def find_unique_coords(coords, found_coords):
    return [o for o in coords if o not in set(found_coords)]

# --- Headfixed tail tracking --- #

def track_headfixed_tail(frame, params, crop_params, smoothing_factor=30): # todo: make smoothing factor user settable
    tail_start_coords = get_relative_coords(params['tail_start_coords'], crop_params['offset'], params['shrink_factor'])
    direction         = params['tail_direction']
    n_tail_points     = params['n_tail_points']

    global fitted_tail, tail_funcs, tail_brightness, background_brightness, tail_length
    
    # check whether we are processing the first frame
    first_frame = tail_funcs == None

    # convert tail direction to a vector
    directions={ "Up": [0,-1], "Down": [0,1], "Left": [-1,0], "Right": [1,0] }

    # set maximum tail points to avoid crazy things happening
    max_tail_points = 200

    # initialize tail fitted coords array for this frame
    frame_fit = np.zeros((max_tail_points, 2))

    # initialize lists of variables
    widths, convolution_results = [],[]
    test, slices                = [], []
    
    # pick an initial guess for the direction vector
    guess_vector = np.array(directions[direction])
    
    # set an approximate width of the tail (px)
    guess_tail_width = 50
    
    # flip x, y in tail start coords
    tail_start_coords = [tail_start_coords[1], tail_start_coords[0]]
    
    if first_frame:
        # set current point
        current_point = np.array(tail_start_coords)
        
        # get histogram of pixel brightness for the frame
        if frame.ndim == 2:
            histogram = np.histogram(frame[:, :], 10, (0, 255))
        elif frame.ndim == 3:
            histogram = np.histogram(frame[:, :, 0], 10, (0, 255))
        
        # get average background brightness
        background_brightness = histogram[1][histogram[0].argmax()]/2 + histogram[1][min(histogram[0].argmax()+1, len(histogram[0]))]/2

        # get average tail brightness from a 2x2 area around the current point
        if frame.ndim == 2:
            tail_brightness = frame[current_point[1]-2:current_point[1]+3, current_point[0]-2:current_point[0]+3].mean()
        elif frame.ndim == 3:
            tail_brightness = frame[current_point[1]-2:current_point[1]+3, current_point[0]-2:current_point[0]+3, 0].mean()
        
        # create a Gaussian pdf (we will use this to find the midline of the tail)
        normpdf = pylab.normpdf(np.arange(-guess_tail_width/4.0, (guess_tail_width/4.0)+1), 0, 8)
    else:
        # set current point to the first point that was found in the last tracked frame
        current_point= fitted_tail[-1][0, :]
    
    # set spacing of tail points
    tail_point_spacing = 5
    
    for count in range(max_tail_points):
        if count == 0:
            guess = current_point
        elif count == 1:
            guess = current_point + guess_vector*tail_point_spacing
        else:
            # normalize guess vector
            guess_vector = guess_vector/np.linalg.norm(guess_vector)
            guess = current_point + guess_vector*tail_point_spacing
        
        # get points that cover a line perpendicular to the direction vector (ie. cover a cross section of the tail)
        guess_line_start = guess + np.array([-guess_vector[1], guess_vector[0]])*guess_tail_width/2
        guess_line_end   = guess + np.array([guess_vector[1], -guess_vector[0]])*guess_tail_width/2
        x_indices = np.linspace(guess_line_start[0], guess_line_end[0], guess_tail_width).astype(int)
        y_indices = np.linspace(guess_line_start[1], guess_line_end[1], guess_tail_width).astype(int)
        
        # clip the perpendicular line to within the size of the image
        if max(y_indices) >= frame.shape[0] or min(y_indices) < 0 or max(x_indices) >= frame.shape[1] or min(x_indices) < 0:
            y_indices = np.clip(y_indices, 0, frame.shape[0]-1)
            x_indices = np.clip(x_indices, 0, frame.shape[1]-1)
        
        # get a cross-sectional slice of the tail
        guess_slice = frame[y_indices, x_indices]

        if guess_slice.ndim == 2:
            guess_slice = guess_slice[:, 0]
        else:
            guess_slice = guess_slice[:]
        
        # extract the tail by subtracting the background brightness
        if tail_brightness < background_brightness:
            guess_slice = (background_brightness - guess_slice)
        else:
            guess_slice = (guess_slice - background_brightness)
        
        # add to list of slices
        slices += [guess_slice]

        # get a histogram of the slice
        histogram = np.histogram(guess_slice, 10)

        # subtract the most common brightness (baseline subtraction)
        guess_slice = guess_slice - guess_slice[((histogram[1][histogram[0].argmax()] <= guess_slice) & (guess_slice < histogram[1][histogram[0].argmax()+1]))].mean()
        
        # remove noise
        sguess = scipy.ndimage.filters.percentile_filter(guess_slice, 50, 5)
    
        if first_frame:
            # first time through, profile the tail

            # find tail edge indices
            tail_edges = np.where(np.diff(sguess > sguess.max()*0.25))[0]

            if len(tail_edges) >= 2:
                # set midpoint of tail edge indices to 0
                tail_edges = tail_edges - len(sguess)/2.0

                # get two closest tail edge indices
                tail_indices = tail_edges[np.argsort(np.abs(tail_edges))[0:2]]

                # get midpoint of tail edges (and change the midpoint back)
                result_index_new = tail_indices.mean() + len(sguess)/2.0

                # add width of tail to tail widths list
                widths += [abs(tail_indices[0] - tail_indices[1])]
            else:
                # end the tail here
                result_index_new = None
                tail_length = count
                break

            # convolve the tail slice with the Gaussian pdf
            results = np.convolve(normpdf, guess_slice, "valid")

            # add to convolution results list
            convolution_results += [results]

            # get the index of the point with max brightness, and adjust to match the size of the tail slice
            result_index = results.argmax() - results.size/2 + guess_slice.size/2

            # get point that corresponds to this index
            new_point = np.array([x_indices[result_index_new], y_indices[result_index_new]])
        else:
            # convolve the tail slice with the tail profile
            results = np.convolve(tail_funcs[count], guess_slice, "valid")

            # get the index of the point with max brightness, and adjust to match the size of the tail slice
            result_index = results.argmax() - results.size/2 + guess_slice.size/2

            # get point that corresponds to this index
            new_point = np.array([x_indices[result_index], y_indices[result_index]])
    
        if first_frame:
            if count > 10:
                # get contrast in all convolution results
                trapz = [pylab.trapz(result - result.mean()) for result in convolution_results]

                slicesnp = np.vstack(slices)

                if np.array(trapz[-3:]).mean() < .2:
                    # contrast in last few slices is too low; end tail here
                    tail_length = count
                    break
                elif slicesnp[-1, result_index-2:result_index+2].mean() < 10:
                    # brightness too low; end tail here
                    tail_length = count
                    break
        elif count > tail_length*.8 and np.sum((new_point - current_point)**2)**.5 > tail_point_spacing*1.5:
            # distance between points is too high; end here
            break             
        elif count == tail_length:
            # reached the length of the tail; end here
            break
        
        # add point to the fitted tail
        frame_fit[count, :] = new_point
        
        if count>0:
            # compute a new guess vector based on the difference between the new point and the last one
            guess_vector = new_point-current_point

        # update current point
        current_point = new_point
    
    if first_frame:
        # remove noise from tail width
        swidths = scipy.ndimage.filters.percentile_filter(widths, 50, 8)

        # pad tail width array
        swidths = np.lib.pad(swidths, [0, 5], mode='edge')

        # compute functions that profile the tail
        tail_funcs = [ calculate_tail_func(np.arange(-guess_tail_width, guess_tail_width), 0, swidth, 1, 0) for swidth in swidths]

    # append fitted tail to list of tail fits
    fitted_tail.append(np.copy(frame_fit[:count]))
    
    # get tail coordinates
    tail_coords = np.fliplr(frame_fit[:count]).T

    r = lambda m, n: [i*n//m + n//(2*m) for i in range(m)] # generates a list of m evenly spaced numbers from 0 to n

    # get number of tail coordinates
    n_tail_coords = tail_coords.shape[1]
    
    if n_tail_coords > n_tail_points:
        # get evenly spaced tail indices
        tail_nums = [0] + r(n_tail_points-2, tail_coords.shape[1]) + [tail_coords.shape[1]-1]
    
        tail_coords = tail_coords[:, tail_nums]

    # get number of tail coordinates
    n_tail_coords = tail_coords.shape[1]
    
    try:
        # make ascending spiral in 3D space
        t = np.zeros(n_tail_coords)
        t[1:] = np.sqrt((tail_coords[1, 1:] - tail_coords[1, :-1])**2 + (tail_coords[0, 1:] - tail_coords[0, :-1])**2)
        t = np.cumsum(t)
        t /= t[-1]
    
        nt = np.linspace(0, 1, 100)
    
        # calculate cubic spline
        spline_y_coords = interpolate.UnivariateSpline(t, tail_coords[0, :], k=3, s=smoothing_factor)(nt)
        spline_x_coords = interpolate.UnivariateSpline(t, tail_coords[1, :], k=3, s=smoothing_factor)(nt)
    
        spline_coords = np.array([spline_y_coords, spline_x_coords])
    except:
        print("Error: Could not calculate tail spline.")
        return [None]*2
    
    # get number of spline coordinates
    n_spline_coords = spline_coords.shape[1]
    
    if n_spline_coords > n_tail_points:
        # get evenly spaced spline indices
        frame_nums = [0] + r(n_tail_points-2, spline_coords.shape[1]) + [spline_coords.shape[1]-1]
    
        spline_coords = spline_coords[:, frame_nums]
    
    return tail_coords, spline_coords

def calculate_tail_func(x, mu, sigma, scale, offset):
    return scale * np.exp(-(x-mu)**4/(2.0*sigma**2))**.2 + offset

def clear_headfixed_tracking():
    global fitted_tail, tail_funcs, tail_brightness, background_brightness, tail_length

    fitted_tail           = []
    tail_funcs            = None
    tail_brightness       = None
    background_brightness = None
    tail_length           = None

# --- Helper functions --- #

def translate_interpolation(interpolation_string):
    # get matching opencv interpolation variable from string
    if interpolation_string == 'Nearest Neighbor':
        interpolation = cv2.INTER_NEAREST
    elif interpolation_string == 'Linear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation_string == 'Bicubic':
        interpolation = cv2.INTER_CUBIC
    elif interpolation_string == 'Lanczos':
        interpolation = cv2.INTER_LANCZOS4

    return interpolation

def get_num_frames_in_folder(folder_path):
    n_frames = 0

    for filename in sort_nicely(os.listdir(folder_path)):
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            n_frames += 1

    return n_frames

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

def add_tracking_to_frame(frame, coords, cropped=False, n_crops=1):
    tail_coords    = coords[0]
    spline_coords  = coords[1]
    heading_angle  = coords[2]
    body_position  = coords[3]
    eye_coords     = coords[4]

    # convert to BGR
    if len(frame.shape) < 3:
        tracked_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        tracked_frame = frame

    if cropped == True:
        # add extra dimension for # of crops
        if tail_coords != None:
            tail_coords   = tail_coords[np.newaxis, :, :]
        if spline_coords != None:
            spline_coords = spline_coords[np.newaxis, :, :]
        if heading_angle != None:
            heading_angle = np.array([[heading_angle]])
        if body_position != None:
            body_position = body_position[np.newaxis, :]
        if eye_coords != None:
            eye_coords    = eye_coords[np.newaxis, :, :]

            # print(eye_coords.shape)

    for k in range(n_crops):
        # add eye points
        if eye_coords != None and eye_coords.shape[-1] == 2:
            for i in range(2):
                if not np.isnan(eye_coords[k, 0, i]) and not np.isnan(eye_coords[k, 1, i]):
                    cv2.circle(tracked_frame, (int(round(eye_coords[k, 1, i])), int(round(eye_coords[k, 0, i]))), 1, (0, 0, 255), -1)

        # if body_position != None and body_position.shape[-1] == 2:
        #     if not np.isnan(body_position[k, 0]) and not np.isnan(body_position[k, 1]):
        #         cv2.circle(tracked_frame, (int(round(body_position[k, 1])), int(round(body_position[k, 0]))), 2, (255, 0, 255), -1)

        if spline_coords != None and spline_coords[k, 0, 0] != np.nan:
            # add tail points
            spline_length = spline_coords.shape[2]
            for i in range(spline_length-1):
                if (not np.isnan(spline_coords[k, 0, i]) and not np.isnan(spline_coords[k, 1, i])
                    and not np.isnan(spline_coords[k, 0, i+1]) and not np.isnan(spline_coords[k, 1, i+1])):
                    cv2.line(tracked_frame, (int(round(spline_coords[k, 1, i])), int(round(spline_coords[k, 0, i]))),
                                            (int(round(spline_coords[k, 1, i+1])), int(round(spline_coords[k, 0, i+1]))), (255, 0, 0), 1)

        if tail_coords != None and tail_coords[k, 0, 0] != np.nan:
            # add tail points
            tail_length = tail_coords.shape[2]
            # print(tail_length)
            for i in range(tail_length):
                if not np.isnan(tail_coords[k, 0, i]) and not np.isnan(tail_coords[k, 1, i]):
                    # cv2.circle(tracked_frame, (int(round(tail_coords[k, 1, i])), int(round(tail_coords[k, 0, i]))), 1, (255, 255, 0), -1)
                    cv2.line(tracked_frame, (int(round(tail_coords[k, 1, i])), int(round(tail_coords[k, 0, i]))),
                                            (int(round(tail_coords[k, 1, i])), int(round(tail_coords[k, 0, i]))), (255, 255, 0), 1)

        # if tail_coords != None and tail_coords[k, 0, 0] != np.nan:
        #     # add tail points
        #     tail_length = tail_coords.shape[2]
        #     for i in range(tail_length-1):
        #         if (not np.isnan(tail_coords[k, 0, i]) and not np.isnan(tail_coords[k, 1, i])
        #             and not np.isnan(tail_coords[k, 0, i+1]) and not np.isnan(tail_coords[k, 1, i+1])):
        #             cv2.line(tracked_frame, (int(round(tail_coords[k, 1, i])), int(round(tail_coords[k, 0, i]))),
        #                                     (int(round(tail_coords[k, 1, i+1])), int(round(tail_coords[k, 0, i+1]))), (255, 0, 0), 1)

    return tracked_frame

def crop_frame(frame, offset, crop):
    if offset != None and crop != None:
        return frame[offset[0]:offset[0] + crop[0], offset[1]:offset[1] + crop[1]]
    else:
        return frame

def shrink_frame(frame, shrink_factor):
    if shrink_factor != 1:
        frame = cv2.resize(frame, (0, 0), fx=shrink_factor, fy=shrink_factor)
    return frame

def get_threshold_frame(frame, threshold):
    _, threshold_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)
    # print(frame)
    np.divide(threshold_frame, 255, out=threshold_frame, casting='unsafe')
    return threshold_frame

def simplify_body_threshold_frame(frame):
    # remove noise from the thresholded image
    kernel = np.ones((2, 2),np.uint8)
    frame = cv2.erode(frame,kernel,iterations = 1)
    frame = cv2.dilate(frame,kernel,iterations = 1)

    return frame

def get_tail_skeleton_frame(tail_threshold_frame):
    # kernel = np.ones((3, 3),np.uint8)

    # return skeletonize(cv2.morphologyEx(tail_threshold_frame, cv2.MORPH_OPEN, kernel)).astype(np.uint8)
    # print(tail_threshold_frame.shape, type(tail_threshold_frame))

    # thin = bwmorph_thin.bwmorph_thin(tail_threshold_frame)

    return skeletonize(tail_threshold_frame).astype(np.uint8)

    # skeleton_matrix = bwmorph_thin.bwmorph_thin(cv2.morphologyEx(tail_threshold_frame, cv2.MORPH_OPEN, kernel)).astype(np.uint8)

    # endpoints     = np.nonzero(skeleton.endPoints(skeleton_matrix))
    # branchpoints  = np.nonzero(skeleton.branchedPoints(skeleton_matrix))
    # print(branchpoints)

    # for (r, c) in zip(branchpoints[0], branchpoints[1]):
    #     skeleton_matrix[r, c] = 0

    # try:
    #     image, contours, _ = cv2.findContours(skeleton_matrix, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # except ValueError:
    #     contours, _ = cv2.findContours(skeleton_matrix, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # contour_areas = [ cv2.contourArea(contour) for contour in contours]

    # for i in range(len(contours)):
    #     if len(contours[i]) < 10:
    #         cv2.drawContours(skeleton_matrix, contours, i, (0, 0, 0))

    # for point in zip(branchpoints[0], branchpoints[1]):
    #     skeleton_matrix[point] = 1

    # return skeleton_matrix
    # return thinning.guo_hall_thinning(tail_threshold_frame.astype(np.ubyte))
    # return skeleton.pruning(thinning.guo_hall_thinning(tail_threshold_frame.astype(np.ubyte)), 3).astype(np.uint8)

def get_relative_tail_crop(tail_crop, shrink_factor):
    return tail_crop*shrink_factor

def get_relative_coords(coords, offset, shrink_factor):
    return (coords - offset)*shrink_factor

def get_absolute_coords(coords, offset, shrink_factor):
    if coords.ndim == 1:
        return coords/shrink_factor + offset
    elif coords.ndim == 2:
        return coords/shrink_factor + offset[np.newaxis, :]
    elif coords.ndim == 3:
        return coords/shrink_factor + offset[np.newaxis, :, np.newaxis]

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

    return l

def split_evenly(n, m, start=0):
    # generate a list of m evenly spaced numbers in the range of (start, start + n)
    # eg. split_evenly(100, 5, 30) = [40, 60, 80, 100, 120]
    return [i*n//m + n//(2*m) + start for i in range(m)]

def get_ellipse(moments, direct):
    # GET_ELLIPSE Equivalent ellipse of an image
    #   E = IM.GET_ELLIPSE(IMG) finds the equivalent ellipse of an image IMG. 
    #   IMG is a n-by-m image, and E is a structure containing the ellipse 
    #   properties.
    #
    #   E = IM.GET_ELLIPSE(..., false) will not try to assign a
    #   direction based on the third moment. The orientation of the object will
    #   be unchanged but the direction is pi-undetermined.
    #
    #   See also: IM.draw_ellipse
     
    # --- Default values
    if direct == None:
        direct = true

    # --- Get the Moments
    m00 = moments['m00']
    m10 = moments['m10']
    m01 = moments['m01']
    m11 = moments['m11']
    m02 = moments['m02']
    m20 = moments['m20']
     
    # --- Ellipse properties

    # Barycenter
    x = m10/m00
    y = m01/m00
     
    # Central moments (intermediary step)
    a = m20/m00 - x**2
    b = 2*(m11/m00 - x*y)
    c = m02/m00 - y**2
     
    # Orientation (radians)
    theta = 1/2*np.arctan(b/(a-c)) + int(a < c)*np.pi/2
     
    # Minor and major axis
    w = np.sqrt(6*(a+c-np.sqrt(b**2+(a-c)**2)))/2
    l = np.sqrt(6*(a+c+np.sqrt(b**2+(a-c)**2)))/2
     
    # Ellipse focal points
    d = np.sqrt(l**2-w**2)
    x1 = x + d*np.cos(theta)
    y1 = y + d*np.sin(theta)
    x2 = x - d*np.cos(theta)
    y2 = y - d*np.sin(theta)
     
    return x1, y1, x2, y2

def yield_chunks_from_list(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def split_list_into_chunks(l, n):
    """Return a list of n-sized chunks from l."""
    return [ l[i:i + n] for i in xrange(0, len(l), n) ]