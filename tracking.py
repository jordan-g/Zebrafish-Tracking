from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cv2
import traceback

import scipy.ndimage
import scipy.stats
from scipy import interpolate
import pylab

import os
import re
import itertools
import time
import pdb
import psutil
from scipy import sparse

import multiprocessing
from multiprocessing import sharedctypes

from functools import partial
from itertools import chain

from moviepy.video.io.ffmpeg_reader import *

from skimage.morphology import skeletonize, thin
from collections import deque

from open_media import open_image, open_video
import utilities
import analysis

try:
    xrange
except:
    xrange = range

# Headfixed tail tracking global variables
fitted_tail           = []
tail_funcs            = None
tail_brightness       = None
background_brightness = None
tail_length           = None

cv2.setNumThreads(0) # Avoids crashes when using multiprocessing with OpenCV

# --- Nick's Added Functions --- #

def subtract_background_from_frames_extended(frames, background, threshold_value = 3, morph = True, kernel_size = [3, 3], n_iterations = 1):
    background_subtracted_frames = []
    for frame in frames:
        absoluate_difference_between_background_and_frame = calculate_absolute_difference_between_background_and_frame(frame, background)
        threshold_frame = apply_threshold_to_frame(absoluate_difference_between_background_and_frame, threshold_value = threshold_value)
        fish_contour_frame = extract_fish_contour_from_threshold_frame(threshold_frame, morph = morph, kernel_size = kernel_size, n_iterations = n_iterations)
        background_subtracted_frames.append(crop_fish_from_frame_using_fish_contour(frame, fish_contour_frame))
    return background_subtracted_frames

def extract_background_extended(video_path, num_backgrounds = 1, threshold_value = 10, morph = True, kernel_size = [3, 3], n_iterations = 1, save_background = False):
    if not os.path.isfile(video_path):
        print('Error! Video: {0} does not exist. Check to make sure the video path has been entered correctly.'.format(video_path))
        return
    background_array = calculate_backgrounds_as_brightest_pixel_value(video_path, num_backgrounds = num_backgrounds)
    try:
        capture = cv2.VideoCapture(video_path)
    except:
        print('Error! Could not open video.'.format(video_path))
        return
    # video_total_frames = get_video_info(video_path)[1]
    video_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    background_chunk_index = int(video_total_frames / num_backgrounds)
    for frame_num in range(video_total_frames):
        print('Extracting running average background using fish contours. Processing frame number: {0}/{1}.'.format(frame_num + 1, video_total_frames), end = '\r')
        success, frame = capture.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if frame_num == 0:
                frame_sum = np.zeros(np.shape(frame))
                contour_sum = np.zeros(np.shape(frame))
            n_background = int(frame_num/background_chunk_index)
            try:
                background_subtracted_frame = calculate_absolute_difference_between_background_and_frame(frame, background_array[n_background])
            except:
                background_subtracted_frame = calculate_absolute_difference_between_background_and_frame(frame, background_array[n_background - 1])
            threshold_frame = apply_threshold_to_frame(background_subtracted_frame, threshold_value = threshold_value)
            fish_contour_frame = extract_fish_contour_from_threshold_frame(threshold_frame, morph = morph, kernel_size = kernel_size, n_iterations = n_iterations)
            fish_contour_frame = -fish_contour_frame/255 + 1
            frame_sum += frame * fish_contour_frame
            contour_sum += fish_contour_frame
    print('Extracting running average background using fish contours. Processing frame number: {0}/{1}.'.format(frame_num + 1, video_total_frames))
    background = frame_sum / contour_sum
    capture.release()
    if save_background:
        background_path = '{0}_background.tif'.format(video_path[:-4])
        cv2.imwrite(background_path, background.astype(np.uint8))
    return background.astype(np.uint8)

def calculate_backgrounds_as_brightest_pixel_value(video_path, num_backgrounds = 1):
    if not os.path.isfile(video_path):
        print('Error! Video: {0} does not exist. Check to make sure the video path has been entered correctly.'.format(video_path))
        return
    try:
        capture = cv2.VideoCapture(video_path)
    except:
        print('Error! Could not open video.')
        return
    background_array = []
    # video_total_frames = get_video_info(video_path)[1]
    video_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    background_chunk_index = int(video_total_frames / num_backgrounds)
    if num_backgrounds >= video_total_frames:
        print('Error! Number of backgrounds requested exceeds the total number of frames in the video.')
        return
    for frame_num in range(video_total_frames):
        print('Calculating background. Processing frame number: {0}/{1}.'.format(frame_num + 1, video_total_frames), end = '\r')
        success, frame = capture.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_num == 0:
                background = frame.copy().astype(np.float32)
            mask = np.less(background, frame)
            background[mask] = frame[mask]
            if frame_num > 0 and frame_num % background_chunk_index == 0:
                background_array.append(background)
            elif len(background_array) < num_backgrounds:
                if (frame_num + 1) == video_total_frames:
                    background_array.append(background)
    print('Calculating background. Processing frame number: {0}/{1}.'.format(frame_num + 1, video_total_frames))
    capture.release()
    return background_array

def calculate_absolute_difference_between_background_and_frame(frame, background):
    background_subtracted_frame = cv2.absdiff(frame, background)
    return background_subtracted_frame

def apply_threshold_to_frame(frame, threshold_value = 10, inverted = False):
    if inverted:
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY
    threshold_frame = cv2.threshold(frame.astype(np.uint8), threshold_value, 255, threshold_type)[1]
    return threshold_frame

def extract_fish_contour_from_threshold_frame(threshold_frame, morph = False, kernel_size = [3, 3], n_iterations = 1):
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size[0], kernel_size[1]))
        threshold_frame = cv2.dilate(threshold_frame, kernel, iterations = n_iterations)
        threshold_frame = cv2.erode(threshold_frame, kernel, iterations = n_iterations)
    contours = cv2.findContours(threshold_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    max_contour = np.concatenate(max(contours, key = cv2.contourArea))
    fish_contour = np.zeros(np.shape(threshold_frame))
    for i in range(len(max_contour)):
        fish_contour[max_contour[i][1]][max_contour[i][0]] = threshold_frame[max_contour[i][1]][max_contour[i][0]]
    fish_contour = cv2.fillPoly(fish_contour, pts = [max_contour], color = (255,255,255))
    return fish_contour

def crop_fish_from_frame_using_fish_contour(frame, fish_contour):
    fish_frame = np.ones(np.shape(fish_contour)) * 255
    fish_contour_values = np.where(fish_contour)
    for i in range(np.shape(fish_contour_values)[1]):
        fish_frame[fish_contour_values[0][i]][fish_contour_values[1][i]] = frame[fish_contour_values[0][i]][fish_contour_values[1][i]]
    return fish_frame

def get_tail_threshold_frame(frame, tail_threshold, inverted = True, morph = True, kernel_size = [3, 3], n_iterations = 1):
    threshold_frame = apply_threshold_to_frame(frame, threshold_value = tail_threshold, inverted = inverted)
    tail_threshold_frame = extract_fish_contour_from_threshold_frame(threshold_frame, morph = morph, kernel_size = kernel_size, n_iterations = n_iterations)
    return tail_threshold_frame

def get_body_threshold_frame(frame, body_threshold, inverted = True, morph = True, kernel_size = [3, 3], n_iterations = 1):
    threshold_frame = apply_threshold_to_frame(frame, threshold_value = body_threshold, inverted = inverted)
    body_threshold_frame = extract_fish_contour_from_threshold_frame(threshold_frame, morph = morph, kernel_size = kernel_size, n_iterations = n_iterations)
    return body_threshold_frame

def extract_body_position_extended(frame, eyes_threshold_value, body_threshold_value, threshold_value = None, threshold_step = 1, erode = False, kernel_size = [3, 3], n_iterations = 1):
    if threshold_value == None:
        threshold_value = eyes_threshold_value
    body_position = None
    try:
        if threshold_value < body_threshold_value:
            threshold_frame = get_threshold_frame(frame, threshold_value)
            if erode:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size[0], kernel_size[1]))
                threshold_frame = cv2.erode(threshold_frame, kernel, iterations = n_iterations)
            contours = cv2.findContours(threshold_frame.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
            if len(contours) == 3:
                moments = [cv2.moments(contours[i]) for i in range(len(contours))]
                if np.min(np.array([moments[i]['m00'] for i in range(len(moments))])) < 1:
                    threshold_value += threshold_step
                    body_position = extract_body_position_extended(frame = frame, eyes_threshold_value = eyes_threshold_value, body_threshold_value = body_threshold_value, threshold_value = threshold_value, threshold_step = threshold_step)
                elif body_position is None:
                    body_position = np.array([np.average([moments[i]['m01'] / moments[i]['m00'] for i in range(len(moments))]), np.average([moments[i]['m10'] / moments[i]['m00'] for i in range(len(moments))])])
            else:
                threshold_value += threshold_step
                body_position = extract_body_position_extended(frame = frame, eyes_threshold_value = eyes_threshold_value, body_threshold_value = body_threshold_value, threshold_value = threshold_value, threshold_step = threshold_step)
        else:
            body_position = None
    except:
        body_position = None
    return body_position

def show_image(image):
    print(np.shape(image), np.max(image), np.min(image))
    cv2.imshow("Image Preview", image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Background subtraction --- #

def subtract_background_from_frames(frames, background, bg_sub_threshold, dark_background=False):
    '''
    Subtract a background image from an array of frames.

    Arguments:
        frames (ndarray)       : (t, y, x) array of frames.
        background (ndarray)   : (y, x) background image array.
        bg_sub_threshold (int) : Threshold of the difference between a pixel and its background value that
                                 will cause to be considered a background pixel (and be set to white/black).
        dark_background (bool) : Whether the video has a dark background and light fish. If so, background pixels
                                 will be set to black rather than white.

    Returns:
        bg_sub_frames (ndarray) : The background-subtracted frames.
    '''

    # create a mask that is True wherever a pixel value is sufficiently close to the background pixel value
    background_mask = (frames - background < bg_sub_threshold) | (frames - background > 255 - bg_sub_threshold)

    # subtract the background from the frames
    bg_sub_frames = frames - background.astype(float)

    # add back the mean background intensity (so the background-subtracted
    # frames have similar brightness to the original frames)
    bg_sub_frames += np.mean(background)

    # set the brightness of the background appropriately
    if dark_background:
        bg_value = 0
    else:
        bg_value = 255
    bg_sub_frames[background_mask] = bg_value

    # make sure the background-subtracted frames are in the range [0, 255]
    bg_sub_frames[bg_sub_frames < 0] = 0
    bg_sub_frames[bg_sub_frames > 255] = 255

    return bg_sub_frames.astype(np.uint8)

# --- Tracking --- #

def open_and_track_video(video_path, background_path, params, tracking_dir, video_number=0, progress_signal=None):
    '''
    Open and perform tracking on the provided video.

    Arguments:
        video_path (str)          : Path to the video.
        params (dict)             : Dictionary of tracking parameters.
        tracking_dir (str)        : Directory in which to save tracking data.
        video_number (int)        : If tracking a batch of videos, which number this video is.
        progress_signal (QSignal) : Signal to use to update the GUI with tracking progress.
    '''

    # start a timer for recording how long tracking takes
    start_time = time.time()

    # extract parameters
    subtract_background = params['subtract_background']
    if params['backgrounds'] is not None:
        background = params['backgrounds'][video_number]
    elif background_path is not None:
        background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    else:
        background = None
    crop_params         = params['crop_params']
    n_tail_points       = params['n_tail_points']
    save_video          = params['save_video']
    tracking_video_fps  = params['tracking_video_fps']
    use_multiprocessing = params['use_multiprocessing']
    n_crops             = len(params['crop_params'])
    bg_sub_threshold    = params['bg_sub_threshold']
    tracking_type       = params['type']
    dark_background     = params['dark_background']

    # initialize a counter for the number of frames that have been tracked
    n_frames_tracked = 0

    if progress_signal:
        # send a progress update signal to the controller
        percent_complete = 0
        progress_signal.emit(video_number, percent_complete)

    # create a video capture object that we can re-use
    try:
        capture = cv2.VideoCapture(video_path)
    except:
        print("Error: Could not open video.")
        return

    # get video info
    # fps, n_frames_total = get_video_info(video_path)
    n_frames_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    print("Total number of frames to track: {}.".format(n_frames_total))

    if tracking_video_fps == 0:
        # set tracking video fps to be the same as the original video
        tracking_video_fps = fps

    if subtract_background and background is None:
        print("Calculating background...")

        # calculate the background
        if n_frames_total > 1000:
            frame_nums = utilities.split_evenly(n_frames_total, 1000)
        else:
            frame_nums = list(range(n_frames_total))
        # background = open_video(video_path, frame_nums, return_frames=False, calc_background=True, capture=capture, dark_background=dark_background)
        background = extract_background_extended(video_path, num_backgrounds = 10, threshold_value = 8, save_background = True)

    # initialize tracking data arrays
    tail_coords_array    = np.zeros((n_crops, n_frames_total, 2, n_tail_points)) + np.nan
    spline_coords_array  = np.zeros((n_crops, n_frames_total, 2, n_tail_points)) + np.nan
    heading_angle_array  = np.zeros((n_crops, n_frames_total, 1)) + np.nan
    body_position_array  = np.zeros((n_crops, n_frames_total, 2)) + np.nan
    eye_coords_array     = np.zeros((n_crops, n_frames_total, 2, 2)) + np.nan

    # set number of frames to load into memory at a time
    big_chunk_size = 500

    # split frame numbers into big chunks - we keep only one big chunk of frames in memory at a time
    big_split_frame_nums = utilities.split_list_into_chunks(range(n_frames_total), big_chunk_size)

    if use_multiprocessing:
        # create a pool of workers
        pool = multiprocessing.Pool(None)

    # create the directory for saving tracking data if it doesn't exist
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)

    for i in range(len(big_split_frame_nums)):

        print("Tracking frames {} to {}...".format(big_split_frame_nums[i][0], big_split_frame_nums[i][-1]))

        # get the frame numbers to process
        frame_nums = big_split_frame_nums[i]

        # boolean indicating whether to have the capture object seek to the starting frame
        # this only needs to be done at the beginning to seek to frame 0
        seek_to_starting_frame = i == 0

        print("Opening frames...")

        # load this big chunk of frames
        frames = open_video(video_path, frame_nums, capture=capture, seek_to_starting_frame=seek_to_starting_frame)

        if i == 0 and params['save_video']:
            # create the video writer, for saving a video with tracking overlaid
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            new_video_path = os.path.join(tracking_dir, "{}_tracked_video.avi".format(os.path.splitext(os.path.basename(video_path))[0]))
            writer = cv2.VideoWriter(new_video_path, 0, tracking_video_fps,
                (frames[0].shape[1], frames[0].shape[0]), True)

        print("Tracking frames...")

        if use_multiprocessing:
            # split the frames into small chunks - we let multiple processes deal with a chunk at a time
            small_chunk_size = 100
            split_frames = utilities.yield_chunks_from_array(frames, small_chunk_size)

            # get the pool of workers to track the chunks of frames
            result_list = []
            for result in pool.imap(partial(track_frames, params, background), split_frames, 20):
                result_list.append(result)

                # increase the number of tracked frames counter
                n_frames_tracked += small_chunk_size

                if progress_signal:
                    # send a progress update signal to the controller
                    percent_complete = 100.0*n_frames_tracked/n_frames_total
                    progress_signal.emit(video_number, percent_complete)

            # get the number of frame chunks that have been processed
            n_chunks = len(result_list)

            # add tracking results to tracking data arrays
            tail_coords_array[:, frame_nums, :, :]   = np.concatenate([result_list[i][0] for i in range(n_chunks)], axis=1)
            spline_coords_array[:, frame_nums, :, :] = np.concatenate([result_list[i][1] for i in range(n_chunks)], axis=1)
            heading_angle_array[:, frame_nums, :]    = np.concatenate([result_list[i][2] for i in range(n_chunks)], axis=1)
            body_position_array[:, frame_nums, :]    = np.concatenate([result_list[i][3] for i in range(n_chunks)], axis=1)
            eye_coords_array[:, frame_nums, :, :]    = np.concatenate([result_list[i][4] for i in range(n_chunks)], axis=1)
        else:
            # track the big chunk of frames and add results to tracking data arrays
            (tail_coords_small_array, spline_coords_small_array,
             heading_angle_small_array, body_position_small_array, eye_coords_small_array) = track_frames(params, background, frames, frame_nums)

            # increase the number of tracked frames counter
            n_frames_tracked += len(frame_nums)

            if progress_signal:
                # send a progress update signal to the controller
                percent_complete = 100.0*n_frames_tracked/n_frames_total
                progress_signal.emit(video_number, percent_complete)

            # add tracking results to tracking data arrays
            tail_coords_array[:, frame_nums, :, :]   = tail_coords_small_array
            spline_coords_array[:, frame_nums, :, :] = spline_coords_small_array
            heading_angle_array[:, frame_nums, :]    = heading_angle_small_array
            body_position_array[:, frame_nums, :]    = body_position_small_array
            eye_coords_array[:, frame_nums, :, :]    = eye_coords_small_array

        # convert tracking coordinates from cropped frame space to original frame space
        for k in range(n_crops):
            tail_coords_array[k]   = get_absolute_coords(tail_coords_array[k], params['crop_params'][k]['offset'])
            spline_coords_array[k] = get_absolute_coords(spline_coords_array[k], params['crop_params'][k]['offset'])
            body_position_array[k] = get_absolute_coords(body_position_array[k], params['crop_params'][k]['offset'])
            eye_coords_array[k]    = get_absolute_coords(eye_coords_array[k], params['crop_params'][k]['offset'])

        if params['save_video']:
            print("Adding frames to tracking video...")

            for k in range(len(frames)):
                # get the frame & the frame number
                frame     = frames[k]
                frame_num = frame_nums[k]

                # create a dictionary with the tracking results for this frame
                results = {'tail_coords'  : tail_coords_array[:, frame_num, :, :],
                           'spline_coords': spline_coords_array[:, frame_num, :, :],
                           'eye_coords'   : eye_coords_array[:, frame_num, :, :],
                           'heading_angle': heading_angle_array[:, frame_num, :],
                           'body_position': body_position_array[:, frame_num, :]}

                # overlay the tracking data onto the frame
                tracked_frame = add_tracking_to_frame(frame, results, n_crops=n_crops)

                # write to the new video file
                writer.write(tracked_frame)

    if params['save_video']:
        print("Video created: {}.".format(new_video_path))

        # release the video writer
        writer.release()

    # make a tracking params dictionary for this video
    tracking_params = params.copy()
    tracking_params['video_num'] = video_number

    if tracking_type == "freeswimming":
        # set tracking variables to None if they weren't used
        if not params['track_eyes']:
            eye_coords_array = None
        if not params['track_tail']:
            tail_coords_array   = None
            spline_coords_array = None

        # calculate the tail angles (in degrees)
        tail_angle_array = analysis.calculate_freeswimming_tail_angles(heading_angle_array, body_position_array, tail_coords_array)

        # save tail angles, body position & heading angle as CSV files
        # for tail angles, rows are video frames, columns points along the tail
        # for body position, rows are video frames, columns are x & y coordinates
        # for heading angle, rows are video frames
        if n_crops > 1:
            for k in range(n_crops):
                np.savetxt(os.path.join(tracking_dir, "{}_tail_angles_crop_{}.csv".format(os.path.splitext(os.path.basename(video_path))[0], k)), tail_angle_array[k], fmt="%.4f", delimiter=",")
                np.savetxt(os.path.join(tracking_dir, "{}_body_position_crop_{}.csv".format(os.path.splitext(os.path.basename(video_path))[0], k)), body_position_array[k], fmt="%.4f", delimiter=",")
                np.savetxt(os.path.join(tracking_dir, "{}_heading_angle_crop_{}.csv".format(os.path.splitext(os.path.basename(video_path))[0], k)), heading_angle_array[k], fmt="%.4f", delimiter=",")
        else:
            np.savetxt(os.path.join(tracking_dir, "{}_tail_angles.csv".format(os.path.splitext(os.path.basename(video_path))[0])), tail_angle_array[0], fmt="%.4f", delimiter=",")
            np.savetxt(os.path.join(tracking_dir, "{}_body_position.csv".format(os.path.splitext(os.path.basename(video_path))[0])), body_position_array[0], fmt="%.4f", delimiter=",")
            np.savetxt(os.path.join(tracking_dir, "{}_heading_angle.csv".format(os.path.splitext(os.path.basename(video_path))[0])), heading_angle_array[0], fmt="%.4f", delimiter=",")
    else:
        # set tracking variables to None if they weren't used
        eye_coords_array    = None
        body_position_array = None

        # calculate the tail angles (in degrees)
        tail_angle_array = analysis.calculate_headfixed_tail_angles(params['heading_angle'], tail_coords_array)

        # save tail angles as CSV files -- rows are points along the tail, columns are video frames
        if n_crops > 1:
            for k in range(n_crops):
                np.savetxt(os.path.join(tracking_dir, "{}_tail_angles_crop_{}.csv".format(os.path.splitext(os.path.basename(video_path))[0], k)), tail_angle_array[k], fmt="%.4f", delimiter=",")
        else:
            np.savetxt(os.path.join(tracking_dir, "{}_tail_angles.csv".format(os.path.splitext(os.path.basename(video_path))[0])), tail_angle_array[0], fmt="%.4f", delimiter=",")

    # save the tracking data
    np.savez(os.path.join(tracking_dir, "{}_tracking.npz".format(os.path.splitext(os.path.basename(video_path))[0])),
                          tail_coords=tail_coords_array, spline_coords=spline_coords_array,
                          heading_angle=heading_angle_array, body_position=body_position_array,
                          eye_coords=eye_coords_array, params=tracking_params)

    if use_multiprocessing:
        # close the pool of workers
        pool.close()
        pool.join()

    # close the video capture object
    capture.release()

    # stop the timer
    end_time = time.time()

    # print the total tracking time
    print("Finished tracking. Total time: {}s.".format(end_time - start_time))

def open_and_track_video_batch(params, tracking_dir, progress_signal=None):
    '''
    Open and perform tracking on a batch of videos.

    Arguments:
        params (dict)             : Dictionary of tracking parameters, including the video paths.
        tracking_dir (str)        : Directory in which to save tracking data.
        progress_signal (QSignal) : Signal to use to update the GUI with tracking progress.
    '''

    # extract video paths
    video_paths = params['video_paths']

    # track each video with the same parameters
    for i in range(len(video_paths)):
        open_and_track_video(video_paths[i], params, tracking_dir, i, progress_signal)

def track_frames(params, background, frames, frame_nums):
    '''
    Perform tracking on the provided frames.

    Arguments:
        params (dict)             : Dictionary of tracking parameters.
        background (ndarray/None) : Background to subtract.
        frames (ndarray)          : Frames to perform tracking on.

    Returns:
        tail_coords_array (ndarray)   : Array containing coordinates of points along the tail.
                                        Dimensions are (# of crops, # of frames, 2, # tail points).
        spline_coords_array (ndarray) : Array containing coordinates of points along a spline fitted to the tail.
                                        Dimensions are (# of crops, # of frames, 2, # tail points).
        heading_angle_array (ndarray) : Array containing the heading angle of the zebrafish.
                                        Dimensions are (# of crops, # of frames, 1).
        body_position_array (ndarray) : Array containing the body center coordinates of the zebrafish.
                                        Dimensions are (# of crops, # of frames, 2).
        eye_position_array (ndarray)  : Array containing the coordinates of the eyes of the zebrafish.
                                        Dimensions are (# of crops, # of frames, 2, 2).
    '''

    # extract parameters
    crop_params         = params['crop_params']
    tracking_type       = params['type']
    n_tail_points       = params['n_tail_points']
    subtract_background = params['subtract_background']
    bg_sub_threshold    = params['bg_sub_threshold']
    dark_background     = params['dark_background']

    # get number of frames & number of crops
    n_frames = frames.shape[0]
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

    if subtract_background and background is not None:
        # subtract the background
        original_frames = frames.copy()
        frames          = subtract_background_from_frames(frames, background, bg_sub_threshold, dark_background=dark_background)
        # frames = subtract_background_from_frames_extended(frames, background, threshold_value = 2)

    else:
        original_frames = frames

    for frame_number in range(n_frames):
        # get the frame
        frame = frames[frame_number]

        for k in range(n_crops):
            # get the crop & offset
            crop   = crop_params[k]['crop']
            offset = crop_params[k]['offset']

            # crop the frame
            cropped_frame = crop_frame(frame, offset, crop)

            # track the frame
            results, _, _ = track_cropped_frame(cropped_frame, frame_nums[frame_number], params, crop_params[k], original_frame=original_frames[frame_number])

            # add coordinates to tracking data arrays
            if results['tail_coords'] is not None:
                tail_coords_array[k, frame_number, :, :results['tail_coords'].shape[1]]     = results['tail_coords']
                spline_coords_array[k, frame_number, :, :results['spline_coords'].shape[1]] = results['spline_coords']
            heading_angle_array[k, frame_number, :] = results['heading_angle']
            body_position_array[k, frame_number, :] = results['body_position']
            eye_coords_array[k, frame_number, :, :] = results['eye_coords']

    return tail_coords_array, spline_coords_array, heading_angle_array, body_position_array, eye_coords_array

def track_cropped_frame(frame, frame_num, params, crop_params, original_frame=None):
    '''
    Perform tracking on the provided frame.

    Arguments:
        frame (ndarray)    : Frame to perform tracking on.
        params (dict)      : Dictionary of tracking parameters.
        crop_params (dict) : Dictionary of extra tracking parameters for the cropped frame.

    Returns:
        results (dict)           : Dictionary containing tracking results..
        skeleton_frame (ndarray) : The result of skeletonizing the thresholded frame (used by the
                                   GUI to preview the skeleton frame).
        body_crop_coords (list)  : List of coordinates of the crop around the tracked body position
                                   that is used to track the eyes and tail (for freeswimming fish).
    '''
    if original_frame is None:
        original_frame = frame

    # extract tracking type
    tracking_type  = params['type']

    if tracking_type == "freeswimming":
        # extract parameters
        body_crop       = params['body_crop']
        track_tail_bool = params['track_tail']
        track_eyes_bool = params['track_eyes']

        # only crop around the body if we're tracking the eyes and/or tail
        crop_around_body = (track_eyes_bool or track_tail_bool) and body_crop is not None

        # track the heading angle and body position
        if crop_around_body:
            heading_angle, body_position, rel_body_position, body_crop_coords, body_crop_frame, body_threshold_frame = track_body(frame, frame_num, params, crop_params, crop_around_body=True)

            _, body_crop_original_frame = crop_frame_around_body(original_frame, body_position, body_crop)
        else:
            heading_angle, body_position = track_body(frame, frame_num, params, crop_params, crop_around_body=False)
            rel_body_position            = body_position
            body_crop_coords             = None
            body_crop_frame              = frame
            body_crop_original_frame     = original_frame

        if track_eyes_bool:
            # track the eyes
            eye_coords = track_eyes(body_crop_frame, frame_num, params, crop_params)

            if eye_coords is not None:
                # update the heading angle based on the found eye coordinates
                heading_angle = update_heading_angle_from_eye_coords(eye_coords, heading_angle, body_position)

                if crop_around_body and body_crop_coords is not None and np.sum(body_crop_frame) > 0:
                    # convert the eye coords to be relative to the initial frame
                    eye_coords += body_crop_coords[:, 0][:, np.newaxis].astype(int)
        else:
            eye_coords = None

        if track_tail_bool and body_position is not None:
            # track the tail only if the body center was found
            tail_coords, spline_coords, skeleton_frame = track_freeswimming_tail(body_crop_frame, frame_num, body_threshold_frame, params, crop_params, rel_body_position, heading_angle, original_frame=body_crop_original_frame)
            if tail_coords is not None:
                if crop_around_body and body_crop_coords is not None and np.sum(body_crop_frame) > 0:
                    # convert the tail coords to be relative to the initial frame
                    tail_coords   += body_crop_coords[:, 0][:, np.newaxis].astype(int)
                    spline_coords += body_crop_coords[:, 0][:, np.newaxis].astype(int)
        else:
            tail_coords, spline_coords, skeleton_frame = [None]*3
    elif tracking_type == "headfixed":
        # set body, heading and eye position variables to None since we aren't interested in them
        heading_angle, body_position, eye_coords = [None]*3

        # track the tail
        tail_coords, spline_coords = track_headfixed_tail(frame, params, crop_params)
        skeleton_frame   = None
        body_crop_coords = None

    # create a dictionary of results
    results = { 'tail_coords'    : tail_coords,
                'spline_coords'  : spline_coords,
                'heading_angle'  : heading_angle,
                'body_position'  : body_position,
                'eye_coords'     : eye_coords }

    return results, skeleton_frame, body_crop_coords

# --- Body position & heading angle tracking --- #

def track_body(frame, frame_num, params, crop_params, crop_around_body=True):
    # extract parameters
    adjust_thresholds = params['adjust_thresholds']
    body_threshold    = crop_params['body_threshold']
    eyes_threshold    = crop_params['eyes_threshold']
    body_crop         = params['body_crop']

    # create body threshold frame
    body_threshold_frame = get_threshold_frame(frame, body_threshold, min_threshold=None, dilate=False)
    # body_threshold_frame = get_body_threshold_frame(frame, body_threshold, kernel_size = [5, 5], n_iterations = 3)

    # get heading angle & body position
    heading_angle, body_position = get_heading_angle_and_body_position(body_threshold_frame, frame, eyes_threshold, body_threshold)
    if crop_around_body:
        # create array of body crop coordinates:
        # [ y_start  y_end ]
        # [ x_start  x_end ]

        # crop the frame around the body
        body_crop_coords, body_crop_frame = crop_frame_around_body(frame, body_position, body_crop)

        if body_position is None:
            rel_body_position = None
        else:
            # get body center position relative to the body crop
            rel_body_position = body_position - body_crop_coords[:, 0]

        return heading_angle, body_position, rel_body_position, body_crop_coords, body_crop_frame, body_threshold_frame

    return heading_angle, body_position

def get_heading_angle_and_body_position(body_threshold_frame, frame, eyes_threshold, body_threshold):
    # find contours in the thresholded frame

    try:
        image, contours, _ = cv2.findContours(body_threshold_frame.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, _ = cv2.findContours(body_threshold_frame.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    try:
        if len(contours) > 0:
            # choose the contour with the largest area as the body
            body_contour = max(contours, key=cv2.contourArea)

            if len(body_contour) >= 10:
                # fit an ellipse and get the angle and center position
                (x, y), (MA, ma), angle = cv2.fitEllipse(body_contour)

                height      = MA
                half_width  = ma

                rad_angle = angle*np.pi/180.0

                # create rotated rectangle mask from the center of the ellipse to one end of the major axis
                # this rectangle covers the half of the ellipse that is in the direction of the heading angle
                mask_1 = np.zeros(body_threshold_frame.shape)
                point_1 = (x + half_width*np.cos(rad_angle), y + half_width*np.sin(rad_angle))
                point_2 = (point_1[0] - height*np.sin(rad_angle), point_1[1] + height*np.cos(rad_angle))

                point_3 = (x - half_width*np.cos(rad_angle), y - half_width*np.sin(rad_angle))
                point_4 = (point_3[0] - height*np.sin(rad_angle), point_3[1] + height*np.cos(rad_angle))

                cv2.fillConvexPoly(mask_1, np.array([point_1, point_2, point_4, point_3]).astype(int), 1)

                # create rotated rectangle mask from the center of the ellipse to the other end of the major axis
                # this rectangle covers the half of the ellipse that is in the opposite direction of the heading angle
                mask_2 = np.zeros(body_threshold_frame.shape)
                point_1 = (x + half_width*np.cos(rad_angle + np.pi), y + half_width*np.sin(rad_angle + np.pi))
                point_2 = (point_1[0] - height*np.sin(rad_angle + np.pi), point_1[1] + height*np.cos(rad_angle + np.pi))

                point_3 = (x - half_width*np.cos(rad_angle + np.pi), y - half_width*np.sin(rad_angle + np.pi))
                point_4 = (point_3[0] - height*np.sin(rad_angle + np.pi), point_3[1] + height*np.cos(rad_angle + np.pi))

                cv2.fillConvexPoly(mask_2, np.array([point_1, point_2, point_4, point_3]).astype(int), 1)

                # if the average brightness of the masked frame in the direction of the the heading angle is larger than
                # that opposite of the heading angle (ie. the heading angle points toward the tail), flip it
                if np.mean(frame[mask_1.astype(bool)]) > np.mean(frame[mask_2.astype(bool)]):
                    angle += 180

                # create an array for the center position
                position = extract_body_position_extended(frame, eyes_threshold, body_threshold, threshold_step = 1)

                if position is None:
                    position = np.array([y, x])

                if position[0] < 0 or position[1] < 0 or 4*MA*ma < 100:
                    # discard results if they're erroneuous and if the body area is too small
                    return [None]*2
            else:
                return [None]*2
        else:
            return [None]*2

        return angle, position
    except:
        return [None]*2

def update_heading_angle_from_eye_coords(eye_coords, body_heading_angle, body_position):
    # get the heading angle based on eye coordinates
    angle = 180.0 + np.arctan((eye_coords[0, 1] - eye_coords[0, 0])/(eye_coords[1, 1] - eye_coords[1, 0]))*180.0/np.pi

    if body_heading_angle is not None:
        # if the angle is too different from the heading angle found by fitting an ellipse to the body,
        # try flipping it
        if angle - body_heading_angle > 90:
            angle -= 180
        elif angle - body_heading_angle < -90:
            angle += 180

        # if it's still not within 90 degrees, just set it to the body threshold heading angle
        if np.abs(angle - body_heading_angle) > 90:
            angle = body_heading_angle
        else:
            # otherwise set the final angle to be a mix of this angle & the body threshold heading angle
            angle = 0.7*body_heading_angle + 0.3*angle

    return angle

def track_eyes(frame, frame_num, params, crop_params):
    # extract parameters
    adjust_thresholds = params['adjust_thresholds']
    eyes_threshold    = crop_params['eyes_threshold']

    # threshold the frame to extract the eyes
    eyes_threshold_frame  = get_threshold_frame(frame, eyes_threshold)

    # get eye positions
    eye_positions = get_eye_positions(eyes_threshold_frame)

    if eye_positions is None and adjust_thresholds: # eyes not found; adjust the threshold & try again
        # initialize counter
        i = 0

        # create a list of head thresholds to go through
        eyes_thresholds = list(range(eyes_threshold-1, eyes_threshold-5, -1)) + list(range(eyes_threshold+1, eyes_threshold+5))

        while eye_positions is None and i < 8:
            # create a thresholded frame using new threshold
            eyes_threshold_frame = get_threshold_frame(frame, eyes_thresholds[i])

            # get eye positions
            eye_positions = get_eye_positions(eyes_threshold_frame)

            # increase counter
            i += 1

    return eye_positions

def get_eye_positions(eyes_threshold_image, prev_eye_coords=None):
    # find contours
    try:
        image, contours, _ = cv2.findContours(eyes_threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, _ = cv2.findContours(eyes_threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) < 2:
        # too few contours found -- we need at least 2 (one for each eye)
        return None

    # choose the two contours with the largest areas as the eyes
    eye_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # stop if the largest area is too small
    if cv2.contourArea(eye_contours[0]) < 2:
        return None

    # get moments
    moments = [cv2.moments(contour) for contour in eye_contours]

    # initialize array to hold eye positions
    positions = np.zeros((2, 2))

    # get coordinates
    for i in range(2):
        M = moments[i]
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            positions[0, i] = cy
            positions[1, i] = cx
        else:
            positions = None

    return positions

# --- Freeswimming tail tracking --- #

def track_freeswimming_tail(frame, frame_num, body_threshold_frame, params, crop_params, body_position, heading_angle, original_frame=None):
    if original_frame is None:
        original_frame = frame

    # extract parameters
    adjust_thresholds  = params['adjust_thresholds']
    min_tail_body_dist = params['min_tail_body_dist']
    max_tail_body_dist = params['max_tail_body_dist']
    n_tail_points      = params['n_tail_points']
    alt_tail_tracking  = params['alt_tail_tracking']
    tail_threshold     = crop_params['tail_threshold']

    # threshold the frame to extract the tail
    tail_threshold_frame = get_threshold_frame(frame, tail_threshold, remove_noise=False)
    # tail_threshold_frame = get_tail_threshold_frame(frame, tail_threshold, kernel_size = [5, 5], n_iterations = 2)

    if alt_tail_tracking:
        tail_coords, spline_coords, skeleton_image = track_freeswimming_tail_alt(frame, frame_num, tail_threshold_frame, body_threshold_frame, params, crop_params, body_position, heading_angle)
    else:
        # get tail coordinates
        tail_coords, spline_coords, skeleton_image = get_freeswimming_tail_coords(tail_threshold_frame, body_position, heading_angle,
                                                                  min_tail_body_dist, max_tail_body_dist,
                                                                  n_tail_points, alt_tracking=alt_tail_tracking)
        if tail_coords is None:
            # try increasing the minimum tail-body distance
            i = 1

            while i <= 5:
                tail_coords, spline_coords, skeleton_image = get_freeswimming_tail_coords(tail_threshold_frame, body_position, heading_angle,
                                                                  min_tail_body_dist+i, max_tail_body_dist,
                                                                  n_tail_points, alt_tracking=alt_tail_tracking)

                i += 1


        if adjust_thresholds and tail_coords is None:
            # initialize counter
            i = 0

            # create a list of tail thresholds to go through
            tail_thresholds = list(range(tail_threshold-1, tail_threshold-5, -1)) + list(range(tail_threshold+1, tail_threshold+5))

            while tail_coords is None and i < 8:
                #  create a thresholded frame using new threshold
                tail_threshold_frame = get_threshold_frame(frame, tail_thresholds[i], remove_noise=False)

                # get tail coordinates
                tail_coords, spline_coords, skeleton_image = get_freeswimming_tail_coords(tail_threshold_frame, body_position, heading_angle,
                                                                          min_tail_body_dist, max_tail_body_dist,
                                                                          n_tail_points, alt_tracking=alt_tail_tracking)

                # increase counter
                i += 1

    return tail_coords, spline_coords, skeleton_image

def track_freeswimming_tail_alt(frame, frame_num, tail_threshold_frame, body_threshold_frame, params, crop_params, body_position, heading_angle):
    tail_threshold     = crop_params['tail_threshold']
    n_tail_points      = params['n_tail_points']
    radius             = params['radius']
    max_tail_value     = params['max_tail_value']
    angle_range        = params['angle_range']

    # get tail skeleton image
    # skeleton_image = get_tail_skeleton_frame(tail_threshold_frame)
    skeleton_image = get_tail_thinned_frame(tail_threshold_frame)

    start_coord = np.array(body_position)

    prev_coord = start_coord
    angle = 180 - heading_angle
    tail_coords = []
    angles = []
    i = 0
    while prev_coord is not None and i < 200:
        next_coord = find_next_coord(frame, prev_coord, angle, radius=radius, max_value=max_tail_value, angle_range=angle_range)
        if next_coord is None:
            r = radius-1
            while next_coord is None and r > 1:
                next_coord = find_next_coord(frame, prev_coord, angle, radius=r, max_value=max_tail_value, angle_range=angle_range)
                r -= 1
        if next_coord is not None:
            angle = np.arctan2(np.array([next_coord[1] - prev_coord[1]]), np.array([next_coord[0] - prev_coord[0]]))[0]*180.0/np.pi
            tail_coords.append(next_coord)
            angles.append(angle)
        prev_coord = next_coord
        i += 1

    if len(tail_coords) > 0:
        tail_coords = np.array(tail_coords).T
        n_tail_coords = tail_coords.shape[1]
    else:
        print("Error: Could not calculate tail spline.")
        # print("Frame number: {0}".format(frame_num))
        return [None]*2 + [skeleton_image]

    if n_tail_coords > n_tail_points:
        # get evenly-spaced tail indices
        tail_nums = np.linspace(0, tail_coords.shape[1]-1, n_tail_points).astype(int)

        # pick evenly-spaced points along the tail
        tail_coords = tail_coords[:, tail_nums]

    n_tail_coords = tail_coords.shape[1]

    try:
        # make ascending spiral in 3D space
        t = np.zeros(n_tail_coords)
        t[1:] = np.sqrt((tail_coords[1, 1:] - tail_coords[1, :-1])**2 + (tail_coords[0, 1:] - tail_coords[0, :-1])**2)
        t = np.cumsum(t)
        t /= t[-1]

        nt = np.linspace(0, 1, 100)

        # calculate cubic spline
        spline_y_coords = interpolate.UnivariateSpline(t, tail_coords[0, :], k=3, s=3)(nt)
        spline_x_coords = interpolate.UnivariateSpline(t, tail_coords[1, :], k=3, s=3)(nt)

        spline_coords = np.array([spline_y_coords, spline_x_coords])

        # get evenly-spaced spline indices
        spline_nums = np.linspace(0, spline_coords.shape[1]-1, n_tail_points).astype(int)

        # pick evenly-spaced points along the spline
        spline_coords = spline_coords[:, spline_nums]
    except:
        print("Error: Could not calculate tail spline.")
        return [None]*2 + [skeleton_image]

    return tail_coords, spline_coords, skeleton_image

def find_next_coord(frame, start_coord, angle, radius=5, max_value=50, angle_range=120):
    angles = np.linspace(angle-angle_range/2, angle+angle_range/2, 100)*np.pi/180.0
    x_coords = np.maximum(0, np.minimum(np.round((radius*np.sin(angles) + start_coord[1])).astype(int), frame.shape[1]-1))
    y_coords = np.maximum(0, np.minimum(np.round((radius*np.cos(angles) + start_coord[0])).astype(int), frame.shape[0]-1))
    coords = np.unique(np.vstack((y_coords, x_coords)).T, axis=0)
    mask = sparse.coo_matrix((np.ones(coords.shape[0]), list(zip(*coords)) ), shape=frame.shape, dtype=bool).toarray()
    masked_frame = np.ones(frame.shape).astype(np.uint8)*255
    masked_frame[mask] = frame[mask]
    min_value = np.amin(masked_frame)
    if min_value > max_value:
        return None
    next_coord = np.unravel_index(masked_frame.argmin(), frame.shape)
    return next_coord

def get_freeswimming_tail_coords(tail_threshold_frame, body_position, heading_angle, min_tail_body_dist, max_tail_body_dist, n_tail_points, max_r=2, smoothing_factor=3, alt_tracking=False): # todo: make max radius & smoothing factor user settable
    # get tail skeleton image
    skeleton_image = get_tail_skeleton_frame(tail_threshold_frame)

    # zero out pixels that are close to the body
    skeleton_image = cv2.circle(skeleton_image, (int(round(body_position[1])), int(round(body_position[0]))), int(min_tail_body_dist), 0, -1).astype(np.uint8)

    # zero out a rectangle rotated in the heading direction of the fish
    # this essentially zeroes out the head of the fish
    height = 20
    width  = 20

    point_1 = (body_position[1] + (width/2)*np.cos(heading_angle*np.pi/180.0), body_position[0] + (width/2)*np.sin(heading_angle*np.pi/180.0))
    point_2 = (point_1[0] - height*np.sin(heading_angle*np.pi/180.0), point_1[1] + height*np.cos(heading_angle*np.pi/180.0))

    point_3 = (body_position[1] - (width/2)*np.cos(heading_angle*np.pi/180.0), body_position[0] - (width/2)*np.sin(heading_angle*np.pi/180.0))
    point_4 = (point_3[0] - height*np.sin(heading_angle*np.pi/180.0), point_3[1] + height*np.cos(heading_angle*np.pi/180.0))

    points = np.array([point_1, point_2, point_4, point_3, point_1]).astype(int)

    cv2.fillConvexPoly(skeleton_image, points, 0)

    if alt_tracking:
        # get an ordered list of coordinates of the tail, from one end to the other
        tail_coords = get_ordered_tail_coords(skeleton_image, max_r, body_position, min_tail_body_dist, max_tail_body_dist)
    else:
        # find contours in the skeleton image
        try:
            image, contours, _ = cv2.findContours(skeleton_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        except ValueError:
            contours, _ = cv2.findContours(skeleton_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            # choose the contour with the most points as the tail contour
            tail_contour = max(contours, key=len)

            # create initial array of tail coordinates
            tail_coords = np.array([ (i[0][1], i[0][0]) for i in tail_contour ]).T

            # pick point closest to the body as the starting point
            startpoint_index = np.argmin((np.sum((tail_coords - body_position[:, np.newaxis])**2, axis=0)))

            # make the starting point be at index 0
            if startpoint_index != 0:
                tail_coords = np.roll(tail_coords, -startpoint_index, axis=1)

            # find endpoint farthest away from the starting point
            min_diff = 10000
            endpoint_index = None

            for i in range(1, tail_coords.shape[1]-1):
                # only consider endpoints -- the point before an endpoint
                # is the same as the point after it
                if tail_coords[:, i-1] is tail_coords[:, i+1]:
                    dist_1 = i
                    dist_2 = tail_coords.shape[1]-i
                    diff = abs(dist_2 - dist_1)
                    if diff < min_diff:
                        min_diff = diff
                        endpoint_index = i

            # only keep tail points up to the farthest endpoint
            tail_coords = tail_coords[:, :endpoint_index]
        else:
            tail_coords = None

    if tail_coords is None:
        # couldn't get tail coordinates; end here.
        return [None]*2 + [skeleton_image]

    # get number of found tail coordinates
    n_tail_coords = tail_coords.shape[1]

    # get size of thresholded image
    y_size = tail_threshold_frame.shape[0]
    x_size = tail_threshold_frame.shape[1]

    if tail_coords is not None:
        # convert tail coordinates to floats
        tail_coords = tail_coords.astype(float)

    if n_tail_coords > n_tail_points:
        # get evenly-spaced tail indices
        tail_nums = np.linspace(0, tail_coords.shape[1]-1, n_tail_points).astype(int)

        # pick evenly-spaced points along the tail
        tail_coords = tail_coords[:, tail_nums]

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

        # get evenly-spaced spline indices
        spline_nums = np.linspace(0, spline_coords.shape[1]-1, n_tail_points).astype(int)

        # pick evenly-spaced points along the spline
        spline_coords = spline_coords[:, spline_nums]
    except:
        print("Error: Could not calculate tail spline.")
        return [None]*2 + [skeleton_image]

    return tail_coords, spline_coords, skeleton_image

def get_ordered_tail_coords(skeleton_image, max_r, body_position, min_tail_body_dist, max_tail_body_dist, min_n_tail_points=10):
    # find contours in the skeleton image
    try:
        image, contours, _ = cv2.findContours(skeleton_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, _ = cv2.findContours(skeleton_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        # choose the contour with the most points as the tail contour
        tail_contour = max(contours, key=len)
    else:
        return None

    # create initial array of tail coordinates
    tail_coords = np.array([ (i[0][1], i[0][0]) for i in tail_contour ]).T

    # pick point closest to the body as the starting point
    startpoint_index = np.argmin((np.sum((tail_coords - body_position[:, np.newaxis])**2, axis=0)))
    tail_start_coords = tail_coords[:, startpoint_index]

    if tail_start_coords is None:
        print("Couldn't find start of the tail.")
        # still could not find start of the tail; end here.
        return None

    # walk along the tail, finding coordinates
    found_coords = walk_along_tail(tail_start_coords, max_r, skeleton_image)

    if len(found_coords) < min_n_tail_points:
        # we didn't get enough tail points; give up here.
        return None

    # convert to an array
    found_coords = np.array(found_coords).T

    return found_coords

def walk_along_tail(tail_start_coords, max_r, skeleton_image):
    # create a list of tuples of tail coordinates, initially only containing the starting coordinates
    found_coords = [tuple(tail_start_coords)]

    # initialize radius (half the side length of square area to look in for the next point)
    r = 1

    # set maximum tail points to avoid crazy things happening
    max_tail_points = 200

    while len(found_coords) < max_tail_points and r <= max_r:
        # find coordinates of the next point
        next_coords = find_next_tail_coords_in_neighborhood(found_coords, r, skeleton_image)

        if next_coords is not None:
            # add coords to found coords list
            found_coords.append(tuple(next_coords))
        else:
            r += 1

    return found_coords

def find_next_tail_coords_in_neighborhood(found_coords, r, skeleton_image):
    # pad the skeleton image with zeros
    padded_matrix = np.zeros((skeleton_image.shape[0] + 2*r, skeleton_image.shape[1] + 2*r))
    padded_matrix[r:-r, r:-r] = skeleton_image

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

    if unique_coords is None or len(unique_coords) == 0:
        # all of the nonzero points have been traversed already; end here
        return None

    if r > 1:
        # get the angles between a vector from the starting point to the last found point, and vectors from the starting point to each of the potential new points
        angles = [ max(0, angle_between(np.array(found_coords[-1]) - np.array(found_coords[0]), np.array(unique_coords[i]) - np.array(found_coords[0]))) for i in xrange(len(unique_coords)) ]

        # if the angles are all too large, give up
        # if we were properly traversing the tail, the vector pointing to the previous
        # coordinate and the one pointing to the next coordinate should be somewhat close to each other
        if np.amin(angles) > 1.5:
            return None

        # set the next coordinate to be the one for which the angle is smallest
        closest_index = np.argmin(angles)
        next_coords = unique_coords[closest_index]
    elif len(unique_coords) > 1:
        # same as above, but don't specify a minimum angle

        angles = [ np.abs(angle_between(np.array(found_coords[-1]) - np.array(found_coords[0]), np.array(unique_coords[i]) - np.array(found_coords[0]))) for i in xrange(len(unique_coords)) ]

        closest_index = np.argmin(angles)
        next_coords = unique_coords[closest_index]
    else:
        # only one potential point was found; pick that one
        next_coords = unique_coords[0]

    return next_coords

def find_unique_coords(coords, found_coords):
    return [o for o in coords if o not in set(found_coords)]

# --- Headfixed tail tracking --- #

def track_headfixed_tail(frame, params, crop_params, smoothing_factor=30, heading_direction=None): # todo: make smoothing factor user settable
    # extract parameters
    tail_start_coords = get_relative_coords(params['tail_start_coords'], crop_params['offset'])
    n_tail_points     = params['n_tail_points']
    heading_angle     = params['heading_angle']

    # shift heading angle so that 0 degrees is down
    heading_angle += 90

    global fitted_tail, tail_funcs, tail_brightness, background_brightness, tail_length

    # check whether we are processing the first frame
    first_frame = tail_funcs is None

    # convert tail direction to a vector
    tail_directions = { "Down": [0,-1], "Up": [0,1], "Right": [-1,0], "Left": [1,0] }

    # set maximum tail points to avoid crazy things happening
    max_tail_points = 200

    # initialize tail fitted coords array for this frame
    frame_fit = np.zeros((max_tail_points, 2))

    # initialize lists of variables
    widths, convolution_results = [],[]
    test, slices                = [], []

    # pick an initial guess for the direction vector
    if heading_angle is not None:
        rad_heading_angle = heading_angle*np.pi/180.0
        guess_vector = np.array([-np.sin(rad_heading_angle), -np.cos(rad_heading_angle)])
    else:
        guess_vector = np.array(tail_directions[heading_direction])

    # set an approximate width of the tail (px)
    guess_tail_width = 50

    # flip x, y in tail start coords
    tail_start_coords = [tail_start_coords[1], tail_start_coords[0]]

    if first_frame:
        # set current point
        current_point = np.array(tail_start_coords).astype(int)

        # get histogram of pixel brightness for the frame
        if frame.ndim == 2:
            histogram = np.histogram(frame[:, :], 10, (0, 255))
        elif frame.ndim == 3:
            histogram = np.histogram(frame[:, :, 0], 10, (0, 255))

        # get average background brightness
        background_brightness = histogram[1][histogram[0].argmax()]/2 + histogram[1][min(histogram[0].argmax()+1, len(histogram[0]))]/2
        # print(type(frame), type(current_point), current_point)
        # get average tail brightness from a 2x2 area around the current point
        if frame.ndim == 2:
            tail_brightness = frame[current_point[1]-2:current_point[1]+3, current_point[0]-2:current_point[0]+3].mean()
        elif frame.ndim == 3:
            tail_brightness = frame[current_point[1]-2:current_point[1]+3, current_point[0]-2:current_point[0]+3, 0].mean()

        # create a Gaussian pdf (we will use this to find the midline of the tail)
        normpdf = pylab.normpdf(np.arange(-guess_tail_width/4.0, (guess_tail_width/4.0)+1), 0, 8)
    else:
        # set current point to the first point that was found in the last tracked frame
        # current_point = fitted_tail[-1][0, :]
        current_point = np.array(tail_start_coords).astype(int)

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
            result_index = int(round(results.argmax() - results.size/2 + guess_slice.size/2))

            # get point that corresponds to this index
            new_point = np.array([x_indices[result_index], y_indices[result_index]])
        else:
            # convolve the tail slice with the tail profile
            results = np.convolve(tail_funcs[count], guess_slice, "valid")

            # get the index of the point with max brightness, and adjust to match the size of the tail slice
            result_index = int(results.argmax() - results.size/2 + guess_slice.size/2)

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

        if len(swidths) > 0:
            swidths = np.lib.pad(swidths, [0, 5], mode='edge')
        else:
            return [None]*2

        # compute functions that profile the tail
        tail_funcs = [ calculate_tail_func(np.arange(-guess_tail_width, guess_tail_width), 0, swidth, 1, 0) for swidth in swidths]

    # append fitted tail to list of tail fits
    fitted_tail.append(np.copy(frame_fit[:count]))

    # get tail coordinates
    tail_coords = np.fliplr(frame_fit[:count]).T

    # get number of tail coordinates
    n_tail_coords = tail_coords.shape[1]

    if n_tail_coords > n_tail_points:
        # get evenly spaced tail indices
        tail_nums = np.linspace(0, tail_coords.shape[1]-1, n_tail_points).astype(int)

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
        spline_nums = np.linspace(0, spline_coords.shape[1]-1, n_tail_points).astype(int)

        spline_coords = spline_coords[:, spline_nums]

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

def crop_frame_around_body(frame, body_position, body_crop):
    # crop the frame around the body position

    if body_crop is None or body_position is None:
        body_crop_coords = np.array([[0, frame.shape[0]], [0, frame.shape[1]]])
    else:
        body_crop_coords = np.array([[np.maximum(0, int((body_position[0]-body_crop[0]))), np.minimum(frame.shape[0], int((body_position[0]+body_crop[0])))],
                                     [np.maximum(0, int((body_position[1]-body_crop[1]))), np.minimum(frame.shape[1], int((body_position[1]+body_crop[1])))]])

    if len(frame.shape) == 3:
        body_crop_frame = frame[body_crop_coords[0, 0]:body_crop_coords[0, 1], body_crop_coords[1, 0]:body_crop_coords[1, 1], :].copy()
    else:
        body_crop_frame = frame[body_crop_coords[0, 0]:body_crop_coords[0, 1], body_crop_coords[1, 0]:body_crop_coords[1, 1]].copy()

    return body_crop_coords, body_crop_frame

def get_video_info(video_path):
    # get video info
    fps      = ffmpeg_parse_infos(video_path)["video_fps"]
    n_frames = ffmpeg_parse_infos(video_path)["video_nframes"]

    return fps, n_frames

def add_tracking_to_frame(frame, tracking_results, cropped=False, n_crops=1):
    # extract parameters
    tail_coords    = tracking_results['tail_coords']
    spline_coords  = tracking_results['spline_coords']
    heading_angle  = tracking_results['heading_angle']
    body_position  = tracking_results['body_position']
    eye_coords     = tracking_results['eye_coords']

    # convert to RGB
    if len(frame.shape) < 3:
        tracked_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        tracked_frame = frame

    if cropped == True:
        # add an extra dimension for # of crops
        if tail_coords is not None:
            tail_coords   = tail_coords[np.newaxis, :, :]
        if spline_coords is not None:
            spline_coords = spline_coords[np.newaxis, :, :]
        if heading_angle is not None:
            heading_angle = np.array([[heading_angle]])
        if body_position is not None:
            body_position = body_position[np.newaxis, :]
        if eye_coords is not None:
            eye_coords    = eye_coords[np.newaxis, :, :]

    for k in range(n_crops):
        # add an arrow showing the heading direction
        if heading_angle is not None and not np.isnan(heading_angle[k, 0]):
                if body_position is not None and not np.isnan(body_position[k, 0]) and not np.isnan(body_position[k, 1]):
                    cv2.arrowedLine(tracked_frame, (int(round(body_position[k, 1])), int(round(body_position[k, 0]))),
                                            (int(round(body_position[k, 1] - 20*np.sin(heading_angle[k, 0]*np.pi/180.0))), int(round(body_position[k, 0] + 20*np.cos(heading_angle[k, 0]*np.pi/180.0)))), (49, 191, 114), 1)

        # add body center point
        if body_position is not None and not np.isnan(body_position[k, 0]) and not np.isnan(body_position[k, 1]):
                cv2.circle(tracked_frame, (int(round(body_position[k, 1])), int(round(body_position[k, 0]))), 1, (255, 128, 50), -1)

        # add eye points
        if eye_coords is not None and eye_coords.shape[-1] == 2:
            for i in range(2):
                if not np.isnan(eye_coords[k, 0, i]) and not np.isnan(eye_coords[k, 1, i]):
                    cv2.circle(tracked_frame, (int(round(eye_coords[k, 1, i])), int(round(eye_coords[k, 0, i]))), 1, (255, 0, 0), -1)

        if spline_coords is not None and spline_coords[k, 0, 0] != np.nan:
            # add spline
            spline_length = spline_coords.shape[2]
            for i in range(spline_length-1):
                if (not np.isnan(spline_coords[k, 0, i]) and not np.isnan(spline_coords[k, 1, i])
                    and not np.isnan(spline_coords[k, 0, i+1]) and not np.isnan(spline_coords[k, 1, i+1])):
                    cv2.line(tracked_frame, (int(round(spline_coords[k, 1, i])), int(round(spline_coords[k, 0, i]))),
                                            (int(round(spline_coords[k, 1, i+1])), int(round(spline_coords[k, 0, i+1]))), (0, 0, 255), 1)

        if spline_coords is not None and spline_coords[k, 0, 0] != np.nan:
            # add tail points
            tail_length = spline_coords.shape[2]
            for i in range(tail_length):
                if not np.isnan(spline_coords[k, 0, i]) and not np.isnan(spline_coords[k, 1, i]):
                    cv2.line(tracked_frame, (int(round(spline_coords[k, 1, i])), int(round(spline_coords[k, 0, i]))),
                                            (int(round(spline_coords[k, 1, i])), int(round(spline_coords[k, 0, i]))), (0, 255, 255), 1)

    return tracked_frame

def crop_frame(frame, offset, crop):
    if offset is not None and crop is not None:
        return frame[offset[0]:offset[0] + crop[0], offset[1]:offset[1] + crop[1]]
    else:
        return frame

def get_threshold_frame(frame, threshold, remove_noise=False, min_threshold=None, dilate=False):
    _, threshold_frame = cv2.threshold(frame.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY_INV)
    if min_threshold is not None:
        _, threshold_frame_2 = cv2.threshold(frame.astype(np.uint8), min_threshold, 255, cv2.THRESH_BINARY_INV)
        threshold_frame = np.logical_and(threshold_frame, np.logical_not(threshold_frame_2)).astype(np.uint8)*255
    np.divide(threshold_frame, 255, out=threshold_frame, casting='unsafe')

    # optionally remove noise from the thresholded image
    if remove_noise:
        kernel = np.ones((3, 3), np.uint8)
        threshold_frame = cv2.erode(threshold_frame, kernel, iterations=1)
        threshold_frame = cv2.dilate(threshold_frame, kernel, iterations=1)

    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        threshold_frame = cv2.erode(threshold_frame, kernel, iterations=1)
        threshold_frame = cv2.dilate(threshold_frame, kernel, iterations=2)

    return threshold_frame

def get_tail_skeleton_frame(tail_threshold_frame):
    return skeletonize(tail_threshold_frame).astype(np.uint8)

def get_tail_thinned_frame(tail_threshold_frame):
    return thin(tail_threshold_frame).astype(np.uint8)

def get_relative_coords(coords, offset):
    return (coords - offset)

def get_absolute_coords(coords, offset):
    '''
    Convert an array of cropped frame coordinates to absolute (original frame) coordinates.

    Arguments:
        coords (ndarray) : Array of coordinates.
        offset (ndarray) : (y, x) frame crop offset.

    Returns:
        abs_coords (ndarray) : Coordinates in the original frame.

    '''

    if coords.ndim == 1:
        return coords + offset
    elif coords.ndim == 2:
        return coords + offset[np.newaxis, :]
    elif coords.ndim == 3:
        return coords + offset[np.newaxis, :, np.newaxis]

# --- Misc. --- #

def get_video_batch_align_offsets(params): # todo: rewrite this using background images to estime transforms
    video_paths = params['video_paths']

    # store first frame of the first video
    source_frame = open_video(video_paths[0], [0], True)[0]

    batch_offsets = [None]

    for k in range(1, len(video_paths)):
        first_frame = open_video(video_paths[k], None, [1], None)[0]

        transform = cv2.estimateRigidTransform(source_frame, first_frame, False)

        offset = transform[:, 2]

        batch_offsets.append(offset)

    return batch_offsets

def apply_align_offset_to_frame(frame, batch_offset):
    transform = np.float32([[1, 0, -batch_offset[0]], [0, 1, -batch_offset[1]]])
    return cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]), borderValue=255)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
