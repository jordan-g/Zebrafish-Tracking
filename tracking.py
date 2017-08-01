from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

import multiprocessing
from multiprocessing import sharedctypes

from functools import partial
from itertools import chain

import imageio
try:
    from moviepy.video.io.ffmpeg_reader import *
except imageio.core.fetching.NeedDownloadError:
    imageio.plugins.ffmpeg.download()
    from moviepy.video.io.ffmpeg_reader import *

from skimage.morphology import skeletonize
from collections import deque

from open_media import open_image, open_video
import utilities

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

# --- Background subtraction --- #

def subtract_background_from_frames(frames, background, bg_sub_threshold, in_place=False):
    '''
    Subtract a background image from an array of frames.

    Arguments:
        frames (ndarray)       : (t, y, x) array of frames.
        background (ndarray)   : (y, x) background image array.
        bg_sub_threshold (int) : Threshold of the difference between a pixel and its background value that
                                 will cause to be considered a background pixel (and be set to white).
        in_place (bool)        : Whether to subtract the background in-place (faster, but overwrites the frames array).

    Returns:
        bg_sub_frames (ndarray) : The background-subtracted frames.
    '''

    if in_place:
        frames[ (frames - background < bg_sub_threshold) | (frames - background > 255 - bg_sub_threshold) ] = 255
    else:
        bg_sub_frames = frames.copy()
        bg_sub_frames[ (frames - background < bg_sub_threshold) | (frames - background > 255 - bg_sub_threshold) ] = 255

        return bg_sub_frames

# --- Noise Removal --- #

def remove_noise(frames):
    '''
    Performs noise removal on an array of frames.

    Arguments:
        frames (ndarray)       : (t, y, x) array of frames.
    Returns:
        denoised_frames (ndarray) : The denoised frames.
    '''

    denoised_frames = []

    for frame in frames:
        denoised_frames.append(cv2.fastNlMeansDenoisingMulti(frame, h=3, templateWindowSize=7, searchWindowSize=7))

    return denoised_frames

# --- Tracking --- #

def open_and_track_video(video_path, params, tracking_dir, video_number=0, progress_signal=None): # todo: add video creation from tracking data
    # Extract parameters
    subtract_background = params['subtract_background']
    background          = params['backgrounds'][video_number]
    crop_params         = params['crop_params']
    n_tail_points       = params['n_tail_points']
    save_video          = params['save_video']
    tracking_video_fps  = params['tracking_video_fps']
    use_multiprocessing = params['use_multiprocessing']
    n_crops             = len(params['crop_params'])
    bg_sub_threshold    = params['bg_sub_threshold']

    # Initialize a counter for the number of frames that have been tracked
    n_frames_tracked = 0

    # Start a timer for seeing how long the tracking took
    start_time = time.time()

    # Create a video capture object that we can re-use
    try:
        capture = cv2.VideoCapture(video_path)
    except:
        print("Error: Could not open video.")
        return

    # Get video info
    fps, n_frames_total = get_video_info(video_path)

    if tracking_video_fps == 0:
        # Set tracking video fps to be the same as the original video
        tracking_video_fps = fps

    if subtract_background and background is None:
        # Calculate the background
        background = open_video(video_path, return_frames=False, calc_background=True, capture=capture)

    # Initialize tracking data arrays
    tail_coords_array    = np.zeros((n_crops, n_frames_total, 2, n_tail_points)) + np.nan
    spline_coords_array  = np.zeros((n_crops, n_frames_total, 2, n_tail_points)) + np.nan
    heading_angle_array  = np.zeros((n_crops, n_frames_total, 1)) + np.nan
    body_position_array  = np.zeros((n_crops, n_frames_total, 2)) + np.nan
    eye_coords_array     = np.zeros((n_crops, n_frames_total, 2, 2)) + np.nan

    # Calculate the amount of frames to keep in memory at a time -- enough to fill 1/2 of the available memory
    mem = psutil.virtual_memory()
    mem_to_use = 0.5*mem.available
    frame = open_video(video_path, [0], capture=capture)[0]
    frame_size = frame.shape[0]*frame.shape[1]
    big_chunk_size = int(mem_to_use / frame_size)

    # Split frame numbers into big chunks - we keep only one big chunk of frames in memory at a time
    big_split_frame_nums = utilities.split_list_into_chunks(range(n_frames_total), big_chunk_size)

    if use_multiprocessing:
        # Create a pool of workers
        pool = multiprocessing.Pool(None)

    if progress_signal:
        # Send an initial progress update signal to the controller
        progress_signal.emit(video_number, 0)

    for i in range(len(big_split_frame_nums)):
        print("Tracking frames {} to {}...".format(big_split_frame_nums[i][0], big_split_frame_nums[i][-1]))

        # Get the frame numbers to process
        frame_nums = big_split_frame_nums[i]

        # Boolean indicating whether to have the capture object seek to the starting frame
        # This only needs to be done at the beginning to seek to frame 0
        seek_to_starting_frame = i == 0

        print("Opening frames...")

        # Load this big chunk of frames
        frames = open_video(video_path, frame_nums, capture=capture, seek_to_starting_frame=seek_to_starting_frame)

        if i == 0 and params['save_video']:
            # Create the video writer, for saving a video with tracking overlaid
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            new_video_path = "{}_tracked_video.avi".format(os.path.splitext(os.path.basename(video_path))[0])
            writer = cv2.VideoWriter(new_video_path, fourcc, tracking_video_fps,
                (frames[0].shape[1], frames[0].shape[0]), True)

        print("Tracking frames...")

        if use_multiprocessing:
            # Split the frames into small chunks - we let multiple processes deal with a chunk at a time
            small_chunk_size = 100
            split_frames = utilities.yield_chunks_from_array(frames, small_chunk_size)

            # Get the pool of workers to track the chunks of frames
            result_list = []
            for result in pool.imap(partial(track_frames, params, background), split_frames):
                result_list.append(result)

                # Increase the number of tracked frames counter
                n_frames_tracked += small_chunk_size

                if progress_signal:
                    # Send a progress update signal to the controller
                    percent_complete = 100.0*n_frames_tracked/n_frames_total
                    progress_signal.emit(video_number, percent_complete)

            # Get the number of frame chunks that have been processed
            n_chunks = len(result_list)

            # Add tracking results to tracking data arrays
            tail_coords_array[:, frame_nums, :, :]   = np.concatenate([result_list[i][0] for i in range(n_chunks)], axis=1)
            spline_coords_array[:, frame_nums, :, :] = np.concatenate([result_list[i][1] for i in range(n_chunks)], axis=1)
            heading_angle_array[:, frame_nums, :]    = np.concatenate([result_list[i][2] for i in range(n_chunks)], axis=1)
            body_position_array[:, frame_nums, :]    = np.concatenate([result_list[i][3] for i in range(n_chunks)], axis=1)
            eye_coords_array[:, frame_nums, :, :]    = np.concatenate([result_list[i][4] for i in range(n_chunks)], axis=1)
        else:
            # Track the big chunk of frames and add results to tracking data arrays
            (tail_coords_small_array, spline_coords_small_array,
             heading_angle_small_array, body_position_small_array, eye_coords_small_array) = track_frames(params, background, frames)

            # Increase the number of tracked frames counter
            n_frames_tracked += len(frame_nums)

            if progress_signal:
                # Send a progress update signal to the controller
                percent_complete = 100.0*n_frames_tracked/n_frames_total
                progress_signal.emit(video_number, percent_complete)

            # Add tracking results to tracking data arrays
            tail_coords_array[:, frame_nums, :, :]   = tail_coords_small_array
            spline_coords_array[:, frame_nums, :, :] = spline_coords_small_array
            heading_angle_array[:, frame_nums, :]    = heading_angle_small_array
            body_position_array[:, frame_nums, :]    = body_position_small_array
            eye_coords_array[:, frame_nums, :, :]    = eye_coords_small_array

        # Convert tracking coordinates from cropped frame space to original frame space
        for k in range(n_crops):
            tail_coords_array[k]   = get_absolute_coords(tail_coords_array[k], params['crop_params'][k]['offset'], params['scale_factor'])
            spline_coords_array[k] = get_absolute_coords(spline_coords_array[k], params['crop_params'][k]['offset'], params['scale_factor'])
            body_position_array[k] = get_absolute_coords(body_position_array[k], params['crop_params'][k]['offset'], params['scale_factor'])
            eye_coords_array[k]    = get_absolute_coords(eye_coords_array[k], params['crop_params'][k]['offset'], params['scale_factor'])

        if params['save_video']:
            print("Adding frames to tracking video...")

            for k in range(len(frames)):
                # Get the frame & the frame number
                frame     = frames[k]
                frame_num = frame_nums[k]

                # Create a dictionary with the tracking results for this frame
                results = {'tail_coords'  : tail_coords_array[:, frame_num, :, :],
                           'spline_coords': spline_coords_array[:, frame_num, :, :],
                           'eye_coords'   : eye_coords_array[:, frame_num, :, :],
                           'heading_angle': heading_angle_array[:, frame_num, :],
                           'body_position': body_position_array[:, frame_num, :]}

                # Overlay the tracking data onto the frame
                tracked_frame = add_tracking_to_frame(frame, results, n_crops=n_crops)

                # Write to the new video file
                writer.write(tracked_frame)

    if params['save_video']:
        print("Video created: {}.".format(new_video_path))

        # Release the video writer
        writer.release()

    if use_multiprocessing:
        # Close the pool of workers
        pool.close()
        pool.join()
    
    # Create the directory for saving tracking data if it doesn't exist
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)

    # Make a tracking params dictionary for this video
    tracking_params = params.copy()
    tracking_params['video_num'] = video_number

    # Save the tracking data
    np.savez(os.path.join(tracking_dir, "{}_tracking.npz".format(os.path.splitext(os.path.basename(video_path))[0])),
                          tail_coords=tail_coords_array, spline_coords=spline_coords_array,
                          heading_angle=heading_angle_array, body_position=body_position_array,
                          eye_coords=eye_coords_array, params=tracking_params)

    # Close the video capture object
    capture.release()

    # Stop the timer
    end_time = time.time()

    # Print the total tracking time
    print("Finished tracking. Total time: {}s.".format(end_time - start_time))

def open_and_track_video_batch(params, tracking_dir, progress_signal=None):
    video_paths = params['video_paths']

    # track each video with the same parameters
    for i in range(len(video_paths)):
        open_and_track_video(video_paths[i], params, tracking_dir, i, progress_signal)

def track_frames(params, background, frames):
    # Extract parameters
    crop_params         = params['crop_params']
    tracking_type       = params['type']
    n_tail_points       = params['n_tail_points']
    scale_factor        = params['scale_factor']
    interpolation       = utilities.translate_interpolation(params['interpolation'])
    subtract_background = params['subtract_background']
    bg_sub_threshold    = params['bg_sub_threshold']
    invert              = params['invert']

    # Get number of frames & number of crops
    n_frames = len(frames)
    n_crops  = len(crop_params)

    # Initialize tracking data arrays
    tail_coords_array    = np.zeros((n_crops, n_frames, 2, n_tail_points)) + np.nan
    spline_coords_array  = np.zeros((n_crops, n_frames, 2, n_tail_points)) + np.nan
    heading_angle_array  = np.zeros((n_crops, n_frames, 1)) + np.nan
    body_position_array  = np.zeros((n_crops, n_frames, 2)) + np.nan
    eye_coords_array     = np.zeros((n_crops, n_frames, 2, 2)) + np.nan

    # Set booleans for head & tail tracking
    track_head = tracking_type == "freeswimming"
    track_tail = tracking_type == "headfixed" or params['track_tail'] == True

    if invert:
        # Invert the frames if necessary
        frames = 255 - frames

    if subtract_background:
        # Subtract the background in-place
        subtract_background_from_frames(frames, background, bg_sub_threshold, in_place=True)

    for frame_number in range(n_frames):
        # Get the frame
        frame = frames[frame_number]

        for k in range(n_crops):
            # Get the crop & offset
            crop   = crop_params[k]['crop']
            offset = crop_params[k]['offset']

            # Shrink & crop the frame
            cropped_frame        = crop_frame(frame, offset, crop)
            scaled_cropped_frame = scale_frame(cropped_frame, scale_factor, interpolation)

            # Track the frame
            results = track_cropped_frame(scaled_cropped_frame, params, crop_params[k])

            # Re-scale coordinates
            if results['tail_coords'] is not None:
                results['tail_coords'] /= scale_factor
            if results['spline_coords'] is not None:
                results['spline_coords'] /= scale_factor
            if results['body_position'] is not None:
                results['body_position'] /= scale_factor
            if results['eye_coords'] is not None:
                results['eye_coords'] /= scale_factor

            # Add coordinates to tracking data arrays
            if results['tail_coords'] is not None:
                tail_coords_array[k, frame_number, :, :results['tail_coords'].shape[1]]     = results['tail_coords']
                spline_coords_array[k, frame_number, :, :results['spline_coords'].shape[1]] = results['spline_coords']
            heading_angle_array[k, frame_number, :] = results['heading_angle']
            body_position_array[k, frame_number, :] = results['body_position']
            eye_coords_array[k, frame_number, :, :] = results['eye_coords']

    return tail_coords_array, spline_coords_array, heading_angle_array, body_position_array, eye_coords_array

def track_cropped_frame(frame, params, crop_params):
    tracking_type  = params['type']
    scale_factor   = params['scale_factor']
    interpolation  = utilities.translate_interpolation(params['interpolation'])

    if tracking_type == "freeswimming":
        body_crop       = params['body_crop']
        track_tail_bool = params['track_tail']
        track_eyes_bool = params['track_eyes']

        body_crop_frame = None
        body_crop_coords = None

        crop_around_body = (track_eyes_bool or track_tail_bool) and body_crop is not None

        # track body
        if crop_around_body:
            heading_angle, body_position, rel_body_position, body_crop_coords, body_crop_frame = track_body(frame, params, crop_params, crop_around_body=True)
        else:
            heading_angle, body_position = track_body(frame, params, crop_params, crop_around_body=False)

        # track eyes
        if track_eyes_bool:
            if crop_around_body and body_crop_coords is not None and body_crop_frame is not None:
                eye_coords = track_eyes(body_crop_frame, params, crop_params)

                if eye_coords is not None:
                    heading_angle = get_heading_angle_from_eye_coords(eye_coords, heading_angle)

                    # convert eye coords to be relative to initial frame
                    eye_coords += body_crop_coords[:, 0][:, np.newaxis].astype(int)
            else:
                eye_coords = track_eyes(frame, params, crop_params)

        if track_tail_bool and body_position is not None:
            # track tail
            if crop_around_body and body_crop_coords is not None and body_crop_frame is not None:
                tail_coords, spline_coords = track_freeswimming_tail(body_crop_frame, params, crop_params, rel_body_position)
                if tail_coords is not None:
                    # convert eye coords to be relative to initial frame
                    tail_coords   += body_crop_coords[:, 0][:, np.newaxis].astype(int)
                    spline_coords += body_crop_coords[:, 0][:, np.newaxis].astype(int)
            else:
                tail_coords, spline_coords = track_freeswimming_tail(frame, params, crop_params, body_position)
        else:
            tail_coords, spline_coords = [None]*2
    elif tracking_type == "headfixed":
        # set head coords to None since we aren't interested in them
        heading_angle, body_position, eye_coords = [None]*3

        # track tail
        tail_coords, spline_coords = track_headfixed_tail(frame, params, crop_params)

    results = { 'tail_coords'  : tail_coords,
                'spline_coords': spline_coords,
                'heading_angle': heading_angle,
                'body_position': body_position,
                'eye_coords'   : eye_coords
              }

    return results

# --- Heading tracking --- #

def track_body(frame, params, crop_params, crop_around_body=True):
    adjust_thresholds = params['adjust_thresholds']
    body_threshold    = crop_params['body_threshold']
    body_crop         = params['body_crop']

    # create body threshold frame
    body_threshold_frame = get_threshold_frame(frame, body_threshold)

    # get heading angle & body position
    heading_angle, body_position = get_heading_angle_and_body_position(body_threshold_frame)

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

        return heading_angle, body_position, rel_body_position, body_crop_coords, body_crop_frame

    return heading_angle, body_position

def get_heading_angle_and_body_position(body_threshold_frame):
    # find contours in the thresholded frame
    try:
        image, contours, _ = cv2.findContours(body_threshold_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, _ = cv2.findContours(body_threshold_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    try:
        if len(contours) > 0:
            # choose the contour with the largest area as the body
            body_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(body_contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # fit an ellipse and get the angle and center position
            (x, y), (MA, ma), angle = cv2.fitEllipse(body_contour)

            # create an array for the center position
            position = np.array([y, x])

            if position[0] < 0 or position[1] < 0:
                return [None]*2
        else:
            return [None]*2

        return angle, position
    except:
        return [None]*2

def get_heading_angle_from_eye_coords(eye_coords, body_heading_angle):
    # get heading angle based on eye coordinates
    angle = 180.0 + np.arctan((eye_coords[0, 1] - eye_coords[0, 0])/(eye_coords[1, 1] - eye_coords[1, 0]))*180.0/np.pi

    if body_heading_angle is not None:
        # make it aligned with the body threshold heading angle
        if np.abs(angle - body_heading_angle) > 90:
            angle -= 180

        # if it's still not within 90º, just set it to the body threshold heading angle
        if np.abs(angle - body_heading_angle) > 90:
            angle = heading_angle
        else:
            # set the angle to the average of this angle & the body threshold heading angle
            angle = (body_heading_angle + angle)/2.0

    return angle

def track_eyes(frame, params, crop_params):
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

def get_eye_positions(eyes_threshold_image, prev_eye_coords=None): # todo: rewrite
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

    # get moments
    moments = [cv2.moments(contour) for contour in eye_contours]

    # initialize array to hold eye positions
    positions = np.zeros((2, 2))

    # get coordinates
    for i in range(2):
        M = moments[i]
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        positions[0, i] = cy
        positions[1, i] = cx

    return positions

# --- Freeswimming tail tracking --- #

def track_freeswimming_tail(frame, params, crop_params, body_position):
    adjust_thresholds  = params['adjust_thresholds']
    scale_factor      = params['scale_factor']
    min_tail_body_dist = params['min_tail_body_dist']*scale_factor
    max_tail_body_dist = params['max_tail_body_dist']*scale_factor
    n_tail_points      = params['n_tail_points']
    tail_threshold     = crop_params['tail_threshold']

    # create a thresholded frame
    tail_threshold_frame = get_threshold_frame(frame, tail_threshold)

    # get tail coordinates
    tail_coords, spline_coords = get_freeswimming_tail_coords(tail_threshold_frame, body_position,
                                                              min_tail_body_dist, max_tail_body_dist,
                                                              n_tail_points)

    if adjust_thresholds and tail_coords is None:
        # initialize counter
        i = 0
        
        # create a list of tail thresholds to go through
        tail_thresholds = list(range(tail_threshold-1, tail_threshold-5, -1)) + list(range(tail_threshold+1, tail_threshold+5))
        
        while tail_coords is None and i < 8:
            #  create a thresholded frame using new threshold
            tail_threshold_frame = get_threshold_frame(frame, tail_thresholds[i])

            # get tail coordinates
            tail_coords, spline_coords = get_freeswimming_tail_coords(tail_threshold_frame, body_position,
                                                                      min_tail_body_dist, max_tail_body_dist,
                                                                      n_tail_points)

            # increase counter
            i += 1

    return tail_coords, spline_coords

def get_freeswimming_tail_coords(tail_threshold_frame, body_position, min_tail_body_dist, max_tail_body_dist, n_tail_points, max_r=4, smoothing_factor=3,): # todo: make max radius & smoothing factor user settable
    # get tail skeleton matrix
    skeleton_matrix = get_tail_skeleton_frame(tail_threshold_frame)

    # get coordinates of nonzero points of thresholded image
    nonzeros = np.nonzero(skeleton_matrix)

    # zero out pixels that are close to body
    skeleton_matrix = cv2.circle(skeleton_matrix, (int(round(body_position[1])), int(round(body_position[0]))), int(min_tail_body_dist), (0, 0, 0), -1)

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

    if tail_coords is None:
        # couldn't get tail coordinates; end here.
        return [None]*2

    # get number of tail coordinates
    n_tail_coords = tail_coords.shape[1]

    # get size of thresholded image
    y_size = tail_threshold_frame.shape[0]
    x_size = tail_threshold_frame.shape[1]

    if tail_coords is not None:
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
                pixel_sum += tail_threshold_frame[np.minimum(y_size-1, np.maximum(0, int(y+k))), np.minimum(x_size-1, np.maximum(0, int(x+l)))]
                y_sum     += k*tail_threshold_frame[np.minimum(y_size-1, np.maximum(0, int(y+k))), np.minimum(x_size-1, np.maximum(0, int(x+l)))]
                x_sum     += l*tail_threshold_frame[np.minimum(y_size-1, np.maximum(0, int(y+k))), np.minimum(x_size-1, np.maximum(0, int(x+l)))]

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
            if (closest_body_distance is None) or (closest_body_distance is not None and body_distance < closest_body_distance):
                # either no point has been found yet or this is a closer point than the previous best

                # get nonzero elements in 3x3 neigbourhood around the point
                nonzero_neighborhood = skeleton_matrix[r-1:r+2, c-1:c+2] != 0

                # if the number of non-zero points in the neighborhood is at least 2
                # (ie. there's at least one direction to move in along the tail),
                # set this to our tail starting point.
                if np.sum(nonzero_neighborhood) == 2:
                    tail_start_coords = np.array([r, c])
                    closest_body_distance = body_distance

    if tail_start_coords is None:
        # still could not find start of the tail; end here.
        return None

    # walk along the tail
    found_coords = walk_along_tail(tail_start_coords, max_r, skeleton_matrix)

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

        if next_coords is not None:
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

    if unique_coords is None or len(unique_coords) == 0:
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
    tail_start_coords = get_relative_coords(params['tail_start_coords'], crop_params['offset'], params['scale_factor'])
    heading_direction = params['heading_direction']
    n_tail_points     = params['n_tail_points']
    heading_angle     = params['heading_angle']

    print(tail_start_coords, frame.shape)

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

def crop_frame_around_body(frame, body_position, body_crop, scale_factor=1):
    if body_crop is None or body_position is None:
        body_crop_coords = np.array([[0, frame.shape[0]], [0, frame.shape[1]]])
    else:
        body_crop_coords = np.array([[np.maximum(0, int((body_position[0]-body_crop[0])/scale_factor)), np.minimum(frame.shape[0], int((body_position[0]+body_crop[0])/scale_factor))],
                                     [np.maximum(0, int((body_position[1]-body_crop[1])/scale_factor)), np.minimum(frame.shape[1], int((body_position[1]+body_crop[1])/scale_factor))]])

    # crop the frame to the tail
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
    tail_coords    = tracking_results['tail_coords']
    spline_coords  = tracking_results['spline_coords']
    heading_angle  = tracking_results['heading_angle']
    body_position  = tracking_results['body_position']
    eye_coords     = tracking_results['eye_coords']

    # convert to BGR
    if len(frame.shape) < 3:
        tracked_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        tracked_frame = frame

    if cropped == True:
        # add extra dimension for # of crops
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
        if heading_angle is not None and not np.isnan(heading_angle[k, 0]):
                if body_position is not None and not np.isnan(body_position[k, 0]) and not np.isnan(body_position[k, 1]):
                    cv2.line(tracked_frame, (int(round(body_position[k, 1] - 20*np.sin(heading_angle[k, 0]*np.pi/180.0))), int(round(body_position[k, 0] + 20*np.cos(heading_angle[k, 0]*np.pi/180.0)))),
                                            (int(round(body_position[k, 1] + 20*np.sin(heading_angle[k, 0]*np.pi/180.0))), int(round(body_position[k, 0] - 20*np.cos(heading_angle[k, 0]*np.pi/180.0)))), (49, 191, 114), 1)

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

        if tail_coords is not None and tail_coords[k, 0, 0] != np.nan:
            # add tail points
            tail_length = tail_coords.shape[2]
            for i in range(tail_length):
                if not np.isnan(tail_coords[k, 0, i]) and not np.isnan(tail_coords[k, 1, i]):
                    cv2.line(tracked_frame, (int(round(tail_coords[k, 1, i])), int(round(tail_coords[k, 0, i]))),
                                            (int(round(tail_coords[k, 1, i])), int(round(tail_coords[k, 0, i]))), (0, 255, 255), 1)

    return tracked_frame

def crop_frame(frame, offset, crop):
    if offset is not None and crop is not None:
        return frame[offset[0]:offset[0] + crop[0], offset[1]:offset[1] + crop[1]]
    else:
        return frame

def scale_frame(frame, scale_factor, interpolation):
    if scale_factor != 1:
        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interpolation)
    return frame

def get_threshold_frame(frame, threshold):
    _, threshold_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)
    np.divide(threshold_frame, 255, out=threshold_frame, casting='unsafe')

    # remove noise from the thresholded image
    kernel = np.ones((2, 2),np.uint8)
    threshold_frame = cv2.erode(threshold_frame, kernel, iterations=1)
    threshold_frame = cv2.dilate(threshold_frame, kernel, iterations=1)

    return threshold_frame

def get_tail_skeleton_frame(tail_threshold_frame):
    return skeletonize(tail_threshold_frame).astype(np.uint8)

def get_relative_body_crop(body_crop, scale_factor):
    return body_crop*scale_factor

def get_relative_coords(coords, offset, scale_factor):
    return (coords - offset)*scale_factor

def get_absolute_coords(coords, offset, scale_factor):
    '''
    Convert an array of cropped frame coordinates to absolute (original frame) coordinates.

    Arguments:
        coords (ndarray) : Array of coordinates.
        offset (ndarray) : (y, x) frame crop offset.
        scale_factor (float) : Scaling factor of the cropped frame.

    Returns:
        abs_coords (ndarray) : Coordinates in the original frame.

    '''

    if coords.ndim == 1:
        return coords/scale_factor + offset
    elif coords.ndim == 2:
        return coords/scale_factor + offset[np.newaxis, :]
    elif coords.ndim == 3:
        return coords/scale_factor + offset[np.newaxis, :, np.newaxis]

# --- Misc. --- #

def get_video_batch_align_offsets(params):
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