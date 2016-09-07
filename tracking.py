from __future__ import division
import numpy as np
import cv2
from scipy import interpolate

import os
import re
import itertools

from moviepy.video.io.ffmpeg_reader import *
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

from bwmorph_thin import bwmorph_thin
import analysis as an

import pdb
import time
# import threading
import multiprocessing
from multiprocessing import sharedctypes
from functools import partial

default_crop_params = { 'offset': [0, 0],      # crop offset
                        'crop': None,          # crop size
                        'tail_threshold': 200, # pixel brightness to use for thresholding to find the tail (0-255)
                        'head_threshold': 50   # pixel brightness to use for thresholding to find the eyes (0-255)
                      }

cv2.setNumThreads(0)

def open_and_track_image(image_path, tracking_dir, **kwargs):
    """
    Open & track an image file.
    
    Args:
        image_path         (str): path to the image.
        tracking_dir       (str): directory in which to save tracking data.
        ---
        crop      (int y, int x): height & width of crop area.
        offset    (int y, int x): coordinates at which to begin the crop.
        shrink_factor    (float): factor to use to shrink the image (0 - 1).
        invert            (bool): whether to invert the image.
        min_tail_eye_dist  (int): minimum distance between the eyes & start of the tail.
        head_threshold     (int): brightness threshold to use to find the head (0 - 255).
        tail_threshold     (int): brightness threshold to use to find the tail (0 - 255).
        track_head        (bool): whether to track the head.
        track_tail        (bool): whether to track the tail.
        n_tail_points      (int): # of coordinates to keep when tracking the tail.
        tail_crop (int y, int x): height & width of crop area for tail tracking.
        adjust_thresholds (bool): whether to adjust thresholds if tracking fails.
    """

    crop_params       = kwargs.get('crop_params', [default_crop_params])
    shrink_factor     = kwargs.get('shrink_factor', 1)
    invert            = kwargs.get('invert', False)
    min_tail_eye_dist = kwargs.get('min_tail_eye_dist', 20)
    max_tail_eye_dist = kwargs.get('max_tail_eye_dist', 30)
    track_head        = kwargs.get('track_head', True)
    track_tail        = kwargs.get('track_tail', True)
    n_tail_points     = kwargs.get('n_tail_points', 30)
    tail_crop         = kwargs.get('tail_crop', None)
    adjust_thresholds = kwargs.get('adjust_thresholds', True)
    eye_resize_factor = kwargs.get('eye_resize_factor', 1)
    interpolation     = kwargs.get('interpolation', cv2.INTER_NEAREST)

    min_tail_eye_dist *= shrink_factor

    # load the original image
    image = load_image(image_path)

    if invert:
        # invert the image
        image = (255 - image)

    for k in range(len(crop_params)):
        crop           = crop_params[k]['crop']
        offset         = crop_params[k]['offset']
        head_threshold = crop_params[k]['head_threshold']
        tail_threshold = crop_params[k]['tail_threshold']

        if crop != None and offset != None:
            # edit crop & offset to take into account the shrink factor
            crop   = (round(crop[0]*shrink_factor), round(crop[1]*shrink_factor))
            offset = (round(offset[0]*shrink_factor), round(offset[1]*shrink_factor))
        
        # shrink & crop the image
        shrunken_image = shrink_image(image, shrink_factor)
        cropped_image  = crop_image(shrunken_image, offset, crop)

        # track the image
        (tail_coords, spline_coords,
         eye_coords, heading_coords, skeleton_matrix) = track_image(cropped_image,
                                                                 head_threshold, tail_threshold,
                                                                 min_tail_eye_dist,
                                                                 track_head, track_tail,
                                                                 n_tail_points, tail_crop,
                                                                 adjust_thresholds,
                                                                 eye_resize_factor, interpolation)
        
        # add tracked points to the image
        tracked_image = add_tracking_to_image(cropped_image,
                                              tail_coords, spline_coords,
                                              eye_coords, heading_coords)

        # save the new image
        cv2.imwrite(os.path.join(tracking_dir, "crop_{}.png".format(k)), tracked_image)

def track_frames(fps, params, new_video_path, frames):
    crop_params       = params['crop_params']
    shrink_factor     = params['shrink_factor']
    invert            = params['invert']
    min_tail_eye_dist = params['min_tail_eye_dist']
    max_tail_eye_dist = params['max_tail_eye_dist']
    track_head        = params['track_head']
    track_tail        = params['track_tail']
    n_tail_points     = params['n_tail_points']
    tail_crop         = params['tail_crop']
    adjust_thresholds = params['adjust_thresholds']
    eye_resize_factor = params['eye_resize_factor']

    if params['interpolation'] == 'Nearest Neighbor':
        interpolation = cv2.INTER_NEAREST
    elif params['interpolation'] == 'Linear':
        interpolation = cv2.INTER_LINEAR
    elif params['interpolation'] == 'Bicubic':
        interpolation = cv2.INTER_CUBIC
    elif params['interpolation'] == 'Lanczos':
        interpolation = cv2.INTER_LANCZOS4

    save_video        = params['save_video']
    new_video_fps     = params['new_video_fps']

    min_tail_eye_dist *= shrink_factor

    n_frames = len(frames)
    n_crops = len(crop_params)

    print(n_frames, n_crops)

    def track_frame_at_number(frame_number):
        # get corresponding frame
        image = frames[frame_number]

        # crop
        cropped_image = crop_image(image, offset, crop)

        # track the image
        (tail_coords, spline_coords,
         eye_coords, heading_coords, skeleton_matrix) = track_image(cropped_image,
                                                                    head_threshold, tail_threshold,
                                                                    min_tail_eye_dist, max_tail_eye_dist,
                                                                    track_head, track_tail,
                                                                    n_tail_points, tail_crop,
                                                                    adjust_thresholds,
                                                                    eye_resize_factor, interpolation, prev_eye_coords=eye_coords_array[k, :frame_number])
        # print("seeya")
        # add tracked coordinates to arrays
        if tail_coords != None:
            tail_coords_array[k, frame_number, :, :]   = tail_coords
            spline_coords_array[k, frame_number, :, :] = spline_coords
        else:
            problematic_tail_frames[k].append(frame_number)

        if eye_coords != None:
            eye_coords_array[k, frame_number, :, :]  = eye_coords
            heading_coords_array[k, frame_number, :, :] = heading_coords
        else:
            problematic_head_frames[k].append(frame_number)

        if save_video:
            # add tracked points to the image
            tracked_image = add_tracking_to_image(cropped_image,
                                                  tail_coords, spline_coords,
                                                  eye_coords, heading_coords,
                                                  heading_coords_array[:frame_number, :, 1])

            return tracked_image

    def track_frame_at_time(t):
        frame_number = int(round(t*new_video_fps))

        tracked_image = track_frame_at_number(frame_number)
        
        return tracked_image

    # animations = [[]]*n_crops

    # initialize tracking data arrays
    eye_coords_array     = np.zeros((n_crops, n_frames, 2, 2)) + np.nan
    heading_coords_array = np.zeros((n_crops, n_frames, 2, 2)) + np.nan
    tail_coords_array    = np.zeros((n_crops, n_frames, 2, n_tail_points)) + np.nan
    spline_coords_array  = np.zeros((n_crops, n_frames, 2, n_tail_points)) + np.nan

    # initialize problematic frame arrays
    problematic_head_frames = [[]]*n_crops
    problematic_tail_frames = [[]]*n_crops

    for k in range(n_crops):
        crop           = crop_params[k]['crop']
        offset         = crop_params[k]['offset']
        head_threshold = crop_params[k]['head_threshold']
        tail_threshold = crop_params[k]['tail_threshold']

        if crop != None and offset != None:
            # edit crop & offset to take into account the shrink factor
            crop   = (round(crop[0]*shrink_factor), round(crop[1]*shrink_factor))
            offset = (round(offset[0]*shrink_factor), round(offset[1]*shrink_factor))

        if save_video:
            # track & make a video
            animation = mpy.VideoClip(track_frame_at_time, duration=n_frames/new_video_fps)
            animation.write_videofile(os.path.join(os.path.dirname(new_video_path), "crop_{}.mov".format(k)), codec='libx264', fps=new_video_fps)
            # animations[k].append(animation)
        else:
            # just track all the frames
            for frame_number in range(n_frames):
                print("Tracking frame {}.".format(frame_number))
                track_frame_at_number(frame_number)

    return [eye_coords_array, heading_coords_array, tail_coords_array, spline_coords_array, problematic_head_frames, problematic_tail_frames]

def track_frames_from_folder(folder_path, fps, params, new_video_path, frame_nums):
    n_frames = len(frame_nums)
    start_frame = frame_nums[0]
    frames = load_frames_from_folder(folder_path, n_frames, start_frame=start_frame, evenly_spaced=False)

    invert = params['invert']
    shrink_factor = params['shrink_factor']
    n_crops = len(params['crop_params'])
    print(n_frames)

    # pre-process all images
    for n in range(n_frames):
        image = frames[n]

        # convert to greyscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if invert:
            # invert the image
            image = (255 - image)

        # shrink & crop the image
        shrunken_image = shrink_image(image, shrink_factor)

        frames[n] = shrunken_image

    return track_frames(fps, params, new_video_path, frames)

def open_and_track_folder(folder_path, new_video_path, params):
    """
    Open & track a folder of frames.
    
    Args:
        folder_path        (str): path to the folder of frames.
        new_video_path     (str): directory in which to save tracking data.
        ---
        crop      (int y, int x): height & width of crop area.
        offset    (int y, int x): coordinates at which to begin the crop.
        shrink_factor    (float): factor to use to shrink the image (0 - 1).
        invert            (bool): whether to invert the image.
        min_tail_eye_dist  (int): minimum distance between the eyes & start of the tail.
        max_tail_eye_dist  (int): maximum distance between the eyes & start of the tail.
        head_threshold     (int): brightness threshold to use to find the head (0 - 255).
        tail_threshold     (int): brightness threshold to use to find the tail (0 - 255).
        track_head        (bool): whether to track the head.
        track_tail        (bool): whether to track the tail.
        n_tail_points      (int): # of coordinates to keep when tracking the tail.
        tail_crop (int y, int x): height & width of crop area for tail tracking.
        adjust_thresholds (bool): whether to adjust thresholds if tracking fails.
        ---
        save_video        (bool): whether to save a video with tracking.
        new_video_fps      (int): fps to use for the created video.
    """

    invert = params['invert']
    shrink_factor = params['shrink_factor']
    n_crops = len(params['crop_params'])
    save_video = params['save_video']
    new_video_fps = params['new_video_fps']

    start_time = time.time()

    # get number of frames & set fps to 1
    n_frames = get_num_frames_in_folder(folder_path)
    fps = 1

    if not save_video:
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in xrange(0, len(l), n):
                yield l[i:i + n]

        split_frames = chunks(range(n_frames), 100)

        result_list = []

        # create a pool of workers
        pool  = multiprocessing.Pool(None)

        def log_result(result):
            result_list.append(result)

        # results = []
        func = partial(track_frames_from_folder, folder_path, fps, params, new_video_path)
        pool.map_async(func, split_frames, callback=log_result)
        pool.close()
        pool.join()

        n_chunks = len(result_list[0])

        eye_coords_array = np.concatenate([result_list[0][i][0] for i in range(n_chunks)], axis=1)
        heading_coords_array = np.concatenate([result_list[0][i][1] for i in range(n_chunks)], axis=1)
        tail_coords_array = np.concatenate([result_list[0][i][2] for i in range(n_chunks)], axis=1)
        spline_coords_array = np.concatenate([result_list[0][i][3] for i in range(n_chunks)], axis=1)

        problematic_tail_frames = np.concatenate([result_list[0][i][4] for i in range(n_chunks)], axis=1)
        problematic_head_frames = np.concatenate([result_list[0][i][5] for i in range(n_chunks)], axis=1)
    else:
        eye_coords_array, heading_coords_array, tail_coords_array, spline_coords_array, problematic_head_frames, problematic_tail_frames = track_frames_from_folder(folder_path, fps, params, new_video_path, range(n_frames))

    # set directory for saving tracking data
    data_dir = os.path.dirname(new_video_path)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # save data
    for k in range(n_crops):
        np.savez(os.path.join(data_dir, "crop_{}_tracking_data.npy".format(k)),
                              eye_coords=eye_coords_array[k], heading_coords=heading_coords_array[k],
                              tail_coords=tail_coords_array[k], spline_coords=spline_coords_array[k],
                              params=params, problematic_head_frames=problematic_head_frames[k],
                              problematic_tail_frames=problematic_tail_frames[k])

    end_time = time.time()

    print("Finished tracking. Total time: {}s.".format(end_time - start_time))

def track_frames_from_video(video_path, fps, params, new_video_path, frame_nums):
    crop_params       = params['crop_params']
    shrink_factor     = params['shrink_factor']
    invert            = params['invert']
    min_tail_eye_dist = params['min_tail_eye_dist']
    max_tail_eye_dist = params['max_tail_eye_dist']
    track_head        = params['track_head']
    track_tail        = params['track_tail']
    n_tail_points     = params['n_tail_points']
    tail_crop         = params['tail_crop']
    adjust_thresholds = params['adjust_thresholds']
    eye_resize_factor = params['eye_resize_factor']

    if params['interpolation'] == 'Nearest Neighbor':
        interpolation = cv2.INTER_NEAREST
    elif params['interpolation'] == 'Linear':
        interpolation = cv2.INTER_LINEAR
    elif params['interpolation'] == 'Bicubic':
        interpolation = cv2.INTER_CUBIC
    elif params['interpolation'] == 'Lanczos':
        interpolation = cv2.INTER_LANCZOS4

    save_video        = params['save_video']
    new_video_fps     = params['new_video_fps']

    min_tail_eye_dist *= shrink_factor

    start_frame = frame_nums[0]
    n_frames = len(frame_nums)

    # open the video
    try:
        cap = FFMPEG_VideoReader(video_path, True)
        cap.initialize()
    except:
        print("Error: Could not open video.")
        return None

    # get number of frames & fps
    # n_frames = ffmpeg_parse_infos(video_path)["video_nframes"]
    fps      = ffmpeg_parse_infos(video_path)["video_fps"]

    print("Original video fps: {0}. Number of frames: {1}.".format(fps, n_frames))

    n_crops = len(params['crop_params'])

    frames = []

    print(n_frames, n_crops)

    def track_frame_at_number(frame_number):
        # get corresponding frame
        image = cap.get_frame(frame_number/fps)

        # convert to greyscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if invert:
            # invert the image
            image = (255 - image)

        # shrink & crop the image
        shrunken_image = shrink_image(image, shrink_factor)

        # crop
        cropped_image = crop_image(shrunken_image, offset, crop)

        # track the image
        (tail_coords, spline_coords,
         eye_coords, heading_coords, skeleton_matrix) = track_image(cropped_image,
                                                                    head_threshold, tail_threshold,
                                                                    min_tail_eye_dist, max_tail_eye_dist,
                                                                    track_head, track_tail,
                                                                    n_tail_points, tail_crop,
                                                                    adjust_thresholds,
                                                                    eye_resize_factor, interpolation, prev_eye_coords=eye_coords_array[k, :frame_number])
        # print("seeya")
        # add tracked coordinates to arrays
        if tail_coords != None:
            tail_coords_array[k, frame_number, :, :]   = tail_coords
            spline_coords_array[k, frame_number, :, :] = spline_coords
        else:
            problematic_tail_frames[k].append(frame_number)

        if eye_coords != None:
            eye_coords_array[k, frame_number, :, :]  = eye_coords
            heading_coords_array[k, frame_number, :, :] = heading_coords
        else:
            problematic_head_frames[k].append(frame_number)

        if save_video:
            # add tracked points to the image
            tracked_image = add_tracking_to_image(cropped_image,
                                                  tail_coords, spline_coords,
                                                  eye_coords, heading_coords,
                                                  heading_coords_array[:frame_number, :, 1])

            return tracked_image

    def track_frame_at_time(t):
        frame_number = int(round(t*new_video_fps))

        tracked_image = track_frame_at_number(frame_number)
        
        return tracked_image

    # initialize tracking data arrays
    eye_coords_array     = np.zeros((n_crops, n_frames, 2, 2)) + np.nan
    heading_coords_array = np.zeros((n_crops, n_frames, 2, 2)) + np.nan
    tail_coords_array    = np.zeros((n_crops, n_frames, 2, n_tail_points)) + np.nan
    spline_coords_array  = np.zeros((n_crops, n_frames, 2, n_tail_points)) + np.nan

    # initialize problematic frame arrays
    problematic_head_frames = [[]]*n_crops
    problematic_tail_frames = [[]]*n_crops

    for k in range(n_crops):
        crop           = crop_params[k]['crop']
        offset         = crop_params[k]['offset']
        head_threshold = crop_params[k]['head_threshold']
        tail_threshold = crop_params[k]['tail_threshold']

        if crop != None and offset != None:
            # edit crop & offset to take into account the shrink factor
            crop   = (round(crop[0]*shrink_factor), round(crop[1]*shrink_factor))
            offset = (round(offset[0]*shrink_factor), round(offset[1]*shrink_factor))

        if save_video:
            # track & make a video
            animation = mpy.VideoClip(track_frame_at_time, duration=n_frames/new_video_fps)
            animation.write_videofile(os.path.join(os.path.dirname(new_video_path), "crop_{}.mov".format(k)), codec='libx264', fps=new_video_fps)
            # animations[k].append(animation)
        else:
            # just track all the frames
            for frame_number in range(n_frames):
                print("Tracking frame {}.".format(frame_number))
                track_frame_at_number(frame_number)

    cap.release()

    return [eye_coords_array, heading_coords_array, tail_coords_array, spline_coords_array, problematic_head_frames, problematic_tail_frames]

def open_and_track_video(video_path, new_video_path, params):
    """
    Open & track a video.
    
    Args:
        video_path         (str): path to the video to be tracked.
        tracking_dir       (str): directory in which to save tracking data.
        ---
        crop      (int y, int x): height & width of crop area.
        offset    (int y, int x): coordinates at which to begin the crop.
        shrink_factor    (float): factor to use to shrink the image (0 - 1).
        invert            (bool): whether to invert the image.
        min_tail_eye_dist  (int): minimum distance between the eyes & start of the tail.
        max_tail_eye_dist  (int): maximum distance between the eyes & start of the tail.
        head_threshold     (int): brightness threshold to use to find the head (0 - 255).
        tail_threshold     (int): brightness threshold to use to find the tail (0 - 255).
        track_head        (bool): whether to track the head.
        track_tail        (bool): whether to track the tail.
        n_tail_points      (int): # of coordinates to keep when tracking the tail.
        tail_crop (int y, int x): height & width of crop area for tail tracking.
        adjust_thresholds (bool): whether to adjust thresholds if tracking fails.
        ---
        save_video        (bool): whether to save a video with tracking.
        new_video_fps      (int): fps to use for the created video.
    """

    start_time = time.time()

    crop_params       = params['crop_params']
    shrink_factor     = params['shrink_factor']
    invert            = params['invert']
    min_tail_eye_dist = params['min_tail_eye_dist']
    max_tail_eye_dist = params['max_tail_eye_dist']
    track_head        = params['track_head']
    track_tail        = params['track_tail']
    n_tail_points     = params['n_tail_points']
    tail_crop         = params['tail_crop']
    adjust_thresholds = params['adjust_thresholds']
    eye_resize_factor = params['eye_resize_factor']
    interpolation     = params['interpolation']

    save_video        = params['save_video']
    new_video_fps     = params['new_video_fps']

    # get number of frames & set fps to 1
    fps, n_frames = get_video_info(video_path)

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in xrange(0, len(l), n):
            yield l[i:i + n]

    if not save_video:
        # split_frames = chunks(range(n_frames), 100)

        # result_list = []

        # # create a pool of workers
        # pool  = multiprocessing.Pool(None)

        # def log_result(result):
        #     result_list.append(result)

        # # results = []
        # func = partial(track_frames_from_video, video_path, fps, params, new_video_path)
        # pool.map_async(func, split_frames, callback=log_result)
        # pool.close()
        # pool.join()

        # n_chunks = len(result_list[0])

        # eye_coords_array = np.concatenate([result_list[0][i][0] for i in range(n_chunks)], axis=1)
        # heading_coords_array = np.concatenate([result_list[0][i][1] for i in range(n_chunks)], axis=1)
        # tail_coords_array = np.concatenate([result_list[0][i][2] for i in range(n_chunks)], axis=1)
        # spline_coords_array = np.concatenate([result_list[0][i][3] for i in range(n_chunks)], axis=1)

        # problematic_tail_frames = np.concatenate([result_list[0][i][4] for i in range(n_chunks)], axis=1)
        # problematic_head_frames = np.concatenate([result_list[0][i][5] for i in range(n_chunks)], axis=1)
        eye_coords_array, heading_coords_array, tail_coords_array, spline_coords_array, problematic_head_frames, problematic_tail_frames = track_frames_from_video(video_path, fps, params, new_video_path, range(n_frames))
    else:
        eye_coords_array, heading_coords_array, tail_coords_array, spline_coords_array, problematic_head_frames, problematic_tail_frames = track_frames_from_video(video_path, fps, params, new_video_path, range(n_frames))

    # set directory for saving tracking data
    data_dir = os.path.dirname(new_video_path)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # save data
    for k in range(n_crops):
        np.savez(os.path.join(data_dir, "crop_{}_tracking_data.npy".format(k)),
                              eye_coords=eye_coords_array[k], heading_coords=heading_coords_array[k],
                              tail_coords=tail_coords_array[k], spline_coords=spline_coords_array[k],
                              params=params, problematic_head_frames=problematic_head_frames[k],
                              problematic_tail_frames=problematic_tail_frames[k])

    end_time = time.time()

    print("Finished tracking. Total time: {}s.".format(end_time - start_time))

def load_frame_from_image(image_path):
    print("Loading {}.".format(image_path))

    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except:
        print("Error: Could not open image.")
        return None

    return image

def get_num_frames_in_folder(folder_path):
    n_frames_orig = 0

    for filename in sort_nicely(os.listdir(folder_path)):
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            n_frames_orig += 1

    return n_frames_orig

def load_frames_from_folder(folder_path, n_frames=None, start_frame=0, evenly_spaced=True):
    print("Loading images from {}.".format(folder_path))

    frame_filenames = []
    n_frames_orig = 0

    for filename in sort_nicely(os.listdir(folder_path)):
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            n_frames_orig += 1
            frame_filenames.append(filename)

    if n_frames == None or n_frames > n_frames_orig: # load all frames
        n_frames = n_frames_orig

        f = 0
        frames = []

        while f < n_frames:
            print(frame_filenames[f])
            # get image
            frame = load_frame_from_image(os.path.join(folder_path, frame_filenames[f]))

            # convert to greyscale
            if len(frame.shape) >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # add to frames list
            frames.append(frame)

            f += 1
    else: # load only some frames
        if evenly_spaced:
            # get evenly spaced frame numbers
            r = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            frame_nums = [0] + r(100, n_frames_orig)
        else:
            frame_nums = range(start_frame, start_frame + n_frames)

        f = 0
        frames = []

        f_last = 0

        for f in sorted(frame_nums):
            frame = load_frame_from_image(os.path.join(folder_path, frame_filenames[f]))

            if frame != None:
                # convert to greyscale
                if len(frame.shape) >= 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # add to frames list
                frames.append(frame)

    if len(frames) == 0:
        print("Could not find any images.")
        return None

    return frames

def get_video_info(video_path):
    # get video info
    fps           = ffmpeg_parse_infos(video_path)["video_fps"]
    n_frames = ffmpeg_parse_infos(video_path)["video_nframes"]

    return fps, n_frames

def load_frames_from_video(video_path, n_frames=None):
    print("Loading video from {}.".format(video_path))

    # open the video
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print("Error: Could not open video.")
        return None

    # get video info
    n_frames_orig = ffmpeg_parse_infos(video_path)["video_nframes"]
    fps = ffmpeg_parse_infos(video_path)["video_fps"]
    print("Original video fps: {0}. n_frames: {1}".format(fps, n_frames_orig))

    if n_frames == None or n_frames > n_frames_orig: # load all frames
        n_frames = n_frames_orig

        f = 0
        frames = []

        while f < n_frames:
            # get image
            ret, frame = cap.read()

            # convert to greyscale
            if len(frame.shape) >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # add to frames list
            frames.append(frame)

            f += 1
    else: # load only some frames

        # get evenly spaced frame numbers
        r = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        frame_nums = [0] + r(100, n_frames_orig)

        f = 0
        frames = []

        f_last = 0

        for f in sorted(frame_nums):
            try:
                cap.set(cv2.CV_CAP_PROP_POS_FRAMES,f)
            except:
                cap.set(1,f)
            ret, frame = cap.read()

            if frame != None:
                # convert to greyscale
                if len(frame.shape) >= 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # add to frames list
                frames.append(frame)

    return frames

def open_and_track_video_frame(video_path, tracking_dir, frame_num, **kwargs):
    """
    Open & track a single frame from a video.
    
    Args:
        video_path         (str): path to the video from which to track a frame.
        tracking_dir       (str): directory in which to save tracking data.
        frame_num          (int): frame number to track.
        ---
        crop      (int y, int x): height & width of crop area.
        offset    (int y, int x): coordinates at which to begin the crop.
        shrink_factor    (float): factor to use to shrink the image (0 - 1).
        invert            (bool): whether to invert the image.
        min_tail_eye_dist  (int): minimum distance between the eyes & start of the tail.
        max_tail_eye_dist  (int): maximum distance between the eyes & start of the tail.
        head_threshold     (int): brightness threshold to use to find the head (0 - 255).
        tail_threshold     (int): brightness threshold to use to find the tail (0 - 255).
        track_head        (bool): whether to track the head.
        track_tail        (bool): whether to track the tail.
        n_tail_points      (int): # of coordinates to keep when tracking the tail.
        tail_crop (int y, int x): height & width of crop area for tail tracking.
        adjust_thresholds (bool): whether to adjust thresholds if tracking fails.
    """

    start_time = time.time()

    crop              = kwargs.get('crop', None)
    offset            = kwargs.get('offset', None)
    shrink_factor     = kwargs.get('shrink_factor', 1)
    invert            = kwargs.get('invert', False)
    min_tail_eye_dist = kwargs.get('min_tail_eye_dist', 20)
    max_tail_eye_dist = kwargs.get('max_tail_eye_dist', 30)
    head_threshold    = kwargs.get('head_threshold')
    tail_threshold    = kwargs.get('tail_threshold')
    track_head        = kwargs.get('track_head', True)
    track_tail        = kwargs.get('track_tail', True)
    n_tail_points     = kwargs.get('n_tail_points', 30)
    tail_crop         = kwargs.get('tail_crop', None)
    adjust_thresholds = kwargs.get('adjust_thresholds', True)
    eye_resize_factor = kwargs.get('eye_resize_factor', 1)
    interpolation     = kwargs.get('interpolation', cv2.INTER_NEAREST)

    if crop != None and offset != None:
        # edit crop & offset to take into account the shrink factor
        crop   = (round(crop[0]*shrink_factor), round(crop[1]*shrink_factor))
        offset = (round(offset[0]*shrink_factor), round(offset[1]*shrink_factor))

    min_tail_eye_dist *= shrink_factor

    # open the video
    try:
        cap = FFMPEG_VideoReader(video_path, True)
        cap.initialize()
    except:
        print("Error: Could not open video.")
        return None

    # get number of frames & fps
    n_frames = ffmpeg_parse_infos(video_path)["video_nframes"]
    fps      = ffmpeg_parse_infos(video_path)["video_fps"]

    print("Original video fps: {0}. Number of frames: {1}.".format(fps, n_frames))

    if frame_num >= n_frames:
        # frame number is too high; end here
        print("Error: Frame {} exceeds total number of frames in the video.".format(frame_num))
        return

    # get corresponding frame
    image = cap.get_frame(frame_num/fps)

    # convert to greyscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if invert:
        # invert the image
        image = (255 - image)

    # shrink & crop the image
    shrunken_image = shrink_image(image, shrink_factor)
    cropped_image  = crop_image(shrunken_image, offset, crop)

    # get thresholded images
    head_threshold_image = get_head_threshold_image(cropped_image, head_threshold)
    tail_threshold_image = get_tail_threshold_image(cropped_image, tail_threshold)

    # save various images
    cv2.imwrite(os.path.join(tracking_dir, "original_image_{}.png".format(frame_num)), image)
    cv2.imwrite(os.path.join(tracking_dir, "cropped_image_{}.png".format(frame_num)), cropped_image)
    cv2.imwrite(os.path.join(tracking_dir, "head_threshold_image_{}.png".format(frame_num)), head_threshold_image*255)
    cv2.imwrite(os.path.join(tracking_dir, "tail_threshold_image_{}.png".format(frame_num)), tail_threshold_image*255)

    # track the image
    (tail_coords, spline_coords,
     eye_coords, heading_coords, skeleton_matrix) = track_image(cropped_image,
                                                             head_threshold, tail_threshold,
                                                             min_tail_eye_dist, max_tail_eye_dist,
                                                             track_head, track_tail,
                                                             n_tail_points, tail_crop,
                                                             adjust_thresholds,
                                                             eye_resize_factor, interpolation)

    if skeleton_matrix != None:
        # save tail skeleton image
        cv2.imwrite(os.path.join(tracking_dir, "skeleton_image_{}.png".format(frame_num)), skeleton_matrix*255)

    # add tracked points to the image
    tracked_image = add_tracking_to_image(cropped_image,
                                          tail_coords, spline_coords,
                                          eye_coords, heading_coords)

    # save the tracked image
    cv2.imwrite(os.path.join(tracking_dir, "tracked_image_{}.png".format(frame_num)), tracked_image)

def open_and_track_video_frames(video_path, tracking_dir,
    first_frame_num, last_frame_num, **kwargs):
    """
    Open & track a range of frames from a video.
    
    Args:
        video_path         (str): path to the video from which to track a frame.
        tracking_dir       (str): directory in which to save tracking data.
        first_frame_num    (int): first frame number to track.
        last_frame_num     (int): last frame number to track.
        ---
        crop      (int y, int x): height & width of crop area.
        offset    (int y, int x): coordinates at which to begin the crop.
        shrink_factor    (float): factor to use to shrink the image (0 - 1).
        invert            (bool): whether to invert the image.
        min_tail_eye_dist  (int): minimum distance between the eyes & start of the tail.
        max_tail_eye_dist  (int): maximum distance between the eyes & start of the tail.
        head_threshold     (int): brightness threshold to use to find the head (0 - 255).
        tail_threshold     (int): brightness threshold to use to find the tail (0 - 255).
        track_head        (bool): whether to track the head.
        track_tail        (bool): whether to track the tail.
        n_tail_points      (int): # of coordinates to keep when tracking the tail.
        tail_crop (int y, int x): height & width of crop area for tail tracking.
        adjust_thresholds (bool): whether to adjust thresholds if tracking fails.
    """

    start_time = time.time()

    crop              = kwargs.get('crop', None)
    offset            = kwargs.get('offset', None)
    shrink_factor     = kwargs.get('shrink_factor', 1)
    invert            = kwargs.get('invert', False)
    min_tail_eye_dist = kwargs.get('min_tail_eye_dist', 15)
    max_tail_eye_dist = kwargs.get('max_tail_eye_dist', 30)
    head_threshold    = kwargs.get('head_threshold')
    tail_threshold    = kwargs.get('tail_threshold')
    track_head        = kwargs.get('track_head', True)
    track_tail        = kwargs.get('track_tail', True)
    n_tail_points     = kwargs.get('n_tail_points', 30)
    tail_crop         = kwargs.get('tail_crop', None)
    adjust_thresholds = kwargs.get('adjust_thresholds', True)
    eye_resize_factor = kwargs.get('eye_resize_factor', 1)
    interpolation     = kwargs.get('interpolation', cv2.INTER_NEAREST)

    if crop != None and offset != None:
        # edit crop & offset to take into account the shrink factor
        crop   = (round(crop[0]*shrink_factor), round(crop[1]*shrink_factor))
        offset = (round(offset[0]*shrink_factor), round(offset[1]*shrink_factor))

    min_tail_eye_dist *= shrink_factor

    # get number of frames & fps
    try:
        cap = FFMPEG_VideoReader(video_path, True)
        cap.initialize()
    except:
        print("Error: Could not open video.")
        return None

    # get original video info
    n_frames = ffmpeg_parse_infos(video_path)["video_nframes"]
    fps      = ffmpeg_parse_infos(video_path)["video_fps"]

    print("Original video fps: {0}. Number of frames: {1}.".format(fps, n_frames))

    for frame_num in range(first_frame_num, last_frame_num+1):
        if frame_num >= n_frames:
            # frame number is too high; end here
            print("Error: Frame {} exceeds total number of frames in the video.".format(frame_num))
            return

        # get corresponding frame
        image = cap.get_frame(frame_num/fps)

        # convert to greyscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if invert:
            # invert the image
            image = (255 - image)

        # shrink & crop the image
        shrunken_image = shrink_image(image, shrink_factor)
        cropped_image  = crop_image(shrunken_image, offset, crop)

        # get thresholded images
        head_threshold_image = get_head_threshold_image(cropped_image, head_threshold)
        tail_threshold_image = get_tail_threshold_image(cropped_image, tail_threshold)

        # save various images
        cv2.imwrite(os.path.join(tracking_dir, "original_image_{}.png".format(frame_num)), image)
        cv2.imwrite(os.path.join(tracking_dir, "cropped_image_{}.png".format(frame_num)), cropped_image)
        cv2.imwrite(os.path.join(tracking_dir, "head_threshold_image_{}.png".format(frame_num)), head_threshold_image*255)
        cv2.imwrite(os.path.join(tracking_dir, "tail_threshold_image_{}.png".format(frame_num)), tail_threshold_image*255)

        # track the image
        (tail_coords, spline_coords,
         eye_coords, heading_coords, skeleton_matrix) = track_image(cropped_image,
                                                                 head_threshold, tail_threshold,
                                                                 min_tail_eye_dist, max_tail_eye_dist,
                                                                 track_head, track_tail,
                                                                 n_tail_points, tail_crop,
                                                                 adjust_thresholds,
                                                                 eye_resize_factor, interpolation)

        if skeleton_matrix != None:
            # save tail skeleton image
            cv2.imwrite(os.path.join(tracking_dir, "skeleton_image_{}.png".format(frame_num)), skeleton_matrix*255)

        # add tracked points to the image
        tracked_image = add_tracking_to_image(cropped_image,
                                              tail_coords, spline_coords,
                                              eye_coords, heading_coords)

        # save the tracked image
        cv2.imwrite(os.path.join(tracking_dir, "tracked_image_{}.png".format(frame_num)), tracked_image)

def track_image(image, head_threshold, tail_threshold,
    min_tail_eye_dist, max_tail_eye_dist, track_head, track_tail, n_tail_points,
    tail_crop, adjust_thresholds, eye_resize_factor, interpolation, prev_eye_coords=None):
    """
    Track the given image.
    
    Args:
        image         (2d array): grayscale image array.
        head_threshold     (int): brightness threshold to use to find the head (0 - 255).
        tail_threshold     (int): brightness threshold to use to find the tail (0 - 255).
        min_tail_eye_dist  (int): minimum distance between the eyes & start of the tail.
        max_tail_eye_dist  (int): maximum distance between the eyes & start of the tail.
        track_head        (bool): whether to track the head.
        track_tail        (bool): whether to track the tail.
        n_tail_points      (int): # of coordinates to keep when tracking the tail.
        tail_crop (int y, int x): height & width of crop area for tail tracking.
        adjust_thresholds (bool): whether to adjust thresholds if tracking fails.

    Returns:
        tail_coords     (2d array): array of tail coordinates - size (2, n_tail_points).
        spline_coords   (2d array): array of fitted spline coordinates - size (2, n_tail_points).
        eye_coords      (2d array): array of eye coordinates - size (2, 2).
        heading_coords     (2d array): array of eye bisector coordinates - size (2, 2).
        skeleton_matrix (2d array): array showing the skeleton of the tail (same size as image).
    """

    # print("yoo", interpolation)

    if eye_resize_factor != 1:
        orig_image = image.copy()

    #     print("yoo")
        image = cv2.resize(image, (0, 0), fx=eye_resize_factor, fy=eye_resize_factor, interpolation=interpolation)

    #     print("yoo")
    if track_head:
        head_threshold_orig = head_threshold

        head_threshold_image = get_head_threshold_image(image, head_threshold)

        # get head coordinates
        (eye_coords, heading_coords) = get_head_coords(head_threshold_image, eye_resize_factor=eye_resize_factor, prev_eye_coords=prev_eye_coords)

        if eye_coords == None and adjust_thresholds: # incorrect # of eyes found
            i = 0

            while eye_coords == None:
                if i < 4:
                    # increase head threshold & try again
                    head_threshold += 1

                    head_threshold_image = get_head_threshold_image(image, head_threshold)

                    (eye_coords, heading_coords) = get_head_coords(head_threshold_image, eye_resize_factor=eye_resize_factor, prev_eye_coords=prev_eye_coords)

                    i += 1
                elif i < 8:
                    if i == 2:
                        # reset to original threshold
                        head_threshold = head_threshold_orig

                    head_threshold -= 1

                    head_threshold_image = get_head_threshold_image(image, head_threshold)

                    (eye_coords, heading_coords) = get_head_coords(head_threshold_image, eye_resize_factor=eye_resize_factor, prev_eye_coords=prev_eye_coords)
                    i += 1
                else:
                    break
    else:
        (eye_coords, heading_coords) = [None]*2

    # print("yoo")

    if eye_resize_factor != 1:
        image = orig_image

    if track_head and eye_coords == None:
        # don't bother tracking the tail; end here
        return [None]*5

    if track_tail:
        if eye_coords != None:
            # get coordinates of the midpoint of the eyes
            mid_coords = [(eye_coords[0, 0] + eye_coords[0, 1])/2.0, (eye_coords[1, 0] + eye_coords[1, 1])/2.0]

            # create array of tail crop coords: [ [y_start, y_end],
            #                                     [x_start, x_end] ]
            if tail_crop == None:
                tail_crop_coords = np.array([[0, image.shape[0]], [0, image.shape[1]]])
            else:
                tail_crop_coords = np.array([[np.maximum(0, mid_coords[0]-tail_crop[0]), np.minimum(image.shape[0], mid_coords[0]+tail_crop[0])],
                                             [np.maximum(0, mid_coords[1]-tail_crop[1]), np.minimum(image.shape[1], mid_coords[1]+tail_crop[1])]])
            
            rel_eye_coords  = (eye_coords.T - tail_crop_coords[:, 0]).T

            tail_crop_image = image[tail_crop_coords[0, 0]:tail_crop_coords[0, 1], tail_crop_coords[1, 0]:tail_crop_coords[1, 1]]
        else:
            rel_eye_coords  = eye_coords
            tail_crop_image = image
        
        tail_threshold_image = get_tail_threshold_image(tail_crop_image, tail_threshold)

        # track tail
        (tail_coords, spline_coords, skeleton_matrix) = get_tail_coords(tail_threshold_image, rel_eye_coords,
                                                                        min_tail_eye_dist, max_tail_eye_dist,
                                                                        n_tail_points=30)

        if tail_coords == None and adjust_thresholds: # tail wasn't able to be tracked
            tail_threshold_orig = tail_threshold
            i = 0

            while tail_coords == None:
                if i < 4:
                    # increase tail threshold & try again
                    tail_threshold += 1

                    tail_threshold_image = get_tail_threshold_image(tail_crop_image, tail_threshold)

                    (tail_coords, spline_coords, skeleton_matrix) = get_tail_coords(tail_threshold_image, rel_eye_coords,
                                                                                    min_tail_eye_dist, max_tail_eye_dist,
                                                                                    n_tail_points)
                    i += 1
                elif i < 8:
                    if i == 2:
                        # reset to original threshold
                        tail_threshold = tail_threshold_orig

                    # decrease tail threshold & try again
                    tail_threshold -= 1

                    tail_threshold_image = get_tail_threshold_image(tail_crop_image, tail_threshold)

                    (tail_coords, spline_coords, skeleton_matrix) = get_tail_coords(tail_threshold_image, rel_eye_coords,
                                                                                    min_tail_eye_dist, max_tail_eye_dist,
                                                                                    n_tail_points)
                    i += 1
                else:
                    break
    else:
        (tail_coords, spline_coords, skeleton_matrix) = [None]*3

    if eye_coords != None and tail_coords != None:
        tail_coords   += tail_crop_coords[:, 0][:, np.newaxis].astype(int)
        spline_coords += tail_crop_coords[:, 0][:, np.newaxis].astype(int)

        tail_distances = np.sqrt((heading_coords[0, :] - spline_coords[0, 0])**2 + (heading_coords[1, :] - spline_coords[1, 0])**2)
        if tail_distances[0] < tail_distances[1]:
            heading_coords = np.fliplr(heading_coords)

    return (tail_coords, spline_coords, eye_coords, heading_coords, skeleton_matrix)

def add_tracking_to_image(image, tail_coords, spline_coords,
    eye_coords, heading_coords, pos_hist_coords=None):
    """
    Plot tracking data on top of the given image.
    
    Args:
        image           (2d array): grayscale image array.
        tail_coords     (2d array): array of tail coordinates - size (2, n_tail_points).
        spline_coords   (2d array): array of fitted spline coordinates - size (2, n_tail_points).
        eye_coords      (2d array): array of eye coordinates - size (2, 2).
        heading_coords     (2d array): array of eye bisector coordinates - size (2, 2).
        pos_hist_coords (2d array): history of head position coordinates over time T - size (T, 2).

    Returns:
        tracked_image   (3d array): RGB image with tracking added
    """

    # convert to BGR
    if len(image.shape) < 3:
        tracked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # add eye points
    if eye_coords != None:
        for i in range(2):
            cv2.circle(tracked_image, (int(round(eye_coords[1, i])), int(round(eye_coords[0, i]))), 1, (0, 0, 255), -1)

    if pos_hist_coords != None:
        hist_length = pos_hist_coords.shape[0]
        # add position history
        for i in range(hist_length-1):
            if not np.isnan(pos_hist_coords[i, 0]) and not np.isnan(pos_hist_coords[i+1, 0]):
                cv2.line(tracked_image, (int(round(pos_hist_coords[i, 1])), int(round(pos_hist_coords[i, 0]))),
                                        (int(round(pos_hist_coords[i+1, 1])), int(round(pos_hist_coords[i+1, 0]))), (255, 0, 0), 1)

    if spline_coords != None and spline_coords[0, 0] != np.nan:
        # add tail points
        spline_length = spline_coords.shape[1]
        for i in range(spline_length-1):
            if (not np.isnan(spline_coords[0, i]) and not np.isnan(spline_coords[0, i+1])
                and not np.isnan(spline_coords[1, i]) and not np.isnan(spline_coords[1, i+1])):
                cv2.line(tracked_image, (int(round(spline_coords[1, i])), int(round(spline_coords[0, i]))),
                                        (int(round(spline_coords[1, i+1])), int(round(spline_coords[0, i+1]))), (0, 255, 0), 1)

    return tracked_image

def get_centroids(head_threshold_image, eye_resize_factor=1, prev_eye_coords=None):
    """
    Find centroids in a binary threshold image.
    
    Args:
        head_threshold_image (2d array): greyscale thresholded image.

    Returns:
        centroid_coords      (2d array): array of n centroid coordinates - size (2, n).
    """

    # find contours
    try:
        image, contours, _ = cv2.findContours(head_threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        contours, _ = cv2.findContours(head_threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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

    # if prev_eye_coords != None and prev_eye_coords.shape[0] >= 10 and prev_eye_coords[-10][0] != None:
    #     for j in range(2):
    #         dy = prev_eye_coords[-10, 0, j] - centroid_coords[0, j]
    #         dx = prev_eye_coords[-10, 1, j] - centroid_coords[1, j]
    #         if np.abs(dy) < 3 and np.abs(dx) < 3:
    #             centroid_coords = prev_eye_coords[-1]

    return centroid_coords

def get_heading_coords(eye_coords, length=5):
    """
    Calculate start & end coordinates of the bisector of the line joining the given eye coordinates.

    Args:
        eye_coords  (2d array): array of eye coordinates - size (2, 2).
        length           (int): desired length of bisector.

    Returns:
        heading_coords (2d array): array of eye bisector coordinates - size (2, 2).
    """
    # get coordinates of eyes
    y_1 = eye_coords[0, 0]
    y_2 = eye_coords[0, 1]
    x_1 = eye_coords[1, 0]
    x_2 = eye_coords[1, 1]

    if y_1 == y_2 and x_1 == x_2:
        # eye coordinates are the same -- can't calculate a bisector
        return None
    elif y_1 == y_2:
        # y coordinates are the same -- make them slightly different
        # to avoid dividing by zero
        y_1 = y_2 + 1e-6
    elif x_1 == x_2:
        # x coordinates are the same -- make them slightly different
        # to avoid dividing by zero
        x_1 = x_2 + 1e-6

    # get the slope of the line segment joining the eyes
    m = (x_2 - x_1)/(y_2 - y_1)

    # get the midpoint of the line segment joining the eyes
    x_mid = (x_2 + x_1)/2.0
    y_mid = (y_2 + y_1)/2.0

    # get the slope of the perpendicular line
    k = -1.0/m

    # create the endpoint of a unit vector with this slope
    y_unit = 1
    x_unit = k*y_unit

    # get the endpoints of scaled vectors pointing in each direction
    y_heading_1 = length*y_unit/np.sqrt(y_unit**2 + x_unit**2)
    x_heading_1 = length*x_unit/np.sqrt(y_unit**2 + x_unit**2)

    y_heading_2 = -length*y_unit/np.sqrt(y_unit**2 + x_unit**2)
    x_heading_2 = -length*x_unit/np.sqrt(y_unit**2 + x_unit**2)

    # add vector endpoints to the midpoint of the eyes
    heading_coords = np.array([[y_mid + y_heading_1, y_mid + y_heading_2],
                            [x_mid + x_heading_1, x_mid + x_heading_2]])

    return heading_coords

def get_head_coords(head_threshold_image, eye_resize_factor, prev_eye_coords=None, min_intereye_dist=3, max_intereye_dist=10):
    """
    Calculate eye coordinates & perpendicular bisector coordinates using a thresholded image.

    Args:
        head_threshold_image (2d array): greyscale thresholded image.
        min_intereye_dist         (int): minimum distance between the eyes.
        max_intereye_dist         (int): maximum distance between the eyes.

    Returns:
        eye_coords           (2d array): array of eye coordinates - size (2, 2).
        heading_coords          (2d array): array of eye bisector coordinates - size (2, 2).
    """
    # get eye centroids
    centroid_coords = get_centroids(head_threshold_image, eye_resize_factor, prev_eye_coords=prev_eye_coords)

    if centroid_coords == None:
        # no centroids found; end here.
        return [None]*2

    # get the number of found eye centroids
    n_centroids = centroid_coords.shape[1]

    # get all permutations of pairs of centroid indices
    perms = itertools.permutations(np.arange(n_centroids), r=2)

    for p in list(perms):
        eye_coords = np.array([[centroid_coords[0, p[0]], centroid_coords[0, p[1]]],
                               [centroid_coords[1, p[0]], centroid_coords[1, p[1]]]])

        intereye_dist = np.sqrt((eye_coords[0, 1] - eye_coords[0, 0])**2 + (eye_coords[1, 1] - eye_coords[1, 0])**2)

        if not (intereye_dist < min_intereye_dist or intereye_dist > max_intereye_dist):
            # eye coords fit the criteria of min & max distance; stop looking.
            break

    # get coords of perpendicular bisector
    heading_coords = get_heading_coords(eye_coords)

    return eye_coords, heading_coords

def get_tail_coords(tail_threshold_image, eye_coords=None, min_tail_eye_dist=None,
    max_tail_eye_dist=None, max_l=9, n_tail_points=30, smoothing_factor=30):
    """
    Calculate tail coordinates & cubic spline using a thresholded image.

    Args:
        tail_threshold_image (2d array): greyscale thresholded image.
        eye_coords           (2d array): array of eye coordinates - size (2, 2). (Optional)
        min_tail_eye_dist         (int): minimum distance between the eyes & *start* of the tail. (Optional)
        max_tail_eye_dist         (int): maximum distance between the eyes & *start* of the tail. (Optional)
        max_l                     (int): maximum side length of area to look in for subsequent points.
        n_tail_points             (int): # of coordinates to keep when tracking the tail.
        smoothing_factor          (int): smoothing factor for cubic spline.

    Returns:
        tail_coords          (2d array): array of tail coordinates - size (2, n_tail_points).
        spline_coords        (2d array): array of fitted spline coordinates - size (2, n_tail_points).
        skeleton_matrix      (2d array): array showing the skeleton of the tail (same size as image).
    """
    # get size of thresholded image
    y_size = tail_threshold_image.shape[0]
    x_size = tail_threshold_image.shape[1]

    # get coordinates of nonzero points of thresholded image
    nonzeros = np.nonzero(tail_threshold_image)

    if eye_coords != None:
        # get coordinates of the midpoint of the eyes
        mid_coords = [(eye_coords[0, 0] + eye_coords[0, 1])/2.0, (eye_coords[1, 0] + eye_coords[1, 1])/2.0]

        # zero out pixels that are close to the eyes
        for (r, c) in zip(nonzeros[0], nonzeros[1]):
            if np.sqrt((r - mid_coords[0])**2 + (c - mid_coords[1])**2) < min_tail_eye_dist:
                tail_threshold_image[r, c] = 0
    else:
        mid_coords = None

    # get tail skeleton matrix
    skeleton_matrix = bwmorph_thin(tail_threshold_image, n_iter=np.inf).astype(np.uint8)

    # get an ordered list of coordinates of the tail, from one end to the other
    tail_coords = get_ordered_tail_coords(skeleton_matrix, max_l, mid_coords, max_tail_eye_dist)

    if tail_coords == None:
        # couldn't get tail coordinates; end here.
        return [None]*2 + [skeleton_matrix]

    if tail_coords.shape[1] > n_tail_points:
        # get evenly spaced tail indices
        r = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        frame_nums = [0] + r(n_tail_points-2, tail_coords.shape[1]) + [tail_coords.shape[1]-1]

        tail_coords = tail_coords[:, frame_nums]

    # get number of tail coordinates
    n_tail_coords = tail_coords.shape[1]

    # modify tail skeleton coordinates (Huang et al., 2013)
    for i in range(n_tail_coords):
        y = tail_coords[0, i]
        x = tail_coords[1, i]

        pixel_sum = 0
        y_sum     = 0
        x_sum     = 0

        for k in range(-1, 2):
            for l in range(-1, 2):
                pixel_sum += tail_threshold_image[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]
                y_sum     += k*tail_threshold_image[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]
                x_sum     += l*tail_threshold_image[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]

        if pixel_sum != 0:
            y_sum /= pixel_sum
            x_sum /= pixel_sum
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
        return [None]*2 + [skeleton_matrix]

    # get number of spline coordinates
    n_spline_coords = spline_coords.shape[1]

    if n_tail_coords > n_tail_points:
        # get evenly spaced tail indices
        r = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        frame_nums = [0] + r(n_tail_points-2, tail_coords.shape[1]) + [tail_coords.shape[1]-1]

        tail_coords = tail_coords[:, frame_nums]

    if n_spline_coords > n_tail_points:
        # get evenly spaced spline indices
        r = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        frame_nums = [0] + r(n_tail_points-2, spline_coords.shape[1]) + [spline_coords.shape[1]-1]

        spline_coords = spline_coords[:, frame_nums]

    return tail_coords, spline_coords, skeleton_matrix

def get_ordered_tail_coords(skeleton_matrix, max_l, mid_coords, max_tail_eye_dist, min_n_tail_points=30):
    """
    Walk through points of a tail to get an ordered array of coordinates (from head to end of tail).

    Args:
        skeleton_matrix     (2d array): array showing the skeleton of the tail (same size as image).
        max_l                    (int): maximum side length of area to look in for subsequent points.
        mid_coords          (1d array): (x, y) coordinate of midpoint of eyes. (Optional)
        max_tail_eye_dist        (int): maximum distance between the eyes & *start* of the tail.
        min_n_tail_points        (int): minimum acceptable number of tail points.

    Returns:
        ordered_tail_coords (2d array): array of ordered n tail coordinates - size (2, n).
    """
    # get size of matrix
    y_size = skeleton_matrix.shape[0]
    x_size = skeleton_matrix.shape[1]

    # get coordinates of nonzero points of skeleton matrix
    nonzeros = np.nonzero(skeleton_matrix)

    # initialize tail starting point coordinates
    starting_coords = None

    # initialize variable for storing the closest distance to the midpoint of the eyes
    closest_eye_dist = None

    if mid_coords != None:
        # loop through all nonzero points of the tail skeleton
        for (r, c) in zip(nonzeros[0], nonzeros[1]):
            # get distance of this point to the midpoint of the eyes
            eye_dist = np.sqrt((r - mid_coords[0])**2 + (c - mid_coords[1])**2)

            # look for an endpoint near the eyes
            if eye_dist < max_tail_eye_dist:
                if (closest_eye_dist == None) or (closest_eye_dist != None and eye_dist < closest_eye_dist):
                    # either no point has been found yet or this is a closer point than the previous best

                    # get nonzero elements in 3x3 neigbourhood around the point
                    nonzero_neighborhood = skeleton_matrix[r-1:r+2, c-1:c+2] != 0

                    # if the number of non-zero points in the neighborhood is at least 2
                    # (ie. there's at least one direction to move in along the tail),
                    # set this to our tail starting point.
                    if np.sum(nonzero_neighborhood) >= 2:
                        starting_coords = np.array([r, c])
                        closest_eye_dist = eye_dist
    else:
        # eyes aren't being tracked; just find any starting point
        for (r, c) in zip(nonzeros[0], nonzeros[1]):
            # get nonzero elements in 3x3 neigbourhood around the point
            nonzero_neighborhood = skeleton_matrix[r-1:r+2, c-1:c+2] != 0

            # if the number of non-zero points in the neighborhood is 2
            # (ie. there's only one direction to move in along the tail),
            # set this to our tail starting point.
            if np.sum(nonzero_neighborhood) == 2:
                starting_coords = np.array([r, c])

    if starting_coords == None:
        # still could not find start of the tail; end here.
        return None

    # walk along the tail
    found_coords = walk_along_tail(starting_coords, max_l, skeleton_matrix)

    if len(found_coords) < min_n_tail_points:
        # we didn't manage to get the full tail; try moving along the tail in reverse
        found_coords = walk_along_tail(found_coords[-1], max_l, skeleton_matrix)

    if len(found_coords) < min_n_tail_points:
        # we still didn't get enough tail points; give up here.
        return None

    # convert to an array
    found_coords = np.array(found_coords).T

    return found_coords

# --- HELPER FUNCTIONS --- #

def find_unique_coords(coords, found_coords):
    """
    Find coordinates in coords that are 'unique' -- ie. not in found_coords.

    Args:
        coords        (2d array): array of n coordinates to look at - size (2, n).
        found_coords      (list): list of found coordinates to check against.

    Returns:
        unique_coords (2d array): array of m coordinates that are not in found_coords - size (2, m).
    """

    unique_coords = []

    for i in range(coords.shape[1]):
        c = coords[:, i]
        if not any((c == x).all() for x in found_coords):
            # found a unique coordinate
            unique_coords.append(c)

    if len(unique_coords) == 0:
        # no unique points found; end here.
        return None

    unique_coords = np.array(unique_coords).T

    return unique_coords

def find_next_tail_coords_in_neighborhood(found_coords, l, skeleton_matrix):
    """
    Find the next point of the tail in the neighbourhood of coord.

    Args:
        found_coords        (list): list of found coordinates.
        l                    (int): side length of the square area around coord to look in for the next point. Must be 1 + some multiple of 2.
        skeleton_matrix (2d array): array showing the skeleton of the tail (same size as image).

    Returns:
        next_coords     (1d array): next coordinates of the tail.
    """

    if (l-1) % 2 != 0:
        print("Error: l-1={} is not divisible by 2.".format(l-1))
        return None

    # get half of side length
    r = int((l-1)/2)

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

    # translate these coordinates to image coordinates
    diff_x = r - x
    diff_y = r - y
    nonzero_y_coords -= diff_y
    nonzero_x_coords -= diff_x

    nonzeros = np.array([nonzero_y_coords, nonzero_x_coords])

    # find the next point
    unique_coords = find_unique_coords(nonzeros, found_coords)

    if unique_coords == None:
        return None

    # find closest point to the last found coordinate
    distances = np.sqrt((unique_coords[0, :] - found_coords[-1][0])**2 + (unique_coords[1, :] - found_coords[-1][1])**2)
    closest_index = np.argmin(distances)

    next_coords = unique_coords[:, closest_index]

    return next_coords

def walk_along_tail(starting_coords, max_l, skeleton_matrix):
    """
    Walk along the tail until no new coordinates are found.

    Args:
        starting_coords (1d array): (x, y) coordinates of the starting point.
        max_l                (int): maximum side length of area to look in for subsequent points.
        skeleton_matrix (2d array): array showing the skeleton of the tail (same size as image).

    Returns:
        found_coords        (list): list of found coordinates.
    """
    # initialize found coords list
    found_coords = [starting_coords]

    # initialize side length of square area to look in for the next point
    l = 3

    while l <= max_l and len(found_coords) < 200:
        # find coordinates of the next point
        next_coords = find_next_tail_coords_in_neighborhood(found_coords, l, skeleton_matrix)

        if next_coords != None:
            # add coords to found coords list
            found_coords.append(next_coords)

            # reset the area size -- next iteration of loop will look for next point
            l = 3
        else:
            # could not find a point; increase the area to look in
            l += 2

    return found_coords

def crop_image(image, offset, crop):
    if offset != None and crop != None:
        return image[offset[0]:offset[0] + crop[0], offset[1]:offset[1] + crop[1]]
    else:
        return image

def shrink_image(image, shrink_factor):
    if shrink_factor != 1:
        image = cv2.resize(image, (0, 0), fx=shrink_factor, fy=shrink_factor)
    return image

def get_head_threshold_image(image, head_threshold):
    _, head_threshold_image = cv2.threshold(image, head_threshold, 255, cv2.THRESH_BINARY_INV)
    np.divide(head_threshold_image, 255, out=head_threshold_image, casting='unsafe')
    return head_threshold_image

def get_tail_threshold_image(image, tail_threshold):
    _, tail_threshold_image = cv2.threshold(image, tail_threshold, 255, cv2.THRESH_BINARY_INV)
    np.divide(tail_threshold_image, 255, out=tail_threshold_image, casting='unsafe')
    return tail_threshold_image

def get_tail_skeleton_image(tail_threshold_image):
    return bwmorph_thin(tail_threshold_image, n_iter=np.inf).astype(np.uint8)

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
