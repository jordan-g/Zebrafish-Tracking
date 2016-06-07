import numpy as np
import cv2
from scipy import interpolate

import os
import re

from moviepy.video.io.ffmpeg_reader import *
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

from bwmorph_thin import bwmorph_thin
import analysis as an

import pdb

# initialize tracking variables
last_eye_y_coords = None
last_eye_x_coords = None
head_threshold = 0
tail_threshold = 0

def crop_image(image, offset, crop):
    if offset != None and crop != None:
        return image[offset[0]:offset[0] + crop[0], offset[1]:offset[1] + crop[1]]
    else:
        return image

def shrink_image(image, shrink_factor):
    if shrink_factor != 1:
        image = cv2.resize(image, (0,0), fx=shrink_factor, fy=shrink_factor)
    return image

def get_head_threshold_image(image, head_threshold):
    _, head_threshold_image = cv2.threshold(image, head_threshold, 255, cv2.THRESH_BINARY_INV)
    head_threshold_image /= 255
    return head_threshold_image

def get_tail_threshold_image(image, tail_threshold):
    _, tail_threshold_image = cv2.threshold(image, tail_threshold, 255, cv2.THRESH_BINARY_INV)
    tail_threshold_image /= 255
    return tail_threshold_image

def track_image(old_image_path, new_image_path, **kwargs):
    crop                  = kwargs.get('crop', None)
    offset                = kwargs.get('offset', None)
    shrink_factor         = kwargs.get('shrink_factor', 1)
    invert                = kwargs.get('invert', False)
    min_eye_distance      = kwargs.get('min_eye_distance', 0)
    eye_1_index           = kwargs.get('eye_1_index', 0)
    eye_2_index           = kwargs.get('eye_2_index', 1)
    head_threshold        = kwargs.get('head_threshold')
    tail_threshold        = kwargs.get('tail_threshold')
    track_head_bool       = kwargs.get('track_head_bool', True)
    track_tail_bool       = kwargs.get('track_tail_bool', True)
    
    # load the original image
    image = load_image(old_image_path)

    if crop != None and offset != None:
        # edit crop & offset to take into account the shrink factor
        crop   = (round(crop[0]*shrink_factor), round(crop[1]*shrink_factor))
        offset = (round(offset[0]*shrink_factor), round(offset[1]*shrink_factor))

    if invert:
        # invert the image
        image = (255 - image)

    # shrink & crop the image
    shrunken_image = shrink_image(image, shrink_factor)
    cropped_image = crop_image(shrunken_image, offset, crop)

    # get thresholded images
    head_threshold_image = get_head_threshold_image(cropped_image, head_threshold)
    tail_threshold_image = get_tail_threshold_image(cropped_image, tail_threshold)

    # track the image
    (tail_y_coords, tail_x_coords, spline_y_coords, spline_x_coords,
        eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords) = track_frame(cropped_image,
                                                                                head_threshold, tail_threshold, 
                                                                                head_threshold_image, tail_threshold_image,
                                                                                min_eye_distance*shrink_factor,
                                                                                eye_1_index, eye_2_index,
                                                                                track_head_bool, track_tail_bool)
    
    # add tracked points to the image
    image = plot_image(cropped_image,
                    tail_y_coords, tail_x_coords,
                    spline_y_coords, spline_x_coords,
                    eye_y_coords, eye_x_coords,
                    perp_y_coords, perp_x_coords)

    # save the new image
    cv2.imwrite(new_image_path, image)

def track_folder(old_folder_path, new_video_path, **kwargs):
    crop                  = kwargs.get('crop', None)
    offset                = kwargs.get('offset', None)
    shrink_factor         = kwargs.get('shrink_factor', 1)
    invert                = kwargs.get('invert', False)
    min_eye_distance      = kwargs.get('min_eye_distance', 0)
    eye_1_index           = kwargs.get('eye_1_index', 0)
    eye_2_index           = kwargs.get('eye_2_index', 1)
    head_threshold        = kwargs.get('head_threshold')
    tail_threshold        = kwargs.get('tail_threshold')
    track_head_bool       = kwargs.get('track_head_bool', True)
    track_tail_bool       = kwargs.get('track_tail_bool', True)
    save_video            = kwargs.get('save_video', True)
    plot_heading_angle    = kwargs.get('plot_heading_angle', False)
    new_video_fps         = kwargs.get('new_video_fps', 30)
    new_video_size_factor = kwargs.get('new_video_size_factor', 1)
    
    # load all frames from the folder
    frames = load_folder(old_folder_path)

    n_frames = len(frames)

    # initialize tracking data arrays
    eye_y_coords_array      = np.zeros((n_frames, 2))
    eye_x_coords_array      = np.zeros((n_frames, 2))
    perp_y_coords_array     = np.zeros((n_frames, 2))
    perp_x_coords_array     = np.zeros((n_frames, 2))
    tail_end_y_coords_array = np.zeros((n_frames, 2))
    tail_end_x_coords_array = np.zeros((n_frames, 2))

    if crop != None and offset != None:
        # edit crop & offset to take into account the shrink factor
        crop   = (round(crop[0]*shrink_factor), round(crop[1]*shrink_factor))
        offset = (round(offset[0]*shrink_factor), round(offset[1]*shrink_factor))

    def track_frame_at_number(frame_number):
        global last_eye_y_coords, last_eye_x_coords

        # get corresponding frame
        image = frames[frame_number]

        if invert:
            # invert the image
            image = (255 - image)

        # shrink & crop the image
        shrunken_image = shrink_image(image, shrink_factor)
        cropped_image = crop_image(shrunken_image, offset, crop)

        # get thresholded images
        head_threshold_image = get_head_threshold_image(cropped_image, head_threshold)
        tail_threshold_image = get_tail_threshold_image(cropped_image, tail_threshold)

        # track the image
        (tail_y_coords, tail_x_coords, spline_y_coords, spline_x_coords,
            eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords) = track_frame(cropped_image,
                                                                                    head_threshold, tail_threshold, 
                                                                                    head_threshold_image, tail_threshold_image,
                                                                                    min_eye_distance*shrink_factor,
                                                                                    eye_1_index, eye_2_index,
                                                                                    track_head_bool, track_tail_bool)

        # update last found coordinates of the eyes
        last_eye_y_coords = eye_y_coords
        last_eye_x_coords = eye_x_coords

        # add tracked coordinates to arrays
        if tail_y_coords is not None and tail_x_coords is not None:
            tail_end_y_coords_array[frame_number, :] = [tail_y_coords[0], tail_y_coords[-1]]
            tail_end_x_coords_array[frame_number, :] = [tail_x_coords[0], tail_x_coords[-1]]

        if eye_y_coords is not None and eye_x_coords is not None:
            eye_y_coords_array[frame_number, :] = eye_y_coords
            eye_x_coords_array[frame_number, :] = eye_x_coords

        if perp_y_coords is not None and perp_x_coords is not None:
            perp_y_coords_array[frame_number, :] = perp_y_coords
            perp_x_coords_array[frame_number, :] = perp_x_coords

        if save_video:
            # add tracked points to the image
            image = plot_image(cropped_image,
                            tail_y_coords, tail_x_coords,
                            spline_y_coords, spline_x_coords,
                            eye_y_coords, eye_x_coords,
                            perp_y_coords, perp_x_coords, (perp_y_coords_array[:frame_number, 0]+perp_y_coords_array[:frame_number, 1])/2.0, (perp_x_coords_array[:frame_number, 0] + perp_x_coords_array[:frame_number, 1])/2.0)

            # resize image
            image = shrink_image(image, new_video_size_factor)

            # add dynamic plot of heading angle
            if plot_heading_angle:
                # initialize plot image
                heading_plot_image = np.ones((90, image.shape[1], 3))*255
                
                # get array of angles
                angle_array = np.nan_to_num(an.get_heading_angle(None, False, perp_y_coords_array, perp_x_coords_array))

                # ratio of total # of plot points to the pixel width of the plot
                pixel_ratio = float(n_frames)/image.shape[1]

                # draw horizontal line at the top
                cv2.line(heading_plot_image, (0, 0), (image.shape[1], 0), (0, 0, 0), 1)

                # add description text
                cv2.putText(heading_plot_image, "Heading Angle", (0, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1, cv2.CV_AA)

                # draw a horizontal line in the middle
                cv2.line(heading_plot_image, (0, 45), (image.shape[1], 45), (160, 160, 160), 1)

                # add plot points
                for i in range(frame_number):
                    cv2.line(heading_plot_image, (int(round(i/pixel_ratio)), int(round(angle_array[i]/2.0))+45), (int(round((i+1)/pixel_ratio)), int(round(angle_array[i+1]/2.0))+45), (0, 0, 255), 1)

                # label y axis
                cv2.putText(heading_plot_image, "0", (0, 47), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
                cv2.putText(heading_plot_image, "90", (0, 8), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
                cv2.putText(heading_plot_image, "-90", (0, 89), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)

                # get sizes of the plot & the video frame
                h1, w1 = image.shape[:2]
                h2, w2 = heading_plot_image.shape[:2]

                # create an empty matrix to store the final combined frame
                vis = np.zeros((h1+h2, max(w1, w2),3), np.uint8)

                # combine the video frame & the plot
                vis[:h1, :w1,:3] = image
                vis[h1:h1+h2, :w2,:3] = heading_plot_image

                image = vis

            return image

    def track_frame_at_time(t):
        # get corresponding image for time t
        frame_number = int(t*new_video_fps)

        new_frame = track_frame_at_number(frame_number)
        
        return new_frame

    if save_video:
        animation = mpy.VideoClip(track_frame_at_time, duration=n_frames/new_video_fps)
        animation.write_videofile(new_video_path, codec='libx264', fps=new_video_fps)
    else:
        for frame_number in range(n_frames):
            track_frame_at_number(frame_number)

    # save tracked coordinates
    save_dir = os.path.join(os.path.dirname(new_video_path), "Tracking Data/")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savetxt(os.path.join(save_dir, "eye_y_coords_array.csv"), eye_y_coords_array)
    np.savetxt(os.path.join(save_dir, "eye_x_coords_array.csv"), eye_x_coords_array)
    np.savetxt(os.path.join(save_dir, "perp_y_coords_array.csv"), perp_y_coords_array)
    np.savetxt(os.path.join(save_dir, "perp_x_coords_array.csv"), perp_x_coords_array)
    np.savetxt(os.path.join(save_dir, "tail_end_y_coords_array.csv"), tail_end_y_coords_array)
    np.savetxt(os.path.join(save_dir, "tail_end_x_coords_array.csv"), tail_end_x_coords_array)

def track_video(old_video_path, new_video_path, **kwargs):
    crop                  = kwargs.get('crop', None)
    offset                = kwargs.get('offset', None)
    shrink_factor         = kwargs.get('shrink_factor', 1)
    invert                = kwargs.get('invert', False)
    min_eye_distance      = kwargs.get('min_eye_distance', 0)
    eye_1_index           = kwargs.get('eye_1_index', 0)
    eye_2_index           = kwargs.get('eye_2_index', 1)
    head_threshold        = kwargs.get('head_threshold')
    tail_threshold        = kwargs.get('tail_threshold')
    track_head_bool       = kwargs.get('track_head_bool', True)
    track_tail_bool       = kwargs.get('track_tail_bool', True)
    save_video            = kwargs.get('save_video', True)
    plot_heading_angle    = kwargs.get('plot_heading_angle', False)
    new_video_fps         = kwargs.get('new_video_fps', 30)
    new_video_size_factor = kwargs.get('new_video_size_factor', 1)

    # open the original video
    try:
        cap = FFMPEG_VideoReader(old_video_path, True)
        cap.initialize()
    except:
        print("Error: Could not open video.")
        return None

    # get original video info
    n_frames = ffmpeg_parse_infos(old_video_path)["video_nframes"]
    old_video_fps = ffmpeg_parse_infos(old_video_path)["video_fps"]

    print("Original video fps: {0}. n_frames: {1}".format(old_video_fps, n_frames))

    # initialize tracking data arrays
    eye_y_coords_array      = np.zeros((n_frames, 2))
    eye_x_coords_array      = np.zeros((n_frames, 2))
    perp_y_coords_array     = np.zeros((n_frames, 2))
    perp_x_coords_array     = np.zeros((n_frames, 2))
    tail_end_y_coords_array = np.zeros((n_frames, 2))
    tail_end_x_coords_array = np.zeros((n_frames, 2))

    if crop != None and offset != None:
        # edit crop & offset to take into account the shrink factor
        crop   = (round(crop[0]*shrink_factor), round(crop[1]*shrink_factor))
        offset = (round(offset[0]*shrink_factor), round(offset[1]*shrink_factor))

    def track_frame_at_number(frame_number):
        global last_eye_y_coords, last_eye_x_coords

        # get corresponding frame
        image = cap.get_frame(frame_number/old_video_fps)

        if invert:
            # invert the image
            image = (255 - image)

        # shrink & crop the image
        shrunken_image = shrink_image(image, shrink_factor)
        cropped_image = crop_image(shrunken_image, offset, crop)

        # get thresholded images
        head_threshold_image = get_head_threshold_image(cropped_image, head_threshold)
        tail_threshold_image = get_tail_threshold_image(cropped_image, tail_threshold)

        # track the image
        (tail_y_coords, tail_x_coords, spline_y_coords, spline_x_coords,
            eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords) = track_frame(cropped_image,
                                                                                    head_threshold, tail_threshold, 
                                                                                    head_threshold_image, tail_threshold_image,
                                                                                    min_eye_distance*shrink_factor,
                                                                                    eye_1_index, eye_2_index,
                                                                                    track_head_bool, track_tail_bool)

        # update last found coordinates of the eyes
        last_eye_y_coords = eye_y_coords
        last_eye_x_coords = eye_x_coords

        # add tracked coordinates to arrays
        if tail_y_coords is not None and tail_x_coords is not None:
            tail_end_y_coords_array[frame_number, :] = [tail_y_coords[0], tail_y_coords[-1]]
            tail_end_x_coords_array[frame_number, :] = [tail_x_coords[0], tail_x_coords[-1]]

        if eye_y_coords is not None and eye_x_coords is not None:
            eye_y_coords_array[frame_number, :] = eye_y_coords
            eye_x_coords_array[frame_number, :] = eye_x_coords

        if perp_y_coords is not None and perp_x_coords is not None:
            perp_y_coords_array[frame_number, :] = perp_y_coords
            perp_x_coords_array[frame_number, :] = perp_x_coords

        if save_video:
            # add tracked points to the image
            image = plot_image(cropped_image,
                            tail_y_coords, tail_x_coords,
                            spline_y_coords, spline_x_coords,
                            eye_y_coords, eye_x_coords,
                            perp_y_coords, perp_x_coords, perp_y_coords_array[:frame_number, 1], perp_x_coords_array[:frame_number, 1])

            # resize image
            image = shrink_image(image, new_video_size_factor)

            # add dynamic plot of heading angle
            if plot_heading_angle:
                # initialize plot image
                heading_plot_image = np.ones((90, image.shape[1], 3))*255
                
                # get array of angles
                angle_array = np.nan_to_num(an.get_heading_angle(None, False, perp_y_coords_array, perp_x_coords_array))

                # ratio of total # of plot points to the pixel width of the plot
                pixel_ratio = float(n_frames)/image.shape[1]

                # draw horizontal line at the top
                cv2.line(heading_plot_image, (0, 0), (image.shape[1], 0), (0, 0, 0), 1)

                # add description text
                cv2.putText(heading_plot_image, "Heading Angle", (0, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1, cv2.CV_AA)

                # draw a horizontal line in the middle
                cv2.line(heading_plot_image, (0, 45), (image.shape[1], 45), (160, 160, 160), 1)

                # add plot points
                for i in range(frame_number):
                    cv2.line(heading_plot_image, (int(round(i/pixel_ratio)), int(round(angle_array[i]/2.0))+45), (int(round((i+1)/pixel_ratio)), int(round(angle_array[i+1]/2.0))+45), (0, 0, 255), 1)

                # label y axis
                cv2.putText(heading_plot_image, "0", (0, 47), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
                cv2.putText(heading_plot_image, "90", (0, 8), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
                cv2.putText(heading_plot_image, "-90", (0, 89), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)

                # get sizes of the plot & the video frame
                h1, w1 = image.shape[:2]
                h2, w2 = heading_plot_image.shape[:2]

                # create an empty matrix to store the final combined frame
                vis = np.zeros((h1+h2, max(w1, w2),3), np.uint8)

                # combine the video frame & the plot
                vis[:h1, :w1,:3] = image
                vis[h1:h1+h2, :w2,:3] = heading_plot_image

                image = vis

            return image

    def track_frame_at_time(t):
        # get corresponding image for time t
        frame_number = int(t*new_video_fps)

        new_frame = track_frame_at_number(frame_number)
        
        return new_frame

    if save_video:
        animation = mpy.VideoClip(track_frame_at_time, duration=n_frames/new_video_fps)
        animation.write_videofile(new_video_path, codec='libx264', fps=new_video_fps)
    else:
        for frame_number in range(n_frames):
            track_frame_at_number(frame_number)

    # save tracked coordinates
    save_dir = os.path.join(os.path.dirname(new_video_path), "Tracking Data/")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savetxt(os.path.join(save_dir, "eye_y_coords_array.csv"), eye_y_coords_array)
    np.savetxt(os.path.join(save_dir, "eye_x_coords_array.csv"), eye_x_coords_array)
    np.savetxt(os.path.join(save_dir, "perp_y_coords_array.csv"), perp_y_coords_array)
    np.savetxt(os.path.join(save_dir, "perp_x_coords_array.csv"), perp_x_coords_array)
    np.savetxt(os.path.join(save_dir, "tail_end_y_coords_array.csv"), tail_end_y_coords_array)
    np.savetxt(os.path.join(save_dir, "tail_end_x_coords_array.csv"), tail_end_x_coords_array)

def load_image(image_path):
    # print("Loading {}.".format(image_path))

    try:
        image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    except:
        print("Error: Could not open image.")
        return None

    return image

def load_folder(folder_path):
    print("Loading images from {}.".format(folder_path))

    frames = []

    for filename in sort_nicely(os.listdir(folder_path)):
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = folder_path + "/" + filename

            # get image
            frame = load_image(image_path)

            # convert to greyscale
            if len(frame.shape) >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # add to frames list
            frames.append(frame)

    if len(frames) == 0:
        print("Could not find any images.")
        return None

    return frames

def load_video(video_path, n_frames=None):
    print("Loading video from {}.".format(video_path))

    # open the video
    try:
        cap = FFMPEG_VideoReader(video_path, True)
        cap.initialize()
    except:
        print("Error: Could not open video.")
        return None

    # get video info
    n_frames_orig = ffmpeg_parse_infos(video_path)["video_nframes"]
    fps = ffmpeg_parse_infos(video_path)["video_fps"]
    print("Original video fps: {0}. n_frames: {1}".format(fps, n_frames))

    if n_frames == None or n_frames > n_frames_orig: # load all frames
        n_frames = n_frames_orig

        f = 0
        frames = []

        while f < n_frames:
            # get image
            frame = cap.get_frame(f/fps)

            # convert to greyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # add to frames list
            frames.append(frame)

            f += 1
    else: # load only some frames

        # get evenly spaced frame numbers
        # r = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        # frame_nums = [0] + r(100, n_frames_orig)
        frame_nums = range(100)

        f = 0
        frames = []

        for f in sorted(frame_nums):
            # get image
            frame = cap.get_frame(f/fps)

            # convert to greyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # add to frames list
            frames.append(frame)

    return frames

def track_frame(image, head_thresh, tail_thresh, head_threshold_image, tail_threshold_image, min_eye_distance,
                eye_1_index, eye_2_index, track_head_bool, track_tail_bool,
                closest_eye_y_coords=None, closest_eye_x_coords=None):
    global head_threshold, tail_threshold

    head_threshold = head_thresh
    tail_threshold = tail_thresh

    if track_head_bool:
        # track head
        (eye_y_coords, eye_x_coords,
        perp_y_coords, perp_x_coords) = track_head(head_threshold_image,
                                                    eye_1_index, eye_2_index,
                                                    closest_eye_y_coords, closest_eye_x_coords)

        if eye_y_coords == None or len(eye_y_coords) < 2: # incorrect # of eyes found
            orig_head_threshold = head_threshold
            i = 0

            while eye_y_coords == None or len(eye_y_coords) < 2:
                if i < 80:
                    # increase head threshold & try again
                    head_threshold += 1

                    head_threshold_image = get_head_threshold_image(image, head_threshold)

                    (eye_y_coords, eye_x_coords,
                    perp_y_coords, perp_x_coords) = track_head(head_threshold_image,
                                                                eye_1_index, eye_2_index,
                                                                closest_eye_y_coords, closest_eye_x_coords)
                    i += 1
                else:
                    break

    else:
        (eye_y_coords, eye_x_coords,
        perp_y_coords, perp_x_coords) = [None]*4

    if track_tail_bool:
        # track tail
        (tail_y_coords, tail_x_coords,
        spline_y_coords, spline_x_coords) = track_tail(tail_threshold_image,
                                                eye_x_coords, eye_y_coords,
                                                min_eye_distance)

        if tail_y_coords == None: # tail wasn't able to be tracked
            orig_tail_threshold = tail_threshold
            i = 0

            while tail_y_coords == None:
                if i < 20:
                    # increase tail threshold & try again
                    tail_threshold += 1

                    tail_threshold_image = get_tail_threshold_image(image, tail_threshold)

                    (tail_y_coords, tail_x_coords,
                    spline_y_coords, spline_x_coords) = track_tail(tail_threshold_image,
                                                            eye_x_coords, eye_y_coords,
                                                            min_eye_distance)
                    i += 1
                elif i < 40:
                    if i == 20:
                        # reset to original threshold
                        tail_threshold = orig_tail_threshold

                    # decrease tail threshold & try again
                    tail_threshold -= 1

                    tail_threshold_image = get_tail_threshold_image(image, tail_threshold)

                    (tail_y_coords, tail_x_coords,
                    spline_y_coords, spline_x_coords) = track_tail(tail_threshold_image,
                                                            eye_x_coords, eye_y_coords,
                                                            min_eye_distance)
                    i += 1
                else:
                    break
    else:
        (tail_y_coords, tail_x_coords,
        spline_y_coords, spline_x_coords) = [None]*4

    return (tail_y_coords, tail_x_coords, spline_y_coords, spline_x_coords,
            eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords)

def plot_image(image,
                tail_y_coords, tail_x_coords,
                spline_y_coords, spline_x_coords,
                eye_y_coords, eye_x_coords,
                perp_y_coords, perp_x_coords, pos_hist_y_coords=None, pos_hist_x_coords=None):
    
    # convert to BGR
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # add eye points
    if eye_y_coords != None and eye_x_coords != None:
        for i in range(2):
            cv2.circle(image, (int(round(eye_x_coords[i])), int(round(eye_y_coords[i]))), 1, (0, 0, 255), -1)

    if pos_hist_y_coords != None and pos_hist_x_coords != None:
        # add position history
        for i in range(len(pos_hist_y_coords)):
            if i != len(pos_hist_y_coords)-1 and pos_hist_x_coords[i] != 0 and pos_hist_x_coords[i+1] != 0:
                cv2.line(image, (int(round(pos_hist_x_coords[i])), int(round(pos_hist_y_coords[i]))), (int(round(pos_hist_x_coords[i+1])), int(round(pos_hist_y_coords[i+1]))), (255, 0, 0), 1)

    try:
        # add tail points
        for i in range(len(spline_y_coords)):
            if i != len(spline_y_coords)-1:
                cv2.line(image, (int(round(spline_x_coords[i])), int(round(spline_y_coords[i]))), (int(round(spline_x_coords[i+1])), int(round(spline_y_coords[i+1]))), (0, 255, 0), 1)
    except:
        pass
    return image

def get_centroids(head_threshold_image):
    # find centroids in an image

    # convert to grayscale
    if len(head_threshold_image.shape) >= 3:
        head_threshold_image = cv2.cvtColor(head_threshold_image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(head_threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    moments = [cv2.moments(contour) for contour in contours]

    centroids_x_coords = []
    centroids_y_coords = []

    for m in moments:
        if m['m00'] != 0.0:
            centroids_y_coords.append(m['m01']/m['m00'])
            centroids_x_coords.append(m['m10']/m['m00'])

    if len(centroids_x_coords) < 2 or len(centroids_y_coords) < 2:
        # print("Error: Could not find centroids.")
        return [None]*2

    return centroids_y_coords, centroids_x_coords

def get_heading(eye_y_coords, eye_x_coords):
    # get coordinates of eyes
    y_1 = eye_y_coords[0]
    y_2 = eye_y_coords[1]
    x_1 = eye_x_coords[0]
    x_2 = eye_x_coords[1]

    if y_1 == y_2 and x_1 == x_2:
        return eye_y_coords, eye_x_coords

    if y_1 == y_2:
        y_1 = y_2 + 1e-6

    # get slope of line segment joining eyes
    m = (x_2 - x_1)/(y_2 - y_1)
    if m == 0:
        m = 1e-6

    # get midpoint of line segment joining eyes
    x_3 = (x_2 + x_1)/2
    y_3 = (y_2 + y_1)/2

    # get slope of perpendicular line
    k = -1.0/m

    # create the endpoint of a vector with this slope
    y_4 = 1
    x_4 = k*y_4

    # get endpoints of normalized & scaled vectors pointing in each direction
    y_5 = 5.0*y_4/np.sqrt(y_4**2 + x_4**2)
    x_5 = 5.0*x_4/np.sqrt(y_4**2 + x_4**2)

    y_6 = -5.0*y_4/np.sqrt(y_4**2 + x_4**2)
    x_6 = -5.0*x_4/np.sqrt(y_4**2 + x_4**2)

    # add vector endpoints to midpoint of eyes
    perp_y_coords = [y_5 + y_3, y_6 + y_3]
    perp_x_coords = [x_5 + x_3, x_6 + x_3]

    return perp_y_coords, perp_x_coords

def track_head(head_threshold_image, eye_1_index, eye_2_index, closest_eye_y_coords=None, closest_eye_x_coords=None):
    # get eye centroids
    centroids_y_coords, centroids_x_coords = get_centroids(head_threshold_image)

    if centroids_x_coords != None:
        n_centroids = len(centroids_x_coords)

        if eye_1_index >= n_centroids:
            eye_1_index = n_centroids - 1
        if eye_2_index >= n_centroids:
            eye_2_index = n_centroids - 1

        if closest_eye_y_coords and closest_eye_x_coords:
            # instead of using the given indices, pick the closest centroid coordinates to previosuly found eyes
            distances_to_eye_1 = np.sqrt((np.array(centroids_y_coords) - closest_eye_y_coords[0])**2 + (np.array(centroids_x_coords) - closest_eye_x_coords[0])**2)
            eye_1_index = np.argmin(distances_to_eye_1)

            distances_to_eye_2 = np.sqrt((np.array(centroids_y_coords) - closest_eye_y_coords[1])**2 + (np.array(centroids_x_coords) - closest_eye_x_coords[1])**2)
            eye_2_index = np.argmin(distances_to_eye_2)

            if min(distances_to_eye_2) > 5 or min(distances_to_eye_1) > 5:
                return [None]*4

            if eye_2_index == eye_1_index:
                distances_to_eye_2[eye_1_index] = 1000
                eye_2_index = np.argmin(distances_to_eye_2)

        if eye_2_index == eye_1_index:
            return [None]*4

        eye_y_coords = [ centroids_y_coords[eye_1_index], centroids_y_coords[eye_2_index] ]
        eye_x_coords = [ centroids_x_coords[eye_1_index], centroids_x_coords[eye_2_index] ]

        if closest_eye_y_coords and closest_eye_x_coords:
            # make sure distance between the eyes is similar to previous distance
            prev_eye_distance = np.sqrt((closest_eye_y_coords[1] - closest_eye_y_coords[0])**2 + (closest_eye_x_coords[1] - closest_eye_x_coords[0])**2)
            new_eye_distance = np.sqrt((eye_y_coords[1] - eye_y_coords[0])**2 + (eye_x_coords[1] - eye_x_coords[0])**2)

            if new_eye_distance > prev_eye_distance*1.2:
                return [None]*4

        perp_y_coords, perp_x_coords = get_heading(eye_y_coords, eye_x_coords)
    else:
        return [None]*4

    return eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords

def track_tail(tail_threshold_image, eye_x_coords, eye_y_coords, min_eye_distance, max_tail_eye_distance=10):
    # get size of thresholded image
    y_size = tail_threshold_image.shape[0]
    x_size = tail_threshold_image.shape[1]

    # convert to grayscale
    if len(tail_threshold_image.shape) >= 3:
        tail_threshold_image = cv2.cvtColor(tail_threshold_image, cv2.COLOR_BGR2GRAY)

    # get tail skeleton matrix
    skeleton_matrix = bwmorph_thin(tail_threshold_image, n_iter=np.inf).astype(int)

    # get an ordered list of coordinates of the tail, from one end to the other
    try:
        tail_y_coords, tail_x_coords = get_ordered_points(skeleton_matrix, eye_y_coords, eye_x_coords, max_tail_eye_distance)
    except:
        return [None]*4

    # get number of coordinates
    n_coords = tail_x_coords.shape[0]

    # modify tail skeleton coordinates (Huang et al., 2013)
    for i in range(n_coords):
        y = tail_y_coords[i]
        x = tail_x_coords[i]

        pixel_sum = 0
        y_sum     = 0
        x_sum     = 0

        radius = 1

        for k in range(-radius, radius+1):
            for l in range(-radius, radius+1):
                pixel_sum += tail_threshold_image[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]
                y_sum     += k*tail_threshold_image[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]
                x_sum     += l*tail_threshold_image[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]

        y_sum /= pixel_sum
        x_sum /= pixel_sum
        tail_y_coords[i] = y + y_sum
        tail_x_coords[i] = x + x_sum

    # remove coordinates that are close to the eyes
    if eye_x_coords and eye_y_coords:
        for i in range(len(eye_x_coords)):
            tail_mask = np.sqrt((tail_x_coords - eye_x_coords[i])**2 + (tail_y_coords - eye_y_coords[i])**2) > min_eye_distance
            tail_y_coords = tail_y_coords[tail_mask]
            tail_x_coords = tail_x_coords[tail_mask]

    try:
        # make ascending spiral in 3D space
        t = np.zeros(tail_x_coords.shape)
        t[1:] = np.sqrt((tail_x_coords[1:] - tail_x_coords[:-1])**2 + (tail_y_coords[1:] - tail_y_coords[:-1])**2)
        t = np.cumsum(t)
        t /= t[-1]

        nt = np.linspace(0, 1, 100)

        # calculate cubic spline
        spline_y_coords = interpolate.UnivariateSpline(t, tail_y_coords, k=3, s=20)(nt)
        spline_x_coords = interpolate.UnivariateSpline(t, tail_x_coords, k=3, s=20)(nt)

        spline = [spline_y_coords, spline_x_coords]
    except:
        print("Error: Could not calculate tail spline.")
        return [None]*4

    return tail_y_coords, tail_x_coords, spline[0], spline[1]

def get_ordered_points(matrix, eye_y_coords, eye_x_coords, max_tail_eye_distance):
    def find_unique_points(y_coords, x_coords):
        '''
        Find points given in y_coords & x_coords that have not yet been found.
        This function looks in found_y_coords & found_x_coords to check for matches.
        '''

        unique_points_y = []
        unique_points_x = []

        for i in range(len(y_coords)):
            y = y_coords[i]
            x = x_coords[i]

            if not (y in found_y_coords and x in found_x_coords):
                # found a unique coordinate
                unique_points_y.append(y)
                unique_points_x.append(x)

        return unique_points_y, unique_points_x

    def find_next_point_in_neighborhood(y, x, r, matrix):
        # pad the matrix with zeros by a given radius
        padded_matrix = np.zeros((matrix.shape[0] + 2*r, matrix.shape[1] + 2*r))
        padded_matrix[r:-r, r:-r] = matrix

        # get neighborhood around the current point
        neighborhood = padded_matrix[np.maximum(0, y):np.minimum(padded_matrix.shape[0], y+2*r+1), np.maximum(0, x):np.minimum(padded_matrix.shape[1], x+2*r+1)]

        # get coordinates of nonzero elements in the neighborhood
        nonzero_points_y, nonzero_points_x = np.nonzero(neighborhood)

        # translate these coordinates to image coordinates
        diff_x = r - x
        diff_y = r - y
        nonzero_points_y -= diff_y
        nonzero_points_x -= diff_x

        # find the next point
        try:
            unique_points_y, unique_points_x = find_unique_points(nonzero_points_y, nonzero_points_x)
        except:
            return None, None

        if len(unique_points_y) == 0 or len(unique_points_x) == 0:
            return None, None

        unique_points_y = np.array(unique_points_y)
        unique_points_x = np.array(unique_points_x)

        distances = np.sqrt((unique_points_y - found_y_coords[-1])**2 + (unique_points_x - found_x_coords[-1])**2)

        closest_index = np.argmin(distances)

        new_y = unique_points_y[closest_index]
        new_x = unique_points_x[closest_index]

        return new_y, new_x

    def move_along_tail():
        r = 1 # radius of neighborhood to look in for the next point

        while r < 10:
            new_y, new_x = find_next_point_in_neighborhood(found_y_coords[-1], found_x_coords[-1], r, matrix)

            if new_y != None and new_x != None:
                found_y_coords.append(new_y)
                found_x_coords.append(new_x)

                r = 1
            else:
                # could not find a point; increase the radius
                r += 1

    # pad the matrix with zeros by a radius of 1
    padded_matrix = np.zeros((matrix.shape[0] + 2, matrix.shape[1] + 2))
    padded_matrix[1:-1, 1:-1] = matrix

    # get size of matrix
    y_size = matrix.shape[0]
    x_size = matrix.shape[1]

    # initialize endpoint coordinates
    endpoint_x = 0
    endpoint_y = 0

    # Find row and column locations that are non-zero
    (rows,cols) = np.nonzero(matrix)

    # Initialize empty list of co-ordinates
    end_coords = None

    endpoint_y = 0
    endpoint_x = 0

    # find an endpoint of the tail
    for (r,c) in zip(rows,cols):
        if eye_y_coords != None:
            # look for an endpoint near the eyes
            if np.sqrt((r - eye_y_coords[0])**2 + (c - eye_x_coords[0])**2) < max_tail_eye_distance or np.sqrt((r - eye_y_coords[1])**2 + (c - eye_x_coords[1])**2) < max_tail_eye_distance:

                # Extract an 8-connected neighbourhood
                (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

                # Cast to int to index into image
                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')

                # Convert into a single 1D array and check for non-zero locations
                pix_neighbourhood = matrix[row_neigh,col_neigh].ravel() != 0

                # If the number of non-zero locations equals 2, add this to 
                # our list of co-ordinates
                if np.sum(pix_neighbourhood) == 2:
                    endpoint_y = r
                    endpoint_x = c
                    break
        else:
            # just find any endpoint

            # Extract an 8-connected neighbourhood
            (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

            # Cast to int to index into image
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')

            # Convert into a single 1D array and check for non-zero locations
            pix_neighbourhood = matrix[row_neigh,col_neigh].ravel() != 0

            # If the number of non-zero locations equals 2, add this to 
            # our list of co-ordinates
            if np.sum(pix_neighbourhood) == 2:
                endpoint_y = r
                endpoint_x = c
                break

    # initialize list of found tail coordinates
    found_y_coords = [endpoint_y]
    found_x_coords = [endpoint_x]

    # move along the tail
    move_along_tail()

    if len(found_x_coords) < 10:
        # we didn't manage to get the full tail; try moving along the tail in reverse
        found_y_coords = [found_y_coords[-1]]
        found_x_coords = [found_x_coords[-1]]
        move_along_tail()

    return np.array(found_y_coords), np.array(found_x_coords)

# --- HELPER FUNCTIONS --- #

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
