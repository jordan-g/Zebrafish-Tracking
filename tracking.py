import matplotlib
matplotlib.use('TkAgg')

import cv2
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from bwmorph_thin import bwmorph_thin
import numpy as np
from scipy import interpolate
import pdb
from moviepy.video.io.ffmpeg_reader import *
import os
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import re

def crop_image(image, offset, crop):
    # print(image.shape, offset, crop)
    if offset != None and crop != None:
        # print("hi")
        # print(image[np.maximum(0, offset[0]):np.minimum(offset[0] + crop[0], image.shape[0]), np.maximum(0, offset[1]):np.minimum(offset[1] + crop[1], image.shape[1])].shape)
        return image[np.maximum(0, offset[0]):np.minimum(offset[0] + crop[0], image.shape[0]), np.maximum(0, offset[1]):np.minimum(offset[1] + crop[1], image.shape[1])]
    else:
        return image

def shrink_image(image, shrink_factor):
    if shrink_factor != 1:
        image = cv2.resize(image, (0,0), fx=shrink_factor, fy=shrink_factor)
    return image

def get_head_threshold_image(image, head_threshold):
    _, head_threshold_image = cv2.threshold(image, head_threshold, 255, cv2.THRESH_BINARY_INV)
    return head_threshold_image

def get_tail_threshold_image(image, tail_threshold):
    _, tail_threshold_image = cv2.threshold(image, tail_threshold, 255, cv2.THRESH_BINARY_INV)
    tail_threshold_image /= 255
    return tail_threshold_image

def track_image(old_image_path, new_image_path, crop, offset, shrink_factor,
                    head_threshold, tail_threshold, min_eye_distance, eye_1_index, eye_2_index,
                    track_head_bool, track_tail_bool):

    image = load_image(old_image_path)

    cropped_image = crop_image(image, offset, crop)
    cropped_image = shrink_image(image, shrink_factor)

    head_threshold_image = get_head_threshold_image(cropped_image, head_threshold)
    tail_threshold_image = get_tail_threshold_image(cropped_image, tail_threshold)


    (tail_y_coords, tail_x_coords, spline_y_coords, spline_x_coords,
            eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords) = track_frame(head_threshold_image,
                                                                            tail_threshold_image,
                                                                            min_eye_distance,
                                                                            eye_1_index, eye_2_index, track_head_bool, track_tail_bool)
    image = plot_image(cropped_image,
                    tail_y_coords, tail_x_coords,
                    spline_y_coords, spline_x_coords,
                    eye_y_coords, eye_x_coords,
                    perp_y_coords, perp_x_coords)

    cv2.imwrite(new_image_path, image)

def track_folder(old_folder_path, new_video_path, crop, offset, shrink_factor,
                    head_threshold, tail_threshold, min_eye_distance, eye_1_index, eye_2_index,
                    track_head_bool, track_tail_bool):
    # print(crop, offset)
    frames = load_folder(old_folder_path)
    n_frames = len(frames)
    def get_frame(t):
        # print(t, n_frames, int(t*20))
        image = frames[int(t*20)]

        # print(crop, offset)

        cropped_image = crop_image(image, offset, crop)
        cropped_image = shrink_image(cropped_image, shrink_factor)

        # print(cropped_image.shape)

        head_threshold_image = get_head_threshold_image(cropped_image, head_threshold)
        tail_threshold_image = get_tail_threshold_image(cropped_image, tail_threshold)


        (tail_y_coords, tail_x_coords, spline_y_coords, spline_x_coords,
                eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords) = track_frame(head_threshold_image,
                                                                                tail_threshold_image,
                                                                                min_eye_distance,
                                                                                eye_1_index, eye_2_index, track_head_bool, track_tail_bool)
        # eye_y_coords_list.append(eye_y_coords)
        # eye_x_coords_list.append(eye_x_coords)
        # print(spline_x_coords)

        image = plot_image(cropped_image,
                        tail_y_coords, tail_x_coords,
                        spline_y_coords, spline_x_coords,
                        eye_y_coords, eye_x_coords,
                        perp_y_coords, perp_x_coords)

        return image

    animation = mpy.VideoClip(get_frame, duration=n_frames/20)
    animation.resize(3.3).write_videofile(new_video_path, codec='png', fps=20)

def track_video(old_video_path, new_video_path, crop, offset, shrink_factor,
                    head_threshold, tail_threshold, min_eye_distance, eye_1_index, eye_2_index,
                    track_head_bool, track_tail_bool):
    try:
        cap = FFMPEG_VideoReader(old_video_path, True)
        cap.initialize()
    except:
        print("Error: Could not open video.")
        return None

    n_frames = ffmpeg_parse_infos(old_video_path)["video_nframes"]
    fps = ffmpeg_parse_infos(old_video_path)["video_fps"]

    print("fps: {0}. n_frames: {1}".format(fps, n_frames))

    def get_frame(t):
        frame_number = int(t*20)

        image = cap.get_frame(frame_number/fps)

        # convert to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cropped_image = crop_image(image, offset, crop)
        cropped_image = shrink_image(cropped_image, shrink_factor)

        head_threshold_image = get_head_threshold_image(cropped_image, head_threshold)
        tail_threshold_image = get_tail_threshold_image(cropped_image, tail_threshold)


        (tail_y_coords, tail_x_coords, spline_y_coords, spline_x_coords,
                eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords) = track_frame(head_threshold_image,
                                                                                tail_threshold_image,
                                                                                min_eye_distance,
                                                                                eye_1_index, eye_2_index, track_head_bool, track_tail_bool)
        # eye_y_coords_list.append(eye_y_coords)
        # eye_x_coords_list.append(eye_x_coords)

        image = plot_image(cropped_image,
                        tail_y_coords, tail_x_coords,
                        spline_y_coords, spline_x_coords,
                        eye_y_coords, eye_x_coords,
                        perp_y_coords, perp_x_coords)

        return image

    animation = mpy.VideoClip(get_frame, duration=n_frames/20)
    animation.resize(3.3).write_videofile(new_video_path, codec='png', fps=20)

def load_video(video_path):
    print("Loading video from {}.".format(video_path))

    try:
        cap = FFMPEG_VideoReader(video_path, True)
        cap.initialize()
    except:
        print("Error: Could not open video.")
        return None

    n_frames = ffmpeg_parse_infos(video_path)["video_nframes"]
    fps = ffmpeg_parse_infos(video_path)["video_fps"]

    print("fps: {0}. n_frames: {1}".format(fps, n_frames))

    f = 0
    frames = []

    while f < n_frames:
        # get image
        frame = cap.get_frame(f/fps)

        # convert to greyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)

        f += 1

    return frames

def load_folder(folder_path):
    print("Loading images from {}.".format(folder_path))

    frames = []
    # print(os.listdir(folder_path))

    for filename in sort_nicely(os.listdir(folder_path)):
        if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = folder_path + "/" + filename

            frame = load_image(image_path)
            frames.append(frame)

    if len(frames) == 0:
        print("Could not find any images.")
        return None

    return frames

def load_image(image_path):
    print("Loading {}.".format(image_path))

    try:
        image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    except:
        print("Error: Could not open image.")
        return None

    return image

def test():
    plt.plot([1, 2], [4, 5])
    plt.show()

def track_frame(head_threshold_image, tail_threshold_image, min_eye_distance,
                eye_1_index, eye_2_index, track_head_bool, track_tail_bool):

    if track_head_bool:
        # track head
        (eye_y_coords, eye_x_coords,
        perp_y_coords, perp_x_coords) = track_head(head_threshold_image,
                                                    eye_1_index, eye_2_index)
    else:
        (eye_y_coords, eye_x_coords,
        perp_y_coords, perp_x_coords) = [None]*4

    if track_tail_bool:
        # track tail
        (tail_y_coords, tail_x_coords,
        spline_y_coords, spline_x_coords) = track_tail(tail_threshold_image,
                                                eye_x_coords, eye_y_coords,
                                                min_eye_distance)
        # print(spline_x_coords, tail_threshold_image.shape)
    else:
        (tail_y_coords, tail_x_coords,
        spline_y_coords, spline_x_coords) = [None]*4

    return (tail_y_coords, tail_x_coords, spline_y_coords, spline_x_coords,
            eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords)

def plot_image(image,
                tail_y_coords, tail_x_coords,
                spline_y_coords, spline_x_coords,
                eye_y_coords, eye_x_coords,
                perp_y_coords, perp_x_coords):

    # print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if eye_y_coords != None and eye_x_coords != None:
        for i in range(2):
            cv2.circle(image, (int(eye_x_coords[i]), int(eye_y_coords[i])), 1, (0, 0, 255), -1)

    if spline_y_coords != None and spline_x_coords != None:
        for i in range(len(spline_y_coords)):
            if i != len(spline_y_coords)-1:
                cv2.line(image, (int(spline_x_coords[i]), int(spline_y_coords[i])), (int(spline_x_coords[i+1]), int(spline_y_coords[i+1])), (0, 255, 0), 1)

    return image

def save_image(fig, axes, image_path):
    # get image directory, name & extension
    image_dir, image_filename = os.path.split(image_path)
    image_name, image_ext = image_filename.split('.')

    # save tracked image
    new_image_path = image_dir + "/" + image_name + "-tracked.jpg"
    plt.savefig(new_image_path, bbox_inches='tight', pad_inches=0)

def track_and_save_image(image_path, shrink_factor,
                            tail_threshold,
                            head_threshold,
                            min_eye_distance,
                            offset=None, crop=None,
                            eye_1_index=0, eye_2_index=1):

    image = load_image(image_path)

    # crop the image
    if crop != None and offset != None:
        cropped_image = crop_image(image, offset, crop)
    else:
        cropped_image = image

    if shrink_factor != None:
        cropped_image = shrink_image(cropped_image, shrink_factor)

    head_threshold_image = get_head_threshold_image(cropped_image, head_threshold)
    tail_threshold_image = get_tail_threshold_image(cropped_image, tail_threshold)

    (tail_y_coords, tail_x_coords,
        spline_y, spline_x,
        eye_y_coords, eye_x_coords,
        perp_y_coords, perp_x_coords) = track_frame(head_threshold_image, tail_threshold_image,
                                            min_eye_distance,
                                            eye_1_index, eye_2_index)

    fig, axes = plot_image(cropped_image,
                    tail_y_coords, tail_x_coords,
                    spline_y_coords, spline_x_coords,
                    eye_y_coords, eye_x_coords,
                    perp_y_coords, perp_x_coords)

    save_image(image_path, fig, axes)

    print("Successfully tracked the image.")

def get_centroids(head_threshold_image):
    # get centroids
    contours, _ = cv2.findContours(head_threshold_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    moments = [cv2.moments(contour) for contour in contours]
    centroids_x_coords = []
    centroids_y_coords = []
    for m in moments:
        if m['m00'] != 0.0:
            centroids_y_coords.append(m['m01']/m['m00'])
            centroids_x_coords.append(m['m10']/m['m00'])
    if len(centroids_x_coords) < 2 or len(centroids_y_coords) < 2:
        print("Error: Could not find centroids.")
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

def track_head(head_threshold_image, eye_1_index, eye_2_index):
    centroids_y_coords, centroids_x_coords = get_centroids(head_threshold_image)
    if centroids_x_coords != None:
        n_centroids = len(centroids_x_coords)
        if eye_1_index >= n_centroids:
            eye_1_index = n_centroids - 1
        if eye_2_index >= n_centroids:
            eye_2_index = n_centroids - 1
        eye_y_coords = [ centroids_y_coords[eye_1_index], centroids_y_coords[eye_2_index] ]
        eye_x_coords = [ centroids_x_coords[eye_1_index], centroids_x_coords[eye_2_index] ]

        perp_y_coords, perp_x_coords = get_heading(eye_y_coords, eye_x_coords)
    else:
        return [None]*4

    return eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords

def track_tail(tail_threshold_image, eye_x_coords, eye_y_coords, min_eye_distance):

    # print(tail_threshold_image.shape, eye_x_coords, eye_y_coords, min_eye_distance)

    # get size of thresholded image
    y_size = tail_threshold_image.shape[0]
    x_size = tail_threshold_image.shape[1]

    # get tail skeleton matrix
    skeleton_matrix = bwmorph_thin(tail_threshold_image, n_iter=np.inf)

    # get an ordered list of coordinates of the tail, from one end to the other
    try:
        tail_y_coords, tail_x_coords = get_ordered_points(skeleton_matrix, eye_y_coords, eye_x_coords)
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

def get_ordered_points(matrix, eye_y_coords, eye_x_coords):
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
            return

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
            try:
                new_y, new_x = find_next_point_in_neighborhood(found_y_coords[-1], found_x_coords[-1], r, matrix)
                found_y_coords.append(new_y)
                found_x_coords.append(new_x)
            except:
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

    # get an endpoint
    for y in range(y_size):
        for x in range(x_size):
            # get big & small neighborhoods of the current point
            neighborhood = padded_matrix[y:y+3, x:x+3]
            big_neighborhood = padded_matrix[np.maximum(0, y-20):np.minimum(padded_matrix.shape[0], y+21), np.maximum(0, x-20):np.minimum(padded_matrix.shape[1], x+21)]

            if (matrix[y, x] == 1) and (np.sum(neighborhood) == 2) and (np.sum(big_neighborhood) > 10):
                if eye_x_coords and eye_y_coords:
                    if (np.sqrt((y - eye_y_coords[0])**2 + (x - eye_x_coords[0])**2) < 60):
                        # (hopefully) found an endpoint
                        endpoint_y = y
                        endpoint_x = x
                        break
                else:
                    # (hopefully) found an endpoint
                    endpoint_y = y
                    endpoint_x = x
                    break
        else:
            continue
        break

    # initialize list of found tail coordinates
    found_y_coords = [endpoint_y]
    found_x_coords = [endpoint_x]

    # move along the tail
    move_along_tail()

    if len(found_x_coords) < 40:
        # we didn't manage to get the full tail; try moving along the tail in reverse
        found_y_coords = [found_y_coords[-1]]
        found_x_coords = [found_x_coords[-1]]
        move_along_tail()

    return np.array(found_y_coords), np.array(found_x_coords)

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
