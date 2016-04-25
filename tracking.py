import matplotlib
matplotlib.use('TkAgg')

import cv2
from bwmorph_thin import bwmorph_thin
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import colorConverter
import matplotlib as mpl
from scipy import interpolate
import pdb

import os

def track_image(image_path, shrink_factor,
                offsets=None, crops=None,
                tail_threshold_brightness=219, head_threshold_brightness=60,
                min_eye_distance=20):
    '''
    Track zebrafish heading & tail motion in a whole image or multiple cropped areas of an image.

    image_path                : path to the image
    shrink_factor             : factor with which to reduce the resolution of the image before processing
    offsets                   : a list of (x, y) tuples that indicate top-left corners of cropped areas to track
    crops                     : a list of (w, h) tuples that indicate widths & heights of cropped areas to track
    tail_threshold_brightness : brightness threshold to use to isolate the zebrafish tail (0 - 255)
    head_threshold_brightness : brightness threshold to use to isolate the zebrafish head (0 - 255)
    min_eye_distance          : minimum distance from the eyes at which to start the tail curve
    '''

    # clear plots
    plt.clf()

    # get image directory, name & extension
    image_dir, image_filename = os.path.split(image_path)
    image_name, image_ext = image_filename.split('.')

    print("Importing image from {}.".format(image_path))

    # open image
    try:
        image = cv2.imread(image_path, 0)
    except:
        print("Error: Could not open image.")
        return

    # get original image size
    x_size = image.shape[1]
    y_size = image.shape[0]

    # print("Original image size is ({0}, {1}).".format(y_size, x_size))

    # shrink the image
    if shrink_factor not in (1.0, None):
        image = cv2.resize(image, (0,0), fx=shrink_factor, fy=shrink_factor)

    # get small image size
    x_size = image.shape[1]
    y_size = image.shape[0]

    # get number of cropped areas
    if crops is not None:
        n_crops = len(crops)
    else:
        n_crops = 1

    for i in range(n_crops):
        # get offsets
        if offsets is not None:
            x_offset = int(offsets[i][1]*shrink_factor)
            y_offset = int(offsets[i][0]*shrink_factor)
        else:
            x_offset = 0
            y_offset = 0

        # crop the image
        if crops is not None:
            # get crops
            x_crop = int(crops[i][1]*shrink_factor)
            y_crop = int(crops[i][0]*shrink_factor)

            if (x_offset + x_crop <= x_size) and (y_offset + y_crop <= y_size):
                cropped_image = image[y_offset:(y_offset + y_crop), x_offset:x_offset + x_crop]
            else:
                print("Error: Cropped area crosses the boundaries of the image.")
                return
        else:
            cropped_image = image

            print("Cropped, small image size is ({0}, {1}).".format(cropped_image.shape[0], cropped_image.shape[1]))

        # track head
        try:
            eye_centroids_x_coords, eye_centroids_y_coords, perpendicular_x_coords, perpendicular_y_coords = track_head(cropped_image, head_threshold_brightness)
        except:
            print("Error: Failed to track eyes.")
            return

        # track tail
        try:
            tail_x_coords, tail_y_coords, spline = track_tail(cropped_image, tail_threshold_brightness, eye_centroids_x_coords, eye_centroids_y_coords, min_eye_distance)
        except:
            print("Error: Failed to track tail.")
            return

        # plot result
        if n_crops == 1:
            # plot tail curve
            plt.plot(spline[1], spline[0], lw=1, c='red')

            # plot tail points
            # plt.scatter(tail_x_coords, tail_y_coords, s=5, c='blue')

            # plot eye centroids
            plt.scatter(eye_centroids_x_coords, eye_centroids_y_coords, s=2, c='orange')

            # join eye centroids
            plt.plot(eye_centroids_x_coords, eye_centroids_y_coords, lw=1, c='orange')

            # plot perpendicular line
            plt.plot(perpendicular_x_coords, perpendicular_y_coords, lw=1, c='green')

            # plot the cropped image
            plt.imshow(cropped_image, 'gray', interpolation='none')

            # set plot parameters
            plt.xlim(0, cropped_image.shape[1])
            plt.ylim(cropped_image.shape[0], 0)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_dpi(100)
            fig.set_size_inches(cropped_image.shape[1]/30.0, cropped_image.shape[0]/30.0)
            plt.axis('off')
        else:
            # convert coordinates of cropped image to full image coordinates
            tail_x_coords = tail_x_coords + x_offset
            tail_y_coords = tail_y_coords + y_offset
            spline[0] += y_offset
            spline[1] += x_offset
            eye_centroids_x_coords = [ x + x_offset for x in eye_centroids_x_coords]
            eye_centroids_y_coords = [ y + y_offset for y in eye_centroids_y_coords]
            perpendicular_x_coords = [ x + x_offset for x in perpendicular_x_coords]
            perpendicular_y_coords = [ y + y_offset for y in perpendicular_y_coords]

            # plot tail curve
            plt.plot(spline[1], spline[0], lw=1, c='red')

            # plot tail points
            # plt.scatter(tail_x_coords, tail_y_coords, s=0.5, c='blue')

            # plot eye centroids
            plt.scatter(eye_centroids_x_coords, eye_centroids_y_coords, s=1.0, c='orange')

            # join eye centroids
            plt.plot(eye_centroids_x_coords, eye_centroids_y_coords, lw=0.5, c='orange')

            # plot perpendicular line
            plt.plot(perpendicular_x_coords, perpendicular_y_coords, lw=0.5, c='green')

            # plot the original image
            plt.imshow(image, 'gray', interpolation='none')

            # set plot parameters
            plt.xlim(0, image.shape[1])
            plt.ylim(image.shape[0], 0)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_dpi(100)
            fig.set_size_inches(image.shape[1]/30.0, image.shape[0]/30.0)
            plt.axis('off')

    # create new directory to save tracked image
    if not os.path.exists(image_dir + "/tracked"):
      os.mkdir(image_dir + "/tracked")

    # save tracked image
    new_image_path = image_dir + "/tracked/" + image_name + ".jpg"
    plt.savefig(new_image_path, bbox_inches='tight', pad_inches=0)

    print("Successfully tracked the image.")
    return True

def track_head(image, head_threshold_brightness):
    # get thresholded image for head tracking & normalize
    _, head_threshold = cv2.threshold(image, head_threshold_brightness, 255, cv2.THRESH_BINARY_INV)
    head_threshold /= 255

    # get eye centroids
    eye_contours, _ = cv2.findContours(head_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    moments = [cv2.moments(contour) for contour in eye_contours]
    eye_centroids_x_coords = []
    eye_centroids_y_coords = []
    for m in moments:
        if m['m00'] != 0.0:
            eye_centroids_y_coords.append(m['m01']/m['m00'])
            eye_centroids_x_coords.append(m['m10']/m['m00'])
    if len(eye_centroids_x_coords) < 2 or len(eye_centroids_y_coords) < 2:
        print("Error: Could not find eye centroids.")
        return

    eye_centroids_x_coords = eye_centroids_x_coords[2:4]
    eye_centroids_y_coords = eye_centroids_y_coords[2:4]

    # get coordinates of centroids
    y_1 = eye_centroids_y_coords[0]
    y_2 = eye_centroids_y_coords[1]
    x_1 = eye_centroids_x_coords[0]
    x_2 = eye_centroids_x_coords[1]

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
    perpendicular_y_coords = [y_5 + y_3, y_6 + y_3]
    perpendicular_x_coords = [x_5 + x_3, x_6 + x_3]

    return eye_centroids_x_coords, eye_centroids_y_coords, perpendicular_x_coords, perpendicular_y_coords

def track_tail(image, tail_threshold_brightness, eye_x_coords, eye_y_coords, min_eye_distance):
    # get thresholded image for tail tracking & normalize
    _, tail_threshold = cv2.threshold(image, tail_threshold_brightness, 255, cv2.THRESH_BINARY_INV)
    tail_threshold /= 255

    # get size of thresholded image
    x_size = tail_threshold.shape[1]
    y_size = tail_threshold.shape[0]

    # get tail skeleton matrix
    skeleton_matrix = bwmorph_thin(tail_threshold, n_iter=np.inf)
    
    # get an ordered list of coordinates of the tail, from one end to the other
    try:
        tail_y_coords, tail_x_coords = get_ordered_points(skeleton_matrix, eye_y_coords, eye_x_coords)
    except:
        return

    # get number of coordinates
    n_coords = tail_x_coords.shape[0]

    # modify tail skeleton coordinates (Huang et al., 2013)
    for i in range(n_coords):
        x = tail_x_coords[i]
        y = tail_y_coords[i]

        pixel_sum = 0
        y_sum     = 0
        x_sum     = 0

        radius = 1

        for k in range(-radius, radius+1):
            for l in range(-radius, radius+1):
                pixel_sum += tail_threshold[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]
                x_sum     += l*tail_threshold[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]
                y_sum     += k*tail_threshold[np.minimum(y_size-1, np.maximum(0, y+k)), np.minimum(x_size-1, np.maximum(0, x+l))]

        y_sum /= pixel_sum
        x_sum /= pixel_sum
        tail_y_coords[i] = y + y_sum
        tail_x_coords[i] = x + x_sum

    # remove coordinates that are close to the eyes
    for i in range(len(eye_x_coords)):
        tail_mask = np.sqrt((tail_x_coords - eye_x_coords[i])**2 + (tail_y_coords - eye_y_coords[i])**2) > min_eye_distance
        tail_x_coords = tail_x_coords[tail_mask]
        tail_y_coords = tail_y_coords[tail_mask]

    try:
        # make ascending spiral in 3D space
        t = np.zeros(tail_x_coords.shape)
        t[1:] = np.sqrt((tail_x_coords[1:] - tail_x_coords[:-1])**2 + (tail_y_coords[1:] - tail_y_coords[:-1])**2)
        t = np.cumsum(t)
        t /= t[-1]

        nt = np.linspace(0, 1, 100)

        # calculate cubic spline
        spline_x_coords = interpolate.UnivariateSpline(t, tail_x_coords, k=3, s=20)(nt)
        spline_y_coords = interpolate.UnivariateSpline(t, tail_y_coords, k=3, s=20)(nt)

        spline = [spline_y_coords, spline_x_coords]
    except:
        print("Error: Could not calculate tail spline.")
        return

    return tail_x_coords, tail_y_coords, spline

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

        unique_points_x = np.array(unique_points_x)
        unique_points_y = np.array(unique_points_y)

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

            if (matrix[y, x] == 1) and (np.sum(neighborhood) == 2) and (np.sum(big_neighborhood) > 10) and (np.sqrt((y - eye_y_coords[0])**2 + (x - eye_x_coords[0])**2) < 60):
                # (hopefully) found an endpoint
                endpoint_y = y
                endpoint_x = x
                break
        else:
            continue
        break

    # initialize list of found tail coordinates
    found_x_coords = [endpoint_x]
    found_y_coords = [endpoint_y]

    # move along the tail
    move_along_tail()

    if len(found_x_coords) < 40:
        # we didn't manage to get the full tail; try moving along the tail in reverse
        found_x_coords = [found_x_coords[-1]]
        found_y_coords = [found_y_coords[-1]]
        move_along_tail()

    return np.array(found_y_coords), np.array(found_x_coords)