import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def open_saved_data(save_dir=None):
    # load first crop
    try:
        # npzfile = np.load(glob.glob(os.path.join(save_dir, "*.npz"))[0])
        npzfile = np.load(save_dir)

        tail_coords_array   = npzfile['tail_coords']
        spline_coords_array = npzfile['spline_coords']
        heading_angle_array = np.radians(npzfile['heading_angle'])
        body_position_array = npzfile['body_position']
        eye_coords_array    = npzfile['eye_coords']
        params              = npzfile['params'][()]

        if params['type'] == "headfixed":
            heading_angle_array = None
            body_position_array = None
            eye_coords_array    = None
        else:
            if params['track_tail'] == False:
                tail_coords_array   = None
                spline_coords_array = None
            if params['track_eyes'] == False:
                eye_coords_array = None
    except:
        print("Error: Tracking data could not be found.")
        return [None]*6

    return tail_coords_array, spline_coords_array, heading_angle_array, body_position_array, eye_coords_array, params

def fix_heading_angles(heading_angle_array):
    # get number of crops, frames & tail points
    n_crops          = heading_angle_array.shape[0]
    n_frames         = heading_angle_array.shape[1]
    n_heading_points = heading_angle_array.shape[-1]

    for k in range(n_crops):
        for j in range(n_heading_points):
            # correct for abrupt jumps in angle due to vectors switching quadrants between frames
            for i in range(1, n_frames):
                if heading_angle_array[k, i, j] - heading_angle_array[k, i-1, j] >= np.pi/2.0:
                    heading_angle_array[k, i, j] -= np.pi
                elif heading_angle_array[k, i, j] - heading_angle_array[k, i-1, j] <= -np.pi/2.0:
                    heading_angle_array[k, i, j] += np.pi

    return heading_angle_array

def get_freeswimming_tail_angles(tail_coords_array, heading_angle_array, body_position_array):
    print("hi")
    # get number of crops, frames & tail points
    n_crops       = tail_coords_array.shape[0]
    n_frames      = tail_coords_array.shape[1]
    n_tail_points = tail_coords_array.shape[-1]

    # initialize array for storing tail angles
    tail_angle_array = np.zeros((n_crops, n_frames, n_tail_points)) + np.nan

    # create heading vectors based on heading angle
    heading_vectors = np.zeros((n_crops, n_frames, 2)) + np.nan
    heading_vectors[:, :, 0][:, :, np.newaxis] = np.cos(heading_angle_array)
    heading_vectors[:, :, 1][:, :, np.newaxis] = np.sin(heading_angle_array)

    # create array of start/end coordinates of the line with the heading angle passing through the body center position
    heading_coords_array = np.zeros((n_crops, n_frames, 2, 2))
    heading_coords_array[:, :, :, 0] = body_position_array + heading_vectors
    heading_coords_array[:, :, :, 1] = body_position_array - heading_vectors

    # get distances between start/end coordinates of the heading line and the starting coordinates of the tail
    tail_distances = np.sqrt((heading_coords_array[:, :, 0, :] - tail_coords_array[:, :, 0, -1][:, :, np.newaxis])**2 + (heading_coords_array[:, :, 1, :] - tail_coords_array[:, :, 1, -1][:, :, np.newaxis])**2)

    # get frames where the "starting" heading coordinate is closer to the tail than the "ending" coordinate
    mask = tail_distances[:, :, 0] < tail_distances[:, :, 1]

    # flip heading vectors for these frames, so that all vectors point toward the tail
    heading_vectors[mask, :] *= -1

    # create tail vectors by subtracting points along the tail and the body position
    tail_vectors = body_position_array[:, :, :, np.newaxis] - tail_coords_array

    tail_distance_start = np.sqrt((heading_coords_array[:, :, 0, 0] - tail_coords_array[:, :, 0, 0])**2 + (heading_coords_array[:, :, 1, 0] - tail_coords_array[:, :, 1, 0])**2)
    tail_distance_end   = np.sqrt((heading_coords_array[:, :, 0, 0] - tail_coords_array[:, :, 0, -1])**2 + (heading_coords_array[:, :, 1, 0] - tail_coords_array[:, :, 1, -1])**2)

    mask = tail_distance_end < tail_distance_start

    tail_coords_array[mask, :, :] = np.fliplr(tail_coords_array[mask, :, :])

    for k in range(n_crops):
        for j in range(n_tail_points):
            # get dot product and determinant between the tail vectors and the heading vectors
            dot = tail_vectors[k, :, 0, j]*heading_vectors[k, :, 1] + tail_vectors[k, :, 1, j]*heading_vectors[k, :, 0] # dot product
            det = tail_vectors[k, :, 0, j]*heading_vectors[k, :, 0] - tail_vectors[k, :, 1, j]*heading_vectors[k, :, 1] # determinant

            # get an angle between 0 and 2*pi
            tail_angle_array[k, :, j] = np.arctan2(dot, det)

            # print(tail_angle_array[k, 0, j] // np.pi/2.0)

            # tail_angle_array[k, :, j] -= (tail_angle_array[k, 0, j] // np.pi/2.0)*np.pi/2.0

            # correct for abrupt jumps in angle due to vectors switching quadrants between frames
            for i in range(1, n_frames):
                # if tail_angle_array[k, i, j] - tail_angle_array[k, i-1, j] >= np.pi:
                #     tail_angle_array[k, i, j] -= 2*np.pi
                # elif tail_angle_array[k, i, j] - tail_angle_array[k, i-1, j] <= -np.pi:
                #     tail_angle_array[k, i, j] += 2*np.pi
                if tail_angle_array[k, i, j] - tail_angle_array[k, i-1, j] >= np.pi/4.0:
                    if k == 3 and i < 30:
                        print(k, j, i, i-1, tail_angle_array[k, i, j], tail_angle_array[k, i-1, j])
                    tail_angle_array[k, i, j] -= np.pi
                elif tail_angle_array[k, i, j] - tail_angle_array[k, i-1, j] <= -np.pi/4.0:
                    if k == 3 and i < 30:
                        print(k, j, i, i-1, tail_angle_array[k, i, j], tail_angle_array[k, i-1, j])
                    tail_angle_array[k, i, j] += np.pi

            print("round 2")

            for i in range(1, n_frames):
                if tail_angle_array[k, i, j] - tail_angle_array[k, i-1, j] >= np.pi/4.0:
                    if k == 3 and i < 30:
                        print(k, j, i, i-1, tail_angle_array[k, i, j], tail_angle_array[k, i-1, j])
                    tail_angle_array[k, i, j] -= np.pi
                elif tail_angle_array[k, i, j] - tail_angle_array[k, i-1, j] <= -np.pi/4.0:
                    if k == 3 and i < 30:
                        print(k, j, i, i-1, tail_angle_array[k, i, j], tail_angle_array[k, i-1, j])
                    tail_angle_array[k, i, j] += np.pi

            # print(tail_angle_array[k, 0, j] // np.pi/2.0)

            if tail_angle_array[k, 0, j] >= np.pi/2.0:
                tail_angle_array[k, :, j] -= np.pi
            elif tail_angle_array[k, 0, j] <= -np.pi/2.0:
                tail_angle_array[k, :, j] += np.pi

            # tail_angle_array[k, :, j] -= (tail_angle_array[k, 0, j] // np.pi/2.0)*np.pi/2.0

    return tail_angle_array

def get_headfixed_tail_angles(tail_coords_array, tail_direction):
    # convert tail direction to heading angle
    if tail_direction == "Left":
        heading_angle = 0
    elif tail_direction == "Down":
        heading_angle = np.pi/2.0
    elif tail_direction == "Right":
        heading_angle = np.pi
    elif tail_direction == "Up":
        heading_angle = -3.0*pi/2.0

    # get number of crops, frames & tail points
    n_crops       = tail_coords_array.shape[0]
    n_frames      = tail_coords_array.shape[1]
    n_tail_points = tail_coords_array.shape[-1]

    # initialize array for storing tail angles
    tail_angle_array = np.zeros((n_crops, n_frames, n_tail_points)) + np.nan

    # create heading vectors based on heading angle
    heading_vectors = np.zeros((n_crops, n_frames, 2)) + np.nan
    heading_vectors[:, :, 0][:, :, np.newaxis] = np.cos(heading_angle)
    heading_vectors[:, :, 1][:, :, np.newaxis] = np.sin(heading_angle)

    # create tail vectors by subtracting points along the tail and the body position
    tail_vectors = tail_coords_array[:, :, :, 0][:, :, :, np.newaxis] - tail_coords_array

    for k in range(n_crops):
        for j in range(n_tail_points):
            # get dot product and determinant between the tail vectors and the heading vectors
            dot = tail_vectors[k, :, 0, j]*heading_vectors[k, :, 1] + tail_vectors[k, :, 1, j]*heading_vectors[k, :, 0] # dot product
            det = tail_vectors[k, :, 0, j]*heading_vectors[k, :, 0] - tail_vectors[k, :, 1, j]*heading_vectors[k, :, 1] # determinant

            # get an angle between 0 and 2*pi
            tail_angle_array[k, :, j] = np.arctan2(dot, det)

            # correct for abrupt jumps in angle due to vectors switching quadrants between frames
            for i in range(1, n_frames):
                if tail_angle_array[k, i, j] - tail_angle_array[k, i-1, j] >= np.pi/2.0:
                    tail_angle_array[k, i, j] -= np.pi
                elif tail_angle_array[k, i, j] - tail_angle_array[k, i-1, j] <= -np.pi/2.0:
                    tail_angle_array[k, i, j] += np.pi

    return tail_angle_array

def get_position_history(body_position_array, plot=True):
    positions_y = body_position_array[:, :, 0]
    positions_x = body_position_array[:, :, 1]

    speed_array = np.sqrt(np.gradient(positions_y)**2 + np.gradient(positions_x)**2)

    speed_array = np.convolve(speed, np.ones((3,))/3, mode='valid')

    # positions_y[positions_y == 0] = np.nan
    # positions_x[positions_x == 0] = np.nan

    if plot:
        plt.plot(positions_x, positions_y)
        plt.show()

    return positions_y, positions_x, speed

def plot_tail_angle_heatmap(perp_vectors, spline_vectors):
    angle_array = np.zeros((spline_vectors.shape[0], spline_vectors.shape[-1]))

    for j in range(angle_array.shape[-1]):
        dot = spline_vectors[:, 1, j]*perp_vectors[:, 1] + spline_vectors[:, 0, j]*perp_vectors[:, 0]      # dot product
        det = spline_vectors[:, 1, j]*perp_vectors[:, 0] - spline_vectors[:, 0, j]*perp_vectors[:, 1]      # determinant

        angle_array[:, j] = np.arctan2(dot, det) - np.pi/2.0

        for i in range(1, angle_array.shape[0]-1):
            if angle_array[i, j] - angle_array[i-1, j] >= np.pi/2.0:
                angle_array[i, j] -= np.pi/2.0
            elif angle_array[i, j] - angle_array[i-1, j] <= -np.pi/2.0:
                angle_array[i, j] += np.pi/2.0

    fig = plt.figure()
    fig.set_size_inches(100, 1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('plasma')
    ax.imshow(angle_array.T, vmin=-np.pi/3, vmax=np.pi/3, aspect = 'auto')
    plt.savefig("heatmap.png", dpi = 300)

def contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx