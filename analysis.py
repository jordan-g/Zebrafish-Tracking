import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def open_saved_data(data_path=None):
    # load first crop
    try:
        npzfile = np.load(data_path)

        tail_coords_array   = npzfile['tail_coords']
        spline_coords_array = npzfile['spline_coords']
        heading_angle_array = npzfile['heading_angle']
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

def load_tail_angles(csv_path):
    tail_angle_array = np.loadtxt(csv_path, delimiter=",")

    return tail_angle_array

def calculate_headfixed_tail_angles(heading_angle, tail_coords_array):
    '''
    Returns a K x J x M array of tail angles along the length of the tail, where
        - K is the number of crops (almost always 1),
        - J is the number of points along the tail, and
        - M is the number of frames in the video.
    '''

    # convert heading angle to radians
    heading_angle = heading_angle*np.pi/180.0

    # get number of crops, frames & tail points
    n_crops       = tail_coords_array.shape[0]
    n_frames      = tail_coords_array.shape[1]
    n_tail_points = tail_coords_array.shape[-1]

    # initialize array for storing tail angles
    tail_angle_array = np.zeros((n_crops, n_frames, n_tail_points)) + np.nan

    # create reference vectors pointing opposite to the heading angle
    reference_vectors = np.zeros((n_crops, n_frames, 2)) + np.nan
    reference_vectors[:, :, 0][:, :, np.newaxis] = -np.sin(heading_angle - np.pi)
    reference_vectors[:, :, 1][:, :, np.newaxis] = np.cos(heading_angle - np.pi)

    # create tail vectors by subtracting points along the tail and the first point along the tail
    tail_vectors = tail_coords_array - tail_coords_array[:, :, :, 0][:, :, :, np.newaxis]

    for k in range(n_crops):
        for j in range(n_tail_points):
            for m in range(n_frames):
                if np.linalg.norm(tail_vectors[k, m, :, j]) != 0:
                    # get an angle between -pi and pi
                    tail_angle_array[k, m, j] = np.arctan2(tail_vectors[k, m, 1, j], tail_vectors[k, m, 0, j]) - np.arctan2(reference_vectors[k, m, 0], reference_vectors[k, m, 1])
                    
    return tail_angle_array*180.0/np.pi

def calculate_freeswimming_tail_angles(heading_angle_array, body_position_array, tail_coords_array):
    '''
    Returns a K x J x M array of tail angles along the length of the tail, where
        - K is the number of crops (almost always 1),
        - J is the number of points along the tail, and
        - M is the number of frames in the video.
    '''

    # convert heading angle to radians
    heading_angle_array = heading_angle_array*np.pi/180.0

    # get number of crops, frames & tail points
    n_crops       = tail_coords_array.shape[0]
    n_frames      = tail_coords_array.shape[1]
    n_tail_points = tail_coords_array.shape[-1]

    # initialize array for storing tail angles
    tail_angle_array = np.zeros((n_crops, n_frames, n_tail_points)) + np.nan

    # create reference vectors pointing opposite to the heading angle
    reference_vectors = np.zeros((n_crops, n_frames, 2)) + np.nan
    reference_vectors[:, :, 0] = -np.sin(heading_angle_array[:, :, 0] - np.pi)
    reference_vectors[:, :, 1] = np.cos(heading_angle_array[:, :, 0] - np.pi)

    # create tail vectors by subtracting points along the tail and the body position
    tail_vectors = tail_coords_array - body_position_array[:, :, :, np.newaxis]

    for k in range(n_crops):
        for j in range(n_tail_points):
            for m in range(n_frames):
                if np.linalg.norm(tail_vectors[k, m, :, j]) != 0:
                    # get an angle between -pi and pi
                    tail_angle_array[k, m, j] = np.arctan2(tail_vectors[k, m, 1, j], tail_vectors[k, m, 0, j]) - np.arctan2(reference_vectors[k, m, 0], reference_vectors[k, m, 1])
                    
    return tail_angle_array*180.0/np.pi

def calculate_tail_end_angles(tail_angle_array, num_to_average=1, plot=False):
    n_frames = tail_angle_array.shape[0]
    tail_end_angles = np.zeros(n_frames)
    time_for_graph = []
##    ytime_for_graph = []

    for i in range(n_frames):
        tail_angles = tail_angle_array[i, :]
        tail_end_angles[i] = np.mean(tail_angles[ ~np.isnan(tail_angles) ][-num_to_average:])
        time_for_graph.append(float(i)/350)

##    print max(tail_end_angles)
##    print min(tail_end_angles)
    yAdd = max(tail_end_angles) + 5
    yTime = np.zeros(n_frames)
    ytime_for_graph = [x + yAdd for x in yTime]

    if plot:
        plt.plot(time_for_graph, tail_end_angles,'r', lw=1)
        plt.plot(time_for_graph, ytime_for_graph, 'b', lw=1)#Second graph for activity
        plt.title("Tail end angles (averaged over last {} point{} along the tail)".format(num_to_average, "s"*(num_to_average > 1)))
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Angle (Degrees)")
        plt.show()

    return tail_end_angles

def get_tail_end_angles(tail_angle_array, num_to_average=1):
    n_crops  = tail_angle_array.shape[0]
    n_frames = tail_angle_array.shape[1]
    tail_end_angles = np.zeros((n_crops, n_frames))

    for k in range(n_crops):
        for i in range(n_frames):
            tail_angles = tail_angle_array[k, i, :]
            tail_end_angles[k, i] = np.mean(tail_angles[ ~np.isnan(tail_angles) ][-num_to_average:])

    return tail_end_angles

def fix_heading_angles(heading_angle_array):
    # get number of crops, frames & tail points
    n_crops          = heading_angle_array.shape[0]
    n_frames         = heading_angle_array.shape[1]
    n_heading_points = heading_angle_array.shape[-1]

    for k in range(n_crops):
        for j in range(n_heading_points):
            heading_angle_array = np.pi - heading_angle_array

            for i in range(1, n_frames):
                if heading_angle_array[k, i, j] - heading_angle_array[k, i-1, j] > np.pi:
                    heading_angle_array[k, i:, j] -= 2*np.pi
                if heading_angle_array[k, i, j] - heading_angle_array[k, i-1, j] < -np.pi:
                    heading_angle_array[k, i:, j] += 2*np.pi

            for i in range(1, n_frames-2):
                if heading_angle_array[k, i, j] == np.nan and heading_angle_array[k, i-1, j] != np.nan and heading_angle_array[k, i+1, j] != np.nan:
                    heading_angle_array[k, i, j] = (heading_angle_array[k, i-1, j] + heading_angle_array[k, i+1, j])/2
                elif heading_angle_array[k, i, j] == np.nan and heading_angle_array[k, i+1, j] == np.nan and heading_angle_array[k, i-1, j] != np.nan and heading_angle_array[k, i+2, j] != np.nan:
                    heading_angle_array[k, i:i+2, j] = (heading_angle_array[k, i-1, j] + heading_angle_array[k, i+2, j])/2

    return heading_angle_array

def get_position_history(body_position_array, plot=True):
    positions_y = body_position_array[:, :, 0]
    positions_x = body_position_array[:, :, 1]

    speed_array = np.sqrt(np.gradient(positions_y)**2 + np.gradient(positions_x)**2)
    speed_array = np.convolve(speed, np.ones((3,))/3, mode='valid')

    if plot:
        plt.plot(positions_x, positions_y)
        plt.show()

    return positions_y, positions_x, speed

def plot_tail_angle_heatmap(tail_angle_array, crop_num=0, relative_range=False, save_path=""):
    fig = plt.figure(figsize=(100, 1))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('plasma')
    if relative_range:
        minimum = np.nanmin(tail_angle_array)
        maximum = np.nanmax(tail_angle_array)
    else:
        minimum = -180
        maximum = 180

    ax.imshow(tail_angle_array[crop_num].T, vmin=minimum, vmax=maximum, aspect='auto')

    if save_path != "":
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

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
