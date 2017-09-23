import numpy as np
import matplotlib.pyplot as plt
import analysis
import scipy.stats
import os
import json

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def find_nearest(array, value, return_index=False):
    idx = (np.abs(array - value)).argmin()

    if return_index:
        return idx
    else:
        return array[idx]

def process_video(folder, video_name, plot=False):
    # set data paths
    tracking_path   = os.path.join(folder, "{}_Image-Data_Video_tracking.npz".format(video_name))
    stim_data_path  = os.path.join(folder, "{}_Stimulus-Data.csv".format(video_name))
    frame_data_path = os.path.join(folder, "{}_Vimba-Data.csv".format(video_name))

    # load tracking data
    tail_coords_array, spline_coords_array, heading_angle, body_position, eye_coords_array, tracking_params = analysis.open_saved_data(tracking_path)
    heading_angle = analysis.fix_heading_angles(heading_angle)
    heading_angle = heading_angle[0, :, 0]
    body_position = body_position[0]

    # load frame timestamp data
    frame_data = np.loadtxt(frame_data_path, skiprows=1)

    # get total number of frames
    n_frames = frame_data.shape[0]
    print("Number of frames: {}.".format(n_frames))

    # calculate milliseconds at which each frame occurs
    frame_milliseconds = np.zeros(n_frames)
    for i in range(n_frames):
        frame_milliseconds[i] = 1000*(60*(60*frame_data[i, 0] + frame_data[i, 1]) + frame_data[i, 2]) + frame_data[i, 3]
    frame_nums = frame_data[:, -1]

    # load stimulus timestamp data
    stim_data = np.loadtxt(stim_data_path, skiprows=1)

    # get total number of stim switches
    n_stim_switches = stim_data.shape[0]
    print("Number of stimulus switches: {}.".format(n_stim_switches))

    # calculate milliseconds and closest frame numbers at which stim switches occur
    stim_switch_milliseconds = np.zeros(n_stim_switches)
    stim_switch_frame_nums   = np.zeros(n_stim_switches)
    for i in range(n_stim_switches):
        stim_switch_milliseconds[i] = 1000*(60*(60*stim_data[i, 0] + stim_data[i, 1]) + stim_data[i, 2]) + stim_data[i, 3]
        stim_switch_frame_nums[i] = frame_nums[find_nearest(frame_milliseconds, stim_switch_milliseconds[i], return_index=True)]
    stim_switch_frame_nums = stim_switch_frame_nums.astype(int)

    # extract stim ids
    stim_ids = stim_data[:, -1]
    stim_ids = stim_ids.astype(int)

    # create array containing the stim id for each frame
    stim_id_frames = np.zeros(n_frames).astype(int)
    for i in range(n_stim_switches):
        if i < n_stim_switches - 1:
            stim_id_frames[stim_switch_frame_nums[i]:stim_switch_frame_nums[i+1]] = stim_ids[i]
        else:
            stim_id_frames[stim_switch_frame_nums[i]:] = stim_ids[i]

    # ---- capture bouts that correspond to turns ---- #

    # smooth the heading angle array using a Savitzky-Golay filter
    smoothing_window_width = 50
    smoothed_heading_angle = savitzky_golay(heading_angle, 51, 3)

    # calculate the difference betweeen the heading angle at each frame and the heading angle 10 frames before
    n = 10
    running_heading_angle_difference = np.abs(smoothed_heading_angle - np.roll(smoothed_heading_angle, -n))
    running_heading_angle_difference[-n:] = 0
    running_heading_angle_difference = np.nan_to_num(running_heading_angle_difference)

    # extract points where the difference is greater than the threshold
    threshold = 0.1
    heading_angle_difference_above_threshold = (running_heading_angle_difference >= threshold)

    # smooth this array
    smoothing_window_width = 20
    normpdf = scipy.stats.norm.pdf(range(-int(smoothing_window_width/2),int(smoothing_window_width/2)),0,3)
    heading_angle_difference_above_threshold[int(smoothing_window_width/2):-int(smoothing_window_width/2) + 1] = np.convolve(heading_angle_difference_above_threshold, normpdf/np.sum(normpdf), mode='valid')
    heading_angle_difference_above_threshold = heading_angle_difference_above_threshold.astype(int)

    # ---- capture bouts that correspond to forward motions ---- #

    # smooth the body position array using a Savitzky-Golay filter
    smoothed_body_position = np.zeros(body_position.shape)
    smoothed_body_position[:, 0] = savitzky_golay(body_position[:, 0], 51, 3)
    smoothed_body_position[:, 1] = savitzky_golay(body_position[:, 1], 51, 3)

    # get the distance from the x-y position
    body_distance          = np.sqrt(body_position[:, 0]**2 + body_position[:, 1]**2)
    smoothed_body_distance = np.sqrt((smoothed_body_position[:, 0])**2 + (smoothed_body_position[:, 1])**2)

    # scale so that it's in the same range as the heading angle array
    body_distance -= np.nanmin(body_distance)
    body_distance = (np.nanmax(heading_angle) - np.nanmin(heading_angle))*body_distance/np.nanmax(body_distance)
    body_distance += np.nanmin(heading_angle)

    smoothed_body_distance -= np.nanmin(smoothed_body_distance)
    smoothed_body_distance = (np.nanmax(smoothed_heading_angle) - np.nanmin(smoothed_heading_angle))*smoothed_body_distance/np.nanmax(smoothed_body_distance)
    smoothed_body_distance += np.nanmin(smoothed_heading_angle)

    # calculate the difference betweeen the body distance at each frame and the heading angle 10 frames before
    n = 10
    running_body_distance_difference = np.abs(smoothed_body_distance - np.roll(smoothed_body_distance, -n))
    running_body_distance_difference[-n:] = 0
    running_body_distance_difference = np.nan_to_num(running_body_distance_difference)

    # extract points where the difference is greater than the threshold
    threshold = 0.1
    body_distance_difference_above_threshold = (running_body_distance_difference >= threshold)

    # smooth this array
    smoothing_window_width = 20
    normpdf = scipy.stats.norm.pdf(range(-int(smoothing_window_width/2),int(smoothing_window_width/2)),0,3)
    body_distance_difference_above_threshold[int(smoothing_window_width/2):-int(smoothing_window_width/2) + 1] = np.convolve(body_distance_difference_above_threshold, normpdf/np.sum(normpdf), mode='valid')
    body_distance_difference_above_threshold = body_distance_difference_above_threshold.astype(int)

    # -------------------------------------------------- #

    # combine bouts obtained by looking at the body position with those obtained by looking at the heading angle
    combined_difference_above_threshold = np.logical_or(body_distance_difference_above_threshold, heading_angle_difference_above_threshold).astype(int)

    # get the frame numbers of the start & end of all the bouts
    above_threshold_difference = combined_difference_above_threshold - np.roll(combined_difference_above_threshold, -1)
    above_threshold_difference[-1] = 0
    bout_start_frames = np.nonzero(above_threshold_difference == -1)[0] + 1
    bout_end_frames   = np.nonzero(above_threshold_difference == 1)[0] - 1

    # get total number of bouts
    n_bouts = len(bout_start_frames)
    print("Number of bouts: {}.".format(n_bouts))

    # create array containing the bout number for each frame
    # we set it to -1 when a frame is not in a bout
    bout_number_frames = np.zeros(n_frames) - 1
    for i in range(n_bouts):
        bout_number_frames[bout_start_frames[i]:bout_end_frames[i]] = i

    # initialize variable to calcualate the mean bout length in milliseconds
    mean_bout_length = 0

    # determine, for each bout, the heading angle and position at the start and end
    # bout_results is a list of 9 lists, one for each type of stimulus
    bout_results = [[] for i in range(9)]
    for i in range(n_bouts):
        # get the stim id, frame where it starts and frame when it ends
        stim_id     = stim_id_frames[bout_start_frames[i]]
        start_frame = bout_start_frames[i]
        end_frame   = bout_end_frames[i]

        # add to the mean bout length variable
        mean_bout_length += frame_milliseconds[end_frame+1] - frame_milliseconds[start_frame]

        # print("Bout {} starts at frame {} and ends at frame {}.".format(i, start_frame, end_frame))

        # save the heading angle & position at the start & end of the bout, and the video name
        results = {'heading_angle_start': heading_angle[start_frame],
                   'heading_angle_end'  : heading_angle[end_frame],
                   'position_start'     : (body_position[start_frame, 0], body_position[start_frame, 1]),
                   'position_end'       : (body_position[end_frame, 0], body_position[end_frame, 1]),
                   'video'              : video_name }

        # add to the bout_results list
        bout_results[stim_id].append(results)

    # get the mean bout length
    mean_bout_length /= n_bouts
    print("Mean bout length is {} ms.".format(mean_bout_length))

    # determine, for each type of stimulus, the heading angle and position at the start and end
    # stim_results is a list of 9 lists, one for each type of stimulus
    stim_results = [[] for i in range(9)]
    for i in range(n_stim_switches):
        # get the stim id, frame where it starts and frame when it ends
        stim_id     = stim_ids[i]
        start_frame = stim_switch_frame_nums[i]
        if i < n_stim_switches - 1:
            end_frame   = stim_switch_frame_nums[i+1] - 1
        else:
            end_frame = n_frames - 1

        # print("Stimulus {} starts at frame {} and ends at frame {}.".format(i, start_frame, end_frame))

        # save the heading angle & position at the start & end of the bout, and the video name
        results = {'heading_angle_start': heading_angle[start_frame],
                   'heading_angle_end'  : heading_angle[end_frame],
                   'position_start'     : (body_position[start_frame, 0], body_position[start_frame, 1]),
                   'position_end'       : (body_position[end_frame, 0], body_position[end_frame, 1]),
                   'video'              : video_name }

        # add to the stim_results list
        stim_results[stim_id].append(results)

    if plot:
        # plot results
        fig, ax = plt.subplots()
        ax.plot(heading_angle, 'black', lw=1)
        ax.plot(body_distance, 'purple', lw=1)
        # ax.plot(smoothed_body_distance_1, 'red', lw=1)
        # ax.plot(running_body_distance_difference, 'green', lw=1)
        # ax.plot(body_distance_difference_above_threshold, 'blue', lw=1)
        # ax.fill_between(np.arange(len(running_heading_angle_difference)), np.amin(np.nan_to_num(smoothed_heading_angle)), np.amax(np.nan_to_num(smoothed_heading_angle)), where=heading_angle_difference_above_threshold.astype(bool), facecolor='black', alpha=0.2)
        ax.fill_between(np.arange(len(running_body_distance_difference)), np.amin(np.nan_to_num(body_distance)), np.amax(np.nan_to_num(body_distance)), where=combined_difference_above_threshold.astype(bool), facecolor='black', alpha=0.2)

        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'brown', 'black', 'cyan', 'magenta']
        stims = ['Circular Grating', 'Left Grating', 'Right Grating', 'Left Dot', 'Right Dot', 'Left Looming', 'Right Looming', 'White', 'Black']
        for i in range(n_stim_switches):
            stim_active = stim_id_frames == stim_ids[i]
            if i < n_stim_switches - 1:
                stim_active[stim_switch_frame_nums[i+1]] = 1
            else:
                stim_active[-1] = 1
            ax.fill_between(np.arange(len(running_heading_angle_difference)), np.amin(np.nan_to_num(heading_angle)), np.amax(np.nan_to_num(heading_angle)), where=stim_active.astype(bool), facecolor=colors[stim_ids[i]], alpha=0.2)
            ax.text(stim_switch_frame_nums[i] + 10, 0, stims[stim_ids[i]], fontsize=8, alpha=0.5)
        plt.show()

    return bout_results, stim_results, mean_bout_length

if __name__ == "__main__":
    folder = "Results 2"
    suffix = "_Image-Data_Video_tracking"

    # get video names using the .npz tracking data filename
    video_names = [ os.path.basename(filename)[:-len(suffix)-4] for filename in os.listdir(folder) if filename.endswith('.npz') ]

    # get number of videos
    n_videos = len(video_names)

    print("Videos to process:")
    for i in range(n_videos):
        video_name = video_names[i]
        print("{}.\t{}".format(i+1, video_name))
    print("----------------------------------------------------------\n")

    # create lists to store all of the data from all of the videos
    full_bout_results = [[] for i in range(9)]
    full_stim_results = [[] for i in range(9)]

    # initialize variable to calcualate the mean bout length in milliseconds
    full_mean_bout_length = 0

    for i in range(n_videos):
        video_name = video_names[i]
        print("Video {}: {}.".format(i+1, video_name))
        bout_results, stim_results, mean_bout_length = process_video(folder, video_name, plot=True)

        # add to the mean bout length variable
        full_mean_bout_length += mean_bout_length

        for i in range(9):
            full_bout_results[i] += bout_results[i]
            full_stim_results[i] += stim_results[i]

        print("----------------------------------------------------------\n")

    # get the total number of bouts
    n_bouts = sum([ len(l) for l in full_bout_results ])
    print("Total number of bouts: {}.".format(n_bouts))

    # get the mean bout length
    full_mean_bout_length /= n_videos
    print("Mean bout length is {} ms.".format(full_mean_bout_length))

    # save data
    json.dump(full_bout_results, open(os.path.join(folder, "full_bout_results.txt"), 'w'))
    json.dump(full_stim_results, open(os.path.join(folder, "full_stim_results.txt"), 'w'))