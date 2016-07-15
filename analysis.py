import numpy as np
import matplotlib.pyplot as plt
import os

def get_heading_angle(save_dir=None, plot=True, perp_y_coords_array=None, perp_x_coords_array=None):
	if save_dir != None:
		perp_y_coords_array = np.loadtxt(os.path.join(save_dir, "perp_y_coords_array.csv"))
		perp_x_coords_array = np.loadtxt(os.path.join(save_dir, "perp_x_coords_array.csv"))

	y_diff_array = perp_y_coords_array[:, 1] - perp_y_coords_array[:, 0]
	x_diff_array = perp_x_coords_array[:, 1] - perp_x_coords_array[:, 0]

	# y_diff_array[y_diff_array == 0] = np.nan
	# x_diff_array[x_diff_array == 0] = np.nan

	angle_array = np.unwrap(np.arctan2(x_diff_array, y_diff_array))

	angle_array = np.convolve(angle_array, np.ones((3,))/3, mode='valid')

	if plot:
		plt.plot(np.arange(angle_array.shape[0]), angle_array)
		plt.show()

	return angle_array

def get_tail_angle(save_dir=None, plot=True, tail_end_y_coords_array=None, tail_end_x_coords_array=None, heading_angle_array=None):
	if save_dir != None:
		tail_end_y_coords_array = np.loadtxt(os.path.join(save_dir, "tail_end_y_coords_array.csv"))
		tail_end_x_coords_array = np.loadtxt(os.path.join(save_dir, "tail_end_x_coords_array.csv"))
		tail_y_coords_array     = np.loadtxt(os.path.join(save_dir, "tail_y_coords_array.csv"))
		tail_x_coords_array     = np.loadtxt(os.path.join(save_dir, "tail_x_coords_array.csv"))
		perp_y_coords_array     = np.loadtxt(os.path.join(save_dir, "perp_y_coords_array.csv"))
		perp_x_coords_array     = np.loadtxt(os.path.join(save_dir, "perp_x_coords_array.csv"))

	center_y_pos_array = (perp_y_coords_array[:, 1] + perp_y_coords_array[:, 0])/2.0
	center_x_pos_array = (perp_x_coords_array[:, 1] + perp_x_coords_array[:, 0])/2.0

	# y_diff_array = tail_end_y_coords_array[:, 1] - tail_end_y_coords_array[:, 0]
	# x_diff_array = tail_end_x_coords_array[:, 1] - tail_end_x_coords_array[:, 0]

	y_diff_array = center_y_pos_array - np.mean(tail_end_y_coords_array[:, -2:], axis=1)
	x_diff_array = center_x_pos_array - np.mean(tail_end_x_coords_array[:, -2:], axis=1)
	
	# y_diff_array[y_diff_array == 0] = np.nan
	# x_diff_array[x_diff_array == 0] = np.nan

	angle_array = np.arctan2(x_diff_array, y_diff_array)

	angle_array = np.convolve(angle_array, np.ones((3,))/3, mode='valid')
	# if heading_angle_array is not None:
	# 	angle_array += heading_angle_array


	if plot:
		plt.plot(np.arange(angle_array.shape[0]), angle_array)
		plt.show()

	return angle_array

def get_position_history(save_dir=None, plot=True, perp_y_coords_array=None, perp_x_coords_array=None):
	if save_dir != None:
		perp_y_coords_array = np.loadtxt(os.path.join(save_dir, "perp_y_coords_array.csv"))
		perp_x_coords_array = np.loadtxt(os.path.join(save_dir, "perp_x_coords_array.csv"))

	positions_y = (perp_y_coords_array[:, 0]+perp_y_coords_array[:, 1])/2.0
	positions_x = (perp_x_coords_array[:, 0]+perp_x_coords_array[:, 1])/2.0

	speed = np.sqrt(np.gradient(positions_y)**2 + np.gradient(positions_x)**2)

	speed = np.convolve(speed, np.ones((3,))/3, mode='valid')

	# positions_y[positions_y == 0] = np.nan
	# positions_x[positions_x == 0] = np.nan

	if plot:
		plt.plot(positions_x, positions_y)
		plt.show()

	return positions_y, positions_x, speed

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

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