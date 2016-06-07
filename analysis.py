import numpy as np
import matplotlib.pyplot as plt
import os

def get_heading_angle(save_dir=None, plot=True, perp_y_coords_array=None, perp_x_coords_array=None):
	if save_dir != None:
		perp_y_coords_array = np.loadtxt(os.path.join(save_dir, "perp_y_coords_array.csv"))
		perp_x_coords_array = np.loadtxt(os.path.join(save_dir, "perp_x_coords_array.csv"))

	y_diff_array = perp_y_coords_array[:, 1] - perp_y_coords_array[:, 0]
	x_diff_array = perp_x_coords_array[:, 1] - perp_x_coords_array[:, 0]

	y_diff_array[y_diff_array == 0] = np.nan
	x_diff_array[x_diff_array == 0] = np.nan

	angle_array = np.arctan2(x_diff_array, y_diff_array)*180.0/np.pi

	if plot:
		plt.plot(np.arange(angle_array.shape[0]), angle_array)
		plt.show()

	return angle_array

def get_tail_angle(save_dir=None, plot=True, tail_end_y_coords_array=None, tail_end_x_coords_array=None):
	if save_dir != None:
		tail_end_y_coords_array = np.loadtxt(os.path.join(save_dir, "tail_end_y_coords_array.csv"))
		tail_end_x_coords_array = np.loadtxt(os.path.join(save_dir, "tail_end_x_coords_array.csv"))

	y_diff_array = tail_end_y_coords_array[:, 1] - tail_end_y_coords_array[:, 0]
	x_diff_array = tail_end_x_coords_array[:, 1] - tail_end_x_coords_array[:, 0]

	y_diff_array[y_diff_array == 0] = np.nan
	x_diff_array[x_diff_array == 0] = np.nan

	angle_array = np.arctan2(x_diff_array, y_diff_array)*180.0/np.pi

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

	positions_y[positions_y == 0] = np.nan
	positions_x[positions_x == 0] = np.nan

	plt.plot(positions_x, positions_y)
	plt.show()