import numpy as np
import matplotlib.pyplot as plt
import os

def open_saved_data(save_dir=None, k=0):
	"""
	Open saved tracking data from the given directory.
	"""
	try:
		npzfile = np.load(os.path.join(save_dir, "crop_{}_tracking_data.npy.npz".format(k)))
		# print("hi")
		# print(npzfile.files)
		eye_coords_array    = npzfile['eye_coords']
		perp_coords_array = npzfile['heading_coords']
		tail_coords_array = npzfile['tail_coords']
		spline_coords_array = npzfile['spline_coords']
		params = npzfile['params'][()]

		# print(eye_coords_array)
		# try:
		# 	perp_coords_array   = np.load(os.path.join(save_dir, "heading_coords.npy"))
		# except:
		# 	perp_coords_array   = np.load(os.path.join(save_dir, "perp_coords.npy"))
		# tail_coords_array   = np.load(os.path.join(save_dir, "tail_coords.npy"))
		# spline_coords_array = np.load(os.path.join(save_dir, "spline_coords.npy"))
	except:
		print("ERROR: Tracking data could not be found.")
		return [None]*4

	return eye_coords_array, perp_coords_array, tail_coords_array, spline_coords_array, params

def get_vectors(perp_coords_array, spline_coords_array, tail_coords_array):
	"""
	Turn arrays of heading & tail spline coordinates into vectors.
	"""
	# create heading vectors -- starting coordinate minus ending coordinate of the heading line
	perp_vectors = perp_coords_array[:, :, 0] - perp_coords_array[:, :, 1]

	# get distances between start/end coordinates of the heading line and the starting coordinates of the tail
	spline_distances = np.sqrt((perp_coords_array[:, 0, :] - spline_coords_array[:, 0, -1][:, np.newaxis])**2 + (perp_coords_array[:, 1, :] - spline_coords_array[:, 1, -1][:, np.newaxis])**2)
	
	# get frames where the "starting" heading coordinate is closer to the tail than the "ending" coordinate
	mask = spline_distances[:, 0] < spline_distances[:, 1]

	# flip heading vectors for these frames, so that all vectors point toward the tail.
	perp_vectors[mask, :] *= -1

	# get vectors of tail coordinates relative to the heading vectors
	spline_vectors = perp_coords_array[:, :, 1][:, :, np.newaxis] - spline_coords_array

	return perp_vectors, spline_vectors

def get_heading_angle(perp_vectors):
	angle_array = np.arctan2(perp_vectors[:, 1], perp_vectors[:, 0])

	angle_array[~np.isnan(angle_array)] = np.unwrap(angle_array[~np.isnan(angle_array)])

	for i in range(1, angle_array.shape[0]-1):
		if angle_array[i] - angle_array[i-1] >= np.pi/2.0:
			angle_array[i] -= np.pi/2.0
		elif angle_array[i] - angle_array[i-1] <= -np.pi/2.0:
			angle_array[i] += np.pi/2.0

	return angle_array

def get_tail_angle(perp_vectors, spline_vectors):
	spline_vectors = np.mean(spline_vectors[:, :, -3:], axis=2)

	dot = spline_vectors[:, 1]*perp_vectors[:, 1] + spline_vectors[:, 0]*perp_vectors[:, 0]      # dot product
	det = spline_vectors[:, 1]*perp_vectors[:, 0] - spline_vectors[:, 0]*perp_vectors[:, 1]      # determinant

	angle_array = np.arctan2(dot, det) - np.pi/2.0

	for i in range(1, angle_array.shape[0]-1):
		if angle_array[i] - angle_array[i-1] >= np.pi/2.0:
			angle_array[i] -= np.pi/2.0
		elif angle_array[i] - angle_array[i-1] <= -np.pi/2.0:
			angle_array[i] += np.pi/2.0

	return angle_array

def get_position_history(save_dir=None, plot=True, perp_y_coords_array=None, perp_x_coords_array=None):
	if save_dir != None:
		perp_y_coords_array = np.loadtxt(os.path.join(save_dir, "heading_y_coords_array.csv"))
		perp_x_coords_array = np.loadtxt(os.path.join(save_dir, "heading_x_coords_array.csv"))

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