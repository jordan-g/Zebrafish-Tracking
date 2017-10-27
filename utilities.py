import open_media
import cv2
import peakdetect
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

try:
    xrange
except:
    xrange = range

def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def estimate_thresholds(frame, delta=0.0001, n_bins=20, plot_histogram=False):
    # get number of pixels in the frame
    n_pixels = np.product(frame.shape)

    # get pixel brightness histogram
    hist = cv2.calcHist([frame], [0], None, [n_bins], [0, 256])

    if plot_histogram:
        x = np.linspace(0, 256 - (256/n_bins), n_bins)
        plt.bar(x, hist, width=(256/n_bins))
        plt.show()

    # get lowest peak location
    max_peaks, _ = peakdetect.peakdet(hist, delta*n_pixels)

    peak_indices = max_peaks[:, 0]
    peak_locations = peak_indices*(256/n_bins)

    if len(peak_locations) >= 4:
        est_eye_threshold  = int(peak_locations[0])
        est_body_threshold = int(peak_locations[1])
        est_tail_threshold = int(peak_locations[2])
    elif len(peak_locations) >= 3:
        est_eye_threshold  = int(peak_locations[0])
        est_body_threshold = int(peak_locations[0]) + 30
        est_tail_threshold = int(peak_locations[1])
    elif len(peak_locations) >= 2:
        est_eye_threshold  = int(peak_locations[0])
        est_body_threshold = int(peak_locations[0]) + 30
        est_tail_threshold = int(peak_locations[0]) + 70
    else:
        est_eye_threshold  = 50
        est_body_threshold = 80
        est_tail_threshold = 120

    print("Estimated thresholds: {}, {}, {}.".format(est_tail_threshold, est_body_threshold, est_eye_threshold))

    return est_tail_threshold, est_body_threshold, est_eye_threshold

def translate_interpolation(interpolation_string):
    # get matching opencv interpolation variable from string
    if interpolation_string == 'Nearest Neighbor':
        interpolation = cv2.INTER_NEAREST
    elif interpolation_string == 'Linear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation_string == 'Bicubic':
        interpolation = cv2.INTER_CUBIC
    elif interpolation_string == 'Lanczos':
        interpolation = cv2.INTER_LANCZOS4

    return interpolation

def split_evenly(n, m, start=0):
    # generate a list of m evenly spaced numbers in the range of (start, start + n)
    # eg. split_evenly(100, 5, 30) = [40, 60, 80, 100, 120]
    return [i*n//m + n//(2*m) + start for i in range(m)]

def yield_chunks_from_array(array, n):
    '''
    Yield successive n-sized chunks from an array.

    Arguments:
        array (ndarray) : Input array.
        n (int)         : Size of each chunk to yield.

    Yields:
        chunk (ndarray) : n-sized chunk from the input array.
    '''
    for i in xrange(0, array.shape[0], n):
        yield array[i:i + n]

def split_list_into_chunks(l, n):
    '''
    Return a list of n-sized chunks from l.

    Arguments:
        l (list) : Input list.
        n (int)  : Size of each chunk.

    Returns:
        l_2 (list) : List of n-sized chunks of l.
    '''

    return [ l[i:i + n] for i in xrange(0, len(l), n) ]