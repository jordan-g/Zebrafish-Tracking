import open_media
import cv2
import peakdetect
import numpy as np
import matplotlib.pyplot as plt

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