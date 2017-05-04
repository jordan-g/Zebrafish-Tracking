import utilities
import open_media

frame = open_media.open_video("Footage/8_fish.mov", [0])[0]

utilities.estimate_thresholds(frame, delta=0.00001, n_bins=50, plot_histogram=True)