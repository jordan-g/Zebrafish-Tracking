from param_window import *
from preview_window import *
from crops_window import *

import tracking
import open_media
import utilities

import time

# import the Qt library
try:
    from PyQt4.QtCore import pyqtSignal, Qt, QThread
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
    pyqt_version = 4
except:
    from PyQt5.QtCore import pyqtSignal, Qt, QThread
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
    pyqt_version = 5

try:
    xrange
except:
    xrange = range

DEFAULT_HEADFIXED_CROP_PARAMS = { 'offset': np.array([0, 0]), # crop (y, x) offset
                                  'crop': None }              # crop area

DEFAULT_FREESWIMMING_CROP_PARAMS = { 'offset': np.array([0, 0]), # crop (y, x) offset
                                     'crop': None,               # crop area
                                     'body_threshold': 140,      # pixel brightness to use for thresholding to find the body (0-255)
                                     'eyes_threshold': 60,       # pixel brightness to use for thresholding to find the eyes (0-255)
                                     'tail_threshold': 200 }     # pixel brightness to use for thresholding to find the tail (0-255)

DEFAULT_HEADFIXED_PARAMS = {'crop_params': [],
                            'dark_background': False,              # whether the video has a dark background and light fish
                            'type': "headfixed",                   # "headfixed" / "freeswimming"
                            'scale_factor': 1.0,                   # factor by which to down/upscale the original video
                            'interpolation': 'Nearest Neighbor',   # interpolation to use when down/upscaling the original video
                            'save_video': True,                    # whether to make a video with tracking overlaid
                            'tracking_video_fps': 0,               # fps for the generated video
                            'n_tail_points': 30,                   # number of tail points to use
                            'subtract_background': False,          # whether to perform background subtraction
                            'heading_direction': "Down",           # "Right" / "Left" / "Up" / "Down"
                            'heading_angle': 0,                    # heading angle (degrees) - overrides the heading direction parameter
                            'bg_sub_threshold': 30,                # threshold used in background subtraction
                            'video_paths': [],                     # paths to videos that will be tracked
                            'backgrounds': [],                     # backgrounds calculated for background subtraction
                            'tail_start_coords': None,             # (y, x) coordinates of the start of the tail
                            'use_multiprocessing': True,           # whether to use multiprocessing
                            'gui_params': { 'auto_track': False }} # automatically track a frame when you switch to it

DEFAULT_FREESWIMMING_PARAMS = {'crop_params': [],
                               'dark_background': False,                     # whether the video has a dark background and light fish
                               'type': "freeswimming",                       # "headfixed" / "freeswimming"
                               'scale_factor': 1.0,                          # factor by which to down/upscale the original video
                               'interpolation': 'Nearest Neighbor',          # interpolation to use when down/upscaling the original video
                               'save_video': True,                           # whether to make a video with tracking overlaid
                               'tracking_video_fps': 30,                     # fps for the generated video
                               'n_tail_points': 30,                          # number of tail points to use
                               'adjust_thresholds': False,                   # whether to adjust thresholds while tracking if necessary
                               'subtract_background': False,                 # whether to perform background subtraction
                               'track_tail': True,                           # whether to track the tail
                               'track_eyes': True,                           # whether to track the eyes
                               'min_tail_body_dist': 10,                     # min. distance between the body center and the tail
                               'max_tail_body_dist': 30,                     # max. distance between the body center and the tail
                               'bg_sub_threshold': 30,                       # threshold used in background subtraction
                               'body_crop': np.array([100, 100]),            # dimensions of crop around zebrafish body to use for tail tracking - (height, width)
                               'video_paths': [],                            # paths to videos that will be tracked
                               'backgrounds': [],                            # backgrounds calculated for background subtraction
                               'use_multiprocessing': True,                  # whether to use multiprocessing
                               'alt_tail_tracking': False,                   # whether to use alternate slower, but more accurate tail tracking
                               'gui_params': { 'show_body_threshold': False, # show body threshold in preview window
                                               'show_eyes_threshold': False, # show eye threshold in preview window
                                               'show_tail_threshold': False, # show tail threshold in preview window
                                               'show_tail_skeleton': False,  # show tail skeleton in preview window
                                               'auto_track': False,          # automatically track a frame when you switch to it
                                               'zoom_body_crop': False }}    # automatically zoom to fit body crop

max_n_frames = 100 # maximum # of frames to load for previewing

class Controller():
    def __init__(self, default_params, default_crop_params):
        # set parameters to the default parameters
        self.default_params = default_params.copy()
        self.params         = self.default_params.copy()

        # save default crop parameters
        self.default_crop_params = default_crop_params.copy()

        # initialize variables
        self.current_frame      = None # currently showing frame (unscaled and uncropped)
        self.scaled_frame       = None # scaled frame
        self.cropped_frame      = None # cropped frame
        self.body_cropped_frame = None # frame cropped around the body

        self.frames                = [] # list of loaded frames, one element per video
        self.bg_sub_frames         = [] # list of background subtracted frames, one element per video
        self.tracking_results      = [] # list of tracking results, one element per video
        self.background_calc_paths = [] # list of video paths for which backgrounds are being calculated, one element per video
        self.background_progress   = [] # list of background calculation progresses, one element per video

        self.current_crop   = -1   # which crop is being looked at (-1 means no crop is loaded)
        self.curr_video_num = 0    # which video (from a loaded batch) is being looked at
        self.n_frames       = 0    # total number of frames to preview
        self.n              = 0    # index of currently selected frame
        self.tracking_path  = None # path to where tracking data will be saved

        self.get_backgrounds_thread = None # thread used for calculating backgrounds
        self.track_videos_thread    = None # thread used for tracking videos

        self.closing    = False # whether the user is closing the app
        self.first_load = True  # False if we are reloading parameters or videos have already been loaded; True otherwise
        self.tracking   = False # whether tracking is currently being done

    def select_and_open_videos(self):
        # ask the user to select one or more videos to open
        if pyqt_version == 4:
            new_video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to track.', '', 'Videos (*.mov *.mp4 *.avi)')
        elif pyqt_version == 5:
            new_video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to track.', '', 'Videos (*.mov *.mp4 *.avi)')[0]

            # convert paths to strings
            new_video_paths = [ str(video_path) for video_path in new_video_paths ]

        # ignore videos that are already loaded
        for i in range(len(new_video_paths)):
            if new_video_paths[i] in self.params['video_paths']:
                del new_video_paths[i]

        if len(new_video_paths) > 0 and new_video_paths[0] != '':
            if self.first_load:
                # set params to defaults
                self.params = self.default_params.copy()

                # reset the current crop
                self.current_crop = -1

            # open the videos
            self.open_video_batch(new_video_paths)

            if self.first_load:
                if self.frames[0] is None:
                    # no frames found; end here
                    return

                # create a crop
                self.create_crop()

                self.first_load = False

            # show the first frame
            self.show_frame(0, new_load=True)

    def open_video_batch(self, video_paths):
        if (not self.first_load) or len(self.params['video_paths']) == 0:
            # update variables to account for the new videos
            self.params['video_paths'] += video_paths
            self.params['backgrounds'] += [None for i in range(len(video_paths))]

        # append to lists
        self.frames              += [None for i in range(len(video_paths))]
        self.bg_sub_frames       += [None for i in range(len(video_paths))]
        self.background_progress += [0 for i in range(len(video_paths))]

        if self.first_load:
            # update current video number to be 0
            self.curr_video_num = 0

        # update loaded videos label
        self.param_window.update_videos_loaded_text(len(self.params['video_paths']), self.curr_video_num)

        if self.first_load:
            # open the first video from the batch
            self.open_video(self.params["video_paths"][self.curr_video_num])

            if self.frames[self.curr_video_num] is None:
                # no frames found; end here
                return

        # get the indices of the newly loaded videos
        new_video_indices = [ len(self.params['video_paths']) - len(video_paths) + i for i in range(len(video_paths)) ]

        # get the paths of videos for which the background needs to be calculated
        background_calc_paths = [ self.params['video_paths'][k] for k in new_video_indices if self.params['backgrounds'][k] is None ]

        # automatically try to determine whether the video has a dark background with a light fish
        if np.mean(self.frames[self.curr_video_num]) < 100:
            self.param_window.param_controls['dark_background'].setChecked(True)
            self.params['dark_background'] = True

        # add the new video paths to the videos list in the param window
        for k in new_video_indices:
            self.param_window.add_video_item(os.path.basename(self.params['video_paths'][k]))

        # set the selected video in the list
        if self.first_load:
            self.param_window.change_selected_video_row(self.curr_video_num)

        # clear the invalid params text
        self.param_window.set_invalid_params_text("")

        if len(background_calc_paths) > 0:
            # reset the background progress text
            self.param_window.update_background_progress_text(len(background_calc_paths), 0)

            # create a background calculation thread if it doesn't exist
            if self.get_backgrounds_thread is None:
                self.get_backgrounds_thread = GetBackgroundThread(self.param_window, self.params['dark_background'])
                self.get_backgrounds_thread.progress.connect(self.background_calculation_progress)
                self.get_backgrounds_thread.finished.connect(self.background_calculated)

            # add the new video paths for it to process
            self.get_backgrounds_thread.add_video_paths(background_calc_paths)
            self.background_calc_paths += background_calc_paths

            # start the thread
            self.get_backgrounds_thread.start()

    def open_video(self, video_path):
        # reset tracking results
        self.tracking_results = []

        if video_path not in ("", None):
            # get video info
            fps, n_frames_total = open_media.get_video_info(video_path)
            
            print("Loaded video with {} frame{}.".format(n_frames_total, "s"*(n_frames_total > 1)))

            # load evenly spaced frames -- TODO: Add an option to load sequential frames
            frame_nums = utilities.split_evenly(min(n_frames_total, 50000), max_n_frames)

            # load frames from the video
            self.frames[self.curr_video_num] = open_media.open_video(video_path, frame_nums, True, greyscale=True, seek_to_starting_frame=False)

            # get background-subtracted frames if the background has already been calculated
            if self.params['backgrounds'][self.curr_video_num] is not None:
                self.bg_sub_frames[self.curr_video_num] = tracking.subtract_background_from_frames(self.frames[self.curr_video_num], self.params['backgrounds'][self.curr_video_num], self.params['bg_sub_threshold'], dark_background=self.params['dark_background'])

            if self.frames[self.curr_video_num] is None:
                # no frames found; end here
                print("Error: Could not load frames.")
                return

            # set the current frame to the first frame
            self.current_frame = self.frames[self.curr_video_num][0]

            # get number of loaded frames
            self.n_frames = len(self.frames[self.curr_video_num])

            if self.params['type'] == "headfixed":
                # estimate tail direction
                total_luminosities = [np.sum(self.current_frame[0:10, :]), np.sum(self.current_frame[:, 0:10]),
                                      np.sum(self.current_frame[-1:-11, :]), np.sum(self.current_frame[:, -1:-11])]

                self.params['heading_direction'] = heading_direction_options[np.argmin(total_luminosities)]

                self.param_window.update_gui_from_params(self.params)

            # enable GUI controls
            self.param_window.set_gui_disabled(False)

    def prev_video(self):
        if self.curr_video_num != 0:
            # update current video number
            self.curr_video_num -= 1

            # update loaded video label
            self.param_window.update_videos_loaded_text(len(self.params['video_paths']), self.curr_video_num)

            if self.frames[self.curr_video_num] is None:
                # open the previous video from the batch
                self.open_video(self.params['video_paths'][self.curr_video_num])

            # switch to first frame
            self.show_frame(0, new_load=True)

            self.param_window.change_selected_video_row(self.curr_video_num)

    def next_video(self):
        if self.curr_video_num != len(self.params['video_paths'])-1:
            # update current video number
            self.curr_video_num += 1

            # update loaded video label
            self.param_window.update_videos_loaded_text(len(self.params['video_paths']), self.curr_video_num)

            if self.frames[self.curr_video_num] is None:
                # open the next video from the batch
                self.open_video(self.params['video_paths'][self.curr_video_num])

            # switch to first frame
            self.show_frame(0, new_load=True)

            self.param_window.change_selected_video_row(self.curr_video_num)

    def switch_video(self, video_num):
        if 0 <= video_num <= len(self.params['video_paths'])-1:
            # update current video number
            self.curr_video_num = video_num

            # update loaded video label
            self.param_window.update_videos_loaded_text(len(self.params['video_paths']), self.curr_video_num)

            if self.frames[self.curr_video_num] is None:
                # open the next video from the batch
                self.open_video(self.params['video_paths'][self.curr_video_num])

            # switch to first frame
            self.show_frame(0, new_load=True)

    def remove_video(self):
        if self.params['video_paths'][self.curr_video_num] in self.background_calc_paths:
            self.background_calc_paths.remove(self.params['video_paths'][self.curr_video_num])

        del self.params['video_paths'][self.curr_video_num]
        del self.params['backgrounds'][self.curr_video_num]
        del self.frames[self.curr_video_num]
        del self.bg_sub_frames[self.curr_video_num]

        self.param_window.remove_video_item(self.curr_video_num)

        if self.curr_video_num == len(self.params['video_paths']):
            self.curr_video_num -= 1

        if self.curr_video_num != -1:
            if self.frames[self.curr_video_num] is None:
                # open the next video from the batch
                self.open_video(self.params['video_paths'][self.curr_video_num])

            # switch to first frame
            self.show_frame(0, new_load=True)
        else:
            # reset everything
            self.current_frame = None
            self.preview_window.plot_image(None, None, None, None)
            self.first_load = True
            self.param_window.param_controls["subtract_background"].setText("Subtract background")
            self.param_window.set_gui_disabled(True)
            self.clear_crops()

        # update loaded video label
        self.param_window.update_videos_loaded_text(len(self.params['video_paths']), self.curr_video_num)

    def track_videos(self):
        # get save path
        self.tracking_path = str(QFileDialog.getExistingDirectory(self.param_window, "Select Directory"))

        if self.tracking_path != "":
            if self.get_backgrounds_thread is not None:
                self.get_backgrounds_thread.running = False

            # track videos
            if self.track_videos_thread is not None:
                # another thread is already tracking something; don't let it affect the GUI
                self.track_videos_thread.progress.disconnect(self.update_video_tracking_progress)
                self.track_videos_thread.finished.disconnect(self.videos_tracked)

            # create new thread to track the videos
            self.track_videos_thread = TrackVideosThread(self.param_window)
            self.track_videos_thread.set_parameters(self.params, self.tracking_path)

            # set callback function to be called when the videos has been tracked
            self.track_videos_thread.finished.connect(self.videos_tracked)

            # set callback function to be called as the videos are being tracked (to show progress)
            self.track_videos_thread.progress.connect(self.update_video_tracking_progress)

            n_videos = len(self.params['video_paths'])

            self.param_window.update_tracking_progress_text(n_videos, 0, 0)

            # start thread
            self.track_videos_thread.start()

            self.tracking = True

    def background_calculation_progress(self, progress, video_path):
        if video_path in self.params["video_paths"]:
            self.background_progress[self.params["video_paths"].index(video_path)] = progress
            n_backgrounds_calculated       = sum([ x is not None for x in self.params['backgrounds'] ])
            n_backgrounds_being_calculated = sum([ x is None for x in self.params['backgrounds'] ])
            n_backgrounds_total            = len(self.params['backgrounds'])

            if n_backgrounds_total == 1:
                true_progress = min(progress, 100)
            else:
                true_progress = min(sum(self.background_progress)/n_backgrounds_being_calculated, 100)

            # update tracking progress label in param window
            self.param_window.update_background_progress_text(n_backgrounds_total - n_backgrounds_calculated, true_progress)

    def background_calculated(self, background, video_path):
        if np.sum(background) == 0:
            background = None
            self.param_window.background_progress_text = ""
        elif video_path in self.background_calc_paths:
            self.background_progress = [0 for i in range(len(self.params['video_paths']))]
            print("Background for {} calculated.".format(video_path))

            video_num = self.params['video_paths'].index(video_path)

            # update params
            self.params['backgrounds'][video_num] = background

            n_backgrounds_calculated = sum([ x is not None for x in self.params['backgrounds'] ])
            n_backgrounds_total      = len(self.params['backgrounds'])
            
            if n_backgrounds_calculated == n_backgrounds_total or background is None:
                self.param_window.update_background_progress_text(n_backgrounds_total - n_backgrounds_calculated, 100)

            if self.curr_video_num == video_num:
                if background is not None:
                    if self.frames[video_num] is not None:
                        # generate background subtracted frames
                        self.bg_sub_frames[video_num] = tracking.subtract_background_from_frames(self.frames[video_num], self.params['backgrounds'][video_num], self.params['bg_sub_threshold'], dark_background=self.params['dark_background'])

                    if self.params['subtract_background'] == True:
                        # reshape the image
                        self.show_frame(self.n)

                    self.param_window.open_background_action.setEnabled(True)
                    self.param_window.save_background_action.setEnabled(True)

    def videos_tracked(self, tracking_time):
        self.param_window.update_tracking_progress_text(1, 0, 100, tracking_time)
        self.tracking = False

    def update_video_tracking_progress(self, video_number, percent):
        n_videos = len(self.params['video_paths'])

        self.param_window.update_tracking_progress_text(n_videos, video_number, percent)

    def load_params(self, select_path=True):
        if select_path:
            # ask the user to select a path
            params_path = str(QFileDialog.getOpenFileName(self.param_window, 'Open saved parameters', '')[0])
        else:
            params_path = self.last_params_path

        if params_path not in ("", None):
            # load params from saved file
            try:
                params_file = np.load(params_path)
            except:
                return
            saved_params = params_file['params'][()]

            # set params to saved params
            incomplete_load = False
            self.params = self.default_params.copy()
            for key in saved_params:
                if key in self.params:
                    self.params[key] = saved_params[key]
                else:
                    incomplete_load = True

            self.current_crop = -1

            # re-open the video paths specified in the loaded params
            self.open_video_batch(self.params['video_paths'])

            self.first_load = False

            self.param_window.crop_tabs_widget.blockSignals(True)
            # create tabs for all saved crops
            for j in range(len(self.params['crop_params'])):
                self.current_crop += 1
                self.param_window.create_crop_tab(self.params['crop_params'][j])
            self.param_window.crop_tabs_widget.blockSignals(False)

            # update gui controls
            self.param_window.update_gui_from_params(self.params)
            self.param_window.update_gui_from_crop_params(self.params['crop_params'])

            # switch to first frame
            self.show_frame(0, new_load=True)

            if incomplete_load:
                self.param_window.set_invalid_params_text("Some parameters couldn't be loaded and were set to their default values.")
            else:
                self.param_window.set_invalid_params_text("")
        else:
            pass

    def save_params(self, select_path=True):
        if select_path:
            # ask user to select a path
            params_path = str(QFileDialog.getSaveFileName(self.param_window, 'Choose directory to save in', '')[0])
        else:
            # set params path to last used params path
            params_path = self.last_params_path

        if params_path not in ("", None):
            # save params to file
            np.savez(params_path, params=self.params)
        else:
            pass

    def show_frame(self, n, new_load=False):
        print("Showing frame {}.".format(n))
        if n != self.n:
            # reset tracking results
            self.tracking_results = None

        # set current frame index
        if n is not None:
            self.n = n

        # set current frame
        if self.params['subtract_background'] and self.bg_sub_frames[self.curr_video_num] is not None:
            self.current_frame = self.bg_sub_frames[self.curr_video_num][self.n]
        else:
            self.current_frame = self.frames[self.curr_video_num][self.n]

        # reshape the image (shrink, crop & invert)
        self.reshape_frame()

        if self.params['type'] == "freeswimming":
            # generate thresholded frames
            self.generate_thresholded_frames()

        # update the image preview
        self.update_preview(None, new_load, new_frame=True)

        if self.params['gui_params']['auto_track']:
            self.track_frame()

    def reshape_frame(self):
        # reset tracking results
        self.tracking_results = None

        if self.current_frame is not None:
            # get params of currently selected crop
            current_crop_params = self.params['crop_params'][self.current_crop]

            # crop the frame
            if current_crop_params['crop'] is not None and current_crop_params['crop'] is not None:
                crop   = current_crop_params['crop']
                offset = current_crop_params['offset']

                self.cropped_frame = tracking.crop_frame(self.current_frame, offset, crop)
            else:
                self.cropped_frame = self.current_frame

            # scale the frame
            if self.params['scale_factor'] is not None:
                self.scaled_frame = tracking.scale_frame(self.cropped_frame, self.params['scale_factor'], utilities.translate_interpolation(self.params['interpolation']))
            else:
                self.scaled_frame = self.cropped_frame

    def update_preview(self, image=None, new_load=False, new_frame=False):
        if image is None:
            # use the cropped current frame by default
            image = self.scaled_frame

        if self.params['type'] == "freeswimming":
            crop_around_body = self.params['gui_params']['zoom_body_crop'] and self.tracking_results is not None and self.tracking_results['body_position'] is not None and self.tracking_results['heading_angle']
        else:
            crop_around_body = False

        if image is not None:
            # if we have more than one frame, show the slider
            show_slider = self.n_frames > 1

            # get gui params
            gui_params = self.params['gui_params']
            
            # send signal to update image in preview window
            self.preview_window.plot_image(image, self.params, self.params['crop_params'][self.current_crop], self.tracking_results, new_load, new_frame, show_slider, crop_around_body=crop_around_body)

    def toggle_dark_background(self, checkbox):
        self.params['dark_background'] = checkbox.isChecked()

        self.params['backgrounds'] = [None for i in range(len(self.params['video_paths']))]
        self.bg_sub_frames         = [None for i in range(len(self.params['video_paths']))]

        if self.get_backgrounds_thread is not None:
            self.get_backgrounds_thread.running = False

        # get the paths of videos for which the background needs to be calculated
        background_calc_paths = self.params['video_paths']

        self.param_window.set_invalid_params_text("")

        if len(background_calc_paths) > 0:
            self.param_window.update_background_progress_text(len(background_calc_paths), 0)

            self.get_backgrounds_thread = GetBackgroundThread(self.param_window, self.params['dark_background'])
            self.get_backgrounds_thread.progress.connect(self.background_calculation_progress)
            self.get_backgrounds_thread.finished.connect(self.background_calculated)

            self.get_backgrounds_thread.set_parameters(background_calc_paths.copy(), self.params['dark_background'])

            self.background_calc_paths = background_calc_paths

            self.get_backgrounds_thread.start()

    def generate_thresholded_frames(self):
        # get params of currently selected crop
        current_crop_params = self.params['crop_params'][self.current_crop]

        # generate thresholded frames
        self.body_threshold_frame = tracking.get_threshold_frame(self.scaled_frame, current_crop_params['body_threshold'])*255
        self.eyes_threshold_frame = tracking.get_threshold_frame(self.scaled_frame, current_crop_params['eyes_threshold'])*255
        self.tail_threshold_frame = tracking.get_threshold_frame(self.scaled_frame, current_crop_params['tail_threshold'])*255
        self.tail_skeleton_frame  = tracking.get_tail_skeleton_frame(self.tail_threshold_frame/255)*255

    def toggle_save_video(self, checkbox):
        self.params['save_video'] = checkbox.isChecked()

    def toggle_adjust_thresholds(self, checkbox):
        self.params['adjust_thresholds'] = checkbox.isChecked()

    def toggle_tail_tracking(self, checkbox):
        self.params['track_tail'] = checkbox.isChecked()

    def toggle_eye_tracking(self, checkbox):
        self.params['track_eyes'] = checkbox.isChecked()

    def toggle_subtract_background(self, checkbox):
        self.params['subtract_background'] = checkbox.isChecked()

        # reshape the image
        self.show_frame(self.n)

    def toggle_multiprocessing(self, checkbox):
        self.params['use_multiprocessing'] = checkbox.isChecked()

    def toggle_auto_tracking(self, checkbox):
        self.params['gui_params']['auto_track'] = checkbox.isChecked()

    def toggle_zoom_body_crop(self, checkbox):
        self.params['gui_params']['zoom_body_crop'] = checkbox.isChecked()

        self.update_preview()

    def toggle_alt_tail_tracking(self, checkbox):
        self.params['alt_tail_tracking'] = checkbox.isChecked()

    def track_frame(self):
        if self.current_frame is not None:
            # track current frame
            self.tracking_results, skeleton_matrix, body_crop_coords = tracking.track_cropped_frame(self.scaled_frame, self.params, self.params['crop_params'][self.current_crop])
            if skeleton_matrix is not None and body_crop_coords is not None:
                self.tail_skeleton_frame[body_crop_coords[0, 0]:body_crop_coords[0, 1], body_crop_coords[1, 0]:body_crop_coords[1, 1]] = skeleton_matrix*255

            # rescale coordinates
            if self.tracking_results['tail_coords'] is not None:
                self.tracking_results['tail_coords'] /= self.params['scale_factor']
            if self.tracking_results['spline_coords'] is not None:
                self.tracking_results['spline_coords'] /= self.params['scale_factor']
            if self.tracking_results['body_position'] is not None:
                self.tracking_results['body_position'] /= self.params['scale_factor']
            if self.tracking_results['eye_coords'] is not None:
                self.tracking_results['eye_coords'] /= self.params['scale_factor']

            self.update_preview(image=None, new_load=False, new_frame=False)

    def save_background(self):
        if self.params['backgrounds'][self.curr_video_num] is not None:
            if pyqt_version == 4:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save background', '{}_background'.format(os.path.splitext(self.params['video_paths'][0])[0]), 'Images (*.png *.tif *.jpg)'))
            elif pyqt_version == 5:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save background', '{}_background'.format(os.path.splitext(self.params['video_paths'][0])[0]), 'Images (*.png *.tif *.jpg)')[0])
            if not (save_path.endswith('.jpg') or save_path.endswith('.tif') or save_path.endswith('.png')):
                save_path += ".png"
            cv2.imwrite(save_path, self.params['backgrounds'][self.curr_video_num])

    def load_background(self):
        if self.current_frame is not None:
            if pyqt_version == 4:
                background_path = str(QFileDialog.getOpenFileName(self.param_window, 'Open image', '', 'Images (*.jpg  *.jpeg *.tif *.tiff *.png)'))
            elif pyqt_version == 5:
                background_path = str(QFileDialog.getOpenFileName(self.param_window, 'Open image', '', 'Images (*.jpg  *.jpeg *.tif *.tiff *.png)')[0])

            background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

            if background.shape == self.current_frame.shape:
                self.params['backgrounds'][self.curr_video_num] = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
                self.background_calculated(self.params['backgrounds'][self.curr_video_num], self.curr_video_num)

    def update_crop_from_selection(self, start_crop_coord, end_crop_coord):
        # get start & end coordinates - end_add adds a pixel to the end coordinates (for a more accurate crop)
        y_start = round(start_crop_coord[0]/self.params['scale_factor'])
        y_end   = round(end_crop_coord[0]/self.params['scale_factor'])
        x_start = round(start_crop_coord[1]/self.params['scale_factor'])
        x_end   = round(end_crop_coord[1]/self.params['scale_factor'])
        end_add = round(1*self.params['scale_factor'])

        # get params of currently selected crop
        current_crop_params = self.params['crop_params'][self.current_crop].copy()

        # update crop params
        crop   = np.array([abs(y_end - y_start)+end_add, abs(x_end - x_start)+end_add])
        offset = np.array([current_crop_params['offset'][0] + min(y_start, y_end), current_crop_params['offset'][1] + min(x_start, x_end)])
        self.params['crop_params'][self.current_crop]['crop']   = crop
        self.params['crop_params'][self.current_crop]['offset'] = offset

        # update crop gui
        self.param_window.update_gui_from_crop_params(self.params['crop_params'])

        # reset headfixed tracking
        tracking.clear_headfixed_tracking()

        # reshape current frame
        self.reshape_frame()

        # update the image preview
        self.update_preview(image=None, new_load=True, new_frame=True)

    def create_crop(self, new_crop_params=None):
        if new_crop_params is None:
            new_crop_params = self.default_crop_params.copy()

            if self.current_frame is not None:
                new_crop_params['crop']   = np.array(self.current_frame.shape)
                new_crop_params['offset'] = np.array([0, 0])

        self.params['crop_params'].append(new_crop_params)

        self.current_crop = len(self.params['crop_params'])-1

        self.param_window.create_crop_tab(new_crop_params)

    def change_crop(self, index):
        if self.current_frame is not None and index != -1:
            # update current crop number
            self.current_crop = index

            # update the gui with these crop params
            self.param_window.update_gui_from_crop_params(self.params['crop_params'])

            # update the image preview
            self.reshape_frame()
            self.update_preview(image=None, new_load=False, new_frame=True)

            if self.params['gui_params']['auto_track']:
                self.track_frame()

    def remove_crop(self, index):
        # get current number of crops
        n_crops = len(self.params['crop_params'])

        if n_crops > 1:
            # delete params for this crop
            del self.params['crop_params'][index]

            # remove crop tab
            self.param_window.remove_crop_tab(index)

            # set current crop to last tab
            self.current_crop = len(self.params['crop_params']) - 1

    def clear_crops(self):
        if self.current_crop != -1:
            # get current number of crops
            n_crops = len(self.params['crop_params'])

            for index in range(n_crops-1, -1, -1):
                # delete params
                del self.params['crop_params'][index]

                # remove crop tab
                self.param_window.remove_crop_tab(index)

            # reset current crop
            self.params['crop_params'] = []
            self.current_crop = -1

    def select_crop(self):
        # user wants to draw a crop selection; start selecting
        self.preview_window.start_selecting_crop()

    def reset_crop(self):
        if self.current_frame is not None:
            # get params of currently selected crop
            current_crop_params = self.params['crop_params'][self.current_crop]

            # reset crop params
            current_crop_params['crop']   = np.array(self.current_frame.shape)
            current_crop_params['offset'] = np.array([0, 0])
            self.params['crop_params'][self.current_crop] = current_crop_params

            # update crop gui
            self.param_window.update_gui_from_crop_params(self.params['crop_params'])

            # reshape current frame
            self.reshape_frame()

            # update the image preview
            self.update_preview(image=None, new_load=False, new_frame=True)

    def update_scale_factor(self, scale_factor):
        if self.current_frame is not None:
            try:
                scale_factor = float(scale_factor)
                if not (0 < scale_factor <= 4):
                    raise

                self.params['scale_factor'] = scale_factor

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid scale factor value.")

    def update_bg_sub_threshold(self, bg_sub_threshold):
        if self.current_frame is not None:
            try:
                bg_sub_threshold = int(float(bg_sub_threshold))
                if not (0 <= bg_sub_threshold <= 255):
                    raise

                self.params['bg_sub_threshold'] = bg_sub_threshold

                for i in range(len(self.params['video_paths'])):
                    if self.params['backgrounds'][i] is not None:
                        self.bg_sub_frames[i] = tracking.subtract_background_from_frames(self.frames[i], self.params['backgrounds'][i], self.params['bg_sub_threshold'], dark_background=self.params['dark_background'])

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid background subtraction threshold value.")

    def update_crop_height(self, crop_height):
        if self.current_frame is not None:
            try:
                crop_height = int(float(crop_height))
                if not(1 <= crop_height <= self.current_frame.shape[0]):
                    raise

                self.params['crop_params'][self.current_crop]['crop'][0] = crop_height

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid crop height value.")

    def update_crop_width(self, crop_width):
        if self.current_frame is not None:
            try:
                crop_width = int(float(crop_width))
                if not(1 <= crop_width <= self.current_frame.shape[1]):
                    raise

                self.params['crop_params'][self.current_crop]['crop'][1] = crop_width

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid crop width value.")

    def update_y_offset(self, y_offset):
        if self.current_frame is not None:
            try:
                y_offset = int(float(y_offset))
                if not(0 <= y_offset <= self.current_frame.shape[0]-1):
                    raise

                self.params['crop_params'][self.current_crop]['offset'][0] = y_offset

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid y offset value.")

    def update_x_offset(self, x_offset):
        if self.current_frame is not None:
            try:
                x_offset = int(float(x_offset))
                if not(0 <= x_offset <= self.current_frame.shape[1]-1):
                    raise

                self.params['crop_params'][self.current_crop]['offset'][1] = x_offset

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid x offset value.")

    def update_tracking_video_fps(self, tracking_video_fps):
        if self.current_frame is not None:
            try:
                tracking_video_fps = float(tracking_video_fps)
                if not(tracking_video_fps >= 0):
                    raise

                self.params['tracking_video_fps'] = tracking_video_fps

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid saved video FPS value.")

    def update_n_tail_points(self, n_tail_points):
        if self.current_frame is not None:
            try:
                n_tail_points = int(float(n_tail_points))
                if not(n_tail_points > 1):
                    raise

                self.params['n_tail_points'] = n_tail_points

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid number of tail points value.")

    def add_angle_overlay(self, angle):
        self.preview_window.add_angle_overlay(angle)

    def remove_angle_overlay(self):
        self.preview_window.remove_angle_overlay()

    def close_all(self):
        self.closing = True
        self.param_window.close()
        self.preview_window.close()

class FreeswimmingController(Controller):
    def __init__(self):
        # initialize variables
        self.body_threshold_frame = None
        self.eyes_threshold_frame = None
        self.tail_threshold_frame = None
        self.tail_skeleton_frame  = None

        # set path to where last used parameters are saved
        self.last_params_path = "last_params_freeswimming.npz"

        Controller.__init__(self, DEFAULT_FREESWIMMING_PARAMS, DEFAULT_FREESWIMMING_CROP_PARAMS)

        # create parameters window
        self.param_window = FreeswimmingParamWindow(self)

        # create preview window
        self.preview_window = PreviewWindow(self)

        self.param_window.set_gui_disabled(True)

    def show_frame(self, n, new_load=False):
        Controller.show_frame(self, n, new_load)

        if new_load:
            self.param_window.param_controls['body_crop_height_slider'].setMaximum(self.current_frame.shape[0])
            self.param_window.param_controls['body_crop_width_slider'].setMaximum(self.current_frame.shape[1])

    def update_preview(self, image=None, new_load=False, new_frame=False):
        if image is None:
            # pick correct image to show in preview window
            if self.params['gui_params']["show_body_threshold"]:
                image = self.body_threshold_frame
            elif self.params['gui_params']["show_eyes_threshold"]:
                image = self.eyes_threshold_frame
            elif self.params['gui_params']["show_tail_threshold"]:
                image = self.tail_threshold_frame
            elif self.params['gui_params']["show_tail_skeleton"]:
                image = self.tail_skeleton_frame
            else:
                image = self.scaled_frame

        Controller.update_preview(self, image, new_load, new_frame)

    def reshape_frame(self):
        Controller.reshape_frame(self)

    def toggle_threshold_image(self, checkbox):
        if self.current_frame is not None and checkbox is not None and checkbox.isChecked():
            # uncheck other threshold checkboxes
            if checkbox.text() == "Show body threshold":
                self.param_window.param_controls["show_eyes_threshold"].setChecked(False)
                self.param_window.param_controls["show_tail_threshold"].setChecked(False)
                self.param_window.param_controls["show_tail_skeleton"].setChecked(False)
            elif checkbox.text() == "Show eye threshold":
                self.param_window.param_controls["show_body_threshold"].setChecked(False)
                self.param_window.param_controls["show_tail_threshold"].setChecked(False)
                self.param_window.param_controls["show_tail_skeleton"].setChecked(False)
            elif checkbox.text() == "Show tail threshold":
                self.param_window.param_controls["show_body_threshold"].setChecked(False)
                self.param_window.param_controls["show_eyes_threshold"].setChecked(False)
                self.param_window.param_controls["show_tail_skeleton"].setChecked(False)
            elif checkbox.text() == "Show tail skeleton":
                self.param_window.param_controls["show_body_threshold"].setChecked(False)
                self.param_window.param_controls["show_eyes_threshold"].setChecked(False)
                self.param_window.param_controls["show_tail_threshold"].setChecked(False)

        self.params['gui_params']['show_body_threshold'] = self.param_window.param_controls["show_body_threshold"].isChecked()
        self.params['gui_params']['show_eyes_threshold']  = self.param_window.param_controls["show_eyes_threshold"].isChecked()
        self.params['gui_params']['show_tail_threshold'] = self.param_window.param_controls["show_tail_threshold"].isChecked()
        self.params['gui_params']['show_tail_skeleton']  = self.param_window.param_controls["show_tail_skeleton"].isChecked()

        # update the image preview
        self.update_preview(image=None, new_load=False, new_frame=True)

    def get_checked_threshold_checkbox(self):
        if self.param_window.param_controls["show_body_threshold"].isChecked():
            return self.param_window.param_controls["show_body_threshold"]
        elif self.param_window.param_controls["show_eyes_threshold"].isChecked():
            return self.param_window.param_controls["show_eyes_threshold"]
        elif self.param_window.param_controls["show_tail_threshold"].isChecked():
            return self.param_window.param_controls["show_tail_threshold"]
        elif self.param_window.param_controls["show_tail_skeleton"].isChecked():
            return self.param_window.param_controls["show_tail_skeleton"]
        else:
            return None

    def toggle_tail_tracking(self, checkbox):
        self.params['track_tail'] = checkbox.isChecked()

    def update_crop_params_from_gui(self):
        old_crop_params = self.params['crop_params']

        # get current shrink factor
        scale_factor = self.params['scale_factor']

        # get crop params from gui
        for c in range(len(self.params['crop_params'])):
            crop_y   = int(float(self.param_window.crop_param_controls[c]['crop_y' + '_textbox'].text()))
            crop_x   = int(float(self.param_window.crop_param_controls[c]['crop_x' + '_textbox'].text()))
            offset_y = int(float(self.param_window.crop_param_controls[c]['offset_y' + '_textbox'].text()))
            offset_x = int(float(self.param_window.crop_param_controls[c]['offset_x' + '_textbox'].text()))

            body_threshold = int(float(self.param_window.crop_param_controls[c]['body_threshold' + '_textbox'].text()))
            eyes_threshold = int(float(self.param_window.crop_param_controls[c]['eyes_threshold' + '_textbox'].text()))
            tail_threshold = int(float(self.param_window.crop_param_controls[c]['tail_threshold' + '_textbox'].text()))

            valid_params = (1 <= crop_y <= self.current_frame.shape[0]
                        and 1 <= crop_x <= self.current_frame.shape[1]
                        and 0 <= offset_y < self.current_frame.shape[0]-1
                        and 0 <= offset_x < self.current_frame.shape[1]-1
                        and 0 <= body_threshold <= 255
                        and 0 <= eyes_threshold <= 255
                        and 0 <= tail_threshold <= 255)

            if valid_params:
                self.params['crop_params'][c]['crop']   = np.array([crop_y, crop_x])
                self.params['crop_params'][c]['offset'] = np.array([offset_y, offset_x])

                self.params['crop_params'][c]['body_threshold'] = body_threshold
                self.params['crop_params'][c]['eyes_threshold']  = eyes_threshold
                self.params['crop_params'][c]['tail_threshold'] = tail_threshold
            else:
                self.param_window.set_invalid_params_text("Invalid crop parameters.")

                self.params['crop_params'] = old_crop_params

                return

        # reshape current frame
        self.reshape_frame()

        if self.params['type'] == "freeswimming":
            # generate thresholded frames
            self.generate_thresholded_frames()

        # update the image preview
        self.update_preview(image=None, new_load=False, new_frame=True)

    def update_params_from_gui(self):
        if self.current_frame is not None:
            # get params from gui
            try:
                scale_factor      = float(self.param_window.param_controls['scale_factor' + '_textbox'].text())
                tracking_video_fps    = int(float(self.param_window.param_controls['tracking_video_fps'].text()))
                n_tail_points      = int(float(self.param_window.param_controls['n_tail_points'].text()))
                body_crop_height   = int(float(self.param_window.param_controls['body_crop_height' + '_textbox'].text()))
                body_crop_width    = int(float(self.param_window.param_controls['body_crop_width' + '_textbox'].text()))
                min_tail_body_dist = int(float(self.param_window.param_controls['min_tail_body_dist'].text()))
                max_tail_body_dist = int(float(self.param_window.param_controls['max_tail_body_dist'].text()))
                bg_sub_threshold   = int(float(self.param_window.param_controls['bg_sub_threshold' + '_textbox'].text()))
                interpolation      = str(self.param_window.param_controls['interpolation'].currentText())
            except ValueError:
                self.param_window.set_invalid_params_text("Invalid tracking parameters.")
                return

            valid_params = (scale_factor > 0
                        and tracking_video_fps >= 0
                        and n_tail_points > 0
                        and body_crop_height > 0
                        and body_crop_width > 0
                        and min_tail_body_dist >= 0
                        and max_tail_body_dist > min_tail_body_dist)

            if valid_params:
                # if self.params['bg_sub_threshold'] != bg_sub_threshold:
                #     self.bg_sub_frames[self.curr_video_num] = tracking.subtract_background_from_frames(self.frames[self.curr_video_num], self.params['backgrounds'][self.curr_video_num], self.params['bg_sub_threshold'])
                
                self.params['scale_factor']       = scale_factor
                self.params['tracking_video_fps']    = tracking_video_fps
                self.params['n_tail_points']      = n_tail_points
                self.params['body_crop']          = np.array([body_crop_height, body_crop_width])
                self.params['min_tail_body_dist'] = min_tail_body_dist
                self.params['max_tail_body_dist'] = max_tail_body_dist
                self.params['bg_sub_threshold']   = bg_sub_threshold
                self.params['interpolation']      = interpolation

                self.param_window.set_invalid_params_text("")
            else:
                self.param_window.set_invalid_params_text("Invalid parameters.")
                return

            # # reshape current frame
            # self.reshape_frame()

            # if self.params['type'] == "freeswimming":
            #     # generate thresholded frames
            #     self.generate_thresholded_frames()

            # # reshape the image
            # self.show_frame(self.n)

            # update the image preview
            # self.update_preview(image=None, new_load=False, new_frame=True)

    def update_min_tail_body_dist(self, min_tail_body_dist):
        if self.current_frame is not None:
            try:
                min_tail_body_dist = int(float(min_tail_body_dist))
                if not(min_tail_body_dist > 1 and min_tail_body_dist <= self.params['max_tail_body_dist']):
                    raise

                self.params['min_tail_body_dist'] = min_tail_body_dist

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid minimum tail-body distance value.")

    def update_max_tail_body_dist(self, max_tail_body_dist):
        if self.current_frame is not None:
            try:
                max_tail_body_dist = int(float(max_tail_body_dist))
                if not(max_tail_body_dist > 1 and max_tail_body_dist >= self.params['min_tail_body_dist']):
                    raise

                self.params['max_tail_body_dist'] = max_tail_body_dist

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid maximum tail-body distance value.")

    def update_body_crop_height(self, body_crop_height):
        if self.current_frame is not None:
            try:
                body_crop_height = int(float(body_crop_height))
                if not(body_crop_height > 1):
                    raise

                self.params['body_crop'][0] = body_crop_height

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid body crop height value.")

    def update_body_crop_width(self, body_crop_width):
        if self.current_frame is not None:
            try:
                body_crop_width = int(float(body_crop_width))
                if not(body_crop_width > 1):
                    raise

                self.params['body_crop'][1] = body_crop_width

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid body crop width value.")

    def update_body_threshold(self, body_threshold):
        if self.current_frame is not None:
            try:
                body_threshold = int(float(body_threshold))
                if not(0 < body_threshold < 255):
                    raise

                self.params['crop_params'][self.current_crop]['body_threshold'] = body_threshold

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid body threshold value.")

    def update_eyes_threshold(self, eyes_threshold):
        if self.current_frame is not None:
            try:
                eyes_threshold = int(float(eyes_threshold))
                if not (0 < eyes_threshold < 255):
                    raise

                self.params['crop_params'][self.current_crop]['eyes_threshold'] = eyes_threshold

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid eyes threshold value.")

    def update_tail_threshold(self, tail_threshold):
        if self.current_frame is not None:
            try:
                tail_threshold = int(float(tail_threshold))
                if not (0 < tail_threshold < 255):
                    raise

                self.params['crop_params'][self.current_crop]['tail_threshold'] = tail_threshold

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid tail threshold value.")

    def update_crop_from_selection(self, start_crop_coord, end_crop_coord):
        Controller.update_crop_from_selection(self, start_crop_coord, end_crop_coord)

        # generate thresholded frames
        self.generate_thresholded_frames()

class HeadfixedController(Controller):
    def __init__(self):
        # set path to where last used parameters are saved
        self.last_params_path = "last_params_headfixed.npz"

        Controller.__init__(self, DEFAULT_HEADFIXED_PARAMS, DEFAULT_HEADFIXED_CROP_PARAMS)

        # create parameters window
        self.param_window = HeadfixedParamWindow(self)

        # create preview window
        self.preview_window  = PreviewWindow(self)

        self.param_window.set_gui_disabled(True)

    def track_frame(self):
        if self.params['tail_start_coords'] is not None:
            self.preview_window.instructions_label.setText("")
            Controller.track_frame(self)
        else:
            self.preview_window.instructions_label.setText("Please select the start of the tail.")

    def update_crop_params_from_gui(self):
        old_crop_params = self.params['crop_params']

        # get crop params from gui
        for c in range(len(self.params['crop_params'])):
            crop_y   = int(float(self.param_window.crop_param_controls[c]['crop_y' + '_textbox'].text()))
            crop_x   = int(float(self.param_window.crop_param_controls[c]['crop_x' + '_textbox'].text()))
            offset_y = int(float(self.param_window.crop_param_controls[c]['offset_y' + '_textbox'].text()))
            offset_x = int(float(self.param_window.crop_param_controls[c]['offset_x' + '_textbox'].text()))

            valid_params = (1 <= crop_y <= self.current_frame.shape[0]
                        and 1 <= crop_x <= self.current_frame.shape[1]
                        and 0 <= offset_y < self.current_frame.shape[0]-1
                        and 0 <= offset_x < self.current_frame.shape[1]-1)

            if valid_params:
                self.params['crop_params'][c]['crop']   = np.array([crop_y, crop_x])
                self.params['crop_params'][c]['offset'] = np.array([offset_y, offset_x])
            else:
                self.param_window.set_invalid_params_text("Invalid crop parameters.")

                self.params['crop_params'] = old_crop_params

                return

        # reshape current frame
        self.reshape_frame()

        # reset headfixed tracking
        tracking.clear_headfixed_tracking()

        # update the image preview
        self.update_preview(image=None, new_load=False, new_frame=True)

    def reshape_frame(self):
        Controller.reshape_frame(self)

    def update_tail_start_coords(self, rel_tail_start_coords):
        self.params['tail_start_coords'] = tracking.get_absolute_coords(rel_tail_start_coords,
                                                                                   self.params['crop_params'][self.current_crop]['offset'],
                                                                                   self.params['scale_factor'])

        # reset headfixed tracking
        tracking.clear_headfixed_tracking()

    def update_params_from_gui(self):
        if self.current_frame is not None:
            # get params from gui
            try:
                scale_factor   = float(self.param_window.param_controls['scale_factor' + '_textbox'].text())
                heading_direction  = str(self.param_window.param_controls['heading_direction'].currentText())
                tracking_video_fps = int(self.param_window.param_controls['tracking_video_fps'].text())
                n_tail_points   = int(self.param_window.param_controls['n_tail_points'].text())
            except ValueError:
                self.param_window.set_invalid_params_text("Invalid parameters.")
                return

            valid_params = (0 < scale_factor <= 1
                        and tracking_video_fps >= 0
                        and n_tail_points > 0)

            if valid_params:
                self.params['scale_factor']   = scale_factor
                self.params['heading_direction']  = heading_direction
                self.params['tracking_video_fps'] = tracking_video_fps
                self.params['n_tail_points']   = n_tail_points

                self.param_window.set_invalid_params_text("")
            else:
                self.param_window.set_invalid_params_text("Invalid parameters.")
                return

            # reshape current frame
            self.reshape_frame()

            # update the image preview
            self.update_preview(image=None, new_load=False, new_frame=True)

    def update_heading_angle(self, heading_angle):
        if self.current_frame is not None:
            try:
                heading_angle = int(float(heading_angle))
                if not (0 <= heading_angle <= 360):
                    raise

                self.params['heading_angle'] = heading_angle

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid heading angle value.")

    def update_heading_direction(self, heading_direction):
        if self.current_frame is not None:
            try:
                self.params['heading_direction'] = heading_direction

                self.param_window.set_invalid_params_text("")

                self.show_frame(self.n)
            except:
                self.param_window.set_invalid_params_text("Invalid heading direction value.")

class GetBackgroundThread(QThread):
    finished = pyqtSignal(np.ndarray, str)
    progress = pyqtSignal(int, str)

    def __init__(self, parent, dark_background):
        QThread.__init__(self, parent)

        # initialize video paths, and set whether the video background is dark
        self.video_paths     = []
        self.dark_background = dark_background

    def add_video_paths(self, video_paths):
        self.video_paths    += video_paths

    def set_parameters(self, video_paths, dark_background):
        self.video_paths     = video_paths
        self.dark_background = dark_background

    def run(self):
        self.running = True

        while len(self.video_paths) > 0 and self.running:
            fps, n_frames_total = open_media.get_video_info(self.video_paths[0])

            frame_nums = utilities.split_evenly(n_frames_total, 1000)

            background = open_media.open_video(self.video_paths[0], frame_nums, False, True, dark_background=self.dark_background, progress_signal=self.progress, thread=self)

            if background is not None:
                self.finished.emit(background, self.video_paths[0])
            else:
                self.finished.emit(np.zeros(1), self.video_paths[0])

            del self.video_paths[0]

        self.running = False

class TrackVideosThread(QThread):
    finished = pyqtSignal(float)
    progress = pyqtSignal(int, float)

    def __init__(self, parent):
        QThread.__init__(self, parent)

    def set_parameters(self, params, tracking_path):
        self.params = params
        self.tracking_path = tracking_path

    def run(self):
        tracking_func = tracking.open_and_track_video_batch

        if self.tracking_path != "":
            start_time = time.time()

            tracking_func(self.params, self.tracking_path, progress_signal=self.progress)

            end_time = time.time()

            self.finished.emit(end_time - start_time)