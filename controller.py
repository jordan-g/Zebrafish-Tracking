from param_window import *
from preview_window import *
from crops_window import *
from analysis_window import *
import tracking

# import the Qt library
from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    try:
        from PyQt4.QtCore import Signal, Qt, QThread
        from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
        pyqt_version = 4
    except:
        from PyQt5.QtCore import Signal, Qt, QThread
        from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
        from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
        pyqt_version = 5

try:
    xrange
except:
    xrange = range

default_headfixed_crop_params = { 'offset': np.array([0, 0]), # crop (y, x) offset
                                  'crop': None }              # crop area

default_freeswimming_crop_params = { 'offset': np.array([0, 0]), # crop (y, x) offset
                                     'crop': None,               # crop area
                                     'body_threshold': 140,      # pixel brightness to use for thresholding to find the body (0-255)
                                     'eye_threshold': 60,        # pixel brightness to use for thresholding to find the eyes (0-255)
                                     'tail_threshold': 200 }     # pixel brightness to use for thresholding to find the tail (0-255)

default_headfixed_params = {'shrink_factor': 1.0,                               # factor by which to shrink the original frame
                            'crop_params': [],
                            'invert': False,                                    # invert the frame
                            'type': "headfixed",                                # "headfixed" / "freeswimming"
                            'save_video': True,                                 # whether to make a video with tracking overlaid
                            'saved_video_fps': 30,                              # fps for the generated video
                            'n_tail_points': 30,                                # number of tail points to use
                            'subtract_background': False,                       # whether to perform background subtraction
                            'tail_direction': "Right",                          # "right" / "left" / "up" / "down"
                            'media_type': None,                                 # type of media that is tracked - "video" / "folder" / "image" / None
                            'media_paths': [],                                  # paths to media that are tracked
                            'backgrounds': None,                                # backgrounds calculated for background subtraction
                            'tail_start_coords': None,                          # (y, x) coordinates of the start of the tail
                            'use_multiprocessing': True,                        # whether to use multiprocessing
                            'batch_offsets': None,
                            'gui_params': { 'auto_track': False }}              # automatically track a frame when you switch to it

default_freeswimming_params = {'shrink_factor': 1.0,                            # factor by which to shrink the original frame
                               'crop_params': [],
                               'invert': False,                                 # invert the frame
                               'type': "freeswimming",                          # "headfixed" / "freeswimming"
                               'save_video': True,                              # whether to make a video with tracking overlaid
                               'saved_video_fps': 30,                           # fps for the generated video
                               'n_tail_points': 30,                             # number of tail points to use
                               'adjust_thresholds': False,                      # whether to adjust thresholds while tracking if necessary
                               'subtract_background': False,                    # whether to perform background subtraction
                               'track_tail': True,                              # whether to track the tail
                               'track_eyes': True,                              # whether to track the eyes
                               'min_tail_body_dist': 10,                        # min. distance between the body center and the tail
                               'max_tail_body_dist': 30,                        # max. distance between the body center and the tail
                               'eye_resize_factor': 1,                          # factor by which to resize frame for reducing noise in eye position tracking
                               'interpolation': 'Nearest Neighbor',             # interpolation to use when resizing frame for eye tracking
                               'tail_crop': np.array([100, 100]),               # dimensions of crop around zebrafish eyes to use for tail tracking - (y, x)
                               'media_type': None,                              # type of media that is tracked - "video" / "folder" / "image" / None
                               'media_paths': [],                               # paths to media that are tracked
                               'backgrounds': None,                             # backgrounds calculated for background subtraction
                               'use_multiprocessing': True,                     # whether to use multiprocessing
                               'batch_offsets': None,
                               'gui_params': { 'show_body_threshold': False,    # show body threshold in preview window
                                               'show_eye_threshold': False,     # show eye threshold in preview window
                                               'show_tail_threshold': False,    # show tail threshold in preview window
                                               'show_tail_skeleton': False,     # show tail skeleton in preview window
                                               'auto_track': False }}           # automatically track a frame when you switch to it

max_n_frames = 200 # maximum # of frames to load for previewing

class Controller():
    def __init__(self, default_params, default_crop_params):
        # set parameters
        self.default_params = default_params
        self.params = self.default_params

        self.default_crop_params = default_crop_params

        # initialize variables
        self.current_frame         = None
        self.shrunken_frame        = None
        self.cropped_frame         = None
        self.frames                = None
        self.tracking_results      = []
        self.current_crop          = -1   # which crop is being looked at
        self.curr_media_num        = 0    # which media (from a loaded batch) is being looked at
        self.n_frames              = 0    # total number of frames to preview
        self.n                     = 0    # index of currently selected frame
        self.tracking_path         = None # path to where tracking data will be saved
        self.get_background_thread = None
        self.track_media_thread    = None
        self.closing               = False

        # create preview window
        self.preview_window  = PreviewWindow(self)

        # create analysis window
        self.analysis_window = AnalysisWindow(self)
        self.analysis_window.hide()

    def select_and_open_media(self, media_type):
        if media_type == "image":
            # ask the user to select an image
            if pyqt_version == 4:
                media_paths = [str(QFileDialog.getOpenFileName(self.param_window, 'Select image to open', '', 'Images (*.jpg  *.jpeg *.tif *.tiff *.png)'))]
            elif pyqt_version == 5:
                media_paths = [str(QFileDialog.getOpenFileName(self.param_window, 'Select image to open', '', 'Images (*.jpg  *.jpeg *.tif *.tiff *.png)')[0])]
        elif media_type == "folder":
            # ask the user to select a directory
            media_paths = [str(QFileDialog.getExistingDirectory(self.param_window, 'Select folder to open'))]
        elif media_type == "video":
            # ask the user to select video(s)
            if pyqt_version == 4:
                media_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select video(s) to open', '', 'Videos (*.mov *.tif *.mp4 *.avi)')
            elif pyqt_version == 5:
                media_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select video(s) to open', '', 'Videos (*.mov *.tif *.mp4 *.avi)')[0]

            # convert paths to str
            media_paths = [ str(media_path) for media_path in media_paths ]

        if len(media_paths) > 0 and media_paths[0] != '':
            # clear all crops
            self.clear_crops()

            # set params to defaults
            self.params = self.default_params.copy()

            self.current_crop = -1

            self.open_media_batch(media_type, media_paths)

            self.create_crop()

            # switch to first frame
            self.switch_frame(0, new_load=True)

    def open_media(self, media_type, media_path):
        # reset tracking results
        self.tracking_results = []

        if media_path not in ("", None):
            # load frames
            if media_type == "image":
                self.frames[self.curr_media_num] = [tracking.load_frame_from_image(media_path)]
            elif media_type == "folder":
                # get filenames of all frame images in the folder
                frame_filenames = tracking.get_frame_filenames_from_folder(media_path)

                if len(frame_filenames) == 0:
                    # no frames found in the folder; end here
                    return

                n_frames_total = len(frame_filenames) # total number of frames in the folder

                # load evenly spaced frames
                frame_nums = split_evenly(n_frames_total, max_n_frames)

                # load frames from the folder
                if self.params['backgrounds'][self.curr_media_num] != None:
                    self.frames[self.curr_media_num], self.bg_sub_frames[self.curr_media_num] = tracking.load_frames_from_folder(media_path, frame_filenames, frame_nums, background=self.params['backgrounds'][self.curr_media_num])
                else:
                    self.frames[self.curr_media_num] = tracking.load_frames_from_folder(media_path, frame_filenames, frame_nums, None, offset=self.params['batch_offsets'][self.curr_media_num])
            elif media_type == "video":
                # get video info
                fps, n_frames_total = tracking.get_video_info(media_path)

                # load evenly spaced frames
                frame_nums = split_evenly(n_frames_total, max_n_frames)

                # load frames from the video
                if self.params['backgrounds'][self.curr_media_num] != None:
                    self.frames[self.curr_media_num], self.bg_sub_frames[self.curr_media_num] = tracking.load_frames_from_video(media_path, None, frame_nums, background=self.params['backgrounds'][self.curr_media_num], offset=self.params['batch_offsets'][self.curr_media_num])
                else:
                    self.frames[self.curr_media_num] = tracking.load_frames_from_video(media_path, None, frame_nums, background=None, offset=self.params['batch_offsets'][self.curr_media_num])

            if self.frames == None:
                # no frames found; end here
                print("Error: Could not load frames.")
                return

            # set current frame to first frame
            self.current_frame = self.frames[self.curr_media_num][0]

            # get number of frames
            self.n_frames = len(self.frames[self.curr_media_num])

            if self.params['type'] == "headfixed":
                # determine tail direction
                total_luminosities = [np.sum(self.current_frame[-1:-10, :]), np.sum(self.current_frame[0:10, :]),
                                      np.sum(self.current_frame[:, -1:-10]), np.sum(self.current_frame[:, 0:10])]

                self.params['tail_direction'] = tail_direction_options[np.argmax(total_luminosities)]

                self.param_window.update_gui_from_params(self.params)
            
            # get background
            if media_type != "image":
                if self.params['backgrounds'][self.curr_media_num] == None:
                    self.param_window.param_controls["subtract_background"].setEnabled(False)
                    self.param_window.open_background_action.setEnabled(False)
                    self.param_window.save_background_action.setEnabled(False)

                    if self.get_background_thread != None:
                        # another thread is already calculating a background; don't let it affect the GUI
                        self.get_background_thread.progress.disconnect(self.update_background_subtract_progress)
                        self.get_background_thread.finished.disconnect(self.background_calculated)

                    # update "Subtract background" text in param window
                    self.param_window.param_controls["subtract_background"].setText("Subtract background (Calculating...)")

                    # create new thread to calculate the background
                    self.get_background_thread = GetBackgroundThread(self.param_window)
                    self.get_background_thread.set_parameters(self.params['media_paths'][self.curr_media_num], media_type, self.curr_media_num, self.params['batch_offsets'][self.curr_media_num])

                    # set callback function to be called when the background has been calculated
                    self.get_background_thread.finished.connect(self.background_calculated)

                    # set callback function to be called as the background is being calculated (to show progress)
                    self.get_background_thread.progress.connect(self.update_background_subtract_progress)

                    # start thread
                    self.get_background_thread.start()
                else:
                    # background is already calculated; call the callback function
                    self.background_calculated(self.params['backgrounds'][self.curr_media_num], self.curr_media_num)

            # enable gui controls
            self.crops_window.set_gui_disabled(False)
            self.param_window.set_gui_disabled(False)

    def open_media_batch(self, media_type, media_paths): #todo: complete this
        # update media paths & type parameters
        self.params['media_paths'] = media_paths
        self.params['media_type']  = media_type
        self.params['backgrounds'] = [None]*len(media_paths)

        self.frames        = [[]]*len(media_paths)
        self.bg_sub_frames = [[]]*len(media_paths)

        # update current media number
        self.curr_media_num = 0

        # update loaded media label
        if len(media_paths) > 1:
            self.param_window.loaded_media_label.setText("Loaded <b>{} {}s</b>. Showing #{}.".format(len(media_paths), media_type, self.curr_media_num+1))
        else:
            self.param_window.loaded_media_label.setText("Loaded <b>{}</b>.".format(os.path.basename(media_paths[0])))

        self.params['batch_offsets'] = tracking.get_video_batch_align_offsets(self.params)

        # open the first media from the batch
        self.open_media(media_type, media_paths[self.curr_media_num])

    def prev_media(self):
        if self.curr_media_num != 0:
            media_paths = self.params['media_paths']
            media_type  = self.params['media_type']
            
            # update current media number
            self.curr_media_num -= 1

            # update loaded media label
            if len(media_paths) > 1:
                self.param_window.loaded_media_label.setText("Loaded <b>{} {}s</b>. Showing #{}.".format(len(media_paths), media_type, self.curr_media_num+1))
            else:
                self.param_window.loaded_media_label.setText("Loaded <b>{}</b>.".format(os.path.basename(media_paths[0])))

            # open the first media from the batch
            self.open_media(media_type, media_paths[self.curr_media_num])

            # switch to first frame
            self.switch_frame(0, new_load=True)

    def next_media(self):
        if self.curr_media_num != len(self.params['media_paths'])-1:
            media_paths = self.params['media_paths']
            media_type  = self.params['media_type']
            
            # update current media number
            self.curr_media_num += 1

            # update loaded media label
            if len(media_paths) > 1:
                self.param_window.loaded_media_label.setText("Loaded <b>{} {}s</b>. Showing #{}.".format(len(media_paths), media_type, self.curr_media_num+1))
            else:
                self.param_window.loaded_media_label.setText("Loaded <b>{}</b>.".format(os.path.basename(media_paths[0])))

            # open the first media from the batch
            self.open_media(media_type, media_paths[self.curr_media_num])

            # switch to first frame
            self.switch_frame(0, new_load=True)

    def background_calculated(self, background, media_num):
        print(media_num)
        if self.current_frame.shape == background.shape:
            print("Background calculated.")

            # update params
            self.params['backgrounds'][media_num] = background

            if self.params['backgrounds'][media_num] != None:
                # generate background subtracted frames
                self.bg_sub_frames[media_num] = tracking.subtract_background_from_frames(self.frames[media_num], self.params['backgrounds'][media_num])

                # update "Subtract background" text and enable checkbox in param window
                self.param_window.param_controls["subtract_background"].setEnabled(True)
                self.param_window.param_controls["subtract_background"].setText("Subtract background")

                self.param_window.open_background_action.setEnabled(True)
                self.param_window.save_background_action.setEnabled(True)

    def media_tracked(self, tracking_time):
        self.param_window.tracking_progress_label.setText("Tracking completed in <b>{:.3f}s</b>.".format(tracking_time))
        
        self.param_window.toggle_analysis_window_button.setEnabled(True)

    def update_background_subtract_progress(self, percent):
        self.param_window.param_controls["subtract_background"].setText("Subtract background (Calculating...{}%)".format(percent))
    
    def update_media_tracking_progress(self, media_type, media_number, percent):
        if len(self.params['media_paths']) > 1:
            self.param_window.tracking_progress_label.setText("Tracking <b>{} {}</b>: {}%.".format(media_type, media_number+1, percent))
        else:
            self.param_window.tracking_progress_label.setText("Tracking <b>{}</b>: {}%.".format(os.path.basename(self.params['media_paths'][0]), percent))

    def load_params(self, params_path=None):
        if params_path == None:
            # ask the user to select a path
            params_path = str(QFileDialog.getOpenFileName(self, 'Open saved parameters', ''))

        if params_path not in ("", None):
            # load params from saved file
            params_file = np.load(params_path)
            saved_params = params_file['params'][()]

            # set params to saved params
            self.params = saved_params

            self.current_crop = -1

            # re-open media path specified in the loaded params
            self.open_media_batch(self.params['media_type'], self.params['media_paths'])

            # create tabs for all saved crops
            for j in range(len(self.params['crop_params'])):
                self.current_crop += 1
                self.crops_window.create_crop_tab(self.params['crop_params'][j])

            # update gui controls
            self.param_window.update_gui_from_params(self.params)
            self.crops_window.update_gui_from_params(self.params['crop_params'])

            # switch to first frame
            self.switch_frame(0, new_load=True)
        else:
            pass

    def load_last_params(self):
        self.load_params(self.last_params_path)

    def save_params(self, params_path=None):
        # get params from gui
        self.update_params_from_gui()
        self.update_crop_params_from_gui()

        if params_path == None:
            # ask user to select a path
            params_path = str(QFileDialog.getSaveFileName(self, 'Choose directory to save in', ''))
        else:
            # set params path to last used params path
            params_path = self.last_params_path
 
        # save params to file
        np.savez(params_path, params=self.params)

    def toggle_analysis_window(self):
        if self.tracking_path != None:
            self.analysis_window.show()

            # analyze tracking data
            self.analysis_window.load_data(os.path.dirname(self.tracking_path))

    def switch_frame(self, n, new_load=False):
        if n != self.n:
            # reset tracking results
            self.tracking_results = []

        # set current frame index
        if n != None:
            self.n = n

        # set current frame
        if self.params['subtract_background']:
            frames = self.bg_sub_frames
        else:
            frames = self.frames
        print(self.curr_media_num)
        self.current_frame = frames[self.curr_media_num][self.n]

        # reshape the image (shrink, crop & invert)
        self.reshape_frame()

        # invert the frame
        if self.params['invert'] == True:
            self.invert_frame()

        if self.params['type'] == "freeswimming":
            # generate thresholded frames
            self.generate_thresholded_frames()

        # update the image preview
        self.update_preview(None, new_load, new_frame=True)

        if self.params['gui_params']['auto_track']:
            self.track_frame()

    def reshape_frame(self):
        # reset tracking results
        self.tracking_results = []

        if self.current_frame != None:
            print(self.current_frame)
            # get params of currently selected crop
            current_crop_params = self.params['crop_params'][self.current_crop]

            # crop the frame
            if current_crop_params['crop'] is not None and current_crop_params['crop'] is not None:
                crop   = current_crop_params['crop']
                offset = current_crop_params['offset']

                self.cropped_frame = tracking.crop_frame(self.current_frame, offset, crop)
            else:
                self.cropped_frame = self.current_frame

            # shrink the frame
            if self.params['shrink_factor'] != None:
                self.shrunken_frame = tracking.shrink_frame(self.cropped_frame, self.params['shrink_factor'])
            else:
                self.shrunken_frame = self.cropped_frame

    def invert_frame(self):
        self.current_frame  = (255 - self.current_frame)
        self.shrunken_frame = (255 - self.shrunken_frame)
        self.cropped_frame  = (255 - self.cropped_frame)

    def update_preview(self, image=None, new_load=False, new_frame=False):
        if image == None:
            # use the cropped current frame by default
            image = self.shrunken_frame

        if image != None:
            # if we have more than one frame, show the slider
            show_slider = self.n_frames > 1

            # get gui params
            gui_params = self.params['gui_params']
            
            # send signal to update image in preview window
            self.preview_window.plot_image(image, self.params, self.params['crop_params'][self.current_crop], self.tracking_results, new_load, new_frame, show_slider)

    def toggle_invert_image(self, checkbox):
        self.params['invert'] = checkbox.isChecked()

        # invert the frame
        self.invert_frame()

        if self.params['type'] == "freeswimming":
            # generate thresholded frames
            self.generate_thresholded_frames()

        # update the image preview
        self.update_preview()

    def generate_thresholded_frames(self):
        # get params of currently selected crop
        current_crop_params = self.params['crop_params'][self.current_crop]

        # generate thresholded frames
        self.body_threshold_frame = tracking.simplify_body_threshold_frame(tracking.get_threshold_frame(self.shrunken_frame, current_crop_params['body_threshold']))*255
        self.eye_threshold_frame  = tracking.get_threshold_frame(self.shrunken_frame, current_crop_params['eye_threshold'])*255
        self.tail_threshold_frame = tracking.get_threshold_frame(self.shrunken_frame, current_crop_params['tail_threshold'])*255
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
        if self.params['backgrounds'][self.curr_media_num] != None:
            self.params['subtract_background'] = checkbox.isChecked()

            # reshape the image
            self.switch_frame(self.n)

    def toggle_multiprocessing(self, checkbox):
        self.params['use_multiprocessing'] = checkbox.isChecked()

    def toggle_auto_tracking(self, checkbox):
        self.params['gui_params']['auto_track'] = checkbox.isChecked()

    def track_frame(self):
        if self.current_frame != None:
            # get params from gui
            self.update_params_from_gui()

            # track current frame
            self.tracking_results = tracking.track_cropped_frame(self.shrunken_frame, self.params, self.params['crop_params'][self.current_crop])

            self.update_preview(image=None, new_load=False, new_frame=False)

    def track_media(self):
        # get save path
        if self.params['media_type'] == "image":
            self.tracking_path = str(QFileDialog.getExistingDirectory(self.param_window, "Select Directory"))
        elif self.params['media_type'] == "folder":
            self.tracking_path = str(QFileDialog.getExistingDirectory(self.param_window, "Select Directory"))
        elif self.params['media_type'] == "video":
            self.tracking_path = str(QFileDialog.getExistingDirectory(self.param_window, "Select Directory"))

        # track media
        if self.track_media_thread != None:
            # another thread is already tracking something; don't let it affect the GUI
            self.track_media_thread.progress.disconnect(self.update_media_tracking_progress)
            self.track_media_thread.finished.disconnect(self.media_tracked)

        # create new thread to track the media
        self.track_media_thread = TrackMediaThread(self.param_window)
        self.track_media_thread.set_parameters(self.params, self.tracking_path)

        # set callback function to be called when the media has been tracked
        self.track_media_thread.finished.connect(self.media_tracked)

        # set callback function to be called as the media is being tracked (to show progress)
        self.track_media_thread.progress.connect(self.update_media_tracking_progress)

        if len(self.params['media_paths']) > 1:
            self.param_window.tracking_progress_label.setText("Tracking <b>{} 1</b>: 0%.".format(self.params['media_type']))
        else:
            self.param_window.tracking_progress_label.setText("Tracking <b>{}</b>: 0%.".format(os.path.basename(self.params['media_paths'][0])))

        # start thread
        self.track_media_thread.start()

    def save_background(self):
        if self.params['backgrounds'][self.curr_media_num] != None:
            if pyqt_version == 4:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save background', '{}_background'.format(os.path.splitext(self.params['media_paths'][0])[0]), 'Images (*.png *.tif *.jpg)'))
            elif pyqt_version == 5:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save background', '{}_background'.format(os.path.splitext(self.params['media_paths'][0])[0]), 'Images (*.png *.tif *.jpg)')[0])
            if not (save_path.endswith('.jpg') or save_path.endswith('.tif') or save_path.endswith('.png')):
                save_path += ".png"
            cv2.imwrite(save_path, self.params['backgrounds'][self.curr_media_num])

    def load_background(self):
        if self.current_frame != None:
            if pyqt_version == 4:
                background_path = str(QFileDialog.getOpenFileName(self.param_window, 'Open image', '', 'Images (*.jpg  *.jpeg *.tif *.tiff *.png)'))
            elif pyqt_version == 5:
                background_path = str(QFileDialog.getOpenFileName(self.param_window, 'Open image', '', 'Images (*.jpg  *.jpeg *.tif *.tiff *.png)')[0])
            print(background_path)
            background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

            if background.shape == self.current_frame.shape:
                print("hey")
                self.params['backgrounds'][self.curr_media_num] = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
                self.background_calculated(self.params['backgrounds'][self.curr_media_num], self.curr_media_num)

    def update_crop_from_selection(self, start_crop_coord, end_crop_coord):
        # get start & end coordinates - end_add adds a pixel to the end coordinates (for a more accurate crop)
        y_start = round(start_crop_coord[0]/self.params['shrink_factor'])
        y_end   = round(end_crop_coord[0]/self.params['shrink_factor'])
        x_start = round(start_crop_coord[1]/self.params['shrink_factor'])
        x_end   = round(end_crop_coord[1]/self.params['shrink_factor'])
        end_add = round(1*self.params['shrink_factor'])

        # get params of currently selected crop
        current_crop_params = self.params['crop_params'][self.current_crop].copy()

        # update crop params
        crop   = np.array([abs(y_end - y_start)+end_add, abs(x_end - x_start)+end_add])
        offset = np.array([current_crop_params['offset'][0] + min(y_start, y_end), current_crop_params['offset'][1] + min(x_start, x_end)])
        self.params['crop_params'][self.current_crop]['crop']   = crop
        self.params['crop_params'][self.current_crop]['offset'] = offset

        # update crop gui
        self.crops_window.update_gui_from_params(self.params['crop_params'])

        # reset headfixed tracking
        tracking.clear_headfixed_tracking()

        # reshape current frame
        self.reshape_frame()

        # update the image preview
        self.update_preview(image=None, new_load=True, new_frame=True)

    def create_crop(self, new_crop_params=None):
        if new_crop_params == None:
            new_crop_params = self.default_crop_params.copy()

            if self.current_frame != None:
                new_crop_params['crop']   = np.array(self.current_frame.shape)
                new_crop_params['offset'] = np.array([0, 0])

        self.params['crop_params'].append(new_crop_params)

        self.current_crop = len(self.params['crop_params'])-1

        self.crops_window.create_crop_tab(new_crop_params)

    def change_crop(self, index):
        if self.current_frame is not None and index != -1:
            # update current crop number
            self.current_crop = index

            # update the gui with these crop params
            self.crops_window.update_gui_from_params(self.params['crop_params'])

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
            self.crops_window.remove_crop_tab(index)

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
                self.crops_window.remove_crop_tab(index)

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
            self.crops_window.update_gui_from_params(self.params['crop_params'])

            # reshape current frame
            self.reshape_frame()

            # update the image preview
            self.update_preview(image=None, new_load=False, new_frame=True)

    def close_all(self):
        self.closing = True
        self.param_window.close()
        self.crops_window.close()
        self.preview_window.close()

class FreeswimmingController(Controller):
    def __init__(self):
        # initialize variables
        self.body_threshold_frame = None
        self.eye_threshold_frame  = None
        self.tail_threshold_frame = None
        self.tail_skeleton_frame  = None

        # set path to where last used parameters are saved
        self.last_params_path = "last_params_freeswimming.npz"

        Controller.__init__(self, default_freeswimming_params, default_freeswimming_crop_params)

        # create parameters window
        self.param_window = FreeswimmingParamWindow(self)

        # create crops window
        self.crops_window = FreeswimmingCropsWindow(self)
        # self.create_crop()

        self.param_window.set_gui_disabled(True)
        self.crops_window.set_gui_disabled(True)

    def switch_frame(self, n, new_load=False):
        Controller.switch_frame(self, n, new_load)

        if new_load:
            self.param_window.param_controls['tail_crop_height_slider'].setMaximum(self.current_frame.shape[0])
            self.param_window.param_controls['tail_crop_width_slider'].setMaximum(self.current_frame.shape[1])

    def update_preview(self, image=None, new_load=False, new_frame=False):
        if image == None:
            # pick correct image to show in preview window
            if self.params['gui_params']["show_body_threshold"]:
                image = self.body_threshold_frame
            elif self.params['gui_params']["show_eye_threshold"]:
                image = self.eye_threshold_frame
            elif self.params['gui_params']["show_tail_threshold"]:
                image = self.tail_threshold_frame
            elif self.params['gui_params']["show_tail_skeleton"]:
                image = self.tail_skeleton_frame
            else:
                image = self.shrunken_frame

        Controller.update_preview(self, image, new_load, new_frame)

    def reshape_frame(self):
        Controller.reshape_frame(self)

    def toggle_threshold_image(self, checkbox):
        if self.current_frame is not None:
            if checkbox.isChecked():
                # uncheck other threshold checkboxes
                if checkbox.text() == "Show body threshold":
                    self.param_window.param_controls["show_eye_threshold"].setChecked(False)
                    self.param_window.param_controls["show_tail_threshold"].setChecked(False)
                    self.param_window.param_controls["show_tail_skeleton"].setChecked(False)
                elif checkbox.text() == "Show eye threshold":
                    self.param_window.param_controls["show_body_threshold"].setChecked(False)
                    self.param_window.param_controls["show_tail_threshold"].setChecked(False)
                    self.param_window.param_controls["show_tail_skeleton"].setChecked(False)
                elif checkbox.text() == "Show tail threshold":
                    self.param_window.param_controls["show_body_threshold"].setChecked(False)
                    self.param_window.param_controls["show_eye_threshold"].setChecked(False)
                    self.param_window.param_controls["show_tail_skeleton"].setChecked(False)
                elif checkbox.text() == "Show tail skeleton":
                    self.param_window.param_controls["show_body_threshold"].setChecked(False)
                    self.param_window.param_controls["show_eye_threshold"].setChecked(False)
                    self.param_window.param_controls["show_tail_threshold"].setChecked(False)

            self.params['gui_params']['show_body_threshold'] = self.param_window.param_controls["show_body_threshold"].isChecked()
            self.params['gui_params']['show_eye_threshold']  = self.param_window.param_controls["show_eye_threshold"].isChecked()
            self.params['gui_params']['show_tail_threshold'] = self.param_window.param_controls["show_tail_threshold"].isChecked()
            self.params['gui_params']['show_tail_skeleton']  = self.param_window.param_controls["show_tail_skeleton"].isChecked()

            # update the image preview
            self.update_preview(image=None, new_load=False, new_frame=True)

    def toggle_tail_tracking(self, checkbox):
        self.params['track_tail'] = checkbox.isChecked()

    def update_crop_params_from_gui(self):
        old_crop_params = self.params['crop_params']

        # get current shrink factor
        shrink_factor = self.params['shrink_factor']

        # get crop params from gui
        for c in range(len(self.params['crop_params'])):
            crop_y   = int(float(self.crops_window.param_controls[c]['crop_y' + '_textbox'].text()))
            crop_x   = int(float(self.crops_window.param_controls[c]['crop_x' + '_textbox'].text()))
            offset_y = int(float(self.crops_window.param_controls[c]['offset_y' + '_textbox'].text()))
            offset_x = int(float(self.crops_window.param_controls[c]['offset_x' + '_textbox'].text()))

            body_threshold = int(self.crops_window.param_controls[c]['body_threshold'].text())
            eye_threshold  = int(self.crops_window.param_controls[c]['eye_threshold'].text())
            tail_threshold = int(self.crops_window.param_controls[c]['tail_threshold'].text())

            valid_params = (1 <= crop_y <= self.current_frame.shape[0]
                        and 1 <= crop_x <= self.current_frame.shape[1]
                        and 0 <= offset_y < self.current_frame.shape[0]-1
                        and 0 <= offset_x < self.current_frame.shape[1]-1
                        and 0 <= body_threshold <= 255
                        and 0 <= eye_threshold <= 255
                        and 0 <= tail_threshold <= 255)

            if valid_params:
                self.params['crop_params'][c]['crop']   = np.array([crop_y, crop_x])
                self.params['crop_params'][c]['offset'] = np.array([offset_y, offset_x])

                self.params['crop_params'][c]['body_threshold'] = body_threshold
                self.params['crop_params'][c]['eye_threshold']  = eye_threshold
                self.params['crop_params'][c]['tail_threshold'] = tail_threshold
            else:
                self.param_window.show_invalid_params_text()

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
        if self.current_frame != None:
            # get params from gui
            try:
                shrink_factor      = float(self.param_window.param_controls['shrink_factor' + '_textbox'].text())
                saved_video_fps    = int(float(self.param_window.param_controls['saved_video_fps'].text()))
                n_tail_points      = int(float(self.param_window.param_controls['n_tail_points'].text()))
                tail_crop_height   = int(float(self.param_window.param_controls['tail_crop_height' + '_textbox'].text()))
                tail_crop_width    = int(float(self.param_window.param_controls['tail_crop_width' + '_textbox'].text()))
                min_tail_body_dist = int(float(self.param_window.param_controls['min_tail_body_dist'].text()))
                max_tail_body_dist = int(float(self.param_window.param_controls['max_tail_body_dist'].text()))
                eye_resize_factor  = int(float(self.param_window.param_controls['eye_resize_factor'].currentText()))
                interpolation      = str(self.param_window.param_controls['interpolation'].currentText())
            except ValueError:
                self.param_window.show_invalid_params_text()
                return

            valid_params = (0 < shrink_factor <= 1
                        and saved_video_fps > 0
                        and n_tail_points > 0
                        and tail_crop_height > 0
                        and tail_crop_width > 0
                        and min_tail_body_dist >= 0
                        and max_tail_body_dist > min_tail_body_dist
                        and eye_resize_factor > 0)

            if valid_params:
                self.params['shrink_factor']      = shrink_factor
                self.params['saved_video_fps']    = saved_video_fps
                self.params['n_tail_points']      = n_tail_points
                self.params['tail_crop']          = np.array([tail_crop_height, tail_crop_width])
                self.params['min_tail_body_dist'] = min_tail_body_dist * shrink_factor
                self.params['max_tail_body_dist'] = max_tail_body_dist
                self.params['eye_resize_factor']  = eye_resize_factor
                self.params['interpolation']      = interpolation

                self.param_window.hide_invalid_params_text()
            else:
                self.param_window.show_invalid_params_text()
                return

            # reshape current frame
            self.reshape_frame()

            if self.params['type'] == "freeswimming":
                # generate thresholded frames
                self.generate_thresholded_frames()

            # update the image preview
            self.update_preview(image=None, new_load=False, new_frame=True)

    def update_crop_from_selection(self, start_crop_coord, end_crop_coord):
        Controller.update_crop_from_selection(self, start_crop_coord, end_crop_coord)

        # generate thresholded frames
        self.generate_thresholded_frames()

class HeadfixedController(Controller):
    def __init__(self):
        # set path to where last used parameters are saved
        self.last_params_path = "last_params_headfixed.npz"

        Controller.__init__(self, default_headfixed_params, default_headfixed_crop_params)

        # create parameters window
        self.param_window = HeadfixedParamWindow(self)

        # create crops window
        self.crops_window = HeadfixedCropsWindow(self)

        self.param_window.set_gui_disabled(True)
        self.crops_window.set_gui_disabled(True)

    def track_frame(self):
        if self.params['tail_start_coords'] != None:
            self.preview_window.instructions_label.setText("")
            Controller.track_frame(self)
        else:
            self.preview_window.instructions_label.setText("Please select the start of the tail.")

    def update_crop_params_from_gui(self):
        old_crop_params = self.params['crop_params']

        # get crop params from gui
        for c in range(len(self.params['crop_params'])):
            crop_y   = int(float(self.crops_window.param_controls[c]['crop_y' + '_textbox'].text()))
            crop_x   = int(float(self.crops_window.param_controls[c]['crop_x' + '_textbox'].text()))
            offset_y = int(float(self.crops_window.param_controls[c]['offset_y' + '_textbox'].text()))
            offset_x = int(float(self.crops_window.param_controls[c]['offset_x' + '_textbox'].text()))

            valid_params = (1 <= crop_y <= self.current_frame.shape[0]
                        and 1 <= crop_x <= self.current_frame.shape[1]
                        and 0 <= offset_y < self.current_frame.shape[0]-1
                        and 0 <= offset_x < self.current_frame.shape[1]-1)

            if valid_params:
                self.params['crop_params'][c]['crop']   = np.array([crop_y, crop_x])
                self.params['crop_params'][c]['offset'] = np.array([offset_y, offset_x])
            else:
                self.param_window.show_invalid_params_text()

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
        self.params['tail_start_coords'] = tracking.get_absolute_tail_start_coords(rel_tail_start_coords,
                                                                                   self.params['crop_params'][self.current_crop]['offset'],
                                                                                   self.params['shrink_factor'])

        # reset headfixed tracking
        tracking.clear_headfixed_tracking()

    def update_params_from_gui(self):
        if self.current_frame != None:

            # get params from gui
            try:
                shrink_factor   = float(self.param_window.param_controls['shrink_factor' + '_textbox'].text())
                tail_direction  = str(self.param_window.param_controls['tail_direction'].currentText())
                saved_video_fps = int(self.param_window.param_controls['saved_video_fps'].text())
                n_tail_points   = int(self.param_window.param_controls['n_tail_points'].text())
            except ValueError:
                self.param_window.show_invalid_params_text()
                return

            valid_params = (0 < shrink_factor <= 1
                        and saved_video_fps > 0
                        and n_tail_points > 0)

            if valid_params:
                self.params['shrink_factor']   = shrink_factor
                self.params['tail_direction']  = tail_direction
                self.params['saved_video_fps'] = saved_video_fps
                self.params['n_tail_points']   = n_tail_points

                self.param_window.hide_invalid_params_text()
            else:
                self.param_window.show_invalid_params_text()
                return

            # reshape current frame
            self.reshape_frame()

            # update the image preview
            self.update_preview(image=None, new_load=False, new_frame=True)

class GetBackgroundThread(QThread):
    finished = Signal(np.ndarray, int)
    progress = Signal(int)

    def set_parameters(self, media_path, media_type, media_num, batch_offset):
        self.media_path   = media_path
        self.media_type   = media_type
        self.media_num    = media_num
        self.batch_offset = batch_offset

    def run(self):
        if self.media_type == "folder":
            background = tracking.get_background_from_folder(self.media_path, None, None, False, progress_signal=self.progress, batch_offset=self.batch_offset)
        elif self.media_type == "video":
            background = tracking.get_background_from_video(self.media_path, None, None, False, progress_signal=self.progress, batch_offset=self.batch_offset)
        
        self.finished.emit(background, self.media_num)

class TrackMediaThread(QThread):
    finished = Signal(float)
    progress = Signal(str, int, int)

    def set_parameters(self, params, tracking_path):
        self.params = params
        self.tracking_path = tracking_path

    def run(self):
        if self.params['media_type'] == "image":
            tracking_func = tracking.open_and_track_image
        elif self.params['media_type'] == "folder":
            tracking_func = tracking.open_and_track_folder
        elif self.params['media_type'] == "video":
            tracking_func = tracking.open_and_track_video_batch

        if self.tracking_path != "":
            start_time = time.time()

            tracking_func(self.params, self.tracking_path, progress_signal=self.progress)

            end_time = time.time()

            self.finished.emit(end_time - start_time)

# --- Helper functions --- #
def split_evenly(n, m, start=0):
    # generate a list of m evenly spaced numbers in the range of (start, start + n)
    # eg. split_evenly(100, 5, 30) = [40, 60, 80, 100, 120]
    return [i*n//m + n//(2*m) + start for i in range(m)]