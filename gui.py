import sys
import os
import tracking as tt
import numpy as np
import pdb
import cv2
import json
import threading
import time

from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     import multiprocessing as mp; mp.set_start_method('forkserver')

# color table to use for showing images
gray_color_table = [QtGui.qRgb(i, i, i) for i in range(256)]


default_crop_params = { 'offset': [0, 0],      # crop (y, x) offset
                        'crop': None,          # crop size
                        'tail_threshold': 200, # pixel brightness to use for thresholding to find the tail (0-255)
                        'head_threshold': 50   # pixel brightness to use for thresholding to find the eyes (0-255)
                      }

default_params = {'shrink_factor': 1.0,               # factor by which to shrink the original frame
                  'crop_params': [],                  # params for crops
                  'invert': False,                    # invert the frame
                  'type_opened': None,                # type of media that is opened - "video", "folder", "image" or None
                  'min_tail_eye_dist': 10,            # min. distance between the eyes and the tail
                  'max_tail_eye_dist': 30,            # max. distance between the eyes and the tail
                  'track_head': True,                 # track head/eyes
                  'track_tail': True,                 # track tail
                  'show_head_threshold': False,       # show head threshold in preview window
                  'show_tail_threshold': False,       # show tail threshold in preview window
                  'show_tail_skeleton': False,        # show tail skeleton in preview window
                  'last_path': "",                    # path to last opened image/folder/video
                  'save_video': True,                 # whether to make a video with tracking overlaid
                  'new_video_fps': 30,                # fps for the generated video
                  'tail_crop': [100, 100],            # dimensions of crop around zebrafish eyes to use for tail tracking - (y, x)
                  'n_tail_points': 30,                # number of tail points to use
                  'adjust_thresholds': False,         # whether to adjust thresholds while tracking if necessary
                  'eye_resize_factor': 1,             # factor by which to resize frame for reducing noise in eye position tracking
                  'interpolation': 'Nearest Neighbor' # interpolation to use when resizing frame for eye tracking
                 }

eye_resize_factor_options = [i for i in range(1, 9)]
interpolation_options = ["Nearest Neighbor", "Linear", "Bicubic", "Lanczos"]

max_n_frames = 100 # maximum # of frames to load for previewing

class PlotQLabel(QtGui.QLabel):
    """
    QLabel subclass used to show a preview image.

    Properties:
        preview_window (PreviewWindow): preview window that contains this label
        scale_factor:          (float): scale factor between label pixels & pixels of the actual image
        start_crop_coord        (y, x): starting coordinate of mouse crop selection
        end_crop_coord          (y, x): ending coordinate of mouse crop selection
    """

    def __init__(self, preview_window):
        QtGui.QLabel.__init__(self)

        self.preview_window = preview_window
        self.scale_factor = None

        # accept clicks
        self.setAcceptDrops(True)

    def mousePressEvent(self, event):
        if self.scale_factor:
            self.start_crop_coord = (int(round(event.y()/self.scale_factor)), int(round(event.x()/self.scale_factor)))

    def mouseMoveEvent(self, event):
        if self.scale_factor:
            self.end_crop_coord = (int(round(event.y()/self.scale_factor)), int(round(event.x()/self.scale_factor)))

            # draw colored crop selection
            self.preview_window.draw_crop_selection(self.start_crop_coord, self.end_crop_coord)

    def mouseReleaseEvent(self, event):
        if self.scale_factor:
            self.end_crop_coord = (int(round(event.y()/self.scale_factor)), int(round(event.x()/self.scale_factor)))

            if self.end_crop_coord != self.start_crop_coord:
                # finished selecting crop area; crop the image
                self.preview_window.crop_selection(self.start_crop_coord, self.end_crop_coord)

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

class PreviewWindow(QtGui.QMainWindow):
    """
    QMainWindow subclass used to show frames & tracking.

    Properties:
        param_window (ParamWindow): parameter window
        scale_factor:          (float): scale factor between label pixels & pixels of the actual image
        start_crop_coord        (y, x): starting coordinate of mouse crop selection
        end_crop_coord          (y, x): ending coordinate of mouse crop selection
    """

    def __init__(self, param_window):
        QtGui.QMainWindow.__init__(self)

        # set param window
        self.param_window = param_window

        # set title
        self.setWindowTitle("Preview")

        # connect signals to functions
        self.param_window.imageLoaded.connect(self.plot_image)
        self.param_window.imageTracked.connect(self.plot_tracked_image)
        self.param_window.thresholdLoaded.connect(self.plot_threshold_image)
        self.param_window.thresholdUnloaded.connect(self.remove_threshold_image)

        # create main widget
        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setStyleSheet("background-color: #666;")

        # create main layout
        self.main_layout = QtGui.QVBoxLayout(self.main_widget)

        # create label that shows frames
        self.image_label = PlotQLabel(self)
        self.image_label.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.addWidget(self.image_label)

        # create label that shows crop instructions
        self.instructions_label = QtGui.QLabel("")
        self.instructions_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.addWidget(self.instructions_label)

        # initialize variables
        self.image_slider   = None  # slider for selecting frames
        self.image          = None  # image to show
        self.pixmap         = None  # image label's pixmap
        self.pixmap_size    = None  # size of image label's pixmap
        self.tracking_data  = None  # list of tracking data
        self.selecting_crop = False # whether user is selecting a crop

        # set main widget
        self.setCentralWidget(self.main_widget)

        # set window buttons
        self.setWindowFlags(QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint)

        self.show()

    def resizeEvent(self, event):
        # get new size
        size = event.size()

        # change width to within min & max values
        width = max(min(size.width(), 900), 400)

        # set new size of pixmap
        self.pixmap_size = width - 40

        if self.pixmap:
            # calculate new label vs. image scale factor
            scale_factor = float(self.pixmap_size)/max(self.pixmap.width(), self.pixmap.height())
            self.image_label.set_scale_factor(scale_factor)

            # scale pixmap
            pix = self.pixmap.scaled(self.pixmap_size, self.pixmap_size, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.FastTransformation)

            # update label's pixmap & size
            self.image_label.setPixmap(pix)
            self.image_label.setFixedSize(pix.size())

        # constrain window to square size
        self.resize(width, width)

    def start_select_crop(self):
        # start selecting crop
        self.selecting_crop = True

        # add instruction text
        self.instructions_label.setText("Click & drag to select crop area.")

    def plot_image(self, image, tail_crop, new_image=False):
        if self.image_slider:
            # update image slider
            self.image_slider.setMaximum(self.param_window.n_frames-1)
        else:
            # create image slider
            self.image_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
            self.image_slider.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.image_slider.setTickPosition(QtGui.QSlider.TicksBothSides)
            self.image_slider.setTickInterval(1)
            self.image_slider.setSingleStep(1)
            self.image_slider.setMinimum(0)
            self.image_slider.setMaximum(self.param_window.n_frames-1)
            self.image_slider.setValue(0)
            self.image_slider.valueChanged.connect(self.switch_frame)
            self.main_layout.addWidget(self.image_slider)

        # update image
        self.image = image

        if new_image:
            # reset tracking data
            self.tracking_data = None

        if self.tracking_data:
            # plot image with tracking
            self.plot_tracked_image(image, tail_crop, self.tracking_data)
        else:
            # convert image to rgb
            if len(image.shape) < 3:
                rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            else:
                rgb_image = image

            # update image label
            self.update_image_label(rgb_image)

    def plot_threshold_image(self, threshold_image, tail_crop, new_image=False):
        new_threshold_image = threshold_image

        if new_image:
            # reset tracking data
            self.tracking_data = None

        if self.tracking_data:
            # plot image with tracking
            self.plot_tracked_image(new_threshold_image, tail_crop, self.tracking_data)
        else:
            # convert image to rgb
            if len(new_threshold_image.shape) < 3:
                rgb_image = np.repeat(new_threshold_image[:, :, np.newaxis], 3, axis=2)
            else:
                rgb_image = new_threshold_image

            # update image label
            self.update_image_label(rgb_image)

    def remove_threshold_image(self, image):
        # plot non-threshold image
        self.plot_image(image, self.param_window.params['tail_crop'])

    def switch_frame(self, value):
        self.param_window.switch_frame(value)

    def plot_tracked_image(self, image, tail_crop, tracking_data):
        # set tracking data
        self.tracking_data = tracking_data

        # get tracking coords
        tail_coords    = self.tracking_data[0]
        spline_coords  = self.tracking_data[1]
        eye_coords     = self.tracking_data[2]
        heading_coords = self.tracking_data[3]

        # add tracking to image
        image = tt.add_tracking_to_image(image, tail_coords, spline_coords,
                                         eye_coords, heading_coords)

        # convert image to rgb
        if len(image.shape) < 3:
            rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        else:
            rgb_image = image

        if eye_coords != None:
            # get coordinates of the midpoint of the eyes
            mid_coords = [(eye_coords[0, 0] + eye_coords[0, 1])/2.0, (eye_coords[1, 0] + eye_coords[1, 1])/2.0]

            if tail_crop != None:
                # copy image
                overlay = image.copy()

                # draw tail crop overlay
                cv2.rectangle(overlay, (int(mid_coords[1]-tail_crop[1]), int(mid_coords[0]-tail_crop[0])),
                                        (int(mid_coords[1]+tail_crop[1]), int(mid_coords[0]+tail_crop[0])), (242, 242, 65), -1)

                # overlay with the original image
                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, rgb_image)

        # update image label
        self.update_image_label(rgb_image)

    def draw_crop_selection(self, start_crop_coord, end_crop_coord):
        if self.selecting_crop:
            # convert image to rgb
            if len(self.image.shape) < 3:
                rgb_image = np.repeat(self.image[:, :, np.newaxis], 3, axis=2)
            else:
                rgb_image = self.image

            # copy image
            overlay = rgb_image.copy()

            # draw crop selection overlay
            cv2.rectangle(overlay, (start_crop_coord[1], start_crop_coord[0]), (end_crop_coord[1], end_crop_coord[0]), (0, 51, 255), -1)

            # overlay with the original image
            cv2.addWeighted(overlay, 0.5, rgb_image, 0.5, 0, rgb_image)

            # update image label
            self.update_image_label(rgb_image)

    def update_image_label(self, image):
        # get image info
        height, width, bytesPerComponent = image.shape
        bytesPerLine = bytesPerComponent * width
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)

        # create qimage
        qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], bytesPerLine, QtGui.QImage.Format_RGB888)
        qimage.setColorTable(gray_color_table)

        # generate pixmap
        self.pixmap = QtGui.QPixmap(qimage)

        # update label vs. image scale factor
        self.image_label.set_scale_factor(float(self.pixmap_size)/max(self.pixmap.width(), self.pixmap.height()))

        # scale pixmap
        pixmap = self.pixmap.scaled(self.pixmap_size, self.pixmap_size, QtCore.Qt.KeepAspectRatio)

        # set image label's pixmap & update label's size
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())

    def crop_selection(self, start_crop_coord, end_crop_coord):
        if self.selecting_crop:
            # stop selecting the crop
            self.selecting_crop = False

            # clear instruction text
            self.instructions_label.setText("")

            # update crop parameters from the selection
            self.param_window.update_crop_from_selection(start_crop_coord, end_crop_coord)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

class ParamWindow(QtGui.QMainWindow):
    # initiate drawing signals
    imageLoaded       = QtCore.Signal(np.ndarray, list, bool)
    imageTracked      = QtCore.Signal(np.ndarray, list, list)
    thresholdLoaded   = QtCore.Signal(np.ndarray, list, bool)
    thresholdUnloaded = QtCore.Signal(np.ndarray)

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        # set window title
        self.setWindowTitle("Parameters")

        # create preview window
        self.preview_window = PreviewWindow(self)

        # initiate image vars
        self.current_frame        = None
        self.cropped_frame        = None
        self.shrunken_frame       = None
        self.head_threshold_frame = None
        self.tail_threshold_frame = None
        self.tail_skeleton_frame  = None
        self.n_frames             = 0
        self.current_crop_num     = -1

        # set default params
        self.current_crop_params = default_crop_params
        self.params              = default_params

        # initialize parameter controls variable
        self.param_controls = None

        # set experiments file
        self.params_file = "last_params.json"

        # add actions
        openImage = QtGui.QAction(QtGui.QIcon('open.png'), 'Open Image', self)
        openImage.setShortcut('Ctrl+O')
        openImage.setStatusTip('Open an image')
        openImage.triggered.connect(lambda:self.open_image(""))

        openFolder = QtGui.QAction(QtGui.QIcon('open.png'), 'Open Folder', self)
        openFolder.setShortcut('Ctrl+Shift+O')
        openFolder.setStatusTip('Open a folder of images')
        openFolder.triggered.connect(lambda:self.open_folder(""))

        openVideo = QtGui.QAction(QtGui.QIcon('open.png'), 'Open Video', self)
        openVideo.setShortcut('Ctrl+Alt+O')
        openVideo.setStatusTip('Open a video')
        openVideo.triggered.connect(lambda:self.open_video(""))

        trackFrame = QtGui.QAction(QtGui.QIcon('open.png'), 'Track Frame', self)
        trackFrame.setShortcut('Ctrl+T')
        trackFrame.setStatusTip('Track current frame')
        trackFrame.triggered.connect(self.track_frame)

        saveParams = QtGui.QAction(QtGui.QIcon('save.png'), 'Save Parameters', self)
        saveParams.setShortcuts(['Ctrl+S'])
        saveParams.setStatusTip('Save parameters')
        saveParams.triggered.connect(self.save_params)

        # add menu bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFolder)
        fileMenu.addAction(openVideo)
        fileMenu.addAction(openImage)
        fileMenu.addAction(saveParams)
        fileMenu.addAction(trackFrame)

        # create main widget & layout
        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)

        self.layout = QtGui.QVBoxLayout(self.main_widget)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addStretch(1)
        self.layout.setSpacing(5)

        # initialize dict used for accessing all crop parameter controls
        self.crop_param_controls = []

        # create tabs widget
        self.crop_tabs_widget = QtGui.QTabWidget()
        self.crop_tabs_widget.setUsesScrollButtons(True)
        self.crop_tabs_widget.tabCloseRequested.connect(self.remove_crop)
        self.crop_tabs_widget.currentChanged.connect(self.change_crop)
        self.crop_tabs_widget.setElideMode(QtCore.Qt.ElideLeft)
        self.crop_tabs_widget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.crop_tabs_widget.setMinimumSize(276, 300)
        self.layout.addWidget(self.crop_tabs_widget)

        # create tabs layout
        crop_tabs_layout  = QtGui.QHBoxLayout(self.crop_tabs_widget)

        # create lists for storing all crop tab widgets & layouts
        self.crop_tab_layouts = []
        self.crop_tab_widgets = []

        # create a default crop
        self.create_crop()

        # add crop button layout
        crop_button_layout = QtGui.QHBoxLayout()
        crop_button_layout.setSpacing(5)
        crop_button_layout.addStretch(1)
        self.layout.addLayout(crop_button_layout)

        # add delete crop button
        self.remove_crop_button = QtGui.QPushButton(u'\u2717 Remove Crop', self)
        self.remove_crop_button.setMinimumHeight(10)
        self.remove_crop_button.setMaximumWidth(120)
        self.remove_crop_button.clicked.connect(lambda:self.remove_crop(self.current_crop_num))
        self.remove_crop_button.setDisabled(True)
        crop_button_layout.addWidget(self.remove_crop_button)

        # add new crop button
        self.create_crop_button = QtGui.QPushButton(u'\u270E New Crop', self)
        self.create_crop_button.setMinimumHeight(10)
        self.create_crop_button.setMaximumWidth(100)
        self.create_crop_button.clicked.connect(lambda:self.create_crop())
        crop_button_layout.addWidget(self.create_crop_button)

        # create form layout
        self.form_layout = QtGui.QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.layout.addLayout(self.form_layout)

        # create dict for storing all parameter controls
        self.param_controls = {}

        # add checkboxes - (key, description, function to call, initial value, parent layout)
        self.add_checkbox('invert', "Invert image", self.toggle_invert_image, self.params['invert'], self.layout)
        self.add_checkbox('show_head_threshold', "Show head threshold", self.toggle_threshold_image, self.params['show_head_threshold'], self.layout)
        self.add_checkbox('show_tail_threshold', "Show tail threshold", self.toggle_threshold_image, self.params['show_tail_threshold'], self.layout)
        self.add_checkbox('show_tail_skeleton', "Show tail skeleton", self.toggle_threshold_image, self.params['show_tail_skeleton'], self.layout)
        self.add_checkbox('track_head', "Track head", self.toggle_tracking, self.params['track_head'], self.layout)
        self.add_checkbox('track_tail', "Track tail", self.toggle_tracking, self.params['track_tail'], self.layout)
        self.add_checkbox('save_video', "Save video", self.toggle_save_video, self.params['save_video'], self.layout)
        self.add_checkbox('adjust_thresholds', 'Adjust thresholds', self.toggle_adjust_thresholds, self.params['adjust_thresholds'], self.layout)

        # add sliders - (key, description, start, end, initial value, parent layout)
        self.add_slider('shrink_factor', 'Shrink factor:', 1, 10, 10.0*self.params['shrink_factor'], self.form_layout, multiplier=10.0)
        self.add_slider('tail_crop_height', 'Tail crop height:', 1, 100, round(self.params['tail_crop'][0]), self.form_layout, tick_interval=10)
        self.add_slider('tail_crop_width', 'Tail crop width:', 1, 100, round(self.params['tail_crop'][1]), self.form_layout, tick_interval=10)

        # add textboxes - (key, decription, initial value, parent layout)
        self.add_textbox('min_tail_eye_dist', 'Head/tail min. dist.:', self.params['min_tail_eye_dist'], self.form_layout)
        self.add_textbox('max_tail_eye_dist', 'Head/tail max. dist.:', self.params['max_tail_eye_dist'], self.form_layout)
        self.add_textbox('new_video_fps', 'Saved video FPS:', self.params['new_video_fps'], self.form_layout)
        self.add_textbox('n_tail_points', '# tail points:', self.params['n_tail_points'], self.form_layout)

        # add comboboxes
        self.add_combobox('eye_resize_factor', 'Resize factor for eyes:', eye_resize_factor_options, self.params['eye_resize_factor'], self.form_layout)
        self.add_combobox('interpolation', 'Interpolation:', interpolation_options, self.params['interpolation'], self.form_layout)

        # add button layouts
        button_layout_1 = QtGui.QHBoxLayout()
        button_layout_1.setSpacing(5)
        button_layout_1.addStretch(1)
        self.layout.addLayout(button_layout_1)

        button_layout_2 = QtGui.QHBoxLayout()
        button_layout_2.setSpacing(5)
        button_layout_2.addStretch(1)
        self.layout.addLayout(button_layout_2)

        # add buttons
        self.save_button = QtGui.QPushButton(u'\u2713 Save', self)
        self.save_button.setMinimumHeight(10)
        self.save_button.setMaximumWidth(80)
        self.save_button.clicked.connect(self.save_params)
        button_layout_1.addWidget(self.save_button)

        self.track_button = QtGui.QPushButton(u'\u279E Track', self)
        self.track_button.setMinimumHeight(10)
        self.track_button.setMaximumWidth(80)
        self.track_button.clicked.connect(self.track_frame)
        button_layout_1.addWidget(self.track_button)

        self.track_all_button = QtGui.QPushButton(u'\u27A0 Track All', self)
        self.track_all_button.setMinimumHeight(10)
        self.track_all_button.setMaximumWidth(180)
        self.track_all_button.clicked.connect(self.track)
        self.track_all_button.setStyleSheet("font-weight: bold")
        button_layout_1.addWidget(self.track_all_button)

        self.reload_last_save_button = QtGui.QPushButton(u'\u27AA Reload', self)
        self.reload_last_save_button.setMinimumHeight(10)
        self.reload_last_save_button.setMaximumWidth(90)
        self.reload_last_save_button.clicked.connect(lambda:self.reload_last_save())
        button_layout_1.addWidget(self.reload_last_save_button)

        self.open_image_button = QtGui.QPushButton('+ Image', self)
        self.open_image_button.setMinimumHeight(10)
        self.open_image_button.setMaximumWidth(70)
        self.open_image_button.clicked.connect(lambda:self.open_image(""))
        button_layout_1.addWidget(self.open_image_button)

        self.open_folder_button = QtGui.QPushButton('+ Folder', self)
        self.open_folder_button.setMinimumHeight(10)
        self.open_folder_button.setMaximumWidth(70)
        self.open_folder_button.clicked.connect(lambda:self.open_folder(""))
        button_layout_1.addWidget(self.open_folder_button)

        self.open_video_button = QtGui.QPushButton('+ Video', self)
        self.open_video_button.setMinimumHeight(10)
        self.open_video_button.setMaximumWidth(70)
        self.open_video_button.clicked.connect(lambda:self.open_video(""))
        button_layout_1.addWidget(self.open_video_button)

        # set window buttons
        self.setWindowFlags(QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint)

        # set main widget
        self.setCentralWidget(self.main_widget)

        # disable controls
        self.set_gui_disabled(True)

        self.show()

    def set_gui_disabled(self, disbaled_bool):
        self.crop_tabs_widget.setDisabled(disbaled_bool)

        self.remove_crop_button.setDisabled(disbaled_bool)
        self.create_crop_button.setDisabled(disbaled_bool)

        for param_control in self.param_controls.values():
            if type(param_control) == list:
                param_control[0].setDisabled(disbaled_bool)
                param_control[1].setDisabled(disbaled_bool)
            else:
                param_control.setDisabled(disbaled_bool)

        self.save_button.setDisabled(disbaled_bool)
        self.track_button.setDisabled(disbaled_bool)
        self.track_all_button.setDisabled(disbaled_bool)

    def create_crop(self, crop_params=None):
        # create crop tab widget & layout
        crop_tab_widget = QtGui.QWidget(self.crop_tabs_widget)
        crop_tab_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        crop_tab_widget.resize(276, 300)

        crop_tab_layout = QtGui.QVBoxLayout(crop_tab_widget)

        # add to list of crop widgets & layouts
        self.crop_tab_layouts.append(crop_tab_layout)
        self.crop_tab_widgets.append(crop_tab_widget)

        # create form layout
        form_layout = QtGui.QFormLayout()
        form_layout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        crop_tab_layout.addLayout(form_layout)

        # add dict for storing param controls for this crop
        self.crop_param_controls.append({})

        # add sliders - (key, description, start, end, initial value, parent layout)
        if self.current_frame != None:
            self.add_slider('crop_y', 'Crop y:', 1, self.current_frame.shape[0], self.current_frame.shape[0], form_layout, tick_interval=50)
            self.add_slider('crop_x', 'Crop x:', 1, self.current_frame.shape[1], self.current_frame.shape[1], form_layout, tick_interval=50)
            self.add_slider('offset_y', 'Offset y:', 0, self.current_frame.shape[0]-1, 0, form_layout, tick_interval=50)
            self.add_slider('offset_x', 'Offset x:', 0, self.current_frame.shape[1]-1, 0, form_layout, tick_interval=50)
        else:
            self.add_slider('crop_y', 'Crop y:', 1, 2, 1, form_layout, tick_interval=50)
            self.add_slider('crop_x', 'Crop x:', 1, 2, 1, form_layout, tick_interval=50)
            self.add_slider('offset_y', 'Offset y:', 0, 1, 0, form_layout, tick_interval=50)
            self.add_slider('offset_x', 'Offset x:', 0, 1, 0, form_layout, tick_interval=50)

        # add textboxes - (key, decription, initial value, parent layout)
        self.add_textbox('head_threshold', 'Head threshold:', self.current_crop_params['head_threshold'], form_layout)
        self.add_textbox('tail_threshold', 'Tail threshold:', self.current_crop_params['tail_threshold'], form_layout)

        # create crop button layout
        crop_button_layout = QtGui.QHBoxLayout()
        crop_button_layout.setSpacing(5)
        crop_button_layout.addStretch(1)
        crop_tab_layout.addLayout(crop_button_layout)

        # add crop buttons
        self.reset_crop_button = QtGui.QPushButton(u'\u25A8 Reset Crop', self)
        self.reset_crop_button.setMinimumHeight(10)
        self.reset_crop_button.setMaximumWidth(110)
        self.reset_crop_button.clicked.connect(self.reset_crop)
        crop_button_layout.addWidget(self.reset_crop_button)

        self.select_crop_button = QtGui.QPushButton(u'\u25A3 Select Crop', self)
        self.select_crop_button.setMinimumHeight(10)
        self.select_crop_button.setMaximumWidth(110)
        self.select_crop_button.clicked.connect(self.select_crop)
        crop_button_layout.addWidget(self.select_crop_button)

        if crop_params == None:
            # no params are given for this crop; set to default parameters
            self.params['crop_params'].append(default_crop_params.copy())
        
        # update current crop number
        self.current_crop_num = len(self.params['crop_params']) - 1

        # update gui
        # self.update_crop_param_gui()

        # add crop widget as a tab
        self.crop_tabs_widget.addTab(crop_tab_widget, str(self.current_crop_num))

        # make this crop the active tab
        self.crop_tabs_widget.setCurrentIndex(self.current_crop_num)

        # update text on all tabs
        for i in range(len(self.params['crop_params'])):
                self.crop_tabs_widget.setTabText(i, str(i))

    def change_crop(self, index):
        if index != -1:
            # get params for this crop
            self.current_crop_params = self.params['crop_params'][index]

            # update current crop number
            self.current_crop_num = index

            # update the gui with these crop params
            self.update_crop_param_gui()

            # update the image preview
            self.reshape_frame()
            self.update_preview(new_frame=True)

    def remove_crop(self, index):
        # get current number of crops
        n_crops = len(self.params['crop_params'])

        if n_crops > 1:
            # delete params for this crop
            del self.params['crop_params'][index]

            # remove the tab
            self.crop_tabs_widget.removeTab(index)

            # delete this tab's controls, widget & layout
            del self.crop_param_controls[index]
            del self.crop_tab_widgets[index]
            del self.crop_tab_layouts[index]

            # set current crop to last tab
            self.current_crop_num = len(self.params['crop_params']) - 1
            self.current_crop_params = self.params['crop_params'][-1]

            # update text on all tabs
            for i in range(len(self.params['crop_params'])):
                self.crop_tabs_widget.setTabText(i, str(i))

    def clear_crops(self):
        # get current number of crops
        n_crops = len(self.params['crop_params'])

        print(n_crops)

        for c in range(n_crops-1, -1, -1):
            # delete params
            del self.params['crop_params'][c]

            # remove tab
            self.crop_tabs_widget.removeTab(c)

            # delete controls, widget & layout
            del self.crop_param_controls[c]
            del self.crop_tab_widgets[c]
            del self.crop_tab_layouts[c]

        # reset current crop
        self.params['crop_params'] = []
        self.current_crop_params = None
        self.current_crop_num = -1

    def select_crop(self):
        # user wants to draw a crop selection; start selecting
        self.preview_window.start_select_crop()

    def hideEvent(self, event):
        # quit app when clicking 'x' button on macOS
        self.preview_window.closeEvent(event)

    def reload_last_save(self):
        # re-load last saved state

        # clear all crops
        self.clear_crops()

        # load saved parameters
        self.load_params()

        # add all saved crops
        for j in range(len(self.params['crop_params'])):
            self.create_crop(self.params['crop_params'][j])

        # re-open the last file
        self.open_last_file()

    def load_params(self):
        try:
            new_params = default_params

            # load params from saved file
            with open(self.params_file, "r") as input_file:
                saved_params = json.load(input_file)
                new_params.update(saved_params)
        except:
            self.params['crop_params'] = [default_crop_params]
            self.current_crop_params = self.params['crop_params'][0]
        else:
            self.params = new_params
            self.current_crop_params = self.params['crop_params'][0]

    def open_last_file(self):
        if self.params['type_opened'] == "video":
            print("Opening video.")
            self.open_video(path=self.params['last_path'], reset_params=False)
        elif self.params['type_opened'] == "folder":
            print("Opening folder.")
            self.open_folder(path=self.params['last_path'], reset_params=False)
        elif self.params['type_opened'] == "image":
            print("Opening image.")
            self.open_image(path=self.params['last_path'], reset_params=False)
        else:
            return

    def add_textbox(self, label, description, default_value, parent):
        # make textbox & add row to form layout
        param_box = QtGui.QLineEdit()
        param_box.setMinimumHeight(10)
        param_box.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        parent.addRow(description, param_box)

        # set default text
        if default_value != None:
            param_box.setText(str(default_value))

        # add to list of crop or global controls
        if label in ('crop_y', 'crop_x', 'offset_y', 'offset_x', 'head_threshold', 'tail_threshold'):
            self.crop_param_controls[-1][label] = param_box
        else:
            self.param_controls[label] = param_box

    def add_slider(self, label, description, minimum, maximum, value, parent, tick_interval=1, single_step=1, multiplier=1):
        # make layout to hold slider and textbox
        control_layout = QtGui.QHBoxLayout()

        # make slider & add to layout
        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        slider.setTickPosition(QtGui.QSlider.TicksBothSides)
        slider.setTickInterval(tick_interval)
        slider.setSingleStep(single_step)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        control_layout.addWidget(slider)

        # make textbox & add to layout
        textbox = QtGui.QLineEdit()
        textbox.setMinimumHeight(10)
        textbox.setFixedWidth(40)
        textbox.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        textbox.setText(str(value/multiplier))
        control_layout.addWidget(textbox)

        # connect slider to set textbox text & update params
        slider.sliderMoved.connect(lambda:self.update_textbox_from_slider(slider, textbox, multiplier))
        slider.sliderMoved.connect(self.update_params_from_gui)

        # connect textbox to 
        textbox.editingFinished.connect(lambda:self.update_slider_from_textbox(slider, textbox, multiplier))
        textbox.editingFinished.connect(self.update_params_from_gui)

        # add row to form layout
        parent.addRow(description, control_layout)

        # add to list of crop or global controls
        if label in ('crop_y', 'crop_x', 'offset_y', 'offset_x', 'head_threshold', 'tail_threshold'):
            self.crop_param_controls[-1][label] = [slider, textbox]
        else:
            self.param_controls[label] = [slider, textbox]

    def update_textbox_from_slider(self, slider, textbox, multiplier=1.0):
        textbox.setText(str(slider.sliderPosition()/multiplier))

    def update_slider_from_textbox(self, slider, textbox, multiplier=1.0):
        slider.setValue(float(textbox.text())*multiplier)

    def add_checkbox(self, label, description, toggle_func, checked, parent):
        # make checkbox & add to layout
        checkbox = QtGui.QCheckBox(description)
        checkbox.setChecked(checked)
        checkbox.toggled.connect(lambda:toggle_func(checkbox))
        parent.addWidget(checkbox)

        # add to list of crop or global controls
        if label in ('crop_y', 'crop_x', 'offset_y', 'offset_x', 'head_threshold', 'tail_threshold'):
            self.crop_param_controls[-1][label] = checkbox
        else:
            self.param_controls[label] = checkbox

    def add_combobox(self, label, description, options, value, parent):
        combobox = QtGui.QComboBox()
        combobox.addItems([ str(o) for o in options])
        combobox.setCurrentIndex(options.index(value))
        parent.addRow(description, combobox)

        self.param_controls[label] = combobox

    def open_folder(self, path="", reset_params=True):
        if path == "":
            # ask the user to select a directory
            path = str(QtGui.QFileDialog.getExistingDirectory(self, 'Open folder'))

        if path not in ("", None):
            # get paths to all the frame images & the number of frames
            self.image_paths = []
            self.n_frames = 0

            self.params['last_path'] = path

            for filename in sorted(os.listdir(self.params['last_path'])):
                if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg'):
                    image_path = self.params['last_path'] + "/" + filename

                    self.image_paths.append(image_path)
                    self.n_frames += 1

            if self.n_frames == 0:
                # no files found; end here
                print("Error: Could not find any images.")
                return

            if self.n_frames >= max_n_frames:
                # get evenly spaced frame numbers (so we don't load all the frames)
                f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
                self.image_paths = [ self.image_paths[i] for i in f(max_n_frames, self.n_frames)]
                self.n_frames = max_n_frames

            if reset_params:
                # clear all crops
                self.clear_crops()

                # set params to defaults
                self.params = default_params

                # set crop params to defaults
                self.current_crop_params = default_crop_params

                # create a default crop
                self.create_crop()

            # set path
            self.params['last_path'] = path

            # set type of opened media
            self.params['type_opened'] = 'folder'

            # switch to first frame
            self.switch_frame(0)

            # update gui
            self.update_crop_param_gui()
            self.update_global_param_gui()

            # enable controls
            self.set_gui_disabled(False)

    def open_image(self, path="", reset_params=True):
        if path == "":
            # ask the user to select an image
            path = str(QtGui.QFileDialog.getOpenFileName(self, 'Open image', '', 'Images (*.jpg  *.jpeg *.tif *.tiff *.png)'))

        if path not in ("", None):
            # set number of frames
            self.n_frames = 1

            if reset_params:
                # clear all crops
                self.clear_crops()

                # set params to defaults
                self.params = default_params

                # set crop params to defaults
                self.current_crop_params = default_crop_params

                # create a default crop
                self.create_crop()

            # set path
            self.params['last_path'] = path

            # set type of opened media
            self.params['type_opened'] = 'image'

            # switch to first (and only) frame
            self.switch_frame(0)
            
            # update gui
            self.update_crop_param_gui()
            self.update_global_param_gui()

            # enable controls
            self.set_gui_disabled(False)

    def open_video(self, path="", reset_params=True):
        if path == "":
            # ask the user to select a video
            path = str(QtGui.QFileDialog.getOpenFileName(self, 'Open video', '', 'Videos (*.mov *.tif *.mp4 *.avi)'))

        if path not in ("", None):
            self.params['last_path'] = path

            # load frames from the video
            self.frames = tt.load_frames_from_video(self.params['last_path'], n_frames=max_n_frames)

            print(len(self.frames))

            if self.frames == None:
                # no frames found; end here
                print("Error: Could not load frames.")
                return

            # set number of frames
            self.n_frames = len(self.frames)

            if reset_params:
                # clear all crops
                self.clear_crops()

                # set params to defaults
                self.params = default_params

                # set crop params to defaults
                self.current_crop_params = default_crop_params

                # create a default crop
                self.create_crop()

            # set path
            self.params['last_path'] = path

            # set type of opened media
            self.params['type_opened'] = 'video'

            # switch to first frame
            self.switch_frame(0)

            # update gui
            self.update_crop_param_gui()
            self.update_global_param_gui()

            # enable controls
            self.set_gui_disabled(False)

    def set_slider_value(self, slider_widgets, value, slider_scale_factor=None):
        # change slider value without sending signals
        slider = slider_widgets[0]

        if value == None:
            value = slider.minimum()

        slider.blockSignals(True)

        if slider_scale_factor != None:
            slider.setValue(value*slider_scale_factor)
        else:
            slider.setValue(value)

        slider.blockSignals(False)

        # change textbox value
        textbox = slider_widgets[1]
        textbox.setText(str(float(value)))

    def update_crop_param_gui(self):
        # update param controls with current parameters
        if self.current_crop_params['crop'] != None:
            self.set_slider_value(self.crop_param_controls[self.current_crop_num]['crop_y'], self.current_crop_params['crop'][0])
            self.set_slider_value(self.crop_param_controls[self.current_crop_num]['crop_x'], self.current_crop_params['crop'][1])
            self.set_slider_value(self.crop_param_controls[self.current_crop_num]['offset_y'], self.current_crop_params['offset'][0])
            self.set_slider_value(self.crop_param_controls[self.current_crop_num]['offset_x'], self.current_crop_params['offset'][1])
        elif self.current_frame != None:
            self.set_slider_value(self.crop_param_controls[self.current_crop_num]['crop_y'], self.current_frame.shape[0])
            self.set_slider_value(self.crop_param_controls[self.current_crop_num]['crop_x'], self.current_frame.shape[1])
            self.set_slider_value(self.crop_param_controls[self.current_crop_num]['offset_y'], 0)
            self.set_slider_value(self.crop_param_controls[self.current_crop_num]['offset_x'], 0)

        self.crop_param_controls[self.current_crop_num]['tail_threshold'].setText(str(self.current_crop_params['tail_threshold']))
        self.crop_param_controls[self.current_crop_num]['head_threshold'].setText(str(self.current_crop_params['head_threshold']))

    def update_global_param_gui(self):
        # update param controls with current parameters
        if self.param_controls != None:
            self.set_slider_value(self.param_controls['shrink_factor'], self.params['shrink_factor'], slider_scale_factor=10)
            self.set_slider_value(self.param_controls['tail_crop_height'], self.params['tail_crop'][0])
            self.set_slider_value(self.param_controls['tail_crop_width'], self.params['tail_crop'][1])

            self.param_controls['min_tail_eye_dist'].setText(str(self.params['min_tail_eye_dist']))
            self.param_controls['max_tail_eye_dist'].setText(str(self.params['max_tail_eye_dist']))
            self.param_controls['new_video_fps'].setText(str(self.params['new_video_fps']))
            self.param_controls['n_tail_points'].setText(str(self.params['n_tail_points']))

            self.param_controls['eye_resize_factor'].setCurrentIndex(eye_resize_factor_options.index(self.params['eye_resize_factor']))
            self.param_controls['interpolation'].setCurrentIndex(interpolation_options.index(self.params['interpolation']))

            self.param_controls['invert'].setChecked(self.params['invert'])
            self.param_controls['show_head_threshold'].setChecked(self.params['show_head_threshold'])
            self.param_controls['show_tail_threshold'].setChecked(self.params['show_tail_threshold'])
            self.param_controls['show_tail_skeleton'].setChecked(self.params['show_tail_skeleton'])
            self.param_controls['track_head'].setChecked(self.params['track_head'])
            self.param_controls['track_tail'].setChecked(self.params['track_tail'])
            self.param_controls['save_video'].setChecked(self.params['save_video'])
            self.param_controls['adjust_thresholds'].setChecked(self.params['adjust_thresholds'])

    def switch_frame(self, n):
        # set current frame
        if self.params['type_opened'] == 'video':
            self.current_frame = self.frames[n]
        elif self.params['type_opened'] == 'folder':
            self.current_frame = tt.load_frame_from_image(self.image_paths[n])
        elif self.params['type_opened'] == 'image':
            self.current_frame = tt.load_frame_from_image(self.params['last_path'])

        if self.param_controls['invert'].isChecked():
            # invert the frame
            self.invert_frame()

        if self.current_crop_params['crop'] == None:
            # update crop
            self.current_crop_params['crop'] = self.current_frame.shape

        for c in range(len(self.params['crop_params'])):
            self.crop_param_controls[c]['crop_y'][0].setMaximum(self.current_frame.shape[0])
            self.crop_param_controls[c]['crop_x'][0].setMaximum(self.current_frame.shape[1])
            self.crop_param_controls[c]['offset_y'][0].setMaximum(self.current_frame.shape[0]-1)
            self.crop_param_controls[c]['offset_x'][0].setMaximum(self.current_frame.shape[1]-1)

        # reshape the image
        self.reshape_frame()

        # update the image preview
        self.update_preview(new_frame=True)
        # self.track_frame(update_params=False)

    def reshape_frame(self):
        if self.current_frame != None:
            # shrink the image
            if self.params['shrink_factor'] != None:
                self.shrunken_frame = tt.shrink_image(self.current_frame, self.params['shrink_factor'])
            else:
                self.shrunken_frame = self.current_frame

            # crop the image
            if self.current_crop_params['crop'] is not None and self.current_crop_params['offset'] is not None:
                crop = (round(self.current_crop_params['crop'][0]*self.params['shrink_factor']), round(self.current_crop_params['crop'][1]*self.params['shrink_factor']))
                offset = (round(self.current_crop_params['offset'][0]*self.params['shrink_factor']), round(self.current_crop_params['offset'][1]*self.params['shrink_factor']))

                self.cropped_frame = tt.crop_image(self.shrunken_frame, offset, crop)
            else:
                self.cropped_frame = self.shrunken_frame

            # generate thresholded frames
            self.generate_threshold_frames()

    def generate_threshold_frames(self):
        # generate head & tail thresholded frames
        if self.current_frame != None:
            self.head_threshold_frame = tt.get_head_threshold_image(self.cropped_frame, self.current_crop_params['head_threshold'])
            self.tail_threshold_frame = tt.get_tail_threshold_image(self.cropped_frame, self.current_crop_params['tail_threshold'])
            self.tail_skeleton_frame = tt.get_tail_skeleton_image(self.tail_threshold_frame)

    def update_preview(self, new_frame=False):
        if self.current_frame != None:
            # plot frame. for threshold/skeleton images, change nonzero pixels to 255 (so they show up as white)
            if self.param_controls["show_head_threshold"].isChecked():
                self.thresholdLoaded.emit(self.head_threshold_frame*255, self.params['tail_crop'], new_frame)
            elif self.param_controls["show_tail_threshold"].isChecked():
                self.thresholdLoaded.emit(self.tail_threshold_frame*255, self.params['tail_crop'], new_frame)
            elif self.param_controls["show_tail_skeleton"].isChecked():
                self.thresholdLoaded.emit(self.tail_skeleton_frame*255, self.params['tail_crop'], new_frame)
            else:
                self.imageLoaded.emit(self.cropped_frame, self.params['tail_crop'], new_frame)

    def invert_frame(self):
        if self.current_frame != None:
            # invert frames
            self.current_frame  = (255 - self.current_frame)
            self.shrunken_frame = (255 - self.shrunken_frame)
            self.cropped_frame  = (255 - self.cropped_frame)

            # generate thresholded frames
            self.generate_threshold_frames()

    def toggle_invert_image(self, checkbox):
        if checkbox.isChecked():
            self.params['invert'] = True
        else:
            self.params['invert'] = False

        self.invert_frame()

        # update the image preview
        self.update_preview(new_frame=True)

    def toggle_save_video(self, checkbox):
        if checkbox.isChecked():
            self.params['save_video'] = True
        else:
            self.params['save_video'] = False

    def toggle_adjust_thresholds(self, checkbox):
        if checkbox.isChecked():
            self.params['adjust_thresholds'] = True
        else:
            self.params['adjust_thresholds'] = False

    def toggle_threshold_image(self, checkbox):
        if self.current_frame != None:
            if checkbox.isChecked():
                # uncheck other threshold checkbox
                if checkbox.text() == "Show head threshold":
                    self.param_controls["show_tail_threshold"].setChecked(False)
                    self.param_controls["show_tail_skeleton"].setChecked(False)
                elif checkbox.text() == "Show tail threshold":
                    self.param_controls["show_head_threshold"].setChecked(False)
                    self.param_controls["show_tail_skeleton"].setChecked(False)
                elif checkbox.text() == "Show tail skeleton":
                    self.param_controls["show_head_threshold"].setChecked(False)
                    self.param_controls["show_tail_threshold"].setChecked(False)

            self.params['show_tail_threshold'] = self.param_controls["show_tail_threshold"].isChecked()
            self.params['show_head_threshold'] = self.param_controls["show_head_threshold"].isChecked()
            self.params['show_tail_skeleton'] = self.param_controls["show_tail_skeleton"].isChecked()

            # update the image preview
            self.update_preview(new_frame=True)

    def toggle_tracking(self, checkbox):
        if checkbox.isChecked():
            track = True
        else:
            track = False

        if checkbox.text() == "Track head":
            self.params['track_head'] = track
        elif checkbox.text() == "Track tail":
            self.params['track_tail'] = track

    def update_params_from_gui(self):
        if self.current_frame != None:
            # get crop params from gui
            for c in range(len(self.params['crop_params'])):
                crop_y = int(float(self.crop_param_controls[c]['crop_y'][1].text()))
                crop_x = int(float(self.crop_param_controls[c]['crop_x'][1].text()))
                offset_y = int(float(self.crop_param_controls[c]['offset_y'][1].text()))
                offset_x = int(float(self.crop_param_controls[c]['offset_x'][1].text()))

                self.params['crop_params'][c]['crop'] = [crop_y, crop_x]
                self.params['crop_params'][c]['offset'] = [offset_y, offset_x]

                self.params['crop_params'][c]['head_threshold'] = int(self.crop_param_controls[c]['head_threshold'].text())
                self.params['crop_params'][c]['tail_threshold'] = int(self.crop_param_controls[c]['tail_threshold'].text())

            self.current_crop_params = self.params['crop_params'][self.current_crop_num]

            # get global params from gui
            self.params['shrink_factor'] = float(self.param_controls['shrink_factor'][1].text())

            tail_crop_height = int(float(self.param_controls['tail_crop_height'][1].text()))
            tail_crop_width = int(float(self.param_controls['tail_crop_width'][1].text()))
            self.params['tail_crop'] = [tail_crop_height, tail_crop_width]

            self.params['min_tail_eye_dist'] = int(self.param_controls['min_tail_eye_dist'].text())
            self.params['max_tail_eye_dist'] = int(self.param_controls['max_tail_eye_dist'].text())
            self.params['new_video_fps'] = int(self.param_controls['new_video_fps'].text())
            self.params['n_tail_points'] = int(self.param_controls['n_tail_points'].text())

            self.params['eye_resize_factor'] = int(self.param_controls['eye_resize_factor'].currentText())
            self.params['interpolation'] = str(self.param_controls['interpolation'].currentText())

            # reshape current frame
            self.reshape_frame()

            # update the image preview
            self.update_preview(new_frame=True)

    def save_params(self):
        # get params from gui
        self.update_params_from_gui()

        # save params to file
        with open(self.params_file, "w") as output_file:
            json.dump(self.params, output_file)

    def track_frame(self, update_params=True):
        if update_params:
            # get params from gui
            self.update_params_from_gui()

        if self.params['interpolation'] == 'Nearest Neighbor':
            interpolation = cv2.INTER_NEAREST
        elif self.params['interpolation'] == 'Linear':
            interpolation = cv2.INTER_LINEAR
        elif self.params['interpolation'] == 'Bicubic':
            interpolation = cv2.INTER_CUBIC
        elif self.params['interpolation'] == 'Lanczos':
            interpolation = cv2.INTER_LANCZOS4

        # track frame
        (tail_coords, spline_coords,
         eye_coords, heading_coords,
         skeleton_matrix) = tt.track_image(self.cropped_frame, self.current_crop_params['head_threshold'], self.current_crop_params['tail_threshold'],
                                           self.params['min_tail_eye_dist'], self.params['max_tail_eye_dist'],
                                           self.params['track_head'], self.params['track_tail'],
                                           self.params['n_tail_points'], self.params['tail_crop'], self.params['adjust_thresholds'],
                                           self.params['eye_resize_factor'], interpolation)

        if not self.signalsBlocked():
            # plot tracked frame. for threshold/skeleton images, change nonzero pixels to 255 (so they show up as white)
            if self.param_controls["show_head_threshold"].isChecked():
                self.imageTracked.emit(self.head_threshold_frame*255, self.params['tail_crop'],
                    [tail_coords, spline_coords, eye_coords, heading_coords])
            elif self.param_controls["show_tail_threshold"].isChecked():
                self.imageTracked.emit(self.tail_threshold_frame*255, self.params['tail_crop'],
                    [tail_coords, spline_coords, eye_coords, heading_coords])
            elif self.param_controls["show_tail_skeleton"].isChecked():
                self.imageTracked.emit(self.tail_skeleton_frame*255, self.params['tail_crop'],
                    [tail_coords, spline_coords, eye_coords, heading_coords])
            else:
                self.imageTracked.emit(self.cropped_frame, self.params['tail_crop'],
                    [tail_coords, spline_coords, eye_coords, heading_coords])

    def track(self):
        if self.params['interpolation'] == 'Nearest Neighbor':
            interpolation = cv2.INTER_NEAREST
        elif self.params['interpolation'] == 'Linear':
            interpolation = cv2.INTER_LINEAR
        elif self.params['interpolation'] == 'Bicubic':
            interpolation = cv2.INTER_CUBIC
        elif self.params['interpolation'] == 'Lanczos':
            interpolation = cv2.INTER_LANCZOS4

        kwargs_dict = { 'crop_params':       self.params['crop_params'],
                        'shrink_factor':     self.params['shrink_factor'],
                        'invert':            self.params['invert'],
                        'min_tail_eye_dist': self.params['min_tail_eye_dist'],
                        'max_tail_eye_dist': self.params['max_tail_eye_dist'],
                        'track_head':        self.params['track_head'],
                        'track_tail':        self.params['track_tail'],
                        'save_video':        self.params['save_video'],
                        'new_video_fps':     self.params['new_video_fps'],
                        'tail_crop':         self.params['tail_crop'],
                        'n_tail_points':     self.params['n_tail_points'],
                        'adjust_thresholds': self.params['adjust_thresholds'],
                        'eye_resize_factor': self.params['eye_resize_factor'],
                        'interpolation':     interpolation
                      }

        if self.params['type_opened'] == "image":
            self.track_image(kwargs_dict)
        elif self.params['type_opened'] == "folder":
            self.track_folder(kwargs_dict)
        elif self.params['type_opened'] == "video":
            self.track_video(kwargs_dict)

    def track_image(self, kwargs_dict):
        # get save path
        self.save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save image', '', 'Images (*.jpg *.tif *.png)'))

        # spawn thread to track image
        t = threading.Thread(target=tt.open_and_track_image, args=(self.params['last_path'], self.save_path), kwargs=kwargs_dict)
        t.start()

    def track_folder(self, kwargs_dict):
        # get save path
        self.save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save video', '', 'Videos (*.mov *.tif *.mp4 *.avi)'))

        # spawn thread to track folder
        t = threading.Thread(target=tt.open_and_track_folder, args=(self.params['last_path'], self.save_path), kwargs=kwargs_dict)
        t.start()

    def track_video(self, kwargs_dict):
        # get save path
        self.save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save video', '', 'Videos (*.mov *.tif *.mp4 *.avi)'))

        # spawn thread to track video
        t = threading.Thread(target=tt.open_and_track_video, args=(self.params['last_path'], self.save_path), kwargs=kwargs_dict)
        t.start()

    def update_crop_from_selection(self, start_crop_coord, end_crop_coord):
        # get start & end coordinates - end_add adds a pixel to the end coordinates (for a more accurate crop)
        y_start = round(start_crop_coord[0]/self.params['shrink_factor'])
        y_end   = round(end_crop_coord[0]/self.params['shrink_factor'])
        x_start = round(start_crop_coord[1]/self.params['shrink_factor'])
        x_end   = round(end_crop_coord[1]/self.params['shrink_factor'])
        end_add = round(1*self.params['shrink_factor'])

        # update crop params
        self.current_crop_params['crop']   = [abs(y_end - y_start)+end_add, abs(x_end - x_start)+end_add]
        self.current_crop_params['offset'] = [self.current_crop_params['offset'][0] + min(y_start, y_end), self.current_crop_params['offset'][1] + min(x_start, x_end)]
        self.params['crop_params'][self.current_crop_num] = self.current_crop_params

        # update crop gui
        self.update_crop_param_gui()

        # reshape current frame
        self.reshape_frame()

        # update the image preview
        self.update_preview(new_frame=True)

    def reset_crop(self):
        # reset crop params
        self.current_crop_params['crop']   = [self.current_frame.shape[0], self.current_frame.shape[1]]
        self.current_crop_params['offset'] = [0, 0]
        self.params['crop_params'][self.current_crop_num] = self.current_crop_params

        # update crop gui
        self.update_crop_param_gui()

        # reshape current frame
        self.reshape_frame()

        # update the image preview
        self.update_preview(new_frame=True)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

if __name__ == "__main__":
    qApp = QtGui.QApplication(sys.argv)

    # create & show param window
    param_window = ParamWindow()
    param_window.show()

    sys.exit(qApp.exec_())
