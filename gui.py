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

gray_color_table = [QtGui.qRgb(i, i, i) for i in range(256)]

default_params = {'shrink_factor': 1.0,                 # factor by which to shrink the original frame
                  'offset': None,                       # crop offset
                  'crop': None,                         # crop size
                  'tail_threshold': 200,                # pixel brightness to use for thresholding to find the tail (0-255)
                  'head_threshold': 50,                 # pixel brightness to use for thresholding to find the eyes (0-255)
                  'invert': False,                      # invert the frame
                  'type_opened': None,                  # type of media that is opened - "video", "folder", "image" or None
                  'min_eye_distance': 10,               # min. distance between the eyes and the tail
                  'max_eye_distance':20,
                  'track_head': True,                   # track head/eyes
                  'track_tail': True,                   # track tail
                  'show_head_threshold': False,         # show head threshold in preview window
                  'show_tail_threshold': False,         # show tail threshold in preview window
                  'last_path': "",                      # path to last opened image/folder/video
                  'save_video': True,                   # whether to make a video with tracking overlaid
                  'new_video_fps': 30,                  # fps for the generated video
                  'closest_eye_y_coords': [None, None], # closest y coordinates of eyes
                  'closest_eye_x_coords': [None, None], # closest x coordinates of eyes
                  'fish_crop_dims': [100, 100]          # dimensions of crop around zebrafish eyes to use for tail tracking - (y, x)
                 }

max_n_frames = 100 # maximum # of frames to load for previewing

class PlotQLabel(QtGui.QLabel):
    '''
    QLabel subclass used to show a preview image.
    '''
    def __init__(self):
        QtGui.QLabel.__init__(self)

        self.scale_factor = None

    def mousePressEvent(self, event):
        if self.scale_factor:
            self.y_start = int(event.y()/self.scale_factor)
            self.x_start = int(event.x()/self.scale_factor)

    def mouseMoveEvent(self, event):
        if self.scale_factor:
            self.y_end = int(event.y()/self.scale_factor)
            self.x_end = int(event.x()/self.scale_factor)

            self.preview_window.draw_crop_selection(self.y_start, self.y_end, self.x_start, self.x_end)

    def mouseReleaseEvent(self, event):
        if self.scale_factor:
            self.y_end = int(event.y()/self.scale_factor)
            self.x_end = int(event.x()/self.scale_factor)

            if self.y_end != self.y_start and self.x_end != self.x_start:
                self.preview_window.crop_selection(self.y_start, self.y_end, self.x_start, self.x_end)
            elif self.y_end == self.y_start and self.x_end == self.x_start:
                if self.preview_window.selecting_eyes == True:
                    self.preview_window.set_eye_coord(self.y_end, self.x_end)

    def set_plot_window(self, plot_window):
        self.preview_window = plot_window

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

class PreviewWindow(QtGui.QMainWindow):
    def __init__(self, param_window):
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Preview")

        self.param_window = param_window

        self.param_window.imageLoaded.connect(self.plot_image)
        self.param_window.imageTracked.connect(self.plot_tracked_image)
        self.param_window.thresholdLoaded.connect(self.plot_threshold_image)
        self.param_window.thresholdUnloaded.connect(self.remove_threshold_image)

        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setStyleSheet("background-color:#666666;")

        self.l = QtGui.QVBoxLayout(self.main_widget)
        self.image_label = PlotQLabel()
        self.image_label.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.image_label.setAcceptDrops(True)
        self.image_label.set_plot_window(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.l.setAlignment(QtCore.Qt.AlignCenter)
        self.l.addWidget(self.image_label)

        self.instructions_label = QtGui.QLabel("")
        self.instructions_label.setAlignment(QtCore.Qt.AlignCenter)
        self.l.addWidget(self.instructions_label)

        self.image_slider = None

        self.orig_image    = None
        self.tracking_list = None
        self.orig_pix      = None
        self.pix_size      = None
        self.rgb_image     = None

        self.selecting_eyes = False
        self.n_eyes_selected = 0
        self.selecting_crop = False

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.setWindowFlags(QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowMinimizeButtonHint)

    def resizeEvent(self, event):
        QtGui.QMainWindow.resizeEvent(self, event)

        width = self.main_widget.width() - 40
        height = self.main_widget.height() - 40

        self.pix_size = min(max(min(width, height), 400), 900)

        if self.orig_pix:
            scale_factor = float(self.pix_size)/max(self.orig_pix.width(), self.orig_pix.height())
            self.image_label.set_scale_factor(float(self.pix_size)/max(self.orig_pix.width(), self.orig_pix.height()))
            pix = self.orig_pix.scaled(self.pix_size, self.pix_size, QtCore.Qt.KeepAspectRatio)
            self.image_label.setPixmap(pix)
            self.image_label.setFixedSize(pix.size())

    def start_select_eyes(self):
        self.selecting_eyes = True
        self.instructions_label.setText("Click to select eye locations.")

    def set_eye_coord(self, y, x):
        if self.n_eyes_selected == 0:
            self.rgb_image = None

        self.n_eyes_selected += 1
        self.param_window.set_eye_coord_from_selection(y, x, self.n_eyes_selected)

        if self.n_eyes_selected == 2:
            self.n_eyes_selected = 0
            self.end_select_eyes()

        if self.rgb_image == None:
            image = np.copy(self.orig_image)

            if len(image.shape) < 3:
                print(image.shape)
                self.rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            else:
                self.rgb_image = image

            cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB, self.rgb_image)

        cv2.circle(self.rgb_image, (x, y), 2, (255, 255, 0), -1)

        height, width, bytesPerComponent = self.rgb_image.shape
        bytesPerLine = bytesPerComponent * width

        qimage = QtGui.QImage(self.rgb_image.data, self.rgb_image.shape[1], self.rgb_image.shape[0], bytesPerLine, QtGui.QImage.Format_RGB888)
        qimage.setColorTable(gray_color_table)

        self.orig_pix = QtGui.QPixmap(qimage)

        pix = self.orig_pix.scaled(self.pix_size, self.pix_size, QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)
        self.image_label.setFixedSize(pix.size())

    def end_select_eyes(self):
        self.selecting_eyes = False
        self.instructions_label.setText("")

    def start_select_crop(self):
        self.selecting_crop = True
        self.instructions_label.setText("Click & drag to select crop area.")

    def plot_threshold_image(self, threshold_image, fish_crop, new_image=False):
        new_threshold_image = threshold_image * 255

        self.orig_image = new_threshold_image

        if new_image:
            self.tracking_list = None

        if self.tracking_list:
            self.plot_tracked_image(new_threshold_image, fish_crop, self.tracking_list)
        else:
            rgb_image = np.repeat(new_threshold_image[:, :, np.newaxis], 3, axis=2)

            height, width, bytesPerComponent = rgb_image.shape
            bytesPerLine = bytesPerComponent * width
            cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB, rgb_image)

            if fish_crop[0] != None:
                overlay = rgb_image.copy()

                cv2.rectangle(overlay, (int(fish_crop[2]), int(fish_crop[0])), (int(fish_crop[3]), int(fish_crop[1])), (195, 144, 212), -1)

                cv2.addWeighted(overlay, 0.5, rgb_image, 0.5, 0, rgb_image)

            qimage = QtGui.QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], bytesPerLine, QtGui.QImage.Format_RGB888)
            qimage.setColorTable(gray_color_table)

            self.orig_pix = QtGui.QPixmap(qimage)

            self.image_label.set_scale_factor(float(self.pix_size)/max(self.orig_pix.width(), self.orig_pix.height()))

            pix = self.orig_pix.scaled(self.pix_size, self.pix_size, QtCore.Qt.KeepAspectRatio)
            self.image_label.setPixmap(pix)
            self.image_label.setFixedSize(pix.size())

    def remove_threshold_image(self, image):
        self.orig_image = np.copy(image)
        self.plot_image(self.orig_image, self.param_window.fish_crop)

    def plot_image(self, image, fish_crop, new_image=False):
        # if not self.param_window.image_opened:
        if self.image_slider:
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

            self.l.addWidget(self.image_slider)

        if new_image:
            self.tracking_list = None

        self.orig_image = np.copy(image)

        if self.tracking_list:
            self.plot_tracked_image(image, self.param_window.fish_crop, self.tracking_list)
        else:
            rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

            height, width, bytesPerComponent = rgb_image.shape
            bytesPerLine = bytesPerComponent * width
            cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB, rgb_image)

            # print(fish_crop)
            if fish_crop[0] != None:
                overlay = rgb_image.copy()

                cv2.rectangle(overlay, (int(fish_crop[2]), int(fish_crop[0])), (int(fish_crop[3]), int(fish_crop[1])), (195, 144, 212), -1)

                cv2.addWeighted(overlay, 0.5, rgb_image, 0.5, 0, rgb_image)

            qimage = QtGui.QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], bytesPerLine, QtGui.QImage.Format_RGB888)
            qimage.setColorTable(gray_color_table)

            self.orig_pix = QtGui.QPixmap(qimage)

            self.image_label.set_scale_factor(float(self.pix_size)/max(self.orig_pix.width(), self.orig_pix.height()))

            pix = self.orig_pix.scaled(self.pix_size, self.pix_size, QtCore.Qt.KeepAspectRatio)
            self.image_label.setPixmap(pix)
            self.image_label.setFixedSize(pix.size())

    def switch_frame(self, value):
        self.param_window.switch_frame(value)

    def plot_tracked_image(self, image, fish_crop, tracking_list):
        self.tracking_list = tracking_list

        tail_y_coords   = self.tracking_list[0]
        tail_x_coords   = self.tracking_list[1]
        spline_y_coords = self.tracking_list[2]
        spline_x_coords = self.tracking_list[3]
        eye_y_coords    = self.tracking_list[4]
        eye_x_coords    = self.tracking_list[5]
        perp_y_coords   = self.tracking_list[6]
        perp_x_coords   = self.tracking_list[7]

        image = tt.plot_image(image, tail_y_coords, tail_x_coords, spline_y_coords, spline_x_coords,
                                eye_y_coords, eye_x_coords, perp_y_coords, perp_x_coords)

        self.orig_image = np.copy(image)

        # print(fish_crop)


        height, width, bytesPerComponent = image.shape
        bytesPerLine = bytesPerComponent * width
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)

        if fish_crop[0] != None:
            overlay = image.copy()

            cv2.rectangle(overlay, (int(fish_crop[2]), int(fish_crop[0])), (int(fish_crop[3]), int(fish_crop[1])), (195, 144, 212), -1)

            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], bytesPerLine, QtGui.QImage.Format_RGB888)
        qimage.setColorTable(gray_color_table)

        self.orig_pix = QtGui.QPixmap(qimage)

        self.image_label.set_scale_factor(float(self.pix_size)/max(self.orig_pix.width(), self.orig_pix.height()))

        pix = self.orig_pix.scaled(self.pix_size, self.pix_size, QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)
        self.image_label.setFixedSize(pix.size())

    def draw_crop_selection(self, y_start, y_end, x_start, x_end):
        if self.selecting_crop:
            image = np.copy(self.orig_image)

            if len(image.shape) < 3:
                rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            else:
                rgb_image = image

            overlay = rgb_image.copy()

            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (0, 51, 255), -1)

            cv2.addWeighted(overlay, 0.5, rgb_image, 0.5, 0, rgb_image)

            height, width, bytesPerComponent = rgb_image.shape
            bytesPerLine = bytesPerComponent * width
            cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB, rgb_image)

            qimage = QtGui.QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], bytesPerLine, QtGui.QImage.Format_RGB888)
            qimage.setColorTable(gray_color_table)

            self.orig_pix = QtGui.QPixmap(qimage)

            pix = self.orig_pix.scaled(self.pix_size, self.pix_size, QtCore.Qt.KeepAspectRatio)
            self.image_label.setPixmap(pix)
            self.image_label.setFixedSize(pix.size())

    def crop_selection(self, y_start, y_end, x_start, x_end):
        if self.selecting_crop:
            self.selecting_crop = False
            self.instructions_label.setText("")

            self.param_window.update_crop_from_selection(y_start, y_end, x_start, x_end)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

class ParamWindow(QtGui.QMainWindow):
    # initiate drawing signals
    imageLoaded       = QtCore.pyqtSignal(np.ndarray, list, bool)
    imageTracked      = QtCore.pyqtSignal(np.ndarray, list, list)
    thresholdLoaded   = QtCore.pyqtSignal(np.ndarray, list, bool)
    thresholdUnloaded = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        # create preview window
        self.preview_window = PreviewWindow(self)
        self.preview_window.setWindowTitle("Preview")
        self.preview_window.show()

        # initiate image vars
        self.current_frame = None
        self.cropped_frame = None
        self.shrunken_frame = None
        self.head_threshold_frame = None
        self.tail_threshold_frame = None
        self.n_frames = 0

        # coords of crop around the eyes of the fish to use for finding the tail
        self.fish_crop = [None, None, None, None]

        # set experiments file
        self.params_file = "last_params.json"

        # set window size
        self.setGeometry(100, 200, 200, 580)

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

        updateParams = QtGui.QAction(QtGui.QIcon('save.png'), 'Update Parameters', self)
        updateParams.setShortcuts(['Enter'])
        updateParams.setStatusTip('Update parameters')
        updateParams.triggered.connect(self.update_params_from_gui)

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

        # create widget & layout
        self.mainWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.mainWidget)

        self.layout = QtGui.QVBoxLayout()
        self.form_layout = QtGui.QFormLayout()

        # set default params
        self.params = default_params

        self.param_controls = {}

        # add checkboxes - (key, description, function to call, initial value)
        self.add_checkbox('invert', "Invert image", self.toggle_invert_image, self.params['invert'])
        self.add_checkbox('show_head_threshold', "Show head threshold", self.toggle_threshold_image, self.params['show_head_threshold'])
        self.add_checkbox('show_tail_threshold', "Show tail threshold", self.toggle_threshold_image, self.params['show_tail_threshold'])
        self.add_checkbox('track_head', "Track head", self.toggle_tracking, self.params['track_head'])
        self.add_checkbox('track_tail', "Track tail", self.toggle_tracking, self.params['track_tail'])

        # add sliders - (key, description, start, end, initial value)
        self.add_slider('crop_y', 'Crop y:', 1, 500, 500)
        self.add_slider('crop_x', 'Crop x:', 1, 500, 500)

        self.add_slider('offset_y', 'Offset y:', 0, 499, 0)
        self.add_slider('offset_x', 'Offset x:', 0, 499, 0)

        self.add_slider('shrink_factor', 'Shrink factor:', 1, 10, round(10*self.params['shrink_factor']))
        self.add_slider('fish_crop_height', 'Fish crop height:', 1, 100, round(self.params['fish_crop_dims'][0]))
        self.add_slider('fish_crop_width', 'Fish crop width:', 1, 100, round(self.params['fish_crop_dims'][1]))

        # add textboxes - (key, decription, initial value)
        self.add_textbox('min_eye_distance', 'Head/tail start min. dist.:', self.params['min_eye_distance'])
        self.add_textbox('max_eye_distance', 'Head/tail start max. dist.:', self.params['max_eye_distance'])
        self.add_textbox('head_threshold', 'Head threshold:', self.params['head_threshold'])
        self.add_textbox('tail_threshold', 'Tail threshold:', self.params['tail_threshold'])
        self.add_checkbox('save_video', "Save video", self.toggle_save_video, self.params['save_video'])
        self.add_textbox('new_video_fps', 'Saved Video FPS:', self.params['new_video_fps'])

        # add eye 1 closest coords label
        if self.params['closest_eye_y_coords'] == None or self.params['closest_eye_y_coords'][0] == None:
            closest_eye_1_coords_text = "None"
        else:
            closest_eye_1_coords_text = str((self.params['closest_eye_y_coords'][0], self.params['closest_eye_x_coords'][0]))
        closest_eye_1_coords_label = QtGui.QLabel(closest_eye_1_coords_text)

        self.form_layout.addRow('Closest eye 1 coords:', closest_eye_1_coords_label)
        self.param_controls['closest_eye_1_coords'] = closest_eye_1_coords_label

        # add eye 2 closest coords label
        if self.params['closest_eye_y_coords'] == None or self.params['closest_eye_y_coords'][1] == None:
            closest_eye_2_coords_text = "None"
        else:
            closest_eye_2_coords_text = str((self.params['closest_eye_y_coords'][1], self.params['closest_eye_x_coords'][1]))
        closest_eye_2_coords_label = QtGui.QLabel(closest_eye_2_coords_text)

        self.form_layout.addRow('Closest eye 2 coords:', closest_eye_2_coords_label)
        self.param_controls['closest_eye_2_coords'] = closest_eye_2_coords_label

        self.layout.addLayout(self.form_layout)

        # add button layouts
        hbox1 = QtGui.QHBoxLayout()
        hbox1.setSpacing(5)
        hbox1.addStretch(1)
        hbox2 = QtGui.QHBoxLayout()
        hbox2.setSpacing(5)
        hbox2.addStretch(1)
        hbox3 = QtGui.QHBoxLayout()
        hbox3.setSpacing(5)
        hbox3.setMargin(0)
        hbox3.addStretch(1)

        # add buttons
        self.reload_last_save_button = QtGui.QPushButton('Reload', self)
        self.reload_last_save_button.setMinimumHeight(10)
        self.reload_last_save_button.setMaximumWidth(70)
        self.reload_last_save_button.clicked.connect(lambda:self.reload_last_save())
        hbox1.addWidget(self.reload_last_save_button)

        self.open_image_button = QtGui.QPushButton('+ Image', self)
        self.open_image_button.setMinimumHeight(10)
        self.open_image_button.setMaximumWidth(70)
        self.open_image_button.clicked.connect(lambda:self.open_image(""))
        hbox1.addWidget(self.open_image_button)

        self.open_folder_button = QtGui.QPushButton('+ Folder', self)
        self.open_folder_button.setMinimumHeight(10)
        self.open_folder_button.setMaximumWidth(70)
        self.open_folder_button.clicked.connect(lambda:self.open_folder(""))
        hbox1.addWidget(self.open_folder_button)

        self.open_video_button = QtGui.QPushButton('+ Video', self)
        self.open_video_button.setMinimumHeight(10)
        self.open_video_button.setMaximumWidth(70)
        self.open_video_button.clicked.connect(lambda:self.open_video(""))
        hbox1.addWidget(self.open_video_button)

        self.save_button = QtGui.QPushButton('Save', self)
        self.save_button.setMinimumHeight(10)
        self.save_button.setMaximumWidth(80)
        self.save_button.clicked.connect(self.save_params)
        hbox2.addWidget(self.save_button)

        self.track_button = QtGui.QPushButton('Track', self)
        self.track_button.setMinimumHeight(10)
        self.track_button.setMaximumWidth(80)
        self.track_button.clicked.connect(self.track_frame)
        hbox2.addWidget(self.track_button)

        self.reset_crop_button = QtGui.QPushButton('Reset Crop', self)
        self.reset_crop_button.setMinimumHeight(10)
        self.reset_crop_button.setMaximumWidth(100)
        self.reset_crop_button.clicked.connect(self.reset_crop)
        hbox2.addWidget(self.reset_crop_button)

        self.select_crop_button = QtGui.QPushButton('Crop', self)
        self.select_crop_button.setMinimumHeight(10)
        self.select_crop_button.setMaximumWidth(60)
        self.select_crop_button.clicked.connect(self.select_crop)
        hbox3.addWidget(self.select_crop_button)

        self.select_eyes_button = QtGui.QPushButton('Select Eyes', self)
        self.select_eyes_button.setMinimumHeight(10)
        self.select_eyes_button.setMaximumWidth(120)
        self.select_eyes_button.clicked.connect(self.select_eyes)
        hbox3.addWidget(self.select_eyes_button)

        self.track_button = QtGui.QPushButton('Track && Save', self)
        self.track_button.setMinimumHeight(10)
        self.track_button.setMaximumWidth(180)
        self.track_button.clicked.connect(self.track)
        self.track_button.setStyleSheet("font-weight: bold")
        hbox3.addWidget(self.track_button)

        self.layout.addLayout(hbox1)
        self.layout.addLayout(hbox2)
        self.layout.addLayout(hbox3)
        self.layout.setSpacing(5)

        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.mainWidget.setLayout(self.layout)

        self.setWindowTitle("Parameters")

        self.setWindowFlags(QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)

        self.show()

    def select_crop(self):
        # user wants to draw a crop selection
        self.preview_window.start_select_crop()

    def select_eyes(self):
        # user wants to select eyes
        self.preview_window.start_select_eyes()

    def hideEvent(self, event):
        self.preview_window.closeEvent(event)

    def reload_last_save(self):
        # re-load last saved state
        self.load_params()
        self.open_last_file()
        print(self.params)
        self.update_gui_with_new_params()
        print(self.params)

    def load_params(self):
        try:
            # load params from saved file
            with open(self.params_file, "r") as input_file:
                self.params = json.load(input_file)

            if sorted(self.params.keys()) != sorted(default_params.keys()):
                raise
        except:
            # set params to defaults
            self.params = default_params

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

    def add_textbox(self, label, description, default_value):
        param_box = QtGui.QLineEdit(self)
        param_box.setMinimumHeight(10)
        self.form_layout.addRow(description, param_box)
        if default_value != None:
            param_box.setText(str(default_value))

        self.param_controls[label] = param_box

    def add_slider(self, label, description, minimum, maximum, value, tick_interval=2, single_step=0.5):
        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        slider.setTickPosition(QtGui.QSlider.TicksBothSides)
        slider.setTickInterval(tick_interval)
        slider.setSingleStep(single_step)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.setMinimumWidth(130)

        slider.valueChanged.connect(self.update_params_from_gui)
        self.form_layout.addRow(description, slider)
        self.param_controls[label] = slider

    def add_checkbox(self, label, description, toggle_func, checked):
        checkbox = QtGui.QCheckBox(description)
        checkbox.setChecked(checked)

        checkbox.toggled.connect(lambda:toggle_func(checkbox))
        self.layout.addWidget(checkbox)
        self.param_controls[label] = checkbox

    def open_folder(self, path="", reset_params=True):
        if path == "":
            # ask the user to select a directory
            self.params['last_path'] = str(QtGui.QFileDialog.getExistingDirectory(self, 'Open folder'))
        else:
            self.params['last_path'] = path

        print(self.params['last_path'])

        if self.params['last_path'] not in ("", None):
            # get paths to all the images & the number of frames
            self.image_paths = []
            self.n_frames = 0

            for filename in sorted(os.listdir(self.params['last_path'])):
                if filename.endswith('.tif') or filename.endswith('.png'):
                    image_path = self.params['last_path'] + "/" + filename
                    self.image_paths.append(image_path)
                    self.n_frames += 1

            if self.n_frames == 0:
                print("Could not find any images.")
                return
            elif self.n_frames >= max_n_frames:
                # get evenly spaced frame numbers (so we don't load all the frames)
                f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
                self.image_paths = [ self.image_paths[i] for i in f(max_n_frames, self.n_frames)]
                self.n_frames = max_n_frames

            print("Opened folder.")

            self.params['type_opened'] = 'folder'

            if reset_params:
                self.params = default_params
                self.update_gui_with_new_params()

            # switch to first frame
            self.switch_frame(0)

    def open_image(self, path="", reset_params=True):
        if path == "":
            # ask the user to select an image
            self.params['last_path'] = str(QtGui.QFileDialog.getOpenFileName(self, 'Open image', '', 'Images (*.jpg  *.jpeg *.tif *.tiff *.png)'))
        else:
            self.params['last_path'] = path

        if self.params['last_path'] not in ("", None):
            self.params['type_opened'] = 'image'

            self.n_frames = 1

            if reset_params:
                self.params = default_params
                self.update_gui_with_new_params()

            # switch to first (and only) frame
            self.switch_frame(0)

    def open_video(self, path="", reset_params=True):
        if path == "":
            # ask the user to select a video
            self.params['last_path'] = str(QtGui.QFileDialog.getOpenFileName(self, 'Open video', '', 'Videos (*.mov *.tif *.mp4 *.avi)'))
        else:
            self.params['last_path'] = path

        if self.params['last_path'] not in ("", None):
            self.frames = tt.load_video(self.params['last_path'], n_frames=max_n_frames)

            if self.frames == None:
                print("Could not load frames.")
                return

            self.params['type_opened'] = 'video'

            self.n_frames = len(self.frames)

            if reset_params == True:
                print("Resetting params")
                self.params = default_params
                self.update_gui_with_new_params()

            # switch to first frame
            self.switch_frame(0)

    def update_gui_with_new_params(self):
        self.param_controls['crop_y'].blockSignals(True)
        self.param_controls['crop_x'].blockSignals(True)
        self.param_controls['offset_y'].blockSignals(True)
        self.param_controls['offset_x'].blockSignals(True)
        self.param_controls['shrink_factor'].blockSignals(True)
        self.param_controls['fish_crop_height'].blockSignals(True)
        self.param_controls['fish_crop_width'].blockSignals(True)

        if self.params['crop'] != None:
            # update sliders
            self.param_controls['crop_y'].setValue(round(500.0*self.params['crop'][0]/self.current_frame.shape[0]))
            self.param_controls['crop_x'].setValue(round(500.0*self.params['crop'][1]/self.current_frame.shape[1]))

            self.param_controls['offset_y'].setValue(round(500.0*self.params['offset'][0]/self.current_frame.shape[0]))
            self.param_controls['offset_x'].setValue(round(500.0*self.params['offset'][1]/self.current_frame.shape[1]))
        else:
            # set sliders to default
            self.param_controls['crop_y'].setValue(500)
            self.param_controls['crop_x'].setValue(500)

            self.param_controls['offset_y'].setValue(0)
            self.param_controls['offset_x'].setValue(0)

        self.param_controls['shrink_factor'].setValue(round(10*self.params['shrink_factor']))
        self.param_controls['fish_crop_height'].setValue(round(self.params['fish_crop_dims'][0]))
        self.param_controls['fish_crop_width'].setValue(round(self.params['fish_crop_dims'][1]))

        self.param_controls['crop_y'].blockSignals(False)
        self.param_controls['crop_x'].blockSignals(False)
        self.param_controls['offset_y'].blockSignals(False)
        self.param_controls['offset_x'].blockSignals(False)
        self.param_controls['shrink_factor'].blockSignals(False)
        self.param_controls['fish_crop_height'].blockSignals(False)
        self.param_controls['fish_crop_width'].blockSignals(False)

        # update param controls
        self.param_controls['invert'].setChecked(self.params['invert'])
        self.param_controls['show_head_threshold'].setChecked(self.params['show_head_threshold'])
        self.param_controls['show_tail_threshold'].setChecked(self.params['show_tail_threshold'])
        self.param_controls['track_head'].setChecked(self.params['track_head'])
        self.param_controls['track_tail'].setChecked(self.params['track_tail'])
        self.param_controls['save_video'].setChecked(self.params['save_video'])
        self.param_controls['min_eye_distance'].setText(str(self.params['min_eye_distance']))
        self.param_controls['max_eye_distance'].setText(str(self.params['max_eye_distance']))
        self.param_controls['tail_threshold'].setText(str(self.params['tail_threshold']))
        self.param_controls['head_threshold'].setText(str(self.params['head_threshold']))
        self.param_controls['new_video_fps'].setText(str(self.params['new_video_fps']))

        print("hi", self.params)

        # update selected eye coordinates
        if self.params['closest_eye_y_coords'] == None or self.params['closest_eye_y_coords'][0] == None:
            closest_eye_1_coords_text = "None"
        else:
            closest_eye_1_coords_text = str((int(self.params['closest_eye_y_coords'][0]), int(self.params['closest_eye_x_coords'][0])))
        self.param_controls['closest_eye_1_coords'].setText(closest_eye_1_coords_text)

        if self.params['closest_eye_y_coords'] == None or self.params['closest_eye_y_coords'][1] == None:
            closest_eye_2_coords_text = "None"
        else:
            closest_eye_2_coords_text = str((int(self.params['closest_eye_y_coords'][1]), int(self.params['closest_eye_x_coords'][1])))
        self.param_controls['closest_eye_2_coords'].setText(closest_eye_2_coords_text)

    def switch_frame(self, n):
        if self.params['type_opened'] == 'video':
            self.current_frame = self.frames[n]
        elif self.params['type_opened'] == 'folder':
            self.current_frame = tt.load_image(self.image_paths[n])
        elif self.params['type_opened'] == 'image':
            self.current_frame = tt.load_image(self.params['last_path'])

        # stop selecting eyes
        if self.preview_window.selecting_eyes:
            self.preview_window.end_select_eyes()

        if self.params['closest_eye_y_coords'][1] == None:
            # only one eye has been selected, revert
            self.params['closest_eye_y_coords'] = [None, None]
            self.params['closest_eye_x_coords'] = [None, None]

        # self.update_params_from_gui()

        if self.param_controls['invert'].isChecked():
            # invert the image
            self.invert_frame()

        # reshape the image
        self.reshape_frame()

        # update the image preview
        self.update_preview(new_frame=True)

    def reshape_frame(self):
        # shrink the image
        if self.current_frame != None:
            if self.params['shrink_factor'] != None:
                self.shrunken_frame = tt.shrink_image(self.current_frame, self.params['shrink_factor'])

            # crop the image
            if self.params['crop'] is not None and self.params['offset'] is not None:
                crop = (round(self.params['crop'][0]*self.params['shrink_factor']), round(self.params['crop'][1]*self.params['shrink_factor']))
                offset = (round(self.params['offset'][0]*self.params['shrink_factor']), round(self.params['offset'][1]*self.params['shrink_factor']))

                self.cropped_frame = tt.crop_image(self.shrunken_frame, offset, crop)

            self.generate_threshold_frames()

    def generate_threshold_frames(self):
        if self.current_frame != None:
            self.head_threshold_frame = tt.get_head_threshold_image(self.cropped_frame, self.params['head_threshold'])
            self.tail_threshold_frame = tt.get_tail_threshold_image(self.cropped_frame, self.params['tail_threshold'])

    def update_preview(self, new_frame=False):
        if self.current_frame != None:
            if self.param_controls["show_head_threshold"].isChecked():
                self.thresholdLoaded.emit(self.head_threshold_frame, self.fish_crop, new_frame)
            elif self.param_controls["show_tail_threshold"].isChecked():
                self.thresholdLoaded.emit(self.tail_threshold_frame, self.fish_crop, new_frame)
            else:
                self.imageLoaded.emit(self.cropped_frame, self.fish_crop, new_frame)

    def invert_frame(self):
        if self.current_frame != None:
            self.current_frame  = (255 - self.current_frame)
            self.shrunken_frame = (255 - self.shrunken_frame)
            self.cropped_frame  = (255 - self.cropped_frame)

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

    def toggle_threshold_image(self, checkbox):
        if self.current_frame != None:
            if checkbox.isChecked():
                if checkbox.text() == "Show head threshold":
                    self.param_controls["show_tail_threshold"].setChecked(False)
                    self.thresholdLoaded.emit(self.head_threshold_frame, self.fish_crop, False)
                elif checkbox.text() == "Show tail threshold":
                    self.param_controls["show_head_threshold"].setChecked(False)
                    self.thresholdLoaded.emit(self.tail_threshold_frame, self.fish_crop, False)
            else:
                self.thresholdUnloaded.emit(self.cropped_frame)

            self.params['show_tail_threshold'] = self.param_controls["show_tail_threshold"].isChecked()
            self.params['show_head_threshold'] = self.param_controls["show_head_threshold"].isChecked()

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
        # get crop settings
        if self.current_frame != None:
            crop_y = self.param_controls['crop_y'].value()*self.current_frame.shape[0]/500
            crop_x = self.param_controls['crop_x'].value()*self.current_frame.shape[1]/500
            offset_y = self.param_controls['offset_y'].value()*self.current_frame.shape[0]/500
            offset_x = self.param_controls['offset_x'].value()*self.current_frame.shape[1]/500

            new_crop = (int(crop_y), int(crop_x))
            new_offset = (int(offset_y), int(offset_x))

        fish_crop_height = self.param_controls['fish_crop_height'].value()
        fish_crop_width = self.param_controls['fish_crop_width'].value()
        new_fish_crop_dims = (int(fish_crop_height), int(fish_crop_width))

        self.params['min_eye_distance'] = int(self.param_controls['min_eye_distance'].text())
        self.params['max_eye_distance'] = int(self.param_controls['max_eye_distance'].text())
        self.params['new_video_fps'] = int(self.param_controls['new_video_fps'].text())

        new_head_threshold = int(self.param_controls['head_threshold'].text())
        new_tail_threshold = int(self.param_controls['tail_threshold'].text())
        new_shrink_factor = float(self.param_controls['shrink_factor'].value())/10.0

        generate_new_frame = False

        if self.current_frame != None:
            if self.params['crop'] != new_crop:
                self.params['crop'] = new_crop
                generate_new_frame = True
            if self.params['offset'] != new_offset:
                self.params['offset'] = new_offset
                generate_new_frame = True
        if self.params['shrink_factor'] != new_shrink_factor:
            self.params['shrink_factor'] = new_shrink_factor
            generate_new_frame = True
        if self.params['head_threshold'] != new_head_threshold:
            self.params['head_threshold'] = new_head_threshold
            generate_new_frame = True
        if self.params['tail_threshold'] != new_tail_threshold:
            self.params['tail_threshold'] = new_tail_threshold
            generate_new_frame = True
        if self.params['fish_crop_dims'] != new_fish_crop_dims:
            self.params['fish_crop_dims'] = new_fish_crop_dims
            generate_new_frame = True

        if generate_new_frame:
            self.reshape_frame()

            # update the image preview
            self.update_preview(new_frame=True)

    def save_params(self):
        self.update_params_from_gui()

        # save params to file
        with open(self.params_file, "w") as output_file:
            json.dump(self.params, output_file)

    def track_frame(self):
        self.update_params_from_gui()

        if self.params['track_head']:
            start_time = time.clock()
            (eye_y_coords, eye_x_coords,
            perp_y_coords, perp_x_coords) = tt.track_head(self.head_threshold_frame,
                                                            0, 1,
                                                            self.params['closest_eye_y_coords'], self.params['closest_eye_x_coords'])

            end_time = time.clock()

            print("Head time: {}".format(end_time - start_time))

            if eye_x_coords == None:
                print("Could not track head.")
            else:
                self.params['closest_eye_y_coords'] = eye_y_coords
                self.params['closest_eye_x_coords'] = eye_x_coords
        else:
            (eye_y_coords, eye_x_coords,
            perp_y_coords, perp_x_coords) = [None]*4

        if self.params['track_tail']:
            if self.params['closest_eye_y_coords'] != None and self.params['closest_eye_y_coords'][0] != None:
                self.pos_y_coord = (self.params['closest_eye_y_coords'][0] + self.params['closest_eye_y_coords'][1])/2.0
                self.pos_x_coord = (self.params['closest_eye_x_coords'][0] + self.params['closest_eye_x_coords'][1])/2.0

                if self.params['fish_crop_dims'][0] == None:
                    self.fish_crop = [0, self.cropped_image.shape[0], 0, self.cropped_image.shape[1]]
                else:
                    self.fish_crop = [np.maximum(0, self.pos_y_coord-self.params['fish_crop_dims'][0]), np.minimum(self.cropped_frame.shape[0], self.pos_y_coord+self.params['fish_crop_dims'][0]), np.maximum(0, self.pos_x_coord-self.params['fish_crop_dims'][1]), np.minimum(self.cropped_frame.shape[1], self.pos_x_coord+self.params['fish_crop_dims'][1])]
                
                if eye_y_coords != None:
                    eye_y_coords = [eye_y_coords[0]-self.fish_crop[0], eye_y_coords[1]-self.fish_crop[0]]
                    eye_x_coords = [eye_x_coords[0]-self.fish_crop[2], eye_x_coords[1]-self.fish_crop[2]]

                tail_threshold_image = self.tail_threshold_frame[self.fish_crop[0]:self.fish_crop[1], self.fish_crop[2]:self.fish_crop[3]]
            else:
                tail_threshold_image = self.tail_threshold_frame
            start_time = time.clock()
            (tail_y_coords, tail_x_coords,
            spline_y_coords, spline_x_coords, skeleton_matrix) = tt.track_tail(tail_threshold_image,
                                                                eye_x_coords, eye_y_coords,
                                                                min_eye_distance=self.params['min_eye_distance']*self.params['shrink_factor'], max_eye_distance=self.params['max_eye_distance']*self.params['shrink_factor'])

            end_time = time.clock()

            print("Tail time: {}".format(end_time - start_time))

            if tail_x_coords == None:
                print("Could not track tail.")
        else:
            (tail_y_coords, tail_x_coords,
            spline_y_coords, spline_x_coords) = [None]*4

        if self.params['closest_eye_y_coords'] != None and self.params['closest_eye_y_coords'][0] != None:
            if eye_y_coords != None:
                eye_y_coords = [eye_y_coords[0]+self.fish_crop[0], eye_y_coords[1]+self.fish_crop[0]]
                eye_x_coords = [eye_x_coords[0]+self.fish_crop[2], eye_x_coords[1]+self.fish_crop[2]]

            if tail_y_coords != None:
                for i in range(len(tail_y_coords)):
                    tail_y_coords[i] += self.fish_crop[0]
                    tail_x_coords[i] += self.fish_crop[2]
                for i in range(len(spline_y_coords)):
                    spline_y_coords[i] += self.fish_crop[0]
                    spline_x_coords[i] += self.fish_crop[2]

        if not self.signalsBlocked():
            if self.param_controls["show_head_threshold"].isChecked():
                self.imageTracked.emit(self.head_threshold_frame*255, self.fish_crop, [tail_y_coords, tail_x_coords,
                    spline_y_coords, spline_x_coords,
                    eye_y_coords,
                    eye_x_coords,
                    perp_y_coords,
                    perp_x_coords])
            elif self.param_controls["show_tail_threshold"].isChecked():
                self.imageTracked.emit(self.tail_threshold_frame*255, self.fish_crop, [tail_y_coords, tail_x_coords,
                    spline_y_coords, spline_x_coords,
                    eye_y_coords,
                    eye_x_coords,
                    perp_y_coords,
                    perp_x_coords])
            else:
                self.imageTracked.emit(self.cropped_frame, self.fish_crop, [tail_y_coords, tail_x_coords,
                    spline_y_coords, spline_x_coords,
                    eye_y_coords,
                    eye_x_coords,
                    perp_y_coords,
                    perp_x_coords])

    def track(self):
        if self.params['type_opened'] == "image":
            self.track_image()
        elif self.params['type_opened'] == "folder":
            self.track_folder()
        elif self.params['type_opened'] == "video":
            self.track_video()

    def track_image(self):
        self.save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save image', '', 'Images (*.jpg *.tif *.png)'))

        kwargs_dict = { 'crop': self.params['crop'],
                        'offset': self.params['offset'],
                        'shrink_factor': self.params['shrink_factor'],
                        'invert': self.params['invert'],
                        'min_eye_distance': self.params['min_eye_distance'],
                        'max_eye_distance': self.params['max_eye_distance'],
                        'closest_eye_y_coords': self.params['closest_eye_y_coords'],
                        'closest_eye_x_coords': self.params['closest_eye_x_coords'],
                        'head_threshold': self.params['head_threshold'],
                        'tail_threshold': self.params['tail_threshold'],
                        'track_head': self.params['track_head'],
                        'track_tail': self.params['track_tail'],
                        'save_video': self.params['save_video'],
                        'new_video_fps': self.params['new_video_fps']
                      }

        t = threading.Thread(target=tt.track_image, args=(self.params['last_path'], self.save_path), kwargs=kwargs_dict)

        t.start()

    def track_folder(self):
        self.save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save video', '', 'Videos (*.mov *.tif *.mp4 *.avi)'))

        kwargs_dict = { 'crop': self.params['crop'],
                        'offset': self.params['offset'],
                        'shrink_factor': self.params['shrink_factor'],
                        'invert': self.params['invert'],
                        'min_eye_distance': self.params['min_eye_distance'],
                        'max_eye_distance': self.params['max_eye_distance'],
                        'closest_eye_y_coords': self.params['closest_eye_y_coords'],
                        'closest_eye_x_coords': self.params['closest_eye_x_coords'],
                        'head_threshold': self.params['head_threshold'],
                        'tail_threshold': self.params['tail_threshold'],
                        'track_head': self.params['track_head'],
                        'track_tail': self.params['track_tail'],
                        'save_video': self.params['save_video'],
                        'new_video_fps': self.params['new_video_fps'],
                        'fish_crop_dims': self.params['fish_crop_dims']
                      }

        t = threading.Thread(target=tt.track_folder, args=(self.params['last_path'], self.save_path), kwargs=kwargs_dict)

        t.start()

    def track_video(self):
        print("Invert:", self.params['invert'])
        self.save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save video', '', 'Videos (*.mov *.tif *.mp4 *.avi)'))

        kwargs_dict = { 'crop': self.params['crop'],
                        'offset': self.params['offset'],
                        'shrink_factor': self.params['shrink_factor'],
                        'invert': self.params['invert'],
                        'min_eye_distance': self.params['min_eye_distance'],
                        'max_eye_distance': self.params['max_eye_distance'],
                        'closest_eye_y_coords': self.params['closest_eye_y_coords'],
                        'closest_eye_x_coords': self.params['closest_eye_x_coords'],
                        'head_threshold': self.params['head_threshold'],
                        'tail_threshold': self.params['tail_threshold'],
                        'track_head': self.params['track_head'],
                        'track_tail': self.params['track_tail'],
                        'save_video': self.params['save_video'],
                        'new_video_fps': self.params['new_video_fps'],
                        'mini_crop_dims': self.params['fish_crop_dims']
                      }

        t = threading.Thread(target=tt.track_video, args=(self.params['last_path'], self.save_path), kwargs=kwargs_dict)

        t.start()

    def update_crop_from_selection(self, y_start, y_end, x_start, x_end):
        y_start = round(y_start/self.params['shrink_factor'])
        y_end   = round(y_end/self.params['shrink_factor'])
        x_start = round(x_start/self.params['shrink_factor'])
        x_end   = round(x_end/self.params['shrink_factor'])
        end_add = round(1*self.params['shrink_factor'])

        self.params['crop'] = (abs(y_end - y_start)+end_add, abs(x_end - x_start)+end_add)
        self.params['offset'] = (self.params['offset'][0] + min(y_start, y_end), self.params['offset'][1] + min(x_start, x_end))

        self.param_controls['crop_y'].blockSignals(True)
        self.param_controls['crop_x'].blockSignals(True)
        self.param_controls['offset_y'].blockSignals(True)
        self.param_controls['offset_x'].blockSignals(True)

        self.param_controls['crop_y'].setValue(round(500.0*self.params['crop'][0]/self.current_frame.shape[0]))
        self.param_controls['crop_x'].setValue(round(500.0*self.params['crop'][1]/self.current_frame.shape[1]))

        self.param_controls['offset_y'].setValue(round(500.0*self.params['offset'][0]/self.current_frame.shape[0]))
        self.param_controls['offset_x'].setValue(round(500.0*self.params['offset'][1]/self.current_frame.shape[1]))

        self.param_controls['crop_y'].blockSignals(False)
        self.param_controls['crop_x'].blockSignals(False)
        self.param_controls['offset_y'].blockSignals(False)
        self.param_controls['offset_x'].blockSignals(False)

        self.reshape_frame()

        # update the image preview
        self.update_preview(new_frame=True)

    def reset_crop(self):
        self.params['crop'] = (self.current_frame.shape[0], self.current_frame.shape[1])
        self.params['offset'] = (0, 0)

        self.param_controls['crop_y'].blockSignals(True)
        self.param_controls['crop_x'].blockSignals(True)
        self.param_controls['offset_y'].blockSignals(True)
        self.param_controls['offset_x'].blockSignals(True)

        self.param_controls['crop_y'].setValue(round(500.0*self.params['crop'][0]/self.current_frame.shape[0]))
        self.param_controls['crop_x'].setValue(round(500.0*self.params['crop'][1]/self.current_frame.shape[1]))

        self.param_controls['offset_y'].setValue(round(500.0*self.params['offset'][0]/self.current_frame.shape[0]))
        self.param_controls['offset_x'].setValue(round(500.0*self.params['offset'][1]/self.current_frame.shape[1]))

        self.param_controls['crop_y'].blockSignals(False)
        self.param_controls['crop_x'].blockSignals(False)
        self.param_controls['offset_y'].blockSignals(False)
        self.param_controls['offset_x'].blockSignals(False)

        self.reshape_frame()

        # update the image preview
        self.update_preview(new_frame=True)

    def set_eye_coord_from_selection(self, y, x, num):
        # y = round(y/self.params['shrink_factor'])
        # x = round(x/self.params['shrink_factor'])

        if num == 1:
            self.params['closest_eye_y_coords'] = [y, self.params['closest_eye_y_coords'][1]]
            self.params['closest_eye_x_coords'] = [x, self.params['closest_eye_x_coords'][1]]

            self.param_controls['closest_eye_1_coords'].setText(str((int(y), int(x))))
        else:
            self.params['closest_eye_y_coords'] = [self.params['closest_eye_y_coords'][0], y]
            self.params['closest_eye_x_coords'] = [self.params['closest_eye_x_coords'][0], x]

            self.param_controls['closest_eye_2_coords'].setText(str((int(y), int(x))))
            # print(self.params['closest_eye_x_coords'])

        print(self.params['closest_eye_y_coords'])
        print(self.params['closest_eye_x_coords'])

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

qApp = QtGui.QApplication(sys.argv)
# qApp.setStyle("cleanlooks")

param_window = ParamWindow()
param_window.show()

sys.exit(qApp.exec_())
