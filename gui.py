import sys
import os
import tracking as tt
import numpy as np
import pdb
import cv2
import json
import threading

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

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class DynamicCanvas(PlotCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        PlotCanvas.__init__(self, *args, **kwargs)

        self.image_tracked   = False
        self.show_threshold  = False
        self.image           = None
        self.threshold_image = None
        self.tracking_list   = None

    def compute_initial_figure(self):
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.fig.tight_layout()

    def init_plot(self):
        self.axes.clear()

    def end_plot(self):
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.fig.tight_layout()
        self.draw()

    def add_image(self, image):
        self.axes.imshow(image, cmap='gray', interpolation='none', vmin=0, vmax=255)
        self.image = image

    def add_threshold_image(self, threshold_image):
        self.axes.imshow(threshold_image, cmap='gray', interpolation='none',visible=True, vmin=0, vmax=1)
        self.threshold_image = threshold_image

    def generate_tracking(self, tracking_list):
        tail_y_coords = tracking_list[0]
        tail_x_coords = tracking_list[1]
        spline_y_coords = tracking_list[2]
        spline_x_coords = tracking_list[3]
        eye_y_coords = tracking_list[4]
        eye_x_coords = tracking_list[5]
        perp_y_coords = tracking_list[6]
        perp_x_coords = tracking_list[7]

        spline_line = None
        eyes_line = None

        if spline_y_coords != None:
            spline_ax = self.fig.add_subplot(111)
            spline_line = spline_ax.plot(spline_x_coords, spline_y_coords, lw=1, c='red')[0]

        if eye_y_coords != None:
            eyes_ax = self.fig.add_subplot(111)
            eyes_line = eyes_ax.plot(eye_x_coords, eye_y_coords, lw=1, c='orange')[0]

        self.tracking_list = tracking_list

        return spline_line, eyes_line

    def add_tracking(self, spline_line, eyes_line):
        if spline_line:
            self.axes.add_line(spline_line)
        if eyes_line:
            self.axes.add_line(eyes_line)

        self.image_tracked = True

    def plot_image(self, image, new_image=False):
        print("Plotting image.")
        self.init_plot()
        if (not new_image) and self.image_tracked:
            spline_line, eyes_line = self.generate_tracking(self.tracking_list)
        self.add_image(image)
        if (not new_image) and self.image_tracked:
            self.add_tracking(spline_line, eyes_line)
        self.end_plot()

    def plot_threshold_image(self, threshold_image, new_image=False):
        print("Plotting threshold image.")
        self.show_threshold = True
        self.init_plot()
        if (not new_image) and self.image_tracked:
            spline_line, eyes_line = self.generate_tracking(self.tracking_list)
        self.add_threshold_image(threshold_image)
        if (not new_image) and self.image_tracked:
            self.add_tracking(spline_line, eyes_line)
        self.end_plot()

    def remove_threshold_image(self):
        print("Removing threshold image.")
        self.show_threshold = False
        self.init_plot()
        if self.image_tracked:
            spline_line, eyes_line = self.generate_tracking(self.tracking_list)
        self.add_image(self.image)
        if self.image_tracked:
            self.add_tracking(spline_line, eyes_line)
        self.end_plot()

    def plot_tracked_image(self, image, tracking_list):
        print("Plotting tracked image.")
        self.init_plot()
        spline_line, eyes_line = self.generate_tracking(tracking_list)
        if self.show_threshold == True:
            self.add_threshold_image(self.threshold_image)
        else:
            self.add_image(image)
        self.add_tracking(spline_line, eyes_line)
        self.end_plot()

class PlotWindow(QtGui.QMainWindow):
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

        self.l = QtGui.QVBoxLayout(self.main_widget)
        self.dc = DynamicCanvas(self.main_widget, width=5, height=4, dpi=100)
        self.l.addWidget(self.dc)

        self.image_slider = None

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def plot_threshold_image(self, threshold_image, new_image=False):
        self.dc.plot_threshold_image(threshold_image, new_image=new_image)

    def remove_threshold_image(self):
        self.dc.remove_threshold_image()

    def plot_image(self, image, new_image=False):
        if not self.param_window.image_opened:
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
                self.image_slider.valueChanged.connect(self.switch_image)

                self.l.addWidget(self.image_slider)

        self.dc.plot_image(image, new_image=new_image)

    def switch_image(self, value):
        self.param_window.switch_image(value)

    def plot_tracked_image(self, image, tracking_list):
        self.dc.plot_tracked_image(image, tracking_list)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

class ParamWindow(QtGui.QMainWindow):
    imageLoaded = QtCore.pyqtSignal(np.ndarray, bool)
    imageTracked = QtCore.pyqtSignal(np.ndarray, list)
    thresholdLoaded = QtCore.pyqtSignal(np.ndarray, bool)
    thresholdUnloaded = QtCore.pyqtSignal()

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.plot_window = PlotWindow(self)
        self.plot_window.setWindowTitle("Preview")
        self.plot_window.show()

        self.head_threshold_image = None
        self.tail_threshold_image = None

        self.setGeometry(100, 200, 300, 800)

        openFolder = QtGui.QAction(QtGui.QIcon('open.png'), 'Open Directory', self)
        openFolder.setShortcut('Ctrl+Shift+O')
        openFolder.setStatusTip('Open a directory of images')
        openFolder.triggered.connect(lambda:self.open_folder(""))

        openVideo = QtGui.QAction(QtGui.QIcon('open.png'), 'Open Video', self)
        openVideo.setShortcut('Ctrl+Alt+O')
        openVideo.setStatusTip('Open a video')
        openVideo.triggered.connect(lambda:self.open_video(""))

        openImage = QtGui.QAction(QtGui.QIcon('open.png'), 'Open Image', self)
        openImage.setShortcut('Ctrl+O')
        openImage.setStatusTip('Open a video')
        openImage.triggered.connect(lambda:self.open_image(""))

        trackFrame = QtGui.QAction(QtGui.QIcon('open.png'), 'Track Frame', self)
        trackFrame.setShortcut('Ctrl+T')
        trackFrame.setStatusTip('Track current image')
        trackFrame.triggered.connect(self.track_frame)

        saveParams = QtGui.QAction(QtGui.QIcon('save.png'), 'Save parameters', self)
        saveParams.setShortcut('Return')
        saveParams.setStatusTip('Save parameters')
        saveParams.triggered.connect(self.save_params)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFolder)
        fileMenu.addAction(openVideo)
        fileMenu.addAction(openImage)
        fileMenu.addAction(saveParams)
        fileMenu.addAction(trackFrame)

        self.mainWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.mainWidget)

        self.layout = QtGui.QVBoxLayout()
        self.form_layout = QtGui.QFormLayout()

        self.load_params()

        self.param_controls = {}

        self.add_checkbox('invert_image', "Invert image", self.toggle_invert_image)
        self.add_checkbox('show_head_threshold', "Show head threshold", self.toggle_threshold_image)
        self.add_checkbox('show_tail_threshold', "Show tail threshold", self.toggle_threshold_image)
        self.add_checkbox('track_head', "Track head", self.toggle_tracking)
        self.add_checkbox('track_tail', "Track tail", self.toggle_tracking)
        self.param_controls['track_head'].setChecked(True)
        self.param_controls['track_tail'].setChecked(True)

        self.open_last_file()

        if self.crop == None:
            self.add_slider('crop_y', 'Crop y:', 1, 100, 100)
            self.add_slider('crop_x', 'Crop x:', 1, 100, 100)
        else:
            self.add_slider('crop_y', 'Crop y:', 1, 100, 100*self.crop[0]/self.image.shape[0])
            self.add_slider('crop_x', 'Crop x:', 1, 100, 100*self.crop[1]/self.image.shape[1])

        if self.offset == None:
            self.add_slider('offset_y', 'Offset y:', 0, 99, 0)
            self.add_slider('offset_x', 'Offset x:', 0, 99, 0)
        else:
            self.add_slider('offset_y', 'Offset y:', 0, 99, 100*self.offset[0]/self.image.shape[0])
            self.add_slider('offset_x', 'Offset x:', 0, 99, 100*self.offset[1]/self.image.shape[1])

        # self.add_slider('crop_y', 'Crop y:', 1, self.image.shape[0], crop_y)
        # self.add_slider('crop_x', 'Crop x:', 1, self.image.shape[1], crop_x)
        # self.add_textbox('crop_y', 'Crop y:', crop_y)
        # self.add_textbox('crop_x', 'Crop x:', crop_x)

        # self.add_textbox('offset_y', 'Offset y:', offset_y)
        # self.add_textbox('offset_x', 'Offset x:', offset_x)

        self.add_slider('shrink_factor', 'Shrink factor:', 1, 10, int(10*self.shrink_factor))
        self.add_slider('eye_1_index', 'Index of eye 1:', 0, 5, self.eye_1_index)
        self.add_slider('eye_2_index', 'Index of eye 2:', 0, 5, self.eye_2_index)
        self.add_textbox('min_eye_distance', 'Minimum distance b/w eye & tail:', self.min_eye_distance)
        self.add_textbox('head_threshold', 'Head threshold:', self.head_threshold)
        self.add_textbox('tail_threshold', 'Tail threshold:', self.tail_threshold)

        self.layout.addLayout(self.form_layout)

        hbox1 = QtGui.QHBoxLayout()
        hbox1.addStretch(1)
        hbox2 = QtGui.QHBoxLayout()
        hbox2.addStretch(1)

        self.open_image_button = QtGui.QPushButton('Open Image', self)
        self.open_image_button.setMinimumHeight(10)
        self.open_image_button.setMaximumWidth(180)
        self.open_image_button.clicked.connect(lambda:self.open_image(""))
        hbox1.addWidget(self.open_image_button)

        self.open_folder_button = QtGui.QPushButton('Open Folder', self)
        self.open_folder_button.setMinimumHeight(10)
        self.open_folder_button.setMaximumWidth(180)
        self.open_folder_button.clicked.connect(lambda:self.open_folder(""))
        hbox1.addWidget(self.open_folder_button)

        self.open_video_button = QtGui.QPushButton('Open Video', self)
        self.open_video_button.setMinimumHeight(10)
        self.open_video_button.setMaximumWidth(180)
        self.open_video_button.clicked.connect(lambda:self.open_video(""))
        hbox1.addWidget(self.open_video_button)

        self.save_button = QtGui.QPushButton('Save', self)
        self.save_button.setMinimumHeight(10)
        self.save_button.setMaximumWidth(80)
        self.save_button.clicked.connect(self.save_params)
        hbox1.addWidget(self.save_button)

        self.track_button = QtGui.QPushButton('Track', self)
        self.track_button.setMinimumHeight(10)
        self.track_button.setMaximumWidth(80)
        self.track_button.clicked.connect(self.track_frame)
        hbox2.addWidget(self.track_button)

        self.track_button = QtGui.QPushButton('Track and Save', self)
        self.track_button.setMinimumHeight(10)
        self.track_button.setMaximumWidth(180)
        self.track_button.clicked.connect(self.track)
        hbox2.addWidget(self.track_button)

        self.layout.addLayout(hbox1)
        self.layout.addLayout(hbox2)

        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.mainWidget.setLayout(self.layout)

        self.setWindowTitle('File dialog')
        self.show()

    def load_params(self):
        # sete xperiments file
        self.params_file = "last_params.json"

        try:
            # load experiments
            with open(self.params_file, "r") as input_file:
                self.params = json.load(input_file)
        except:
            # if none exist, create & save a default experiment
            self.params = {'shrink_factor': 1.0,
                           'offset': None,
                           'crop': None,
                           'tail_threshold': 200,
                           'head_threshold': 50,
                           'video_opened': False,
                           'folder_opened': False,
                           'image_opened': False,
                           'min_eye_distance': 20,
                           'eye_1_index': 0,
                           'eye_2_index': 1,
                           'track_head': True,
                           'track_tail': True,
                           'last_path': "",
                           }
            self.save_params_file()

        if self.params['last_path'] == "":
            self.params = {'shrink_factor': 1.0,
                           'offset': None,
                           'crop': None,
                           'tail_threshold': 200,
                           'head_threshold': 50,
                           'video_opened': False,
                           'folder_opened': False,
                           'image_opened': False,
                           'min_eye_distance': 20,
                           'eye_1_index': 0,
                           'eye_2_index': 1,
                           'track_head': True,
                           'track_tail': True,
                           'last_path': "",
                           }

        self.shrink_factor = self.params['shrink_factor']
        self.offset = self.params['offset']
        self.crop = self.params['crop']
        self.tail_threshold = self.params['tail_threshold']
        self.head_threshold = self.params['head_threshold']
        self.min_eye_distance = self.params['min_eye_distance']
        self.eye_1_index = self.params['eye_1_index']
        self.eye_2_index = self.params['eye_2_index']
        self.track_head = self.params['track_head']
        self.track_tail = self.params['track_tail']
        self.video_opened = self.params['video_opened']
        self.folder_opened = self.params['folder_opened']
        self.image_opened = self.params['image_opened']

    def open_last_file(self):
        try:
            if self.video_opened == True:
                self.open_video(path=self.params['last_path'])
            elif self.folder_opened == True:
                self.open_folder(path=self.params['last_path'])
            elif self.image_opened == True:
                self.open_image(path=self.params['last_path'])
        except:
            pass

    def save_params_file(self):
        # save experiments to file
        with open(self.params_file, "w") as output_file:
            json.dump(self.params, output_file)

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

        slider.valueChanged.connect(self.save_params)
        self.form_layout.addRow(description, slider)
        self.param_controls[label] = slider

    def add_checkbox(self, label, description, toggle_func):
        checkbox = QtGui.QCheckBox(description)
        checkbox.setChecked(False)

        checkbox.toggled.connect(lambda:toggle_func(checkbox))
        self.layout.addWidget(checkbox)
        self.param_controls[label] = checkbox

    def open_folder(self, path=""):
        if path == "":
            self.path = str(QtGui.QFileDialog.getExistingDirectory(self, 'Open folder'))
        else:
            self.path = path

        if self.path != "":
            self.n_frames = 0
            self.image_paths = []

            for filename in sorted(os.listdir(self.path)):
                if filename.endswith('.tif') or filename.endswith('.png'):
                    image_path = self.path + "/" + filename
                    self.image_paths.append(image_path)
                    self.n_frames += 1

            if self.n_frames == 0:
                print("Could not find any images.")
                return

            if len(self.image_paths) >= 100:
                f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
                self.image_paths = [ self.image_paths[i] for i in f(100, self.n_frames)]
                self.n_frames = 100

            self.folder_opened = True
            self.video_opened  = False
            self.image_opened  = False

            self.switch_image(0)

    def open_image(self, path=""):
        if path == "":
            self.path = str(QtGui.QFileDialog.getOpenFileName(self, 'Open image', '', 'Images (*.jpg *.tif *.png)'))
        else:
            self.path = path

        if self.path != "":
            self.n_frames = 1

            self.folder_opeend = False
            self.video_opened  = False
            self.image_opened  = True

            self.switch_image(0)

    def open_video(self, path=""):
        if path == "":
            self.path = str(QtGui.QFileDialog.getOpenFileName(self, 'Open video', '', 'Videos (*.mov *.tif *.mp4)'))
        else:
            self.path = path

        if self.path != "":
            self.frames = tt.load_video(self.path)

            if self.frames != None:
                self.n_frames = len(self.frames)

                if len(self.frames) >= 100:
                    f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
                    self.frames = [self.frames[i] for i in f(100, self.n_frames)]
                    self.n_frames = 100
                print(self.n_frames, len(self.frames))

                self.folder_opeend = False
                self.video_opened  = True
                self.image_opened  = False

                self.switch_image(0)

    def switch_image(self, n):
        if self.video_opened:
            print("Switching to frame {}".format(n))
            self.image = self.frames[n]
        elif self.folder_opened:
            self.image = tt.load_image(self.image_paths[n])
        elif self.image_opened:
            self.image = tt.load_image(self.path)

        if self.crop == None or self.offset == None:
            self.crop = self.image.shape
            self.param_controls['crop_y'].setValue(100*self.image.shape[0])
            self.param_controls['crop_x'].setValue(100*self.image.shape[1])

            self.offset = (0, 0)
            self.param_controls['offset_y'].setValue(0)
            self.param_controls['offset_x'].setValue(0)

        # print(self.crop, self.offset)

        # self.param_controls['show_head_threshold'].setChecked(False)
        # self.param_controls['show_tail_threshold'].setChecked(False)
        if self.param_controls['invert_image'].isChecked():
            self.invert_image()
        self.reshape_image(new_image=True)

    def reshape_image(self, new_image=False):
        # crop the image
        if self.crop != None and self.offset != None:
            self.cropped_image = tt.crop_image(self.image, self.offset, self.crop)
        else:
            self.cropped_image = self.image

        # shrink the image
        if self.shrink_factor != None:
            self.cropped_image = tt.shrink_image(self.cropped_image, self.shrink_factor)

        self.threshold_image()

        self.update_plot(new_image=new_image)

    def threshold_image(self):
        self.head_threshold_image = tt.get_head_threshold_image(self.cropped_image, self.head_threshold)
        self.tail_threshold_image = tt.get_tail_threshold_image(self.cropped_image, self.tail_threshold)

    def update_plot(self, new_image=False):
        if not self.signalsBlocked():
            if self.param_controls["show_head_threshold"].isChecked():
                self.thresholdLoaded.emit(self.head_threshold_image, new_image)
            elif self.param_controls["show_tail_threshold"].isChecked():
                self.thresholdLoaded.emit(self.tail_threshold_image, new_image)
            else:
                self.imageLoaded.emit(self.cropped_image, new_image)

    def invert_image(self):
        self.image = (255 - self.image)

    def toggle_invert_image(self, checkbox):
        self.invert_image()
        self.reshape_image()

    def toggle_threshold_image(self, checkbox):
        if self.head_threshold_image != None:
            if checkbox.isChecked():
                if not self.signalsBlocked():
                    if checkbox.text() == "Show head threshold":
                        self.param_controls["show_tail_threshold"].setChecked(False)
                        self.thresholdLoaded.emit(self.head_threshold_image, False)
                    elif checkbox.text() == "Show tail threshold":
                        self.param_controls["show_head_threshold"].setChecked(False)
                        self.thresholdLoaded.emit(self.tail_threshold_image, False)
            else:
                if not self.signalsBlocked():
                    self.thresholdUnloaded.emit()

    def toggle_tracking(self, checkbox):
        if checkbox.isChecked():
            track = True
        else:
            track = False

        if checkbox.text() == "Track head":
            self.track_head = track
        elif checkbox.text() == "Track tail":
            self.track_tail = track

    def save_params(self):
        crop_y = self.param_controls['crop_y'].value()*self.image.shape[0]/100
        crop_x = self.param_controls['crop_x'].value()*self.image.shape[1]/100
        offset_y = self.param_controls['offset_y'].value()*self.image.shape[0]/100
        offset_x = self.param_controls['offset_x'].value()*self.image.shape[1]/100

        self.eye_1_index = int(self.param_controls['eye_1_index'].value())
        self.eye_2_index = int(self.param_controls['eye_2_index'].value())
        self.min_eye_distance = int(self.param_controls['min_eye_distance'].text())

        new_head_threshold = int(self.param_controls['head_threshold'].text())
        new_tail_threshold = int(self.param_controls['tail_threshold'].text())
        new_crop = (int(crop_y), int(crop_x))
        new_offset = (int(offset_y), int(offset_x))
        new_shrink_factor = float(self.param_controls['shrink_factor'].value())/10.0

        generate_new_image = False

        if self.crop != new_crop:
            self.crop = new_crop
            generate_new_image = True
        if self.offset != new_offset:
            self.offset = new_offset
            generate_new_image = True
        if self.shrink_factor != new_shrink_factor:
            self.shrink_factor = new_shrink_factor
            generate_new_image = True
        if self.head_threshold != new_head_threshold:
            self.head_threshold = new_head_threshold
            generate_new_image = True
        if self.tail_threshold != new_tail_threshold:
            self.tail_threshold = new_tail_threshold
            generate_new_image = True

        self.params['shrink_factor'] = self.shrink_factor
        self.params['offset'] = self.offset
        self.params['crop'] = self.crop
        self.params['tail_threshold'] = self.tail_threshold
        self.params['head_threshold'] = self.head_threshold
        self.params['min_eye_distance'] = self.min_eye_distance
        self.params['eye_1_index'] = self.eye_1_index
        self.params['eye_2_index'] = self.eye_2_index
        self.params['track_head'] = self.track_head
        self.params['track_tail'] = self.track_tail
        self.params['video_opened'] = self.video_opened
        self.params['folder_opened'] = self.folder_opened
        self.params['image_opened'] = self.image_opened
        self.params['last_path'] = self.path

        self.save_params_file()

        if generate_new_image:
            self.reshape_image(new_image=True)
            # self.update_plot()

    def track_frame(self):
        self.save_params()

        if self.track_head:
            (eye_y_coords, eye_x_coords,
            perp_y_coords, perp_x_coords) = tt.track_head(self.head_threshold_image,
                                                            self.eye_1_index, self.eye_2_index)

            if eye_x_coords == None:
                print("Could not track head.")
        else:
            (eye_y_coords, eye_x_coords,
            perp_y_coords, perp_x_coords) = [None]*4

        if self.track_tail:
            (tail_y_coords, tail_x_coords,
            spline_y_coords, spline_x_coords) = tt.track_tail(self.tail_threshold_image,
                                                                eye_x_coords, eye_y_coords,
                                                                min_eye_distance=self.min_eye_distance*self.shrink_factor)
            if tail_x_coords == None:
                print("Could not track tail.")
        else:
            (tail_y_coords, tail_x_coords,
            spline_y_coords, spline_x_coords) = [None]*4

        if not self.signalsBlocked():
            self.imageTracked.emit(self.cropped_image, [tail_y_coords, tail_x_coords,
                spline_y_coords, spline_x_coords,
                eye_y_coords,
                eye_x_coords,
                perp_y_coords,
                perp_x_coords])

    def track(self):
        if self.image_opened:
            self.track_image()
        elif self.folder_opened:
            self.track_folder()
        elif self.video_opened:
            self.track_video()

    def track_image(self):
        self.save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save image', '', 'Images (*.jpg *.tif *.png)'))

        t = threading.Thread(target=tt.track_image, args=(self.path, self.save_path, self.crop, self.offset, self.shrink_factor,
                            self.head_threshold, self.tail_threshold, self.min_eye_distance*self.shrink_factor,
                            self.eye_1_index, self.eye_2_index, self.track_head, self.track_tail))

        t.start()

    def track_folder(self):
        self.save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save video', '', 'Videos (*.mov *.tif *.mp4 *.avi)'))

        t = threading.Thread(target=tt.track_folder, args=(self.path, self.save_path, self.crop, self.offset, self.shrink_factor,
                            self.head_threshold, self.tail_threshold, self.min_eye_distance*self.shrink_factor,
                            self.eye_1_index, self.eye_2_index, self.track_head, self.track_tail))

        t.start()

    def track_video(self):
        self.save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save video', '', 'Videos (*.mov *.tif *.mp4 *.avi)'))

        t = threading.Thread(target=tt.track_video, args=(self.path, self.save_path, self.crop, self.offset, self.shrink_factor,
                            self.head_threshold, self.tail_threshold, self.min_eye_distance*self.shrink_factor,
                            self.eye_1_index, self.eye_2_index, self.track_head, self.track_tail))

        t.start()

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

qApp = QtGui.QApplication(sys.argv)
qApp.setStyle("cleanlooks")

param_window = ParamWindow()
param_window.setWindowTitle("Parameters")
param_window.show()

sys.exit(qApp.exec_())
