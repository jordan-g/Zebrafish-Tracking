import sys
import os
import numpy as np
import pdb
import cv2
import json
import threading
import time

# import the Qt library
from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    try:
        from PyQt4.QtCore import Signal, Qt, QThread
        from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

    except:
        from PyQt5.QtCore import Signal, Qt, QThread
        from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
        from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import analysis as an
import numpy.fft as fft
import scipy
import scipy.stats
import peakdetect

default_params = {'deriv_threshold': 1.0 # threshold to use for angle derivative when finding tail bouts
                 }

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # make figure
        fig = Figure(figsize=(width, height), dpi=dpi)

        # make axis
        self.axes = fig.add_subplot(111)

        # run init of superclass
        FigureCanvas.__init__(self, fig)

        # set background to match window background
        self.setStyleSheet('background: #e2e2e2;')
        fig.patch.set_visible(False)

        # set tight layout
        fig.tight_layout()

        # set parent
        self.setParent(parent)

        # set resize policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def plot_tail_angle_array(self, array, bouts=None, maxes_y=None, maxes_x=None, mins_y=None, mins_x=None, freqs=None, keep_xlim=False):
        self.axes.cla()

        if keep_xlim:
            # store existing xlim
            xlim = self.axes.get_xlim()

        # plot data
        self.axes.plot(array)

        if freqs != None:
            # overlay frequencies
            self.axes.plot(freqs, 'k')

        if maxes_y != None:
            # overlay min & max peaks
            self.axes.plot(maxes_x, maxes_y, 'g.')
            self.axes.plot(mins_x, mins_y, 'r.')

        if bouts != None:
            # overlay bouts
            for i in range(bouts.shape[0]):
                self.axes.axvspan(bouts[i, 0], bouts[i, 1], color='red', alpha=0.2)

        if keep_xlim:
            # restore previous xlim
            self.axes.set_xlim(xlim)

        self.draw()

    def plot_heading_angle_array(self, array, keep_xlim=False):
        self.axes.cla()

        if keep_xlim:
            # store existing xlim
            xlim = self.axes.get_xlim()

        # plot data
        self.axes.plot(array)

        if keep_xlim:
            # restore previous xlim
            self.axes.set_xlim(xlim)

        self.draw()

class AnalysisWindow(QMainWindow):
    def __init__(self, parent):
        # create window
        QMainWindow.__init__(self)
        
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Plot Window")
        self.setGeometry(100, 200, 900, 600)

        # initialize variables
        self.tail_angle_array                   = None
        self.heading_angle_array                = None
        self.smoothed_abs_deriv_abs_angle_array = None
        self.speed_array                        = None
        self.bouts                              = None
        self.peak_maxes_y                       = None
        self.peak_maxes_x                       = None
        self.peak_mins_y                        = None
        self.peak_mins_x                        = None
        self.freqs                              = None
        self.tracking_params_window             = None

        # initialize lists for storing GUI controls
        self.crop_tab_layouts  = []
        self.crop_tab_widgets  = []
        self.tail_tab_layouts  = []
        self.tail_tab_widgets  = []
        self.head_tab_layouts  = []
        self.head_tab_widgets  = []
        self.tail_toolbars     = []
        self.head_toolbars     = []
        self.tail_canvases     = []
        self.head_canvases     = []
        self.plot_tabs_widgets = []
        self.plot_tabs_layouts = []

        self.n_crops          = 0
        self.current_crop_num = -1

        self.smoothing_window_width = 10
        self.threshold              = 0.01
        self.min_width              = 30

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.addStretch(1)

        self.crop_tabs_widget = QTabWidget()
        self.crop_tabs_widget.currentChanged.connect(self.change_crop)
        self.crop_tabs_widget.setElideMode(Qt.ElideLeft)
        self.crop_tabs_layout = QVBoxLayout(self.crop_tabs_widget)
        self.main_layout.addWidget(self.crop_tabs_widget)

        self.create_crop()

        # create button layout for main widget
        main_button_layout = QHBoxLayout()
        main_button_layout.setSpacing(5)
        main_button_layout.addStretch(1)
        self.main_layout.addLayout(main_button_layout)

        # add buttons
        self.load_data_button = QPushButton('Load Data', self)
        self.load_data_button.setMinimumHeight(30)
        self.load_data_button.setMaximumWidth(90)
        self.load_data_button.clicked.connect(lambda:self.load_data())
        main_button_layout.addWidget(self.load_data_button)

        # add buttons
        self.show_tracking_params_button = QPushButton('Tracking Parameters', self)
        self.show_tracking_params_button.setMinimumHeight(30)
        self.show_tracking_params_button.setMaximumWidth(180)
        self.show_tracking_params_button.clicked.connect(lambda:self.show_tracking_params())
        main_button_layout.addWidget(self.show_tracking_params_button)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.show()

    def show_tracking_params(self):
        if self.tracking_params_window != None:
            self.tracking_params_window.close()

        self.tracking_params_window = TrackingParamsWindow(self)
        self.tracking_params_window.show()

        # self.tracking_params_window.create_param_controls(self.params, self.current_crop_num)

    def create_crop(self):
        crop_tab_widget = QWidget(self.crop_tabs_widget)
        crop_tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        crop_tab_layout = QVBoxLayout(crop_tab_widget)

        # add to list of crop widgets & layouts
        self.crop_tab_layouts.append(crop_tab_layout)
        self.crop_tab_widgets.append(crop_tab_widget)

        # create tabs widget & layout
        plot_tabs_widget = QTabWidget()
        plot_tabs_layout  = QVBoxLayout(plot_tabs_widget)
        crop_tab_layout.addWidget(plot_tabs_widget)

        # add to list of crop widgets & layouts
        self.plot_tabs_layouts.append(plot_tabs_layout)
        self.plot_tabs_widgets.append(plot_tabs_widget)

        # create tail tab widget & layout
        tail_tab_widget = QWidget()
        tail_tab_layout = QVBoxLayout(tail_tab_widget)

        # add to list of crop widgets & layouts
        self.tail_tab_layouts.append(tail_tab_layout)
        self.tail_tab_widgets.append(tail_tab_widget)

        # create tail plot canvas & toolbar
        tail_canvas = MplCanvas(tail_tab_widget, width=10, height=10, dpi=75)
        tail_canvas.resize(400, 300)
        tail_toolbar = NavigationToolbar(tail_canvas, self)
        tail_toolbar.hide()
        self.tail_toolbars.append(tail_toolbar)
        self.tail_canvases.append(tail_canvas)

        zoom_button = QPushButton('+ Zoom')
        zoom_button.clicked.connect(self.zoom_tail)
         
        pan_button = QPushButton('> Pan')
        pan_button.clicked.connect(self.pan_tail)
         
        home_button = QPushButton('# Home')
        home_button.clicked.connect(self.home_tail)

        save_button = QPushButton(u'\u2714 Save')
        save_button.clicked.connect(self.save_tail)

        top_tail_button_layout = QHBoxLayout()
        top_tail_button_layout.setSpacing(5)
        top_tail_button_layout.addStretch(1)
        tail_tab_layout.addLayout(top_tail_button_layout)

        top_tail_button_layout.addWidget(zoom_button)
        top_tail_button_layout.addWidget(pan_button)
        top_tail_button_layout.addWidget(home_button)
        top_tail_button_layout.addWidget(save_button)

        # add tail plot canvas & toolbar to tail tab widget
        tail_tab_layout.addWidget(tail_toolbar)
        tail_tab_layout.addWidget(tail_canvas)

        # create button layout for tail tab
        bottom_tail_button_layout = QHBoxLayout()
        tail_tab_layout.addLayout(bottom_tail_button_layout)

        # add buttons
        track_bouts_button = QPushButton('Track Bouts', self)
        track_bouts_button.setMinimumHeight(30)
        track_bouts_button.setMaximumWidth(100)
        track_bouts_button.clicked.connect(lambda:self.track_bouts())
        bottom_tail_button_layout.addWidget(track_bouts_button)

        track_freqs_button = QPushButton('Track Freq', self)
        track_freqs_button.setMinimumHeight(30)
        track_freqs_button.setMaximumWidth(100)
        track_freqs_button.clicked.connect(lambda:self.track_freqs())
        bottom_tail_button_layout.addWidget(track_freqs_button)

        # add checkbox for switching plots
        self.smoothed_deriv_checkbox = QCheckBox("Show smoothed derivative")
        self.smoothed_deriv_checkbox.toggled.connect(lambda:self.show_smoothed_deriv(self.smoothed_deriv_checkbox))
        bottom_tail_button_layout.addWidget(self.smoothed_deriv_checkbox)

        # add param labels & textboxes
        smoothing_window_label = QLabel()
        smoothing_window_label.setText("Smoothing window:")
        bottom_tail_button_layout.addWidget(smoothing_window_label)

        self.smoothing_window_param_box = QLineEdit(self)
        self.smoothing_window_param_box.setMinimumHeight(20)
        self.smoothing_window_param_box.setMinimumWidth(40)
        self.smoothing_window_param_box.setText(str(self.smoothing_window_width))
        bottom_tail_button_layout.addWidget(self.smoothing_window_param_box)

        threshold_label = QLabel()
        threshold_label.setText("Threshold:")
        bottom_tail_button_layout.addWidget(threshold_label)

        self.threshold_param_box = QLineEdit(self)
        self.threshold_param_box.setMinimumHeight(20)
        self.threshold_param_box.setMinimumWidth(40)
        self.threshold_param_box.setText(str(self.threshold))
        bottom_tail_button_layout.addWidget(self.threshold_param_box)

        min_width_label = QLabel()
        min_width_label.setText("Min width:")
        bottom_tail_button_layout.addWidget(min_width_label)

        self.min_width_param_box = QLineEdit(self)
        self.min_width_param_box.setMinimumHeight(20)
        self.min_width_param_box.setMinimumWidth(40)
        self.min_width_param_box.setText(str(self.min_width))
        bottom_tail_button_layout.addWidget(self.min_width_param_box)

        # create head tab widget & layout
        head_tab_widget = QWidget()
        head_tab_layout = QVBoxLayout(head_tab_widget)

        # add to list of crop widgets & layouts
        self.head_tab_layouts.append(head_tab_layout)
        self.head_tab_widgets.append(head_tab_widget)

        # create head plot canvas & toolbar
        head_canvas = MplCanvas(head_tab_widget, width=10, height=10, dpi=75)
        head_toolbar = NavigationToolbar(head_canvas, self)
        head_toolbar.hide()
        self.head_toolbars.append(head_toolbar)
        self.head_canvases.append(head_canvas)

        zoom_button = QPushButton('+ Zoom')
        zoom_button.clicked.connect(self.zoom_head)
         
        pan_button = QPushButton('> Pan')
        pan_button.clicked.connect(self.pan_head)
         
        home_button = QPushButton('# Home')
        home_button.clicked.connect(self.home_head)

        save_button = QPushButton(u'\u2714 Save')
        save_button.clicked.connect(self.save_head)

        top_head_button_layout = QHBoxLayout()
        top_head_button_layout.setSpacing(5)
        top_head_button_layout.addStretch(1)
        head_tab_layout.addLayout(top_head_button_layout)

        top_head_button_layout.addWidget(zoom_button)
        top_head_button_layout.addWidget(pan_button)
        top_head_button_layout.addWidget(home_button)
        top_head_button_layout.addWidget(save_button)

        # add tail plot canvas & toolbar to tail tab widget
        head_tab_layout.addWidget(head_toolbar)
        head_tab_layout.addWidget(head_canvas)

        # create button layout for head tab
        bottom_head_button_layout = QHBoxLayout()
        head_tab_layout.addLayout(bottom_head_button_layout)

        # add buttons
        track_position_button = QPushButton('Track Pos', self)
        track_position_button.setMinimumHeight(30)
        track_position_button.setMaximumWidth(100)
        track_position_button.clicked.connect(lambda:self.track_position())
        bottom_head_button_layout.addWidget(track_position_button)

        # add checkbox for switching plots
        speed_checkbox = QCheckBox("Show speed")
        speed_checkbox.toggled.connect(lambda:self.show_speed(self.speed_checkbox))
        bottom_head_button_layout.addWidget(speed_checkbox)
    
        # set plot tabs widget size
        # plot_tabs_widget.resize(450, 350)

        plot_tabs_widget.addTab(tail_tab_widget, "Tail")
        plot_tabs_widget.addTab(head_tab_widget, "Head")

        self.n_crops += 1

        self.current_crop_num = self.n_crops - 1

        # add tabs to plot tabs widget
        self.crop_tabs_widget.addTab(crop_tab_widget, str(self.current_crop_num))

        self.crop_tabs_widget.setCurrentIndex(self.current_crop_num)

    def change_crop(self, index):
        if index != -1:
            # update current crop number
            self.current_crop_num = index

            self.tail_toolbar = self.tail_toolbars[index]
            self.head_toolbar = self.head_toolbars[index]
            self.tail_canvas = self.tail_canvases[index]
            self.head_canvas = self.head_canvases[index]

    def clear_crops(self):
        self.crop_tab_layouts  = []
        self.crop_tab_widgets  = []
        self.tail_tab_layouts  = []
        self.tail_tab_widgets  = []
        self.head_tab_layouts  = []
        self.head_tab_widgets  = []
        self.tail_toolbars     = []
        self.head_toolbars     = []
        self.tail_canvases     = []
        self.head_canvases     = []
        self.plot_tabs_widgets = []
        self.plot_tabs_layouts = []

        self.head_angle_arrays = []
        self.tail_angle_arrays = []

        for c in range(self.n_crops-1, -1, -1):
            # remove tab
            self.crop_tabs_widget.removeTab(c)

        self.n_crops = 0
        self.current_crop_num = -1

    def home_tail(self):
        self.tail_toolbar.home()
    def zoom_tail(self):
        self.tail_toolbar.zoom()
    def pan_tail(self):
        self.tail_toolbar.pan()
    def save_tail(self):
        self.tail_toolbar.save_figure()

    def home_head(self):
        self.head_toolbar.home()
    def zoom_head(self):
        self.head_toolbar.zoom()
    def pan_head(self):
        self.head_toolbar.pan()
    def save_head(self):
        self.head_toolbar.save_figure()

    def show_smoothed_deriv(self, checkbox):
        if self.smoothed_abs_deriv_abs_angle_array != None:
            if checkbox.isChecked():
                self.tail_canvas.plot_tail_angle_array(self.smoothed_abs_deriv_abs_angle_array, self.bouts, keep_xlim=True)
            else:
                self.tail_canvas.plot_tail_angle_array(self.tail_end_angle_array[self.current_crop_num], self.bouts, self.peak_maxes_y, self.peak_maxes_x, self.peak_mins_y, self.peak_mins_x, self.freqs, keep_xlim=True)

    def show_speed(self, checkbox):
        if self.speed_array != None:
            if checkbox.isChecked():
                self.head_canvas.plot_head_array(self.speed_array, keep_xlim=True)
            else:
                self.head_canvas.plot_head_array(self.head_angle_array, keep_xlim=True)

    def load_data(self, data_path=None):
        if data_path == None:
            # ask the user to select a directory
            self.path = str(QFileDialog.getExistingDirectory(self, 'Open folder'))
        else:
            self.path = data_path

        # load saved tracking data
        (self.tail_coords_array, self.spline_coords_array,
         self.heading_angle_array, self.body_position_array,
         self.eye_coords_array, self.params) = an.open_saved_data(self.path)

        if self.params != None:
            # calculate tail angles
            if self.params['type'] == "freeswimming" and self.params['track_tail']:
                self.tail_angle_array = an.get_freeswimming_tail_angles(self.tail_coords_array, self.heading_angle_array, self.body_position_array)
            elif self.params['type'] == "headfixed":
                self.tail_angle_array = an.get_headfixed_tail_angles(self.tail_coords_array, self.params['tail_direction'])

            # get array of average angle of the last few points of the tail
            self.tail_end_angle_array = np.mean(self.tail_angle_array[:, :, -3:], axis=-1)
            
            # clear crops
            self.clear_crops()

            # get number of saved crops
            n_crops_total = len(self.params['crop_params'])

            for k in range(n_crops_total):
                # create a crop
                self.create_crop()

                # plot heading angle
                if self.heading_angle_array is not None:
                    self.head_canvases[k].plot_heading_angle_array(self.heading_angle_array[k])

                # plot tail angle
                if self.tail_angle_array is not None:
                    self.tail_canvases[k].plot_tail_angle_array(self.tail_end_angle_array[k])

    def track_bouts(self):
        if self.tail_angle_array != None:
            # get params
            self.smoothing_window_width = int(self.smoothing_window_param_box.text())
            self.threshold = float(self.threshold_param_box.text())
            self.min_width = int(self.min_width_param_box.text())

            # get smoothed derivative
            abs_angle_array = np.abs(self.tail_end_angle_array[self.current_crop_num])
            deriv_abs_angle_array = np.gradient(abs_angle_array)
            abs_deriv_abs_angle_array = np.abs(deriv_abs_angle_array)
            normpdf = scipy.stats.norm.pdf(range(-int(self.smoothing_window_width/2),int(self.smoothing_window_width/2)),0,3)
            self.smoothed_abs_deriv_abs_angle_array =  np.convolve(abs_deriv_abs_angle_array,  normpdf/np.sum(normpdf),mode='valid')

            # calculate bout periods
            self.bouts = an.contiguous_regions(self.smoothed_abs_deriv_abs_angle_array > self.threshold)

            # remove bouts that don't have the minimum bout length
            for i in range(self.bouts.shape[0]-1, -1, -1):
                if self.bouts[i, 1] - self.bouts[i, 0] < self.min_width:
                    self.bouts = np.delete(self.bouts, (i), 0)

            # update plot
            self.smoothed_deriv_checkbox.setChecked(False)
            self.tail_canvas.plot_tail_angle_array(self.tail_end_angle_array[self.current_crop_num], self.bouts, keep_xlim=True)

    def track_freqs(self):
        if self.bouts != None:
            # initiate bout maxima & minima coord lists
            self.peak_maxes_y = []
            self.peak_maxes_x = []
            self.peak_mins_y = []
            self.peak_mins_x = []

            # initiate instantaneous frequency array
            self.freqs = np.zeros(self.tail_angle_array.shape[0])

            for i in range(self.bouts.shape[0]):
                # get local maxima & minima
                peak_max, peak_min = peakdetect.peakdet(self.tail_end_angle_array[self.current_crop_num][self.bouts[i, 0]:self.bouts[i, 1]], 0.02)

                # change local coordinates (relative to the start of the bout) to global coordinates
                peak_max[:, 0] += self.bouts[i, 0]
                peak_min[:, 0] += self.bouts[i, 0]

                # add to the bout maxima & minima coord lists
                self.peak_maxes_y += list(peak_max[:, 1])
                self.peak_maxes_x += list(peak_max[:, 0])
                self.peak_mins_y += list(peak_min[:, 1])
                self.peak_mins_x += list(peak_min[:, 0])

            # calculate instantaneous frequencies
            for i in range(len(self.peak_maxes_x)-1):
                self.freqs[self.peak_maxes_x[i]:self.peak_maxes_x[i+1]] = 1.0/(self.peak_maxes_x[i+1] - self.peak_maxes_x[i])

            # update plot
            self.smoothed_deriv_checkbox.setChecked(False)
            self.tail_canvas.plot_tail_angle_array(self.tail_end_angle_array[self.current_crop_num], self.bouts, self.peak_maxes_y, self.peak_maxes_x, self.peak_mins_y, self.peak_mins_x, self.freqs, keep_xlim=True)

    def track_position(self):
        if self.head_angle_array != None:
            # get params
            self.smoothing_window_width = int(self.smoothing_window_param_box.text())

            abs_angle_array = np.abs(self.tail_angle_array)
            deriv_abs_angle_array = np.gradient(abs_angle_array)
            abs_deriv_abs_angle_array = np.abs(deriv_abs_angle_array)
            self.smoothed_abs_deriv_abs_angle_array = np.convolve(abs_deriv_abs_angle_array, np.ones((self.smoothing_window_width,))/self.smoothing_window_width, mode='valid')

            positions_y, positions_x, self.speed_array = an.get_position_history(self.path, plot=False)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def resizeEvent(self, re):
        QMainWindow.resizeEvent(self, re)

        self.plot_tabs_widgets[self.current_crop_num].resize(self.frameGeometry().width()-80, self.frameGeometry().height()-160)

class TrackingParamsWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self)

        # set parent
        self.parent = parent

        # set position & size
        self.setGeometry(550, 100, 10, 10)

        # set title
        self.setWindowTitle("Parameters")

        # create main widget
        self.main_widget = QWidget(self)

        # create main layout
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addStretch(1)
        self.main_layout.setSpacing(5)

        # initialize list of dicts used for accessing all crop parameter controls
        self.param_controls = []

        # create parameter controls
        self.create_param_controls_layout()
        self.create_param_controls(self.parent.params, self.parent.current_crop_num)

        # set main widget
        self.setCentralWidget(self.main_widget)

        # set window buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.show()

    def create_param_controls_layout(self):
        # initialize parameter controls variable
        self.param_controls = None

        # create form layout for param controls with textboxes
        self.form_layout = QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.main_layout.addLayout(self.form_layout)

        # create dict for storing all parameter controls
        self.param_controls = {}

    def create_param_controls(self, params, current_crop_num):
        self.add_parameter_label("type", "Tracking type:", params['type'], self.form_layout)
        self.add_parameter_label("shrink_factor", "Shrink factor:", params['shrink_factor'], self.form_layout)
        self.add_parameter_label("invert", "Invert:", params['invert'], self.form_layout)
        self.add_parameter_label("n_tail_points", "Number of tail points:", params['n_tail_points'], self.form_layout)
        self.add_parameter_label("subtract_background", "Subtract background:", params['subtract_background'], self.form_layout)
        self.add_parameter_label("media_type", "Media type:", params['media_type'], self.form_layout)
        self.add_parameter_label("media_path", "Media path:", params['media_path'], self.form_layout)
        self.add_parameter_label("use_multiprocessing", "Use multiprocessing:", params['use_multiprocessing'], self.form_layout)

        if params['type'] == "freeswimming":
            self.add_parameter_label("adjust_thresholds", "Adjust thresholds:", params['adjust_thresholds'], self.form_layout)
            self.add_parameter_label("track_tail", "Track tail:", params['track_tail'], self.form_layout)
            self.add_parameter_label("track_eyes", "Track eyes:", params['track_eyes'], self.form_layout)
            self.add_parameter_label("min_tail_body_dist", "Body/tail min. dist.:", params['min_tail_body_dist'], self.form_layout)
            self.add_parameter_label("max_tail_body_dist", "Body/tail max. dist.:", params['max_tail_body_dist'], self.form_layout)
            self.add_parameter_label("eye_resize_factor", "Eye resize factor:", params['eye_resize_factor'], self.form_layout)
            self.add_parameter_label("interpolation", "Interpolation:", params['interpolation'], self.form_layout)
            self.add_parameter_label("tail_crop", "Tail crop:", params['tail_crop'], self.form_layout)
        else:
            self.add_parameter_label("tail_direction", "Tail direction:", params['tail_direction'], self.form_layout)
            self.add_parameter_label("tail_start_coords", "Tail start coords:", params['tail_start_coords'], self.form_layout)

    def add_parameter_label(self, label, description, value, parent):
        # make value label & add row to form layout
        param_label = QLabel()
        param_label.setText(str(value))
        parent.addRow(description, param_label)

        # add to list of crop or global controls
        self.param_controls[label] = param_label

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

def convert_direction_to_angle(direction):
    if direction == "up":
        return np.pi/2.0
    elif direction == "left":
        return np.pi
    elif direction == "down":
        return 3.0*np.pi/2.0
    else:
        return 2.0*np.pi

if __name__ == "__main__":
    qApp = QApplication(sys.argv)

    plot_window = AnalysisWindow(None)
    plot_window.show()

    sys.exit(qApp.exec_())