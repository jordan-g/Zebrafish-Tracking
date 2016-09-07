import sys
import os
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
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import analysis as an
import numpy.fft as fft
import scipy
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
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.updateGeometry()

    def plot_tail_array(self, array, bouts=None, maxes_y=None, maxes_x=None, mins_y=None, mins_x=None, freqs=None, keep_xlim=False):
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

    def plot_head_array(self, array, keep_xlim=False):
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

class PlotWindow(QtGui.QMainWindow):
    def __init__(self):
        # create window
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Plot Window")
        self.setGeometry(100, 200, 900, 600)

        # initialize variables
        self.tail_angle_array                   = None
        self.head_angle_array                   = None
        self.smoothed_abs_deriv_abs_angle_array = None
        self.speed_array                        = None
        self.bouts                              = None
        self.peak_maxes_y                       = None
        self.peak_maxes_x                       = None
        self.peak_mins_y                        = None
        self.peak_mins_x                        = None
        self.freqs                              = None
        self.crop_tab_layouts = []
        self.crop_tab_widgets = []
        self.tail_tab_layouts = []
        self.tail_tab_widgets = []
        self.head_tab_layouts = []
        self.head_tab_widgets = []
        self.tail_toolbars = []
        self.head_toolbars = []
        self.tail_canvases = []
        self.head_canvases = []
        self.plot_tabs_widgets = []
        self.plot_tabs_layouts = []

        self.tail_angle_arrays = []
        self.head_angle_arrays = []

        self.n_crops = 0
        self.current_crop_num = -1

        self.smoothing_window_width = 10
        self.threshold = 0.01
        self.min_width = 30

        # add file menu
        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        # create main widget & layout
        self.main_widget = QtGui.QWidget(self)
        self.main_layout = QtGui.QVBoxLayout(self.main_widget)
        self.main_layout.addStretch(1)

        self.crop_tabs_widget = QtGui.QTabWidget()
        self.crop_tabs_widget.currentChanged.connect(self.change_crop)
        self.crop_tabs_widget.setElideMode(QtCore.Qt.ElideLeft)
        # self.crop_tabs_widget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.crop_tabs_layout = QtGui.QVBoxLayout(self.crop_tabs_widget)
        self.main_layout.addWidget(self.crop_tabs_widget)

        self.create_crop()

        # create button layout for main widget
        main_button_layout = QtGui.QHBoxLayout()
        self.main_layout.addLayout(main_button_layout)

        # add buttons
        self.load_data_button = QtGui.QPushButton('Load Data', self)
        self.load_data_button.setMinimumHeight(30)
        self.load_data_button.setMaximumWidth(90)
        self.load_data_button.clicked.connect(lambda:self.load_data())
        main_button_layout.addWidget(self.load_data_button)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def create_crop(self):
        crop_tab_widget = QtGui.QWidget(self.crop_tabs_widget)
        crop_tab_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        crop_tab_layout = QtGui.QVBoxLayout(crop_tab_widget)

        # add to list of crop widgets & layouts
        self.crop_tab_layouts.append(crop_tab_layout)
        self.crop_tab_widgets.append(crop_tab_widget)

        # create tabs widget & layout
        plot_tabs_widget = QtGui.QTabWidget()
        plot_tabs_layout  = QtGui.QVBoxLayout(plot_tabs_widget)
        crop_tab_layout.addWidget(plot_tabs_widget)

        # add to list of crop widgets & layouts
        self.plot_tabs_layouts.append(plot_tabs_layout)
        self.plot_tabs_widgets.append(plot_tabs_widget)

        # create tail tab widget & layout
        tail_tab_widget = QtGui.QWidget()
        tail_tab_layout = QtGui.QVBoxLayout(tail_tab_widget)

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

        zoom_button = QtGui.QPushButton(u'\u2A01 Zoom')
        zoom_button.clicked.connect(self.zoom_tail)
         
        pan_button = QtGui.QPushButton(u'\u2921 Pan')
        pan_button.clicked.connect(self.pan_tail)
         
        home_button = QtGui.QPushButton(u'\u2B51 Home')
        home_button.clicked.connect(self.home_tail)

        save_button = QtGui.QPushButton(u'\u2714 Save')
        save_button.clicked.connect(self.save_tail)

        top_tail_button_layout = QtGui.QHBoxLayout()
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
        bottom_tail_button_layout = QtGui.QHBoxLayout()
        tail_tab_layout.addLayout(bottom_tail_button_layout)

        # add buttons
        track_bouts_button = QtGui.QPushButton('Track Bouts', self)
        track_bouts_button.setMinimumHeight(30)
        track_bouts_button.setMaximumWidth(100)
        track_bouts_button.clicked.connect(lambda:self.track_bouts())
        bottom_tail_button_layout.addWidget(track_bouts_button)

        track_freqs_button = QtGui.QPushButton('Track Freq', self)
        track_freqs_button.setMinimumHeight(30)
        track_freqs_button.setMaximumWidth(100)
        track_freqs_button.clicked.connect(lambda:self.track_freqs())
        bottom_tail_button_layout.addWidget(track_freqs_button)

        # add checkbox for switching plots
        smoothed_deriv_checkbox = QtGui.QCheckBox("Show smoothed deriv")
        smoothed_deriv_checkbox.toggled.connect(lambda:self.show_smoothed_deriv(smoothed_deriv_checkbox))
        bottom_tail_button_layout.addWidget(smoothed_deriv_checkbox)

        # add param labels & textboxes
        smoothing_window_label = QtGui.QLabel()
        smoothing_window_label.setText("Smoothing Window: =")
        bottom_tail_button_layout.addWidget(smoothing_window_label)

        smoothing_window_param_box = QtGui.QLineEdit(self)
        smoothing_window_param_box.setMinimumHeight(10)
        smoothing_window_param_box.setText(str(self.smoothing_window_width))
        bottom_tail_button_layout.addWidget(smoothing_window_param_box)

        threshold_label = QtGui.QLabel()
        threshold_label.setText("Threshold: =")
        bottom_tail_button_layout.addWidget(threshold_label)

        threshold_param_box = QtGui.QLineEdit(self)
        threshold_param_box.setMinimumHeight(10)
        threshold_param_box.setText(str(self.threshold))
        bottom_tail_button_layout.addWidget(threshold_param_box)

        min_width_label = QtGui.QLabel()
        min_width_label.setText("Min width: =")
        bottom_tail_button_layout.addWidget(min_width_label)

        min_width_param_box = QtGui.QLineEdit(self)
        min_width_param_box.setMinimumHeight(10)
        min_width_param_box.setText(str(self.min_width))
        bottom_tail_button_layout.addWidget(min_width_param_box)

        # create head tab widget & layout
        head_tab_widget = QtGui.QWidget()
        head_tab_layout = QtGui.QVBoxLayout(head_tab_widget)

        # add to list of crop widgets & layouts
        self.head_tab_layouts.append(head_tab_layout)
        self.head_tab_widgets.append(head_tab_widget)

        # create head plot canvas & toolbar
        head_canvas = MplCanvas(head_tab_widget, width=10, height=10, dpi=75)
        head_toolbar = NavigationToolbar(head_canvas, self)
        head_toolbar.hide()
        self.head_toolbars.append(head_toolbar)
        self.head_canvases.append(head_canvas)

        zoom_button = QtGui.QPushButton(u'\u2A01 Zoom')
        zoom_button.clicked.connect(self.zoom_head)
         
        pan_button = QtGui.QPushButton(u'\u2921 Pan')
        pan_button.clicked.connect(self.pan_head)
         
        home_button = QtGui.QPushButton(u'\u2B51 Home')
        home_button.clicked.connect(self.home_head)

        save_button = QtGui.QPushButton(u'\u2714 Save')
        save_button.clicked.connect(self.save_head)

        top_head_button_layout = QtGui.QHBoxLayout()
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
        bottom_head_button_layout = QtGui.QHBoxLayout()
        head_tab_layout.addLayout(bottom_head_button_layout)

        # add buttons
        track_position_button = QtGui.QPushButton('Track Pos', self)
        track_position_button.setMinimumHeight(30)
        track_position_button.setMaximumWidth(100)
        track_position_button.clicked.connect(lambda:self.track_position())
        bottom_head_button_layout.addWidget(track_position_button)

        # add checkbox for switching plots
        speed_checkbox = QtGui.QCheckBox("Show speed")
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
        print(index)
        if index != -1:
            # get params for this crop
            # self.current_crop_params = self.params['crop_params'][index]

            # update current crop number
            self.current_crop_num = index

            self.tail_toolbar = self.tail_toolbars[index]
            self.head_toolbar = self.head_toolbars[index]
            self.tail_canvas = self.tail_canvases[index]
            self.head_canvas = self.head_canvases[index]

    def clear_crops(self):
        print(self.n_crops)

        self.crop_tab_layouts = []
        self.crop_tab_widgets = []
        self.tail_tab_layouts = []
        self.tail_tab_widgets = []
        self.head_tab_layouts = []
        self.head_tab_widgets = []
        self.tail_toolbars = []
        self.head_toolbars = []
        self.tail_canvases = []
        self.head_canvases = []
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
                self.tail_canvas.plot_tail_array(self.smoothed_abs_deriv_abs_angle_array, self.bouts, keep_xlim=True)
            else:
                self.tail_canvas.plot_tail_array(self.tail_angle_array, self.bouts, self.peak_maxes_y, self.peak_maxes_x, self.peak_mins_y, self.peak_mins_x, self.freqs, keep_xlim=True)

    def show_speed(self, checkbox):
        if self.speed_array != None:
            if checkbox.isChecked():
                self.head_canvas.plot_head_array(self.speed_array, keep_xlim=True)
            else:
                self.head_canvas.plot_head_array(self.head_angle_array, keep_xlim=True)

    def load_data(self):
        # ask the user to select a directory
        self.path = str(QtGui.QFileDialog.getExistingDirectory(self, 'Open folder'))
        eye_coords_array, perp_coords_array, tail_coords_array, spline_coords_array, self.params = an.open_saved_data(self.path)
        self.clear_crops()

        n_crops_tot = len(self.params['crop_params'])

        print(self.n_crops, self.current_crop_num)

        for k in range(n_crops_tot):
            self.create_crop()

            eye_coords_array, perp_coords_array, tail_coords_array, spline_coords_array, self.params = an.open_saved_data(self.path, k)
            perp_vectors, spline_vectors = an.get_vectors(perp_coords_array, spline_coords_array, tail_coords_array)
            self.head_angle_arrays.append(an.get_heading_angle(perp_vectors))
            self.tail_angle_arrays.append(an.get_tail_angle(perp_vectors, spline_vectors))

        # an.plot_tail_angle_heatmap(perp_vectors, spline_vectors)

            self.tail_canvases[k].plot_tail_array(self.tail_angle_arrays[k])
            self.head_canvases[k].plot_head_array(self.head_angle_arrays[k])

    def track_bouts(self):
        if self.tail_angle_array != None:
            # get params
            self.smoothing_window_width = int(self.smoothing_window_param_box.text())
            self.threshold = float(self.threshold_param_box.text())
            self.min_width = int(self.min_width_param_box.text())

            # get smoothed derivative
            abs_angle_array = np.abs(self.tail_angle_array)
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
            self.tail_canvas.plot_tail_array(self.tail_angle_array, self.bouts, keep_xlim=True)

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
                peak_max, peak_min = peakdetect.peakdet(self.tail_angle_array[self.bouts[i, 0]:self.bouts[i, 1]], 0.02)

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
            self.tail_canvas.plot_tail_array(self.tail_angle_array, self.bouts, self.peak_maxes_y, self.peak_maxes_x, self.peak_mins_y, self.peak_mins_x, self.freqs, keep_xlim=True)

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
        QtGui.QMainWindow.resizeEvent(self, re)

        self.plot_tabs_widgets[self.current_crop_num].resize(self.frameGeometry().width()-80, self.frameGeometry().height()-160)

qApp = QtGui.QApplication(sys.argv)

plot_window = PlotWindow()
plot_window.show()

sys.exit(qApp.exec_())
