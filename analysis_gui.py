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
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import analysis as an
import numpy.fft as fft
import scipy
import peakdetect

default_params = {'deriv_threshold': 1.0                     # threshold
                 }

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        fig.tight_layout()

    def compute_initial_figure(self):
        pass

    def plot_tail_array(self, array, bouts=None, maxes_y=None, maxes_x=None, mins_y=None, mins_x=None, freqs=None, keep_xlim=False):
        self.axes.cla()

        if keep_xlim:
            xlim = self.axes.get_xlim()

        self.axes.plot(array)

        if freqs != None:
            self.axes.plot(freqs, 'k')

        if maxes_y != None:
            self.axes.plot(maxes_x, maxes_y, 'g.')
            self.axes.plot(mins_x, mins_y, 'r.')

        if bouts != None:
            for i in range(bouts.shape[0]):
                self.axes.axvspan(bouts[i, 0], bouts[i, 1], color='red', alpha=0.2)

        if keep_xlim:
            self.axes.set_xlim(xlim)

        self.draw()

    def plot_head_array(self, array, keep_xlim=False):
        self.axes.cla()

        if keep_xlim:
            xlim = self.axes.get_xlim()

        self.axes.plot(array)

        if keep_xlim:
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

        self.smoothing_window_width = 10
        self.threshold = 0.01
        self.min_width = 30

        # add file menu
        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        # create main widget & layout
        self.main_widget = QtGui.QWidget(self)
        main_layout = QtGui.QVBoxLayout(self.main_widget)
        main_layout.addStretch(1)

        # create tabs widget & layout
        self.plot_tabs_widget = QtGui.QTabWidget(self.main_widget)
        plot_tabs_layout  = QtGui.QVBoxLayout(self.plot_tabs_widget)

        # create tail tab widget & layout
        tail_tab_widget = QtGui.QWidget()
        tail_tab_layout = QtGui.QVBoxLayout(tail_tab_widget)

        # create tail plot canvas & toolbar
        self.tail_canvas = MplCanvas(tail_tab_widget, width=10, height=10, dpi=75)
        self.tail_toolbar = NavigationToolbar(self.tail_canvas, self)
        self.tail_canvas.resize(400, 300)
        # self.tail_toolbar.hide()

        # add tail plot canvas & toolbar to tail tab widget
        tail_tab_layout.addWidget(self.tail_toolbar)
        tail_tab_layout.addWidget(self.tail_canvas)

        # create button layout for tail tab
        tail_button_layout = QtGui.QHBoxLayout()
        tail_tab_layout.addLayout(tail_button_layout)

        # add buttons
        self.track_bouts_button = QtGui.QPushButton('Track Bouts', self)
        self.track_bouts_button.setMinimumHeight(30)
        self.track_bouts_button.setMaximumWidth(100)
        self.track_bouts_button.clicked.connect(lambda:self.track_bouts())
        tail_button_layout.addWidget(self.track_bouts_button)

        self.track_freqs_button = QtGui.QPushButton('Track Freq', self)
        self.track_freqs_button.setMinimumHeight(30)
        self.track_freqs_button.setMaximumWidth(100)
        self.track_freqs_button.clicked.connect(lambda:self.track_freqs())
        tail_button_layout.addWidget(self.track_freqs_button)

        # add checkbox for switching plots
        self.smoothed_deriv_checkbox = QtGui.QCheckBox("Show smoothed deriv")
        self.smoothed_deriv_checkbox.toggled.connect(lambda:self.show_smoothed_deriv(self.smoothed_deriv_checkbox))
        tail_button_layout.addWidget(self.smoothed_deriv_checkbox)

        # add param labels & textboxes
        smoothing_window_label = QtGui.QLabel()
        smoothing_window_label.setText("Smoothing Window: =")
        tail_button_layout.addWidget(smoothing_window_label)

        self.smoothing_window_param_box = QtGui.QLineEdit(self)
        self.smoothing_window_param_box.setMinimumHeight(10)
        self.smoothing_window_param_box.setText(str(self.smoothing_window_width))
        tail_button_layout.addWidget(self.smoothing_window_param_box)

        threshold_label = QtGui.QLabel()
        threshold_label.setText("Threshold: =")
        tail_button_layout.addWidget(threshold_label)

        self.threshold_param_box = QtGui.QLineEdit(self)
        self.threshold_param_box.setMinimumHeight(10)
        self.threshold_param_box.setText(str(self.threshold))
        tail_button_layout.addWidget(self.threshold_param_box)

        min_width_label = QtGui.QLabel()
        min_width_label.setText("Min width: =")
        tail_button_layout.addWidget(min_width_label)

        self.min_width_param_box = QtGui.QLineEdit(self)
        self.min_width_param_box.setMinimumHeight(10)
        self.min_width_param_box.setText(str(self.min_width))
        tail_button_layout.addWidget(self.min_width_param_box)

        # create head tab widget & layout
        head_tab_widget = QtGui.QWidget()
        head_plot_layout = QtGui.QVBoxLayout(head_tab_widget)

        # create head plot canvas & toolbar
        self.head_canvas = MplCanvas(head_tab_widget, width=10, height=10, dpi=75)
        self.head_toolbar = NavigationToolbar(self.head_canvas, self)
        # self.head_toolbar.hide()

        # add head plot canvas & toolbar to head tab widget
        head_plot_layout.addWidget(self.head_toolbar)
        head_plot_layout.addWidget(self.head_canvas)

        # create button layout for head tab
        head_button_layout = QtGui.QHBoxLayout()
        head_plot_layout.addLayout(head_button_layout)

        # add buttons
        self.track_position_button = QtGui.QPushButton('Track Pos', self)
        self.track_position_button.setMinimumHeight(30)
        self.track_position_button.setMaximumWidth(100)
        self.track_position_button.clicked.connect(lambda:self.track_position())
        head_button_layout.addWidget(self.track_position_button)

        # add checkbox for switching plots
        self.speed_checkbox = QtGui.QCheckBox("Show speed")
        self.speed_checkbox.toggled.connect(lambda:self.show_speed(self.speed_checkbox))
        head_button_layout.addWidget(self.speed_checkbox)
    
        # set plot tabs widget size
        self.plot_tabs_widget.resize(450, 350)

        # add tabs to plot tabs widget
        self.plot_tabs_widget.addTab(tail_tab_widget,"Tail")
        self.plot_tabs_widget.addTab(head_tab_widget,"Head")

        # create button layout for main widget
        main_button_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(main_button_layout)

        # add buttons
        self.load_data_button = QtGui.QPushButton('Load Data', self)
        self.load_data_button.setMinimumHeight(30)
        self.load_data_button.setMaximumWidth(90)
        self.load_data_button.clicked.connect(lambda:self.load_data())
        main_button_layout.addWidget(self.load_data_button)

        # self.button1 = QtGui.QPushButton('Zoom')
        # self.button1.clicked.connect(self.zoom)
         
        # self.button2 = QtGui.QPushButton('Pan')
        # self.button2.clicked.connect(self.pan)
         
        # self.button3 = QtGui.QPushButton('Home')
        # self.button3.clicked.connect(self.home)
        
        # main_button_layout.addWidget(self.button1)
        # main_button_layout.addWidget(self.button2)
        # main_button_layout.addWidget(self.button3)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

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

        self.head_angle_array = an.get_heading_angle(self.path, plot=False)
        self.tail_angle_array = an.get_tail_angle(self.path, plot=False, heading_angle_array=self.head_angle_array)

        self.tail_canvas.plot_tail_array(self.tail_angle_array)
        self.head_canvas.plot_head_array(self.head_angle_array)

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

        self.plot_tabs_widget.resize(self.frameGeometry().width(), self.frameGeometry().height()-80)

qApp = QtGui.QApplication(sys.argv)

plot_window = PlotWindow()
plot_window.show()

sys.exit(qApp.exec_())
