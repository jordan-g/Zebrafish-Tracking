import sys
import os
import numpy as np
import pdb
import cv2
import json
import threading
import time

# import the Qt library
try:
    from PyQt4.QtCore import Qt, QThread
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QListWidget, QListWidgetItem, QStackedWidget, QTabBar
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
    pyqt_version = 4
except:
    from PyQt5.QtCore import Qt, QThread
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QListWidget, QListWidgetItem, QStackedWidget, QTabBar

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    pyqt_version = 5

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import analysis as an
import numpy.fft as fft
import scipy
import scipy.stats
import peakdetect

default_params = {'deriv_threshold': 1.0, # threshold to use for angle derivative when finding tail bouts
                  'tracking_paths': [] }

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=3, height=3, dpi=75):
        # make figure
        fig = Figure(figsize=(width, height), dpi=dpi)

        # make axis
        self.axes = fig.add_subplot(111)

        # run init of superclass
        FigureCanvas.__init__(self, fig)

        # set background to match window background
        fig.patch.set_visible(False)

        # set tight layout
        fig.tight_layout()

        # set parent
        self.setParent(parent)

        # set resize policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def plot_tail_angle_array(self, array, extra_tracking=None, keep_xlim=True):
        self.axes.cla()

        if keep_xlim:
            # store existing xlim
            xlim = self.axes.get_xlim()

        # plot data
        self.axes.plot(array)

        if extra_tracking != None:
            bouts         = extra_tracking['bouts']
            peak_points   = extra_tracking['peak_points']
            valley_points = extra_tracking['valley_points']
            frequencies   = extra_tracking['frequencies']

            if frequencies is not None:
                # overlay frequencies
                self.axes.plot(frequencies, 'k')

            if peak_points is not None and valley_points is not None:
                # overlay peak & valley points
                self.axes.plot(peak_points[0], peak_points[1], 'g.')
                self.axes.plot(valley_points[0], valley_points[1], 'r.')

            if bouts != None:
                # overlay bouts
                for i in range(bouts.shape[0]):
                    self.axes.axvspan(bouts[i, 0], bouts[i, 1], color='red', alpha=0.2)

        if keep_xlim:
            # restore previous xlim
            self.axes.set_xlim(xlim)

        self.draw()

    def plot_heading_angle_array(self, array, keep_xlim=True):
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

class PlotWindow(QMainWindow):
    def __init__(self, parent, controller):
        # create window
        QMainWindow.__init__(self)
        self.setWindowTitle("Plot")
        self.setGeometry(600, 200, 10, 10)

        # set controller
        self.controller = controller

        # set parent's plot_window to point to self
        parent.plot_window = self

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.main_layout = QVBoxLayout(self.main_widget)

        # create plot canvas
        self.plot_canvas  = MplCanvas(None, width=10, height=5, dpi=75)
        self.main_layout.addWidget(self.plot_canvas)

        # create matplotlib's toolbar but hide it
        self.mpl_plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        self.mpl_plot_toolbar.hide()

        # create toolbar
        self.create_plot_toolbar(self.main_layout)

        # set central widget
        self.setCentralWidget(self.main_widget)

        # set window titlebar buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.show()

    def create_plot_toolbar(self, parent_layout):
        # create buttons
        zoom_button = QPushButton('Zoom')
        zoom_button.clicked.connect(self.mpl_plot_toolbar.zoom)
         
        pan_button = QPushButton('Pan')
        pan_button.clicked.connect(self.mpl_plot_toolbar.pan)
         
        home_button = QPushButton('Home')
        home_button.clicked.connect(self.mpl_plot_toolbar.home)

        save_button = QPushButton('Save')
        save_button.clicked.connect(self.mpl_plot_toolbar.save_figure)

        # create button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)
        button_layout.addStretch(1)

        # add buttons to button layout
        button_layout.addWidget(zoom_button)
        button_layout.addWidget(pan_button)
        button_layout.addWidget(home_button)
        button_layout.addWidget(save_button)

        # add button layout to parent layout
        parent_layout.addLayout(button_layout)

    def plot_tail_angle_array(self, array, extra_tracking=None, keep_xlim=True):
        self.plot_canvas.plot_tail_angle_array(array, extra_tracking, keep_xlim)

    def plot_heading_angle_array(self, array, keep_xlim=True):
        print(array.shape)
        self.plot_canvas.plot_heading_angle_array(array, keep_xlim)

class AnalysisWindow(QMainWindow):
    def __init__(self, parent, controller):
        # create window
        QMainWindow.__init__(self)

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Tracking Analysis")
        self.setGeometry(100, 200, 10, 10)

        # set controller
        self.controller = controller

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.main_layout = QGridLayout(self.main_widget)

        # create left widget & layout
        self.left_widget = QWidget(self)
        self.main_layout.addWidget(self.left_widget, 0, 0)

        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setAlignment(Qt.AlignTop)

        # create list of tracking items
        self.tracking_list_items = []
        self.tracking_list = QListWidget(self)
        self.tracking_list.currentRowChanged.connect(self.controller.switch_tracking_file)
        self.left_layout.addWidget(self.tracking_list)

        # create tracking list buttons
        self.tracking_list_buttons = QHBoxLayout(self)
        self.left_layout.addLayout(self.tracking_list_buttons)

        self.add_tracking_button = QPushButton('+')
        self.add_tracking_button.clicked.connect(self.controller.select_and_open_tracking_files)
        self.add_tracking_button.setToolTip("Add tracking file.")
        self.tracking_list_buttons.addWidget(self.add_tracking_button)

        self.remove_tracking_button = QPushButton('-')
        self.remove_tracking_button.clicked.connect(self.controller.remove_tracking_file)
        self.remove_tracking_button.setToolTip("Remove selected tracking file.")
        self.tracking_list_buttons.addWidget(self.remove_tracking_button)

        self.prev_tracking_button = QPushButton('<')
        self.prev_tracking_button.clicked.connect(self.controller.prev_tracking_file)
        self.prev_tracking_button.setToolTip("Switch to previous tracking file.")
        self.tracking_list_buttons.addWidget(self.prev_tracking_button)

        self.next_tracking_button = QPushButton('>')
        self.next_tracking_button.clicked.connect(self.controller.next_tracking_file)
        self.next_tracking_button.setToolTip("Switch to next tracking file.")
        self.tracking_list_buttons.addWidget(self.next_tracking_button)

        # create right widget & layout
        self.right_widget = QWidget(self)
        self.right_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.main_layout.addWidget(self.right_widget, 0, 1)
        
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setAlignment(Qt.AlignTop)
        self.right_layout.setSpacing(5)

        # create button layout for main widget
        plot_horiz_layout = QHBoxLayout()
        self.right_layout.addLayout(plot_horiz_layout)

        # add param labels & textboxes
        plot_type_label = QLabel()
        plot_type_label.setText("Plot:")
        plot_horiz_layout.addWidget(plot_type_label)
        plot_horiz_layout.addStretch(1)

        # create tab widget for plot type
        self.plot_tabs_widget = QTabBar()
        self.plot_tabs_widget.setDrawBase(False)
        self.plot_tabs_widget.setExpanding(False)
        self.plot_tabs_widget.currentChanged.connect(self.controller.change_plot_type)
        plot_horiz_layout.addWidget(self.plot_tabs_widget)

        # create button layout for main widget
        crop_horiz_layout = QHBoxLayout()
        self.right_layout.addLayout(crop_horiz_layout)

        # add param labels & textboxes
        crop_type_label = QLabel()
        crop_type_label.setText("Crop #:")
        crop_horiz_layout.addWidget(crop_type_label)
        crop_horiz_layout.addStretch(1)

        # create tab widget for crop number
        self.crop_tabs_widget = QTabBar()
        self.crop_tabs_widget.setDrawBase(False)
        self.crop_tabs_widget.setExpanding(False)
        self.crop_tabs_widget.currentChanged.connect(self.controller.change_crop)
        crop_horiz_layout.addWidget(self.crop_tabs_widget)

        # create button layout for main widget
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.right_layout.addLayout(button_layout)

        # add buttons
        self.show_tracking_params_button = QPushButton('Tracking Parameters', self)
        self.show_tracking_params_button.setMinimumHeight(30)
        self.show_tracking_params_button.clicked.connect(self.controller.show_tracking_params)
        button_layout.addWidget(self.show_tracking_params_button)

        # create stacked widget & layout
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.right_layout.addWidget(self.stacked_widget)

        self.create_tail_tracking_widget(self.stacked_widget)
        self.create_body_tracking_widget(self.stacked_widget)

        # self.right_layout = QVBoxLayout(self.right_widget)
        # self.right_layout.setAlignment(Qt.AlignTop)
        # self.right_layout.setSpacing(5)

        # self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        # set window titlebar buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.show()

    def update_plot(self, array, plot_type, extra_tracking=None, keep_xlim=True):
        print("Updating plot")

        if plot_type == "tail":
            self.stacked_widget.setCurrentIndex(0)
            self.plot_window.plot_tail_angle_array(array, extra_tracking=extra_tracking, keep_xlim=keep_xlim)
        elif plot_type == "body":
            self.stacked_widget.setCurrentIndex(1)
            self.plot_window.plot_heading_angle_array(array, keep_xlim=keep_xlim)
        else:
            pass

    def switch_tracking_item(self, row_number):
        print("Switching tracking item")

        tracking_params = self.controller.tracking_params[row_number]

        self.change_selected_tracking_row(row_number)

        self.plot_tabs_widget.blockSignals(True)

        # add plot tabs
        for i in range(self.plot_tabs_widget.count()-1, -1, -1):
            self.plot_tabs_widget.removeTab(i)

        if tracking_params['type'] == "freeswimming":
            if tracking_params['track_tail']:
                self.plot_tabs_widget.addTab("Tail")

            self.plot_tabs_widget.addTab("Body")

            if tracking_params['track_eyes']:
                self.plot_tabs_widget.addTab("Eyes")
        else:
            self.plot_tabs_widget.addTab("Tail")

        self.plot_tabs_widget.blockSignals(False)

        self.crop_tabs_widget.blockSignals(True)
        for i in range(self.crop_tabs_widget.count()-1, -1, -1):
            self.crop_tabs_widget.removeTab(i)

        # add crop tabs
        n_crops = len(tracking_params['crop_params'])

        for i in range(n_crops):
            self.crop_tabs_widget.addTab("{}".format(i+1))
        self.crop_tabs_widget.blockSignals(False)

    def add_tracking_item(self, item_name):
        print("Adding tracking item")
        self.tracking_list_items.append(QListWidgetItem(item_name, self.tracking_list))

        # self.update_plot()

    def change_selected_tracking_row(self, row_number):
        self.tracking_list.blockSignals(True)
        self.tracking_list.setCurrentRow(row_number)
        self.tracking_list.blockSignals(False)

    def create_tail_tracking_widget(self, parent_widget):
        # create tail tab widget & layout
        tail_tab_widget = QWidget()
        tail_tab_layout = QVBoxLayout(tail_tab_widget)

        # create button layout for tail tab
        bottom_tail_button_layout = QVBoxLayout()
        # bottom_tail_button_layout.setSpacing(5)
        bottom_tail_button_layout.addStretch(1)
        tail_tab_layout.addLayout(bottom_tail_button_layout)

        # add buttons
        track_bouts_button = QPushButton('Track Bouts', self)
        track_bouts_button.setMinimumHeight(30)
        track_bouts_button.setMaximumWidth(100)
        track_bouts_button.clicked.connect(lambda:self.controller.track_bouts())
        bottom_tail_button_layout.addWidget(track_bouts_button)

        track_freqs_button = QPushButton('Track Freq', self)
        track_freqs_button.setMinimumHeight(30)
        track_freqs_button.setMaximumWidth(100)
        track_freqs_button.clicked.connect(lambda:self.controller.track_freqs())
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
        self.smoothing_window_param_box.setMaximumWidth(40)
        self.smoothing_window_param_box.setText(str(self.controller.smoothing_window_width))
        bottom_tail_button_layout.addWidget(self.smoothing_window_param_box)

        threshold_label = QLabel()
        threshold_label.setText("Threshold:")
        bottom_tail_button_layout.addWidget(threshold_label)

        self.threshold_param_box = QLineEdit(self)
        self.threshold_param_box.setMinimumHeight(20)
        self.threshold_param_box.setMaximumWidth(40)
        self.threshold_param_box.setText(str(self.controller.threshold))
        bottom_tail_button_layout.addWidget(self.threshold_param_box)

        min_width_label = QLabel()
        min_width_label.setText("Min width:")
        bottom_tail_button_layout.addWidget(min_width_label)

        self.min_width_param_box = QLineEdit(self)
        self.min_width_param_box.setMinimumHeight(20)
        self.min_width_param_box.setMaximumWidth(40)
        self.min_width_param_box.setText(str(self.controller.min_width))
        bottom_tail_button_layout.addWidget(self.min_width_param_box)

        parent_widget.addWidget(tail_tab_widget)

    def create_body_tracking_widget(self, parent_widget):
        # create head tab widget & layout
        head_tab_widget = QWidget()
        head_tab_layout = QVBoxLayout(head_tab_widget)

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

        parent_widget.addWidget(head_tab_widget)

    def create_crops(self, parent_layout):
        crop_tabs_widget = QTabWidget()
        crop_tabs_widget.currentChanged.connect(self.change_crop)
        crop_tabs_widget.setElideMode(Qt.ElideLeft)
        crop_tabs_layout = QVBoxLayout(crop_tabs_widget)
        parent_layout.addWidget(crop_tabs_widget)

        self.crop_tabs_widgets.append(crop_tabs_widget)
        self.crop_tabs_layouts.append(crop_tabs_layout)

        n_crops = len(self.controller.tracking_params[self.controller.curr_tracking_num]['crop_params'])

        for k in range(n_crops):
            self.create_crop()

    def clear_crops(self):
        self.crop_tab_layouts  = [[]]
        self.crop_tab_widgets  = [[]]
        self.plot_tab_layouts  = [{'tail': [],
                                  'eyes': [],
                                  'body': []}]
        self.plot_tab_widgets  = [{'tail': [],
                                  'eyes': [],
                                  'body': []}]
        self.plot_tabs_widgets = [[]]
        self.plot_tabs_layouts = [[]]

        self.head_angle_arrays = []
        self.tail_angle_arrays = []

        for c in range(self.n_crops-1, -1, -1):
            # remove tab
            self.crop_tabs_widget.removeTab(c)

        self.n_crops = 0
        self.current_crop = -1

    def show_smoothed_deriv(self, checkbox):
        if self.smoothed_abs_deriv_abs_angle_array != None:
            if checkbox.isChecked():
                self.tail_canvas.plot_tail_angle_array(self.smoothed_abs_deriv_abs_angle_array, self.bouts, keep_limits=True)
            else:
                self.tail_canvas.plot_tail_angle_array(self.tail_end_angle_array[self.current_crop], self.bouts, self.peak_maxes_y, self.peak_maxes_x, self.peak_mins_y, self.peak_mins_x, self.freqs, keep_limits=True)

    def show_speed(self, checkbox):
        if self.speed_array != None:
            if checkbox.isChecked():
                self.head_canvas.plot_head_array(self.speed_array, keep_limits=True)
            else:
                self.head_canvas.plot_head_array(self.head_angle_array, keep_limits=True)

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
            # self.tail_end_angle_array = np.mean(self.tail_angle_array[:, :, -3:], axis=-1)
            # self.tail_end_angle_array = self.tail_angle_array[:, :, -1]
            self.tail_end_angle_array = an.get_tail_end_angles(self.tail_angle_array, num_to_average=3)
            
            # clear crops
            self.clear_crops()

            # get number of saved crops
            n_crops_total = len(self.params['crop_params'])

            for k in range(n_crops_total):
                # create a crop
                self.create_crop()

                # plot heading angle
                if self.heading_angle_array is not None:
                    self.plot_canvases[k].plot_heading_angle_array(self.heading_angle_array[k])

                # plot tail angle
                if self.tail_angle_array is not None:
                    self.tail_canvases[k].plot_tail_angle_array(self.tail_end_angle_array[k])

    # def track_bouts(self):
    #     if self.tail_angle_array != None:
    #         # get params
    #         self.smoothing_window_width = int(self.smoothing_window_param_box.text())
    #         self.threshold = float(self.threshold_param_box.text())
    #         self.min_width = int(self.min_width_param_box.text())

    #         # get smoothed derivative
    #         abs_angle_array = np.abs(self.tail_end_angle_array[self.current_crop])
    #         deriv_abs_angle_array = np.gradient(abs_angle_array)
    #         abs_deriv_abs_angle_array = np.abs(deriv_abs_angle_array)
    #         normpdf = scipy.stats.norm.pdf(range(-int(self.smoothing_window_width/2),int(self.smoothing_window_width/2)),0,3)
    #         self.smoothed_abs_deriv_abs_angle_array =  np.convolve(abs_deriv_abs_angle_array,  normpdf/np.sum(normpdf),mode='valid')

    #         # calculate bout periods
    #         self.bouts = an.contiguous_regions(self.smoothed_abs_deriv_abs_angle_array > self.threshold)

    #         # remove bouts that don't have the minimum bout length
    #         for i in range(self.bouts.shape[0]-1, -1, -1):
    #             if self.bouts[i, 1] - self.bouts[i, 0] < self.min_width:
    #                 self.bouts = np.delete(self.bouts, (i), 0)

    #         # update plot
    #         self.smoothed_deriv_checkbox.setChecked(False)
    #         self.tail_canvas.plot_tail_angle_array(self.tail_end_angle_array[self.current_crop], self.bouts, keep_limits=True)

    # def track_freqs(self):
    #     if self.bouts != None:
    #         # initiate bout maxima & minima coord lists
    #         self.peak_maxes_y = []
    #         self.peak_maxes_x = []
    #         self.peak_mins_y = []
    #         self.peak_mins_x = []

    #         # initiate instantaneous frequency array
    #         self.freqs = np.zeros(self.tail_angle_array.shape[0])

    #         for i in range(self.bouts.shape[0]):
    #             # get local maxima & minima
    #             peak_max, peak_min = peakdetect.peakdet(self.tail_end_angle_array[self.current_crop][self.bouts[i, 0]:self.bouts[i, 1]], 0.02)

    #             # change local coordinates (relative to the start of the bout) to global coordinates
    #             peak_max[:, 0] += self.bouts[i, 0]
    #             peak_min[:, 0] += self.bouts[i, 0]

    #             # add to the bout maxima & minima coord lists
    #             self.peak_maxes_y += list(peak_max[:, 1])
    #             self.peak_maxes_x += list(peak_max[:, 0])
    #             self.peak_mins_y += list(peak_min[:, 1])
    #             self.peak_mins_x += list(peak_min[:, 0])

    #         # calculate instantaneous frequencies
    #         for i in range(len(self.peak_maxes_x)-1):
    #             self.freqs[self.peak_maxes_x[i]:self.peak_maxes_x[i+1]] = 1.0/(self.peak_maxes_x[i+1] - self.peak_maxes_x[i])

    #         # update plot
    #         self.smoothed_deriv_checkbox.setChecked(False)
    #         self.tail_canvas.plot_tail_angle_array(self.tail_end_angle_array[self.current_crop], self.bouts, self.peak_maxes_y, self.peak_maxes_x, self.peak_mins_y, self.peak_mins_x, self.freqs, keep_limits=True)

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

    def closeEvent(self, event):
        self.controller.close_all()

    def resizeEvent(self, re):
        QMainWindow.resizeEvent(self, re)

        # self.crop_tab_widgets[self.controller.curr_tracking_num][self.current_crop].resize(self.frameGeometry().width()-80, self.frameGeometry().height()-160)

class Controller():
    def __init__(self, default_params):
        # set parameters
        self.default_params = default_params
        self.params         = self.default_params

        # initialize variables
        self.current_crop         = -1     # which crop is being looked at
        self.current_tracking_num = 0      # which tracking data (from a loaded batch) is being looked at
        self.current_plot_type    = "tail" # which plot type is being looked at ("tail" / "body" / "eyes")

        self.n_frames          = 0    # total number of frames to preview
        self.n                 = 0    # index of currently selected frame
        self.first_load        = True

        # intialize analysis variables
        self.smoothing_window_width = 10
        self.threshold              = 0.01
        self.min_width              = 30

        self.tail_angle_arrays                  = []
        self.heading_angle_arrays               = []
        self.body_position_arrays               = []
        self.eye_position_arrays                = []
        self.tail_end_angle_arrays              = []
        self.tracking_params                    = []
        self.tracking_paths                     = None
        self.smoothed_abs_deriv_abs_angle_array = None
        self.speed_arrays                       = None
        self.bouts                              = None
        self.peak_points                        = None
        self.valley_points                      = None
        self.tail_frequencies                   = None
        self.tracking_params_window             = None

        self.analysis_window = AnalysisWindow(None, self)
        self.plot_window     = PlotWindow(self.analysis_window, self)
        self.tracking_params_window = None

    def select_and_open_tracking_files(self):
        if pyqt_version == 4:
            tracking_paths = QFileDialog.getOpenFileNames(self.analysis_window, 'Select tracking files', '', 'Numpy (*.npz)')
        elif pyqt_version == 5:
            tracking_paths = QFileDialog.getOpenFileNames(self.analysis_window, 'Select tracking files', '', 'Numpy (*.npz)')[0]

            # convert paths to str
            tracking_paths = [ str(tracking_path) for tracking_path in tracking_paths ]

        if len(tracking_paths) > 0 and tracking_paths[0] != '':
            if self.first_load:
                # clear all crops
                self.clear_crops()

                # set params to defaults
                self.params = self.default_params.copy()

                self.current_crop = -1

            self.open_tracking_file_batch(tracking_paths)

            if self.first_load:
                self.first_load = False

    def open_tracking_file_batch(self, tracking_paths):
        self.first_load = self.first_load or len(self.params['tracking_paths']) == 0

        if (self.first_load and len(self.params['tracking_paths']) == 0) or not self.first_load:
            # update tracking paths & type parameters
            self.params['tracking_paths'] += tracking_paths

        self.tail_angle_arrays     += [None]*len(tracking_paths)
        self.heading_angle_arrays  += [None]*len(tracking_paths)
        self.body_position_arrays  += [None]*len(tracking_paths)
        self.eye_position_arrays   += [None]*len(tracking_paths)
        self.tail_end_angle_arrays += [None]*len(tracking_paths)
        self.tracking_params       += [None]*len(tracking_paths)

        if self.first_load:
            # update current tracking data number
            self.current_tracking_num = 0

            # open the first tracking file from the batch
            self.open_tracking_file(tracking_paths[self.current_tracking_num])

        for k in range(len(self.params['tracking_paths']) - len(tracking_paths), len(self.params['tracking_paths'])):
            self.analysis_window.add_tracking_item(os.path.basename(self.params['tracking_paths'][k]))

        self.analysis_window.switch_tracking_item(self.current_tracking_num)
        
    def open_tracking_file(self, tracking_path):
        print("Loading {}".format(tracking_path))

        # load saved tracking data
        (tail_coords_array, spline_coords_array,
         heading_angle_array, body_position_array,
         eye_coords_array, tracking_params) = an.open_saved_data(tracking_path)

        # calculate tail angles
        if tracking_params['type'] == "freeswimming":
            heading_angle_array = an.fix_heading_angles(heading_angle_array)

            if tracking_params['track_tail']:
                tail_angle_array = an.get_freeswimming_tail_angles(tail_coords_array, heading_angle_array, body_position_array)
            else:
                tail_angle_array = None
        elif tracking_params['type'] == "headfixed":
            tail_angle_array = an.get_headfixed_tail_angles(tail_coords_array, tail_angle=tracking_params['tail_angle'], tail_direction=tracking_params['tail_direction'])
        else:
            tail_angle_array = None

        if tail_angle_array != None:
            self.current_plot_type = "tail"

            # get array of average angle of the last few points of the tail
            # tail_end_angle_array = tail_angle_array[:, :, -1]
            tail_end_angle_array = an.get_tail_end_angles(tail_angle_array, num_to_average=3)
            # print(tail_end_angle_array.shape)
        else:
            self.current_plot_type = "body"
            tail_end_angle_array = None

        self.tail_angle_arrays[self.current_tracking_num]     = tail_angle_array
        self.tail_end_angle_arrays[self.current_tracking_num] = tail_end_angle_array
        self.heading_angle_arrays[self.current_tracking_num]  = heading_angle_array
        self.body_position_arrays[self.current_tracking_num]  = body_position_array
        self.eye_position_arrays[self.current_tracking_num]   = eye_coords_array
        self.tracking_params[self.current_tracking_num]       = tracking_params

        if self.current_plot_type == "tail":
            self.plot_array = self.tail_end_angle_arrays[self.current_tracking_num][self.current_crop]
        elif self.current_plot_type == "body":
            self.plot_array = self.heading_angle_arrays[self.current_tracking_num][self.current_crop]
        elif self.current_plot_type == "eyes":
            self.plot_array = self.eye_position_arrays[self.current_tracking_num][self.current_crop]

        self.analysis_window.update_plot(self.plot_array, self.current_plot_type, keep_xlim=False)

    def create_crops(self, n_crops):
        self.analysis_window.create_crops()

    def change_crop(self, index):
        print("Changing crop", index)

        self.current_crop = index

        if self.current_plot_type == "tail":
            self.plot_array = self.tail_end_angle_arrays[self.current_tracking_num][self.current_crop]
        elif self.current_plot_type == "body":
            self.plot_array = self.heading_angle_arrays[self.current_tracking_num][self.current_crop]
        elif self.current_plot_type == "eyes":
            self.plot_array = self.eye_position_arrays[self.current_tracking_num][self.current_crop]

        self.analysis_window.update_plot(self.plot_array, self.current_plot_type)

    def change_plot_type(self, index):
        print("Changing plot type", index)
        if index == 0:
            self.current_plot_type = "tail"
        elif index == 1:
            self.current_plot_type = "body"
        else:
            self.current_plot_type = "eyes"

        if self.current_plot_type == "tail":
            self.plot_array = self.tail_end_angle_arrays[self.current_tracking_num][self.current_crop]
        elif self.current_plot_type == "body":
            self.plot_array = self.heading_angle_arrays[self.current_tracking_num][self.current_crop]
        elif self.current_plot_type == "eyes":
            self.plot_array = self.eye_position_arrays[self.current_tracking_num][self.current_crop]

        self.analysis_window.update_plot(self.plot_array, self.current_plot_type)

    def show_tracking_params(self):
        if self.tracking_params_window != None:
            self.tracking_params_window.close()

        self.tracking_params_window = TrackingParamsWindow(self)
        self.tracking_params_window.show()

        # self.tracking_params_window.create_param_controls(self.params, self.current_crop)

    def clear_crops(self):
        pass

    def remove_tracking_file(self):
        pass

    def prev_tracking_file(self):
        pass

    def next_tracking_file(self):
        pass

    def switch_tracking_file(self, tracking_num):
        if 0 <= tracking_num <= len(self.params['tracking_paths'])-1:
            tracking_paths = self.params['tracking_paths']

            # update current media number
            self.current_tracking_num = tracking_num

            if self.tracking_params[self.current_tracking_num] == None:
                # open the next media from the batch
                self.open_tracking_file(tracking_paths[self.current_tracking_num])

            # self.analysis_window.switch_tracking_widget(tracking_num)
            self.analysis_window.switch_tracking_item(tracking_num)

    def track_bouts(self):
        if len(self.tail_angle_arrays) > 0:
            # get params
            self.smoothing_window_width = int(self.analysis_window.smoothing_window_param_box.text())
            self.threshold = float(self.analysis_window.threshold_param_box.text())
            self.min_width = int(self.analysis_window.min_width_param_box.text())

            # get smoothed derivative
            abs_angle_array = np.abs(self.tail_end_angle_arrays[self.current_tracking_num][self.current_crop])
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

            extra_tracking = { 'bouts': self.bouts,
                               'peak_points': self.peak_points,
                               'valley_points': self.valley_points,
                               'frequencies': self.tail_frequencies }

            # update plot
            self.analysis_window.update_plot(self.plot_array, self.current_plot_type, extra_tracking, keep_limits=True)

    def track_freqs(self):
        if self.bouts != None:
            # initiate bout maxima & minima coord lists
            self.peak_maxes_y = []
            self.peak_maxes_x = []
            self.peak_mins_y = []
            self.peak_mins_x = []

            # initiate instantaneous frequency array
            self.tail_frequencies = np.zeros(self.tail_angle_arrays[self.current_tracking_num].shape[0])

            for i in range(self.bouts.shape[0]):
                # get local maxima & minima
                peak_max, peak_min = peakdetect.peakdet(self.tail_end_angle_arrays[self.current_tracking_num][self.current_crop][self.bouts[i, 0]:self.bouts[i, 1]], 0.02)

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

            extra_tracking = { 'bouts': self.bouts,
                               'peak_points': self.peak_points,
                               'valley_points': self.valley_points,
                               'frequencies': self.tail_frequencies }

            # update plot
            # self.smoothed_deriv_checkbox.setChecked(False)
            # self.tail_canvas.plot_tail_angle_array(self.tail_end_angle_array[self.current_crop], self.bouts, self.peak_maxes_y, self.peak_maxes_x, self.peak_mins_y, self.peak_mins_x, self.freqs, keep_limits=True)
            
            # update plot
            self.analysis_window.update_plot(self.plot_array, self.current_plot_type, extra_tracking=extra_tracking, keep_limits=True)

    def close_all(self):
        self.analysis_window.close()
        self.plot_window.close()

class TrackingParamsWindow(QMainWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

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
        self.create_param_controls(self.controller.tracking_params[self.controller.current_tracking_num], self.controller.current_crop)

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

    def create_param_controls(self, params, current_crop):
        self.add_parameter_label("type", "Tracking type:", params['type'], self.form_layout)
        self.add_parameter_label("shrink_factor", "Shrink factor:", params['shrink_factor'], self.form_layout)
        self.add_parameter_label("invert", "Invert:", params['invert'], self.form_layout)
        self.add_parameter_label("n_tail_points", "Number of tail points:", params['n_tail_points'], self.form_layout)
        self.add_parameter_label("subtract_background", "Subtract background:", params['subtract_background'], self.form_layout)
        self.add_parameter_label("media_types", "Media types:", params['media_types'], self.form_layout)
        self.add_parameter_label("media_paths", "Media paths:", params['media_paths'], self.form_layout)
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

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == "__main__":
    qApp = QApplication(sys.argv)

    controller = Controller(default_params)

    sys.exit(qApp.exec_())
