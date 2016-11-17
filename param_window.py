import sys
import os
import time
import threading
import json

import numpy as np
import cv2

# import the Qt library
from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    try:
        from PyQt4.QtCore import Signal, Qt, QThread
        from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
    except:
        from PyQt5.QtCore import Signal, Qt, QThread
        from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
        from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog

# options for dropdown selectors for interpolation
eye_resize_factor_options = [i for i in range(1, 9)]
interpolation_options     = ["Nearest Neighbor", "Linear", "Bicubic", "Lanczos"]
tail_direction_options    = ["Left", "Right", "Up", "Down"]

class ParamWindow(QMainWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # set window title
        self.setWindowTitle("Parameters")

        # set position & size
        self.setGeometry(147, 100, 10, 10)

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addStretch(1)
        self.main_layout.setSpacing(5)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        # create the menu bar
        self.create_menubar()

        # create parameter controls
        self.create_param_controls_layout()
        self.create_param_controls(controller.params)

        # create buttons
        self.create_main_buttons()

        # set window titlebar buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        # disable controls
        self.set_gui_disabled(True)
        
        self.show()

    def create_menubar(self):
        # create actions
        open_image = QAction(QIcon('open.png'), 'Open Image', self)
        open_image.setShortcut('Ctrl+O')
        open_image.setStatusTip('Open an image')
        open_image.triggered.connect(lambda:self.controller.select_and_open_media("image"))

        open_folder = QAction(QIcon('open.png'), 'Open Folder', self)
        open_folder.setShortcut('Ctrl+Shift+O')
        open_folder.setStatusTip('Open a folder of images')
        open_folder.triggered.connect(lambda:self.controller.select_and_open_media("folder"))

        open_video = QAction(QIcon('open.png'), 'Open Video', self)
        open_video.setShortcut('Ctrl+Alt+O')
        open_video.setStatusTip('Open a video')
        open_video.triggered.connect(lambda:self.controller.select_and_open_media("video"))

        track_frame = QAction(QIcon('open.png'), 'Track Frame', self)
        track_frame.setShortcut('Ctrl+T')
        track_frame.setStatusTip('Track current frame')
        track_frame.triggered.connect(self.controller.track_frame)

        save_params = QAction(QIcon('save.png'), 'Save Parameters', self)
        save_params.setShortcuts(['Ctrl+S'])
        save_params.setStatusTip('Save parameters')
        save_params.triggered.connect(self.controller.save_params)

        # create menu bar & add actions
        menubar  = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(open_folder)
        file_menu.addAction(open_video)
        file_menu.addAction(open_image)
        file_menu.addAction(save_params)
        file_menu.addAction(track_frame)

    def create_param_controls_layout(self):
        # initialize parameter controls variable
        self.param_controls = None

        # create invalid parameters label
        self.invalid_params_label = QLabel("")
        self.invalid_params_label.setStyleSheet("font-weight: bold; color: red;")
        self.main_layout.addWidget(self.invalid_params_label)

        # create form layout for param controls with textboxes
        self.form_layout = QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.main_layout.addLayout(self.form_layout)

        # create dict for storing all parameter controls
        self.param_controls = {}

    def create_param_controls(self, params):
        pass

    def create_main_buttons(self):
        # add button layouts
        button_layout_1 = QHBoxLayout()
        button_layout_1.setSpacing(5)
        button_layout_1.addStretch(1)
        self.main_layout.addLayout(button_layout_1)

        button_layout_2 = QHBoxLayout()
        button_layout_2.setSpacing(5)
        button_layout_2.addStretch(1)
        self.main_layout.addLayout(button_layout_2)

        button_layout_3 = QHBoxLayout()
        button_layout_3.setSpacing(5)
        button_layout_3.addStretch(1)
        self.main_layout.addLayout(button_layout_3)

        button_layout_4 = QHBoxLayout()
        button_layout_4.setSpacing(5)
        button_layout_4.addStretch(1)
        self.main_layout.addLayout(button_layout_4)

        # add buttons
        self.save_button = QPushButton(u'\u2713 Save', self)
        self.save_button.setMaximumWidth(80)
        self.save_button.clicked.connect(self.controller.save_params)
        button_layout_1.addWidget(self.save_button)

        self.track_button = QPushButton(u'\u279E Track', self)
        self.track_button.setMaximumWidth(80)
        self.track_button.clicked.connect(self.controller.track_frame)
        button_layout_1.addWidget(self.track_button)

        self.track_all_button = QPushButton(u'\u27A0 Track All', self)
        self.track_all_button.setMaximumWidth(180)
        self.track_all_button.clicked.connect(self.controller.track_media)
        self.track_all_button.setStyleSheet("font-weight: bold")
        button_layout_1.addWidget(self.track_all_button)

        self.open_image_button = QPushButton('+ Image', self)
        self.open_image_button.setMaximumWidth(90)
        self.open_image_button.clicked.connect(lambda:self.controller.select_and_open_media("image"))
        button_layout_2.addWidget(self.open_image_button)

        self.open_folder_button = QPushButton('+ Folder', self)
        self.open_folder_button.setMaximumWidth(90)
        self.open_folder_button.clicked.connect(lambda:self.controller.select_and_open_media("folder"))
        button_layout_2.addWidget(self.open_folder_button)

        self.open_video_button = QPushButton('+ Video', self)
        self.open_video_button.setMaximumWidth(90)
        self.open_video_button.clicked.connect(lambda:self.controller.select_and_open_media("video"))
        button_layout_2.addWidget(self.open_video_button)

        self.reload_last_save_button = QPushButton(u'\u27AA Reload', self)
        self.reload_last_save_button.setMaximumWidth(90)
        self.reload_last_save_button.clicked.connect(self.controller.load_last_params)
        button_layout_3.addWidget(self.reload_last_save_button)

        self.load_params_button = QPushButton(u'Load Params\u2026', self)
        self.load_params_button.setMaximumWidth(180)
        self.load_params_button.clicked.connect(self.controller.load_params)
        button_layout_3.addWidget(self.load_params_button)

        self.save_params_button = QPushButton(u'Save Params\u2026', self)
        self.save_params_button.setMaximumWidth(180)
        self.save_params_button.clicked.connect(self.controller.save_params)
        button_layout_3.addWidget(self.save_params_button)

        self.load_background_button = QPushButton(u'Load Background', self)
        self.load_background_button.setMaximumWidth(180)
        self.load_background_button.clicked.connect(self.controller.load_background)
        button_layout_4.addWidget(self.load_background_button)

        self.save_background_button = QPushButton(u'Save Background', self)
        self.save_background_button.setMaximumWidth(180)
        self.save_background_button.clicked.connect(self.controller.save_background)
        button_layout_4.addWidget(self.save_background_button)

        self.toggle_analysis_window_button = QPushButton(u'Analyse\u2026', self)
        self.toggle_analysis_window_button.setMaximumWidth(180)
        self.toggle_analysis_window_button.clicked.connect(self.controller.toggle_analysis_window)
        button_layout_4.addWidget(self.toggle_analysis_window_button)

    def set_gui_disabled(self, disabled_bool):
        for param_control in self.param_controls.values():
            # don't enable "Subtract background" checkbox when enabling the gui
            if param_control.objectName() == "subtract_background" and disabled_bool == False:
                pass
            else:
                param_control.setDisabled(disabled_bool)

        self.save_button.setDisabled(disabled_bool)
        self.track_button.setDisabled(disabled_bool)
        self.track_all_button.setDisabled(disabled_bool)

    def add_textbox(self, label, description, default_value, parent):
        # make textbox & add row to form layout
        param_box = QLineEdit()
        param_box.setObjectName(label)
        param_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        parent.addRow(description, param_box)

        # set default text
        if default_value != None:
            param_box.setText(str(default_value))

        # add to list of crop or global controls
        self.param_controls[label] = param_box

    def add_slider(self, label, description, minimum, maximum, slider_moved_func, value, parent, tick_interval=1, single_step=1, multiplier=1):
        # make layout to hold slider and textbox
        control_layout = QHBoxLayout()

        slider_label  = label + "_slider"
        textbox_label = label + "_textbox"

        # make slider & add to layout
        slider = QSlider(Qt.Horizontal)
        slider.setObjectName(slider_label)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(tick_interval)
        slider.setSingleStep(single_step)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        control_layout.addWidget(slider)

        # make textbox & add to layout
        textbox = QLineEdit()
        textbox.setObjectName(textbox_label)
        textbox.setFixedWidth(40)
        textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        textbox.setText(str(value/multiplier))
        control_layout.addWidget(textbox)

        # connect slider to set textbox text & update params
        slider.valueChanged.connect(lambda:self.update_textbox_from_slider(slider, textbox, multiplier))
        slider.valueChanged.connect(slider_moved_func)

        # connect textbox to 
        textbox.editingFinished.connect(lambda:self.update_slider_from_textbox(slider, textbox, multiplier))
        textbox.editingFinished.connect(slider_moved_func)

        # add row to form layout
        parent.addRow(description, control_layout)

        # add to list of controls
        self.param_controls[slider_label]  = slider
        self.param_controls[textbox_label] = textbox

    def update_textbox_from_slider(self, slider, textbox, multiplier=1.0):
        textbox.setText(str(slider.sliderPosition()/multiplier))

    def update_slider_from_textbox(self, slider, textbox, multiplier=1.0):
        slider.setValue(float(textbox.text())*multiplier)

    def set_slider_value(self, label, value, slider_scale_factor=None):
        # change slider value without sending signals

        slider_label  = label + "_slider"
        textbox_label = label + "_textbox"

        slider = self.param_controls[slider_label]

        if value == None:
            value = slider.minimum()

        slider.blockSignals(True)

        if slider_scale_factor != None:
            slider.setValue(value*slider_scale_factor)
        else:
            slider.setValue(value)

        slider.blockSignals(False)

        # change textbox value
        textbox = self.param_controls[textbox_label]
        textbox.setText(str(float(value)))

    def add_checkbox(self, label, description, toggle_func, checked, parent):
        # make checkbox & add to layout
        checkbox = QCheckBox(description)
        checkbox.setObjectName(label)
        checkbox.setChecked(checked)
        checkbox.clicked.connect(lambda:toggle_func(checkbox))
        parent.addWidget(checkbox)

        # add to list of crop or global controls
        self.param_controls[label] = checkbox

    def add_combobox(self, label, description, options, value, parent):
        combobox = QComboBox()
        combobox.setObjectName(label)
        combobox.addItems([ str(o) for o in options])
        combobox.setCurrentIndex(options.index(value))
        parent.addRow(description, combobox)

        self.param_controls[label] = combobox

    def show_invalid_params_text(self):
        self.invalid_params_label.setText("Invalid parameters.")

    def hide_invalid_params_text(self):
        self.invalid_params_label.setText("")

    def closeEvent(self, event):
        self.controller.close_all()

class HeadfixedParamWindow(ParamWindow):
    def __init__(self, controller):
        ParamWindow.__init__(self, controller)

    def create_param_controls(self, params):
        # add checkboxes - (key, description, function to call, initial value, parent layout)
        self.add_checkbox('invert', "Invert image", self.controller.toggle_invert_image, params['invert'], self.main_layout)
        self.add_checkbox('save_video', "Save video", self.controller.toggle_save_video, params['save_video'], self.main_layout)
        self.add_checkbox('subtract_background', 'Subtract background', self.controller.toggle_subtract_background, params['subtract_background'], self.main_layout)
        self.add_checkbox('use_multiprocessing', 'Use multiprocessing', self.controller.toggle_multiprocessing, params['use_multiprocessing'], self.main_layout)
        self.add_checkbox('auto_track', 'Auto track', self.controller.toggle_auto_tracking, params['gui_params']['auto_track'], self.main_layout)

        # add sliders - (key, description, start, end, initial value, parent layout)
        self.add_slider('shrink_factor', 'Shrink factor:', 1, 10, self.controller.update_params_from_gui, 10.0*params['shrink_factor'], self.form_layout, multiplier=10.0)

        # add textboxes - (key, decription, initial value, parent layout)
        self.add_textbox('saved_video_fps', 'Saved video FPS:', params['saved_video_fps'], self.form_layout)
        self.add_textbox('n_tail_points', '# tail points:', params['n_tail_points'], self.form_layout)

        # add comboboxes
        self.add_combobox('tail_direction', 'Tail direction:', tail_direction_options, params['tail_direction'], self.form_layout)

    def update_gui_from_params(self, params):
        # update param controls with current parameters
        if self.param_controls != None:
            self.param_controls['invert'].setChecked(params['invert'])
            self.param_controls['save_video'].setChecked(params['save_video'])
            self.param_controls['subtract_background'].setChecked(params['subtract_background'])
            self.param_controls['use_multiprocessing'].setChecked(params['use_multiprocessing'])
            self.param_controls['auto_track'].setChecked(params['gui_params']['auto_track'])

            self.set_slider_value('shrink_factor', params['shrink_factor'], slider_scale_factor=10)

            self.param_controls['saved_video_fps'].setText(str(params['saved_video_fps']))
            self.param_controls['n_tail_points'].setText(str(params['n_tail_points']))

            self.param_controls['tail_direction'].setCurrentIndex(tail_direction_options.index(params['tail_direction']))

class FreeswimmingParamWindow(ParamWindow):
    def __init__(self, controller):
        ParamWindow.__init__(self, controller)

    def create_param_controls(self, params):
        # add checkboxes - (key, description, function to call, initial value, parent layout)
        self.add_checkbox('invert', "Invert image", self.controller.toggle_invert_image, params['invert'], self.main_layout)
        self.add_checkbox('show_body_threshold', "Show body threshold", self.controller.toggle_threshold_image, params['gui_params']['show_body_threshold'], self.main_layout)
        self.add_checkbox('show_eye_threshold', "Show eye threshold", self.controller.toggle_threshold_image, params['gui_params']['show_eye_threshold'], self.main_layout)
        self.add_checkbox('show_tail_threshold', "Show tail threshold", self.controller.toggle_threshold_image, params['gui_params']['show_tail_threshold'], self.main_layout)
        self.add_checkbox('show_tail_skeleton', "Show tail skeleton", self.controller.toggle_threshold_image, params['gui_params']['show_tail_skeleton'], self.main_layout)
        self.add_checkbox('track_tail', "Track tail", self.controller.toggle_tail_tracking, params['track_tail'], self.main_layout)
        self.add_checkbox('track_eyes', "Track eyes", self.controller.toggle_eye_tracking, params['track_eyes'], self.main_layout)
        self.add_checkbox('save_video', "Save video", self.controller.toggle_save_video, params['save_video'], self.main_layout)
        self.add_checkbox('adjust_thresholds', 'Adjust thresholds', self.controller.toggle_adjust_thresholds, params['adjust_thresholds'], self.main_layout)
        self.add_checkbox('subtract_background', 'Subtract background', self.controller.toggle_subtract_background, params['subtract_background'], self.main_layout)
        self.add_checkbox('use_multiprocessing', 'Use multiprocessing', self.controller.toggle_multiprocessing, params['use_multiprocessing'], self.main_layout)
        self.add_checkbox('auto_track', 'Auto track', self.controller.toggle_auto_tracking, params['gui_params']['auto_track'], self.main_layout)

        # add sliders - (key, description, start, end, initial value, parent layout)
        self.add_slider('shrink_factor', 'Shrink factor:', 1, 10, self.controller.update_params_from_gui, 10.0*params['shrink_factor'], self.form_layout, multiplier=10.0)
        self.add_slider('tail_crop_height', 'Tail crop height:', 1, 100, self.controller.update_params_from_gui, round(params['tail_crop'][0]), self.form_layout, tick_interval=10)
        self.add_slider('tail_crop_width', 'Tail crop width:', 1, 100, self.controller.update_params_from_gui, round(params['tail_crop'][1]), self.form_layout, tick_interval=10)

        # add textboxes - (key, decription, initial value, parent layout)
        self.add_textbox('min_tail_body_dist', 'Body/tail min. dist.:', params['min_tail_body_dist'], self.form_layout)
        self.add_textbox('max_tail_body_dist', 'Body/tail max. dist.:', params['max_tail_body_dist'], self.form_layout)
        self.add_textbox('saved_video_fps', 'Saved video FPS:', params['saved_video_fps'], self.form_layout)
        self.add_textbox('n_tail_points', '# tail points:', params['n_tail_points'], self.form_layout)

        # add comboboxes
        self.add_combobox('eye_resize_factor', 'Resize factor for eyes:', eye_resize_factor_options, params['eye_resize_factor'], self.form_layout)
        self.add_combobox('interpolation', 'Interpolation:', interpolation_options, params['interpolation'], self.form_layout)

    def update_gui_from_params(self, params):
        # update param controls with current parameters
        if self.param_controls != None:
            self.param_controls['invert'].setChecked(params['invert'])
            self.param_controls['show_body_threshold'].setChecked(params['gui_params']['show_body_threshold'])
            self.param_controls['show_eye_threshold'].setChecked(params['gui_params']['show_eye_threshold'])
            self.param_controls['show_tail_threshold'].setChecked(params['gui_params']['show_tail_threshold'])
            self.param_controls['show_tail_skeleton'].setChecked(params['gui_params']['show_tail_skeleton'])
            self.param_controls['track_tail'].setChecked(params['track_tail'])
            self.param_controls['track_eyes'].setChecked(params['track_eyes'])
            self.param_controls['save_video'].setChecked(params['save_video'])
            self.param_controls['adjust_thresholds'].setChecked(params['adjust_thresholds'])
            self.param_controls['subtract_background'].setChecked(params['subtract_background'])
            self.param_controls['use_multiprocessing'].setChecked(params['use_multiprocessing'])
            self.param_controls['auto_track'].setChecked(params['gui_params']['auto_track'])

            self.set_slider_value('shrink_factor', params['shrink_factor'], slider_scale_factor=10)
            self.set_slider_value('tail_crop_height', params['tail_crop'][0])
            self.set_slider_value('tail_crop_width', params['tail_crop'][1])

            self.param_controls['min_tail_body_dist'].setText(str(params['min_tail_body_dist']))
            self.param_controls['max_tail_body_dist'].setText(str(params['max_tail_body_dist']))
            self.param_controls['saved_video_fps'].setText(str(params['saved_video_fps']))
            self.param_controls['n_tail_points'].setText(str(params['n_tail_points']))

            self.param_controls['eye_resize_factor'].setCurrentIndex(eye_resize_factor_options.index(params['eye_resize_factor']))
            self.param_controls['interpolation'].setCurrentIndex(interpolation_options.index(params['interpolation']))