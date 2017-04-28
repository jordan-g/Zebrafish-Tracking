import sys
import os
import time
import threading
import json

import numpy as np
import cv2
import pdb

# import the Qt library
try:
    from PyQt4.QtCore import Qt, QThread
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QListWidget, QListWidgetItem, QAbstractItemView
except:
    from PyQt5.QtCore import Qt, QThread
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QListWidget, QListWidgetItem, QAbstractItemView

# options for dropdown selectors for interpolation
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

        self.main_layout = QGridLayout(self.main_widget)

        # create left widget & layout
        self.left_widget = QWidget(self)
        self.main_layout.addWidget(self.left_widget, 0, 0)

        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setAlignment(Qt.AlignTop)

        self.media_list = QListWidget(self)
        self.media_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.media_list.currentRowChanged.connect(self.controller.switch_media)
        self.left_layout.addWidget(self.media_list)

        self.media_list_items = []

        self.media_list_buttons = QHBoxLayout()
        self.left_layout.addLayout(self.media_list_buttons)

        self.add_media_button = QPushButton('+')
        self.add_media_button.clicked.connect(self.controller.select_and_open_media)
        self.add_media_button.setToolTip("Add media.")
        self.media_list_buttons.addWidget(self.add_media_button)

        self.remove_media_button = QPushButton('-')
        self.remove_media_button.clicked.connect(self.controller.remove_media)
        self.remove_media_button.setToolTip("Remove selected media.")
        self.media_list_buttons.addWidget(self.remove_media_button)

        # add media switching buttons
        self.prev_media_button = QPushButton('<')
        self.prev_media_button.clicked.connect(self.controller.prev_media)
        self.prev_media_button.setToolTip("Switch to previous loaded media.")
        # self.prev_media_button.setMaximumWidth(50)
        self.media_list_buttons.addWidget(self.prev_media_button)

        self.next_media_button = QPushButton('>')
        self.next_media_button.clicked.connect(self.controller.next_media)
        self.next_media_button.setToolTip("Switch to next loaded media.")
        # self.next_media_button.setMaximumWidth(50)
        self.media_list_buttons.addWidget(self.next_media_button)

        # create right widget & layout
        self.right_widget = QWidget(self)
        self.right_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.main_layout.addWidget(self.right_widget, 0, 1)

        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setAlignment(Qt.AlignTop)
        self.right_layout.addStretch(1)
        self.right_layout.setSpacing(5)

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

    def clear_media_list(self):
        for k in range(len(self.media_list_items)-1, -1, -1):
            self.media_list.takeItem(k)
            del self.media_list_items[k]

    def add_media_item(self, item_name):
        self.media_list_items.append(QListWidgetItem(item_name, self.media_list))

    def remove_media_item(self, item_num):
        if len(self.media_list_items) > 0 and item_num < len(self.media_list_items):
            self.media_list.blockSignals(True)
            self.media_list.takeItem(item_num)
            del self.media_list_items[item_num]
            self.media_list.blockSignals(False)

    def change_selected_media_row(self, row_number):
        self.media_list.blockSignals(True)
        self.media_list.setCurrentRow(row_number)
        self.media_list.blockSignals(False)

    def create_menubar(self):
        # create actions
        self.open_image_action = QAction('Open Image', self)
        self.open_image_action.setShortcut('Ctrl+O')
        self.open_image_action.setStatusTip('Open an image.')
        self.open_image_action.triggered.connect(lambda:self.controller.select_and_open_media("image"))

        self.open_folder_action = QAction('Open Folder', self)
        self.open_folder_action.setShortcut('Ctrl+Shift+O')
        self.open_folder_action.setStatusTip('Open a folder of images.')
        self.open_folder_action.triggered.connect(lambda:self.controller.select_and_open_media("folder"))

        self.open_video_action = QAction('Open Video(s)', self)
        self.open_video_action.setShortcut('Ctrl+Alt+O')
        self.open_video_action.setStatusTip('Open one or more videos.')
        self.open_video_action.triggered.connect(lambda:self.controller.select_and_open_media("video"))

        self.save_params_action = QAction('Save Parameters', self)
        self.save_params_action.setShortcuts(['Ctrl+S'])
        self.save_params_action.setStatusTip('Quick-save the current parameters.')
        self.save_params_action.triggered.connect(self.controller.save_params)

        self.track_frame_action = QAction('Track Frame', self)
        self.track_frame_action.setShortcut('Ctrl+T')
        self.track_frame_action.setStatusTip('Track the currently previewed frame.')
        self.track_frame_action.triggered.connect(self.controller.track_frame)

        self.track_all_action = QAction('Track All Media', self)
        self.track_all_action.setShortcut('Ctrl+Shift+T')
        self.track_all_action.setStatusTip('Track all of the currently loaded media.')
        self.track_all_action.triggered.connect(self.controller.track_media)

        # create menu bar & add actions
        menubar  = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(self.open_image_action)
        file_menu.addAction(self.open_folder_action)
        file_menu.addAction(self.open_video_action)
        file_menu.addAction(self.save_params_action)
        file_menu.addAction(self.track_frame_action)
        file_menu.addAction(self.track_all_action)

        self.open_background_action = QAction('Open Background...', self)
        self.open_background_action.setShortcut('Ctrl+Alt+B')
        self.open_background_action.setStatusTip('Open a saved background.')
        self.open_background_action.triggered.connect(self.controller.load_background)
        self.open_background_action.setEnabled(False)

        self.save_background_action = QAction('Save Background...', self)
        self.save_background_action.setShortcut('Ctrl+B')
        self.save_background_action.setStatusTip('Save the currently calculated background.')
        self.save_background_action.triggered.connect(self.controller.save_background)
        self.save_background_action.setEnabled(False)

        background_menu = menubar.addMenu('&Background')
        background_menu.addAction(self.open_background_action)
        background_menu.addAction(self.save_background_action)

    def create_param_controls_layout(self):
        # initialize parameter controls variable
        self.param_controls = None

        top_layout = QHBoxLayout()
        self.right_layout.addLayout(top_layout)

        # create loaded media label
        self.loaded_media_label = QLabel("No media loaded.")
        top_layout.addWidget(self.loaded_media_label)

        # create tracking progress label
        self.tracking_progress_label = QLabel("---")
        self.right_layout.addWidget(self.tracking_progress_label)

        # create invalid parameters label
        self.invalid_params_label = QLabel("")
        self.invalid_params_label.setStyleSheet("font-weight: bold; color: red;")
        self.right_layout.addWidget(self.invalid_params_label)

        # create form layout for param controls with textboxes
        self.form_layout = QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.right_layout.addLayout(self.form_layout)

        # create dict for storing all parameter controls
        self.param_controls = {}

    def create_param_controls(self, params):
        pass

    def create_main_buttons(self):
        # add button layouts
        button_layout_1 = QHBoxLayout()
        button_layout_1.setSpacing(5)
        button_layout_1.addStretch(1)
        self.right_layout.addLayout(button_layout_1)

        button_layout_2 = QHBoxLayout()
        button_layout_2.setSpacing(5)
        button_layout_2.addStretch(1)
        self.right_layout.addLayout(button_layout_2)

        button_layout_3 = QHBoxLayout()
        button_layout_3.setSpacing(5)
        button_layout_3.addStretch(1)
        self.right_layout.addLayout(button_layout_3)

        # add buttons
        self.save_button = QPushButton(u'\u2713 Save', self)
        self.save_button.setMaximumWidth(80)
        self.save_button.clicked.connect(self.controller.save_params)
        self.save_button.setToolTip("Quick-save the current parameters. Use 'Reload' to load these parameters later.")
        button_layout_1.addWidget(self.save_button)

        self.track_button = QPushButton(u'\u279E Track', self)
        self.track_button.setMaximumWidth(90)
        self.track_button.clicked.connect(self.controller.track_frame)
        self.track_button.setToolTip("Track the currently-previewed frame (for parameter tuning).")
        button_layout_1.addWidget(self.track_button)

        self.track_all_button = QPushButton(u'\u27A0 Track All', self)
        self.track_all_button.setMaximumWidth(180)
        self.track_all_button.clicked.connect(self.controller.track_media)
        self.track_all_button.setStyleSheet("font-weight: bold")
        self.track_all_button.setToolTip("Track all of the currently loaded media with the current parameters.")
        button_layout_1.addWidget(self.track_all_button)

        self.reload_last_save_button = QPushButton(u'\u27AA Reload', self)
        self.reload_last_save_button.setMaximumWidth(90)
        self.reload_last_save_button.clicked.connect(self.controller.load_last_params)
        self.reload_last_save_button.setToolTip("Reload the previously auto-saved parameters.")
        button_layout_2.addWidget(self.reload_last_save_button)

        self.load_params_button = QPushButton(u'Load Params\u2026', self)
        self.load_params_button.setMaximumWidth(180)
        self.load_params_button.clicked.connect(lambda:self.controller.load_params(None))
        self.load_params_button.setToolTip("Load a set of parameters.")
        button_layout_2.addWidget(self.load_params_button)

        self.save_params_button = QPushButton(u'Save Params\u2026', self)
        self.save_params_button.setMaximumWidth(180)
        self.save_params_button.clicked.connect(self.controller.save_params)
        self.save_params_button.setToolTip("Save the current set of parameters.")
        button_layout_2.addWidget(self.save_params_button)

        self.load_background_button = QPushButton(u'Load Background', self)
        self.load_background_button.setMaximumWidth(180)
        self.load_background_button.clicked.connect(self.controller.load_background)
        self.load_background_button.setToolTip("Load a saved background image for background subtraction.")
        button_layout_3.addWidget(self.load_background_button)

        self.save_background_button = QPushButton(u'Save Background', self)
        self.save_background_button.setMaximumWidth(180)
        self.save_background_button.clicked.connect(self.controller.save_background)
        self.save_background_button.setToolTip("Save the calculated background subtraction image.")
        button_layout_3.addWidget(self.save_background_button)

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
            param_box.setText(str(int(default_value)))

        # add to list of crop or global controls
        self.param_controls[label] = param_box

    def add_slider(self, label, description, minimum, maximum, slider_moved_func, value, parent, tick_interval=1, single_step=1, multiplier=1, int_values=False):
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
        if int_values:
            textbox.setText(str(int(value/multiplier)))
        else:
            textbox.setText(str(value/multiplier))
        control_layout.addWidget(textbox)

        # connect slider to set textbox text & update params
        slider.valueChanged.connect(lambda:self.update_textbox_from_slider(slider, textbox, multiplier, int_values))
        slider.valueChanged.connect(slider_moved_func)

        # connect textbox to 
        textbox.editingFinished.connect(lambda:self.update_slider_from_textbox(slider, textbox, multiplier))
        textbox.editingFinished.connect(slider_moved_func)

        # add row to form layout
        parent.addRow(description, control_layout)

        # add to list of controls
        self.param_controls[slider_label]  = slider
        self.param_controls[textbox_label] = textbox

    def update_textbox_from_slider(self, slider, textbox, multiplier=1.0, int_values=False):
        if int_values:
            textbox.setText(str(int(slider.sliderPosition()/multiplier)))
        else:
            textbox.setText(str(slider.sliderPosition()/multiplier))

    def update_slider_from_textbox(self, slider, textbox, multiplier=1.0):
        slider.setValue(float(textbox.text())*multiplier)

    def set_slider_value(self, label, value, slider_scale_factor=1.0, int_values=False):
        # change slider value without sending signals

        slider_label  = label + "_slider"
        textbox_label = label + "_textbox"

        slider = self.param_controls[slider_label]

        if value == None:
            value = slider.minimum()

        slider.blockSignals(True)

        slider.setValue(value*slider_scale_factor)

        slider.blockSignals(False)

        # change textbox value
        textbox = self.param_controls[textbox_label]
        self.update_textbox_from_slider(slider, textbox, multiplier=slider_scale_factor, int_values=int_values)

    def add_checkbox(self, label, description, toggle_func, checked, parent, row=-1, column=-1):
        # make checkbox & add to layout
        checkbox = QCheckBox(description)
        checkbox.setObjectName(label)
        checkbox.setChecked(checked)
        checkbox.clicked.connect(lambda:toggle_func(checkbox))
        if column == -1:
            parent.addWidget(checkbox)
        else:
            parent.addWidget(checkbox, row, column) 

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
        # create layout for checkboxes
        self.checkbox_layout = QGridLayout()
        self.checkbox_layout.setColumnStretch(0, 1)
        self.checkbox_layout.setColumnStretch(1, 1)
        self.right_layout.addLayout(self.checkbox_layout)

        # add checkboxes - (key, description, function to call, initial value, parent layout)
        self.add_checkbox('invert', "Invert image", self.controller.toggle_invert_image, params['invert'], self.checkbox_layout, 0, 0)
        self.add_checkbox('save_video', "Save video", self.controller.toggle_save_video, params['save_video'], self.checkbox_layout, 1, 0)
        self.add_checkbox('subtract_background', 'Subtract background', self.controller.toggle_subtract_background, params['subtract_background'], self.checkbox_layout, 2, 0)
        self.add_checkbox('align_batches', 'Auto alignment', self.controller.toggle_align_batches, params['align_batches'], self.checkbox_layout, 0, 1)
        self.add_checkbox('use_multiprocessing', 'Use multiprocessing', self.controller.toggle_multiprocessing, params['use_multiprocessing'], self.checkbox_layout, 1, 1)
        self.add_checkbox('auto_track', 'Auto track', self.controller.toggle_auto_tracking, params['gui_params']['auto_track'], self.checkbox_layout, 2, 1)

        # add sliders - (key, description, start, end, initial value, parent layout)
        self.add_slider('scale_factor', 'Scale factor:', 1, 40, self.controller.update_params_from_gui, 10.0*params['scale_factor'], self.form_layout, multiplier=10.0)

        # add textboxes - (key, decription, initial value, parent layout)
        self.add_textbox('saved_video_fps', 'Saved video FPS:', params['saved_video_fps'], self.form_layout)
        self.add_textbox('n_tail_points', '# tail points:', params['n_tail_points'], self.form_layout)

        # add comboboxes
        self.add_combobox('tail_direction', 'Tail direction:', tail_direction_options, params['tail_direction'], self.form_layout)
        self.add_combobox('interpolation', 'Interpolation:', interpolation_options, params['interpolation'], self.form_layout)

    def update_gui_from_params(self, params):
        # update param controls with current parameters
        if self.param_controls != None:
            self.param_controls['invert'].setChecked(params['invert'])
            self.param_controls['save_video'].setChecked(params['save_video'])
            self.param_controls['subtract_background'].setChecked(params['subtract_background'])
            self.param_controls['use_multiprocessing'].setChecked(params['use_multiprocessing'])
            self.param_controls['auto_track'].setChecked(params['gui_params']['auto_track'])

            self.set_slider_value('scale_factor', params['scale_factor'], slider_scale_factor=10)

            self.param_controls['saved_video_fps'].setText(str(params['saved_video_fps']))
            self.param_controls['n_tail_points'].setText(str(params['n_tail_points']))

            self.param_controls['tail_direction'].setCurrentIndex(tail_direction_options.index(params['tail_direction']))
            self.param_controls['interpolation'].setCurrentIndex(interpolation_options.index(params['interpolation']))

class FreeswimmingParamWindow(ParamWindow):
    def __init__(self, controller):
        ParamWindow.__init__(self, controller)

    def create_param_controls(self, params):
        # create layout for checkboxes
        self.checkbox_layout = QGridLayout()
        self.checkbox_layout.setColumnStretch(0, 1)
        self.checkbox_layout.setColumnStretch(1, 1)
        self.right_layout.addLayout(self.checkbox_layout)

        # add checkboxes - (key, description, function to call, initial value, parent layout)
        self.add_checkbox('invert', "Invert image", self.controller.toggle_invert_image, params['invert'], self.checkbox_layout, 0, 0)
        self.add_checkbox('show_body_threshold', "Show body threshold", self.controller.toggle_threshold_image, params['gui_params']['show_body_threshold'], self.checkbox_layout, 1, 0)
        self.add_checkbox('show_eye_threshold', "Show eye threshold", self.controller.toggle_threshold_image, params['gui_params']['show_eye_threshold'], self.checkbox_layout, 2, 0)
        self.add_checkbox('show_tail_threshold', "Show tail threshold", self.controller.toggle_threshold_image, params['gui_params']['show_tail_threshold'], self.checkbox_layout, 3, 0)
        self.add_checkbox('show_tail_skeleton', "Show tail skeleton", self.controller.toggle_threshold_image, params['gui_params']['show_tail_skeleton'], self.checkbox_layout, 4, 0)
        self.add_checkbox('track_tail', "Track tail", self.controller.toggle_tail_tracking, params['track_tail'], self.checkbox_layout, 5, 0)
        self.add_checkbox('track_eyes', "Track eyes", self.controller.toggle_eye_tracking, params['track_eyes'], self.checkbox_layout, 6, 0)
        self.add_checkbox('save_video', "Save video", self.controller.toggle_save_video, params['save_video'], self.checkbox_layout, 0, 1)
        self.add_checkbox('adjust_thresholds', 'Adjust thresholds', self.controller.toggle_adjust_thresholds, params['adjust_thresholds'], self.checkbox_layout, 1, 1)
        self.add_checkbox('subtract_background', 'Subtract background', self.controller.toggle_subtract_background, params['subtract_background'], self.checkbox_layout, 2, 1)
        self.add_checkbox('align_batches', 'Auto alignment', self.controller.toggle_align_batches, params['align_batches'], self.checkbox_layout, 3, 1)
        self.add_checkbox('use_multiprocessing', 'Use multiprocessing', self.controller.toggle_multiprocessing, params['use_multiprocessing'], self.checkbox_layout, 4, 1)
        self.add_checkbox('auto_track', 'Auto track', self.controller.toggle_auto_tracking, params['gui_params']['auto_track'], self.checkbox_layout, 5, 1)

        # add sliders - (key, description, start, end, callback function, initial value, parent layout)
        self.add_slider('scale_factor', 'Scale factor:', 1, 40, self.controller.update_params_from_gui, 10.0*params['scale_factor'], self.form_layout, multiplier=10.0)
        self.add_slider('body_crop_height', 'Body crop height:', 1, 100, self.controller.update_params_from_gui, round(params['body_crop'][0]), self.form_layout, tick_interval=10, int_values=True)
        self.add_slider('body_crop_width', 'Body crop width:', 1, 100, self.controller.update_params_from_gui, round(params['body_crop'][1]), self.form_layout, tick_interval=10, int_values=True)

        # add textboxes - (key, decription, initial value, parent layout)
        self.add_textbox('min_tail_body_dist', 'Body/tail min. dist.:', params['min_tail_body_dist'], self.form_layout)
        self.add_textbox('max_tail_body_dist', 'Body/tail max. dist.:', params['max_tail_body_dist'], self.form_layout)
        self.add_textbox('saved_video_fps', 'Saved video FPS:', params['saved_video_fps'], self.form_layout)
        self.add_textbox('n_tail_points', '# tail points:', params['n_tail_points'], self.form_layout)

        # add comboboxes
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

            self.set_slider_value('scale_factor', params['scale_factor'], slider_scale_factor=10)
            self.set_slider_value('body_crop_height', params['body_crop'][0], int_values=True)
            self.set_slider_value('body_crop_width', params['body_crop'][1], int_values=True)

            self.param_controls['min_tail_body_dist'].setText(str(params['min_tail_body_dist']))
            self.param_controls['max_tail_body_dist'].setText(str(params['max_tail_body_dist']))
            self.param_controls['saved_video_fps'].setText(str(params['saved_video_fps']))
            self.param_controls['n_tail_points'].setText(str(params['n_tail_points']))

            self.param_controls['interpolation'].setCurrentIndex(interpolation_options.index(params['interpolation']))