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
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QListWidget, QListWidgetItem, QAbstractItemView, QFrame
except:
    from PyQt5.QtCore import Qt, QThread
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QListWidget, QListWidgetItem, QAbstractItemView, QFrame

# options for dropdown selectors for interpolation
interpolation_options     = ["Nearest Neighbor", "Linear", "Bicubic", "Lanczos"]
heading_direction_options = ["Down", "Right", "Up", "Left"]

class ParamWindow(QMainWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set tracking mode
        self.tracking_mode = "freeswimming"

        # set controller
        self.controller = controller

        # set window title
        self.setWindowTitle("Parameters")

        # set position & size
        self.setGeometry(147, 100, 10, 10)

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.main_widget.setStyleSheet("font-size: 12px;")

        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # create left widget & layout
        self.left_widget = QWidget(self)
        self.main_layout.addWidget(self.left_widget, 0, 0)

        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_layout.setSpacing(5)

        video_list_label = QLabel("Loaded Videos")
        video_list_label.setStyleSheet("font-weight: bold;")
        self.left_layout.addWidget(video_list_label)

        self.videos_list = QListWidget(self)
        self.videos_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.videos_list.currentRowChanged.connect(self.controller.switch_video)
        self.left_layout.addWidget(self.videos_list)

        self.videos_list_items = []

        self.videos_list_buttons = QHBoxLayout()
        self.videos_list_buttons.setSpacing(5)
        self.left_layout.addLayout(self.videos_list_buttons)

        self.add_videos_button = QPushButton('+')
        self.add_videos_button.clicked.connect(self.controller.select_and_open_videos)
        self.add_videos_button.setToolTip("Add videos to track.")
        self.videos_list_buttons.addWidget(self.add_videos_button)

        self.remove_video_button = QPushButton('-')
        self.remove_video_button.clicked.connect(self.controller.remove_video)
        self.remove_video_button.setToolTip("Remove selected video.")
        self.videos_list_buttons.addWidget(self.remove_video_button)

        # add video switching buttons
        self.prev_video_button = QPushButton('<')
        self.prev_video_button.clicked.connect(self.controller.prev_video)
        self.prev_video_button.setToolTip("Switch to the previous loaded video.")
        self.videos_list_buttons.addWidget(self.prev_video_button)

        self.next_video_button = QPushButton('>')
        self.next_video_button.clicked.connect(self.controller.next_video)
        self.next_video_button.setToolTip("Switch to next loaded video.")
        self.videos_list_buttons.addWidget(self.next_video_button)

        # create freeswimming params widget & layout
        self.right_widget = QWidget(self)
        self.right_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.main_layout.addWidget(self.right_widget, 0, 1)

        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setAlignment(Qt.AlignTop)
        # self.right_layout.addStretch(1)
        self.right_layout.setSpacing(5)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        # create the menu bar
        self.create_menubar()

        # create parameter controls
        self.create_param_controls_layout()
        self.create_param_controls(controller.params)

        # create crops widget
        self.create_crops_widget()

        # create main buttons
        self.create_main_buttons()

        self.videos_loaded_text       = "No videos loaded."
        self.background_progress_text = ""
        self.tracking_progress_text   = ""

        self.create_status_layout()

        # set window titlebar buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)

        # disable controls
        self.set_gui_disabled(True)

        
        self.show()

    def create_status_layout(self):
        status_layout = QHBoxLayout()
        self.main_layout.addLayout(status_layout, 2, 0, 1, 2)
        status_layout.setContentsMargins(5, 5, 5, 5)

        # Create status label
        self.status_label = QLabel()
        self.update_status_text()
        self.status_label.setStyleSheet("font-size: 11px;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch(1)

        self.invalid_params_label = QLabel("")
        self.invalid_params_label.setStyleSheet("font-weight: bold; color: red; font-size: 11px;")
        status_layout.addWidget(self.invalid_params_label)

    def create_crops_widget(self):
        crops_label = QLabel("Crop Parameters")
        crops_label.setStyleSheet("font-weight: bold;")
        self.left_layout.addWidget(crops_label)

        # initialize list of dicts used for accessing all crop parameter controls
        self.crop_param_controls = []

        self.crops_widget = QWidget(self)
        self.crops_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.crops_layout = QVBoxLayout(self.crops_widget)
        self.crops_widget.setMaximumHeight(400)
        self.crops_layout.setContentsMargins(0, 0, 0, 0)
        self.crops_layout.setSpacing(0)

        # create tabs widget
        self.crop_tabs_widget = QTabWidget()
        self.crop_tabs_widget.setUsesScrollButtons(True)
        self.crop_tabs_widget.tabCloseRequested.connect(self.controller.remove_crop)
        self.crop_tabs_widget.currentChanged.connect(self.controller.change_crop)
        self.crop_tabs_widget.setElideMode(Qt.ElideLeft)
        self.crop_tabs_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.crop_tabs_widget.setMinimumSize(10, 100)
        self.crop_tabs_widget.setMaximumHeight(300)
        self.crops_layout.addWidget(self.crop_tabs_widget)

        # create tabs layout
        crop_tabs_layout  = QHBoxLayout(self.crop_tabs_widget)

        # create lists for storing all crop tab widgets & layouts
        self.crop_tab_layouts = []
        self.crop_tab_widgets = []

        # add crop button layout
        crop_button_layout = QHBoxLayout()
        crop_button_layout.setSpacing(5)
        crop_button_layout.addStretch(1)
        self.crops_layout.addLayout(crop_button_layout)

        # add delete crop button
        self.remove_crop_button = QPushButton(u'\u2717 Remove Crop', self)
        # self.remove_crop_button.setMaximumWidth(120)
        self.remove_crop_button.clicked.connect(lambda:self.controller.remove_crop(self.controller.current_crop))
        self.remove_crop_button.setDisabled(True)
        crop_button_layout.addWidget(self.remove_crop_button)

        # add new crop button
        self.create_crop_button = QPushButton(u'\u270E New Crop', self)
        # self.create_crop_button.setMaximumWidth(100)
        self.create_crop_button.clicked.connect(lambda:self.controller.create_crop())
        self.create_crop_button.setDisabled(True)
        crop_button_layout.addWidget(self.create_crop_button)

        self.left_layout.addWidget(self.crops_widget)

    def create_crop_tab(self, crop_params):
        # create crop tab widget & layout
        crop_tab_widget = QWidget(self.crop_tabs_widget)
        crop_tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        crop_tab_widget.resize(276, 300)

        crop_tab_layout = QVBoxLayout(crop_tab_widget)
        crop_tab_layout.setContentsMargins(0, 0, 0, 0)
        crop_tab_layout.setSpacing(5)

        # add to list of crop widgets & layouts
        self.crop_tab_layouts.append(crop_tab_layout)
        self.crop_tab_widgets.append(crop_tab_widget)

        # create form layout
        self.crop_param_form_layout = QFormLayout()
        self.crop_param_form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        crop_tab_layout.addLayout(self.crop_param_form_layout)

        # add dict for storing param controls for this crop
        self.crop_param_controls.append({})

        # add param controls
        self.create_crop_param_controls(crop_params)

        # create crop button layout
        crop_button_layout = QHBoxLayout()
        crop_button_layout.setSpacing(5)
        crop_button_layout.addStretch(1)
        crop_tab_layout.addLayout(crop_button_layout)

        # add crop buttons
        self.reset_crop_button = QPushButton(u'\u25A8 Reset Crop', self)
        # self.reset_crop_button.setMaximumWidth(110)
        self.reset_crop_button.clicked.connect(self.controller.reset_crop)
        crop_button_layout.addWidget(self.reset_crop_button)

        self.select_crop_button = QPushButton(u'\u25A3 Select Crop', self)
        # self.select_crop_button.setMaximumWidth(110)
        self.select_crop_button.clicked.connect(self.controller.select_crop)
        crop_button_layout.addWidget(self.select_crop_button)

        # add crop widget as a tab
        self.crop_tabs_widget.addTab(crop_tab_widget, str(self.controller.current_crop))

        # make this crop the active tab
        self.crop_tabs_widget.setCurrentIndex(self.controller.current_crop)

        # update text on all tabs
        for i in range(len(self.controller.params['crop_params'])):
                self.crop_tabs_widget.setTabText(i, str(i))

    def remove_crop_tab(self, index):
        # remove the tab
        self.crop_tabs_widget.removeTab(index)

        # delete this tab's controls, widget & layout
        del self.crop_param_controls[index]
        del self.crop_tab_widgets[index]
        del self.crop_tab_layouts[index]

        # update text on all tabs
        for i in range(len(self.controller.params['crop_params'])):
            self.crop_tabs_widget.setTabText(i, str(i))

    def create_crop_param_controls(self, crop_params):
        if self.controller.current_frame is not None:
            # add sliders - (key, description, start, end, initial value, parent layout)
            self.add_crop_param_slider('crop_y', 'Crop height:', 1, self.controller.current_frame.shape[0], crop_params['crop'][0], self.controller.update_crop_height, self.crop_param_form_layout, tick_interval=50, int_values=True)
            self.add_crop_param_slider('crop_x', 'Crop width:', 1, self.controller.current_frame.shape[1], crop_params['crop'][1], self.controller.update_crop_width, self.crop_param_form_layout, tick_interval=50, int_values=True)
            self.add_crop_param_slider('offset_y', 'Offset y:', 0, self.controller.current_frame.shape[0]-1, crop_params['offset'][0], self.controller.update_y_offset, self.crop_param_form_layout, tick_interval=50, int_values=True)
            self.add_crop_param_slider('offset_x', 'Offset x:', 0, self.controller.current_frame.shape[1]-1, crop_params['offset'][1], self.controller.update_x_offset, self.crop_param_form_layout, tick_interval=50, int_values=True)
        else:
            self.add_crop_param_slider('crop_y', 'Crop height:', 1, 2, 1, self.controller.update_crop_height, self.crop_param_form_layout, tick_interval=50, int_values=True)
            self.add_crop_param_slider('crop_x', 'Crop width:', 1, 2, 1, self.controller.update_crop_width, self.crop_param_form_layout, tick_interval=50, int_values=True)
            self.add_crop_param_slider('offset_y', 'Offset y:', 0, 1, 0, self.controller.update_y_offset, self.crop_param_form_layout, tick_interval=50, int_values=True)
            self.add_crop_param_slider('offset_x', 'Offset x:', 0, 1, 0, self.controller.update_x_offset, self.crop_param_form_layout, tick_interval=50, int_values=True)
    
    def update_gui_from_crop_params(self, crop_params):
        if len(crop_params) == len(self.crop_param_controls):
            # update crop gui for each crop
            for c in range(len(crop_params)):
                self.crop_param_controls[c]['crop_y' + '_slider'].setMaximum(self.controller.current_frame.shape[0])
                self.crop_param_controls[c]['crop_x' + '_slider'].setMaximum(self.controller.current_frame.shape[1])
                self.crop_param_controls[c]['offset_y' + '_slider'].setMaximum(self.controller.current_frame.shape[0]-1)
                self.crop_param_controls[c]['offset_x' + '_slider'].setMaximum(self.controller.current_frame.shape[1]-1)

                # update param controls with current parameters
                self.set_crop_param_slider_value('crop_y', crop_params[c]['crop'][0], c, int_values=True)
                self.set_crop_param_slider_value('crop_x', crop_params[c]['crop'][1], c, int_values=True)
                self.set_crop_param_slider_value('offset_y', crop_params[c]['offset'][0], c, int_values=True)
                self.set_crop_param_slider_value('offset_x', crop_params[c]['offset'][1], c, int_values=True)

    def clear_videos_list(self):
        for k in range(len(self.videos_list_items)-1, -1, -1):
            self.videos_list.takeItem(k)
            del self.videos_list_items[k]

    def add_video_item(self, item_name):
        self.videos_list_items.append(QListWidgetItem(item_name, self.videos_list))

    def remove_video_item(self, item_num):
        if len(self.videos_list_items) > 0 and item_num < len(self.videos_list_items):
            self.videos_list.blockSignals(True)
            self.videos_list.takeItem(item_num)
            del self.videos_list_items[item_num]
            self.videos_list.blockSignals(False)

    def change_selected_video_row(self, row_number):
        self.videos_list.blockSignals(True)
        self.videos_list.setCurrentRow(row_number)
        self.videos_list.blockSignals(False)

    def create_menubar(self):
        # create actions
        self.open_video_action = QAction('Open Video(s)', self)
        self.open_video_action.setShortcut('Ctrl+O')
        self.open_video_action.setStatusTip('Open one or more videos to track.')
        self.open_video_action.triggered.connect(lambda:self.controller.select_and_open_videos)

        self.save_params_action = QAction('Save Parameters', self)
        self.save_params_action.setShortcuts(['Ctrl+S'])
        self.save_params_action.setStatusTip('Quick-save the current parameters.')
        self.save_params_action.triggered.connect(self.controller.save_params)

        self.track_frame_action = QAction('Track Frame', self)
        self.track_frame_action.setShortcut('Ctrl+T')
        self.track_frame_action.setStatusTip('Track the currently previewed frame.')
        self.track_frame_action.triggered.connect(self.controller.track_frame)

        self.track_all_action = QAction('Track All Videos', self)
        self.track_all_action.setShortcut('Ctrl+Shift+T')
        self.track_all_action.setStatusTip('Track all of the currently loaded videos.')
        self.track_all_action.triggered.connect(self.controller.track_videos)

        self.exit_action = QAction('Exit', self)
        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.triggered.connect(self.controller.close_all)

        # create menu bar & add actions
        menubar  = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(self.open_video_action)
        file_menu.addAction(self.save_params_action)
        file_menu.addAction(self.track_frame_action)
        file_menu.addAction(self.track_all_action)
        file_menu.addAction(self.exit_action)

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

        # create heading label
        param_label = QLabel("Tracking Parameters")
        param_label.setStyleSheet("font-weight: bold;")
        self.right_layout.addWidget(param_label)

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
        self.right_layout.addLayout(button_layout_1)

        button_layout_2 = QHBoxLayout()
        button_layout_2.setSpacing(5)
        self.right_layout.addLayout(button_layout_2)

        button_layout_3 = QHBoxLayout()
        button_layout_3.setSpacing(5)
        self.right_layout.addLayout(button_layout_3)

        # add buttons
        self.save_button = QPushButton(u'\u2713 Save', self)
        # self.save_button.setMaximumWidth(80)
        self.save_button.clicked.connect(self.controller.save_params)
        self.save_button.setToolTip("Quick-save the current parameters. Use 'Reload' to load these parameters later.")
        button_layout_1.addWidget(self.save_button)

        self.track_button = QPushButton(u'\u279E Track', self)
        # self.track_button.setMaximumWidth(90)
        self.track_button.clicked.connect(self.controller.track_frame)
        self.track_button.setToolTip("Track the currently-previewed frame (for parameter tuning).")
        button_layout_1.addWidget(self.track_button)

        self.track_all_button = QPushButton(u'\u27A0 Track All', self)
        # self.track_all_button.setMaximumWidth(180)
        self.track_all_button.clicked.connect(self.controller.track_videos)
        self.track_all_button.setStyleSheet("font-weight: bold")
        self.track_all_button.setToolTip("Track all of the currently loaded videos with the current parameters.")
        button_layout_1.addWidget(self.track_all_button)

        self.reload_last_save_button = QPushButton(u'\u27AA Reload', self)
        # self.reload_last_save_button.setMaximumWidth(90)
        self.reload_last_save_button.clicked.connect(self.controller.load_last_params)
        self.reload_last_save_button.setToolTip("Reload the previously auto-saved parameters.")
        button_layout_2.addWidget(self.reload_last_save_button)

        self.load_params_button = QPushButton(u'Load Params\u2026', self)
        # self.load_params_button.setMaximumWidth(180)
        self.load_params_button.clicked.connect(lambda:self.controller.load_params(None))
        self.load_params_button.setToolTip("Load a set of parameters.")
        button_layout_2.addWidget(self.load_params_button)

        self.save_params_button = QPushButton(u'Save Params\u2026', self)
        # self.save_params_button.setMaximumWidth(180)
        self.save_params_button.clicked.connect(self.controller.save_params)
        self.save_params_button.setToolTip("Save the current set of parameters.")
        button_layout_2.addWidget(self.save_params_button)

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

        self.crop_tabs_widget.setDisabled(disabled_bool)

        self.remove_crop_button.setDisabled(disabled_bool)
        self.create_crop_button.setDisabled(disabled_bool)

    def add_crop_param_slider(self, label, description, minimum, maximum, value, slider_moved_func, parent, tick_interval=1, single_step=1, slider_scale_factor=1, int_values=False):
        # make layout to hold slider and textbox
        control_layout = QHBoxLayout()

        slider_label  = label + "_slider"
        textbox_label = label + "_textbox"

        # make slider & add to layout
        slider = QSlider(Qt.Horizontal)
        slider.setObjectName(label)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.NoTicks)
        slider.setTickInterval(tick_interval)
        slider.setSingleStep(single_step)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        control_layout.addWidget(slider)

        # make textbox & add to layout
        textbox = QLineEdit()
        textbox.setAlignment(Qt.AlignHCenter)
        textbox.setObjectName(label)
        textbox.setFixedWidth(40)
        textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.update_textbox_from_slider(slider, textbox, slider_scale_factor, int_values)
        control_layout.addWidget(textbox)

        # connect slider to set textbox text & update params
        slider.sliderMoved.connect(lambda:self.update_textbox_from_slider(slider, textbox, slider_scale_factor, int_values))
        slider.sliderMoved.connect(slider_moved_func)
        slider.sliderMoved.connect(lambda:self.slider_moved(slider))
        slider.sliderReleased.connect(lambda:self.slider_released(slider))

        # connect textbox to 
        textbox.editingFinished.connect(lambda:self.update_slider_from_textbox(slider, textbox, slider_scale_factor))
        textbox.editingFinished.connect(lambda:slider_moved_func(textbox.text()))

        # add row to form layout
        parent.addRow(description, control_layout)

        # add to list of controls
        self.crop_param_controls[-1][slider_label]  = slider
        self.crop_param_controls[-1][textbox_label] = textbox

    def get_checked_threshold_checkbox(self):
        if self.param_controls["show_body_threshold"].isChecked():
            return self.param_controls["show_body_threshold"]
        elif self.param_controls["show_eyes_threshold"].isChecked():
            return self.param_controls["show_eyes_threshold"]
        elif self.param_controls["show_tail_threshold"].isChecked():
            return self.param_controls["show_tail_threshold"]
        elif self.param_controls["show_tail_skeleton"].isChecked():
            return self.param_controls["show_tail_skeleton"]
        else:
            return None

    def slider_moved(self, slider):
        label = slider.objectName()

        if "threshold" in label:
            # store previously-checked threshold checkbox
            self.prev_checked_threshold_checkbox = self.get_checked_threshold_checkbox()

            checkbox = self.param_controls["show_{}".format(label)]
            checkbox.setChecked(True)
            self.controller.toggle_threshold_image(checkbox)
        elif label == "heading_angle_slider":
            textbox = self.param_controls["heading_angle_textbox"]
            angle = float(textbox.text())
            self.controller.add_angle_overlay(angle)

            if angle == 0 or angle == 360:
                index = 0
            elif angle == 90:
                index = 1
            elif angle == 180:
                index = 2
            elif angle == 270:
                index = 3
            else:
                index = None

            if index != None:
                self.param_controls["heading_direction"].setCurrentIndex(index)

    def slider_released(self, slider):
        label = slider.objectName()
        if "threshold" in label:
            checkbox = self.param_controls["show_{}".format(label)]
            checkbox.setChecked(False)

            if self.prev_checked_threshold_checkbox != None:
                self.prev_checked_threshold_checkbox.setChecked(True)
            self.controller.toggle_threshold_image(self.prev_checked_threshold_checkbox)
        elif label == "heading_angle_slider":
            self.controller.remove_angle_overlay()

    def set_crop_param_slider_value(self, label, value, crop_index, slider_scale_factor=1.0, int_values=False):
        # change slider value without sending signals

        slider_label  = label + "_slider"
        textbox_label = label + "_textbox"

        slider = self.crop_param_controls[crop_index][slider_label]

        if value == None:
            value = slider.minimum()

        slider.blockSignals(True)

        slider.setValue(value*slider_scale_factor)

        slider.blockSignals(False)

        # change textbox value
        textbox = self.crop_param_controls[crop_index][textbox_label]
        self.update_textbox_from_slider(slider, textbox, slider_scale_factor=slider_scale_factor, int_values=int_values)

    def add_textbox(self, label, description, return_func, default_value, parent):
        # make textbox & add row to form layout
        param_box = QLineEdit()
        param_box.setObjectName(label)
        param_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        param_box.editingFinished.connect(lambda:return_func(float(param_box.text())))
        parent.addRow(description, param_box)

        # set default text
        if default_value != None:
            param_box.setText(str(int(default_value)))

        # add to list of crop or global controls
        self.param_controls[label] = param_box

    def add_slider(self, label, description, minimum, maximum, slider_moved_func, value, parent, tick_interval=1, single_step=1, slider_scale_factor=1, int_values=False):
        # make layout to hold slider and textbox
        control_layout = QHBoxLayout()

        slider_label  = label + "_slider"
        textbox_label = label + "_textbox"

        # make slider & add to layout
        slider = QSlider(Qt.Horizontal)
        slider.setObjectName(slider_label)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.NoTicks)
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
        self.update_textbox_from_slider(slider, textbox, slider_scale_factor, int_values)
        control_layout.addWidget(textbox)

        # connect slider to set textbox text & update params
        slider.sliderMoved.connect(lambda:self.update_textbox_from_slider(slider, textbox, slider_scale_factor, int_values))
        slider.sliderMoved.connect(lambda:slider_moved_func(int(float(textbox.text()))))
        slider.sliderMoved.connect(lambda:self.slider_moved(slider))
        slider.sliderReleased.connect(lambda:self.slider_released(slider))

        # connect textbox to 
        textbox.editingFinished.connect(lambda:self.update_slider_from_textbox(slider, textbox, slider_scale_factor))
        textbox.editingFinished.connect(lambda:slider_moved_func(int(float(textbox.text()))))

        # add row to form layout
        parent.addRow(description, control_layout)

        # add to list of controls
        self.param_controls[slider_label]  = slider
        self.param_controls[textbox_label] = textbox

    def update_textbox_from_slider(self, slider, textbox, slider_scale_factor=1.0, int_values=False):
        if int_values:
            textbox.setText(str(int(slider.sliderPosition()/slider_scale_factor)))
        else:
            textbox.setText(str(slider.sliderPosition()/slider_scale_factor))

    def update_slider_from_textbox(self, slider, textbox, slider_scale_factor=1.0):
        try:
            value = float(textbox.text())
            slider.setValue(float(textbox.text())*slider_scale_factor)
        except:
            pass

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
        self.update_textbox_from_slider(slider, textbox, slider_scale_factor=slider_scale_factor, int_values=int_values)

    def set_crop_param_slider_value(self, label, value, crop_index, slider_scale_factor=1.0, int_values=False):
        # change slider value without sending signals

        slider_label  = label + "_slider"
        textbox_label = label + "_textbox"

        slider = self.crop_param_controls[crop_index][slider_label]

        if value == None:
            value = slider.minimum()

        slider.blockSignals(True)

        slider.setValue(value*slider_scale_factor)

        slider.blockSignals(False)

        # change textbox value
        textbox = self.crop_param_controls[crop_index][textbox_label]
        self.update_textbox_from_slider(slider, textbox, slider_scale_factor=slider_scale_factor, int_values=int_values)

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
        combobox.currentIndexChanged.connect(lambda:self.combobox_index_changed(combobox))
        parent.addRow(description, combobox)

        self.param_controls[label] = combobox

    def combobox_index_changed(self, combobox):
        label = combobox.objectName()

        if label == "heading_direction":
            index = combobox.currentIndex()

            if index == 0:
                angle = 0
            elif index == 1:
                angle = 90
            elif index == 2:
                angle = 180
            elif index == 3:
                angle = 270

            self.param_controls["heading_angle_slider"].setValue(angle)
            self.param_controls["heading_angle_textbox"].setText(str(angle))

    def update_videos_loaded_text(self, n_videos, curr_video_num):
        if n_videos > 0:
            self.videos_loaded_text = "Loaded <b>{}</b> video{}, showing <b>#{}</b>.".format(n_videos, "s"*(n_videos > 1), curr_video_num+1)
        else:
            self.videos_loaded_text       = "No videos loaded."
            self.background_progress_text = ""
            self.tracking_progress_text   = ""

        self.update_status_text()

    def set_videos_loaded_text(self, text):
        self.videos_loaded_text = text

        self.update_status_text()

    def update_background_progress_text(self, n_backgrounds, percent):
        if percent < 100:
            self.background_progress_text = "Calculating <b>{}</b> background{}... {:.1f}%.".format(n_backgrounds, "s"*(n_backgrounds > 1), percent)
        else:
            self.background_progress_text = ""

        self.update_status_text()

    def set_background_progress_text(self, text):
        self.background_progress_text = text

        self.update_status_text()

    def update_tracking_progress_text(self, n_videos, curr_video_num, percent, total_tracking_time=None):
        if percent < 100:
            self.tracking_progress_text = "Tracking <b>video {}/{}</b>... {:.1f}%.".format(curr_video_num+1, n_videos, percent)
        else:
            if total_tracking_time != None:
                self.tracking_progress_text = "Tracking completed in <b>{:.3f}s</b>.".format(total_tracking_time)
            else:
                self.tracking_progress_text = ""

        self.update_status_text()

    def update_status_text(self):
        self.status_label.setText("{} {} {}".format(self.videos_loaded_text, self.background_progress_text, self.tracking_progress_text))

    def set_invalid_params_text(self, text):
        self.invalid_params_label.setText(text)

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
        self.add_checkbox('use_multiprocessing', 'Use multiprocessing', self.controller.toggle_multiprocessing, params['use_multiprocessing'], self.checkbox_layout, 1, 1)
        self.add_checkbox('auto_track', 'Auto track', self.controller.toggle_auto_tracking, params['gui_params']['auto_track'], self.checkbox_layout, 2, 1)

        # add sliders - (key, description, start, end, initial value, parent layout)
        self.add_slider('scale_factor', 'Scale factor:', 1, 40, self.controller.update_scale_factor, 10.0*params['scale_factor'], self.form_layout, tick_interval=10, slider_scale_factor=10.0)
        self.add_slider('bg_sub_threshold', 'Background subtraction threshold:', 1, 100, self.controller.update_bg_sub_threshold, round(params['bg_sub_threshold']), self.form_layout, tick_interval=10, int_values=True)
        self.add_slider('heading_angle', 'Heading angle:', 0, 360, self.controller.update_heading_angle, params['heading_angle'], self.form_layout, tick_interval=50)
        self.add_combobox('heading_direction', 'Heading direction:', heading_direction_options, params['heading_direction'], self.form_layout)

        # add textboxes - (key, decription, initial value, parent layout)
        self.add_textbox('saved_video_fps', 'Saved video FPS:', self.controller.update_saved_video_fps, params['saved_video_fps'], self.form_layout)
        self.add_textbox('n_tail_points', '# tail points:', self.controller.update_n_tail_points, params['n_tail_points'], self.form_layout)

        # add comboboxes
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

            self.param_controls['heading_direction'].setCurrentIndex(heading_direction_options.index(params['heading_direction']))
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
        self.add_checkbox('show_eyes_threshold', "Show eyes threshold", self.controller.toggle_threshold_image, params['gui_params']['show_eyes_threshold'], self.checkbox_layout, 2, 0)
        self.add_checkbox('show_tail_threshold', "Show tail threshold", self.controller.toggle_threshold_image, params['gui_params']['show_tail_threshold'], self.checkbox_layout, 3, 0)
        self.add_checkbox('show_tail_skeleton', "Show tail skeleton", self.controller.toggle_threshold_image, params['gui_params']['show_tail_skeleton'], self.checkbox_layout, 4, 0)
        self.add_checkbox('track_tail', "Track tail", self.controller.toggle_tail_tracking, params['track_tail'], self.checkbox_layout, 5, 0)
        self.add_checkbox('track_eyes', "Track eyes", self.controller.toggle_eye_tracking, params['track_eyes'], self.checkbox_layout, 6, 0)
        self.add_checkbox('save_video', "Save video", self.controller.toggle_save_video, params['save_video'], self.checkbox_layout, 0, 1)
        self.add_checkbox('adjust_thresholds', 'Adjust thresholds', self.controller.toggle_adjust_thresholds, params['adjust_thresholds'], self.checkbox_layout, 1, 1)
        self.add_checkbox('subtract_background', 'Subtract background', self.controller.toggle_subtract_background, params['subtract_background'], self.checkbox_layout, 2, 1)
        self.add_checkbox('use_multiprocessing', 'Use multiprocessing', self.controller.toggle_multiprocessing, params['use_multiprocessing'], self.checkbox_layout, 3, 1)
        self.add_checkbox('auto_track', 'Auto track', self.controller.toggle_auto_tracking, params['gui_params']['auto_track'], self.checkbox_layout, 4, 1)
        self.add_checkbox('zoom_body_crop', 'Zoom around body crop', self.controller.toggle_zoom_body_crop, params['gui_params']['zoom_body_crop'], self.checkbox_layout, 5, 1)

        # add sliders - (key, description, start, end, callback function, initial value, parent layout)
        self.add_slider('scale_factor', 'Scale factor:', 1, 40, self.controller.update_scale_factor, 10.0*params['scale_factor'], self.form_layout, slider_scale_factor=10.0)
        self.add_slider('body_crop_height', 'Body crop height:', 1, 100, self.controller.update_body_crop_height, round(params['body_crop'][0]), self.form_layout, tick_interval=10, int_values=True)
        self.add_slider('body_crop_width', 'Body crop width:', 1, 100, self.controller.update_body_crop_width, round(params['body_crop'][1]), self.form_layout, tick_interval=10, int_values=True)
        self.add_slider('bg_sub_threshold', 'Background subtraction threshold:', 1, 100, self.controller.update_bg_sub_threshold, round(params['bg_sub_threshold']), self.form_layout, tick_interval=10, int_values=True)

        # add textboxes - (key, decription, initial value, parent layout)
        self.add_textbox('min_tail_body_dist', 'Body/tail min. dist.:', self.controller.update_min_tail_body_dist, params['min_tail_body_dist'], self.form_layout)
        self.add_textbox('max_tail_body_dist', 'Body/tail max. dist.:', self.controller.update_max_tail_body_dist, params['max_tail_body_dist'], self.form_layout)
        self.add_textbox('saved_video_fps', 'Saved video FPS:', self.controller.update_saved_video_fps, params['saved_video_fps'], self.form_layout)
        self.add_textbox('n_tail_points', '# tail points:', self.controller.update_n_tail_points, params['n_tail_points'], self.form_layout)

        # add comboboxes
        self.add_combobox('interpolation', 'Interpolation:', interpolation_options, params['interpolation'], self.form_layout)

    def update_gui_from_params(self, params):
        # update param controls with current parameters
        if self.param_controls != None:
            self.param_controls['invert'].setChecked(params['invert'])
            self.param_controls['show_body_threshold'].setChecked(params['gui_params']['show_body_threshold'])
            self.param_controls['show_eyes_threshold'].setChecked(params['gui_params']['show_eyes_threshold'])
            self.param_controls['show_tail_threshold'].setChecked(params['gui_params']['show_tail_threshold'])
            self.param_controls['show_tail_skeleton'].setChecked(params['gui_params']['show_tail_skeleton'])
            self.param_controls['track_tail'].setChecked(params['track_tail'])
            self.param_controls['track_eyes'].setChecked(params['track_eyes'])
            self.param_controls['save_video'].setChecked(params['save_video'])
            self.param_controls['adjust_thresholds'].setChecked(params['adjust_thresholds'])
            self.param_controls['subtract_background'].setChecked(params['subtract_background'])
            self.param_controls['use_multiprocessing'].setChecked(params['use_multiprocessing'])
            self.param_controls['auto_track'].setChecked(params['gui_params']['auto_track'])
            self.param_controls['zoom_body_crop'].setChecked(params['gui_params']['zoom_body_crop'])

            self.set_slider_value('scale_factor', params['scale_factor'], slider_scale_factor=10)
            self.set_slider_value('body_crop_height', params['body_crop'][0], int_values=True)
            self.set_slider_value('body_crop_width', params['body_crop'][1], int_values=True)
            self.set_slider_value('bg_sub_threshold', params['bg_sub_threshold'], int_values=True)

            self.param_controls['min_tail_body_dist'].setText(str(params['min_tail_body_dist']))
            self.param_controls['max_tail_body_dist'].setText(str(params['max_tail_body_dist']))
            self.param_controls['saved_video_fps'].setText(str(params['saved_video_fps']))
            self.param_controls['n_tail_points'].setText(str(params['n_tail_points']))

            self.param_controls['interpolation'].setCurrentIndex(interpolation_options.index(params['interpolation']))

    def create_crop_param_controls(self, crop_params):
        ParamWindow.create_crop_param_controls(self, crop_params)

        # add sliders & textboxes - (key, decription, initial value, parent layout)
        self.add_crop_param_slider('body_threshold', 'Body threshold:', 0, 255, crop_params['body_threshold'], self.controller.update_body_threshold, self.crop_param_form_layout, tick_interval=10, int_values=True)
        self.add_crop_param_slider('eyes_threshold', 'Eyes threshold:', 0, 255, crop_params['eyes_threshold'], self.controller.update_eyes_threshold, self.crop_param_form_layout, tick_interval=10, int_values=True)
        self.add_crop_param_slider('tail_threshold', 'Tail threshold:', 0, 255, crop_params['tail_threshold'], self.controller.update_tail_threshold, self.crop_param_form_layout, tick_interval=10, int_values=True)

    def update_gui_from_crop_params(self, crop_params):
        ParamWindow.update_gui_from_crop_params(self, crop_params)

        if len(crop_params) == len(self.crop_param_controls):
            for c in range(len(crop_params)):
                self.set_crop_param_slider_value('body_threshold', crop_params[c]['body_threshold'], c, int_values=True)
                self.set_crop_param_slider_value('eyes_threshold', crop_params[c]['eyes_threshold'], c, int_values=True)
                self.set_crop_param_slider_value('tail_threshold', crop_params[c]['tail_threshold'], c, int_values=True)

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)