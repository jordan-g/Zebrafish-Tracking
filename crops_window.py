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

class CropsWindow(QMainWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # set position & size
        self.setGeometry(550, 100, 10, 10)

        # set title
        self.setWindowTitle("Crops")

        # create main widget
        self.main_widget = QWidget(self)

        # create main layout
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addStretch(1)
        self.main_layout.setSpacing(5)

        # initialize list of dicts used for accessing all crop parameter controls
        self.param_controls = []

        # create tabs widget
        self.crop_tabs_widget = QTabWidget()
        self.crop_tabs_widget.setUsesScrollButtons(True)
        self.crop_tabs_widget.tabCloseRequested.connect(self.controller.remove_crop)
        self.crop_tabs_widget.currentChanged.connect(self.controller.change_crop)
        self.crop_tabs_widget.setElideMode(Qt.ElideLeft)
        self.crop_tabs_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.crop_tabs_widget.setMinimumSize(276, 300)
        self.main_layout.addWidget(self.crop_tabs_widget)

        # create tabs layout
        crop_tabs_layout  = QHBoxLayout(self.crop_tabs_widget)

        # create lists for storing all crop tab widgets & layouts
        self.crop_tab_layouts = []
        self.crop_tab_widgets = []

        # add crop button layout
        crop_button_layout = QHBoxLayout()
        crop_button_layout.setSpacing(5)
        crop_button_layout.addStretch(1)
        self.main_layout.addLayout(crop_button_layout)

        # add delete crop button
        self.remove_crop_button = QPushButton(u'\u2717 Remove Crop', self)
        self.remove_crop_button.setMaximumWidth(120)
        self.remove_crop_button.clicked.connect(lambda:self.controller.remove_crop(self.controller.current_crop))
        self.remove_crop_button.setDisabled(True)
        crop_button_layout.addWidget(self.remove_crop_button)

        # add new crop button
        self.create_crop_button = QPushButton(u'\u270E New Crop', self)
        self.create_crop_button.setMaximumWidth(100)
        self.create_crop_button.clicked.connect(lambda:self.controller.create_crop())
        crop_button_layout.addWidget(self.create_crop_button)

        # set main widget
        self.setCentralWidget(self.main_widget)

        # set window buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.show()

    def add_textbox(self, label, description, default_value, parent):
        # make textbox & add row to form layout
        param_box = QLineEdit()
        param_box.setObjectName(label)
        param_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        param_box.returnPressed.connect(self.controller.update_crop_params_from_gui)
        parent.addRow(description, param_box)

        # set default text
        if default_value != None:
            param_box.setText(str(default_value))

        # add to list of crop or global controls
        self.param_controls[-1][label] = param_box

    def add_slider(self, label, description, minimum, maximum, value, slider_moved_func, parent, tick_interval=1, single_step=1, multiplier=1):
        # make layout to hold slider and textbox
        control_layout = QHBoxLayout()

        slider_label  = label + "_slider"
        textbox_label = label + "_textbox"

        # make slider & add to layout
        slider = QSlider(Qt.Horizontal)
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
        # textbox.setMinimumHeight(10)
        textbox.setFixedWidth(40)
        textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        textbox.setText(str(value/multiplier))
        textbox.returnPressed.connect(self.controller.update_crop_params_from_gui)
        control_layout.addWidget(textbox)

        # connect slider to set textbox text & update params
        slider.sliderMoved.connect(lambda:self.update_textbox_from_slider(slider, textbox, multiplier))
        slider.sliderMoved.connect(slider_moved_func)

        # connect textbox to 
        textbox.editingFinished.connect(lambda:self.update_slider_from_textbox(slider, textbox, multiplier))
        textbox.editingFinished.connect(slider_moved_func)

        # add row to form layout
        parent.addRow(description, control_layout)

        # add to list of controls
        self.param_controls[-1][slider_label]  = slider
        self.param_controls[-1][textbox_label] = textbox

    def update_textbox_from_slider(self, slider, textbox, multiplier=1.0):
        textbox.setText(str(slider.sliderPosition()/multiplier))

    def update_slider_from_textbox(self, slider, textbox, multiplier=1.0):
        slider.setValue(float(textbox.text())*multiplier)

    def set_slider_value(self, label, value, crop_index, slider_scale_factor=None):
        # change slider value without sending signals

        slider_label  = label + "_slider"
        textbox_label = label + "_textbox"

        slider = self.param_controls[crop_index][slider_label]

        if value == None:
            value = slider.minimum()

        slider.blockSignals(True)

        if slider_scale_factor != None:
            slider.setValue(value*slider_scale_factor)
        else:
            slider.setValue(value)

        slider.blockSignals(False)

        # change textbox value
        textbox = self.param_controls[crop_index][textbox_label]
        textbox.setText(str(int(value)))

    def set_gui_disabled(self, disbaled_bool):
        self.crop_tabs_widget.setDisabled(disbaled_bool)

        self.remove_crop_button.setDisabled(disbaled_bool)
        self.create_crop_button.setDisabled(disbaled_bool)

    def create_crop_tab(self, crop_params):
        # create crop tab widget & layout
        crop_tab_widget = QWidget(self.crop_tabs_widget)
        crop_tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        crop_tab_widget.resize(276, 300)

        crop_tab_layout = QVBoxLayout(crop_tab_widget)

        # add to list of crop widgets & layouts
        self.crop_tab_layouts.append(crop_tab_layout)
        self.crop_tab_widgets.append(crop_tab_widget)

        # create form layout
        self.form_layout = QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        crop_tab_layout.addLayout(self.form_layout)

        # add dict for storing param controls for this crop
        self.param_controls.append({})

        # add param controls
        self.create_param_controls(crop_params)

        # create crop button layout
        crop_button_layout = QHBoxLayout()
        crop_button_layout.setSpacing(5)
        crop_button_layout.addStretch(1)
        crop_tab_layout.addLayout(crop_button_layout)

        # add crop buttons
        self.reset_crop_button = QPushButton(u'\u25A8 Reset Crop', self)
        self.reset_crop_button.setMaximumWidth(110)
        self.reset_crop_button.clicked.connect(self.controller.reset_crop)
        crop_button_layout.addWidget(self.reset_crop_button)

        self.select_crop_button = QPushButton(u'\u25A3 Select Crop', self)
        self.select_crop_button.setMaximumWidth(110)
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
        del self.param_controls[index]
        del self.crop_tab_widgets[index]
        del self.crop_tab_layouts[index]

        # update text on all tabs
        for i in range(len(self.controller.params['crop_params'])):
            self.crop_tabs_widget.setTabText(i, str(i))

    def create_param_controls(self, crop_params):
        if self.controller.current_frame is not None:
            # add sliders - (key, description, start, end, initial value, parent layout)
            self.add_slider('crop_y', 'Crop y:', 1, self.controller.current_frame.shape[0], crop_params['crop'][0], self.controller.update_crop_params_from_gui, self.form_layout, tick_interval=50)
            self.add_slider('crop_x', 'Crop x:', 1, self.controller.current_frame.shape[1], crop_params['crop'][1], self.controller.update_crop_params_from_gui, self.form_layout, tick_interval=50)
            self.add_slider('offset_y', 'Offset y:', 0, self.controller.current_frame.shape[0]-1, crop_params['offset'][0], self.controller.update_crop_params_from_gui, self.form_layout, tick_interval=50)
            self.add_slider('offset_x', 'Offset x:', 0, self.controller.current_frame.shape[1]-1, crop_params['offset'][1], self.controller.update_crop_params_from_gui, self.form_layout, tick_interval=50)
        else:
            self.add_slider('crop_y', 'Crop y:', 1, 2, 1, self.controller.update_crop_params_from_gui, self.form_layout, tick_interval=50)
            self.add_slider('crop_x', 'Crop x:', 1, 2, 1, self.controller.update_crop_params_from_gui, self.form_layout, tick_interval=50)
            self.add_slider('offset_y', 'Offset y:', 0, 1, 0, self.controller.update_crop_params_from_gui, self.form_layout, tick_interval=50)
            self.add_slider('offset_x', 'Offset x:', 0, 1, 0, self.controller.update_crop_params_from_gui, self.form_layout, tick_interval=50)
    
    def update_gui_from_params(self, crop_params):
        if len(crop_params) == len(self.param_controls):
            # update crop gui for each crop
            for c in range(len(crop_params)):
                self.param_controls[c]['crop_y' + '_slider'].setMaximum(self.controller.current_frame.shape[0])
                self.param_controls[c]['crop_x' + '_slider'].setMaximum(self.controller.current_frame.shape[1])
                self.param_controls[c]['offset_y' + '_slider'].setMaximum(self.controller.current_frame.shape[0]-1)
                self.param_controls[c]['offset_x' + '_slider'].setMaximum(self.controller.current_frame.shape[1]-1)

                # update param controls with current parameters
                self.set_slider_value('crop_y', crop_params[c]['crop'][0], c)
                self.set_slider_value('crop_x', crop_params[c]['crop'][1], c)
                self.set_slider_value('offset_y', crop_params[c]['offset'][0], c)
                self.set_slider_value('offset_x', crop_params[c]['offset'][1], c)

    def closeEvent(self, ce):
        if not self.controller.closing:
            ce.ignore()
        else:
            ce.accept()

class FreeswimmingCropsWindow(CropsWindow):
    def __init__(self, controller):
        CropsWindow.__init__(self, controller)

    def create_param_controls(self, crop_params):
        CropsWindow.create_param_controls(self, crop_params)

        # add textboxes - (key, decription, initial value, parent layout)
        self.add_textbox('body_threshold', 'Body threshold:', crop_params['body_threshold'], self.form_layout)
        self.add_textbox('eye_threshold', 'Eye threshold:', crop_params['eye_threshold'], self.form_layout)
        self.add_textbox('tail_threshold', 'Tail threshold:', crop_params['tail_threshold'], self.form_layout)

    def update_gui_from_params(self, crop_params):
        CropsWindow.update_gui_from_params(self, crop_params)

        if len(crop_params) == len(self.param_controls):
            for c in range(len(crop_params)):
                self.param_controls[c]['body_threshold'].setText(str(crop_params[c]['body_threshold']))
                self.param_controls[c]['eye_threshold'].setText(str(crop_params[c]['eye_threshold']))
                self.param_controls[c]['tail_threshold'].setText(str(crop_params[c]['tail_threshold']))

class HeadfixedCropsWindow(CropsWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # set position & size
        self.setGeometry(550, 100, 10, 10)

        # set title
        self.setWindowTitle("Crops")

        # create main widget
        self.main_widget = QWidget(self)

        # create main layout
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addStretch(1)
        self.main_layout.setSpacing(5)

        # initialize list of dicts used for accessing all crop parameter controls
        self.param_controls = []

        # create tabs widget
        self.crop_tabs_widget = QTabWidget()
        self.crop_tabs_widget.setUsesScrollButtons(True)
        self.crop_tabs_widget.tabCloseRequested.connect(self.controller.remove_crop)
        self.crop_tabs_widget.currentChanged.connect(self.controller.change_crop)
        self.crop_tabs_widget.setElideMode(Qt.ElideLeft)
        self.crop_tabs_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.crop_tabs_widget.setMinimumSize(276, 300)
        self.main_layout.addWidget(self.crop_tabs_widget)

        # create tabs layout
        crop_tabs_layout  = QHBoxLayout(self.crop_tabs_widget)

        # create lists for storing all crop tab widgets & layouts
        self.crop_tab_layouts = []
        self.crop_tab_widgets = []

        # add crop button layout
        crop_button_layout = QHBoxLayout()
        crop_button_layout.setSpacing(5)
        crop_button_layout.addStretch(1)
        self.main_layout.addLayout(crop_button_layout)

        # set main widget
        self.setCentralWidget(self.main_widget)

        # set window buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.show()

    def set_gui_disabled(self, disbaled_bool):
        self.crop_tabs_widget.setDisabled(disbaled_bool)