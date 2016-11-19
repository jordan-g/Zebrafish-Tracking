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

import tracking

# color table to use for showing images
gray_color_table = [qRgb(i, i, i) for i in range(256)]

class PlotQLabel(QLabel):
    """
    QLabel subclass used to show a preview image.

    Properties:
        preview_window (PreviewWindow): preview window that contains this label
        scale_factor:          (float): scale factor between label pixels & pixels of the actual image
        start_crop_coord        (y, x): starting coordinate of mouse crop selection
        end_crop_coord          (y, x): ending coordinate of mouse crop selection
    """

    def __init__(self, preview_window):
        QLabel.__init__(self)

        self.preview_window = preview_window
        self.scale_factor = None

        # accept clicks
        self.setAcceptDrops(True)

    def mousePressEvent(self, event):
        if self.scale_factor:
            self.start_crop_coord = (int(round(event.y()/self.scale_factor)), int(round(event.x()/self.scale_factor)))

    def mouseMoveEvent(self, event):
        if self.scale_factor:
            self.end_crop_coord = (int(round(event.y()/self.scale_factor)), int(round(event.x()/self.scale_factor)))

            # draw colored crop selection
            self.preview_window.draw_crop_selection(self.start_crop_coord, self.end_crop_coord)

    def mouseReleaseEvent(self, event):
        if self.scale_factor:
            self.end_crop_coord = (int(round(event.y()/self.scale_factor)), int(round(event.x()/self.scale_factor)))

            print("user clicked {}.".format(self.end_crop_coord))

            if self.end_crop_coord != self.start_crop_coord:
                # finished selecting crop area; crop the image
                self.preview_window.crop_selection(self.start_crop_coord, self.end_crop_coord)
            else:
                # draw tail start coordinate
                self.preview_window.draw_tail_start(np.array(self.end_crop_coord))


    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

class PreviewWindow(QMainWindow):
    """
    QMainWindow subclass used to show frames & tracking.

    Properties:
        param_window (ParamWindow): parameter window
        scale_factor:          (float): scale factor between label pixels & pixels of the actual image
        start_crop_coord        (y, x): starting coordinate of mouse crop selection
        end_crop_coord          (y, x): ending coordinate of mouse crop selection
    """

    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # set title
        self.setWindowTitle("Preview")

        # set position & size
        self.setGeometry(851, 100, 10, 10)

        # create main widget
        self.main_widget = QWidget(self)
        self.main_widget.setStyleSheet("background-color: #666;")

        # create main layout
        self.main_layout = QVBoxLayout(self.main_widget)

        # create label that shows frames
        self.image_label = PlotQLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.image_label)

        # create label that shows crop instructions
        self.instructions_label = QLabel("")
        self.instructions_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.instructions_label)

        # create image slider
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setFocusPolicy(Qt.StrongFocus)
        self.image_slider.setTickPosition(QSlider.TicksBothSides)
        self.image_slider.setTickInterval(1)
        self.image_slider.setSingleStep(1)
        self.image_slider.setValue(0)
        self.image_slider.valueChanged.connect(self.controller.switch_frame)
        self.image_slider.hide()
        self.main_layout.addWidget(self.image_slider)

        # initialize variables
        # self.image_slider   = None  # slider for selecting frames
        self.image                  = None  # image to show
        self.pixmap                 = None  # image label's pixmap
        self.pixmap_size            = None  # size of image label's pixmap
        self.tracking_data          = None  # list of tracking data
        self.selecting_crop         = False # whether user is selecting a crop
        self.changing_heading_angle = False # whether the user is changing the heading angle

        # set main widget
        self.setCentralWidget(self.main_widget)

        # set window buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.show()

    def resizeEvent(self, event):
        # get new size
        size = event.size()

        # change width to within min & max values
        width = max(min(size.width(), 900), 400)

        # set new size of pixmap
        self.pixmap_size = width - 40

        if self.pixmap:
            # calculate new label vs. image scale factor
            scale_factor = float(self.pixmap_size)/max(self.pixmap.width(), self.pixmap.height())
            self.image_label.set_scale_factor(scale_factor)

            # scale pixmap
            pix = self.pixmap.scaled(self.pixmap.width()*scale_factor, self.pixmap.height()*scale_factor, Qt.IgnoreAspectRatio, Qt.FastTransformation)

            # update label's pixmap & size
            self.image_label.setPixmap(pix)
            self.image_label.setFixedSize(pix.size())

        # constrain window to square size
        self.resize(width, width)

    def start_selecting_crop(self):
        # start selecting crop
        self.selecting_crop = True

        # add instruction text
        self.instructions_label.setText("Click & drag to select crop area.")

    def plot_image(self, image, params, crop_params, tracking_results, new_load=False, new_frame=False, show_slider=True):
        if new_load:
            if show_slider:
                if not self.image_slider.isVisible():
                    self.image_slider.setValue(0)
                    self.image_slider.setMaximum(self.controller.n_frames-1)
                    self.image_slider.show()
            else:
                self.image_slider.hide()

        # convert to BGR
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # update image
        self.image = image.copy()

        try:
            tail_crop = tracking.get_relative_tail_crop(params['tail_crop'], params['shrink_factor'])
        except KeyError:
            tail_crop = None

        try:
            tail_start_coords = tracking.get_relative_tail_start_coords(params['tail_start_coords'], crop_params['offset'], params['shrink_factor'])

            # add tail start point to image
            cv2.circle(image, (int(round(tail_start_coords[1])), int(round(tail_start_coords[0]))), 1, (180, 50, 180), -1)
        except KeyError:
            tail_start_coords = None

        if len(tracking_results) > 0:
            # get tracking coords
            tail_coords    = tracking_results[0]
            spline_coords  = tracking_results[1]
            if len(tracking_results) > 2:
                heading_angle  = tracking_results[2]
                body_position  = tracking_results[3]
                eye_coords     = tracking_results[4]
            else:
                heading_angle  = None
                body_position  = None
                eye_coords     = None

            # add tracking to image
            image = tracking.add_tracking_to_frame(image, tracking_results)

            if tail_crop != None and eye_coords != None:
                # copy image
                overlay = image.copy()

                # draw tail crop overlay
                cv2.rectangle(overlay, (int(body_position[1]-tail_crop[1]), int(body_position[0]-tail_crop[0])),
                                        (int(body_position[1]+tail_crop[1]), int(body_position[0]+tail_crop[0])), (242, 242, 65), -1)

                # overlay with the original image
                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

        # update image label
        self.update_image_label(image)

    def draw_crop_selection(self, start_crop_coords, end_crop_coords):
        if self.selecting_crop and self.image != None:
            # convert image to rgb
            if len(self.image.shape) < 3:
                image = np.repeat(self.image[:, :, np.newaxis], 3, axis=2)
            else:
                image = self.image.copy()

            # copy image
            overlay = image.copy()

            # draw crop selection overlay
            cv2.rectangle(overlay, (start_crop_coords[1], start_crop_coords[0]), (end_crop_coords[1], end_crop_coords[0]), (255, 51, 0), -1)

            # overlay with the original image
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

            # update image label
            self.update_image_label(image)

    def draw_tail_start(self, rel_tail_start_coords):
        if self.controller.params['type'] == "headfixed":
            # send new tail start coordinates to controller
            self.controller.update_tail_start_coords(rel_tail_start_coords)

            # clear instructions text
            self.instructions_label.setText("")

        if self.image != None:
            image = self.image

            cv2.circle(image, (int(round(rel_tail_start_coords[1])), int(round(rel_tail_start_coords[0]))), 1, (180, 50, 180), -1)

            # update image label
            self.update_image_label(image)

    def update_image_label(self, image):
        # get image info
        height, width, bytesPerComponent = image.shape
        bytesPerLine = bytesPerComponent * width
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)

        # create qimage
        qimage = QImage(image.data, image.shape[1], image.shape[0], bytesPerLine, QImage.Format_RGB888)
        qimage.setColorTable(gray_color_table)

        # generate pixmap
        self.pixmap = QPixmap(qimage)

        # update label vs. image scale factor
        self.image_label.set_scale_factor(float(self.pixmap_size)/max(self.pixmap.width(), self.pixmap.height()))

        # scale pixmap
        pixmap = self.pixmap.scaled(self.pixmap_size, self.pixmap_size, Qt.KeepAspectRatio)

        # set image label's pixmap & update label's size
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())

    def crop_selection(self, start_crop_coord, end_crop_coord):
        if self.selecting_crop:
            # stop selecting the crop
            self.selecting_crop = False

            # clear instruction text
            self.instructions_label.setText("")

            # update crop parameters from the selection
            self.controller.update_crop_from_selection(start_crop_coord, end_crop_coord)

    def closeEvent(self, ce):
        if not self.controller.closing:
            ce.ignore()
        else:
            ce.accept()
