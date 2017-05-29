import sys
import os
import time
import threading
import json

import numpy as np
import cv2

# import the Qt library
try:
    from PyQt4.QtCore import Qt, QThread, QSize
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QGraphicsDropShadowEffect, QColor
except:
    from PyQt5.QtCore import Qt, QThread, QSize
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon, QColor
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QGraphicsDropShadowEffect

import tracking
import utilities

# color table to use for showing images
gray_color_table = [qRgb(i, i, i) for i in range(256)]

class PreviewQLabel(QLabel):
    """
    QLabel subclass used to show a preview image.

    Properties:
        preview_window (PreviewWindow): preview window that contains this label
        scale_factor:          (float): scale factor between label pixels & pixels of the actual image
        start_crop_coord        (y, x): starting coordinate of mouse crop selection
        end_crop_coord          (y, x): ending coordinate of mouse crop selection
        pixmap                 (Array): label's pixmap
    """

    def __init__(self, preview_window):
        QLabel.__init__(self)

        self.preview_window = preview_window
        self.scale_factor   = None
        self.pix            = None  # image label's pixmap
        self.pix_size       = None  # size of image label's pixmap

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
                self.preview_window.remove_tail_start()
                self.preview_window.draw_tail_start(np.array(self.end_crop_coord))


    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    def update_size(self):
        if self.pix:
            # calculate new label vs. image scale factor
            scale_factor = float(self.pix_size)/max(self.pix.width(), self.pix.height())
            self.scale_factor = scale_factor

            # scale pixmap
            pix = self.pix.scaled(self.pix.width()*scale_factor, self.pix.height()*scale_factor, Qt.KeepAspectRatio, Qt.FastTransformation)

            # update pixmap & size
            self.setPixmap(pix)
            self.setFixedSize(pix.size())

    def update_pixmap(self, image):
        if image == None:
            self.scale_factor   = None
            self.pix            = None  # image label's pixmap
            self.pix_size       = None  # size of image label's pixmap
            self.clear()
        else:
            # get image info
            height, width, bytesPerComponent = image.shape
            bytesPerLine = bytesPerComponent * width
            
            # create qimage
            qimage = QImage(image.data, image.shape[1], image.shape[0], bytesPerLine, QImage.Format_RGB888)
            qimage.setColorTable(gray_color_table)

            # generate pixmap
            self.pix = QPixmap(qimage)

class PreviewWindow(QMainWindow):
    """
    QMainWindow subclass used to show frames & tracking.

    Properties:
        controller        (Controller): controller object
        main_widget          (QWidget): main window widget
        main_layout          (QLayout): main window layout
        image_label    (PreviewQLabel): label for showing a preview image
        instructions_label    (QLabel): label for showing instructions
        image_slider         (QSlider): slider for switching between frames
        image                  (Array): image to show
        pixmap                 = None  # image label's pixmap
        pixmap_size            = None  # size of image label's pixmap
        tracking_data          = None  # list of tracking data
        selecting_crop         = False # whether user is selecting a crop
        changing_heading_angle = False # whether the user is changing the heading angle
    """

    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # set title
        self.setWindowTitle("Preview")

        # get parameter window position & size
        param_window_x      = self.controller.param_window.x()
        param_window_y      = self.controller.param_window.y()
        param_window_width  = self.controller.param_window.width()

        # set position & size
        self.setGeometry(param_window_x + param_window_width, param_window_y, 10, 10)

        # create main widget
        self.main_widget = QWidget(self)
        self.main_widget.setStyleSheet("background-color: #b3b9bc;")
        self.main_widget.setMinimumSize(QSize(500, 500))

        # create main layout
        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # create label that shows frames
        self.image_widget = QWidget(self)
        self.image_layout = QHBoxLayout(self.image_widget)
        self.image_layout.setContentsMargins(16, 16, 16, 16)
        self.image_label = PreviewQLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.hide()
        self.image_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.image_widget, 0, 0)

        self.image_label.setStyleSheet("border: 1px solid rgba(122, 127, 130, 0.5)")
        image_label_shadow = QGraphicsDropShadowEffect()
        image_label_shadow.setBlurRadius(16)
        image_label_shadow.setColor(QColor(122, 127, 130, 100))
        image_label_shadow.setOffset(0)
        self.image_label.setGraphicsEffect(image_label_shadow)

        self.bottom_widget = QWidget(self)
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.bottom_layout.setContentsMargins(8, 0, 8, 8)
        self.main_layout.addWidget(self.bottom_widget, 1, 0)

        # create label that shows crop instructions
        self.instructions_label = QLabel("")
        self.instructions_label.setStyleSheet("font-size: 11px;")
        self.instructions_label.setAlignment(Qt.AlignCenter)
        self.bottom_layout.addWidget(self.instructions_label)

        # create image slider
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setFocusPolicy(Qt.StrongFocus)
        self.image_slider.setTickPosition(QSlider.NoTicks)
        self.image_slider.setTickInterval(1)
        self.image_slider.setSingleStep(1)
        self.image_slider.setValue(0)
        self.image_slider.valueChanged.connect(self.controller.switch_frame)
        self.image_slider.hide()
        self.bottom_layout.addWidget(self.image_slider)

        # initialize variables
        self.image                  = None  # image to show
        self.tracking_data          = None  # list of tracking data
        self.selecting_crop         = False # whether user is selecting a crop
        self.changing_heading_angle = False # whether the user is changing the heading angle
        self.body_crop              = None

        # set main widget
        self.setCentralWidget(self.main_widget)

        # set window buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)

        self.show()

    def resizeEvent(self, event):
        # get new size
        size = event.size()

        # update label's pixmap size
        self.image_label.pix_size = self.get_available_pix_size()

        # update size of image label
        self.image_label.update_size()

    def get_available_pix_size(self):
        available_width  = self.width() - 16
        available_height = self.height() - self.bottom_widget.height() - 30

        if available_height < available_width:
            return available_height
        else:
            return available_width

    def start_selecting_crop(self):
        # start selecting crop
        self.selecting_crop = True

        # add instruction text
        self.instructions_label.setText("Click & drag to select crop area.")

    def plot_image(self, image, params, crop_params, tracking_results, new_load=False, new_frame=False, show_slider=True, crop_around_body=False):
        if image == None:
            self.update_image_label(None)
            self.image_slider.hide()
            self.image_label.hide()
        else:
            if new_load:
                self.image_label.show()
                self.remove_tail_start()
                if show_slider:
                    if not self.image_slider.isVisible():
                        self.image_slider.setValue(0)
                        self.image_slider.setMaximum(self.controller.n_frames-1)
                        self.image_slider.show()
                else:
                    self.image_slider.hide()

            # convert to RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # update image
            self.image = tracking.scale_frame(image, 1.0/params['scale_factor'], utilities.translate_interpolation(self.controller.params['interpolation']))
            image = self.image.copy()

            try:
                body_crop = params['body_crop']
            except:
                body_crop = None

            try:
                tail_start_coords = params['tail_start_coords']

                # add tail start point to image
                cv2.circle(image, (int(round(tail_start_coords[1] - crop_params['offset'][1])), int(round(tail_start_coords[0] - crop_params['offset'][0]))), 1, (180, 180, 50), -1)
            except (KeyError, TypeError) as error:
                # print(error)
                tail_start_coords = None

            if tracking_results != None:
                body_position = tracking_results['body_position']
                heading_angle = tracking_results['heading_angle']

                # add tracking to image
                image = tracking.add_tracking_to_frame(image, tracking_results, cropped=True)

                if body_crop != None and body_position != None:
                    if not crop_around_body:
                        # copy image
                        overlay = image.copy()

                        # draw tail crop overlay
                        cv2.rectangle(overlay, (int(body_position[1]-body_crop[1]), int(body_position[0]-body_crop[0])),
                                                (int(body_position[1]+body_crop[1]), int(body_position[0]+body_crop[0])), (242, 242, 65), -1)

                        # overlay with the original image
                        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

                        self.body_crop = None
                    else:
                        self.body_crop = body_crop
            
            if crop_around_body:
                _, image = tracking.crop_frame_around_body(image, body_position, params['body_crop'], params['scale_factor'])

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
            image = self.image.copy()

            cv2.circle(image, (int(round(rel_tail_start_coords[1])), int(round(rel_tail_start_coords[0]))), 1, (180, 180, 50), -1)

            # update image label
            self.update_image_label(image)

    def remove_tail_start(self):
        self.update_image_label(self.image)

    def add_angle_overlay(self, angle):
        image = self.image.copy()
        image_height = self.image.shape[0]
        image_width  = self.image.shape[1]
        center_y = image_height/2
        center_x = image_width/2

        cv2.arrowedLine(image, (int(center_x - 0.3*image_height*np.sin(angle*np.pi/180)), int(center_y - 0.3*image_width*np.cos(angle*np.pi/180))),
                        (int(center_x + 0.3*image_height*np.sin(angle*np.pi/180)), int(center_y + 0.3*image_width*np.cos(angle*np.pi/180))),
                        (50, 255, 50), 2)

        self.update_image_label(image)

    def remove_angle_overlay(self):
        self.update_image_label(self.image)

    def update_image_label(self, image):
        self.image_label.update_pixmap(image)

        # update label's pixmap size
        self.image_label.pix_size = self.get_available_pix_size()

        # update size of image label
        self.image_label.update_size()

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
