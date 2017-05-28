import sys
import os
import time
import threading
import json

import numpy as np
import cv2

# import the Qt library
try:
    from PyQt4.QtCore import Qt, QThread
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
except:
    from PyQt5.QtCore import Qt, QThread
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog

from controller import *

if __name__ == "__main__":
    qApp = QApplication(sys.argv)

    mode_select_dialog = QMessageBox()
    mode_select_dialog.setText("What type of footage are you tracking?")
    cancel_buton = mode_select_dialog.addButton("Cancel", QMessageBox.RejectRole)
    headfixed_button = mode_select_dialog.addButton("Headfixed", QMessageBox.YesRole)
    freeswimming_button = mode_select_dialog.addButton("Free-swimming", QMessageBox.NoRole)
     
    mode_select_dialog.exec_()

    # create controller
    if mode_select_dialog.clickedButton() == headfixed_button:
        controller = HeadfixedController()
        sys.exit(qApp.exec_())
    elif mode_select_dialog.clickedButton() == freeswimming_button:
        controller = FreeswimmingController()
        sys.exit(qApp.exec_())
    else:
        mode_select_dialog.close()
        qApp.quit()
