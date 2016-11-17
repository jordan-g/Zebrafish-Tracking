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

from controller import *

if __name__ == "__main__":
    qApp = QApplication(sys.argv)

    mode_select_dialog = QMessageBox()
    mode_select_dialog.setText("What type of footage are you tracking?")
    cancel_buton = mode_select_dialog.addButton("Cancel", QMessageBox.RejectRole)
    headfixed_button = mode_select_dialog.addButton("Headfixed", QMessageBox.YesRole)
    freeswimming_button = mode_select_dialog.addButton("Free-swimming", QMessageBox.NoRole)
     
    mode_select_dialog.exec_()
     
    if mode_select_dialog.clickedButton() == headfixed_button:
        print("Headfixed")
        # create controller
        controller = HeadfixedController()
    elif mode_select_dialog.clickedButton() == freeswimming_button:
        print("Free-swimming")
        # create controller
        controller = FreeswimmingController()
    else:
        mode_select_dialog.close()

    sys.exit(qApp.exec_())
