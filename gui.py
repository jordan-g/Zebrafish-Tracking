import sys
import os
import tracking as tt

from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class DynamicCanvas(PlotCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        PlotCanvas.__init__(self, *args, **kwargs)
        # timer = QtCore.QTimer(self)
        # timer.timeout.connect(self.update_figure)
        # timer.start(1000)

    def compute_initial_figure(self):
        # self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        pass

    def update_image(self, image):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        self.axes.imshow(image, cmap='gray', interpolation='none')
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.fig.tight_layout()
        plt.axis('off')
        self.draw()

class PlotWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Preview")

        self.main_widget = QtGui.QWidget(self)

        l = QtGui.QVBoxLayout(self.main_widget)
        self.dc = DynamicCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(self.dc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def set_image(self, image):
        self.dc.update_image(image)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

class ParamWindow(QtGui.QMainWindow):
    def __init__(self, plot_window):
        QtGui.QMainWindow.__init__(self)

        self.plot_window = plot_window

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('File dialog')
        self.show()

    def showDialog(self):
        fname = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                '/home'))
        shrink_factor = 1.0
        offsets = None
        crops = None

        print(fname, type(fname))

        image, cropped_images, y_offsets, x_offsets, image_dir, image_name = tt.load_image(fname, shrink_factor, offsets=offsets, crops=crops)

        self.plot_window.set_image(cropped_images[0])

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

qApp = QtGui.QApplication(sys.argv)

plot_window = PlotWindow()
plot_window.setWindowTitle("Preview")
plot_window.show()

param_window = ParamWindow(plot_window)
param_window.setWindowTitle("Parameters")
param_window.show()

sys.exit(qApp.exec_())
