# Reference: Hand-Digits-Recognition @ franneck94
# Link : https://github.com/franneck94/Hand-Digits-Recognition/blob/master/drawer/src/drawing_gui.py

import os
import sys
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import uic

# Some imports related to the model
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

import wiki2bio

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = wiki2bio.Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


