# Reference: Hand-Digits-Recognition @ franneck94
# Link : https://github.com/franneck94/Hand-Digits-Recognition/blob/master/drawer/src/drawing_gui.py

import os
import sys
# Some imports related to the model
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

import wiki2bio_main

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = wiki2bio_main.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


