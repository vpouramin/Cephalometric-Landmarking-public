# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CephalometricLandmark(object):
    def setupUi(self, CephalometricLandmark):
        CephalometricLandmark.setObjectName("CephalometricLandmark")
        CephalometricLandmark.setEnabled(True)
        CephalometricLandmark.resize(572, 768)
        CephalometricLandmark.setToolTipDuration(7)
        self.centralwidget = QtWidgets.QWidget(CephalometricLandmark)
        self.centralwidget.setObjectName("centralwidget")
        self.brows_btn = QtWidgets.QPushButton(self.centralwidget)
        self.brows_btn.setGeometry(QtCore.QRect(10, 20, 75, 23))
        self.brows_btn.setObjectName("brows_btn")
        self.FileName_tbox = QtWidgets.QLineEdit(self.centralwidget)
        self.FileName_tbox.setEnabled(False)
        self.FileName_tbox.setGeometry(QtCore.QRect(90, 20, 471, 20))
        self.FileName_tbox.setText("")
        self.FileName_tbox.setObjectName("FileName_tbox")
        self.Image_View = QtWidgets.QGraphicsView(self.centralwidget)
        self.Image_View.setGeometry(QtCore.QRect(10, 110, 551, 621))
        self.Image_View.setObjectName("Image_View")
        self.confidence_label = QtWidgets.QLabel(self.centralwidget)
        self.confidence_label.setGeometry(QtCore.QRect(10, 50, 121, 16))
        self.confidence_label.setObjectName("confidence_label")
        self.IOU_label = QtWidgets.QLabel(self.centralwidget)
        self.IOU_label.setGeometry(QtCore.QRect(43, 80, 81, 16))
        self.IOU_label.setObjectName("IOU_label")
        self.Apply_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Apply_btn.setGeometry(QtCore.QRect(480, 64, 75, 23))
        self.Apply_btn.setObjectName("Apply_btn")
        self.Exit_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Exit_btn.setGeometry(QtCore.QRect(480, 740, 81, 23))
        self.Exit_btn.setObjectName("Exit_btn")
        self.Confodence_Slider = QtWidgets.QSlider(self.centralwidget)
        self.Confodence_Slider.setGeometry(QtCore.QRect(170, 50, 301, 22))
        self.Confodence_Slider.setMaximum(100)
        self.Confodence_Slider.setProperty("value", 50)
        self.Confodence_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Confodence_Slider.setObjectName("Confodence_Slider")
        self.IOU_Slider = QtWidgets.QSlider(self.centralwidget)
        self.IOU_Slider.setGeometry(QtCore.QRect(170, 80, 301, 22))
        self.IOU_Slider.setMaximum(100)
        self.IOU_Slider.setProperty("value", 40)
        self.IOU_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.IOU_Slider.setObjectName("IOU_Slider")
        self.confidence_value = QtWidgets.QLabel(self.centralwidget)
        self.confidence_value.setGeometry(QtCore.QRect(125, 54, 47, 13))
        self.confidence_value.setAlignment(QtCore.Qt.AlignCenter)
        self.confidence_value.setObjectName("confidence_value")
        self.IOU_value = QtWidgets.QLabel(self.centralwidget)
        self.IOU_value.setGeometry(QtCore.QRect(125, 83, 47, 13))
        self.IOU_value.setAlignment(QtCore.Qt.AlignCenter)
        self.IOU_value.setObjectName("IOU_value")
        CephalometricLandmark.setCentralWidget(self.centralwidget)

        self.retranslateUi(CephalometricLandmark)
        QtCore.QMetaObject.connectSlotsByName(CephalometricLandmark)
        CephalometricLandmark.setTabOrder(self.brows_btn, self.FileName_tbox)
        CephalometricLandmark.setTabOrder(self.FileName_tbox, self.Confodence_Slider)
        CephalometricLandmark.setTabOrder(self.Confodence_Slider, self.IOU_Slider)
        CephalometricLandmark.setTabOrder(self.IOU_Slider, self.Apply_btn)
        CephalometricLandmark.setTabOrder(self.Apply_btn, self.Image_View)
        CephalometricLandmark.setTabOrder(self.Image_View, self.Exit_btn)

    def retranslateUi(self, CephalometricLandmark):
        _translate = QtCore.QCoreApplication.translate
        CephalometricLandmark.setWindowTitle(_translate("CephalometricLandmark", "MainWindow"))
        self.brows_btn.setText(_translate("CephalometricLandmark", "Browse"))
        self.confidence_label.setText(_translate("CephalometricLandmark", "Confidence Threshold :"))
        self.IOU_label.setText(_translate("CephalometricLandmark", "IOU Threshold :"))
        self.Apply_btn.setText(_translate("CephalometricLandmark", "Apply"))
        self.Exit_btn.setText(_translate("CephalometricLandmark", "Exit"))
        self.confidence_value.setText(_translate("CephalometricLandmark", "0.5"))
        self.IOU_value.setText(_translate("CephalometricLandmark", "0.4"))
