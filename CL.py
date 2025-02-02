from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage
from yolov5.detect import run
import cv2

class Ui_CephalometricLandmark(QWidget):
    def __init__(self, windows):
        super().__init__()
        self.windows = windows
        self.windows.setObjectName("CephalometricLandmark")
        self.windows.setEnabled(True)
        #self.windows.resize(505, 650)
        self.windows.resize(572, 768)
        self.centralwidget = QtWidgets.QWidget(self.windows)
        self.centralwidget.setObjectName("centralwidget")
    
    def setupUi(self):
        self.brows_btn = QtWidgets.QPushButton(self.centralwidget)
        self.brows_btn.setGeometry(QtCore.QRect(10, 20, 75, 23))
        self.brows_btn.setObjectName("brows_btn")
        self.brows_btn.clicked.connect(self.__brows_btn_click)
        self.FileName_tbox = QtWidgets.QLineEdit(self.centralwidget)
        self.FileName_tbox.setEnabled(False)
        self.FileName_tbox.setGeometry(QtCore.QRect(90, 20, 471, 20)    )
        self.FileName_tbox.setText("")
        self.FileName_tbox.setObjectName("FileName_tbox")
        self.FileName_tbox.textChanged.connect(self.__FileName_text_change)
        self.Image_View = QtWidgets.QGraphicsView(self.centralwidget)
        self.Image_View.setGeometry(QtCore.QRect(10, 110, 551, 621))
        self.Image_View.setSizeIncrement(QtCore.QSize(0, 0))
        self.Image_View.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Image_View.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.Image_View.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.Image_View.setObjectName("Image_View")
        self.confidence_label = QtWidgets.QLabel(self.centralwidget)
        self.confidence_label.setGeometry(QtCore.QRect(10, 50, 121, 16))
        self.confidence_label.setObjectName("confidence_label")
        self.IOU_label = QtWidgets.QLabel(self.centralwidget)
        self.IOU_label.setGeometry(QtCore.QRect(43, 80, 81, 16))
        self.IOU_label.setObjectName("IOU_label")
        self.Apply_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Apply_btn.setGeometry(QtCore.QRect(480, 64, 75, 23))
        self.Apply_btn.setEnabled(False)
        self.Apply_btn.setObjectName("Apply_btn")
        self.Apply_btn.clicked.connect(self.__run_inference)
        self.Exit_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Exit_btn.setGeometry(QtCore.QRect(480, 740, 81, 23))
        self.Exit_btn.setObjectName("Exit_btn")
        self.Exit_btn.clicked.connect(self.__exit_btn_click)
        self.Confodence_Slider = QtWidgets.QSlider(self.centralwidget)
        self.Confodence_Slider.setGeometry(QtCore.QRect(170, 50, 301, 22))
        self.Confodence_Slider.setMaximum(100)
        self.Confodence_Slider.setProperty("value", 50)
        self.Confodence_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Confodence_Slider.setObjectName("Confodence_Slider")
        self.Confodence_Slider.valueChanged.connect( \
            lambda: self.confidence_value.setText(str(self.Confodence_Slider.value()/100)))
        self.IOU_Slider = QtWidgets.QSlider(self.centralwidget)
        self.IOU_Slider.setGeometry(QtCore.QRect(170, 80, 301, 22))
        self.IOU_Slider.setMaximum(100)
        self.IOU_Slider.setProperty("value", 40)
        self.IOU_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.IOU_Slider.setObjectName("IOU_Slider")
        self.IOU_Slider.valueChanged.connect( \
            lambda: self.IOU_value.setText(str(self.IOU_Slider.value()/100)))
        self.confidence_value = QtWidgets.QLabel(self.centralwidget)
        self.confidence_value.setGeometry(QtCore.QRect(130, 54, 47, 13))
        self.confidence_value.setAlignment(QtCore.Qt.AlignCenter)
        self.confidence_value.setObjectName("confidence_value")
        self.IOU_value = QtWidgets.QLabel(self.centralwidget)
        self.IOU_value.setGeometry(QtCore.QRect(130, 83, 47, 13))
        self.IOU_value.setAlignment(QtCore.Qt.AlignCenter)
        self.IOU_value.setObjectName("IOU_value")

        self.retranslateUi(self.windows)
        
        self.windows.setTabOrder(self.brows_btn, self.FileName_tbox)
        self.windows.setTabOrder(self.FileName_tbox, self.Confodence_Slider)
        self.windows.setTabOrder(self.Confodence_Slider, self.IOU_Slider)
        self.windows.setTabOrder(self.IOU_Slider, self.Apply_btn)
        self.windows.setTabOrder(self.Apply_btn, self.Image_View)
        self.windows.setTabOrder(self.Image_View, self.Exit_btn)
        

    def retranslateUi(self, windows):
        _translate = QtCore.QCoreApplication.translate
        windows.setWindowTitle(_translate("CephalometricLandmark", "Cephalometric Landmark"))
        windows.setWindowIcon(QtGui.QIcon('cl.ico'))
        self.brows_btn.setText(_translate("CephalometricLandmark", "Browse"))
        self.confidence_label.setText(_translate("CephalometricLandmark", "Confidence Threshold :"))
        self.IOU_label.setText(_translate("CephalometricLandmark", "IOU Threshold :"))
        self.Apply_btn.setText(_translate("CephalometricLandmark", "Apply"))
        self.Exit_btn.setText(_translate("CephalometricLandmark", "Exit"))
        self.confidence_value.setText(_translate("CephalometricLandmark", "0.5"))
        self.IOU_value.setText(_translate("CephalometricLandmark", "0.4"))
    
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open Image", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.FileName_tbox.setText(fileName)
        scene = QtWidgets.QGraphicsScene(self)
        photo = QPixmap(fileName)
        photo_item = QtWidgets.QGraphicsPixmapItem(photo)
        scene.addItem(photo_item)
        self.Image_View.fitInView(photo_item)
        self.Image_View.setScene(scene)
        self.Image_View.fitInView(photo_item)
        self.Image_View.setScene(scene)
        
        
    def __brows_btn_click(self):
        self.openFileNameDialog()
    
    def __exit_btn_click(self):
        self.windows.close()
    
    def __run_inference(self):
        img, points = run(weights='best.pt', conf_thres=float(self.confidence_value.text()), hide_conf=True, exist_ok=True, \
                  source = self.FileName_tbox.text(), iou_thres=float(self.IOU_value.text()), max_det=20, augment=True)
        print(len(points))
        print(points)
        #print(self.FileName_tbox.text().split('/')[-1].replace('.tif','.jpg'))
        #cv2.imwrite('res.jpg', img)
        #result_filename = self.FileName_tbox.text().split('/')[-1].replace('.tif','.jpg')
        result_filename = self.FileName_tbox.text().replace('.tif','.jpg')
        cv2.imwrite(result_filename, img)
        scene = QtWidgets.QGraphicsScene(self)
        photo = QPixmap(result_filename)
        photo_item = QtWidgets.QGraphicsPixmapItem(photo)
        scene.addItem(photo_item)
        self.Image_View.fitInView(photo_item)
        self.Image_View.setScene(scene)
    
    def __FileName_text_change(self):
        if (self.FileName_tbox.text() == ''):
            self.Apply_btn.setEnabled(False)
        else:
            self.Apply_btn.setEnabled(True)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QDialog()
    ui = Ui_CephalometricLandmark(window)
    ui.setupUi()
    window.show()
    sys.exit(app.exec_())