import sys
import cv2
import torch
import os
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class YOLO_GUI(QWidget):
    def __init__(self):
        super().__init__()

        # ????????? ??????
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "yolov5", "custom_model", "nuts_detection_pi", "weights", "best.pt")
        self.img_size = 512
        self.conf_thres = 0.15
        self.iou_thres = 0.25
        self.frame_rate = 20

        # ???????? ?????? YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.path.dirname(__file__), "..", "yolov5", "custom_model", "nuts_detection_pi", "weights", "best.pt"), force_reload=True)

        self.model.conf = self.conf_thres
        self.model.iou = self.iou_thres

        # ... (????????? ??? ???????? ??? ?????????)

    def change_model(self):
        model_name = self.model_selector.currentText()
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "yolov5", "custom_model", "nuts_detection_pi", "weights", f"{model_name}.pt")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
        self.model.conf = self.conf_thres
        self.model.iou = self.iou_thres

    # ... (????????? ?????? ???????? ??? ?????????)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLO_GUI()
    window.show()
    sys.exit(app.exec_())
