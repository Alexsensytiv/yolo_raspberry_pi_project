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

        # Model parameters
        self.model_path = "yolo_project/yolov5/custom_model/nuts_detection_pi/weights/best.pt"
        self.img_size = 512  # Image size
        self.conf_thres = 0.15  # Confidence threshold
        self.iou_thres = 0.25  # IoU threshold
        self.frame_rate = 20  # FPS

        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path=self.model_path, force_reload=True)
        self.model.conf = self.conf_thres
        self.model.iou = self.iou_thres

        # GUI setup
        self.setWindowTitle("Nut Detection GUI")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setFixedSize(self.img_size, 480)

        self.count_label = QLabel("Detected Nuts: 0", self)

        # Buttons
        self.start_button = QPushButton("Start Detection", self)
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton("Stop Detection", self)
        self.stop_button.clicked.connect(self.stop_detection)

        self.open_camera_button = QPushButton("Open Camera", self)
        self.open_camera_button.clicked.connect(self.open_camera)

        self.start_labeling_button = QPushButton("Start Labeling", self)
        self.start_labeling_button.clicked.connect(self.start_labeling)

        self.start_training_button = QPushButton("Start Training", self)
        self.start_training_button.clicked.connect(self.start_training)

        # Model selector
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(["best", "last", "custom"])
        self.model_selector.currentIndexChanged.connect(self.change_model)

        # Layout setup
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.model_selector)
        left_layout.addWidget(self.open_camera_button)
        left_layout.addWidget(self.start_labeling_button)
        left_layout.addWidget(self.start_training_button)
        left_layout.addStretch()

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.start_button)
        bottom_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.label)
        main_layout.addWidget(self.count_label)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_objects)

    def start_detection(self):
        self.timer.start(self.frame_rate)

    def stop_detection(self):
        self.timer.stop()
        self.cap.release()
        self.label.clear()

    def detect_objects(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame = cv2.resize(frame, (self.img_size, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_frame, size=self.img_size)

        count = sum(1 for obj in results.xyxy[0] if obj[-1] == 0)
        
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0 and conf >= self.conf_thres:
                cv2.rectangle(rgb_frame, (int(xyxy[0]), int(xyxy[1])), 
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))
        self.count_label.setText(f"Detected Nuts: {count}")

    def open_camera(self):
        os.system("cheese &")
    
    def start_labeling(self):
        os.system("labelImg &")
    
    def start_training(self):
        os.system("python train.py --data yolov5/custom_model/data.yaml --weights yolov5s.pt --epochs 50")
    
    def change_model(self):
        model_name = self.model_selector.currentText()
        self.model_path = f"yolov5/custom_model/nuts_detection_pi/weights/{model_name}.pt"
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                    path=self.model_path, force_reload=True)
        self.model.conf = self.conf_thres
        self.model.iou = self.iou_thres

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLO_GUI()
    window.show()
    sys.exit(app.exec_())
