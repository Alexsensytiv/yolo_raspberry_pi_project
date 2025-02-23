import sys
import os
import cv2
import torch
import subprocess
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

sys.path.append(os.path.expanduser("~/yolo_project/yolov5"))
sys.path.append(os.path.expanduser("~/yolo_project/scripts"))

class YOLO_GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLOv5 GUI - Raspberry Pi 5")
        self.setGeometry(100, 100, 900, 700)

        # Buttons
        self.start_button = QPushButton("Start YOLO")
        self.start_button.clicked.connect(self.start_detection)

        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)

        self.labelimg_button = QPushButton("Open LabelImg")
        self.labelimg_button.clicked.connect(self.open_labelimg)

        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.train_model)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close_app)

        # Video feed label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        self.count_label = QLabel("Detected Nuts: 0")
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.labelimg_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.exit_button)
        layout.addWidget(self.log_output)
        self.setLayout(layout)
         # Load YOLOv5 model
        self.model_path = os.path.expanduser("~/yolo_project/yolov5/custom_model/nuts_detection_pi/weights/best.pt")
        if not os.path.exists(self.model_path):
            self.log_output.append(f"? Model file not found: {self.model_path}")
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
            self.model.conf = 0.5

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_detection(self):
        """Start YOLO detection"""
        self.timer.start(30)

    def update_frame(self):
        """Capture frame, run YOLO, and display in GUI"""
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.model(frame)
        nut_count = sum(1 for *_, conf, cls in results.xyxy[0] if int(cls) == 0)

        for x1, y1, x2, y2, conf, cls in results.xyxy[0]:
            if int(cls) == 0:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        self.count_label.setText(f"Detected Nuts: {nut_count}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def capture_image(self):
        """Save captured image"""
        ret, frame = self.cap.read()
        if not ret:
            return

        dataset_path = os.path.expanduser("~/dataset/images")
        os.makedirs(dataset_path, exist_ok=True)

        filename = os.path.join(dataset_path, f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)
        self.log_output.append(f"?? Image saved: {filename}")

    def open_labelimg(self):
        """Open LabelImg for annotation"""
        subprocess.Popen(["python", "labelImg.py"])
        self.log_output.append("?? LabelImg opened.")

    def train_model(self):
        """Start YOLO model training"""
        self.log_output.append("?? Starting YOLO training...")
        train_command = [
            "python", "train.py",
            "--img", "512",
            "--batch", "16",
            "--epochs", "50",
            "--data", "dataset/dataset.yaml",
            "--weights", "yolov5n.pt"
        ]
        subprocess.Popen(train_command)
        self.log_output.append("? Training started!")

    def close_app(self):
        """Exit application"""
        self.timer.stop()
        self.cap.release()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLO_GUI()
    window.show()
    sys.exit(app.exec_())
