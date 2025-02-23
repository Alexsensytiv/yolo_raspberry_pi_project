import sys
import os
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class YOLO_GUI(QWidget):
    def __init__(self):
        super().__init__()

        # Path to the model
        model_path = "yolo_project/yolov5/custom_model/nuts_detection_pi/weights/best.pt"

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found! Please check the path.")

        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

        # GUI setup
        self.setWindowTitle("Nut Detection")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.count_label = QLabel("Detected Nuts: 0", self)

        self.start_button = QPushButton("Start Detection", self)
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_detection)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        # Initialize camera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_objects)

    def start_detection(self):
        """Start video capture and object detection."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)  # Open camera
        self.timer.start(30)  # Run detection every 30ms

    def stop_detection(self):
        """Stop video capture and clear display."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None  # Release camera resource
        self.label.clear()

    def detect_objects(self):
        """Capture frame, run object detection, and display results."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLOv5 inference
        results = self.model(rgb_frame)

        # Count detected objects (if any)
        count = len([obj for obj in results.xyxy[0] if len(obj) > 0 and obj[-1] == 0])

        # Draw bounding boxes
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0 and conf > 0.15:  # Check confidence threshold
                cv2.rectangle(rgb_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

        # Convert frame to QImage for PyQt5
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(rgb_frame, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

        # Update object count label
        self.count_label.setText(f"Detected Nuts: {count}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLO_GUI()
    window.show()
    sys.exit(app.exec_())
