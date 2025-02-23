import sys
import cv2
import torch
import os
import subprocess
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QComboBox, QTextEdit
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class YOLO_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model_path = None
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.models_dir = "yolo_project/yolov5/custom_model/nuts_detection_pi/weights"
        self.load_model_list()

    def initUI(self):
        layout = QVBoxLayout()
        
        self.model_label = QLabel("Select Model:")
        layout.addWidget(self.model_label)

        self.model_selector = QComboBox()
        layout.addWidget(self.model_selector)
        
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        layout.addWidget(self.start_button)
        
        self.camera_button = QPushButton("Open Camera")
        self.camera_button.clicked.connect(self.open_camera)
        layout.addWidget(self.camera_button)
        
        self.labeling_button = QPushButton("Start Labeling")
        self.labeling_button.clicked.connect(self.start_labeling)
        layout.addWidget(self.labeling_button)
        
        self.training_button = QPushButton("Start Training")
        self.training_button.clicked.connect(self.start_training)
        layout.addWidget(self.training_button)
        
        self.count_label = QLabel("Objects Detected: 0")
        layout.addWidget(self.count_label)
        
        self.video_label = QLabel()
        layout.addWidget(self.video_label)
        
        self.setLayout(layout)
        self.setWindowTitle("YOLO Detection GUI")
        self.setGeometry(100, 100, 800, 600)
        
    def load_model_list(self):
        if os.path.exists(self.models_dir):
            models = [f for f in os.listdir(self.models_dir) if f.endswith(".pt")]
            self.model_selector.addItems(models)
    
    def start_detection(self):
        selected_model = self.model_selector.currentText()
        if not selected_model:
            self.count_label.setText("No model selected")
            return
        
        self.model_path = os.path.join(self.models_dir, selected_model)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 20)
            
        self.timer.start(50)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame)
            detected_objects = len(results.pred[0])
            self.count_label.setText(f"Objects Detected: {detected_objects}")
            
            for det in results.pred[0]:
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_image))
    
    def open_camera(self):
        os.system("libcamera-hello")
    
    def start_labeling(self):
        os.system("labelImg")
    
    def start_training(self):
        os.system("python yolo_project/yolov5/train.py --img 512 --batch 8 --epochs 50 --data yolov5/data.yaml --weights yolov5s.pt")
    
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLO_GUI()
    window.show()
    sys.exit(app.exec_())
