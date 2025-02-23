import sys
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class YOLO_GUI(QWidget):
    def __init__(self):
        super().__init__()

        # Загрузка обученной модели YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                    path='yolo_project/yolov5/custom_model/nuts_detection_pi/weights/best.pt', 
                                    force_reload=True)  
        
        self.model.conf = 0.15  # Изменение порога уверенности
        self.model.iou = 0.25   # Изменение порога IoU

        # GUI
        self.setWindowTitle("Nut Detection")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setFixedSize(512, 480)

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

        # Захват видео
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_objects)

    def start_detection(self):
        self.timer.start(20)  # Обновление каждые 30 мс

    def stop_detection(self):
        self.timer.stop()
        self.cap.release()
        self.label.clear()

    def detect_objects(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Установка разрешения кадра в 512x512
        frame = cv2.resize(frame, (512, 480))

        # Конвертация в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Запуск модели YOLO
        results = self.model(rgb_frame, size=512)  

        # Подсчет объектов класса "nut"
        count = sum(1 for obj in results.xyxy[0] if obj[-1] == 0)  # 'nut' class ID = 0

        # Отрисовка bounding box
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0 and conf >= 0.15:  # Проверка confidence threshold
                cv2.rectangle(rgb_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

        # Конвертация кадра в QImage
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

        # Обновление счетчика
        self.count_label.setText(f"Detected Nuts: {count}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLO_GUI()
    window.show()
    sys.exit(app.exec_())
