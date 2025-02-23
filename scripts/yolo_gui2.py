import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QListWidget
from PyQt5.QtCore import Qt


class YOLO_GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Nut Detection GUI")
        self.setGeometry(100, 100, 900, 600)

        # Video stream placeholder
        self.video_label = QLabel("Video Stream", self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Control buttons
        self.camera_button = QPushButton("Camera", self)
        self.labeling_button = QPushButton("Labeling", self)
        self.training_button = QPushButton("Training", self)
        self.select_model_button = QPushButton("Select Model", self)

        # Model selection list
        self.model_list = QListWidget(self)
        self.model_list.addItems(["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"])

        # Detection buttons
        self.start_button = QPushButton("Start Detection", self)
        self.stop_button = QPushButton("Stop", self)

        # Left panel layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.camera_button)
        left_layout.addWidget(self.labeling_button)
        left_layout.addWidget(self.training_button)
        left_layout.addWidget(self.select_model_button)
        left_layout.addWidget(self.model_list)
        left_layout.addStretch()

        # Main layout (video display)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)

        # Bottom layout (start/stop buttons)
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.start_button)
        bottom_layout.addWidget(self.stop_button)

        main_layout.addLayout(bottom_layout)

        # Container for left panel and main video display
        main_container = QHBoxLayout()
        main_container.addLayout(left_layout)
        main_container.addLayout(main_layout)

        self.setLayout(main_container)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLO_GUI()
    window.show()
    sys.exit(app.exec_())
