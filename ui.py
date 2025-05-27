import sys
import os
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from reportlab.pdfgen import canvas

from main import run_prediction  # Import your prediction function from main.py


class OsteoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Osteoporosis Detection UI")
        self.setStyleSheet("background-color: #2d2d30; color: white;")
        self.image_path = ""
        self.result = {}

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Label for showing image path
        self.path_label = QLabel("Selected image path: None")
        self.path_label.setStyleSheet(
            "padding: 5px; font-size: 18px; background-color: #3c3f41; border-radius: 10px;")
        layout.addWidget(self.path_label)

        # Grid for image and results
        grid = QGridLayout()

        # Image display button (clickable to open image externally)
        self.image_label = QPushButton("No Image Selected")
        self.image_label.setStyleSheet(
            "background-color: #3c6eb4; border-radius: 15px; font-size: 18px; padding: 10px;")
        self.image_label.setFixedSize(400, 400)
        self.image_label.clicked.connect(self.open_image_viewer)
        grid.addWidget(self.image_label, 0, 0, 11, 1)  # Increased row span for new fields

        # Result fields
        self.result_labels = {}
        result_fields = [
            "Bone type",
            "Prediction",
            "Description",
            "Estimated BMD",
            "Confidence score",
            "Inference time",
            "Bone Type Model Accuracy",      # New field
            "Osteoporosis Model Accuracy"      # New field
        ]

        for i, field in enumerate(result_fields):
            label_btn = QPushButton(field)
            label_btn.setStyleSheet("background-color: #3c6eb4; border-radius: 10px; font-size: 20px; padding: 8px;")  # increased font-size & padding
            label_val = QLabel("N/A")
            label_val.setStyleSheet("background-color: #3c6eb4; border-radius: 10px; padding: 8px; font-size: 22px; font-weight: bold;")  # bigger font + bold
            grid.addWidget(label_btn, i, 1)
            grid.addWidget(label_val, i, 2)
            self.result_labels[field] = label_val


        layout.addLayout(grid)

        # Buttons below
        btn_layout = QHBoxLayout()

        self.select_btn = QPushButton("Select Image")
        self.select_btn.setStyleSheet(
            "background-color: #3c6eb4; border-radius: 10px; font-size: 18px; padding: 5px;")
        self.select_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(self.select_btn)

        self.report_btn = QPushButton("Save PDF Report")
        self.report_btn.setStyleSheet(
            "background-color: #3c6eb4; border-radius: 10px; font-size: 18px; padding: 5px;")
        self.report_btn.clicked.connect(self.save_pdf_report)
        btn_layout.addWidget(self.report_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            self.path_label.setText(f"Selected image path: {file_name}")

            pixmap = QPixmap(file_name).scaled(200, 400, Qt.KeepAspectRatio)
            self.image_label.setIconSize(pixmap.size())
            self.image_label.setIcon(QIcon(pixmap))
            self.image_label.setText("")  # Clear the button text once image is shown

            self.run_model()

    def run_model(self):
        if not self.image_path:
            return

        start_time = time.time()
        self.result = run_prediction(self.image_path)
        inference_time = time.time() - start_time

        # Update UI with results
        self.result_labels["Bone type"].setText(str(self.result.get("bone_type", "N/A")))
        self.result_labels["Prediction"].setText(str(self.result.get("condition", "N/A")))
        self.result_labels["Description"].setText(str(self.result.get("description", "N/A")))
        self.result_labels["Estimated BMD"].setText(str(self.result.get("estimated_bmd", "N/A")))
        self.result_labels["Confidence score"].setText(f"{self.result.get('confidence', 'N/A'):.2f}%")
        self.result_labels["Inference time"].setText(f"{inference_time:.2f} sec")

        # Set accuracy values; show N/A if not provided
        bone_acc = self.result.get("bone_accuracy")
        cond_acc = self.result.get("condition_accuracy")

        self.result_labels["Bone Type Model Accuracy"].setText(
            f"{bone_acc:.2f}%" if bone_acc is not None else "N/A"
        )
        self.result_labels["Osteoporosis Model Accuracy"].setText(
            f"{cond_acc:.2f}%" if cond_acc is not None else "N/A"
        )

    def open_image_viewer(self):
        if self.image_path:
            if os.name == "nt":
                os.startfile(self.image_path)
            elif sys.platform == "darwin":
                os.system(f"open '{self.image_path}'")
            else:
                os.system(f"xdg-open '{self.image_path}'")

    def save_pdf_report(self):
        if not self.result or not self.image_path:
            return

        pdf_path = os.path.splitext(self.image_path)[0] + "_report.pdf"
        c = canvas.Canvas(pdf_path)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 800, "Osteoporosis Detection Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 780, f"Image Path: {self.image_path}")

        y = 750
        for key, label in self.result_labels.items():
            c.drawString(50, y, f"{key}: {label.text()}")
            y -= 25

        c.save()
        print(f"PDF report saved: {pdf_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OsteoApp()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec_())