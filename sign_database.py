from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
import os

# Auto-generate the database from captured images
def build_sign_database():
    database = {}
    if os.path.exists("sign_images"):
        for filename in os.listdir("sign_images"):
            if filename.endswith(".png"):
                sign_name = filename[:-4].replace("_", " ")
                database[sign_name] = os.path.join("sign_images", filename)
    return database

SIGN_DATABASE = build_sign_database()

class SignDisplay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("""
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #ecf0f1;
            border-radius: 10px;
        """)
        
    def show_sign(self, text):
        text = text.lower().strip()
        if text in SIGN_DATABASE:
            image_path = SIGN_DATABASE[text]
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                self.setPixmap(pixmap.scaled(
                    self.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                ))
            else:
                self.clear()
                self.setText(f"Image missing: {image_path}")
        else:
            self.clear()
            self.setText(f"No sign for: {text}")