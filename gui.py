import sys
import subprocess
import speech_recognition as sr
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QLabel, QSpacerItem, QSizePolicy,
    QHBoxLayout, QLineEdit, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPalette, QColor, QFont
from sign_database import SignDisplay

class Worker(QThread):
    log = pyqtSignal(str)

    def __init__(self, script_name):
        super().__init__()
        self.script_name = script_name

    def run(self):
        try:
            process = subprocess.Popen(
                [sys.executable, self.script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            for line in iter(process.stdout.readline, ''):
                self.log.emit(line)
            process.stdout.close()
            process.wait()
        except Exception as e:
            self.log.emit(f"[ERROR] {e}")

class ISLGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translator")
        self.setGeometry(200, 100, 800, 600)
        self.setup_ui()
        
        # Audio setup
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.is_listening = False
        self.audio_level = 0
        self.audio_timer = QTimer(self)
        self.audio_timer.timeout.connect(self.update_audio_level)

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2c3e50, stop:1 #3498db
                );
                color: white;
                font-family: 'Segoe UI';
            }
            QLabel#titleLabel {
                font-size: 32px;
                font-weight: bold;
                color: #ffffff;
                margin-bottom: 20px;
            }
            QPushButton {
                background-color: #1abc9c;
                border: none;
                border-radius: 12px;
                padding: 12px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #16a085;
                color: #ecf0f1;
            }
            QPushButton#voiceButton {
                background-color: #e74c3c;
            }
            QPushButton#voiceButton:hover {
                background-color: #c0392b;
            }
            QPushButton#voiceButton:disabled {
                background-color: #7f8c8d;
            }
            QTextEdit, QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid #ecf0f1;
                border-radius: 10px;
                padding: 10px;
                color: #f1f1f1;
            }
            QLineEdit {
                font-size: 16px;
            }
            QProgressBar {
                border: 2px solid #ecf0f1;
                border-radius: 5px;
                text-align: center;
                background: rgba(0, 0, 0, 0.2);
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2ecc71, stop:1 #f1c40f
                );
            }
        """)

        # Title
        self.title = QLabel("Sign Language Translator")
        self.title.setObjectName("titleLabel")
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        # Action Buttons
        self.btn_collect = QPushButton("🧠 Collect Training Data")
        self.btn_collect.clicked.connect(lambda: self.run_script("collect1.py"))
        layout.addWidget(self.btn_collect)

        self.btn_train = QPushButton("⚙️ Train Model")
        self.btn_train.clicked.connect(lambda: self.run_script("train1.py"))
        layout.addWidget(self.btn_train)

        self.btn_recognize = QPushButton("📹 Start Recognition")
        self.btn_recognize.clicked.connect(lambda: self.run_script("recognize1.py"))
        layout.addWidget(self.btn_recognize)

        # Translation Section
        translation_layout = QHBoxLayout()
        
        self.translation_input = QLineEdit()
        self.translation_input.setPlaceholderText("Enter text or use voice input...")
        
        self.btn_translate = QPushButton("Translate to Sign")
        self.btn_translate.clicked.connect(self.show_sign_translation)
        
        self.btn_voice = QPushButton("🎤 Voice Input")
        self.btn_voice.setObjectName("voiceButton")
        self.btn_voice.clicked.connect(self.toggle_voice_input)
        
        translation_layout.addWidget(self.translation_input, 70)
        translation_layout.addWidget(self.btn_translate, 15)
        translation_layout.addWidget(self.btn_voice, 15)

        # Audio Level Indicator
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setValue(0)
        self.audio_level_bar.setTextVisible(False)
        self.audio_level_bar.setFixedHeight(8)
        self.audio_level_bar.hide()

        layout.addLayout(translation_layout)
        layout.addWidget(self.audio_level_bar)

        # Sign Display
        self.sign_display = SignDisplay()
        layout.addWidget(self.sign_display)

        # Log Area
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def toggle_voice_input(self):
        if not self.is_listening:
            self.start_voice_input()
        else:
            self.stop_voice_input()

    def start_voice_input(self):
        self.is_listening = True
        self.btn_voice.setText("Listening...")
        self.btn_voice.setStyleSheet("background-color: #2ecc71;")
        self.audio_level_bar.show()
        self.audio_timer.start(50)  # Update every 50ms
        
        # Start listening in background
        QTimer.singleShot(100, self.process_voice_input)

    def stop_voice_input(self):
        self.is_listening = False
        self.btn_voice.setText("🎤 Voice Input")
        self.btn_voice.setStyleSheet("background-color: #e74c3c;")
        self.audio_level_bar.hide()
        self.audio_timer.stop()

    def update_audio_level(self):
        # Simulate audio level (replace with real audio processing if needed)
        self.audio_level = min(100, self.audio_level + np.random.randint(-5, 10))
        if self.audio_level < 0:
            self.audio_level = 0
        self.audio_level_bar.setValue(self.audio_level)

    def process_voice_input(self):
        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source)
                self.log_box.append("[INFO] Listening... Speak now")
                audio = self.recognizer.listen(source, timeout=5)
                
            text = self.recognizer.recognize_google(audio)
            self.translation_input.setText(text)
            self.show_sign_translation()
            self.log_box.append(f"[INFO] Recognized: {text}")
            
        except sr.WaitTimeoutError:
            self.log_box.append("[INFO] Listening timeout")
        except sr.UnknownValueError:
            self.log_box.append("[ERROR] Could not understand audio")
        except sr.RequestError as e:
            self.log_box.append(f"[ERROR] Recognition error: {e}")
        except Exception as e:
            self.log_box.append(f"[ERROR] {str(e)}")
        finally:
            self.stop_voice_input()

    def show_sign_translation(self):
        text = self.translation_input.text()
        if text:
            self.sign_display.show_sign(text)

    def run_script(self, script_name):
        self.log_box.append(f"[INFO] Running {script_name}...\n")
        self.worker = Worker(script_name)
        self.worker.log.connect(self.update_log)
        self.worker.start()

    def update_log(self, text):
        self.log_box.append(text)

    def closeEvent(self, event):
        self.stop_voice_input()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ISLGui()
    gui.show()
    sys.exit(app.exec_())