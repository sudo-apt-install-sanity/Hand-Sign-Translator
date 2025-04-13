import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
from gtts import gTTS
import pygame
import time
import threading
import queue
import os

class ISLRecognizer:
    def __init__(self):
        try:
            self.nn_model = load_model('custom_isl_dataset/isl_nn_model.h5')
            with open('custom_isl_dataset/label_encoder.pkl', 'rb') as f:
                self.le = pickle.load(f)
            print("[INFO] Models loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            exit()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )

        pygame.mixer.init()
        os.makedirs('tts_cache', exist_ok=True)
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()

        self.pred_history = deque(maxlen=8)
        self.last_prediction = ""
        self.cooldown = 1.5
        self.last_pred_time = 0

    def tts_worker(self):
        while True:
            text = self.tts_queue.get()
            if text:
                try:
                    audio_path = f'tts_cache/{text}.mp3'
                    if not os.path.exists(audio_path):
                        tts = gTTS(text=text, lang='en')
                        tts.save(audio_path)
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"[TTS ERROR] {e}")
            self.tts_queue.task_done()

    def text_to_speech(self, text):
        if text:
            self.tts_queue.put(text)

    def normalize_landmarks(self, landmarks):
        landmarks = np.array(landmarks)
        mean = landmarks.mean(axis=0)
        std = landmarks.std(axis=0) + 1e-6
        return ((landmarks - mean) / std).flatten()

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        current_time = time.time()
        prediction = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                raw_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                if len(raw_landmarks) == 21:
                    norm_landmarks = self.normalize_landmarks(raw_landmarks)
                    proba = self.nn_model.predict(norm_landmarks.reshape(1, -1), verbose=0)[0]
                    self.pred_history.append(proba)

                    if len(self.pred_history) >= 5:
                        avg_proba = np.mean(self.pred_history, axis=0)
                        pred_idx = np.argmax(avg_proba)
                        prediction = self.le.inverse_transform([pred_idx])[0]

                        if (current_time - self.last_pred_time > self.cooldown):
                            self.last_pred_time = current_time
                            if prediction != self.last_prediction:
                                self.last_prediction = prediction
                                self.text_to_speech(prediction)

        if prediction:
            display_text = f"Sign: {prediction}"
            cv2.putText(frame, display_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame

def main():
    recognizer = ISLRecognizer()
    cap = cv2.VideoCapture(0)

    print("[INFO] Starting ISL Recognition. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        processed_frame = recognizer.process_frame(frame)
        cv2.imshow('ISL Recognition', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()