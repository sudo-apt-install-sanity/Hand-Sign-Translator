import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from collections import defaultdict
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

# Data storage
dataset = defaultdict(list)
current_label = None
collecting = False
frame_count = 0
samples_per_gesture = 100

# Create dataset directory
os.makedirs("custom_isl_dataset", exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 120)
print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    mean = landmarks.mean(axis=0)
    std = landmarks.std(axis=0) + 1e-6
    return ((landmarks - mean) / std).flatten()

# Key-to-label mapping (consistent with sign database)
key_label_map = {
    ord('h'): 'hello',
    ord('t'): 'thank you',
    ord('y'): 'yes',
    ord('n'): 'no',
    ord('p'): 'please',
    ord('i'): 'i love you',
    ord('f'): 'food',
    ord('w'): 'water',
    ord('s'): 'sorry',
    ord('g'): 'good morning',
    ord('j'): 'good night',
    ord('l'): 'friends',
    ord('z'): 'help',
    ord('a'): 'happy',
    ord('d'): 'sad',
    ord('k'): 'angry',
    ord('x'): 'tired',
    ord('c'): 'scared',
    ord('v'): 'excited',
    ord('b'): 'bored',
    ord('m'): 'save_dataset'
}

last_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            if len(raw_landmarks) == 21:
                norm_landmarks = normalize_landmarks(raw_landmarks)
                if collecting:
                    dataset[current_label].append(norm_landmarks)
                    frame_count += 1

                    if frame_count >= samples_per_gesture:
                        collecting = False
                        print(f"[INFO] Collected {len(dataset[current_label])} samples for '{current_label}'")

    # Display recording status
    if collecting:
        status = f"Recording '{current_label}'... {frame_count}/{samples_per_gesture}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, "Keys: H=Hello, T=Thank You, Y=Yes, N=No, P=Please, M=Save, Q=Quit", 
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Show FPS
    current_time = time.time()
    fps = 1 / (current_time - last_time)
    last_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('ISL Data Collector', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in key_label_map:
        label = key_label_map[key]
        if label == 'save_dataset':
            with open('custom_isl_dataset/isl_data.pkl', 'wb') as f:
                pickle.dump(dict(dataset), f)
            print(f"[INFO] Dataset saved with {len(dataset)} gestures.")
        else:
            current_label = label
            collecting = True
            frame_count = 0
            print(f"[INFO] Started recording for '{current_label}'")

cap.release()
cv2.destroyAllWindows()