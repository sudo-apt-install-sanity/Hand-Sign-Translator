import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os

def load_custom_dataset():
    try:
        with open('custom_isl_dataset/isl_data.pkl', 'rb') as f:
            data = pickle.load(f)

        X, y = [], []
        for label, samples in data.items():
            X.extend(samples)
            y.extend([label] * len(samples))

        return np.array(X), np.array(y)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def normalize_dataset(X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)

def train_neural_network(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    X = normalize_dataset(X)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(
        X, y_onehot,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=12, restore_best_weights=True)]
    )

    model.save('custom_isl_dataset/isl_nn_model.h5')
    with open('custom_isl_dataset/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("[INFO] Neural network model saved successfully.")
    return model

if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    X, y = load_custom_dataset()

    if X is None or len(X) == 0:
        print("[ERROR] No valid data found. Please collect data first.")
    else:
        print(f"[INFO] Dataset loaded: {len(X)} samples, {len(np.unique(y))} classes")
        train_neural_network(X, y)