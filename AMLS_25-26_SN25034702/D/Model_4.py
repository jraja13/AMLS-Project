# Build a neural network with tensorflow to classify breast cancer images
# this file will be imported in main.py
import numpy as np
import time
from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models

def model_4(data):
    """
    data: dictionary with keys 'X_train', 'y_train', 'X_val', 'y_val'
    """

    # ----------------------------------
    # 1. Prepare data
    # ----------------------------------
    X_train = data['X_train'].astype('float32') / 255.0
    y_train = data['y_train'].flatten()
    X_val = data['X_val'].astype('float32') / 255.0
    y_val = data['y_val'].flatten()

    # Add channel dimension if needed
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

    # ----------------------------------
    # 2. Build CNN model
    # ----------------------------------
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # ----------------------------------
    # 3. Train the model
    # ----------------------------------
    start_time = time.time()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    training_time = time.time() - start_time

    # ----------------------------------
    # 4. Evaluate the model
    # --------------------------------
    start_time = time.time()
    y_pred_probs = model.predict(X_val)
    prediction_time = time.time() - start_time
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    # Return metrics as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
        "prediction_time": prediction_time
    }

