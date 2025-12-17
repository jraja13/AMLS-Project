import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def model_2(images):
    X_train, y_train = images['X_train'], images['y_train']
    X_test, y_test   = images['X_test'], images['y_test']

    # Reshape grayscale images and normalize
    X_train = X_train.reshape(-1,28,28,1).astype('float32') / 255.0
    X_test  = X_test.reshape(-1,28,28,1).astype('float32') / 255.0
    
    num_classes = len(np.unique(y_train))

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:],
               kernel_regularizer=regularizers.l2(0.0001)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu',
               kernel_regularizer=regularizers.l2(0.0001)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        Dropout(0.5),   
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    start_time = time.time()
    model.fit(
        X_train, y_train,
        epochs=75,
        batch_size=32,
        verbose=2
    )
    training_time = time.time() - start_time

    start_pred = time.time()
    y_pred = np.argmax(model.predict(X_test), axis=1)
    pred_time = time.time() - start_pred

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'training_time': training_time,
        'prediction_time': pred_time
    }

    return metrics
