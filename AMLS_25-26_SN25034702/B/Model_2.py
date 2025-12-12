import numpy as np
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def model_2(images, epochs=100):

    X_train, y_train = images['X_train'], images['y_train']
    X_val, y_val     = images['X_val'], images['y_val']
    X_test, y_test   = images['X_test'], images['y_test']

    # Combine train + val
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)

    # Reshape for CNN
    X_combined = X_combined.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test     = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    num_classes = len(np.unique(y_combined))

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=X_combined.shape[1:]),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(X_combined, y_combined, epochs=epochs, batch_size=32, verbose=2)
    training_time = time.time() - start_time

    start_pred_time = time.time()
    y_pred = np.argmax(model.predict(X_test), axis=1)
    prediction_time = time.time() - start_pred_time

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'training_time': training_time,
        'prediction_time': prediction_time
    }

    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Baseline Model Training Loss (Train + Val)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return metrics