import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def model_2(images):
    X_train, y_train = images['X_train'], images['y_train']
    X_val, y_val     = images['X_val'], images['y_val']
    X_test, y_test   = images['X_test'], images['y_test']

    X_train = X_train.reshape(-1,28,28,1).astype('float32') / 255.0
    X_val   = X_val.reshape(-1,28,28,1).astype('float32') / 255.0
    X_test  = X_test.reshape(-1,28,28,1).astype('float32') / 255.0

    num_classes = len(np.unique(y_train))

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:],
               kernel_regularizer=regularizers.l2(0.0001)),
        MaxPooling2D((2,2)),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

    #Find the best epoch using early stopping but run full 100 epochs to plot
    patience = 5
    best_val_loss = np.inf
    wait = 0
    best_epoch = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, 101):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            batch_size=32,
            verbose=0
        )

        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1,len(val_losses)+1), val_losses, label='Val Loss')
    plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch {best_epoch}')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("B/early_stopping_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Rebuild model and train on train data for exactly best_epoch
    final_model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:],
               kernel_regularizer=regularizers.l2(0.0001)),
        MaxPooling2D((2,2)),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    final_model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    start_time = time.time()
    final_model.fit(X_train, y_train, epochs=best_epoch, batch_size=32, verbose=2)
    training_time = time.time() - start_time

    start_pred = time.time()
    y_prob = final_model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    pred_time = time.time() - start_pred

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'training_time': training_time,
        'prediction_time': pred_time,
        'best_epoch': best_epoch
    }

    return metrics