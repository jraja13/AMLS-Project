import numpy as np
from sklearn import svm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import product
from skimage.feature import hog
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- HOG Feature Extraction ----------------
def extract_hog_features(images, pixels_per_cell=(2,2), cells_per_block=(2,2)):
    features = []
    for img in images:
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze()
        hog_feat = hog(img,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       feature_vector=True)
        features.append(hog_feat)
    return np.array(features)

# ---------------- Hyperparameter Search ----------------
def best_parameters(data, use_hog=True, combine_raw=False):
    # Prepare data
    X_train_flat = data['train_images'].reshape((data['train_images'].shape[0], -1))
    X_val_flat   = data['val_images'].reshape((data['val_images'].shape[0], -1))
    y_train = data['train_labels'].flatten()
    y_val   = data['val_labels'].flatten()

    # Extract HOG if needed
    if use_hog:
        X_train_hog = extract_hog_features(data['train_images'], pixels_per_cell=(2,2))
        X_val_hog   = extract_hog_features(data['val_images'], pixels_per_cell=(2,2))

        if combine_raw:
            X_train = np.concatenate([X_train_flat, X_train_hog], axis=1)
            X_val   = np.concatenate([X_val_flat, X_val_hog], axis=1)
        else:
            X_train = X_train_hog
            X_val   = X_val_hog
    else:
        X_train = X_train_flat
        X_val   = X_val_flat


    # Parameter grid
    C_vals = [1, 10, 50, 100]
    kernels = ['rbf', 'poly']
    gammas = ['scale', 0.01, 0.1]
    degrees = [2, 3, 4]
    coef0_vals = [0.0, 0.1, 0.5]

    best_params = None
    best_acc = -1

    # Iterate over all combinations
    for C, kernel, gamma, degree, coef0 in product(C_vals, kernels, gammas, degrees, coef0_vals):
        if kernel == 'rbf':
            degree = 3
            coef0 = 0.0

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", svm.SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_params = {
                "svm__C": C,
                "svm__kernel": kernel,
                "svm__gamma": gamma,
                "svm__degree": degree,
                "svm__coef0": coef0
            }

    print("BEST PARAMETERS:", best_params)
    print("Validation Accuracy:", best_acc)
    return best_params

# ---------------- Train & Evaluate Model ----------------
def model_1(data, SVM_parameters, use_hog=True, combine_raw=False):
    # Prepare train/test
    X_train_flat = data['train_images'].reshape((data['train_images'].shape[0], -1))
    X_test_flat  = data['test_images'].reshape((data['test_images'].shape[0], -1))
    y_train = data['train_labels'].flatten()
    y_test  = data['test_labels'].flatten()

    if use_hog:
        X_train_hog = extract_hog_features(data['train_images'], pixels_per_cell=(2,2))
        X_test_hog  = extract_hog_features(data['test_images'], pixels_per_cell=(2,2))
        if combine_raw:
            X_train = np.concatenate([X_train_flat, X_train_hog], axis=1)
            X_test  = np.concatenate([X_test_flat, X_test_hog], axis=1)
        else:
            X_train = X_train_hog
            X_test  = X_test_hog
    else:
        X_train = X_train_flat
        X_test  = X_test_flat

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Build SVM with best parameters
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm.SVC(
            C=SVM_parameters["svm__C"],
            kernel=SVM_parameters["svm__kernel"],
            gamma=SVM_parameters["svm__gamma"],
            degree=SVM_parameters["svm__degree"],
            coef0=SVM_parameters["svm__coef0"]
        ))
    ])

    # Train
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    # Predict
    y_pred = clf.predict(X_test)
    t3 = time.time()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": t2 - t1,
        "prediction_time": t3 - t2
    }
