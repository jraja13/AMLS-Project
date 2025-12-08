import numpy as np
from sklearn import svm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import product
from skimage.feature import hog
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.feature import local_binary_pattern

#LBP Feature Extraction
def extract_lbp_features(images, P=8, R=1, method='uniform'):
    features = []
    for img in images:
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze()

        # Compute LBP image
        lbp = local_binary_pattern(img, P=P, R=R, method=method)

        # Convert to histogram (common approach for LBP)
        (hist, _) = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, P + 3),   # P + 2 bins for uniform patterns
            range=(0, P + 2)
        )

        # Normalize histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        features.append(hist)

    return np.array(features)


# Hyperparameter Tuning
def best_parameters(data, use_hog=True, combine_raw=False):
    # Prepare data
    X_train_flat = data['train_images'].reshape((data['train_images'].shape[0], -1))
    X_val_flat   = data['val_images'].reshape((data['val_images'].shape[0], -1))
    y_train = data['train_labels'].flatten()
    y_val   = data['val_labels'].flatten()

    # Extract LBP features
    X_train = extract_lbp_features(data['train_images'], P=8, R=1)
    X_val   = extract_lbp_features(data['val_images'], P=8, R=1)


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

# Training and Evaluation
def model_1(data, SVM_parameters, use_hog=True, combine_raw=False):
    # Prepare data
    X_train_flat = data['train_images'].reshape((data['train_images'].shape[0], -1))
    X_val_flat   = data['val_images'].reshape((data['val_images'].shape[0], -1))
    y_train = data['train_labels'].flatten()
    y_val   = data['val_labels'].flatten()

    # Extract LBP features
    X_train = extract_lbp_features(data['train_images'], P=8, R=1)
    X_val   = extract_lbp_features(data['val_images'], P=8, R=1)

    X_test = data['test_images'].reshape((data['test_images'].shape[0], -1))
    y_test = data['test_labels'].flatten()
    X_test = extract_lbp_features(data['test_images'], P=8, R=1)

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