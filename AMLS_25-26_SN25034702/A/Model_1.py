
import numpy as np
from sklearn import svm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import product

def best_parameters(data):

    # Extract train + val sets
    X_train = data['train_images'].reshape((data['train_images'].shape[0], -1))
    y_train = data['train_labels'].flatten()

    X_val = data['val_images'].reshape((data['val_images'].shape[0], -1))
    y_val = data['val_labels'].flatten()

    # Parameter grid
    C_vals = [0.01, 0.1, 1, 10, 100]
    kernels = ['rbf', 'poly']
    gammas = ['scale', 'auto', 0.001, 0.01, 0.1]
    degrees = [2, 3, 4]
    coef0_vals = [0.0, 0.1, 0.5]

    best_params = None
    best_acc = -1

    # Iterate over all combinations manually
    for C, kernel, gamma, degree, coef0 in product(C_vals, kernels, gammas, degrees, coef0_vals):

        # Ignore degree & coef0 when kernel=rbf
        if kernel == 'rbf':
            degree = 3
            coef0 = 0.0

        model = Pipeline([
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

    print("BEST PARAMETERS (Flatten Baseline):", best_params)
    print("Validation Accuracy:", best_acc)

    return best_params


def model_1(data, SVM_parameters):

    # Train
    X_train = data['train_images'].reshape((data['train_images'].shape[0], -1))
    y_train = data['train_labels'].flatten()

    # Test
    X_test = data['test_images'].reshape((data['test_images'].shape[0], -1))
    y_test = data['test_labels'].flatten()

    # Build model with best params found from validation
    clf = Pipeline([
        ("svm", svm.SVC(
            C=SVM_parameters["svm__C"],
            kernel=SVM_parameters["svm__kernel"],
            gamma=SVM_parameters["svm__gamma"],
            degree=SVM_parameters["svm__degree"],
            coef0=SVM_parameters["svm__coef0"]
        ))
    ])

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
