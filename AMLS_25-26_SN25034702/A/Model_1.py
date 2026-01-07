import numpy as np
from sklearn import svm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import product
from sklearn.decomposition import PCA  

# Hyperparameter Tuning
def best_parameters(data):  
    # Prepare data
    X_train_flat = data['X_train'].reshape((data['X_train'].shape[0], -1))
    X_val_flat   = data['X_val'].reshape((data['X_val'].shape[0], -1))
    y_train = data['y_train'].flatten()
    y_val   = data['y_val'].flatten()

    pca = PCA(n_components=0.90, random_state=42, svd_solver='full', whiten=True)   
    X_train = pca.fit_transform(X_train_flat)
    X_val = pca.transform(X_val_flat)

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
            ("pca", pca),   
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

# Train & Evaluate Model
def model_1(data, SVM_parameters):  
    # Prepare train/test
    X_train_flat = data['X_train'].reshape((data['X_train'].shape[0], -1))
    X_test_flat  = data['X_test'].reshape((data['X_test'].shape[0], -1))
    y_train = data['y_train'].flatten()
    y_test  = data['y_test'].flatten()

    pca = PCA(n_components=0.90, random_state=42, svd_solver='full', whiten=True)
    X_train = pca.fit_transform(X_train_flat)
    X_test = pca.transform(X_test_flat)

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
