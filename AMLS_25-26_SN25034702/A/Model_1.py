#build a svm model to classify breast cancer images
#this file will be imported in main.py

import numpy as np
from sklearn import svm
import time
from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skimage.feature import hog

def best_parameters(data):
    #extract images and labels
    images = data['train_images']
    labels = data['train_labels'].flatten()
    
    #reshape images for SVM input
    n_samples = images.shape[0]
    reshaped_images = images.reshape((n_samples, -1))
    
    hog_features = np.array([
        hog(img.reshape(28,28),
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            orientations=9,
            block_norm='L2-Hys')
        for img in reshaped_images
    ])

    #define parameter grid for grid search
    param_grid = {
    "svm__C": [0.01, 0.1, 1, 10, 100],        # regularization strength
    "svm__kernel": ["linear", "rbf", "poly"],  # kernel types
    "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],  # only for 'rbf' and 'poly'
    "svm__degree": [2, 3, 4]                   # only for 'poly' kernel
}

     #pipeline with StandardScaler + SVM
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm.SVC())
    ])
    
    #perform grid search with cross-validation
    grid = GridSearchCV(pipe, param_grid, cv=3)
    grid.fit(hog_features, labels)
    
    #return best parameters
    return grid.best_params_

def model_1(data, SVM_parameters):
    # Training data
    train_images = data['train_images']
    train_labels = data['train_labels'].flatten()
    
    # Compute HOG features for training images
    train_hog = np.array([
        hog(img.reshape(28,28),
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            orientations=9,
            block_norm='L2-Hys')
        for img in train_images
    ])
    
    # Test data
    test_images = data['test_images']
    test_labels = data['test_labels'].flatten()
    
    # Compute HOG features for test images
    test_hog = np.array([
        hog(img.reshape(28,28),
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            orientations=9,
            block_norm='L2-Hys')
        for img in test_images
    ])
    
    # Train SVM
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm.SVC(
            C=SVM_parameters["svm__C"], 
            kernel=SVM_parameters["svm__kernel"], 
            gamma=SVM_parameters["svm__gamma"],
            degree=SVM_parameters.get("svm__degree", 3)
        ))
    ])
    
    clf.fit(train_hog, train_labels)
    t2 = time.time()
    # Predict
    y_pred = clf.predict(test_hog)
    t3 = time.time()
    
    # Accuracy
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)
    
    # Return metrics as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": t2,
        "prediction_time": t3 - t2
    }

