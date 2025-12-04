#build a svm model to classify breast cancer images
#this file will be imported in main.py

from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise

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
from sklearn.decomposition import PCA

def augment_data(images, labels, augmentation_factor=3):
    """Generates augmented images using simple geometric transformations."""
    augmented_images = [images]
    augmented_labels = [labels]
    
    # Target size after augmentation: 3x original size
    for i in range(1, augmentation_factor + 1):
        new_images = images.copy()
        
        # 1. Random Rotation (0 to 10 degrees)
        angle = np.random.uniform(-10, 10)
        new_images = np.array([rotate(img.reshape(28, 28), angle).reshape(784) for img in new_images])
        
        # 2. Add Gaussian Noise (Simulating real-world imaging variations)
        new_images = np.array([random_noise(img_flat.reshape(28, 28), mode='gaussian', var=0.005).reshape(784) for img_flat in new_images])
        
        # 3. Small Shear/Shift (Using AffineTransform)
        tform = AffineTransform(shear=np.random.uniform(-0.1, 0.1))
        new_images = np.array([warp(img_flat.reshape(28, 28), tform).reshape(784) for img_flat in new_images])

        augmented_images.append(new_images)
        augmented_labels.append(labels)

    # Concatenate all augmented sets
    final_images = np.concatenate(augmented_images, axis=0)
    final_labels = np.concatenate(augmented_labels, axis=0)
    
    print(f"Data Augmentation complete. New training size: {final_images.shape[0]} samples.")
    
    # Return the flattened images
    return final_images.reshape((-1, 784)), final_labels

def best_parameters(data):
    #extract images and labels
    images = data['train_images']
    labels = data['train_labels'].flatten()
    
    #reshape images for SVM input
    n_samples = images.shape[0]
    reshaped_images = images.reshape((n_samples, -1))
    reshaped_images, labels = augment_data(reshaped_images, labels)
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
    'svm__C': [0.01, 0.1, 1, 10, 100, 1000],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'svm__degree': [2, 3, 4, 5],   # only used for poly
    'svm__coef0': [0.0, 0.1, 0.5, 1.0],  # for poly
    # 'pca__n_components': [20, 30, 40, 50, 60, 70, 80, 90, 100]
}

 
    #pipeline with StandardScaler + SVM
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        # ("pca", PCA(n_components=50, svd_solver='full')),
        ("svm", svm.SVC())
    ])
    grid = GridSearchCV(pipe, param_grid, cv=2, n_jobs=2)
    grid.fit(hog_features, labels)
    return grid.best_params_

def model_1(data, SVM_parameters):
    # Training data
    train_images = data['train_images']
    train_labels = data['train_labels'].flatten()
    train_images = train_images.reshape((train_images.shape[0], -1))
    train_images, train_labels = augment_data(train_images.reshape((train_images.shape[0], -1)), train_labels)
    
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
    test_images = test_images.reshape((test_images.shape[0], -1))
    test_labels = data['test_labels'].flatten()

    #use 
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
        # ("pca", PCA(n_components=SVM_parameters["pca__n_components"], svd_solver='full')),
        ("svm", svm.SVC(
            C=SVM_parameters["svm__C"], 
            kernel=SVM_parameters["svm__kernel"], 
            gamma=SVM_parameters["svm__gamma"],
            degree=SVM_parameters.get("svm__degree", 3)
        ))
    ])
    t1 = time.time()
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
        "training_time": t2-t1,
        "prediction_time": t3 - t2
    }

