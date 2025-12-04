# build a random forest model to classify breast cancer images
# this file will be imported in main.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skimage.feature import hog

from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise

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
    # calculate best parameters for Random Forest using GridSearchCV
    # extract images and labels
    images = data['train_images']
    labels = data['train_labels'].flatten()
    images = images.reshape((images.shape[0], -1))
    images, labels = augment_data(images, labels)
    hog_features = np.array([
        hog(img.reshape(28,28),
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            orientations=9,
            block_norm='L2-Hys')
        for img in images
    ])
    param_grid = {
        'rf__n_estimators': [400, 500],
        'rf__max_depth': [30, 50, None],
        'rf__random_state': [42],
        'rf__min_samples_split': [5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier())
    ])  
    grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(hog_features, labels)
    return grid_search.best_params_

def model_2(data, params):
    # Extract images and labels
    train_images = data['train_images']
    train_labels = data['train_labels'].flatten()
    train_images = train_images.reshape((train_images.shape[0], -1))
    
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
    test_images = test_images.reshape((test_images.shape[0], -1))
    

    # Compute HOG features for test images
    test_hog = np.array([
        hog(img.reshape(28,28),
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            orientations=9,
            block_norm='L2-Hys')
        for img in test_images
    ])
    
    # Train Random Forest
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=params['rf__n_estimators'],
            max_depth=params['rf__max_depth'],
            random_state=params['rf__random_state'],
            min_samples_split=params['rf__min_samples_split'],
            min_samples_leaf=params['rf__min_samples_leaf']
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
    precision = precision_score(test_labels, y_pred, zero_division=0)
    recall = recall_score(test_labels, y_pred, zero_division=0)
    f1 = f1_score(test_labels, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': t2 - t1,
        'prediction_time': t3 - t2
    }
    
    return metrics