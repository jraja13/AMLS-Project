import numpy as np
from memory_profiler import memory_usage
import time
from PIL import Image
from A.Model_1 import best_parameters, model_1
from B.Model_2 import find_best_epochs, model_2


def run_model_1(images):

    start_time = time.time()
    
    # Find best hyperparameters using grid search
    best_params = best_parameters(images) 
    
    param_time = time.time() - start_time
    def run():
        return model_1(images, best_params)
    mem_usage, metrics = memory_usage(run, retval=True)
    peak_memory = max(mem_usage) - min(mem_usage)
    metrics['Peak_Memory_Usage_MB'] = peak_memory
    metrics['parameter_tuning_time'] = param_time
    return metrics
    
def run_model_2(images):

    bestepoch = find_best_epochs(images)
    def run():
        return model_2(images, bestepoch)
    mem_usage, metrics = memory_usage(run, retval=True)
    peak_memory = max(mem_usage) - min(mem_usage)
    metrics['Peak_Memory_Usage_MB'] = peak_memory
    return metrics

def extract_images():
    global images
    train_images = np.load('Datasets/breastmnist.npz')['train_images']
    train_labels = np.load('Datasets/breastmnist.npz')['train_labels']
    val_images = np.load('Datasets/breastmnist.npz')['val_images']
    val_labels = np.load('Datasets/breastmnist.npz')['val_labels']
    test_images = np.load('Datasets/breastmnist.npz')['test_images']
    test_labels = np.load('Datasets/breastmnist.npz')['test_labels']
    
    images = {
        'X_train': train_images,
        'y_train': train_labels,
        'X_val': val_images,
        'y_val': val_labels,
        'X_test': test_images,
        'y_test': test_labels
    }  
    return images

def metricsprint(metrics):
    if 'parameter_tuning_time' in metrics:
        print("SVM Metrics:")
    else:
        print("CNN Metrics:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
    if 'parameter_tuning_time' in metrics:
        print(f"Parameter Tuning Time: {metrics['parameter_tuning_time']:.2f} seconds")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Prediction Time: {metrics['prediction_time']:.2f} seconds")
    print(f"Peak Memory Usage: {metrics['Peak_Memory_Usage_MB']:.2f} MB")

if __name__ == "__main__":
    images = extract_images()
    m1 = run_model_1(images)
    m2 = run_model_2(images)
    metricsprint(m1)
    metricsprint(m2)