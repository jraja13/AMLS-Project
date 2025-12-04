import numpy as np
from memory_profiler import memory_usage
import time
from PIL import Image
from A.Model_1 import best_parameters, model_1
from B.Model_2 import best_parameters as best_parameters_2, model_2
from C.model_3 import model_3
from D.Model_4 import model_4

def run_model_1():
    data = np.load('Datasets/breastmnist.npz')
    start_time = time.time()
    best_params = best_parameters(data)
    param_time = time.time() - start_time
    def run():
        return model_1(data, best_params)
    # run model ONCE and capture both memory + metrics
    mem_usage, metrics = memory_usage(run, retval=True)
    peak_memory = max(mem_usage) - min(mem_usage)

    print(f"Model SVM+SS+HOG+Data Augmentation:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Parameter Tuning Time: {param_time:.2f} seconds")
    print(f"Prediction Time: {metrics['prediction_time']:.2f} seconds") 
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")
    
def run_model_2():
    data = np.load('Datasets/breastmnist.npz')
    start_time = time.time()
    best_params = best_parameters_2(data)
    print("Best Parameters for Model 2 (Random Forest):", best_params)
    param_time = time.time() - start_time
    def run():
        return model_2(data, best_params)
    # run model ONCE and capture both memory + metrics
    mem_usage, metrics = memory_usage(run, retval=True)
    peak_memory = max(mem_usage) - min(mem_usage)

    print(f"Model Random Forest+SS+HOG:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Parameter Tuning Time: {param_time:.2f} seconds")
    print(f"Prediction Time: {metrics['prediction_time']:.2f} seconds") 
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")

def run_model_3():

    train_images = np.load('Datasets/breastmnist.npz')['train_images']
    #create a dictionary wiyth keys X_train, y_train, X_val, y_val
    train_labels = np.load('Datasets/breastmnist.npz')['train_labels']
    test_images = np.load('Datasets/breastmnist.npz')['test_images']
    test_labels = np.load('Datasets/breastmnist.npz')['test_labels']
    
    images = {
        'X_train': train_images,
        'y_train': train_labels,
        'X_val': test_images,
        'y_val': test_labels
    }

    def run():
        return model_3(images)
    # run model ONCE and capture both memory + metrics
    mem_usage, metrics = memory_usage(run, retval=True)
    peak_memory = max(mem_usage) - min(mem_usage)

    print(f"Model Neural Network:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Prediction Time: {metrics['prediction_time']:.2f} seconds") 
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")

def run_model_4():

    train_images = np.load('Datasets/breastmnist.npz')['train_images']
    train_labels = np.load('Datasets/breastmnist.npz')['train_labels']
    test_images = np.load('Datasets/breastmnist.npz')['test_images']
    test_labels = np.load('Datasets/breastmnist.npz')['test_labels']
    
    images = {
        'X_train': train_images,
        'y_train': train_labels,
        'X_val': test_images,
        'y_val': test_labels
    }

    def run():
        return model_4(images)
    # run model ONCE and capture both memory + metrics
    mem_usage, metrics = memory_usage(run, retval=True)
    peak_memory = max(mem_usage) - min(mem_usage)

    print(f"Model Neural Network:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Prediction Time: {metrics['prediction_time']:.2f} seconds") 
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")


if __name__ == "__main__":
    # run_model_1()
    # run_model_2()
    # run_model_3()
 run_model_4()
