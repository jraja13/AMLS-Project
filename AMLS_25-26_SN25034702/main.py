import numpy as np
from memory_profiler import memory_usage
import time
from A.Model_1 import best_parameters, model_1
def run_model_1():
    data = np.load('Datasets/breastmnist.npz')
    start_time = time.time()
    best_params = best_parameters(data)

    def run():
        return model_1(data, best_params)

    # run model ONCE and capture both memory + metrics
    mem_usage, metrics = memory_usage(run, retval=True)
    peak_memory = max(mem_usage) - min(mem_usage)

    print(f"Model SVM+SS+HOG Metrics:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"Training Time: {metrics['training_time']-start_time:.2f} seconds")
    print(f"Prediction Time: {metrics['prediction_time']:.2f} seconds") 
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")
    


def run_model_2():
    print("Model 2 is not implemented yet.")


if __name__ == "__main__":
    run_model_1()
    run_model_2()