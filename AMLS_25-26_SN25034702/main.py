import numpy as np
from A.Model_1 import best_parameters, model_1
def run_model_1():
    data = np.load('Datasets/breastmnist.npz')
    best_params = best_parameters(data)
    metrics = model_1(data, best_params)
    
    print(f"Model SVM+SS+HOG Metrics:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

def run_model_2():
    print("Model 2 is not implemented yet.")


if __name__ == "__main__":
    run_model_1()
    run_model_2()