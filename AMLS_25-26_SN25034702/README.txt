Description of the Project

To deploy, optimize and compare two machine learning models.

Model A - Support Vector Machines
Model B - Convolutional Neural Network

This is the folder structure used in this project.

-AMLS_25-26_SN25034702
    -A
    -B
    -Datasets
    main.py
    README.txt

main.py - contains the main script which triggers codes of both the models and evaluates them.
A - contains the code for model 1 (SVM)
B - contains the code for model 2 (CNN)
Datasets - contains the dataset in .npz format. (BreastMNIST images)
README.txt - info file

For the CNN, I have added a chart which shows the point for early stopping, along with both 
the curves of training loss and validation loss. It will be saved inside the folder 'B'.

The performance of the final models can be viewed in the terminal as I have printed them.

The packages required for my code to run smoothly are:
- Python 3.7+ (tested)
- tensorflow
- numpy
- scikit-learn
- matplotlib
- pillow
- memory-profiler 
