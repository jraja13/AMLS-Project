# build a neural network model with pytorch to classify breast cancer images
# this file will be imported in main.py
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models


def model_3(data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    data: dictionary with keys 'X_train', 'y_train', 'X_val', 'y_val'
    device: 'cuda' or 'cpu'
    """

    # ----------------------------------
    # 1. Prepare data
    # ----------------------------------
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'])
    X_val = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'])

    # FIX LABELS: handle (N,1), (N,2), floats, etc.
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        # one-hot → integer
        y_train = torch.argmax(y_train, dim=1)
        y_val = torch.argmax(y_val, dim=1)

    y_train = y_train.view(-1).long()
    y_val = y_val.view(-1).long()

    # Normalize
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    # Add channel dimension if needed
    if X_train.ndim == 3:
        X_train = X_train.unsqueeze(1)
        X_val = X_val.unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    # ----------------------------------
    # 2. Build ResNet18
    # ----------------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Adjust for grayscale
    if X_train.shape[1] == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------------------
    # 3. Training
    # ----------------------------------
    start_time = time.time()
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time

    # ----------------------------------
    # 4. Evaluation
    # ----------------------------------
    model.eval()
    all_preds, all_labels = [], []
    pred_start_time = time.time()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    prediction_time = time.time() - pred_start_time

    # ----------------------------------
    # 5. Metrics
    # ----------------------------------
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1_score': f1_score(all_labels, all_preds, average='weighted'),
        'training_time': training_time,
        'prediction_time': prediction_time
    }

    return metrics