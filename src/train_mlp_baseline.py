"""
train_mlp_baseline.py – MLP on hand‑crafted MAR/motion statistics.

Run this after the data split to get a baseline accuracy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load the 62‑dim data
X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
y_train = np.load(DATA_DIR / "y_train.npy").astype(np.int64)
X_test  = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
y_test  = np.load(DATA_DIR / "y_test.npy").astype(np.int64)

# Extract only MAR + motion (last 2 channels) and compute per‑window statistics
X_train_hc = X_train[:, :, -2:]   # (N, 30, 2)
X_test_hc  = X_test[:, :, -2:]

def handcraft_features(X):
    mean = X.mean(axis=1)
    std  = X.std(axis=1)
    minv = X.min(axis=1)
    maxv = X.max(axis=1)
    return np.hstack([mean, std, minv, maxv])   # (N, 8)

X_train_feat = handcraft_features(X_train_hc)
X_test_feat  = handcraft_features(X_test_hc)

# Normalise
mean = X_train_feat.mean(axis=0, keepdims=True)
std  = X_train_feat.std(axis=0, keepdims=True) + 1e-8
X_train_feat = (X_train_feat - mean) / std
X_test_feat  = (X_test_feat  - mean) / std

# Create datasets
train_set = TensorDataset(torch.tensor(X_train_feat), torch.tensor(y_train))
test_set  = TensorDataset(torch.tensor(X_test_feat),  torch.tensor(y_test))
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False)

# MLP model
class MLPBaseline(nn.Module):
    def __init__(self, input_dim=8, num_emotions=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_emotions)
        )
    def forward(self, x):
        return self.net(x)

model = MLPBaseline()

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_acc = 0.0
for epoch in range(30):
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            preds = model(X_b).argmax(1)
            correct += (preds == y_b.cpu()).sum().item()
            total += y_b.size(0)
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
    print(f"Epoch {epoch+1:2d} | Test Acc {acc:.4f}")

print(f"\nMLP Baseline Best Accuracy: {best_acc:.4f}")

# Final evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        X_b = X_b.to(device)
        preds = model(X_b).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_b.numpy())

emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
print("\nClassification Report (MLP Baseline):")
print(classification_report(all_labels, all_preds, target_names=emotions))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))