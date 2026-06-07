"""
train_model.py - 1D CNN for emotion recognition from lip landmarks.

Requirements: torch, numpy, scikit-learn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
y_train = np.load(DATA_DIR / "y_train.npy").astype(np.int64)
X_test  = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
y_test  = np.load(DATA_DIR / "y_test.npy").astype(np.int64)

# Normalise features (z‑score per channel)
mean = X_train.mean(axis=(0,1), keepdims=True)
std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_test  = (X_test - mean) / std

# Convert to PyTorch tensors
train_set = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_set  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False)

# ------------------------------------------------------------
# 2. Define the 1D‑CNN
# ------------------------------------------------------------
class LipEmotionCNN(nn.Module):
    def __init__(self, input_dim=62, num_emotions=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)   # (batch, 128, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions)
        )

    def forward(self, x):
        # x: (batch, 30, 60) → (batch, 60, 30) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)   # (batch, 128)
        return self.fc(x)

model = LipEmotionCNN()

# ------------------------------------------------------------
# 3. Train
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

epochs = 50
best_acc = 0.0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_set)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    test_acc = correct / total
    scheduler.step(test_acc)

    print(f"Epoch {epoch+1:2d} | Train Loss {train_loss:.4f} | Test Acc {test_acc:.4f}")

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), DATA_DIR / "best_model.pth")
        print("  → Best model saved")

print(f"\nBest test accuracy: {best_acc:.4f}")

# ------------------------------------------------------------
# 4. Final evaluation
# ------------------------------------------------------------
model.load_state_dict(torch.load(DATA_DIR / "best_model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=emotions))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))