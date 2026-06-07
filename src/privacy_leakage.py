"""
privacy_leakage.py – Speaker identification from lip‑motion embeddings.

This script evaluates whether the CNN‑GRU's internal representation
can identify who is speaking. A low accuracy means the model is
privacy‑preserving: it recognizes emotion but not identity.

Requirements: torch, numpy, scikit‑learn, pandas
Data: windows.npz (contains X, y, speakers), best_model_cnn_gru.pth
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pathlib import Path

# ------------------------------------------------------------
# 1. Load the original dataset and recreate the speaker split
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

data = np.load(DATA_DIR / "windows.npz", allow_pickle=True)
X = data["X"].astype(np.float32)          # (27116, 30, 62)
y = data["y"].astype(np.int64)            # emotion labels (not used here)
speakers = data["speakers"]               # string IDs like '1001'

# Use the same random seed and logic as in split_data.py
unique_speakers = np.unique(speakers)
np.random.seed(42)
np.random.shuffle(unique_speakers)

n_train = int(0.8 * len(unique_speakers))
train_speakers = unique_speakers[:n_train]   # 72 speakers
test_speakers  = unique_speakers[n_train:]   # 19 speakers

train_mask = np.isin(speakers, train_speakers)
test_mask  = np.isin(speakers, test_speakers)

X_train = X[train_mask]
X_test  = X[test_mask]
speakers_train = speakers[train_mask]
speakers_test  = speakers[test_mask]

print(f"Train windows: {len(X_train)} | Test windows: {len(X_test)}")

# Normalise (same as during training)
mean = X_train.mean(axis=(0, 1), keepdims=True)
std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

# ------------------------------------------------------------
# 2. Load the trained CNN‑GRU and create an embedding extractor
# ------------------------------------------------------------
class LipEmotionCNNGRU(nn.Module):
    def __init__(self, input_dim=62, num_emotions=6, gru_hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gru = nn.GRU(128, gru_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.fc(x)

class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv = model.conv
        self.gru = model.gru

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]           # (batch, 256)
        return x

full_model = LipEmotionCNNGRU(input_dim=62)
full_model.load_state_dict(torch.load(DATA_DIR / "best_model_cnn_gru.pth", map_location="cpu"))
full_model.eval()

extractor = EmbeddingExtractor(full_model)
extractor.eval()

# ------------------------------------------------------------
# 3. Extract embeddings
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor.to(device)

def get_embeddings(model, X):
    model.eval()
    dataset = TensorDataset(torch.tensor(X))
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)

print("Extracting training embeddings...")
emb_train = get_embeddings(extractor, X_train)   # (N_train, 256)
print("Extracting test embeddings...")
emb_test  = get_embeddings(extractor, X_test)     # (N_test, 256)

# ------------------------------------------------------------
# 4. Speaker classification (with all 91 speakers encoded)
# ------------------------------------------------------------
# Encode speaker IDs for all 91 unique speakers
le = LabelEncoder()
le.fit(np.unique(speakers))                     # 91 classes
y_speaker_train = le.transform(speakers_train)  # integers 0..71 (but from full set)
y_speaker_test  = le.transform(speakers_test)   # integers for 19 unseen speakers

num_speaker_classes = len(le.classes_)          # 91

class SpeakerMLP(nn.Module):
    def __init__(self, input_dim=256, num_speakers=91):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_speakers)
        )
    def forward(self, x):
        return self.net(x)

speaker_model = SpeakerMLP(num_speakers=num_speaker_classes)
speaker_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(speaker_model.parameters(), lr=0.001)
train_set = TensorDataset(torch.tensor(emb_train), torch.tensor(y_speaker_train))
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

speaker_model.train()
for epoch in range(20):
    total_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(speaker_model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_b.size(0)
    print(f"Speaker Epoch {epoch+1:2d} | Loss {total_loss/len(emb_train):.4f}")

# Evaluate on test speakers (unseen during training)
speaker_model.eval()
with torch.no_grad():
    X_test_t = torch.tensor(emb_test).to(device)
    logits = speaker_model(X_test_t)
    preds = logits.argmax(1).cpu().numpy()

test_acc = accuracy_score(y_speaker_test, preds)
chance = 1.0 / num_speaker_classes
print(f"\nSpeaker Identification Accuracy (19 unseen speakers): {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Chance level (1/{num_speaker_classes}): {chance:.4f} ({chance*100:.2f}%)")
print(f"Relative to chance: {test_acc/chance:.2f}x")

# Within‑training accuracy for reference
with torch.no_grad():
    X_train_t = torch.tensor(emb_train).to(device)
    logits_train = speaker_model(X_train_t)
    preds_train = logits_train.argmax(1).cpu().numpy()
train_acc = accuracy_score(y_speaker_train, preds_train)
print(f"\nWithin‑training Speaker Accuracy (72 seen speakers): {train_acc:.4f} ({train_acc*100:.2f}%)")