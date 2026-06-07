import numpy as np
from pathlib import Path

# Load windows
data = np.load(Path(__file__).resolve().parent.parent / "data" / "windows.npz")
X = data["X"]
y = data["y"]
speakers = data["speakers"]

# Get unique speaker IDs
unique_speakers = np.unique(speakers)
np.random.seed(42)   # for reproducibility
np.random.shuffle(unique_speakers)

# 80/20 split
n_train = int(0.8 * len(unique_speakers))
train_speakers = unique_speakers[:n_train]
test_speakers  = unique_speakers[n_train:]

train_mask = np.isin(speakers, train_speakers)
test_mask  = np.isin(speakers, test_speakers)

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

# Save
out_dir = Path(__file__).resolve().parent.parent / "data"
np.save(out_dir / "X_train.npy", X_train)
np.save(out_dir / "y_train.npy", y_train)
np.save(out_dir / "X_test.npy",  X_test)
np.save(out_dir / "y_test.npy",  y_test)

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Train speakers: {len(train_speakers)} | Test speakers: {len(test_speakers)}")