"""
Step 2 (revised) – Window landmark sequences WITH hand‑crafted features.

Includes mar and motion alongside the 60 landmark coordinates.
Output: data/windows.npz  (X shape: (N, 30, 62))
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
METADATA_CSV = BASE_DIR / "data" / "metadata.csv"
OUTPUT_NPZ = BASE_DIR / "data" / "windows.npz"

WINDOW_SIZE = 30          # frames per window (1 second @ 30 fps)
STRIDE = 15               # 50% overlap

# Feature columns – 60 landmarks + 2 hand‑crafted = 62 total
LM_COLS = [f"lm_{i}_{axis}" for i in range(20) for axis in ("x", "y", "z")]
HAND_COLS = ["mar", "motion"]
ALL_COLS = LM_COLS + HAND_COLS          # exactly 62 columns

# ------------------------------------------------------------
# Load metadata (filename -> emotion, speaker)
# ------------------------------------------------------------
meta = pd.read_csv(METADATA_CSV)
meta_dict = {}
for _, row in meta.iterrows():
    meta_dict[row["filename"]] = (row["emotion"], row["speaker"])

# ------------------------------------------------------------
# Emotion encoding
# ------------------------------------------------------------
emotions = sorted(meta["emotion"].unique())
emotion_to_idx = {e: i for i, e in enumerate(emotions)}

# ------------------------------------------------------------
# Windowing
# ------------------------------------------------------------
X_list = []
y_list = []
speaker_list = []

for csv_file in tqdm(sorted(FEATURES_DIR.glob("*.csv")), desc="Windowing"):
    if csv_file.name not in meta_dict:
        continue

    emotion_label, speaker_id = meta_dict[csv_file.name]
    try:
        df = pd.read_csv(csv_file, usecols=ALL_COLS).astype(np.float32)
    except (ValueError, pd.errors.EmptyDataError):
        continue

    if len(df) < WINDOW_SIZE:
        continue

    data = df[ALL_COLS].values   # (frames, 62)

    for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
        window = data[start:start + WINDOW_SIZE]    # (30, 62)
        X_list.append(window)
        y_list.append(emotion_to_idx[emotion_label])
        speaker_list.append(speaker_id)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
speakers = np.array(speaker_list)

np.savez(OUTPUT_NPZ, X=X, y=y, speakers=speakers)
print(f"\nSaved {len(X)} windows to {OUTPUT_NPZ}")
print(f"Shape X: {X.shape}, y: {y.shape}")
print(f"Emotion counts: {np.bincount(y)}")