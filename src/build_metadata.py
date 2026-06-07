import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
FEATURES_DIR = BASE_DIR / "data" / "features"
OUTPUT_CSV = BASE_DIR / "data" / "metadata.csv"

# ------------------------------------------------------------
# Emotion mapping (abbreviation → full label)
# ------------------------------------------------------------
EMOTION_MAP = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}

# ------------------------------------------------------------
# Collect metadata
# ------------------------------------------------------------
records = []
for csv_file in sorted(FEATURES_DIR.glob("*.csv")):
    name = csv_file.stem                     # e.g. "1001_DFA_ANG_XX"
    parts = name.split("_")
    if len(parts) < 4:                       # skip unexpected files
        continue

    speaker_id = parts[0]                    # e.g. "1001"
    emotion_code = parts[2]                  # e.g. "ANG"
    emotion_full = EMOTION_MAP.get(emotion_code, emotion_code)

    records.append({
        "filename": csv_file.name,           # e.g. "1001_DFA_ANG_XX.csv"
        "speaker": speaker_id,
        "emotion": emotion_full,
    })

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Metadata saved to {OUTPUT_CSV}")
print(f"Total files: {len(df)}")
print(f"Unique speakers: {df['speaker'].nunique()}")
print(f"Emotion counts:\n{df['emotion'].value_counts()}")