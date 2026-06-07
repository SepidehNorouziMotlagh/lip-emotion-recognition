# Lip‑Motion Emotion Recognition with Privacy Evaluation

This project trains a lightweight **CNN‑GRU hybrid** to recognise six basic
emotions (Anger, Disgust, Fear, Happy, Neutral, Sad) **solely from inner‑lip
landmarks** extracted from video. The model is designed to be
privacy‑preserving: it reads *how* a person feels without learning *who*
they are.

**Key Result:** 45.1% emotion accuracy on unseen speakers, and **0%
speaker identification accuracy** on the same speakers, emotion can be
recognised while identity remains private.

---

## Dataset

[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)  
7 442 video clips | 91 speakers | 6 emotions (Anger, Disgust, Fear, Happy,
Neutral, Sad)

The data is split **speaker‑independently**: 72 speakers for training, 19
held‑out speakers for testing.

---

## Pipeline

1. **Landmark extraction** – MediaPipe Face Landmarker extracts 20
   inner‑lip landmarks (60‑dim) plus Mouth Aspect Ratio (MAR) and
   inter‑frame motion, resulting in a 62‑dimensional feature vector per
   frame.
2. **Windowing** – Each video is cut into 1‑second windows (30 frames)
   with 50 % overlap.
3. **Model training** – A CNN‑GRU hybrid is trained on the 62‑dimensional
   sequences to predict emotion.
4. **Privacy evaluation** – A separate speaker classifier is trained on
   the model’s internal embeddings and tested on 19 unseen speakers.

---

## Results

### Emotion Recognition

| Model | Test Accuracy |
|:---|:---:|
| Random baseline | ~16.7% |
| Majority‑class baseline | ~20% |
| MLP on hand‑crafted MAR/motion stats | 36.0% |
| 1D‑CNN (60‑dim) | 42.8% |
| 1D‑CNN (62‑dim, +MAR/motion) | 44.6% |
| **CNN‑GRU (62‑dim)** | **45.1%** |

### Privacy Leakage

**Speaker Identification on 19 unseen speakers:** **0%**  
(Chance level: 1/91 ≈ 1.1%)

The model’s internal representation does not transfer speaker identity to
new individuals. It learns to recognise *emotion* from lip dynamics while
discarding the cues that would identify *who* is speaking.

---

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
The following packages are required:

mediapipe >= 0.10.0

opencv-python

numpy

pandas

tqdm

torch

scikit-learn

You also need face_landmarker.task. Download it from
Google Storage.

Usage
Download CREMA-D and place all .mp4 files in data/raw/.

Run the scripts in order:

bash
python src/build_metadata.py
python src/windowing.py
python src/split_data.py
python src/train_cnn_gru.py
python src/privacy_leakage.py
Trained model weights will be saved as data/best_model_cnn_gru.pth.

The repository also includes a pure 1D‑CNN (train_model.py) and an MLP
baseline (train_mlp_baseline.py) for comparison.

Author
Sepideh Norouzi Motlagh
GitHub: @SepidehNorouziMotlagh