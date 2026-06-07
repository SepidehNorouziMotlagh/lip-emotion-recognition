# Lip-Motion Emotion Recognition with Privacy Evaluation

This project trains a lightweight **CNN–GRU hybrid model** to recognize six basic emotions (**Anger, Disgust, Fear, Happy, Neutral, Sad**) solely from **inner-lip landmarks** extracted from video.

The system is designed to be **privacy-preserving**: it learns *how a person feels* without learning *who they are*.

> **Key Result:**
> **45.1% emotion recognition accuracy** on unseen speakers, while achieving **0% speaker identification accuracy** on the same speakers — demonstrating that emotion can be recognized while identity remains private.

---

# Dataset

Dataset: [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

* **7,442** video clips
* **91** speakers
* **6** emotions:

  * Anger
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad

The dataset is split **speaker-independently**:

* **72 speakers** for training
* **19 unseen speakers** for testing

---

# Pipeline

## 1. Landmark Extraction

MediaPipe Face Landmarker extracts:

* 20 inner-lip landmarks (**60 dimensions**)
* Mouth Aspect Ratio (**MAR**)
* Inter-frame lip motion

Resulting in a **62-dimensional feature vector per frame**.

## 2. Windowing

Each video is segmented into:

* **1-second windows**
* **30 frames per window**
* **50% overlap**

## 3. Model Training

A lightweight **CNN–GRU hybrid** is trained on the 62-dimensional temporal sequences to predict emotion labels.

## 4. Privacy Evaluation

A separate speaker classifier is trained on the model’s internal embeddings and evaluated on unseen speakers to measure identity leakage.

---

# Results

## Emotion Recognition

| Model                                    | Test Accuracy |
| ---------------------------------------- | ------------: |
| Random baseline                          |        ~16.7% |
| Majority-class baseline                  |        ~20.0% |
| MLP on handcrafted MAR/motion statistics |         36.0% |
| 1D-CNN (60-dim)                          |         42.8% |
| 1D-CNN (62-dim, +MAR/motion)             |         44.6% |
| **CNN–GRU (62-dim)**                     |     **45.1%** |

---

## Privacy Leakage

### Speaker Identification on 19 Unseen Speakers

**Accuracy: 0%**

Chance level:

```text
1 / 91 ≈ 1.1%
```

The learned representation does not transfer speaker identity to new individuals.
The model captures **emotion-related lip dynamics** while discarding cues that would identify the speaker.

---

# Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Required packages:

```text
mediapipe >= 0.10.0
opencv-python
numpy
pandas
tqdm
torch
scikit-learn
```

You will also need the `face_landmarker.task` model file.

Download it from Google AI / MediaPipe model storage and place it in the appropriate project directory.

---

# Usage

## 1. Download CREMA-D

Place all `.mp4` files inside:

```text
data/raw/
```

## 2. Run the preprocessing and training pipeline

Execute the scripts in the following order:

```bash
python src/build_metadata.py
python src/windowing.py
python src/split_data.py
python src/train_cnn_gru.py
python src/privacy_leakage.py
```

---

# Output

Trained model weights are saved as:

```text
data/best_model_cnn_gru.pth
```

---

# Additional Models

The repository also includes:

* `train_model.py` → pure 1D-CNN baseline
* `train_mlp_baseline.py` → MLP baseline using handcrafted features

---

# Author

**Sepideh Norouzi Motlagh**

GitHub: [@SepidehNorouziMotlagh](https://github.com/SepidehNorouziMotlagh)
