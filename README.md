# Lip-Motion Emotion Recognition

Emotion recognition from inner-lip motion dynamics using 1D-CNN and CNN–GRU architectures, with an exploratory analysis of speaker-identity information.

## Overview

This project investigates whether emotional expressions can be recognized using inner-lip motion dynamics without providing the emotion recognition model with full-face imagery.

Rather than operating on full-face images, the model uses a restricted set of visual features:

- Inner-lip landmarks
- Mouth Aspect Ratio (MAR)
- Inter-frame lip motion magnitude

The goal is to investigate the extent to which emotion-related information can be retained when the model input is restricted to articulatory features around the mouth.

**Important:** Restricting the model input to inner-lip features does not guarantee speaker anonymity. Motion dynamics may still contain identity-related information, which requires a dedicated privacy evaluation.

The repository includes:

- Handcrafted feature baselines
- 1D-CNN models
- CNN–GRU temporal architectures
- Speaker-disjoint emotion recognition evaluation
- An exploratory speaker-identity classification script

## Dataset

**Dataset:** CREMA-D

- 7,442 video clips
- 91 speakers
- 6 emotion classes

### Speaker-disjoint split

| Partition | Speakers |
|---|---:|
| Training | 72 |
| Testing | 19 |

The emotion recognition models are evaluated on speakers excluded from the training partition.

## Method

### Feature Extraction

Each frame is represented using:

- 20 inner-lip landmarks (60 dimensions)
- Mouth Aspect Ratio (MAR)
- Inter-frame motion magnitude

This produces a 62-dimensional feature vector.

Videos are segmented into 30-frame windows with 50% overlap.

### Model Architectures

The evaluated models include:

- MLP handcrafted-feature baseline
- 1D-CNN with 60-dimensional input
- 1D-CNN with 62-dimensional input
- CNN–GRU with 62-dimensional input

The best-performing configuration combines:

1. 1D convolutional feature extraction
2. Bidirectional GRU temporal modeling
3. Fully connected emotion classification

## Emotion Recognition Results

### Overall Accuracy

| Model | Accuracy |
|---|---:|
| MLP baseline | 36.0% |
| 1D-CNN (60-dim) | 42.8% |
| 1D-CNN (62-dim) | 44.6% |
| CNN–GRU (62-dim) | 45.1% |

The CNN–GRU model achieved the highest reported accuracy among the evaluated configurations.

### Per-Class Results

Results for the best-performing CNN–GRU model:

| Emotion | Precision | Recall | F1 |
|---|---:|---:|---:|
| Anger | 0.50 | 0.48 | 0.49 |
| Disgust | 0.52 | 0.49 | 0.51 |
| Fear | 0.28 | 0.24 | 0.26 |
| Happy | 0.55 | 0.74 | 0.63 |
| Neutral | 0.40 | 0.44 | 0.42 |
| Sad | 0.38 | 0.34 | 0.36 |

### Key Observations

- Happy achieved the highest F1-score (0.63) among the evaluated emotion classes.
- Fear had the lowest F1-score (0.26).
- The 62-dimensional 1D-CNN configuration outperformed the 60-dimensional configuration in the reported comparison.
- The CNN–GRU achieved a modest improvement over the best 1D-CNN configuration.

## Exploratory Speaker-Identity Analysis

The repository includes an exploratory speaker-classification script (`src/privacy_leakage.py`) intended to investigate whether speaker information can be recovered from the learned representations.

However, the initial evaluation has an important methodological limitation.

The speaker classifier was trained using identities from the 72-speaker training partition and evaluated on 19 speakers whose identities were absent from its training classes. Consequently, the classifier could not correctly predict the identities of these held-out speakers using its predefined output classes.

The resulting 0% speaker-identification accuracy does **not** demonstrate privacy protection or resistance to speaker identification.

A valid evaluation of identity leakage requires an attack protocol capable of assessing previously unseen speakers, such as speaker verification using held-out identities and genuine/impostor trials.

**The current speaker-identification result is therefore excluded from the project's privacy claims.**

## Privacy Motivation and Scope

Conventional facial emotion recognition systems may expose identity-related information through full-face imagery.

This project explores a more restricted visual representation based on inner-lip motion rather than full-face appearance.

However, restricting the input representation is not equivalent to removing biometric identity information. Speaker characteristics may remain encoded in articulatory movement patterns.

The project therefore provides an exploratory investigation into lip-motion-based emotion recognition, rather than a validated speaker-anonymization system.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required dependencies include:

- PyTorch
- NumPy
- Pandas
- OpenCV
- MediaPipe
- scikit-learn
- tqdm

The repository includes the MediaPipe `face_landmarker.task` model in its root directory.

## Usage

Place CREMA-D videos inside:

```text
data/raw/
```

Run the data preparation and training pipeline:

```bash
python src/build_metadata.py
python src/windowing.py
python src/split_data.py
python src/train_model.py
python src/train_cnn_gru.py
```

The exploratory speaker-identity classification script can be found at:

```text
src/privacy_leakage.py
```

Its existing evaluation should not be interpreted as a valid privacy measurement, for the reasons described in the previous section.

## Repository Structure

```text
.
├── data/
├── results/
├── src/
├── face_landmarker.task
├── requirements.txt
└── README.md
```

## Limitations

- The current emotion recognition results are obtained from a single speaker-disjoint train/test split.
- Performance varies considerably across emotion classes.
- The current speaker-classification experiment does not provide valid evidence of identity protection.
- No validated speaker-verification privacy evaluation has been completed.
- Generalization to other datasets and recording conditions has not been established.

## Future Work

Potential extensions include:

- Speaker-verification attacks using previously unseen identities
- Adversarial privacy training
- Cross-dataset evaluation
- Transformer-based temporal architectures
- Explainability analysis
- Exploration of multimodal audio-visual approaches

## Citation

```bibtex
@misc{norouzi2026lipmotion,
  title={Lip-Motion Emotion Recognition with Exploratory Speaker-Identity Analysis},
  author={Sepideh Norouzi Motlagh},
  year={2026},
  note={GitHub repository}
}
```

## Author

Sepideh Norouzi Motlagh

GitHub: [@SepidehNorouziMotlagh](https://github.com/SepidehNorouziMotlagh)
