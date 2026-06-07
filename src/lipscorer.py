"""
lip_scorer.py
=============
Lip‑motion feature extractor using MediaPipe Face Landmarker (Tasks API).

Provides two modes:
  1. **Activity scoring** – original per‑frame lip‑activity score (0‑1)
     based on Mouth Aspect Ratio and inter‑frame motion.
  2. **Landmark export** – extracts raw inner‑lip landmark coordinates
     (60‑dim vectors) plus MAR and motion for each frame, suitable for
     training a temporal CNN for emotion recognition.

Requirements (already in requirements.txt):
  - mediapipe >= 0.10.0
  - opencv-python
  - numpy
  - pandas (for CSV export)

Place the file `face_landmarker.task` in the project root.
Download it from:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Project structure (as seen on your machine):
  F:/sepide/Projects/Emotion Recognition from Lip-Motion Dynamics for Privacy‑Preserving Multimodal Systems/
  ├── face_landmarker.task
  ├── src/
  │   └── lipscorer.py   (this file)
  ├── data/
  │   ├── raw/            (your CREMA‑D .flv / .mp4 files, all flat)
  │   └── features/       (output CSVs will be saved here)
  └── ...
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from typing import Optional
import pandas as pd

# =============================================================================
# Project‑relative path to the face landmarker model file
# =============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FACE_LANDMARKER_PATH = _PROJECT_ROOT / "face_landmarker.task"

if not _FACE_LANDMARKER_PATH.exists():
    raise FileNotFoundError(
        f"Face Landmarker task file not found: {_FACE_LANDMARKER_PATH}\n"
        "Please download it from https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

# Convenience paths for batch processing
_RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
_FEATURES_DIR = _PROJECT_ROOT / "data" / "features"


class LipScorer:
    """
    Multi‑purpose lip‑motion analyser using MediaPipe Face Landmarker.
    """

    # Indices of inner lip landmarks (closed loop around the inner mouth)
    INNER_LIP_INDICES = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95
    ]

    def __init__(self, smooth_window: int = 3, motion_threshold: float = 0.001):
        self.smooth_window = smooth_window
        self.motion_threshold = motion_threshold

        # Store the base options so we can create a fresh landmarker per video
        self._base_options = python.BaseOptions(model_asset_path=str(_FACE_LANDMARKER_PATH))
        self._landmarker_options = vision.FaceLandmarkerOptions(
            base_options=self._base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )

        # Initial landmarker (will be replaced for each video to avoid timestamp issues)
        self.landmarker = vision.FaceLandmarker.create_from_options(self._landmarker_options)

    def _create_fresh_landmarker(self):
        """Create a brand‑new FaceLandmarker instance for a new video."""
        self.landmarker = vision.FaceLandmarker.create_from_options(self._landmarker_options)

    # ------------------------------------------------------------------
    # Original activity‑scoring methods (unchanged)
    # ------------------------------------------------------------------
    def _compute_mar(self, landmarks: dict) -> float:
        vert_pairs = [(13, 14), (82, 87), (312, 317)]
        horiz_pair = (78, 308)

        def distance(p1, p2):
            return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

        vert_dists = [distance(landmarks[a], landmarks[b]) for a, b in vert_pairs]
        horiz_dist = distance(landmarks[horiz_pair[0]], landmarks[horiz_pair[1]])

        if horiz_dist < 1e-6:
            return 0.0
        return (sum(vert_dists) / len(vert_dists)) / horiz_dist

    def process_video(self, video_path: str) -> np.ndarray:
        """Original activity‑score extraction (unchanged)."""
        self._create_fresh_landmarker()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        scores = []
        prev_landmarks = None
        frame_idx = -1
        prev_timestamp_ms = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= prev_timestamp_ms:
                timestamp_ms = prev_timestamp_ms + 1
            prev_timestamp_ms = timestamp_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                face_lm = result.face_landmarks[0]
                h, w, _ = frame.shape

                landmarks = {}
                for idx in self.INNER_LIP_INDICES:
                    lm = face_lm[idx]
                    landmarks[idx] = (lm.x * w, lm.y * h, lm.z)

                mar = self._compute_mar(landmarks)
                motion = 0.0
                if prev_landmarks is not None:
                    disp_sum = 0.0
                    count = 0
                    for idx in self.INNER_LIP_INDICES:
                        if idx in landmarks and idx in prev_landmarks:
                            p1 = np.array(landmarks[idx][:2])
                            p2 = np.array(prev_landmarks[idx][:2])
                            disp_sum += np.linalg.norm(p1 - p2)
                            count += 1
                    if count > 0:
                        motion = disp_sum / count

                mar_norm = min(mar / 0.8, 1.0)
                motion_norm = min(motion / 0.005, 1.0)
                score = 0.6 * mar_norm + 0.4 * motion_norm
                score = max(0.0, min(score, 1.0))
                if mar < 0.05 and motion < self.motion_threshold:
                    score = 0.0
                prev_landmarks = landmarks.copy()
            else:
                score = 0.0
                prev_landmarks = None

            scores.append(score)

        cap.release()
        scores = np.array(scores)
        if len(scores) > 0 and self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            scores = np.convolve(scores, kernel, mode='same')
            scores = np.clip(scores, 0.0, 1.0)
        return scores

    # ------------------------------------------------------------------
    # Landmark‑sequence extraction for emotion recognition
    # ------------------------------------------------------------------
    def extract_landmark_sequences(
            self,
            video_path: str,
            save_csv: Optional[str] = None,
            handle_missing: str = "skip"
    ) -> pd.DataFrame:
        """
        Extract per‑frame inner‑lip landmark coordinates (60‑dim),
        MAR, and motion from a video, optionally saving to CSV.
        """
        self._create_fresh_landmarker()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames_data = []
        prev_landmarks = None
        frame_idx = -1
        prev_timestamp_ms = -1

        # Helper to create a NaN row for missing frames (interpolate mode)
        def _nan_row(f_idx):
            row = {"frame": f_idx, "mar": np.nan, "motion": np.nan}
            for i in range(len(self.INNER_LIP_INDICES)):
                row[f"lm_{i}_x"] = np.nan
                row[f"lm_{i}_y"] = np.nan
                row[f"lm_{i}_z"] = np.nan
            return row

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= prev_timestamp_ms:
                timestamp_ms = prev_timestamp_ms + 1
            prev_timestamp_ms = timestamp_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                face_lm = result.face_landmarks[0]
                h, w, _ = frame.shape

                landmarks = {}
                for idx in self.INNER_LIP_INDICES:
                    lm = face_lm[idx]
                    landmarks[idx] = (lm.x * w, lm.y * h, lm.z)

                mar = self._compute_mar(landmarks)
                motion = 0.0
                if prev_landmarks is not None:
                    disp_sum = 0.0
                    count = 0
                    for idx in self.INNER_LIP_INDICES:
                        if idx in landmarks and idx in prev_landmarks:
                            p1 = np.array(landmarks[idx][:2])
                            p2 = np.array(prev_landmarks[idx][:2])
                            disp_sum += np.linalg.norm(p1 - p2)
                            count += 1
                    if count > 0:
                        motion = disp_sum / count

                # Flatten landmarks into 60‑dim vector
                lm_vec = []
                for idx in self.INNER_LIP_INDICES:
                    x, y, z = landmarks.get(idx, (0.0, 0.0, 0.0))
                    lm_vec.extend([x, y, z])

                row = {"frame": frame_idx, "mar": mar, "motion": motion}
                for i, idx in enumerate(self.INNER_LIP_INDICES):
                    row[f"lm_{i}_x"] = lm_vec[3 * i]
                    row[f"lm_{i}_y"] = lm_vec[3 * i + 1]
                    row[f"lm_{i}_z"] = lm_vec[3 * i + 2]

                frames_data.append(row)
                prev_landmarks = landmarks.copy()
            else:
                if handle_missing == "skip":
                    prev_landmarks = None
                    continue
                elif handle_missing == "zeros":
                    row = {"frame": frame_idx, "mar": 0.0, "motion": 0.0}
                    for i in range(len(self.INNER_LIP_INDICES)):
                        row[f"lm_{i}_x"] = 0.0
                        row[f"lm_{i}_y"] = 0.0
                        row[f"lm_{i}_z"] = 0.0
                    frames_data.append(row)
                    prev_landmarks = None
                elif handle_missing == "interpolate":
                    # Append a row of NaN values; interpolation will fill later
                    frames_data.append(_nan_row(frame_idx))
                    prev_landmarks = None
                else:
                    raise ValueError(f"Unknown handle_missing mode: {handle_missing}")

        cap.release()

        df = pd.DataFrame(frames_data)

        # Ensure all feature columns are numeric (avoid object-dtype interpolation error)
        numeric_cols = ["frame", "mar", "motion"] + [
            f"lm_{i}_{axis}" for i in range(len(self.INNER_LIP_INDICES)) for axis in ["x", "y", "z"]
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if handle_missing == "interpolate" and not df.empty:
            try:
                df.interpolate(method="linear", inplace=True, limit_direction="both")
                df.fillna(0.0, inplace=True)
            except (TypeError, ValueError):
                df.fillna(0.0, inplace=True)

        if save_csv is not None and not df.empty:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_csv, index=False)

        return df

    # ------------------------------------------------------------------
    # Convenience batch processor
    # ------------------------------------------------------------------
    def process_directory(
        self,
        video_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        handle_missing: str = "skip",
        video_extensions: tuple = (".mp4",)
    ) -> None:
        video_dir = Path(video_dir or _RAW_DATA_DIR)
        output_dir = Path(output_dir or _FEATURES_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.rglob(f"*{ext}"))

        print(f"Found {len(video_files)} video(s) to process.")
        for vf in sorted(video_files):
            rel_path = vf.relative_to(video_dir)
            csv_path = output_dir / rel_path.with_suffix(".csv")
            print(f"Processing: {vf.name} → {csv_path}")
            self.extract_landmark_sequences(
                str(vf), save_csv=str(csv_path), handle_missing=handle_missing
            )
        print("Done.")


# =============================================================================
if __name__ == "__main__":
    scorer = LipScorer()
    scorer.process_directory(handle_missing="interpolate")