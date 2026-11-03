# Heart Sound Classification with a CNN

A course project that classifies heart sound recordings as *normal* or *abnormal* using a convolutional neural network (CNN) on 3-channel spectrograms.

> Individual project · CSC532 Machine Learning, King Mongkut's University of Technology Thonburi (KMUTT)

## Overview

- **Dataset:** [CirCor DigiScope Phonocardiogram Dataset](https://physionet.org/content/circor-heart-sound/) (PhysioNet): 2,633 recordings from 779 patients, labelled *Normal* / *Abnormal* from each patient's clinical outcome.
- **Features:** each recording is trimmed or padded to 5 seconds and converted to a 128 × 128 × 3 input (log-Mel spectrogram, delta and delta-delta).
- **Model:** three Conv2D blocks (BatchNorm, Dropout, pooling) → Dense(128) → sigmoid, trained with class weights and early stopping.
- **Imbalance handling:** augmentation of abnormal recordings (time-stretch, pitch-shift, noise) and class weights.

## Results and what I learned

The course version reported about **82% accuracy**. When I later re-audited the notebook, I found two problems in how it was evaluated:

1. **Augmentation leakage.** Abnormal recordings were augmented *before* the train/test split, so test data contained altered audio and the model could learn "altered audio = abnormal".
2. **Patient leakage.** Each patient has several recordings, and the random split put recordings from the same patient in both training and testing.

After fixing both in `HeartSound_Reaudit.ipynb` (augmenting the training data only, and splitting by **patient** with `StratifiedGroupKFold`), performance on unseen patients dropped to about **chance level** (around 50% accuracy). In other words, this model does not reliably predict the clinical outcome from short recordings.

**Key lessons:**
- Evaluate at the right level: split by **patient**, not by recording, whenever one patient has several samples.
- Apply augmentation **only to training data**, after the split.
- A high headline accuracy is a reason to check the evaluation, not a reason to stop.

## Repository contents

| File | Description |
|---|---|
| `HeartSound_Course.ipynb` | **Course version** (Google Colab): data loading, 3-channel feature extraction, CNN training and evaluation, as submitted for the course |
| `HeartSound_Reaudit.ipynb` | **Re-audit**: the same model with leak-free evaluation (augmentation on training data only, patient-level `StratifiedGroupKFold`, separate early-stopping patients, 20% patient hold-out test set) |

## How to run

1. Download the CirCor dataset from PhysioNet and upload the `training_data/` folder to Google Drive.
2. Open the notebook in **Google Colab** (a GPU runtime is recommended).
3. Set `DATASET_PATH` to your Drive folder, e.g. `/content/drive/MyDrive/training_data`.
4. Run all cells.

## Tech stack
Python · TensorFlow / Keras · librosa · NumPy · pandas · scikit-learn · Matplotlib · Google Colab

## Possible next steps
- Use full-length recordings split into windows, instead of a single 5-second clip
- Try pretrained audio models (e.g. PANNs, YAMNet) as feature extractors
- Predict murmur presence per recording location, which is more directly audible than overall clinical outcome

> **Disclaimer:** for educational purposes only. This is not a medical diagnostic tool.
