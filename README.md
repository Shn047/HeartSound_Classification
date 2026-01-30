# Heart Sound Classification Using Deep Learning

## Overview
This project applies deep learning techniques to classify heart sound recordings as **normal** or **abnormal**, aiming to support automated and objective cardiac screening. The work was completed as an individual Machine Learning course project using real-world medical audio data.

---

## Dataset
- CIRCor Heart Sound Dataset (PhysioNet)
- Expert-labeled normal and abnormal heart sound recordings
- Challenges: noise, variable signal length, class imbalance

---

## Approach
- Audio preprocessing and segmentation
- Feature extraction using spectrogram-based representations and MFCCs
- CNN-based binary classification
- Data augmentation and weighted loss to handle class imbalance

---

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score
- Emphasis on recall and precision for abnormal heart sounds
- Validation using stratified cross-validation

---

## Results
- ~82% classification accuracy
- Stable performance across validation folds
- Promising ability to distinguish abnormal heart sounds in noisy data

---

## Key Learnings
- Preprocessing is critical for audio-based ML
- Metric choice must align with real-world priorities
- Ethical considerations are important in medical AI

---

## Tools
Python, CNN, Audio Processing, MFCCs, Spectrograms

---

## Disclaimer
This project is for educational and research purposes only and is not a medical diagnostic tool.
