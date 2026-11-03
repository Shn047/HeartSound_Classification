# Heart Sound Classification with Deep Learning

**A convolutional neural network (CNN) that classifies heart sound recordings as _normal_ or _abnormal_, as a first step toward objective, low-cost screening support for clinicians.**

> Individual project · CSC532 Machine Learning, King Mongkut's University of Technology Thonburi (KMUTT)

**Case study:** https://shn047.github.io/projects/heart-sound.html

---

## Why

Heart disease is the leading cause of death worldwide. Screening often starts with a doctor listening through a stethoscope (auscultation), which depends on experience, can be subjective, and needs specialists who are scarce in many places. An automated model can make first-line screening more objective and more widely available.

## Results (final model)

Test set: 404 recordings (stratified 80/20 split)

| Metric | Test set | 5-fold stratified CV |
|---|---|---|
| Accuracy | **82%** | **80.19% ± 1.08%** |
| Abnormal precision | **1.00** | 0.97 |
| Abnormal recall | 0.39 | 0.38 |
| Abnormal F1-score | 0.56 | 0.54 |

**Confusion matrix (test set)**

| | Predicted normal | Predicted abnormal |
|---|---|---|
| **Actually normal** | 283 | 0 |
| **Actually abnormal** | 74 | 47 |

**What this means:** the model raises **no false alarms**. When it flags a recording as abnormal, it's right. But it **misses about 6 in 10 abnormal recordings**. For screening, catching cases (recall) matters most, so this is the main limitation and the focus of future work.

Predictions were also **reviewed with a physician** experienced in auscultation, who found them mostly consistent with clinical judgement. They valued the high precision for avoiding unnecessary referrals, and stressed the need for interpretable outputs.

## Data

- **Dataset:** [CirCor DigiScope Phonocardiogram Dataset](https://physionet.org/content/circor-heart-sound/) (PhysioNet)
- **2,633 labelled recordings**, with labels taken from each patient's outcome field (normal / abnormal)
- **Class balance:** about 60% normal and 40% abnormal
- Files: `.wav` audio, `.hea` headers, `.tsv` segmentation annotations, `.txt` patient metadata

## Method

1. **Standardise:** trim or pad every recording to **5 seconds**.
2. **Features:** convert each clip into a **128 × 128 × 3 spectrogram**:
   - channel 1: log-Mel spectrogram
   - channel 2: delta (1st derivative)
   - channel 3: delta-delta (2nd derivative)
3. **Imbalance:** augment abnormal recordings (time-stretching, pitch-shifting, white noise) and use **class weights** in the loss.
4. **Split:** stratified 80/20 train/test, plus **5-fold stratified cross-validation**.

### Model

```
Input 128×128×3
→ Conv2D(32)  → BatchNorm → Dropout(0.2) → MaxPooling
→ Conv2D(64)  → BatchNorm → Dropout(0.2) → MaxPooling
→ Conv2D(128) → BatchNorm → Dropout(0.3) → GlobalAveragePooling
→ Dense(128)  → Dropout(0.5)
→ Dense(1, sigmoid)
```
- Loss: binary cross-entropy · Optimiser: Adam · Batch size: 32
- Early stopping (patience 5), restoring the best validation weights

## Repository contents

| File | Description |
|---|---|
| `ML_Project.ipynb` | Google Colab notebook: data loading, preprocessing, augmentation, CNN training and evaluation |

> **Note on versions:** the notebook currently in this repository is an **earlier baseline** of the project (a simpler feature setup, evaluated on 260 test samples). The results above come from the **final model** presented at the end of the course. The final notebook will be added here.

## How to run

1. Download the CirCor dataset from PhysioNet and upload the `training_data/` folder to Google Drive.
2. Open `ML_Project.ipynb` in **Google Colab** (a GPU runtime is recommended).
3. Update `dataset_path` in the notebook to your Drive folder, e.g. `/content/drive/MyDrive/training_data`.
4. Run all cells.

## Tech stack
Python · TensorFlow / Keras · librosa · NumPy · scikit-learn · Matplotlib / Seaborn · Google Colab

## Next steps
- **Raise recall:** tune the decision threshold to trade some precision for catching more abnormal cases.
- **Cleaner signal:** add bandpass filtering to remove noise outside the heart-sound frequency range.
- **Explainability:** add Grad-CAM so clinicians can see which parts of the sound drove a prediction.

## Ethics
- **Decision support, not diagnosis.** Final decisions stay with healthcare professionals.
- **Privacy.** Uses only anonymised, public data from PhysioNet.
- **Fairness.** Class imbalance is addressed with augmentation and class weights.

> **Disclaimer:** for educational and research purposes only. This is **not** a medical diagnostic tool.
