# Research Project: Predicting Stress Resilience from Nap EEG

Stress resilience is the capacity to maintain, or rapidly recover, adaptive psychological and behavioral functioning under adversity, uncertainty, and sustained pressure. In this project you will build a **subject-level regression** that predicts a CD-RISC-10 score from one afternoon-nap EEG recording.

This drop is for **group discussions 1–3**. The hidden test set is **not** included and will be released later.

## Target and recordings

- **Target:** CD-RISC-10, integer-valued in practice, range **0–40**. Higher = greater stress resilience. One score per person.
- **Hardware:** 129-channel EGI HydroCel Geodesic Sensor Net (128 scalp + 1 reference), **250 Hz**.
- **Length:** at least 30 minutes per recording.
- **IDs:** zero-padded filename stem, e.g. `067.mat` → `"067"`. Always treat IDs as **strings**.

## Repository layout

```text
Group_Project_Resilience/
  README.md          this file
  EEG/               50 EEG recordings (.mat): 40 train + 10 validation
  label.csv          true CD-RISC-10 scores for the 40 training IDs
  val.csv            the 10 validation IDs and labels
```

## What is inside each file type

### `EEG/<id>.mat` — one subject, one nap

Each file is a MATLAB workspace. 


| Variable                    | Shape                  | Meaning                                                                                |
| --------------------------- | ---------------------- | -------------------------------------------------------------------------------------- |
| Signal matrix               | **129 × T**, `float32` | Channels **E1–E129** (rows) × time samples (columns). `T` is at least `30 × 60 × 250`. |
| `EEGSamplingRate`           | `1 × 1`                | Sampling rate in Hz (**250**).                                                         |
| Impedance vector (optional) | **129 × 1**            | Present in **22 / 50** files only.                                                     |


### Preprocessing

The `.mat` files are **exported nap EEG**, not a finished analysis dataset. A reasonable first clean-up, consistent with that setup and with the course’s EEG practice, is:

- **High-pass** to remove slow drift (sweat, drying sponges, baseline wander) while leaving delta (from ~0.5 Hz) intact  
- **Low-pass** to keep sleep-relevant bands (up through beta) and attenuate muscle / high-frequency noise.
- **Notch 50 Hz** if mains hum remains.

You may use a different pipeline. Report what you did.

### `label.csv` — training labels

UTF-8 CSV with header `id,label`.


| Column  | Type             | Meaning                                                |
| ------- | ---------------- | ------------------------------------------------------ |
| `id`    | string, 3 digits | Matches `EEG/<id>.mat`.                                |
| `label` | float            | Ground-truth CD-RISC-10 for that **training** subject. |


### `val.csv` — validation set

Same columns as `label.csv`: `id,label`.


| Column  | Meaning                                                    |
| ------- | ---------------------------------------------------------- |
| `id`    | Validation subject. File is `EEG/<id>.mat`.                |
| `label` | Ground-truth CD-RISC-10 for scoring **after** you predict. |


10 rows. These people are **unseen** during training. Predict a score for each ID, then compute **RMSE** (lower is better) and **R²** (higher is better) against the true `label` values.

**Leakage (training on validation EEG or labels, or using test data) → zero marks for that discussion.**

## Split


| Split      | n   | EEG              | Labels           | Use now                               |
| ---------- | --- | ---------------- | ---------------- | ------------------------------------- |
| Train      | 40  | `EEG/`           | `label.csv`      | Fit the model                         |
| Validation | 10  | `EEG/`           | `val.csv`        | Unseen evaluation for discussions 1–3 |
| Test       | 10  | not in this drop | not in this drop | Final group project only              |


## What to do in this round

1. Load and preprocess the 40 training `.mat` files and join them to `label.csv` by `id`.
2. Train a regression: EEG sequence → one score in [0, 40].
3. Predict the 10 IDs in `val.csv` without using those recordings or labels in fitting.
4. Report RMSE and R² on the validation set in the discussion slides, plus your code.

The submitted adapter selects Ridge regularization with leave-one-subject-out
cross-validation on the 40 training participants. Predictions are clipped to the
valid CD-RISC-10 range [0, 40] after fitting.

## Data-use requirement

The data are anonymized and were collected with participant consent. They are provided only for this course/research project. **Do not redistribute** the EEG recordings, labels, or derived subject-level data.
