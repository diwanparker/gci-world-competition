# GCI World 2026 In-Class Machine Learning Competition

## Home Credit Default Risk Prediction — Solution Repository
* **Author:** Muhammad Rafi Arsyad (`diwanparker`)
* **Organization:** The University of Tokyo — Matsuo-Iwasawa Laboratory
* **Competition Assignments:**
  * Assignment ID `1038`: CSV Prediction Submission (`output/submission.csv`)
  * Assignment ID `1046`: Reproducible Archive Bundle (`competition_solution.zip` for Honors consideration)
* **Public Leaderboard Benchmark:** **`0.76821`** (Rank **#3 Worldwide** out of 176 participants)
* **Evaluation Metric:** Area Under the ROC Curve (**ROC-AUC**)

---

## 1. Executive Summary & Solution Architecture

This repository contains the verified, reproducible, clean-code solution for the GCI World 2026 In-Class Machine Learning Competition. The task models loan default risk using applicant demographic, credit application, and credit bureau features.

### Key Engineering Highlights:
1. **Target Anomaly & Missingness Repair**:
   * Resolved anomalous sentinel values (`DAYS_EMPLOYED == 365243`, indicating non-working/retired status) by converting to `np.nan` and creating an explicit anomaly indicator `DAYS_EMPLOYED_ANOM`.
   * Explicit missingness flags and total record NaN count features (`NAN_COUNT`).
2. **Domain-Specific Financial Engineering (121 Features)**:
   * **Credit & Debt Burdens**: `CREDIT_TERM`, `ANNUITY_CREDIT`, `CREDIT_GOODS`, `CREDIT_GOODS_DIFF`, and `CREDIT_INCOME`.
   * **Family Economics**: `INCOME_PER_PERSON`, `INCOME_PER_CHILD`, `ANNUITY_PER_PERSON`, `CHILDREN_RATIO`, and estimated disposable income `INCOME_LEFT`.
   * **Golden Features Composite Signals (`EXT_SOURCE_1, 2, 3`)**: Weighted external score, statistical aggregations (mean, std, min, max, product), pairwise interactions (`EXT_1 * EXT_2`, `EXT_1 - EXT_2`), and temporal interactions (`EXT / AGE`).
   * **Demographic Group-Relative Aggregations**: Normalized income, credit limit, and credit term relative to occupation, education, and organization peers.
3. **Out-of-Fold Empirical Bayes Target Encoding**:
   * Target-encoded high-cardinality nominals (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`) strictly within fold splits with smoothing parameter $\alpha = 50.0$ to guarantee zero target leakage.
4. **Validation Strategy (5-Fold Stratified CV)**:
   * 5-Fold Stratified K-Fold CV preserves the ~8.07% minority default rate across all training folds.
5. **Multi-Architecture Rank-Averaging Ensemble**:
   * Evaluated and blended regularized **LightGBM** (CV AUC: `0.76365`) and **CatBoost** (CV AUC: `0.76236`).
   * Predictions are transformed into uniform quantile ranks before blending (`0.5 * rank(LightGBM) + 0.5 * rank(CatBoost)`), maximizing the rank-order ROC-AUC metric and achieving an ensemble CV AUC of `0.76481` and Public Leaderboard Score of **`0.76821`**.

---

## 2. Directory Structure

```text
.
├── README.md                      # Comprehensive project documentation & reproduction guide
├── solution_reproducible.ipynb    # Clean Jupyter notebook reproducing the complete pipeline
├── competition_solution.zip      # ZIP bundle for Assignment 1046 (Honors consideration)
├── download_data.py               # Dataset acquisition script from Matsuo Lab Google Drive
├── train.py                       # Single-command end-to-end training and inference entrypoint
├── submit.py                      # Automated Omnicampus API submitter & leaderboard monitor
├── src/                           # Clean architecture module
│   ├── __init__.py                # Package exports
│   ├── features.py                # 121 domain features + out-of-fold target encoding
│   └── models.py                  # 5-Fold CV LightGBM, CatBoost, and Rank-Averaging ensembler
└── output/
    ├── submission.csv             # The winning 0.76821 prediction file (61,500 rows)
    └── preds/                     # Cached out-of-fold and test prediction matrices (.npy)
```

---

## 3. Reproduction Instructions

### Prerequisites
* Python 3.10+
* Required libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `catboost`, `requests`, `httpx`

```bash
pip install numpy pandas scipy scikit-learn lightgbm catboost requests httpx
```

### Step 1: Download Data
Download the official dataset directly to `./input`:
```bash
python download_data.py
```

### Step 2: Run End-to-End Training
To train 5-Fold LightGBM + 5-Fold CatBoost from scratch and output the final predictions:
```bash
python train.py
```
*(Optional: To instantly rebuild `output/submission.csv` using the cached model fold predictions, run `python train.py --use-cached-preds`)*

### Step 3: Submit to Omnicampus Autograder
To submit the generated CSV and poll the leaderboard:
```bash
python submit.py --csv output/submission.csv
```

---

## 4. Benchmark Validation Results

| Model Architecture | 5-Fold OOF ROC-AUC | Public Leaderboard Score |
| :--- | :---: | :---: |
| Baseline Benchmark | 0.66819 | 0.66819 |
| Tuned LightGBM v1 | 0.76120 | 0.76468 |
| Deep LightGBM v2 | 0.76280 | 0.76544 |
| LightGBM (121 Features + TE) | 0.76365 | 0.76813 |
| CatBoost (121 Features) | 0.76236 | — |
| **Champion Ensemble (LightGBM + CatBoost Rank Average)** | **`0.76481`** | **`0.76821`** (Rank **#3 Worldwide**) |
