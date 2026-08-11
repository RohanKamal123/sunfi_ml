# Machine Learning-Based Prediction of Polycystic Ovary Syndrome (PCOS)

**Group 5** — Fatema Ferdous (0152410052) · Wafa Haque (0152420023) · Khondoker Sazzad Sunfi (0152310002)

An end-to-end machine learning study predicting PCOS from clinical, hormonal and
physical measurements, built to address the gaps identified in our literature review of
five recent papers.

---

## Headline result

| | Value |
|---|---|
| Best model | Random Forest (SMOTE + Chi-Square feature selection) |
| Cross-validated ROC-AUC | **0.954 ± 0.029** (10-fold × 3 repeats) |
| Cross-validated accuracy | **0.899 ± 0.036** |
| Held-out test accuracy | **0.917** |
| Held-out test ROC-AUC | **0.942** |
| Test recall / specificity | 0.833 / 0.959 |
| **Accuracy manufactured by a leaky protocol** | **+3.2 points** (+9.0 F1) |

The last row is the finding we consider most important. See
[The leakage experiment](#the-leakage-experiment) below.

---

## What this project does differently

Our gap analysis of the five reviewed papers found five recurring problems. Here is what
was done about each:

| Gap identified in the review | What this project does |
|---|---|
| **Unaddressed class imbalance** (364 vs 177) — no paper explains how it was handled | SMOTE applied *inside* each CV fold, plus a 3-way ablation against no-balancing and class-weighting |
| **Possible overfit/leaked accuracy** — 98–99% on 541 records with no protocol stated | Every fitted step lives in a pipeline; a dedicated experiment **measures** how much accuracy a leaky protocol invents |
| **Limited explanation** — only 1 of 5 papers used XAI | SHAP global importance, directional effects, and per-patient waterfall plots |
| **Single train/test split** on 541 rows | Repeated stratified 10-fold CV (3 repeats) *and* a held-out test set that nothing touches until the end |
| **No comparison of feature selection methods** | Correlation, Chi-Square and RFE all implemented as transformers and compared under identical CV |

Two gaps we identified and did **not** close, stated plainly: no external validation
cohort was available to us, and the dataset contains no ultrasound imagery. Both remain
open problems — see [Limitations](#limitations).

---

## The leakage experiment

Papers 4 and 5 report 98.9–99.3% accuracy on the same 541-row dataset. Our correctly
validated pipeline reaches ~90%. Rather than speculate about the gap, we measured it.

The same classifier, the same data and the same folds are run under two protocols that
differ in exactly one respect — **where SMOTE and feature selection happen**:

| Protocol | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Correct (SMOTE + selection inside CV) | 0.893 | 0.840 | 0.831 | 0.834 | 0.951 |
| Leaky (SMOTE + selection before CV) | 0.924 | 0.924 | 0.929 | 0.924 | 0.974 |
| **Inflation** | **+0.032** | **+0.083** | **+0.098** | **+0.090** | **+0.023** |

When SMOTE runs before the split, synthetic minority rows interpolated from training
patients end up in the validation fold — the model is partly scored on points derived
from its own training data. When RFE runs before the split, the selector has already read
every label, including the test labels.

This does not prove the reviewed papers made these mistakes; most do not describe their
protocol in enough detail to tell, which is itself the finding. What it shows is that a
98–99% headline on this dataset is **reachable through methodology alone**, so such
numbers cannot be taken at face value without a stated protocol.

---

## Results

### Model comparison (repeated stratified 10-fold CV, training split only)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.899 ± 0.036 | 0.871 | 0.818 | 0.840 | **0.954 ± 0.029** |
| Stacking Ensemble | 0.897 ± 0.032 | 0.847 | 0.839 | 0.840 | 0.953 ± 0.031 |
| SVM (RBF) | 0.889 ± 0.045 | 0.825 | 0.844 | 0.831 | 0.948 ± 0.034 |
| Logistic Regression | 0.887 ± 0.041 | 0.805 | 0.870 | 0.833 | 0.947 ± 0.036 |
| XGBoost | 0.875 ± 0.043 | 0.825 | 0.794 | 0.804 | 0.945 ± 0.035 |

Read the standard deviations before the means. All five models sit within roughly one
SD of each other — on 432 training rows they are **not** meaningfully distinguishable.
Random Forest is selected on a tie-break, not on a real margin. Notably, the stacking
ensemble that dominates the literature gives no advantage here once the protocol is sound.

### Feature selection

| Strategy | Features kept | Mean CV ROC-AUC |
|---|---|---|
| All features | 41 | 0.9388 |
| Correlation | 19 | 0.9470 |
| **Chi-Square** | **15** | **0.9473** |
| RFE | 15 | 0.9439 |

Differences are within noise. The practical argument for the reduced sets is that 15
measurements are cheaper to collect in a clinic than 41, and fewer parameters means less
room to overfit.

### Class imbalance ablation (test set)

| Strategy | Accuracy | Recall | F1 | Specificity |
|---|---|---|---|---|
| No balancing | 0.917 | 0.833 | 0.870 | 0.959 |
| Class weighting | 0.908 | 0.861 | 0.861 | 0.932 |
| **SMOTE** | **0.927** | **0.861** | **0.886** | 0.959 |

SMOTE buys recall — it catches PCOS cases the unbalanced model misses — at no cost in
specificity. For a screening tool that trade is worth taking.

### What the model relies on (SHAP)

| Rank | Feature | Higher value means |
|---|---|---|
| 1 | Follicle count (right) | more likely PCOS |
| 2 | Follicle count (left) | more likely PCOS |
| 3 | Skin darkening | more likely PCOS |
| 4 | Hair growth | more likely PCOS |
| 5 | Weight gain | more likely PCOS |

The top features are follicle counts, then the hyperandrogenism symptom cluster, then
cycle irregularity — in order, the three legs of the **Rotterdam criteria**. The model
rediscovered the clinical diagnostic standard from data rather than latching onto a
spurious correlate, which is a stronger argument for trusting it than any accuracy figure.

---

## Quick start

```bash
git clone <this-repo>
cd sunfi_ml
pip install -r requirements.txt

# Full study: ~3-4 minutes, writes every figure and table
python -m src.run_pipeline

# Fast sanity check: fewer folds, skips SHAP
python -m src.run_pipeline --quick

# Tests, including the anti-leakage guarantees
python -m pytest tests/ -v
```

Or work through `notebooks/PCOS_Prediction.ipynb`, which walks the whole study with
narrative and inline figures (committed with outputs, so it reads without running).

---

## Repository layout

```
sunfi_ml/
├── data/raw/PCOS_data_without_infertility.xlsx   Source dataset (541 patients)
├── src/
│   ├── config.py         Paths, seed, column semantics
│   ├── data.py           Loading + the five classes of cleaning
│   ├── eda.py            Exploratory figures and summary statistics
│   ├── features.py       Correlation / Chi-Square / RFE selectors
│   ├── models.py         Pipelines — where leakage is prevented
│   ├── evaluate.py       CV, held-out testing, the leakage experiment
│   ├── explain.py        SHAP global + per-patient explanations
│   ├── literature.py     The five reviewed papers, as comparable data
│   ├── plots.py          Model-behaviour figures
│   └── run_pipeline.py   End-to-end runner
├── notebooks/PCOS_Prediction.ipynb
├── reports/figures/      18 generated figures
├── reports/results/      Result tables (CSV) + summary.json
└── tests/test_pipeline.py
```

---

## Method

### The pipeline

Every model is one `imblearn.Pipeline`:

```
impute → scale / one-hot encode → select features → SMOTE → classifier
```

This structure is the point. When the pipeline is handed to `cross_val_score`,
scikit-learn refits **every** step on each training fold, so the imputer, scaler,
selector and SMOTE never observe the data they are scored on. `tests/test_pipeline.py`
asserts this directly: `test_preprocessor_is_not_fitted_on_validation_folds` plants a
column whose scale differs between halves of the data and checks the fitted mean
reflects only the training fold.

`imblearn`'s pipeline rather than scikit-learn's is required because SMOTE changes the
number of rows: it must run on `fit` but be skipped on `predict`.

### Data cleaning

The raw workbook has five defects, each handled in `src/data.py`:

1. **Text-typed numeric columns** — `II beta-HCG` contains `"1.99."` (stray trailing dot)
   and `AMH` contains a literal `"a"`, so pandas reads both entire columns as `object`.
   Coerced with `errors="coerce"`.
2. **`Unnamed: 44`** — 2 non-null values out of 541. Dropped.
3. **Inconsistent whitespace** in column names (`" Age (yrs)"`, `"Height(Cm) "`). Stripped.
4. **Undefined cycle code** — `Cycle(R/I)` is coded 2 = regular, 4 = irregular, but one
   row contains 5. Recoded to a binary `Cycle_Irregular` indicator; the 5 becomes NaN.
5. **Impossible zeros** — a 0-day cycle length or 0 mm endometrium is a recording
   failure, not a measurement. Set to NaN.

Identifier columns (`Sl. No`, `Patient File No.`, which are duplicates of each other) are
dropped as clinically meaningless. Blood group is coded 11–18 for the eight ABO/Rh types;
the ordering is nominal, so it is one-hot encoded rather than treated as a number.

Missing values are deliberately **not** filled here — imputation is a fitted step and
belongs inside the pipeline.

### Evaluation protocol

1. Stratified 80/20 split, made **before** any modelling decision.
2. All model and feature selection on the training split only, via repeated stratified
   10-fold CV (3 repeats).
3. One pass over the test set, at the end.

---

## Dataset

Kaggle **Polycystic Ovary Syndrome (PCOS)** dataset by Prasoon Kottarathil —
541 patients from 10 hospitals in Kerala, India, with 43 clinical, hormonal and
physical parameters.

<https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos>

`PCOS_data_without_infertility.xlsx` (sheet `Full_new`) is committed under `data/raw/`
for reproducibility. Class distribution: **364 non-PCOS / 177 PCOS** (2.06 : 1).

The project uses only the main file. `PCOS_infertility.csv` covers a different question
(infertility outcomes) and is not needed for the diagnosis task.

---

## Limitations

1. **No external validation.** We criticised the reviewed papers for this and did not fix
   it either — no second cohort was available. Our held-out test set comes from the same
   541 patients and the same 10 Kerala hospitals, so it measures generalisation to new
   *patients*, not to new *populations*. This is the single biggest open problem in
   PCOS prediction work.
2. **Small sample.** 109 test rows, 36 of them PCOS cases. Three patients changing sides
   moves test accuracy by ~3 points. The cross-validated figures are the more trustworthy
   ones; every single-split number here should be read with that granularity in mind.
3. **No ultrasound imagery.** Also in our gap analysis, also not addressed. The dataset
   has follicle *counts* derived from ultrasound, but not the images.
4. **Self-reported symptoms.** Several strong features (weight gain, hair growth, fast
   food) are patient-reported and may be recorded *after* a clinical suspicion of PCOS has
   formed, making them partly a consequence of the diagnosis rather than a predictor.
5. **Ambiguous column semantics.** `Cycle length(days)` has a median of 5 and a range of
   2–12, so it is almost certainly bleeding duration rather than the ~28-day cycle
   interval the name suggests. The Kaggle documentation does not clarify this.
6. **Not a diagnostic tool.** This is coursework. PCOS diagnosis requires the Rotterdam
   criteria applied by a clinician.

---

## References

1. Thakre, V., Vedpathak, S., Thakre, K., & Sonawani, S. (2020). *PCOcare: PCOS Detection
   and Prediction using Machine Learning Algorithms*. Biosciences, Biotechnology Research
   Asia, 13(14), 240–244. https://doi.org/10.21786/bbrc/13.14/56
2. Zad, Z., Jiang, V. S., Wolf, A. T., Wang, T., Cheng, J. J., Paschalidis, I. C., &
   Mahalingaiah, S. (2024). *Predicting polycystic ovary syndrome with machine learning
   algorithms from electronic health records*. Frontiers in Endocrinology, 15, 1298628.
   https://doi.org/10.3389/fendo.2024.1298628
3. Bhat, S. A. (2021). *Detection of Polycystic Ovary Syndrome using Machine Learning
   Algorithms*. MSc Research Project, National College of Ireland.
4. *Polycystic Ovary Syndrome (PCOS) Disease Prediction by Using Traditional Machine
   Learning and Deep Learning Algorithms* (2024).
5. Elmannai, H., El-Rashidy, N., Mashal, I., Alohali, M. A., Farag, S., El-Sappagh, S., &
   Saleh, H. (2023). *Polycystic ovary syndrome detection machine learning model based on
   optimized feature selection and explainable artificial intelligence*. Diagnostics,
   13(8), 1506. https://doi.org/10.3390/diagnostics13081506
