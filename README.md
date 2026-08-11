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
| **Algorithms compared** | **9** — of which **0** differ practically from the best |
| **Gain from a 120s AutoML search** | **−0.005 ROC-AUC** (i.e. none) |

The last three rows are the findings we consider most important. See
[The leakage experiment](#the-leakage-experiment) and
[There is no best algorithm here](#there-is-no-best-algorithm-here).

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
| **Winners declared from tiny margins** | Paired t-tests across identical folds, reporting statistical *and* practical separability |
| **Only ranking metrics reported** | Brier score, log loss and calibration curves — whether the probabilities mean anything |
| **No check that the approach itself is sound** | AutoML benchmark under the same protocol, to test for headroom |

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

## There is no best algorithm here

Nine algorithms were compared under identical repeated cross-validation folds, then tested
with paired t-tests. The result is the most useful thing the project has to say about
model selection:

| Model | CV ROC-AUC | Δ vs best | p-value | Statistically separable? | **Practically separable?** |
|---|---|---|---|---|---|
| **Random Forest** | 0.9537 ± 0.030 | — | — | — (reference) | — |
| Stacking Ensemble | 0.9527 ± 0.031 | −0.0010 | 0.42 | no | **no** |
| Gaussian NB | 0.9504 ± 0.028 | −0.0033 | 0.31 | no | **no** |
| SVM (RBF) | 0.9482 ± 0.034 | −0.0055 | 0.023 | yes | **no** |
| CatBoost | 0.9481 ± 0.035 | −0.0056 | 0.020 | yes | **no** |
| Elastic-Net LR | 0.9474 ± 0.036 | −0.0063 | 0.082 | no | **no** |
| Logistic Regression | 0.9473 ± 0.037 | −0.0064 | 0.071 | no | **no** |
| XGBoost | 0.9449 ± 0.035 | −0.0088 | 0.003 | yes | **no** |
| KNN | 0.9415 ± 0.039 | −0.0122 | 0.0005 | yes | **no** |

**Zero of eight** models differ from the best by a practically meaningful margin. The
largest gap in the whole table is 0.012 AUC — against a fold-to-fold standard deviation
of 0.030, six times larger.

Note the two rightmost columns disagree, and that disagreement is the lesson. With 30
folds, a paired t-test flags XGBoost's 0.009 AUC deficit as "significant" (p = 0.003).
No clinical decision turns on 0.009 AUC. **Statistical separability is not the same as
mattering**, and a paper that reports the former while implying the latter is
over-claiming.

Two consequences worth stating in the report:

- **The stacking ensemble buys nothing.** Papers 4 and 5 both crown a stacking classifier.
  Here it lands 0.001 AUC below a plain Random Forest — inside the noise — while costing
  far more to train and explain.
- **Gaussian Naive Bayes is competitive.** A model with no hyperparameters and a
  famously wrong independence assumption sits third, statistically tied with the best.
  That is a strong signal the dataset's signal is simple and largely additive.

Caveat on the p-values: repeated-CV folds overlap, so the scores are not independent and
the paired t-test is known to be optimistic. Large p-values (evidence of *no* difference)
are therefore trustworthy; small ones should be read as suggestive only. That direction
happens to be the one this project needs.

### Model comparison, full table (repeated stratified 10-fold CV, training split only)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.899 ± 0.036 | 0.871 | 0.818 | 0.840 | **0.954 ± 0.029** |
| Stacking Ensemble | 0.897 ± 0.032 | 0.847 | 0.839 | 0.840 | 0.953 ± 0.031 |
| Gaussian NB | 0.867 ± 0.049 | 0.849 | 0.734 | 0.782 | 0.950 ± 0.027 |
| SVM (RBF) | 0.889 ± 0.045 | 0.825 | 0.844 | 0.831 | 0.948 ± 0.034 |
| CatBoost | 0.886 ± 0.040 | 0.837 | 0.816 | 0.823 | 0.948 ± 0.034 |
| Elastic-Net LR | 0.891 ± 0.033 | 0.817 | 0.865 | 0.837 | 0.947 ± 0.035 |
| Logistic Regression | 0.887 ± 0.041 | 0.805 | 0.870 | 0.833 | 0.947 ± 0.037 |
| XGBoost | 0.875 ± 0.043 | 0.825 | 0.794 | 0.804 | 0.945 ± 0.035 |
| KNN | 0.882 ± 0.046 | 0.822 | 0.833 | 0.822 | 0.942 ± 0.038 |

**CatBoost** is included specifically because Paper 3 reports it as its best model at
95.7% accuracy. Run under our protocol it reaches 88.6% CV accuracy — the gap is the
protocol, not the algorithm.

**KNN carries a caveat:** it is the model most damaged by SMOTE. Oversampling by
interpolating between neighbours, then classifying by neighbours, is close to circular.
Its number should be read with that in mind.

### Is the ranking stable? No.

The CV ranking and the test ranking disagree substantially — further evidence that the
ordering is noise:

| Model | CV rank | Test rank |
|---|---|---|
| Random Forest | 1 | 8 |
| Stacking | 2 | 5 |
| CatBoost | 5 | **1** |
| Logistic Regression | 7 | 2 |

Random Forest tops cross-validation and lands *last but one* on the test set. Any paper
declaring a winner from a single split is reporting which model got the lucky fold.

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

### Probability calibration — the tie-break ROC-AUC cannot see

If nine models are tied on ranking ability, pick on something else. ROC-AUC only measures
*ordering*; it is completely blind to whether a predicted 0.8 corresponds to an 80% chance
of PCOS. That distinction is invisible in every metric reported by the five reviewed
papers, and it is exactly what matters when a probability is used to set a screening
threshold.

| Model | Brier ↓ | Log loss ↓ | ECE ↓ | ROC-AUC |
|---|---|---|---|---|
| **Stacking Ensemble** | **0.0820** | 0.2762 | **0.0350** | 0.9522 |
| Random Forest | 0.0835 | 0.2744 | 0.0537 | 0.9537 |
| Elastic-Net LR | 0.0836 | 0.2858 | 0.0559 | 0.9439 |
| CatBoost | 0.0837 | 0.3044 | 0.0487 | 0.9476 |
| SVM (RBF) | 0.0840 | 0.2787 | 0.0503 | 0.9482 |
| KNN | 0.0875 | 0.5709 | 0.0459 | 0.9429 |
| Logistic Regression | 0.0880 | 0.3010 | 0.0550 | 0.9406 |
| XGBoost | 0.0949 | 0.3390 | 0.0723 | 0.9445 |
| Gaussian NB | 0.1110 | 0.5249 | 0.0988 | 0.9441 |

The ranking here is **not** the ROC-AUC ranking. Gaussian NB is statistically tied with
the best on AUC but is by far the worst-calibrated model — its probabilities are
near-useless even though its rankings are fine. XGBoost has the second-worst calibration.
This is the kind of difference that should decide between tied models, and no reviewed
paper measures it.

**A negative result worth reporting:** wrapping Random Forest in isotonic calibration made
things *worse* — Brier +0.0013, log loss +0.228, AUC −0.007. Isotonic regression is
non-parametric and needs more data than 432 rows to estimate a mapping; it overfits the
calibration folds. The lesson is that calibration is not free, and on small datasets the
standard fix can backfire.

### Does a better pipeline exist? An AutoML check

The model comparison answers "which algorithm is best". It cannot answer the more useful
question: *is this whole approach leaving performance on the table?* Every entry shares
the same pipeline, so a common handicap would be invisible.

[FLAML](https://github.com/microsoft/FLAML) (MIT licence) was given the training split and
120 seconds to search over algorithms and hyperparameters, then scored once on the same
held-out test set:

| Approach | Test accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|
| Hand-built pipeline (Random Forest) | 0.9174 | 0.8696 | 0.9418 |
| AutoML / FLAML (tuned RF) | 0.9083 | 0.8529 | 0.9372 |
| **Difference** | **−0.0092** | **−0.0166** | **−0.0046** |

An untargeted search did not beat a considered design — it landed slightly below, well
inside the ±0.03 fold noise. FLAML independently converged on a Random Forest, which is
mild corroboration of the hand-built choice.

The conclusion: **the dataset is the binding constraint, not the pipeline.** Further
hyperparameter tuning on 541 rows is wasted effort, and any paper claiming a large gain
from architecture search on this data should be read sceptically.

*Caveat: FLAML searches against a wall-clock budget, so the model it returns varies
between runs (we observed 0.935–0.943 test AUC across runs). The conclusion is stable;
the exact number is not.*

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
│   ├── calibration.py    Brier/ECE calibration + paired significance tests
│   ├── automl.py         FLAML benchmark — is there headroom?
│   ├── explain.py        SHAP global + per-patient explanations
│   ├── literature.py     The five reviewed papers, as comparable data
│   ├── plots.py          Model-behaviour figures
│   └── run_pipeline.py   End-to-end runner
├── notebooks/PCOS_Prediction.ipynb
├── reports/figures/      20 generated figures
├── reports/results/      Result tables (CSV) + summary.json
└── tests/test_pipeline.py    36 tests
```

### Optional dependencies

The pipeline degrades gracefully if these are missing — CatBoost is skipped from the model
zoo, and the AutoML section reports itself as unavailable:

| Package | Licence | Used for |
|---|---|---|
| `catboost` | Apache 2.0 | Paper 3's best model, so its claim is tested rather than quoted |
| `flaml` | MIT | AutoML headroom benchmark |

Everything else in `requirements.txt` is BSD/MIT/Apache — the whole stack is free and
open source, with no account, API key or paid tier anywhere in the pipeline.

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

### Why these nine algorithms

Two dataset properties drive the choice:

- **n/p ≈ 13** (541 rows, 41 features). This is the variance-limited regime: model
  capacity is not the bottleneck, data is. Anything more flexible than a shallow ensemble
  spends its capacity fitting noise.
- **The signal is largely additive.** Follicle counts plus a cluster of binary symptoms,
  each pushing one way. There are few interactions for a tree ensemble to exploit that a
  linear model cannot capture — which is why logistic regression keeps pace with XGBoost
  here, and why Gaussian NB is competitive despite its independence assumption.

The nine span the useful hypothesis space — linear (LR, elastic-net), kernel (SVM),
probabilistic (Gaussian NB), instance-based (KNN), bagged trees (RF), boosted trees
(XGBoost, CatBoost) and a stacked ensemble. That spread is what makes the
"they're indistinguishable" finding credible rather than an artefact of testing five
variations on one idea. Naive Bayes and KNN also serve as weak baselines, so 0.95 AUC is
measured against something other than the 67% majority-class floor.

### What was deliberately excluded

The deep learning models from Paper 4 (ANN, **CNN, RNN, LSTM, Bi-LSTM**) are not included,
and not because they would score badly. CNN, RNN and LSTM are *structurally inapplicable*
to this data: convolutions assume spatial locality, recurrence assumes sequential order,
and a row of clinical measurements has neither — column 7 is not "after" column 6. Any
apparent gain from such a model on static tabular data comes from the dense layers wrapped
around it, not from the architecture that names it.

Reporting 99.32% from an LSTM on 541 rows of static tabular data is, on its own, a reason
to distrust that paper's protocol.

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
