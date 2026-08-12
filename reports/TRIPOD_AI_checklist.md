# TRIPOD+AI reporting checklist

**Study:** Machine Learning-Based Prediction of Polycystic Ovary Syndrome (PCOS)
**Group 5** — Fatema Ferdous (0152410052) · Wafa Haque (0152420023) · Khondoker Sazzad Sunfi (0152310002)

TRIPOD+AI (Collins et al., *BMJ* 2024) is the reporting standard for clinical
prediction models that use machine learning. It exists because prediction-model
papers routinely omit the details a reader needs to judge whether a reported
accuracy is trustworthy — which is exactly the problem our literature review
identified in four of the five papers we examined.

Applying the checklist to ourselves is the point. Items we **cannot** satisfy are
marked ⚠️ and explained rather than skipped.

Legend: ✅ reported · ⚠️ partially reported or not applicable, with reason · ❌ not done

---

## Title and abstract

| # | Item | Status | Where / note |
|---|---|---|---|
| 1 | Identify as a study developing a prediction model, state target population and outcome | ✅ | `README.md` title and headline; notebook §0 |
| 2 | Structured summary: objective, data source, participants, predictors, outcome, metrics, conclusions | ✅ | `README.md` "Headline result" + "Results" |

## Introduction

| # | Item | Status | Where / note |
|---|---|---|---|
| 3a | Background: rationale, existing models, why another is needed | ✅ | Project proposal §Introduction; `src/literature.py` encodes all five reviewed models |
| 3b | Objectives, including intended use and intended users | ✅ | `README.md` "What this project does differently". Intended use is **screening triage**, not diagnosis; intended users are clinicians in settings with limited access to ultrasound |

## Methods — data

| # | Item | Status | Where / note |
|---|---|---|---|
| 4a | Data source and rationale | ✅ | `README.md` "Dataset". Kaggle PCOS (Kottarathil), 10 hospitals, Kerala, India. Chosen because 4 of the 5 reviewed papers use it, making comparison direct |
| 4b | Study dates | ⚠️ | **Not documented by the data provider.** The Kaggle dataset states no collection period. This is a real gap we cannot close and it limits any claim about temporal validity |
| 5a | Eligibility criteria | ⚠️ | Not documented upstream. Described as women of reproductive age attending the contributing hospitals; no inclusion/exclusion criteria published |
| 5b | Setting and locations | ✅ | 10 hospitals, Kerala, India (internal); Sfax, Tunisia (external validation cohort) |
| 5c | Treatments received | ❌ | Not recorded in the dataset. Relevant because hormonal contraception alters several predictors |
| 6a | Outcome definition | ✅ | Binary PCOS diagnosis by Rotterdam criteria. Same criteria in both cohorts — verified before external validation |
| 6b | Whether outcome assessors were blinded to predictors | ❌ | **Not blinded, and this matters.** The label is a clinical diagnosis made by the same clinicians who recorded the predictors. See "Risk of bias" below |
| 7a | Predictors: definition, timing, measurement | ✅ | `src/config.py` `DISPLAY_NAMES`; `reports/results/data_quality_report.csv` |
| 7b | Whether predictor assessors were blinded to the outcome | ❌ | Not blinded — same limitation as 6b |
| 8 | Sample size justification | ⚠️ | **No a priori calculation** — the dataset size was fixed at 541 by availability. We report the consequences instead: bootstrap CIs (`reports/results/bootstrap_ci.csv`), a 30-seed split sweep, and learning curves showing whether more data would help |
| 9 | Missing data handling | ✅ | Median/mode imputation **fitted inside each CV fold** (`src/models.py::build_preprocessor`). 18 values missing or blanked from 22,181 cells |
| 10 | Data preprocessing, including how it avoided leakage | ✅ | `src/models.py` module docstring; asserted by `tests/test_pipeline.py::test_preprocessor_is_not_fitted_on_validation_folds` |

## Methods — model development

| # | Item | Status | Where / note |
|---|---|---|---|
| 11 | Model type and rationale | ✅ | `README.md` "Why these nine algorithms" — nine models spanning linear, kernel, probabilistic, instance-based, bagged, boosted and stacked |
| 12 | Predictor selection method | ✅ | Correlation / Chi-Square / RFE, each a transformer refitted per fold (`src/features.py`) |
| 13 | Class imbalance handling | ✅ | SMOTE inside CV folds, with a 3-way ablation against no-balancing and class weighting (`reports/results/imbalance_ablation.csv`) |
| 14 | Hyperparameter tuning and how it was validated | ✅ | Fixed sensible defaults, plus a nested-CV measurement of the selection bias this could introduce (`reports/results/nested_cv.json`) and a FLAML AutoML search testing for headroom |
| 15 | Model evaluation: internal validation method | ✅ | Repeated stratified 10-fold CV (3 repeats) + an untouched held-out test split |
| 16 | Performance measures and rationale | ✅ | Discrimination (ROC-AUC, PR-AUC), classification (accuracy, precision, recall, F1, specificity), **calibration** (Brier, log loss, ECE) and **clinical utility** (decision curve net benefit) |
| 17 | Model updating / recalibration | ⚠️ | Isotonic recalibration was tested and **made performance worse** on 432 rows; reported as a negative result rather than dropped (`reports/results/calibration_effect.csv`) |

## Methods — AI-specific

| # | Item | Status | Where / note |
|---|---|---|---|
| A1 | Software, packages and versions | ✅ | `requirements.txt`, all open-source (BSD/MIT/Apache) |
| A2 | Reproducibility: seeds, code availability | ✅ | Single seed in `src/config.py`; full source in this repository; 36 automated tests |
| A3 | Computational resources | ✅ | Runs on a laptop CPU in ~7 minutes; no GPU |
| A4 | Fairness / subgroup analysis | ✅ | Performance by age band and BMI category (`reports/results/subgroup_performance.csv`). No subgroup failure detected, but subgroup sizes are small |
| A5 | Explainability | ✅ | SHAP global, directional and per-patient (`src/explain.py`) |

## Results

| # | Item | Status | Where / note |
|---|---|---|---|
| 18 | Participant flow | ✅ | 541 → 541 (no exclusions); 432 train / 109 test |
| 19 | Participant characteristics | ✅ | `reports/results/summary_statistics.csv` |
| 20 | Comparison of development and validation data | ✅ | `reports/results/external_cohort_comparison.csv` — standardised differences between the Kerala and Tunisian cohorts |
| 21 | Number of participants and outcome events | ✅ | 177 PCOS / 364 non-PCOS (2.06:1); 36 events in the test set |
| 22 | Full model specification | ✅ | Serialised pipeline in `models/best_model.joblib`; selected features in `reports/results/selected_features.json` |
| 23 | Performance with confidence intervals | ✅ | `reports/results/bootstrap_ci.csv` — 95% percentile bootstrap on all test metrics |
| 24 | Model performance in subgroups | ✅ | See A4 |

## Discussion

| # | Item | Status | Where / note |
|---|---|---|---|
| 25 | Limitations | ✅ | `README.md` "Limitations" — six stated, including the ones we could not fix |
| 26 | Interpretation vs prior evidence | ✅ | `src/literature.py::DISCUSSION` and the leakage experiment |
| 27 | Generalisability | ✅ | External validation on the Tunisian cohort, with an explicit statement of what it can and cannot establish |
| 28 | Implications for practice | ✅ | Cost-tiered screening analysis: a questionnaire-only model is deployable where ultrasound is not |

## Other

| # | Item | Status | Where / note |
|---|---|---|---|
| 29 | Funding | ✅ | None. Undergraduate coursework |
| 30 | Conflicts of interest | ✅ | None declared |
| 31 | Data and code availability | ✅ | Code in this repository. Internal data CC-licensed on Kaggle; external cohort CC BY 4.0 on Mendeley (doi:10.17632/tw34c7hv7z.1) |
| 32 | Protocol / registration | ❌ | Not pre-registered. Coursework project; analysis plan evolved during development, which is itself a source of researcher degrees of freedom |

---

## Risk of bias — self-assessment (PROBAST-style)

Applying the checklist honestly means recording where the study is weak, not
just where it is strong.

| Domain | Risk | Reasoning |
|---|---|---|
| **Participants** | ⚠️ Unclear | No published eligibility criteria or recruitment period from the data provider. Hospital-attending women are not a general population, so prevalence here (33%) far exceeds the 8–13% population estimate |
| **Predictors** | ⚠️ High | Predictors were recorded by clinicians who also made the diagnosis, and were not blinded to it. Several strong features are self-reported symptoms (weight gain, hair growth) that may be elicited *because* PCOS is already suspected — making them partly a consequence of the diagnosis rather than a predictor of it |
| **Outcome** | ⚠️ Unclear | Rotterdam criteria are stated but per-patient application is not auditable. Follicle count is both our strongest predictor **and** one leg of the diagnostic criteria, so some circularity is unavoidable |
| **Analysis** | ✅ Low | This is the domain the project was built to get right: leakage-safe pipeline, repeated CV, untouched test set, imbalance handled inside folds, nested CV to measure selection bias, CIs reported, no selective outcome reporting |

**The honest summary:** our *analysis* is low-risk and we can defend it in detail.
The *data* is medium-to-high risk for reasons entirely outside our control, and no
amount of modelling rigour fixes a label recorded by an unblinded assessor. A model
this good on data this uncertain should be read as a promising screening triage
tool that needs prospective validation — not as a diagnostic instrument.

---

## The circularity problem, stated plainly

Follicle count is our top SHAP feature by a wide margin. It is *also* one of the
three Rotterdam criteria used to assign the label. So the model is partly learning
"patients with many follicles are diagnosed with PCOS", which is true by
definition rather than by discovery.

This does not invalidate the work, but it changes what the result means:

- The **full model** (AUC 0.95) is best read as *automating consistent application
  of the Rotterdam criteria*, not as discovering novel biology.
- The **questionnaire-only model** (AUC 0.89) is the more scientifically interesting
  one, because none of its inputs are diagnostic criteria. It predicts the
  diagnosis from things a patient can report about herself — which is both a
  genuine prediction task and the deployable one.

None of the five reviewed papers raise this issue.

**We have now measured it rather than only asserting it.** Applying the Rotterdam
follicle criterion as a bare rule — threshold 12, zero learned parameters — reaches
0.903 cross-validated ROC-AUC, which is 94.8% of the best model's 0.953. A depth-2
decision tree given all 41 features independently rediscovers the same rule.

That is the circularity, quantified: most of what the model "learns" is the
diagnostic definition it is being scored against.

What survives the deduction is still worth something, and it is specific:
the model converts that same signal into materially better sensitivity
(recall 0.833 vs 0.611 — 30 of 36 test cases found rather than 22, for one extra
false alarm). A screening instrument is judged on exactly that, so the
contribution is real; it is just narrower than an AUC of 0.95 makes it sound.

---

## Reference

Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for
reporting clinical prediction models that use regression or machine learning
methods. *BMJ* 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
