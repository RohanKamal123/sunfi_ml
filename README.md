# Machine Learning-Based Prediction of Polycystic Ovary Syndrome (PCOS)

**Group 5** — Fatema Ferdous (0152410052) · Wafa Haque (0152420023) · Khondoker Sazzad Sunfi (0152310002)

An end-to-end machine learning study predicting PCOS from clinical, hormonal and
physical measurements, built to address the gaps identified in our literature review of
five recent papers.

---

## Headline result

| | Value |
|---|---|
| Best model | Stacking Ensemble (SMOTE + correlation-based feature selection, 20 features) |
| Cross-validated ROC-AUC | **0.956 ± 0.028** (10-fold × 3 repeats) |
| Held-out test accuracy | **0.927** |
| Held-out test ROC-AUC | **0.945** — 95% CI **[0.886, 0.989]** |
| Test recall / specificity | 0.833 / 0.973 |

### The five findings that matter more than the accuracy

| Finding | Result |
|---|---|
| **A leaky protocol manufactures accuracy** | **+3.2 points** accuracy, **+9.0** F1 — same model, same data, same folds |
| **There is essentially no best algorithm** | 9 compared; only **1 of 8** (KNN) differs practically from the best. Across 30 random splits, **8 of 9 models won at least once**; the most frequent winner took just **27%** |
| **A free questionnaire gets you most of the way** | **0.887 AUC** with no clinician, lab or ultrasound — **93%** of the full model. The entire blood panel adds **+0.011** |
| **The model transfers, but the useful features don't exist elsewhere** | External AUC drop ≈ **−0.006** on an independent Tunisian cohort — but only 8 features are shared, and the headline model **cannot be externally validated on any public data** |
| **AutoML finds no headroom** | **+0.0004 ROC-AUC** from a 120s architecture + hyperparameter search |

Jump to: [leakage](#the-leakage-experiment) · [no best algorithm](#there-is-no-best-algorithm-here) ·
[cost tiers](#what-does-each-tier-of-testing-actually-buy) · [external validation](#external-validation)

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
| **Winners declared from tiny margins** | Paired t-tests across identical folds, reporting statistical *and* practical separability; plus a 30-split sweep showing all 9 models win sometimes |
| **Only ranking metrics reported** | Brier score, log loss and calibration curves — whether the probabilities mean anything |
| **No check that the approach itself is sound** | AutoML benchmark under the same protocol, to test for headroom |
| **No external validation** — *nobody closed this* | **Closed as far as public data allows**: an independent Tunisian cohort (n=88, CC BY 4.0), with provenance checks against re-uploads. Finding: the headline model *cannot* be externally validated on any public data |
| **No point estimates without uncertainty** | 95% bootstrap CIs on every test metric |
| **No clinical utility measure** | Decision curve analysis (net benefit) and cost-tiered deployment models |
| **No reporting standard followed** | [TRIPOD+AI checklist](reports/TRIPOD_AI_checklist.md) filled in item by item, including a PROBAST-style risk-of-bias self-assessment |

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
| **Stacking Ensemble** | 0.9557 ± 0.028 | — | — | — (reference) | — |
| Random Forest | 0.9544 ± 0.028 | −0.0012 | 0.39 | no | **no** |
| CatBoost | 0.9516 ± 0.032 | −0.0041 | 0.047 | yes | **no** |
| Gaussian NB | 0.9500 ± 0.025 | −0.0057 | 0.15 | no | **no** |
| XGBoost | 0.9485 ± 0.031 | −0.0071 | 0.002 | yes | **no** |
| Elastic-Net LR | 0.9471 ± 0.035 | −0.0086 | 0.005 | yes | **no** |
| Logistic Regression | 0.9468 ± 0.035 | −0.0089 | 0.002 | yes | **no** |
| SVM (RBF) | 0.9456 ± 0.035 | −0.0101 | 0.0001 | yes | **no** |
| KNN | 0.9268 ± 0.043 | −0.0289 | <0.0001 | yes | **yes** |

**Only one of eight** models — KNN — differs from the best by a practically meaningful
margin. Every other gap is under 0.011 AUC, against a fold-to-fold standard deviation of
0.028, roughly three times larger.

The two rightmost columns disagree for six models, and that disagreement is the lesson.
With 30 folds, a paired t-test flags SVM's 0.010 AUC deficit as highly significant
(p = 0.0001). No clinical decision turns on 0.010 AUC. **Statistical separability is not
the same as mattering**, and a paper reporting the former while implying the latter is
over-claiming.

Two observations worth putting in the report:

- **Gaussian Naive Bayes is competitive.** A model with no hyperparameters and a
  famously wrong independence assumption sits fourth, statistically tied with the best,
  and it *wins the most splits* in the stability sweep below. Strong evidence that this
  dataset's signal is simple and largely additive.
- **KNN is the one genuine laggard, and the reason is instructive.** It is the model most
  damaged by SMOTE: oversampling by interpolating between neighbours, then classifying by
  neighbours, is close to circular. Papers 1 and 4 both include KNN without noting this.

Caveat on the p-values: repeated-CV folds overlap, so the scores are not independent and
the paired t-test is known to be optimistic. Large p-values (evidence of *no* difference)
are therefore trustworthy; small ones should be read as suggestive only. That direction
happens to be the one this project needs.

### Model comparison, full table (repeated stratified 10-fold CV, training split only)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Stacking Ensemble** | 0.894 ± 0.042 | 0.845 | 0.837 | 0.837 | **0.956 ± 0.028** |
| Random Forest | 0.891 ± 0.039 | 0.853 | 0.815 | 0.829 | 0.954 ± 0.027 |
| CatBoost | 0.888 ± 0.047 | 0.835 | 0.830 | 0.829 | 0.952 ± 0.031 |
| Gaussian NB | 0.862 ± 0.047 | 0.769 | 0.846 | 0.801 | 0.950 ± 0.025 |
| XGBoost | 0.888 ± 0.047 | 0.846 | 0.813 | 0.825 | 0.949 ± 0.031 |
| Elastic-Net LR | 0.895 ± 0.035 | 0.815 | 0.884 | 0.845 | 0.947 ± 0.034 |
| Logistic Regression | 0.891 ± 0.039 | 0.806 | 0.887 | 0.842 | 0.947 ± 0.035 |
| SVM (RBF) | 0.882 ± 0.052 | 0.820 | 0.834 | 0.821 | 0.946 ± 0.034 |
| KNN | 0.865 ± 0.053 | 0.778 | 0.839 | 0.802 | 0.927 ± 0.042 |

**CatBoost** is included specifically because Paper 3 reports it as its best model at
95.7% accuracy. Run under our protocol it reaches 88.8% CV accuracy — the gap is the
protocol, not the algorithm.

### Is the ranking stable? No.

The CV ranking and the test ranking disagree substantially — further evidence that the
ordering is noise:

| Model | CV rank | Test rank |
|---|---|---|
| **Stacking Ensemble** | **1** | 6 |
| Random Forest | 2 | 7 |
| CatBoost | 3 | **1** |
| XGBoost | 5 | 2 |
| Logistic Regression | 7 | 4 |

The model that tops cross-validation lands sixth on the test set; the model that tops the
test set was third in CV. Any paper declaring a winner from a single split is reporting
which model got the lucky fold.

**A note on our own headline.** Stacking is reported as "best" because it topped the
cross-validated ranking, which is the defensible selection rule — but by the project's own
evidence that title is close to meaningless. Six of the eight alternatives are
indistinguishable from it, it placed sixth on the held-out test set, and it won only 10%
of the 30 random splits. Anyone reproducing this work should expect a different winner.

### Feature selection

| Strategy | Features kept | Mean CV ROC-AUC |
|---|---|---|
| All features | 41 | 0.9230 |
| **Correlation** | **20** | **0.9449** |
| Chi-Square | 15 | 0.9447 |
| RFE | 15 | 0.9380 |

Correlation and Chi-Square are separated by 0.0002 — noise. What *is* meaningful is that
all three selectors beat using all 41 features by 0.015–0.022 AUC: on 432 training rows,
26 extra weakly-informative features cost more in variance than they contribute in signal.
The practical argument reinforces it — 15–20 measurements are cheaper to collect in a
clinic than 41.

### The winner changes with the split

The significance tests say the models are tied. This says it more bluntly. We
re-split the data 30 times and recorded which model came first on each:

| Model | Times ranked #1 | Win rate | Own AUC range across splits |
|---|---|---|---|
| **Gaussian NB** | **8** | **27%** | 0.102 |
| Random Forest | 6 | 20% | 0.089 |
| CatBoost | 4 | 13% | 0.100 |
| Stacking Ensemble | 3 | 10% | 0.090 |
| Elastic-Net LR | 3 | 10% | 0.114 |
| XGBoost | 3 | 10% | 0.085 |
| SVM (RBF) | 2 | 7% | 0.113 |
| KNN | 1 | 3% | 0.129 |
| Logistic Regression | 0 | 0% | 0.118 |

**Eight of the nine models won at least once**, and the most frequent winner took
only 27% of the draws. Two details are worth pointing at:

- The winner is **Gaussian Naive Bayes** — a model with no hyperparameters and an
  assumption everyone knows is false — which finished *fourth* in the
  cross-validated ranking.
- Even **KNN**, the one model that is genuinely and measurably worse, still won a
  split.

Each model's own test AUC swings by 0.085–0.129 depending only on which patients
landed in the test set. A paper that reports "model X achieved the best accuracy"
from a single 80/20 split is reporting a coin flip, and this table is what that
coin flip looks like when you flip it thirty times.

### What does each tier of testing actually buy?

Your introduction argues PCOS testing "may not always be easy to access,
affordable, or quick, especially in areas with limited healthcare resources".
This turns that argument into numbers. Features are grouped by what they cost to
obtain, and a model is trained on each cumulative tier:

| Tier | What it needs | Features | CV ROC-AUC | Marginal gain |
|---|---|---|---|---|
| **Questionnaire** | A form, a scale, a tape measure | 19 | **0.887** | — |
| + clinic vitals | A nurse, 5 minutes | 25 | 0.880 | −0.007 |
| + blood panel | Venous draw, endocrine assays | 36 | 0.890 | **+0.011** |
| + ultrasound | Sonographer, machine | 41 | **0.952** | **+0.062** |

Three conclusions, and the middle one is the surprise:

1. **A questionnaire alone reaches 0.887 AUC** — 93% of the full model's
   performance, with no clinician, no laboratory and no ultrasound. On the test
   set it scores 0.897 AUC at 83.5% accuracy. That is a screening tool
   deployable by a health worker with a clipboard.
2. **The entire endocrine panel adds +0.011 AUC.** FSH, LH, AMH, TSH, prolactin,
   vitamin D, progesterone, blood sugar and beta-HCG — nine assays, a venous
   draw, and a laboratory — together buy roughly one hundredth of an AUC point.
   For a screening application, the bloodwork is close to worthless.
3. **Only the ultrasound pays for itself** (+0.062).

So the honest recommendation for a low-resource deployment is: **run the
questionnaire, skip the bloods, and spend the budget on ultrasound access for
the women the questionnaire flags.** None of the five reviewed papers ask this
question, and none of them could answer it, because they all optimise a single
model on all 41 features at once.

### Probability calibration — the tie-break ROC-AUC cannot see

If nine models are tied on ranking ability, pick on something else. ROC-AUC only measures
*ordering*; it is completely blind to whether a predicted 0.8 corresponds to an 80% chance
of PCOS. That distinction is invisible in every metric reported by the five reviewed
papers, and it is exactly what matters when a probability is used to set a screening
threshold.

| Model | Brier ↓ | Log loss ↓ | ECE ↓ | ROC-AUC |
|---|---|---|---|---|
| **Stacking Ensemble** | **0.0808** | **0.2724** | **0.0328** | 0.9496 |
| Elastic-Net LR | 0.0817 | 0.2819 | 0.0482 | 0.9454 |
| Random Forest | 0.0827 | 0.2758 | 0.0605 | **0.9559** |
| CatBoost | 0.0838 | 0.2985 | 0.0477 | 0.9484 |
| Logistic Regression | 0.0839 | 0.2846 | 0.0561 | 0.9450 |
| SVM (RBF) | 0.0839 | 0.2989 | 0.0343 | 0.9425 |
| XGBoost | 0.0932 | 0.3313 | 0.0649 | 0.9453 |
| KNN | 0.0983 | 0.7656 | 0.0718 | 0.9284 |
| **Gaussian NB** | **0.1090** | 0.4406 | **0.1003** | 0.9510 |

The ranking here is **not** the ROC-AUC ranking, and the contrast at the two ends makes
the point:

- **Gaussian NB** has the *fourth-best* AUC and the *worst* calibration by a wide margin
  (Brier 0.109, ECE 0.100). It ranks patients well and its probabilities mean almost
  nothing. This is the same model that won the most splits in the sweep above — a model
  you would happily pick on AUC alone, and should not deploy on a probability threshold.
- **Random Forest** has the *best* AUC and only middling calibration (ECE 0.061).
- **SVM** has the *worst* AUC of the non-KNN models and the second-best ECE.

Discrimination and calibration are close to uncorrelated here. No reviewed paper reports
the second one.

**A negative result worth reporting:** wrapping the selected model in isotonic calibration
made things *worse* — see `reports/results/calibration_effect.csv`. Isotonic regression is
non-parametric and needs more data than 432 rows to estimate a stable mapping, so it
overfits the calibration folds. Calibration is not free, and on small datasets the standard
fix can backfire.

### External validation

Every reviewed paper was criticised in our gap analysis for never testing on a
second population. We closed that gap as far as public data allows — and what we
found reframes the criticism.

**The cohort.** A case-control study from Sfax, Tunisia
([Mendeley, doi:10.17632/tw34c7hv7z.1](https://data.mendeley.com/datasets/tw34c7hv7z/1),
CC BY 4.0). 88 women, same Rotterdam criteria. Genuinely independent, and the
differences are the point:

| | Kerala (train) | Tunisia (external) |
|---|---|---|
| n | 541 | 88 |
| Country | India | Tunisia |
| Class ratio | 2.06 : 1 | 1.05 : 1 |
| Mean age | 31.4 | 25.5 |
| Mean BMI | 24.3 | 27.1 |

Before using it we checked it was not a re-upload of the training data — the
single biggest hazard here, since many public "PCOS datasets" are copies of the
same Kerala file. Provenance checks for every candidate are recorded in
`reports/results/external_search_log.txt`, and enforced by
`tests/test_pipeline.py::test_external_cohort_is_not_a_reupload`.

**Three findings:**

1. **The restricted model transfers cleanly.** Mean AUC change from Kerala CV to
   Tunisian patients is **−0.006** across nine algorithms, despite a different
   country, a 6-year-younger cohort and a near-inverted class balance. What the
   model learned is not Kerala-specific.
2. **But the restricted model is weak** — both sides sit near 0.65 AUC. Only 8
   features are shared, and per SHAP, essentially all the signal lives in
   follicle counts and symptoms. The Tunisian study recorded neither. **The
   shared features are the weak ones.**
3. **Therefore the headline model cannot be externally validated at all** — not
   by us, and not by anyone, on currently public data. No public cohort records
   follicle counts and PCOS symptoms in a compatible schema.

That third point is the useful contribution. The five papers did not merely
neglect external validation; **they had no dataset with which to perform it.**
That is a stronger criticism, because it points at what the field needs —
compatible multi-centre data collection — rather than at what five authors
failed to do.

**A fourth observation, on prevalence shift.** Logistic regression achieved 0.96
recall but 0.21 specificity externally: it labelled almost everyone PCOS.
Discrimination survived the move; the *threshold* did not. The 0.5 cut-off was
implicitly tuned for Kerala's 33% prevalence, and Tunisia runs at 51%. A model
deployed in a new population needs its threshold re-set for that population's
base rate — and AUC will never reveal the problem. This is the practical argument
for the calibration analysis above.

### Does a better pipeline exist? An AutoML check

The model comparison answers "which algorithm is best". It cannot answer the more useful
question: *is this whole approach leaving performance on the table?* Every entry shares
the same pipeline, so a common handicap would be invisible.

[FLAML](https://github.com/microsoft/FLAML) (MIT licence) was given the training split and
120 seconds to search over algorithms and hyperparameters, then scored once on the same
held-out test set:

| Approach | Test accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|
| Hand-built pipeline (Stacking Ensemble) | 0.9266 | 0.8824 | 0.9452 |
| AutoML / FLAML (tuned RF) | 0.8991 | 0.8308 | 0.9456 |
| **Difference** | **−0.0275** | **−0.0516** | **+0.0004** |

A 120-second search over algorithms *and* hyperparameters moved test ROC-AUC by
**four ten-thousandths** — and left accuracy and F1 lower. FLAML converged on a Random
Forest, mild corroboration of the hand-built choice.

The conclusion: **the dataset is the binding constraint, not the pipeline.** Further
hyperparameter tuning on 541 rows is wasted effort, and any paper claiming a large gain
from architecture search on this data should be read sceptically.

*Caveat: FLAML searches against a wall-clock budget, so the model it returns varies
between runs (we observed 0.935–0.946 test AUC across runs). The conclusion is stable;
the exact number is not.*

### Applying our own standards to ourselves

The project criticises other papers for leakage. Two checks apply the same
scrutiny inward.

**Nested cross-validation.** The pipeline picks a feature set using CV on the
training split, then reports a CV score on that same split — selection and
evaluation share data, so the number is optimistic in principle. Nested CV (inner
loop selects, outer loop scores) measures the bias:

| Protocol | ROC-AUC |
|---|---|
| Flat CV (selector chosen on the same data) | 0.9520 |
| Nested CV (selector chosen inside folds) | 0.9523 ± 0.027 |
| **Selection bias** | **−0.0002** |

Negligible — because the four feature sets perform almost identically, so
choosing between them leaks almost nothing. We flagged a real concern, measured
it, and found it immaterial *here*. Reporting the measurement rather than the
worry is the point; the same check on a study that tuned 50 hyperparameters would
not come out this way.

**Confidence intervals.** 109 test patients, 36 of them positive. Point estimates
alone are misleading, so every test metric gets a 95% percentile bootstrap:

| Metric | Estimate | 95% CI | Width |
|---|---|---|---|
| Accuracy | 0.927 | [0.872, 0.973] | 0.101 |
| Precision | 0.938 | [0.846, 1.000] | 0.154 |
| **Recall** | **0.833** | **[0.703, 0.943]** | **0.240** |
| F1 | 0.882 | [0.793, 0.955] | 0.162 |
| Specificity | 0.973 | [0.930, 1.000] | 0.070 |
| ROC-AUC | 0.945 | [0.886, 0.989] | 0.103 |

Recall — the metric that matters most for a screening tool — is pinned down only
to within ±12 points. That is the honest precision of *any* single-split number
on a dataset this size, including the 98–99% figures in the literature, none of
which report an interval.

### Is it clinically worth using? Decision curve analysis

Accuracy says whether the model is right. **Net benefit** (Vickers & Elkin, 2006)
says whether *using* it beats the two things a clinic could do instead — refer
everyone for confirmatory testing, or refer nobody. It is the standard analysis
in the clinical prediction literature and absent from all five reviewed papers.

The model beats both defaults across essentially the whole plausible threshold
range (5%–80%). At a 20% threshold — "a missed case is 4× worse than an
unnecessary follow-up", a reasonable stance for screening — it flags **40 of 109**
patients and delivers net benefit **0.264** against **0.163** for referring everyone.

Translated: at that threshold the model finds effectively the same cases while
sending **63% fewer women** for unnecessary confirmatory testing than a
refer-everyone policy would. On a scarce-ultrasound assumption, that is the number
that matters — and it is not derivable from any metric the five papers report.

### Does it work for everyone?

Average metrics hide subgroup failures, so performance is broken out by age band
and BMI category. Across subgroups with at least 15 patients, AUC ranges
**0.918–0.990** — no subgroup failure detected. The caveat is honest: the largest
subgroup has 53 patients, so only a large failure would be visible.

### Would more data help?

Learning curves separate "the model is too simple" from "the dataset is too
small". Over the last third of the curve the validation score moves
**+0.0003 AUC per 100 additional patients** — flat. Training score sits at 0.9999
against a validation score of 0.956, the classic signature of a model that has
saturated what these features can tell it.

Combined with the AutoML result (no headroom from better architecture), this
locates the ceiling: **neither more rows nor a better model would move this much.**
What would move it is better features — which is precisely what the external
validation says no public dataset currently offers.

### Class imbalance ablation (test set)

| Strategy | Accuracy | Recall | F1 | Specificity |
|---|---|---|---|---|
| No balancing | 0.917 | 0.833 | 0.870 | 0.959 |
| Class weighting | 0.908 | 0.861 | 0.861 | 0.932 |
| **SMOTE** | **0.927** | **0.861** | **0.886** | 0.959 |

SMOTE buys recall — it catches PCOS cases the unbalanced model misses — at no cost in
specificity. For a screening tool that trade is worth taking.

### What the model relies on (SHAP)

| Rank | Feature | Mean \|SHAP\| | Higher value means |
|---|---|---|---|
| 1 | Follicle count (right) | 0.128 | more likely PCOS |
| 2 | Follicle count (left) | 0.093 | more likely PCOS |
| 3 | Weight gain | 0.066 | more likely PCOS |
| 4 | Skin darkening | 0.059 | more likely PCOS |
| 5 | Hair growth | 0.057 | more likely PCOS |
| 6 | Irregular cycle | 0.038 | more likely PCOS |

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

# Full study: ~13 minutes, writes all 29 figures and 29 result tables
python -m src.run_pipeline

# Fast sanity check: fewer folds and seeds, skips SHAP (~2 minutes)
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
│   ├── validation.py     Nested CV, seed sweep, learning curves, bootstrap CIs
│   ├── clinical.py       Decision curves, cost tiers, subgroup performance
│   ├── external.py       Independent Tunisian cohort + provenance log
│   ├── automl.py         FLAML benchmark — is there headroom?
│   ├── explain.py        SHAP global + per-patient explanations
│   ├── literature.py     The five reviewed papers, as comparable data
│   ├── plots.py          Model-behaviour figures
│   └── run_pipeline.py   End-to-end runner
├── data/raw/             Kerala cohort (541 patients)
├── data/external/        Tunisian cohort (88 patients, CC BY 4.0)
├── notebooks/PCOS_Prediction.ipynb
├── reports/figures/      29 generated figures
├── reports/results/      29 result tables (CSV/JSON) + summary.json
├── reports/TRIPOD_AI_checklist.md
└── tests/test_pipeline.py    52 tests
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
6. **Physiologically impossible values** — 13 of them, listed below.

Item 6 was only discovered by comparing feature distributions against the
external cohort, which is a small argument for external data beyond validation:

| Row | Feature | Value | Plausible |
|---|---|---|---|
| 161 | BP systolic | 12 | 70–250 |
| 200 | BP diastolic | 8 | 40–150 |
| 223, 296 | Pulse rate | 18, 13 bpm | 35–200 |
| 329 | FSH | 5052 | 0.1–200 |
| 455 | LH | 2018 | 0.01–200 |
| 191, 195 | Vitamin D3 | 6014, 5419 | 1–150 |
| 267 | AMH | 66 | 0.01–50 |
| 216 | Vitamin D3 | 0 | 1–150 |
| 39, 82 | Endometrium | 0 mm | 1–30 |
| 39 | Cycle length | 0 days | 1–60 |

The blood pressures are obvious digit-drops from 120/80, and three hormone values
are off by three orders of magnitude. They are **blanked, not corrected** —
inferring the intended value would be guessing, and one imputed median is more
honest than a plausible-looking invention. All four of the papers using this
dataset presumably trained on these values as-is.

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

1. **External validation is partial, and structurally limited.** We validated on an
   independent Tunisian cohort, but only 8 features are shared, so what transferred was a
   weak restricted model (≈0.65 AUC), not the headline one. The headline model cannot be
   externally validated on any currently public data — see
   [External validation](#external-validation).
2. **Circularity between predictor and label.** Follicle count is our strongest SHAP
   feature *and* one of the three Rotterdam criteria used to assign the label. The full
   model is therefore best read as **automating consistent application of the diagnostic
   criteria**, not as discovering new biology. The questionnaire-only model is the more
   scientifically interesting one, because none of its inputs are diagnostic criteria.
3. **Unblinded outcome assessment.** Predictors were recorded by the clinicians who also
   made the diagnosis. Several strong features (weight gain, hair growth) are
   patient-reported and may be elicited *because* PCOS is already suspected — making them
   partly a consequence of the diagnosis rather than a predictor of it.
4. **Small sample.** 109 test rows, 36 of them positive. Recall's 95% CI spans
   [0.70, 0.94]. Cross-validated figures are the more trustworthy ones.
5. **Undocumented provenance.** The Kaggle dataset publishes no collection period, no
   eligibility criteria and no treatment history. Hospital-attending women are also not a
   general population — prevalence here is 33% against a population estimate of 8–13%.
6. **No ultrasound imagery.** Also in our gap analysis, also not addressed. The dataset
   has follicle *counts* derived from ultrasound, but not the images.
7. **Ambiguous column semantics.** `Cycle length(days)` has a median of 5 and a range of
   2–12, so it is almost certainly bleeding duration rather than the ~28-day cycle
   interval the name suggests. The Kaggle documentation does not clarify this.
8. **Not a diagnostic tool.** This is coursework. PCOS diagnosis requires the Rotterdam
   criteria applied by a clinician.

A structured risk-of-bias assessment is in
[`reports/TRIPOD_AI_checklist.md`](reports/TRIPOD_AI_checklist.md). Its summary: the
*analysis* is low-risk and defensible in detail; the *data* is medium-to-high risk for
reasons outside our control, and no amount of modelling rigour fixes a label recorded by
an unblinded assessor.

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
