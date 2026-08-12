"""Clinical decision rules, as the baseline machine learning has to beat.

Every reviewed paper compares its models only against other models. None asks
the prior question: **does any of this beat the rule a clinician already
uses?**

That question matters here more than usual. SHAP puts follicle count far ahead
of every other feature (mean |SHAP| roughly twice the next), its Cohen's *d* is
about 1.7, and the Rotterdam criteria already specify a threshold -- 12 or more
follicles on either ovary. If "count the follicles" reaches 0.90 AUC on its
own, then a nine-algorithm comparison, SMOTE, three feature selectors and a
stacking ensemble are buying very little, and saying so is more useful than
another decimal place.

The rules implemented here span the range from "no learning at all" to "the
simplest thing that learns":

``RotterdamFollicleRule``
    The published clinical criterion, threshold fixed at 12. Nothing is
    estimated from the data at all.
``TunedThresholdRule``
    The same shape of rule, but with the cut-off chosen on the training fold.
    One parameter learned.
``SymptomCountRule``
    Count the positive symptoms, flag at k or more. Requires no equipment.
``SingleFeatureLogistic``
    Logistic regression on one feature. The smallest real model.
``ShallowTreeRule``
    A depth-2 decision tree -- still small enough to print on a card.

All are scikit-learn classifiers, so they run through exactly the same
cross-validation, held-out test and bootstrap machinery as the nine models,
with no special-casing.

A note on ROC-AUC for hard rules. A fixed threshold produces one point on the
ROC curve, so its AUC is not comparable to a probabilistic model's. Each rule
therefore exposes ``predict_proba`` built from the *continuous* quantity
underneath it (the follicle count, the symptom tally), which is the honest
reading: accuracy measures the specific cut-off, AUC measures the ranking power
of the quantity the rule is built on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config

FOLLICLE_L = "Follicle No. (L)"
FOLLICLE_R = "Follicle No. (R)"

# The Rotterdam consensus (2003) polycystic-ovary morphology criterion.
ROTTERDAM_FOLLICLE_THRESHOLD = 12

SYMPTOM_COLUMNS = [
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Cycle_Irregular",
]


def _as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _minmax(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Squash a score into [0, 1] for ``predict_proba``, preserving order."""
    if high <= low:
        return np.full_like(values, 0.5, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


class _RuleBase(ClassifierMixin, BaseEstimator):
    """Shared plumbing: class labels, NaN handling, probability shaping.

    ``ClassifierMixin`` must precede ``BaseEstimator`` in the bases. scikit-learn
    resolves estimator tags through the MRO, so with the order reversed
    ``BaseEstimator.__sklearn_tags__`` wins, ``is_classifier()`` returns False,
    and scorers such as ``roc_auc`` silently take a regression path -- which
    surfaces as an unrelated "y should be a 1d array" error, or worse, as a
    NaN score that ``cross_val_score`` swallows by default.
    """

    def _fit_common(self, X, y):
        X = _as_frame(X)
        self.classes_ = np.array([0, 1])
        # Rules must cope with the handful of missing values on their own,
        # since they are not wrapped in the imputing pipeline.
        self.medians_ = X.median(numeric_only=True)
        self.n_features_in_ = X.shape[1]
        return X

    def _clean(self, X) -> pd.DataFrame:
        return _as_frame(X).fillna(self.medians_)

    def _proba(self, score: np.ndarray) -> np.ndarray:
        p = _minmax(np.asarray(score, dtype=float), self.score_low_, self.score_high_)
        return np.column_stack([1 - p, p])


class RotterdamFollicleRule(_RuleBase):
    """Flag PCOS when either ovary shows >= 12 follicles.

    This is the published clinical criterion applied verbatim. Nothing is
    estimated from the training data except the range used to scale
    ``predict_proba`` and the medians used to fill missing values, neither of
    which affects the hard predictions.
    """

    def __init__(self, threshold: int = ROTTERDAM_FOLLICLE_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = self._fit_common(X, y)
        score = self._score_frame(X)
        self.score_low_, self.score_high_ = float(score.min()), float(score.max())
        return self

    def _score_frame(self, X) -> np.ndarray:
        return X[[FOLLICLE_L, FOLLICLE_R]].max(axis=1).to_numpy(dtype=float)

    def predict(self, X):
        X = self._clean(X)
        return (self._score_frame(X) >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self._proba(self._score_frame(self._clean(X)))


class TunedThresholdRule(_RuleBase):
    """The same rule shape, with the cut-off learned from the training fold.

    One parameter. It answers whether the Rotterdam threshold of 12 is right
    *for this dataset*, or whether a different cut-off does better -- without
    giving the rule any additional information to work with.
    """

    def __init__(self, feature: str | None = None, criterion: str = "f1"):
        self.feature = feature
        self.criterion = criterion

    def fit(self, X, y):
        from sklearn.metrics import f1_score

        X = self._fit_common(X, y)
        X = X.fillna(self.medians_)
        y = np.asarray(y)

        score = (
            X[self.feature].to_numpy(dtype=float)
            if self.feature
            else X[[FOLLICLE_L, FOLLICLE_R]].max(axis=1).to_numpy(dtype=float)
        )
        self.score_low_, self.score_high_ = float(score.min()), float(score.max())

        candidates = np.unique(score)
        best_value, best_threshold = -np.inf, candidates[0]
        for threshold in candidates:
            pred = (score >= threshold).astype(int)
            if self.criterion == "f1":
                value = f1_score(y, pred, zero_division=0)
            else:  # Youden's J: sensitivity + specificity - 1
                tp = ((pred == 1) & (y == 1)).sum()
                fn = ((pred == 0) & (y == 1)).sum()
                tn = ((pred == 0) & (y == 0)).sum()
                fp = ((pred == 1) & (y == 0)).sum()
                sens = tp / (tp + fn) if (tp + fn) else 0.0
                spec = tn / (tn + fp) if (tn + fp) else 0.0
                value = sens + spec - 1
            if value > best_value:
                best_value, best_threshold = value, threshold

        self.threshold_ = float(best_threshold)
        return self

    def _score_frame(self, X) -> np.ndarray:
        return (
            X[self.feature].to_numpy(dtype=float)
            if self.feature
            else X[[FOLLICLE_L, FOLLICLE_R]].max(axis=1).to_numpy(dtype=float)
        )

    def predict(self, X):
        return (self._score_frame(self._clean(X)) >= self.threshold_).astype(int)

    def predict_proba(self, X):
        return self._proba(self._score_frame(self._clean(X)))


class SymptomCountRule(_RuleBase):
    """Count positive symptoms; flag at ``min_symptoms`` or more.

    Needs no clinician, no laboratory and no ultrasound -- a patient could
    self-administer it. Included because the cost-tier analysis suggests the
    symptom cluster carries most of the questionnaire tier's signal.
    """

    def __init__(self, min_symptoms: int | None = None, criterion: str = "f1"):
        self.min_symptoms = min_symptoms
        self.criterion = criterion

    def fit(self, X, y):
        from sklearn.metrics import f1_score

        X = self._fit_common(X, y)
        X = X.fillna(self.medians_)
        self.symptoms_ = [c for c in SYMPTOM_COLUMNS if c in X.columns]

        score = X[self.symptoms_].sum(axis=1).to_numpy(dtype=float)
        self.score_low_, self.score_high_ = 0.0, float(len(self.symptoms_))

        if self.min_symptoms is not None:
            self.min_symptoms_ = self.min_symptoms
            return self

        y = np.asarray(y)
        best_value, best_k = -np.inf, 1
        for k in range(1, len(self.symptoms_) + 1):
            value = f1_score(y, (score >= k).astype(int), zero_division=0)
            if value > best_value:
                best_value, best_k = value, k
        self.min_symptoms_ = best_k
        return self

    def _score_frame(self, X) -> np.ndarray:
        return X[self.symptoms_].sum(axis=1).to_numpy(dtype=float)

    def predict(self, X):
        return (self._score_frame(self._clean(X)) >= self.min_symptoms_).astype(int)

    def predict_proba(self, X):
        return self._proba(self._score_frame(self._clean(X)))


class SingleFeatureLogistic(_RuleBase):
    """Logistic regression on exactly one feature: the smallest real model."""

    def __init__(self, feature: str = FOLLICLE_R):
        self.feature = feature

    def fit(self, X, y):
        X = self._fit_common(X, y).fillna(self.medians_)
        self.model_ = LogisticRegression(max_iter=5000).fit(
            X[[self.feature]].to_numpy(dtype=float), np.asarray(y)
        )
        return self

    def predict(self, X):
        X = self._clean(X)
        return self.model_.predict(X[[self.feature]].to_numpy(dtype=float))

    def predict_proba(self, X):
        X = self._clean(X)
        return self.model_.predict_proba(X[[self.feature]].to_numpy(dtype=float))


class ShallowTreeRule(_RuleBase):
    """A depth-limited decision tree — small enough to print on a card.

    The bridge between a hand-written rule and a real model: it learns both
    which features to use and where to cut, but stays legible.
    """

    def __init__(self, max_depth: int = 2, random_state: int = config.RANDOM_STATE):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        X = self._fit_common(X, y).fillna(self.medians_)
        self.columns_ = list(X.columns)
        self.model_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=10,
            random_state=self.random_state,
        ).fit(X.to_numpy(dtype=float), np.asarray(y))
        return self

    def predict(self, X):
        return self.model_.predict(self._clean(X)[self.columns_].to_numpy(dtype=float))

    def predict_proba(self, X):
        return self.model_.predict_proba(
            self._clean(X)[self.columns_].to_numpy(dtype=float)
        )

    def as_text(self) -> str:
        """The learned rule, written out."""
        return export_text(
            self.model_,
            feature_names=[config.pretty(c) for c in self.columns_],
            decimals=1,
        )


class MajorityClassRule(_RuleBase):
    """Always predict the majority class. The absolute floor."""

    def fit(self, X, y):
        X = self._fit_common(X, y)
        y = np.asarray(y)
        self.majority_ = int(np.bincount(y).argmax())
        self.rate_ = float(y.mean())
        self.score_low_, self.score_high_ = 0.0, 1.0
        return self

    def predict(self, X):
        return np.full(len(_as_frame(X)), self.majority_, dtype=int)

    def predict_proba(self, X):
        n = len(_as_frame(X))
        return np.column_stack([np.full(n, 1 - self.rate_), np.full(n, self.rate_)])


def build_baselines() -> dict:
    """Every clinical rule, cheapest first."""
    return {
        "Majority class": MajorityClassRule(),
        "Symptom count (tuned k)": SymptomCountRule(),
        "Rotterdam rule (follicles >= 12)": RotterdamFollicleRule(),
        "Follicle threshold (tuned)": TunedThresholdRule(),
        "Logistic on follicle count": SingleFeatureLogistic(),
        "Decision tree (depth 2)": ShallowTreeRule(),
    }


# --------------------------------------------------------------------------
# The comparison: does machine learning beat the rules?
# --------------------------------------------------------------------------
def compare_against_ml(
    X_train,
    y_train,
    X_test,
    y_test,
    ml_models: dict,
    folds: int = 10,
) -> pd.DataFrame:
    """Score every clinical rule and every ML model under one protocol.

    Identical folds, identical test split, identical metrics. The rules get no
    handicap and no advantage -- the point is a like-for-like number.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from .evaluate import classification_metrics

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)

    entries = [(name, model, "clinical rule") for name, model in build_baselines().items()]
    entries += [(name, model, "machine learning") for name, model in ml_models.items()]

    rows = []
    for name, model, kind in entries:
        # error_score="raise" on purpose: the default swallows failures as NaN,
        # which is how a broken estimator can quietly look like a bad one.
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc",
                                 n_jobs=-1, error_score="raise")
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy",
                                 n_jobs=-1, error_score="raise")

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, model.predict(X_test), proba)

        rows.append({
            "approach": name,
            "kind": kind,
            "cv_roc_auc": cv_auc.mean(),
            "cv_roc_auc_std": cv_auc.std(),
            "cv_accuracy": cv_acc.mean(),
            "test_roc_auc": metrics["roc_auc"],
            "test_accuracy": metrics["accuracy"],
            "test_recall": metrics["recall"],
            "test_specificity": metrics["specificity"],
            "test_f1": metrics["f1"],
        })

    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)


def value_added_by_ml(
    comparison: pd.DataFrame, n_positive: int = 0, n_negative: int = 0
) -> dict:
    """How much does the ML apparatus buy over the best clinical rule?

    Deliberately compares against the best rule, not the average one. The
    question a reviewer asks is "could I have got this with a ruler and a
    threshold", so the strongest rule is the fair comparator.

    Pass the held-out class counts to also get the answer in patients rather
    than decimals, which is the form a clinician can act on.
    """
    rules = comparison[comparison["kind"] == "clinical rule"]
    ml = comparison[comparison["kind"] == "machine learning"]

    # The majority-class floor is not a clinical rule anyone would use; exclude
    # it so "best rule" means a rule with actual clinical content.
    real_rules = rules[rules["approach"] != "Majority class"]

    best_rule = real_rules.loc[real_rules["cv_roc_auc"].idxmax()]
    best_ml = ml.loc[ml["cv_roc_auc"].idxmax()]

    # A rule with no learned parameters at all is the strictest comparator.
    zero_param = rules[rules["approach"].str.startswith("Rotterdam")]
    rotterdam = zero_param.iloc[0] if len(zero_param) else None

    verdict = {
        "best_rule": best_rule["approach"],
        "best_rule_cv_auc": round(float(best_rule["cv_roc_auc"]), 4),
        "best_rule_test_auc": round(float(best_rule["test_roc_auc"]), 4),
        "best_rule_test_accuracy": round(float(best_rule["test_accuracy"]), 4),
        "best_rule_test_recall": round(float(best_rule["test_recall"]), 4),
        "best_ml": best_ml["approach"],
        "best_ml_cv_auc": round(float(best_ml["cv_roc_auc"]), 4),
        "best_ml_test_auc": round(float(best_ml["test_roc_auc"]), 4),
        "best_ml_test_accuracy": round(float(best_ml["test_accuracy"]), 4),
        "best_ml_test_recall": round(float(best_ml["test_recall"]), 4),
        "cv_auc_gain": round(float(best_ml["cv_roc_auc"] - best_rule["cv_roc_auc"]), 4),
        "test_auc_gain": round(float(best_ml["test_roc_auc"] - best_rule["test_roc_auc"]), 4),
        "test_recall_gain": round(float(best_ml["test_recall"] - best_rule["test_recall"]), 4),
        "rotterdam_cv_auc": round(float(rotterdam["cv_roc_auc"]), 4) if rotterdam is not None else None,
        "rotterdam_test_accuracy": round(float(rotterdam["test_accuracy"]), 4) if rotterdam is not None else None,
        "rotterdam_test_recall": round(float(rotterdam["test_recall"]), 4) if rotterdam is not None else None,
        "pct_of_ml_auc_from_rule": round(
            float(best_rule["cv_roc_auc"] / best_ml["cv_roc_auc"]), 4
        ),
    }

    # The most communicative framing is patients, not decimals. Averaged
    # metrics hide that a rule with excellent specificity can still be missing
    # a third of the cases it exists to find.
    if n_positive and n_negative:
        rule_found = round(best_rule["test_recall"] * n_positive)
        ml_found = round(best_ml["test_recall"] * n_positive)
        rule_fp = round((1 - best_rule["test_specificity"]) * n_negative)
        ml_fp = round((1 - best_ml["test_specificity"]) * n_negative)
        verdict.update({
            "test_positives": n_positive,
            "test_negatives": n_negative,
            "cases_found_by_rule": rule_found,
            "cases_found_by_ml": ml_found,
            "extra_cases_found_by_ml": ml_found - rule_found,
            "false_alarms_rule": rule_fp,
            "false_alarms_ml": ml_fp,
            "extra_false_alarms_from_ml": ml_fp - rule_fp,
        })
    return verdict


def plot_comparison(comparison: pd.DataFrame, verdict: dict) -> Path:
    """Rules and models on one axis, coloured by kind."""
    order = comparison.sort_values("cv_roc_auc")
    colors = [
        config.PALETTE["pcos"] if k == "machine learning" else "#9CA3AF"
        for k in order["kind"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    axes[0].barh(order["approach"], order["cv_roc_auc"],
                 xerr=order["cv_roc_auc_std"], capsize=3, color=colors)
    axes[0].set_xlim(0.4, 1.02)
    axes[0].axvline(verdict["best_rule_cv_auc"], color="black",
                    linestyle="--", linewidth=1.2, label="Best clinical rule")
    axes[0].set_xlabel("Cross-validated ROC-AUC")
    axes[0].set_title("Does machine learning beat the rule a clinician already uses?",
                      fontsize=11)
    axes[0].legend(loc="lower right", fontsize=9)

    # Recall, not accuracy, on the right. Accuracy flatters the rules: a
    # high-specificity rule on a 2:1 imbalanced set looks respectable while
    # missing a third of the cases it exists to find.
    axes[1].barh(order["approach"], order["test_recall"], color=colors)
    axes[1].set_xlim(0.0, 1.02)
    axes[1].set_xlabel("Held-out test recall (share of PCOS cases actually found)")
    axes[1].set_title("The metric that matters for screening", fontsize=11)
    axes[1].set_yticklabels([])
    for i, value in enumerate(order["test_recall"]):
        axes[1].text(value + 0.012, i, f"{value:.2f}", va="center", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=config.PALETTE["pcos"]),
        plt.Rectangle((0, 0), 1, 1, color="#9CA3AF"),
    ]
    fig.legend(handles, ["Machine learning", "Clinical rule"],
               loc="lower center", ncol=2, frameon=False)

    if "extra_cases_found_by_ml" in verdict:
        headline = (
            f"Counting follicles gets {verdict['pct_of_ml_auc_from_rule']:.0%} of the AUC — "
            f"but machine learning finds {verdict['extra_cases_found_by_ml']} more of the "
            f"{verdict['test_positives']} PCOS cases\n"
            f"({verdict['cases_found_by_rule']} → {verdict['cases_found_by_ml']}), "
            f"for {verdict['extra_false_alarms_from_ml']} extra false alarm"
            f"{'s' if verdict['extra_false_alarms_from_ml'] != 1 else ''}"
        )
    else:
        headline = (
            f"Best ML beats the best clinical rule by {verdict['cv_auc_gain']:+.4f} AUC"
        )
    fig.suptitle(headline, fontsize=12)
    fig.subplots_adjust(bottom=0.12)

    config.ensure_dirs()
    path = config.FIGURES_DIR / "29_clinical_rule_baseline.png"
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path
