"""Cross-validation, held-out testing, and the leakage experiment.

Three levels of evidence, in increasing order of trustworthiness:

1. :func:`cross_validate_models` — repeated stratified k-fold on the training
   split. Used to *choose* a model.
2. :func:`evaluate_on_test` — a single pass over data that was held out
   before any modelling decision was made. Used to *report* a number.
3. :func:`leakage_experiment` — deliberately does the wrong thing (SMOTE and
   feature selection before splitting) to quantify how much accuracy that
   mistake manufactures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from . import config

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def make_splits(X, y, test_size: float = config.TEST_SIZE):
    """Stratified train/test split, held fixed by the global seed.

    The test split is created once, at the very start, and nothing in the
    model-selection process is allowed to look at it.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


def cross_validate_models(
    models: dict,
    X,
    y,
    folds: int = config.CV_FOLDS,
    repeats: int = config.CV_REPEATS,
) -> pd.DataFrame:
    """Repeated stratified k-fold CV for a dict of ``name -> pipeline``.

    Repeating the k-fold with different shuffles matters here: with only
    ~430 training rows, a single 10-fold run swings by several points
    depending on the seed, and reporting one lucky split is exactly the
    fragility this project is trying to avoid.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=config.RANDOM_STATE
    )

    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score="raise"
        )
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[metric] = values.mean()
            row[f"{metric}_std"] = values.std()
        row["fit_time_s"] = scores["fit_time"].mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def compare_feature_sets(
    build_models_fn,
    X,
    y,
    selectors=("all", "correlation", "chi2", "rfe"),
    k_features: int = 15,
    folds: int = config.CV_FOLDS,
    repeats: int = 1,
) -> pd.DataFrame:
    """Cross-validate every (model, feature-set) combination.

    ``repeats`` defaults to 1 here because this grid is 4x larger than the
    single-feature-set run and is only used to pick a selector.
    """
    frames = []
    for selector in selectors:
        models = build_models_fn(X, selector=selector, k_features=k_features)
        result = cross_validate_models(models, X, y, folds=folds, repeats=repeats)
        result.insert(0, "feature_set", selector)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Every headline metric for one set of predictions."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["avg_precision"] = average_precision_score(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            # Specificity matters clinically: it is the rate at which healthy
            # women are correctly cleared.
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        }
    )
    return metrics


def evaluate_on_test(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Fit each model on the full training split and score it once on the test split."""
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        rows.append({"model": name, **classification_metrics(y_test, y_pred, y_proba)})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def curve_data(models: dict, X_test, y_test) -> dict:
    """ROC and precision-recall curve points for already-fitted models."""
    curves = {}
    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        curves[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": auc(fpr, tpr),
            "precision": precision,
            "recall": recall,
            "avg_precision": average_precision_score(y_test, proba),
        }
    return curves


def leakage_experiment(
    X,
    y,
    classifier_fn,
    folds: int = config.CV_FOLDS,
    k_features: int = 15,
) -> pd.DataFrame:
    """Quantify the accuracy inflation from resampling before splitting.

    Two protocols are compared on identical folds and an identical model:

    *correct*
        SMOTE and feature selection sit inside the pipeline, so they are
        refitted per fold on training data only.
    *leaky*
        SMOTE and feature selection are applied once to the whole dataset,
        then the oversampled data is cross-validated. Synthetic minority
        points generated from a training row can then land in the validation
        fold, and the selector has already read every label.

    The gap between the two is a direct, measured estimate of how much of a
    reported 98-99% accuracy can be an artefact of methodology rather than a
    property of the model.
    """
    from imblearn.over_sampling import SMOTE

    from .models import build_pipeline, build_preprocessor
    from .features import build_selector

    # --- Correct protocol ------------------------------------------------
    correct_pipeline = build_pipeline(
        X, classifier_fn(), selector="rfe", k_features=k_features, balance="smote"
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    correct = cross_validate(
        correct_pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score="raise"
    )

    # --- Leaky protocol --------------------------------------------------
    # Preprocess, select and oversample the ENTIRE dataset up front.
    preprocessor = build_preprocessor(X)
    X_prep = preprocessor.fit_transform(X, y)
    selector = build_selector("rfe", k=k_features).fit(X_prep, y)
    X_sel = selector.transform(X_prep)
    X_leaky, y_leaky = SMOTE(random_state=config.RANDOM_STATE).fit_resample(X_sel, y)

    cv_leaky = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE
    )
    leaky = cross_validate(
        classifier_fn(),
        X_leaky,
        y_leaky,
        cv=cv_leaky,
        scoring=SCORING,
        n_jobs=-1,
        error_score="raise",
    )

    rows = []
    for protocol, scores in (("correct (SMOTE inside CV)", correct), ("leaky (SMOTE before CV)", leaky)):
        row = {"protocol": protocol}
        for metric in SCORING:
            row[metric] = scores[f"test_{metric}"].mean()
        rows.append(row)

    result = pd.DataFrame(rows)
    inflation = {"protocol": "inflation (leaky - correct)"}
    for metric in SCORING:
        inflation[metric] = result.loc[1, metric] - result.loc[0, metric]
    return pd.concat([result, pd.DataFrame([inflation])], ignore_index=True)


def balance_ablation(X_train, y_train, X_test, y_test, classifier_fn, k_features: int = 15) -> pd.DataFrame:
    """Compare the three ways of handling the 2:1 class imbalance.

    None of the reviewed papers state which they used, so the effect is
    measured rather than assumed.
    """
    from .models import build_pipeline

    variants = {
        "no balancing": (classifier_fn(), "none"),
        "class weighting": (classifier_fn(class_weight="balanced"), "none"),
        "SMOTE": (classifier_fn(), "smote"),
    }

    rows = []
    for label, (clf, balance) in variants.items():
        pipeline = build_pipeline(
            X_train, clf, selector="rfe", k_features=k_features, balance=balance
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        rows.append({"strategy": label, **classification_metrics(y_test, y_pred, y_proba)})
    return pd.DataFrame(rows)


def threshold_sweep(model, X_test, y_test, thresholds=None) -> pd.DataFrame:
    """Metrics across decision thresholds.

    The default 0.5 cut-off is rarely the right one for a screening tool,
    where a missed PCOS case costs more than a false alarm sent for
    confirmatory testing.
    """
    thresholds = thresholds if thresholds is not None else np.arange(0.05, 0.96, 0.05)
    proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        metrics = classification_metrics(y_test, y_pred)
        metrics.pop("balanced_accuracy", None)
        rows.append({"threshold": round(float(t), 2), **metrics})
    return pd.DataFrame(rows)
