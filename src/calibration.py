"""Probability calibration and statistical comparison between models.

Two questions that ROC-AUC cannot answer:

1. **Are the predicted probabilities meaningful?** AUC only measures ranking.
   A model can rank patients perfectly and still claim 0.8 for cases that are
   right 60% of the time. That distinction is invisible in every metric
   reported by the reviewed papers, and it is the one that matters when a
   probability is used to set a screening threshold.

2. **Is the best model actually better?** When five models sit within one
   standard deviation of each other, ranking them by mean score is ranking
   noise. Paired tests across folds say whether any difference survives.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config


def _save(fig, name: str) -> Path:
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibration_metrics(y_true, y_proba) -> dict:
    """Brier score, log loss and expected calibration error.

    Brier score is a proper scoring rule: it rewards being both correct and
    appropriately confident, and is minimised only by honest probabilities.
    Lower is better for all three.
    """
    return {
        "brier": brier_score_loss(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "ece": expected_calibration_error(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Mean gap between predicted confidence and observed frequency.

    Predictions are bucketed by probability; within each bucket the mean
    predicted probability is compared to the actual positive rate. The buckets
    are averaged weighted by how many predictions fall in each.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        # Include the left edge on the first bin so 0.0 is never dropped.
        in_bin = (y_proba > lower) & (y_proba <= upper)
        if lower == 0.0:
            in_bin |= y_proba == 0.0
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        error += weight * abs(y_true[in_bin].mean() - y_proba[in_bin].mean())
    return float(error)


def compare_calibration(models: dict, X, y, folds: int = 5) -> pd.DataFrame:
    """Cross-validated calibration metrics for each model.

    Uses out-of-fold predictions rather than the test set so that every
    patient contributes, which matters when the test split holds only 36
    positive cases.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)

    rows = []
    for name, model in models.items():
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        rows.append({"model": name, **calibration_metrics(y, proba)})

    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)


def plot_calibration_curves(models: dict, X, y, folds: int = 5) -> Path:
    """Reliability diagram: predicted probability vs observed frequency."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")

    for name, model in models.items():
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        true_freq, mean_pred = calibration_curve(y, proba, n_bins=8, strategy="quantile")
        brier = brier_score_loss(y, proba)
        axes[0].plot(mean_pred, true_freq, marker="o", ms=4, linewidth=1.5,
                     label=f"{name} (Brier {brier:.3f})")
        axes[1].hist(proba, bins=25, histtype="step", linewidth=1.5, label=name)

    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed frequency of PCOS")
    axes[0].set_title("Reliability diagram (out-of-fold predictions)")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of predicted probabilities")
    axes[1].legend(fontsize=8)

    fig.suptitle("Are the predicted probabilities trustworthy?", fontsize=13)
    return _save(fig, "19_calibration_curves.png")


def calibration_effect(base_model, calibrated_model, X, y, folds: int = 5) -> pd.DataFrame:
    """Before/after table for wrapping a model in a calibrator."""
    rows = []
    for label, model in (("uncalibrated", base_model), ("calibrated", calibrated_model)):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        rows.append({"variant": label, **calibration_metrics(y, proba)})

    result = pd.DataFrame(rows)
    delta = {"variant": "change"}
    for column in ("brier", "log_loss", "ece", "roc_auc"):
        delta[column] = result.loc[1, column] - result.loc[0, column]
    return pd.concat([result, pd.DataFrame([delta])], ignore_index=True)


# --------------------------------------------------------------------------
# Statistical comparison
# --------------------------------------------------------------------------
def paired_fold_scores(
    models: dict, X, y, scoring: str = "roc_auc", folds: int = 10, repeats: int = 3
) -> pd.DataFrame:
    """Per-fold scores for every model on *identical* folds.

    Sharing folds is what makes the comparison paired: each model is scored on
    exactly the same patients, so the difference per fold isolates the model
    rather than the split.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    cv = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=config.RANDOM_STATE
    )
    scores = {
        name: cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        for name, model in models.items()
    }
    return pd.DataFrame(scores)


def significance_matrix(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test p-values for every pair of models.

    Note the caveat that applies to all of these: repeated-CV folds overlap,
    so the scores are not independent and the paired t-test is known to be
    optimistic (it under-estimates p). A p-value that is *already* large is
    therefore strong evidence of no difference, which is the direction this
    project needs. Small p-values here should be treated as suggestive only.
    """
    names = list(fold_scores.columns)
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)

    for a in names:
        for b in names:
            if a == b:
                matrix.loc[a, b] = np.nan
            else:
                _, p = stats.ttest_rel(fold_scores[a], fold_scores[b])
                matrix.loc[a, b] = p
    return matrix


def significance_against_best(
    fold_scores: pd.DataFrame, practical_threshold: float = 0.02
) -> pd.DataFrame:
    """Test every model against the best-scoring one.

    Answers the question the report actually needs: is the winner
    distinguishable from the rest, or is the ranking noise?

    Two separate columns, because they answer different questions and
    conflating them is a common way to over-claim:

    ``statistically_sep``
        Did a paired test detect *any* consistent difference? With 30 folds
        even a 0.005 AUC gap can clear p < 0.05.
    ``practically_sep``
        Is the gap large enough to matter? A difference smaller than
        ``practical_threshold`` (default 0.02 AUC, well inside this dataset's
        ~0.03 fold-to-fold spread) would not change a single clinical
        decision, however small its p-value.
    """
    means = fold_scores.mean().sort_values(ascending=False)
    best = means.index[0]

    rows = []
    for name in means.index:
        if name == best:
            rows.append({
                "model": name,
                "mean_score": means[name],
                "std": fold_scores[name].std(),
                "delta_vs_best": 0.0,
                "p_value": np.nan,
                "statistically_sep": "— (reference)",
                "practically_sep": "— (reference)",
            })
            continue

        _, p = stats.ttest_rel(fold_scores[best], fold_scores[name])
        delta = means[name] - means[best]
        rows.append({
            "model": name,
            "mean_score": means[name],
            "std": fold_scores[name].std(),
            "delta_vs_best": delta,
            "p_value": p,
            "statistically_sep": "yes" if p < 0.05 else "no",
            "practically_sep": "yes" if abs(delta) >= practical_threshold else "no",
        })

    return pd.DataFrame(rows)


def plot_significance(fold_scores: pd.DataFrame) -> Path:
    """Box plot of per-fold scores, which shows the overlap directly."""
    order = fold_scores.mean().sort_values(ascending=False).index.tolist()
    data = fold_scores[order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    box = ax.boxplot(
        [data[c] for c in order],
        tick_labels=order,
        patch_artist=True,
        medianprops={"color": "black"},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(config.PALETTE["no_pcos"])
        patch.set_alpha(0.55)

    # Scatter the individual folds so the reader sees the raw spread.
    for i, column in enumerate(order, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, len(data))
        ax.scatter(i + jitter, data[column], s=8, color="#333333", alpha=0.35, zorder=3)

    ax.set_ylabel("ROC-AUC per fold")
    ax.set_title(
        "Per-fold score distributions — the overlap is the point\n"
        f"({config.CV_FOLDS} folds x {config.CV_REPEATS} repeats)",
        fontsize=12,
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, "20_significance_boxplot.png")
