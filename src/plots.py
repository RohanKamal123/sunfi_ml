"""Figures that describe model behaviour rather than the raw data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str) -> Path:
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cv_comparison(cv_results: pd.DataFrame) -> Path:
    """Mean CV metric per model, with the fold-to-fold standard deviation.

    The error bars are the important part of this figure: on 541 rows they
    are wide enough that most of the models are not distinguishable, which
    is worth saying out loud rather than ranking them to three decimals.
    """
    metrics = ["accuracy", "f1", "roc_auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.4), sharey=True)

    order = cv_results.sort_values("roc_auc", ascending=False)["model"].tolist()
    palette = sns.color_palette("crest", len(order))

    for ax, metric in zip(axes, metrics):
        subset = cv_results.set_index("model").loc[order]
        ax.barh(
            range(len(order)),
            subset[metric].values,
            xerr=subset[f"{metric}_std"].values,
            color=palette,
            capsize=3,
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_xlim(0.5, 1.02)
        ax.set_title(metric.replace("_", " ").upper())
        for i, value in enumerate(subset[metric].values):
            ax.text(value + 0.012, i, f"{value:.3f}", va="center", fontsize=9)

    fig.suptitle(
        f"Repeated stratified {config.CV_FOLDS}-fold CV "
        f"({config.CV_REPEATS} repeats), error bars = 1 SD across folds",
        fontsize=12,
    )
    return _save(fig, "08_cv_model_comparison.png")


def plot_feature_set_comparison(comparison: pd.DataFrame) -> Path:
    """Heatmap of mean ROC-AUC for every model x feature-set combination."""
    pivot = comparison.pivot_table(
        index="model", columns="feature_set", values="roc_auc", aggfunc="mean"
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(pivot) + 2.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Mean CV ROC-AUC"},
        ax=ax,
    )
    ax.set_title("Feature selection strategy vs model")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("")
    return _save(fig, "09_feature_set_comparison.png")


def plot_roc_curves(curves: dict) -> Path:
    """Test-set ROC curves for every model on one axis."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, data in sorted(curves.items(), key=lambda kv: -kv[1]["roc_auc"]):
        ax.plot(data["fpr"], data["tpr"], linewidth=1.8, label=f"{name} (AUC = {data['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "10_roc_curves.png")


def plot_pr_curves(curves: dict) -> Path:
    """Precision-recall curves, the more informative view under imbalance."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, data in sorted(curves.items(), key=lambda kv: -kv[1]["avg_precision"]):
        ax.plot(
            data["recall"],
            data["precision"],
            linewidth=1.8,
            label=f"{name} (AP = {data['avg_precision']:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves on the held-out test set")
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, "11_pr_curves.png")


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> Path:
    """Confusion matrix with counts and row-normalised percentages."""
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    annotations = np.array(
        [
            [f"{matrix[i, j]}\n({normalised[i, j]:.1%})" for j in range(2)]
            for i in range(2)
        ]
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=0.6,
        xticklabels=["Predicted: No PCOS", "Predicted: PCOS"],
        yticklabels=["Actual: No PCOS", "Actual: PCOS"],
        ax=ax,
    )
    ax.set_title(f"Confusion matrix — {model_name} (test set)")
    return _save(fig, "12_confusion_matrix.png")


def plot_leakage_experiment(leakage: pd.DataFrame) -> Path:
    """Side-by-side bars for the correct vs leaky protocol."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    data = leakage[~leakage["protocol"].str.startswith("inflation")]

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for i, (_, row) in enumerate(data.iterrows()):
        values = [row[m] for m in metrics]
        bars = ax.bar(
            x + (i - 0.5) * width,
            values,
            width,
            label=row["protocol"],
            color=[config.PALETTE["no_pcos"], config.PALETTE["pcos"]][i],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics])
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("Cross-validated score")
    ax.set_title("How much accuracy does a leaky protocol manufacture?")
    ax.legend(loc="lower right")
    return _save(fig, "13_leakage_experiment.png")


def plot_threshold_sweep(sweep: pd.DataFrame) -> Path:
    """Precision / recall / F1 as the decision threshold moves."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for metric, color in [
        ("precision", "#4C78A8"),
        ("recall", "#E45756"),
        ("f1", "#54A24B"),
        ("specificity", "#B279A2"),
    ]:
        ax.plot(sweep["threshold"], sweep[metric], marker="o", ms=3.5, label=metric, color=color)

    best = sweep.loc[sweep["f1"].idxmax(), "threshold"]
    ax.axvline(0.5, color="grey", linestyle=":", label="default (0.50)")
    ax.axvline(best, color="black", linestyle="--", linewidth=1, label=f"best F1 ({best:.2f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sensitivity on the test set")
    ax.legend(fontsize=9)
    return _save(fig, "14_threshold_sweep.png")


def plot_paper_comparison(comparison: pd.DataFrame) -> Path:
    """Our result placed against the accuracies reported by the five papers."""
    data = comparison.sort_values("reported_accuracy")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = [
        config.PALETTE["pcos"] if row["is_ours"] else "#9CA3AF"
        for _, row in data.iterrows()
    ]
    bars = ax.barh(data["label"], data["reported_accuracy"], color=colors)

    for bar, (_, row) in zip(bars, data.iterrows()):
        value = row["reported_accuracy"]
        # Paper 2 reports only AUC, so its bar is not an accuracy at all.
        metric = "AUC" if row["accuracy_is_proxy"] else "acc"
        ax.text(
            value + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f} {metric}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0.7, 1.06)
    ax.set_xlabel("Reported accuracy (Paper 2 reports AUC only)")
    ax.set_title(
        "Headline scores across the reviewed literature and this project\n"
        "No study, including this one, validated on an external cohort",
        fontsize=12,
    )
    return _save(fig, "15_paper_comparison.png")
