"""Exploratory data analysis figures.

Every function saves a PNG into ``reports/figures`` and returns its path, so
the pipeline script can collect them and the notebook can display them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the pipeline runs without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid", context="notebook")

CLASS_LABELS = {0: "No PCOS", 1: "PCOS"}
CLASS_COLORS = [config.PALETTE["no_pcos"], config.PALETTE["pcos"]]


def _save(fig, name: str) -> Path:
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame) -> Path:
    """The 364 / 177 split that motivates the whole SMOTE discussion."""
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        [CLASS_LABELS[i] for i in counts.index], counts.values, color=CLASS_COLORS
    )
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 6,
            f"{value}\n({value / total:.1%})",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Patients")
    ax.set_title(f"Class distribution (n = {total}, ratio {counts[0] / counts[1]:.1f}:1)")
    ax.set_ylim(0, counts.max() * 1.2)
    return _save(fig, "01_class_balance.png")


def plot_missingness(df: pd.DataFrame) -> Path:
    """Which columns had missing or invalid values after cleaning."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(missing) + 1)))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh([config.pretty(c) for c in missing.index], missing.values, color="#7C7C7C")
        ax.set_xlabel("Missing values")
        ax.set_title("Missing / invalid values remaining after cleaning")
        ax.set_xlim(0, max(missing.values) * 1.35)
        for y, v in enumerate(missing.values):
            ax.text(v + 0.03, y, str(v), va="center")
    return _save(fig, "02_missingness.png")


def plot_target_correlations(df: pd.DataFrame, top_n: int = 20) -> Path:
    """The features most linearly associated with a PCOS diagnosis."""
    corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET)
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(7, 0.38 * len(top) + 1.5))
    colors = [config.PALETTE["pcos"] if v > 0 else config.PALETTE["no_pcos"] for v in top]
    ax.barh([config.pretty(c) for c in top.index], top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with PCOS diagnosis")
    ax.set_title(f"Top {top_n} features by correlation with the target")
    return _save(fig, "03_target_correlations.png")


def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 18) -> Path:
    """Correlations among the strongest features, to expose redundancy."""
    corr_with_target = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET).abs()
    cols = list(corr_with_target.sort_values(ascending=False).head(top_n).index)
    cols = [config.TARGET] + cols

    matrix = df[cols].corr()
    matrix.index = [config.pretty(c) for c in matrix.index]
    matrix.columns = [config.pretty(c) for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(
        matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Correlation structure of the strongest predictors")
    return _save(fig, "04_correlation_heatmap.png")


def plot_key_distributions(df: pd.DataFrame) -> Path:
    """Distributions of the clinically important continuous variables by class."""
    features = [
        "Follicle No. (L)",
        "Follicle No. (R)",
        "AMH(ng/mL)",
        "BMI",
        "Cycle length(days)",
        "Age (yrs)",
        "Waist:Hip Ratio",
        "LH(mIU/mL)",
        "Endometrium (mm)",
    ]
    features = [f for f in features if f in df.columns]

    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        for label, color in zip([0, 1], CLASS_COLORS):
            values = df.loc[df[config.TARGET] == label, feature].dropna()
            # AMH and the hormone panels have extreme outliers that flatten
            # the plot; clip the display range at the 99th percentile.
            upper = df[feature].quantile(0.99)
            sns.kdeplot(
                values.clip(upper=upper),
                ax=ax,
                fill=True,
                alpha=0.4,
                color=color,
                label=CLASS_LABELS[label],
                warn_singular=False,
            )
        ax.set_title(config.pretty(feature), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[len(features):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Feature distributions by PCOS status", y=1.0, fontsize=13)
    fig.subplots_adjust(bottom=0.08)
    return _save(fig, "05_key_distributions.png")


def plot_symptom_prevalence(df: pd.DataFrame) -> Path:
    """Prevalence of each binary symptom within each class."""
    symptoms = [c for c in config.BINARY_COLUMNS if c in df.columns]
    rows = []
    for symptom in symptoms:
        for label in (0, 1):
            subset = df.loc[df[config.TARGET] == label, symptom].dropna()
            rows.append(
                {
                    "symptom": config.pretty(symptom),
                    "group": CLASS_LABELS[label],
                    "prevalence": subset.mean(),
                }
            )
    prevalence = pd.DataFrame(rows)

    order = (
        prevalence[prevalence["group"] == "PCOS"]
        .sort_values("prevalence", ascending=False)["symptom"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=prevalence,
        x="symptom",
        y="prevalence",
        hue="group",
        order=order,
        palette=CLASS_COLORS,
        ax=ax,
    )
    ax.set_ylabel("Proportion of patients")
    ax.set_xlabel("")
    ax.set_title("Symptom prevalence by PCOS status")
    ax.legend(title="")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _save(fig, "06_symptom_prevalence.png")


def plot_follicle_separation(df: pd.DataFrame) -> Path:
    """Follicle counts, the single clearest signal in the dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    plot_df = df.copy()
    plot_df["group"] = plot_df[config.TARGET].map(CLASS_LABELS)

    sns.boxplot(
        data=plot_df,
        x="group",
        y="Follicle No. (L)",
        hue="group",
        palette=CLASS_COLORS,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Left ovary follicle count")
    axes[0].set_xlabel("")

    sns.scatterplot(
        data=plot_df,
        x="Follicle No. (L)",
        y="Follicle No. (R)",
        hue="group",
        palette=CLASS_COLORS,
        alpha=0.65,
        s=32,
        ax=axes[1],
    )
    # The Rotterdam criteria use a threshold of 12 follicles per ovary.
    axes[1].axvline(12, color="grey", linestyle="--", linewidth=1)
    axes[1].axhline(12, color="grey", linestyle="--", linewidth=1)
    axes[1].set_title("Left vs right follicle count (dashed: 12-follicle criterion)")
    axes[1].legend(title="")

    fig.suptitle("Follicle counts separate the classes most cleanly", fontsize=13)
    return _save(fig, "07_follicle_separation.png")


def run_all(df: pd.DataFrame) -> list[Path]:
    """Generate every EDA figure."""
    return [
        plot_class_balance(df),
        plot_missingness(df),
        plot_target_correlations(df),
        plot_correlation_heatmap(df),
        plot_key_distributions(df),
        plot_symptom_prevalence(df),
        plot_follicle_separation(df),
    ]


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature summary split by class, with a t-test style effect size."""
    rows = []
    for col in df.columns:
        if col == config.TARGET:
            continue
        no_pcos = df.loc[df[config.TARGET] == 0, col].dropna()
        pcos = df.loc[df[config.TARGET] == 1, col].dropna()
        pooled_sd = np.sqrt((no_pcos.var() + pcos.var()) / 2)
        cohens_d = (pcos.mean() - no_pcos.mean()) / pooled_sd if pooled_sd else 0.0
        rows.append(
            {
                "feature": config.pretty(col),
                "mean_no_pcos": round(no_pcos.mean(), 3),
                "mean_pcos": round(pcos.mean(), 3),
                "std_no_pcos": round(no_pcos.std(), 3),
                "std_pcos": round(pcos.std(), 3),
                "cohens_d": round(cohens_d, 3),
            }
        )
    return (
        pd.DataFrame(rows)
        .reindex(pd.DataFrame(rows)["cohens_d"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )
