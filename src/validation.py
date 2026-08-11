"""Rigorous validation: nested CV, split stability, learning curves, and CIs.

Four questions the flat cross-validation in :mod:`src.evaluate` cannot answer:

1. **Is the reported CV score itself biased?** The pipeline picks a feature
   set using CV on the training split, then reports a CV score on that same
   split. Selection and evaluation share data, so the number is optimistic.
   Nested CV measures the bias and removes it.
2. **How arbitrary is the "winner"?** Re-run the whole thing over many random
   splits and count how often each model comes first.
3. **Would more data help?** Learning curves separate "the model is too
   simple" from "the dataset is too small".
4. **How precise is the test-set number?** 109 rows deserve an interval, not
   a point estimate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)

from . import config


def _save(fig, name: str) -> Path:
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 1. Nested cross-validation
# --------------------------------------------------------------------------
def nested_cv(
    X,
    y,
    build_pipeline_fn,
    classifier_fn,
    selectors=("all", "correlation", "chi2", "rfe"),
    inner_folds: int = 5,
    outer_folds: int = 10,
    scoring: str = "roc_auc",
) -> dict:
    """Nested CV, and the size of the bias it removes.

    The outer loop splits the data. Inside each outer training fold, an inner
    loop chooses the feature-selection strategy. The chosen pipeline is then
    scored on the outer *test* fold, which the selection never saw.

    Compare that to the flat protocol, where the selector is chosen using the
    whole training set and then scored by CV on the same rows. The difference
    between the two is optimism: the amount by which choosing a
    hyperparameter on your evaluation data inflates the result.

    It is the same error as the SMOTE leak measured in
    :func:`src.evaluate.leakage_experiment`, one level up. Reporting it makes
    the project's methodological argument apply to the project itself.
    """
    outer = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=config.RANDOM_STATE
    )
    inner = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=config.RANDOM_STATE
    )

    pipeline = build_pipeline_fn(X, classifier_fn(), selector="all")
    grid = {"select": [_selector_instance(name) for name in selectors]}

    search = GridSearchCV(
        pipeline, grid, scoring=scoring, cv=inner, n_jobs=-1, refit=True
    )

    nested_scores = cross_val_score(search, X, y, cv=outer, scoring=scoring, n_jobs=-1)

    # Flat protocol: choose the selector once on everything, then CV on the
    # same data with that choice fixed.
    flat_by_selector = {}
    for name in selectors:
        candidate = build_pipeline_fn(X, classifier_fn(), selector=name)
        flat_by_selector[name] = cross_val_score(
            candidate, X, y, cv=outer, scoring=scoring, n_jobs=-1
        ).mean()

    best_selector = max(flat_by_selector, key=flat_by_selector.get)
    flat_score = flat_by_selector[best_selector]

    return {
        "nested_mean": float(nested_scores.mean()),
        "nested_std": float(nested_scores.std()),
        "nested_scores": nested_scores.tolist(),
        "flat_mean": float(flat_score),
        "flat_best_selector": best_selector,
        "flat_by_selector": {k: float(v) for k, v in flat_by_selector.items()},
        "optimism": float(flat_score - nested_scores.mean()),
        "scoring": scoring,
    }


def _selector_instance(name: str):
    from .features import build_selector

    return build_selector(name, k=15)


def plot_nested_cv(result: dict) -> Path:
    """Flat vs nested estimate, with the optimism gap called out."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    labels = ["Flat CV\n(selector chosen on same data)", "Nested CV\n(selector chosen inside folds)"]
    values = [result["flat_mean"], result["nested_mean"]]
    colors = [config.PALETTE["pcos"], config.PALETTE["no_pcos"]]

    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.errorbar(
        1, result["nested_mean"], yerr=result["nested_std"],
        fmt="none", ecolor="black", capsize=5,
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002,
                f"{value:.4f}", ha="center", fontsize=11)

    ax.set_ylim(min(values) - 0.03, max(values) + 0.02)
    ax.set_ylabel(f"{result['scoring'].upper()}")
    ax.set_title(
        f"Selection bias in the reported CV score: {result['optimism']:+.4f}",
        fontsize=12,
    )
    return _save(fig, "22_nested_cv.png")


# --------------------------------------------------------------------------
# 2. Split stability — how arbitrary is the winner?
# --------------------------------------------------------------------------
def seed_sweep(
    X,
    y,
    build_models_fn,
    n_seeds: int = 50,
    selector: str = "chi2",
    test_size: float = config.TEST_SIZE,
) -> pd.DataFrame:
    """Re-split the data many times and record which model wins each time.

    A single train/test split is one draw from this distribution. If the
    winner changes from draw to draw, then any paper that reports "model X was
    best" from one split is reporting which model got the lucky fold.
    """
    records = []
    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        models = build_models_fn(X_train, selector=selector)

        for name, model in models.items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            records.append(
                {"seed": seed, "model": name, "roc_auc": roc_auc_score(y_test, proba)}
            )

    return pd.DataFrame(records)


def win_counts(sweep: pd.DataFrame) -> pd.DataFrame:
    """How often each model took first place across the seeds."""
    winners = sweep.loc[sweep.groupby("seed")["roc_auc"].idxmax(), "model"]
    counts = winners.value_counts()
    n_seeds = sweep["seed"].nunique()

    summary = (
        sweep.groupby("model")["roc_auc"]
        .agg(mean_auc="mean", std_auc="std", min_auc="min", max_auc="max")
        .reset_index()
    )
    summary["wins"] = summary["model"].map(counts).fillna(0).astype(int)
    summary["win_rate"] = (summary["wins"] / n_seeds).round(3)
    summary["auc_range"] = (summary["max_auc"] - summary["min_auc"]).round(4)
    return summary.sort_values("wins", ascending=False).reset_index(drop=True)


def plot_seed_sweep(sweep: pd.DataFrame, wins: pd.DataFrame) -> Path:
    """Win counts, and the spread of test AUC each model shows across splits."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    n_seeds = sweep["seed"].nunique()

    order = wins.sort_values("wins", ascending=False)
    axes[0].barh(order["model"], order["wins"], color=config.PALETTE["pcos"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f"Times ranked #1 (out of {n_seeds} random splits)")
    axes[0].set_title("Which model 'wins' depends on the split", fontsize=11)
    for i, (_, row) in enumerate(order.iterrows()):
        axes[0].text(row["wins"] + 0.3, i, f"{row['win_rate']:.0%}", va="center", fontsize=9)

    model_order = order["model"].tolist()
    axes[1].boxplot(
        [sweep.loc[sweep["model"] == m, "roc_auc"] for m in model_order],
        tick_labels=model_order,
        patch_artist=True,
        boxprops={"facecolor": config.PALETTE["no_pcos"], "alpha": 0.55},
        medianprops={"color": "black"},
    )
    axes[1].set_ylabel("Test ROC-AUC")
    axes[1].set_title(f"Test-set AUC across {n_seeds} random splits", fontsize=11)
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(
        "A single train/test split is one draw from this distribution", fontsize=13
    )
    return _save(fig, "23_seed_sweep.png")


# --------------------------------------------------------------------------
# 3. Learning curves — would more data help?
# --------------------------------------------------------------------------
def compute_learning_curve(
    model, X, y, folds: int = 5, points: int = 8
) -> pd.DataFrame:
    """Cross-validated score as a function of training-set size."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    fractions = np.linspace(0.15, 1.0, points)

    sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=fractions,
        scoring="roc_auc", n_jobs=-1, shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    return pd.DataFrame(
        {
            "train_size": sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "cv_mean": test_scores.mean(axis=1),
            "cv_std": test_scores.std(axis=1),
        }
    )


def plot_learning_curve(curve: pd.DataFrame, model_name: str) -> Path:
    """Train vs validation score against training-set size.

    Reading it: a validation curve still climbing at the right-hand edge means
    more patients would help. A flat one means the ceiling is the features and
    the label noise, not the sample size.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(curve["train_size"], curve["train_mean"], marker="o", ms=5,
            color=config.PALETTE["pcos"], label="Training score")
    ax.fill_between(
        curve["train_size"],
        curve["train_mean"] - curve["train_std"],
        curve["train_mean"] + curve["train_std"],
        alpha=0.15, color=config.PALETTE["pcos"],
    )

    ax.plot(curve["train_size"], curve["cv_mean"], marker="s", ms=5,
            color=config.PALETTE["no_pcos"], label="Cross-validated score")
    ax.fill_between(
        curve["train_size"],
        curve["cv_mean"] - curve["cv_std"],
        curve["cv_mean"] + curve["cv_std"],
        alpha=0.15, color=config.PALETTE["no_pcos"],
    )

    # Slope over the last third, as a rough "is it still climbing" read.
    tail = curve.tail(max(3, len(curve) // 3))
    slope = np.polyfit(tail["train_size"], tail["cv_mean"], 1)[0]
    per_100 = slope * 100

    ax.set_xlabel("Training set size (patients)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        f"Learning curve — {model_name}\n"
        f"Validation score is changing by {per_100:+.4f} AUC per 100 extra patients",
        fontsize=12,
    )
    ax.legend(loc="lower right")
    return _save(fig, "24_learning_curve.png")


# --------------------------------------------------------------------------
# 4. Bootstrap confidence intervals
# --------------------------------------------------------------------------
def bootstrap_test_metrics(
    model, X_test, y_test, n_boot: int = 2000, alpha: float = 0.05
) -> pd.DataFrame:
    """Percentile bootstrap CIs for the held-out metrics.

    Resamples the *test set* with replacement, keeping the fitted model fixed,
    so the interval reflects uncertainty from having only 109 evaluation
    patients. It does not capture uncertainty from the training split -- the
    seed sweep covers that.
    """
    from .evaluate import classification_metrics

    y_test = np.asarray(y_test)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    point = classification_metrics(y_test, pred, proba)
    metrics = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]

    rng = np.random.default_rng(config.RANDOM_STATE)
    samples = {m: [] for m in metrics}
    n = len(y_test)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # A resample can miss one class entirely; those draws are unusable.
        if len(np.unique(y_test[idx])) < 2:
            continue
        boot = classification_metrics(y_test[idx], pred[idx], proba[idx])
        for m in metrics:
            samples[m].append(boot[m])

    rows = []
    for m in metrics:
        values = np.array(samples[m])
        lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows.append(
            {
                "metric": m,
                "estimate": round(point[m], 4),
                "ci_lower": round(float(lower), 4),
                "ci_upper": round(float(upper), 4),
                "ci_width": round(float(upper - lower), 4),
            }
        )
    return pd.DataFrame(rows)


def plot_bootstrap_ci(ci: pd.DataFrame, model_name: str, n_test: int) -> Path:
    """Point estimates with their 95% intervals."""
    fig, ax = plt.subplots(figsize=(8, 4.8))

    y = np.arange(len(ci))
    ax.errorbar(
        ci["estimate"], y,
        xerr=[ci["estimate"] - ci["ci_lower"], ci["ci_upper"] - ci["estimate"]],
        fmt="o", ms=8, capsize=5, color=config.PALETTE["no_pcos"],
    )
    for i, (_, row) in enumerate(ci.iterrows()):
        ax.text(row["ci_upper"] + 0.012, i,
                f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]",
                va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ") for m in ci["metric"]])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.15)
    ax.set_xlabel("Score")
    ax.set_title(
        f"Test-set metrics with 95% bootstrap CIs — {model_name}\n"
        f"n = {n_test} patients: the intervals are wide, and that is the honest picture",
        fontsize=12,
    )
    return _save(fig, "25_bootstrap_ci.png")
