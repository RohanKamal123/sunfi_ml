"""Clinical utility: decision curves, cost-tiered screening, subgroup checks.

Accuracy and AUC answer "is the model right". Clinicians need "is using this
model better than what we do now, and for whom". Three analyses:

* **Decision curve analysis** — net benefit against the two trivial policies
  (test everyone / test nobody), across the range of thresholds a clinician
  might reasonably adopt.
* **Cost-tiered screening** — what does each tier of measurement actually buy?
  The project's motivation is deployment where testing is scarce, so this
  quantifies the ultrasound and the blood panel in AUC terms.
* **Subgroup performance** — a model that works on average but fails for
  women over 35 is not deployable, and average metrics hide that.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def _save(fig, name: str) -> Path:
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Decision curve analysis
# --------------------------------------------------------------------------
def net_benefit(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Net benefit of the model vs treat-all and treat-none.

    Net benefit (Vickers & Elkin, 2006) converts a confusion matrix into a
    single number on the scale of "true positives per patient", after
    discounting false positives by how much a clinician cares about them::

        NB = TP/n - FP/n * (p_t / (1 - p_t))

    The threshold probability ``p_t`` encodes that trade-off: adopting a
    threshold of 0.2 says a missed PCOS case is 4x worse than an unnecessary
    follow-up. Sweeping ``p_t`` shows over which range of clinical opinions
    the model is worth using.

    The comparators matter as much as the model:

    * **treat all** — refer every woman for confirmatory testing.
    * **treat none** — refer nobody; net benefit is 0 by construction.

    A model is only useful where its curve sits above *both*.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = np.arange(0.01, 0.81, 0.01) if thresholds is None else np.asarray(thresholds)

    n = len(y_true)
    prevalence = y_true.mean()

    rows = []
    for p_t in thresholds:
        odds = p_t / (1 - p_t)

        predicted = y_proba >= p_t
        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        model_nb = tp / n - (fp / n) * odds

        # Treat-all: everyone is flagged, so TP = all positives, FP = all negatives.
        all_nb = prevalence - (1 - prevalence) * odds

        rows.append(
            {
                "threshold": round(float(p_t), 3),
                "net_benefit_model": model_nb,
                "net_benefit_treat_all": all_nb,
                "net_benefit_treat_none": 0.0,
                "n_flagged": int(predicted.sum()),
            }
        )

    df = pd.DataFrame(rows)
    df["benefit_over_best_default"] = df["net_benefit_model"] - df[
        ["net_benefit_treat_all", "net_benefit_treat_none"]
    ].max(axis=1)
    return df


def plot_decision_curve(curve: pd.DataFrame, model_name: str) -> Path:
    """The decision curve, with the useful threshold range shaded."""
    fig, ax = plt.subplots(figsize=(8, 5.2))

    ax.plot(curve["threshold"], curve["net_benefit_model"], linewidth=2.2,
            color=config.PALETTE["pcos"], label=f"{model_name}")
    ax.plot(curve["threshold"], curve["net_benefit_treat_all"], linewidth=1.5,
            color="#888888", linestyle="--", label="Refer everyone")
    ax.axhline(0, color="black", linewidth=1.2, linestyle=":", label="Refer nobody")

    useful = curve[curve["benefit_over_best_default"] > 0]
    if not useful.empty:
        lo, hi = useful["threshold"].min(), useful["threshold"].max()
        ax.axvspan(lo, hi, alpha=0.12, color=config.PALETTE["no_pcos"])
        ax.text(
            (lo + hi) / 2, ax.get_ylim()[1] * 0.9,
            f"model adds value\n{lo:.0%}–{hi:.0%}",
            ha="center", fontsize=9, color=config.PALETTE["no_pcos"],
        )

    ax.set_xlabel("Threshold probability (how much worse a missed case is than a false alarm)")
    ax.set_ylabel("Net benefit")
    ax.set_ylim(bottom=min(-0.05, curve["net_benefit_model"].min()))
    ax.set_title("Decision curve analysis — is the model clinically worth using?", fontsize=12)
    ax.legend(loc="upper right")
    return _save(fig, "26_decision_curve.png")


# --------------------------------------------------------------------------
# Cost-tiered screening models
# --------------------------------------------------------------------------
def tiered_models(
    X_train, y_train, X_test, y_test, build_pipeline_fn, classifier_fn
) -> pd.DataFrame:
    """Train one model per feature-acquisition tier.

    Answers a question the reviewed papers never ask: if a clinic cannot
    afford an ultrasound, how much predictive power is actually lost? The
    introduction to this project argues PCOS testing is often inaccessible;
    this turns that argument into a number.
    """
    from .evaluate import classification_metrics
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.RANDOM_STATE)

    rows = []
    previous_auc = None
    for tier_name, columns in config.FEATURE_TIERS.items():
        available = [c for c in columns if c in X_train.columns]
        if not available:
            continue

        X_tr, X_te = X_train[available], X_test[available]
        pipeline = build_pipeline_fn(X_tr, classifier_fn(), selector="all")

        cv_auc = cross_val_score(
            pipeline, X_tr, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        pipeline.fit(X_tr, y_train)
        proba = pipeline.predict_proba(X_te)[:, 1]
        metrics = classification_metrics(y_test, pipeline.predict(X_te), proba)

        rows.append(
            {
                "tier": tier_name,
                "n_features": len(available),
                "cv_roc_auc": cv_auc.mean(),
                "cv_roc_auc_std": cv_auc.std(),
                "test_roc_auc": metrics["roc_auc"],
                "test_accuracy": metrics["accuracy"],
                "test_recall": metrics["recall"],
                "test_specificity": metrics["specificity"],
                "gain_over_previous": (
                    cv_auc.mean() - previous_auc if previous_auc is not None else np.nan
                ),
            }
        )
        previous_auc = cv_auc.mean()

    return pd.DataFrame(rows)


def plot_tiered_models(tiers: pd.DataFrame) -> Path:
    """AUC per acquisition tier, with the marginal gain of each step."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    x = np.arange(len(tiers))
    axes[0].bar(x - 0.2, tiers["cv_roc_auc"], 0.4,
                yerr=tiers["cv_roc_auc_std"], capsize=4,
                label="Cross-validated", color=config.PALETTE["no_pcos"])
    axes[0].bar(x + 0.2, tiers["test_roc_auc"], 0.4,
                label="Held-out test", color=config.PALETTE["pcos"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [f"{t}\n({n} features)" for t, n in zip(tiers["tier"], tiers["n_features"])],
        fontsize=8,
    )
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("What does each tier of testing buy?", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=9)
    for xi, value in zip(x, tiers["cv_roc_auc"]):
        axes[0].text(xi - 0.2, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)

    gains = tiers.dropna(subset=["gain_over_previous"])
    axes[1].barh(gains["tier"], gains["gain_over_previous"],
                 color=[config.PALETTE["pcos"] if g > 0.01 else "#9CA3AF"
                        for g in gains["gain_over_previous"]])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Marginal CV ROC-AUC gain over the previous tier")
    axes[1].set_title("Is the extra testing worth it?", fontsize=11)
    for i, value in enumerate(gains["gain_over_previous"]):
        axes[1].text(value + 0.001, i, f"{value:+.4f}", va="center", fontsize=9)

    fig.suptitle(
        "Cost-tiered screening: questionnaire → vitals → bloods → ultrasound",
        fontsize=13,
    )
    return _save(fig, "27_cost_tiers.png")


# --------------------------------------------------------------------------
# Subgroup performance
# --------------------------------------------------------------------------
def subgroup_performance(model, X_test, y_test, min_size: int = 15) -> pd.DataFrame:
    """Performance within age and BMI bands.

    A model can look strong on average while failing a subgroup entirely.
    Groups smaller than ``min_size`` are reported but flagged, because an AUC
    computed on a dozen patients means very little.
    """
    from .evaluate import classification_metrics

    y_test = pd.Series(np.asarray(y_test), index=X_test.index)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    bands = {}
    if "Age (yrs)" in X_test.columns:
        bands["Age"] = pd.cut(
            X_test["Age (yrs)"], bins=[0, 25, 30, 35, 100],
            labels=["<25", "25-29", "30-34", "35+"],
        )
    if "BMI" in X_test.columns:
        # WHO categories.
        bands["BMI"] = pd.cut(
            X_test["BMI"], bins=[0, 18.5, 25, 30, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
        )

    rows = []
    for band_name, series in bands.items():
        for level in series.cat.categories:
            mask = (series == level).to_numpy()
            n = int(mask.sum())
            if n == 0:
                continue

            y_sub = y_test.to_numpy()[mask]
            row = {
                "grouping": band_name,
                "subgroup": str(level),
                "n": n,
                "n_pcos": int(y_sub.sum()),
                "prevalence": round(float(y_sub.mean()), 3),
                "reliable": n >= min_size and 0 < y_sub.sum() < n,
            }

            if 0 < y_sub.sum() < n:
                metrics = classification_metrics(y_sub, pred[mask], proba[mask])
                row.update({
                    "accuracy": round(metrics["accuracy"], 3),
                    "recall": round(metrics["recall"], 3),
                    "specificity": round(metrics["specificity"], 3),
                    "roc_auc": round(metrics["roc_auc"], 3),
                })
            else:
                # Only one class present: AUC is undefined.
                row.update({"accuracy": np.nan, "recall": np.nan,
                            "specificity": np.nan, "roc_auc": np.nan})
            rows.append(row)

    return pd.DataFrame(rows)


def plot_subgroups(subgroups: pd.DataFrame) -> Path:
    """AUC by subgroup, with unreliable (small-n) bars greyed out."""
    usable = subgroups.dropna(subset=["roc_auc"])
    if usable.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No subgroup had both classes present", ha="center")
        ax.axis("off")
        return _save(fig, "28_subgroup_performance.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r['grouping']}: {r['subgroup']} (n={r['n']})" for _, r in usable.iterrows()]
    colors = [
        config.PALETTE["no_pcos"] if r["reliable"] else "#C9CDD4"
        for _, r in usable.iterrows()
    ]

    ax.barh(labels, usable["roc_auc"], color=colors)
    ax.axvline(usable["roc_auc"].mean(), color=config.PALETTE["pcos"],
               linestyle="--", linewidth=1.4, label="Mean across subgroups")
    ax.set_xlim(0.4, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC within subgroup")
    ax.set_title(
        "Does the model work for everyone?\n(grey bars: fewer than 15 patients — read with caution)",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "28_subgroup_performance.png")
