"""Explainable AI (SHAP) for the selected model.

Accuracy alone does not tell a clinician anything actionable. SHAP assigns
each feature a signed contribution to each individual prediction, which
answers two different questions:

* *globally* — which measurements is the model actually relying on, and does
  that agree with the Rotterdam criteria?
* *locally* — for this particular patient, what pushed the prediction toward
  or away from PCOS?

The explainer is fitted on the model's view of the data — i.e. after
preprocessing and feature selection — so the feature names in the plots are
the ones the classifier really sees.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config


def transform_for_explainer(fitted_pipeline, X) -> pd.DataFrame:
    """Push raw features through every pipeline step except the classifier.

    SMOTE is skipped: it is a resampler, so on ``transform`` the imblearn
    pipeline passes data through it untouched, which is exactly what we want
    when explaining real patients rather than synthetic ones.
    """
    transformed = X
    for name, step in fitted_pipeline.steps[:-1]:
        if hasattr(step, "transform"):
            transformed = step.transform(transformed)
    return pd.DataFrame(transformed).reset_index(drop=True)


def build_explainer(fitted_pipeline, X_background):
    """Return a SHAP explainer appropriate to the final estimator.

    Tree ensembles get the exact TreeExplainer; anything else falls back to
    the model-agnostic (and much slower) KernelExplainer over a k-means
    summary of the background data.
    """
    classifier = fitted_pipeline.named_steps["classifier"]
    background = transform_for_explainer(fitted_pipeline, X_background)

    tree_models = ("RandomForest", "XGB", "GradientBoosting", "DecisionTree", "ExtraTrees")
    if any(name in type(classifier).__name__ for name in tree_models):
        return shap.TreeExplainer(classifier), background

    summary = shap.kmeans(background, min(50, len(background)))
    return shap.KernelExplainer(lambda d: classifier.predict_proba(d)[:, 1], summary), background


def shap_values_for(fitted_pipeline, X) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP values for ``X``, plus the transformed frame they refer to.

    For binary classifiers SHAP may return either one matrix or one per
    class depending on the model; this normalises to the positive class.
    """
    explainer, _ = build_explainer(fitted_pipeline, X)
    data = transform_for_explainer(fitted_pipeline, X)

    values = explainer.shap_values(data)
    values = np.asarray(values)

    # Shape can be (n, features), (n, features, 2) or (2, n, features).
    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        else:
            values = values[1]
    return values, data


def _pretty_columns(data: pd.DataFrame) -> pd.DataFrame:
    renamed = data.copy()
    renamed.columns = [config.pretty(str(c)) for c in renamed.columns]
    return renamed


def _save(fig, name: str) -> Path:
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_global_importance(shap_values, data: pd.DataFrame, model_name: str) -> Path:
    """Mean absolute SHAP value per feature — the global ranking."""
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, _pretty_columns(data), plot_type="bar", show=False, max_display=15
    )
    fig = plt.gcf()
    fig.suptitle(f"Global feature importance — {model_name}", fontsize=12, y=1.01)
    return _save(fig, "16_shap_global_importance.png")


def plot_beeswarm(shap_values, data: pd.DataFrame, model_name: str) -> Path:
    """Beeswarm: importance *and* the direction of each feature's effect."""
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, _pretty_columns(data), show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle(f"SHAP value distribution — {model_name}", fontsize=12, y=1.01)
    return _save(fig, "17_shap_beeswarm.png")


def plot_individual_explanations(
    fitted_pipeline, X_test, y_test, shap_values, data: pd.DataFrame, n: int = 2
) -> list[Path]:
    """Waterfall plots for a few individual patients.

    One correctly identified PCOS case and one correctly cleared case, so the
    write-up can show the model's reasoning in both directions.
    """
    proba = fitted_pipeline.predict_proba(X_test)[:, 1]
    y_test = np.asarray(y_test)

    explainer, _ = build_explainer(fitted_pipeline, X_test)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).ravel()[-1]

    pretty_data = _pretty_columns(data)
    paths = []

    # Most confident correct positive, and most confident correct negative.
    correct_pos = np.where((y_test == 1) & (proba >= 0.5))[0]
    correct_neg = np.where((y_test == 0) & (proba < 0.5))[0]
    picks = []
    if len(correct_pos):
        picks.append(("pcos_case", correct_pos[np.argmax(proba[correct_pos])]))
    if len(correct_neg):
        picks.append(("no_pcos_case", correct_neg[np.argmin(proba[correct_neg])]))

    for label, idx in picks[:n]:
        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=base_value,
            data=pretty_data.iloc[idx].values,
            feature_names=list(pretty_data.columns),
        )
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        fig = plt.gcf()
        fig.suptitle(
            f"Individual prediction — {label.replace('_', ' ')} "
            f"(P(PCOS) = {proba[idx]:.2f})",
            fontsize=11,
            y=1.01,
        )
        paths.append(_save(fig, f"18_shap_waterfall_{label}.png"))

    return paths


def importance_table(shap_values, data: pd.DataFrame) -> pd.DataFrame:
    """Global SHAP importance as a sortable table for the report.

    ``mean_abs_shap`` ranks features by how much they move predictions.

    Direction needs more care than the mean signed SHAP value, which is
    dominated by the class balance: on a majority-negative test set almost
    every feature has a negative mean, which says nothing about the feature.
    Instead ``value_shap_corr`` correlates each feature's *value* with its
    own SHAP contribution. A positive correlation means higher values of the
    feature push the prediction toward PCOS.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    directions = []
    for i, col in enumerate(data.columns):
        values = pd.Series(np.asarray(data[col], dtype=float))
        contributions = pd.Series(shap_values[:, i])
        # Constant features (or constant contributions) have no direction.
        if values.nunique() <= 1 or contributions.nunique() <= 1:
            directions.append(np.nan)
        else:
            directions.append(values.corr(contributions))

    table = pd.DataFrame(
        {
            "feature": [config.pretty(str(c)) for c in data.columns],
            "mean_abs_shap": mean_abs,
            "value_shap_corr": directions,
        }
    )
    table["higher_value_means"] = np.where(
        table["value_shap_corr"].isna(),
        "n/a",
        np.where(table["value_shap_corr"] > 0, "more likely PCOS", "less likely PCOS"),
    )
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
