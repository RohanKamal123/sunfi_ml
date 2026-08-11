"""The five reviewed papers, as data, so results can be compared honestly.

Numbers here are the figures the papers themselves report. They are NOT
directly comparable to ours and the ``comparable`` flag records why: a score
obtained from a single train/test split on 541 rows, with the resampling
done before splitting, measures something different from a repeated
cross-validated score with resampling inside the folds.
"""

from __future__ import annotations

import pandas as pd

REVIEWED_PAPERS = [
    {
        "paper": "Paper 1 — PCOcare (Thakre et al., 2020)",
        "best_model": "Random Forest",
        "dataset": "Kaggle PCOS (541)",
        "reported_accuracy": 0.909,
        "reported_auc": None,
        "n_samples": 541,
        "external_validation": False,
        "imbalance_handled": "not stated",
        "validation_protocol": "single train/test split",
        "explainability": False,
    },
    {
        "paper": "Paper 2 — EHR study (Zad et al., 2024)",
        "best_model": "Gradient Boosted Trees / MLP",
        "dataset": "Boston Medical Center EHR (30,601)",
        "reported_accuracy": None,
        "reported_auc": 0.825,  # midpoint of the reported 0.80-0.85 range
        "n_samples": 30601,
        "external_validation": False,
        "imbalance_handled": "not stated",
        "validation_protocol": "held-out split on large EHR cohort",
        "explainability": False,
    },
    {
        "paper": "Paper 3 — Bhat (2021)",
        "best_model": "CatBoost",
        "dataset": "Kaggle PCOS (541)",
        "reported_accuracy": 0.957,
        "reported_auc": None,
        "n_samples": 541,
        "external_validation": False,
        "imbalance_handled": "not stated",
        "validation_protocol": "single train/test split",
        "explainability": False,
    },
    {
        "paper": "Paper 4 — Traditional ML + DL (2024)",
        "best_model": "Stacking Classifier",
        "dataset": "Kaggle PCOS (541)",
        "reported_accuracy": 0.9932,
        "reported_auc": None,
        "n_samples": 541,
        "external_validation": False,
        "imbalance_handled": "not stated",
        "validation_protocol": "single train/test split",
        "explainability": False,
    },
    {
        "paper": "Paper 5 — RFE + XAI (Elmannai et al., 2023)",
        "best_model": "Stacking + RFE",
        "dataset": "Kaggle PCOS (541)",
        "reported_accuracy": 0.9887,
        "reported_auc": None,
        "n_samples": 541,
        "external_validation": False,
        "imbalance_handled": "not stated",
        "validation_protocol": "single train/test split",
        "explainability": True,
    },
]


def comparison_table(
    our_accuracy: float,
    our_auc: float,
    our_model: str,
    our_accuracy_std: float | None = None,
) -> pd.DataFrame:
    """Build the literature comparison table including this project's result."""
    rows = [dict(p) for p in REVIEWED_PAPERS]
    for row in rows:
        row["is_ours"] = False
        row["label"] = row["paper"].split(" — ")[0] + f" ({row['best_model']})"

    ours = {
        "paper": "This project",
        "best_model": our_model,
        "dataset": "Kaggle PCOS (541)",
        "reported_accuracy": our_accuracy,
        "reported_auc": our_auc,
        "n_samples": 541,
        "external_validation": False,
        "imbalance_handled": "SMOTE inside CV folds",
        "validation_protocol": (
            "repeated stratified 10-fold CV + untouched held-out test set"
        ),
        "explainability": True,
        "is_ours": True,
        "label": f"This project ({our_model})",
    }
    if our_accuracy_std is not None:
        ours["accuracy_std"] = our_accuracy_std

    rows.append(ours)
    df = pd.DataFrame(rows)

    # Papers reporting only AUC have no accuracy to plot; fill so the figure
    # still renders, and flag it in the table.
    df["accuracy_is_proxy"] = df["reported_accuracy"].isna()
    df["reported_accuracy"] = df["reported_accuracy"].fillna(df["reported_auc"])
    return df


DISCUSSION = """
Reading the comparison table
----------------------------

Papers 4 and 5 report 98.9-99.3% accuracy on the same 541-row dataset used
here. This project reports a lower number, and that gap is the most
important result in the report -- it is a methodology difference, not a
modelling failure.

Three things separate the protocols:

1. **Where the resampling happens.** If SMOTE runs before the train/test
   split, synthetic minority rows interpolated from training patients end up
   in the evaluation set. The model is then partly scored on points derived
   from data it trained on. The leakage experiment in this project measures
   that effect directly on the same dataset and the same classifier.

2. **Where feature selection happens.** Running RFE once on the full dataset
   lets the selector read every label, including the test labels, before the
   split. The chosen feature set is then tuned to the test set.

3. **How many splits.** A single 80/20 split of 541 rows puts about 35 PCOS
   cases in the test set. Three patients changing sides moves accuracy by
   roughly 3 points, so a single split can produce a very high number by
   chance. Repeated 10-fold CV averages that variance away.

None of this proves the reviewed papers made these mistakes -- most simply do
not report enough detail to tell, which is itself the finding. The honest
claim this project can make is narrower and better supported: under a
protocol where nothing downstream of the split can see the evaluation data,
this is the performance the Kaggle PCOS dataset supports.
""".strip()
