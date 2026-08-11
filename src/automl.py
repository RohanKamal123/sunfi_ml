"""AutoML benchmark: is the hand-built pipeline near-optimal?

The model comparison answers "which of these algorithms is best". It does not
answer the more useful question: **is this whole approach leaving performance
on the table?** A hand-built pipeline could be losing several points to a
better architecture or better hyperparameters and the comparison would never
reveal it, because every entry shares the same handicap.

An AutoML search is the cheapest way to find out. FLAML (MIT licence,
Microsoft Research) searches over algorithms and hyperparameters under a time
budget. If it beats the hand-built pipeline substantially, there is headroom;
if it lands in the same place, the hand-built pipeline is at the dataset's
ceiling and further tuning is wasted effort.

The critical detail: FLAML is given the **training split only**, and is scored
by the same protocol as everything else. An AutoML tool handed the full
dataset would tune itself against the test set, which is the same leak the
project is built to avoid — just automated.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from . import config


def _sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip characters LightGBM refuses in feature names.

    Our column names carry clinical units -- "FSH(mIU/mL)", "Waist:Hip Ratio"
    -- and LightGBM rejects any name containing JSON-special characters. FLAML
    may pick LightGBM, so the names are flattened before the search. Only the
    AutoML branch needs this; the hand-built pipeline keeps the readable names
    for its SHAP plots.
    """
    renamed = df.copy()
    renamed.columns = [
        re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_") for c in renamed.columns
    ]
    # Flattening can collide ("BP _Systolic (mmHg)" style pairs); suffix dupes.
    seen: dict[str, int] = {}
    unique = []
    for name in renamed.columns:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    renamed.columns = unique
    return renamed

try:
    from flaml import AutoML

    HAS_FLAML = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_FLAML = False


def run_automl_benchmark(
    X_train,
    y_train,
    X_test,
    y_test,
    time_budget: int = 120,
    metric: str = "roc_auc",
) -> dict:
    """Search for a better pipeline under a time budget, then score it once.

    Parameters
    ----------
    time_budget:
        Seconds of search. On 432 rows even a short budget explores a lot;
        the returns flatten quickly because the dataset, not the search, is
        the binding constraint.

    Returns
    -------
    dict with the winning configuration and its held-out test scores, or a
    ``{"available": False}`` marker if FLAML is not installed.
    """
    if not HAS_FLAML:
        return {"available": False, "reason": "flaml is not installed"}

    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # FLAML handles encoding and model choice, but not missing values in the
    # way our pipeline does, so impute first -- fitted on train only.
    imputer = SimpleImputer(strategy="median").fit(X_train)
    X_train_imputed = _sanitise_columns(
        pd.DataFrame(
            imputer.transform(X_train), columns=X_train.columns, index=X_train.index
        )
    )
    X_test_imputed = _sanitise_columns(
        pd.DataFrame(
            imputer.transform(X_test), columns=X_test.columns, index=X_test.index
        )
    )

    automl = AutoML()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        automl.fit(
            X_train=X_train_imputed,
            y_train=np.asarray(y_train),
            task="classification",
            metric=metric,
            time_budget=time_budget,
            eval_method="cv",
            n_splits=5,
            split_type="stratified",
            seed=config.RANDOM_STATE,
            verbose=0,
            early_stop=True,
        )

    y_pred = automl.predict(X_test_imputed)
    y_proba = automl.predict_proba(X_test_imputed)[:, 1]

    return {
        "available": True,
        "best_estimator": str(automl.best_estimator),
        "best_config": {k: _plain(v) for k, v in (automl.best_config or {}).items()},
        "cv_score_on_train": float(1 - automl.best_loss) if metric == "roc_auc" else None,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "time_budget_s": time_budget,
    }


def _plain(value):
    """Make numpy scalars JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def comparison_table(automl_result: dict, our_result: dict) -> pd.DataFrame:
    """Side-by-side of the hand-built pipeline against the AutoML search."""
    if not automl_result.get("available"):
        return pd.DataFrame(
            [{"approach": "AutoML (FLAML)", "note": automl_result.get("reason", "unavailable")}]
        )

    rows = [
        {
            "approach": f"Hand-built pipeline ({our_result['model']})",
            "test_accuracy": our_result["accuracy"],
            "test_f1": our_result["f1"],
            "test_roc_auc": our_result["roc_auc"],
            "interpretable": "yes (SHAP + fixed feature set)",
            "search_cost": "none",
        },
        {
            "approach": f"AutoML / FLAML ({automl_result['best_estimator']})",
            "test_accuracy": automl_result["test_accuracy"],
            "test_f1": automl_result["test_f1"],
            "test_roc_auc": automl_result["test_roc_auc"],
            "interpretable": "harder (tuned black box)",
            "search_cost": f"{automl_result['time_budget_s']}s search",
        },
    ]

    table = pd.DataFrame(rows)
    delta = {
        "approach": "difference (AutoML - hand-built)",
        "test_accuracy": rows[1]["test_accuracy"] - rows[0]["test_accuracy"],
        "test_f1": rows[1]["test_f1"] - rows[0]["test_f1"],
        "test_roc_auc": rows[1]["test_roc_auc"] - rows[0]["test_roc_auc"],
        "interpretable": "",
        "search_cost": "",
    }
    return pd.concat([table, pd.DataFrame([delta])], ignore_index=True)
