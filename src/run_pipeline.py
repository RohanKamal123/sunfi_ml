"""End-to-end pipeline runner.

    python -m src.run_pipeline              # full run
    python -m src.run_pipeline --quick      # fewer CV repeats, skip SHAP

Writes every figure to ``reports/figures`` and every table to
``reports/results``, then prints a summary.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from . import config, data, eda, evaluate, explain, literature, models, plots

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def _save_table(df: pd.DataFrame, name: str) -> None:
    config.ensure_dirs()
    path = config.RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  saved -> {path.relative_to(config.PROJECT_ROOT)}")


def main(quick: bool = False, k_features: int = 15) -> dict:
    start = time.time()
    config.ensure_dirs()
    repeats = 1 if quick else config.CV_REPEATS

    # ---------------------------------------------------------------- data
    _banner("1. Loading and cleaning data")
    raw = data.load_raw()
    df = data.clean(raw)
    X, y = data.split_xy(df)

    print(f"  raw shape      : {raw.shape}")
    print(f"  cleaned shape  : {df.shape}  ({X.shape[1]} features)")
    print(f"  class balance  : {int((y == 0).sum())} no-PCOS / {int((y == 1).sum())} PCOS "
          f"({(y == 0).sum() / (y == 1).sum():.2f}:1)")
    print(f"  missing values : {int(df.isna().sum().sum())} across "
          f"{int((df.isna().sum() > 0).sum())} columns")

    quality = data.data_quality_report(raw, df)
    _save_table(quality, "data_quality_report")
    df.to_csv(config.PROCESSED_DIR / "pcos_clean.csv", index=False)

    # ----------------------------------------------------------------- EDA
    _banner("2. Exploratory data analysis")
    figures = eda.run_all(df)
    for path in figures:
        print(f"  figure -> {path.relative_to(config.PROJECT_ROOT)}")
    stats = eda.summary_statistics(df)
    _save_table(stats, "summary_statistics")
    print("\n  Largest class differences (Cohen's d):")
    print(stats.head(8).to_string(index=False))

    # --------------------------------------------------------------- split
    _banner("3. Train / test split")
    X_train, X_test, y_train, y_test = evaluate.make_splits(X, y)
    print(f"  train: {X_train.shape[0]} rows ({int((y_train == 1).sum())} PCOS)")
    print(f"  test : {X_test.shape[0]} rows ({int((y_test == 1).sum())} PCOS)")
    print("  The test split is not touched again until step 7.")

    # --------------------------------------------------- feature selection
    _banner("4. Feature selection comparison (cross-validated)")
    feature_comparison = evaluate.compare_feature_sets(
        models.build_all_models,
        X_train,
        y_train,
        k_features=k_features,
        folds=5 if quick else config.CV_FOLDS,
        repeats=1,
    )
    _save_table(feature_comparison, "feature_set_comparison")
    plots.plot_feature_set_comparison(feature_comparison)

    best_set = (
        feature_comparison.groupby("feature_set")["roc_auc"].mean().idxmax()
    )
    print(feature_comparison.groupby("feature_set")["roc_auc"]
          .agg(["mean", "max"]).round(4).to_string())
    print(f"\n  best feature set by mean ROC-AUC: {best_set!r}")

    # Record which features each strategy actually keeps.
    selection_detail = {}
    preprocessor = models.build_preprocessor(X_train)
    X_prep = preprocessor.fit_transform(X_train, y_train)
    from .features import build_selector

    for name in ("correlation", "chi2", "rfe"):
        selector = build_selector(name, k=k_features).fit(X_prep, y_train)
        selection_detail[name] = [str(c) for c in selector.selected_features_]
        print(f"  {name:12s} kept {len(selection_detail[name]):2d} features")
    (config.RESULTS_DIR / "selected_features.json").write_text(
        json.dumps(selection_detail, indent=2)
    )

    # ------------------------------------------------------ model training
    _banner(f"5. Model comparison — repeated stratified CV on {best_set!r} features")
    cv_models = models.build_all_models(X_train, selector=best_set, k_features=k_features)
    cv_results = evaluate.cross_validate_models(
        cv_models, X_train, y_train, folds=5 if quick else config.CV_FOLDS, repeats=repeats
    )
    _save_table(cv_results, "cv_results")
    plots.plot_cv_comparison(cv_results)

    display_cols = ["model", "accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std"]
    print(cv_results[display_cols].round(4).to_string(index=False))

    best_model_name = cv_results.iloc[0]["model"]
    print(f"\n  best model by CV ROC-AUC: {best_model_name}")

    # ------------------------------------------------- imbalance ablation
    _banner("6. Class-imbalance ablation")
    from sklearn.ensemble import RandomForestClassifier

    def rf(**kwargs):
        return RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            random_state=config.RANDOM_STATE, n_jobs=-1, **kwargs
        )

    ablation = evaluate.balance_ablation(
        X_train, y_train, X_test, y_test, rf, k_features=k_features
    )
    _save_table(ablation, "imbalance_ablation")
    print(ablation[["strategy", "accuracy", "precision", "recall", "f1",
                    "specificity", "roc_auc"]].round(4).to_string(index=False))

    # ------------------------------------------------------- held-out test
    _banner("7. Held-out test set evaluation")
    test_models = models.build_all_models(X_train, selector=best_set, k_features=k_features)
    test_results = evaluate.evaluate_on_test(test_models, X_train, y_train, X_test, y_test)
    _save_table(test_results, "test_results")

    print(test_results[["model", "accuracy", "precision", "recall", "f1",
                        "specificity", "roc_auc"]].round(4).to_string(index=False))

    curves = evaluate.curve_data(test_models, X_test, y_test)
    plots.plot_roc_curves(curves)
    plots.plot_pr_curves(curves)

    best_pipeline = test_models[best_model_name]
    y_pred = best_pipeline.predict(X_test)
    plots.plot_confusion_matrix(y_test, y_pred, best_model_name)

    sweep = evaluate.threshold_sweep(best_pipeline, X_test, y_test)
    _save_table(sweep, "threshold_sweep")
    plots.plot_threshold_sweep(sweep)

    joblib.dump(best_pipeline, config.MODELS_DIR / "best_model.joblib")
    print(f"\n  model saved -> models/best_model.joblib")

    # ---------------------------------------------------------- leakage
    _banner("8. Leakage experiment — how much does a leaky protocol inflate results?")
    leakage = evaluate.leakage_experiment(X, y, rf, folds=5 if quick else config.CV_FOLDS,
                                          k_features=k_features)
    _save_table(leakage, "leakage_experiment")
    plots.plot_leakage_experiment(leakage)
    print(leakage.round(4).to_string(index=False))

    # ------------------------------------------------------------- SHAP
    shap_table = None
    if not quick:
        _banner("9. Explainability (SHAP)")
        # Explain a tree model: TreeExplainer is exact and fast, whereas
        # KernelExplainer on the stacking ensemble takes minutes.
        explain_name = next(
            (n for n in ("Random Forest", "XGBoost") if n in test_models), best_model_name
        )
        explain_pipeline = test_models[explain_name]
        print(f"  explaining: {explain_name}")

        shap_values, shap_data = explain.shap_values_for(explain_pipeline, X_test)
        explain.plot_global_importance(shap_values, shap_data, explain_name)
        explain.plot_beeswarm(shap_values, shap_data, explain_name)
        explain.plot_individual_explanations(
            explain_pipeline, X_test, y_test, shap_values, shap_data
        )

        shap_table = explain.importance_table(shap_values, shap_data)
        _save_table(shap_table, "shap_importance")
        print("\n  Top 10 features by mean |SHAP|:")
        print(shap_table.head(10).round(4).to_string(index=False))
    else:
        _banner("9. Explainability (SHAP) — skipped in --quick mode")

    # -------------------------------------------------- literature compare
    _banner("10. Comparison against the reviewed literature")
    best_row = test_results[test_results["model"] == best_model_name].iloc[0]
    cv_row = cv_results[cv_results["model"] == best_model_name].iloc[0]

    comparison = literature.comparison_table(
        our_accuracy=float(best_row["accuracy"]),
        our_auc=float(best_row["roc_auc"]),
        our_model=best_model_name,
        our_accuracy_std=float(cv_row["accuracy_std"]),
    )
    _save_table(comparison, "literature_comparison")
    plots.plot_paper_comparison(comparison)

    print(comparison[["paper", "best_model", "reported_accuracy",
                      "validation_protocol", "explainability"]].to_string(index=False))
    print("\n" + literature.DISCUSSION)

    # ------------------------------------------------------------ summary
    elapsed = time.time() - start
    summary = {
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "class_balance": {"no_pcos": int((y == 0).sum()), "pcos": int((y == 1).sum())},
        "best_feature_set": best_set,
        "n_features_selected": len(selection_detail.get(best_set, X.columns)),
        "best_model": best_model_name,
        "cv_roc_auc": round(float(cv_row["roc_auc"]), 4),
        "cv_roc_auc_std": round(float(cv_row["roc_auc_std"]), 4),
        "cv_accuracy": round(float(cv_row["accuracy"]), 4),
        "cv_accuracy_std": round(float(cv_row["accuracy_std"]), 4),
        "test_accuracy": round(float(best_row["accuracy"]), 4),
        "test_precision": round(float(best_row["precision"]), 4),
        "test_recall": round(float(best_row["recall"]), 4),
        "test_f1": round(float(best_row["f1"]), 4),
        "test_specificity": round(float(best_row["specificity"]), 4),
        "test_roc_auc": round(float(best_row["roc_auc"]), 4),
        "leakage_accuracy_inflation": round(
            float(leakage.iloc[2]["accuracy"]), 4
        ),
        "runtime_seconds": round(elapsed, 1),
    }
    if shap_table is not None:
        summary["top_shap_features"] = shap_table.head(5)["feature"].tolist()

    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    _banner("Done")
    print(json.dumps(summary, indent=2))
    print(f"\nFigures : {config.FIGURES_DIR.relative_to(config.PROJECT_ROOT)}")
    print(f"Tables  : {config.RESULTS_DIR.relative_to(config.PROJECT_ROOT)}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PCOS prediction pipeline.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer folds and repeats, and skip SHAP. For a fast sanity check.",
    )
    parser.add_argument(
        "--k-features",
        type=int,
        default=15,
        help="Number of features kept by the chi2 and RFE selectors (default: 15).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(quick=args.quick, k_features=args.k_features)
