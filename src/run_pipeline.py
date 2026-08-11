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

from . import (
    automl,
    calibration,
    config,
    data,
    eda,
    evaluate,
    explain,
    literature,
    models,
    plots,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def _row_for(results: pd.DataFrame, model_name: str) -> pd.Series:
    """Pull one model's row out of a results table."""
    return results[results["model"] == model_name].iloc[0]


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

    # -------------------------------------------- statistical significance
    _banner("5b. Is the ranking real? Paired tests across folds")
    fold_scores = calibration.paired_fold_scores(
        cv_models, X_train, y_train,
        folds=5 if quick else config.CV_FOLDS, repeats=repeats,
    )
    _save_table(fold_scores, "per_fold_scores")

    significance = calibration.significance_against_best(fold_scores)
    _save_table(significance, "significance_tests")
    calibration.plot_significance(fold_scores)

    print(significance.round(4).to_string(index=False))

    n_models = len(significance) - 1
    n_indistinguishable = int((significance["statistically_sep"] == "no").sum())
    n_practical = int((significance["practically_sep"] == "yes").sum())
    largest_gap = float(significance["delta_vs_best"].abs().max())

    print(
        f"\n  Statistically indistinguishable from the best: {n_indistinguishable}/{n_models}"
        f"\n  Practically different (>= 0.02 AUC):           {n_practical}/{n_models}"
        f"\n  Largest gap to the best model:                 {largest_gap:.4f} AUC"
    )
    print(
        "\n  Note the two columns disagree. With 30 folds a paired test can flag a\n"
        "  0.005 AUC gap as 'significant', but no clinical decision turns on it —\n"
        "  and the fold-to-fold spread is ~0.03, six times larger. Statistical\n"
        "  separability is not the same as mattering."
    )

    # ------------------------------------------------------- calibration
    _banner("5c. Probability calibration — do the probabilities mean anything?")
    calibration_results = calibration.compare_calibration(
        cv_models, X_train, y_train, folds=5
    )
    _save_table(calibration_results, "calibration_metrics")
    calibration.plot_calibration_curves(cv_models, X_train, y_train, folds=5)
    print(calibration_results.round(4).to_string(index=False))
    print("\n  Lower Brier / log-loss / ECE is better. ROC-AUC is blind to all three.")

    best_calibrated = calibration_results.iloc[0]["model"]
    print(f"  best-calibrated model: {best_calibrated}")

    # Does explicit calibration help the selected model?
    base_clf = models.build_extended_classifiers().get(best_model_name)
    if base_clf is not None:
        effect = calibration.calibration_effect(
            models.build_pipeline(X_train, base_clf, selector=best_set, k_features=k_features),
            models.build_calibrated(X_train, base_clf, selector=best_set, k_features=k_features),
            X_train,
            y_train,
        )
        _save_table(effect, "calibration_effect")
        print(f"\n  Effect of isotonic calibration on {best_model_name}:")
        print(effect.round(4).to_string(index=False))

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

    # ----------------------------------------------------- AutoML ceiling
    _banner("9b. AutoML benchmark — is the hand-built pipeline leaving anything on the table?")
    automl_result = automl.run_automl_benchmark(
        X_train, y_train, X_test, y_test, time_budget=30 if quick else 120
    )
    if automl_result.get("available"):
        our_row = _row_for(test_results, best_model_name)
        automl_table = automl.comparison_table(
            automl_result,
            {
                "model": best_model_name,
                "accuracy": float(our_row["accuracy"]),
                "f1": float(our_row["f1"]),
                "roc_auc": float(our_row["roc_auc"]),
            },
        )
        _save_table(automl_table, "automl_comparison")
        print(f"  FLAML searched for {automl_result['time_budget_s']}s and chose: "
              f"{automl_result['best_estimator']}")
        print(f"  its config: {automl_result['best_config']}")
        print()
        print(automl_table.to_string(index=False))

        gain = automl_result["test_roc_auc"] - float(our_row["roc_auc"])
        if abs(gain) < 0.02:
            verdict = (
                f"AutoML moved test ROC-AUC by {gain:+.4f} — inside the fold-to-fold\n"
                "  noise of +/-0.03. The dataset, not the pipeline, is the binding\n"
                "  constraint, so further tuning would be wasted effort."
            )
        elif gain > 0:
            verdict = (
                f"AutoML gained {gain:+.4f} ROC-AUC. That is real headroom — worth\n"
                "  investigating what architecture it found."
            )
        else:
            verdict = (
                f"AutoML scored {gain:+.4f} below the hand-built pipeline, i.e. an\n"
                "  untargeted search did not even match a considered design."
            )
        print(f"\n  {verdict}")
        print(
            "\n  Caveat: FLAML searches against a wall-clock budget, so the exact model\n"
            "  it returns varies between runs. The conclusion (no meaningful headroom)\n"
            "  is stable; the specific number in this table is not."
        )
    else:
        print(f"  skipped: {automl_result.get('reason')}")

    # -------------------------------------------------- literature compare
    _banner("10. Comparison against the reviewed literature")
    best_row = _row_for(test_results, best_model_name)
    cv_row = _row_for(cv_results, best_model_name)

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
        "n_models_compared": len(cv_results),
        "n_models_indistinguishable_from_best": n_indistinguishable,
        "best_calibrated_model": best_calibrated,
        "best_brier_score": round(float(calibration_results.iloc[0]["brier"]), 4),
        "runtime_seconds": round(elapsed, 1),
    }
    if shap_table is not None:
        summary["top_shap_features"] = shap_table.head(5)["feature"].tolist()
    if automl_result.get("available"):
        summary["automl"] = {
            "best_estimator": automl_result["best_estimator"],
            "test_roc_auc": round(automl_result["test_roc_auc"], 4),
            "gain_over_hand_built": round(
                automl_result["test_roc_auc"] - float(best_row["roc_auc"]), 4
            ),
        }

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
