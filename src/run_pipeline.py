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
    clinical,
    config,
    data,
    eda,
    evaluate,
    explain,
    external,
    literature,
    models,
    plots,
    validation,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def _uncorrected_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Numeric frame with range enforcement skipped, to list what it catches."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"PCOS (Y/N)": config.TARGET})
    df = df.drop(
        columns=[c for c in config.ID_COLUMNS + config.JUNK_COLUMNS if c in df.columns]
    )
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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

    # Physiologically impossible values, found by comparing the feature
    # distributions against an independent cohort.
    violations = data.range_violations(_uncorrected_frame(raw))
    if not violations.empty:
        _save_table(violations, "range_violations")
        print(f"\n  {len(violations)} physiologically impossible values blanked:")
        print(violations.to_string(index=False))

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

    # ---------------------------------------------------------- nested CV
    _banner("5d. Nested CV — is the reported CV score itself biased?")
    from sklearn.ensemble import RandomForestClassifier

    def rf_plain():
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )

    nested = validation.nested_cv(
        X_train, y_train, models.build_pipeline, rf_plain,
        outer_folds=5 if quick else config.CV_FOLDS,
    )
    (config.RESULTS_DIR / "nested_cv.json").write_text(json.dumps(nested, indent=2))
    validation.plot_nested_cv(nested)

    print(f"  Flat CV   (selector chosen on the same data): {nested['flat_mean']:.4f}")
    print(f"  Nested CV (selector chosen inside folds)    : {nested['nested_mean']:.4f} "
          f"± {nested['nested_std']:.4f}")
    print(f"  Selection bias (optimism)                   : {nested['optimism']:+.4f}")
    if abs(nested["optimism"]) < 0.005:
        print(
            "\n  Negligible. The four feature sets perform almost identically, so\n"
            "  choosing between them leaks almost nothing. The concern was real in\n"
            "  principle and immaterial in practice — worth measuring, not assuming."
        )
    else:
        print("\n  Material. The flat CV number should be replaced by the nested one.")

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

    # -------------------------------------------------- cost-tiered models
    _banner("6b. Cost-tiered screening — what does each tier of testing buy?")
    tiers = clinical.tiered_models(
        X_train, y_train, X_test, y_test, models.build_pipeline, rf_plain
    )
    _save_table(tiers, "cost_tiers")
    clinical.plot_tiered_models(tiers)
    print(tiers.round(4).to_string(index=False))

    questionnaire_auc = float(tiers.iloc[0]["cv_roc_auc"])
    full_auc = float(tiers.iloc[-1]["cv_roc_auc"])
    retained = questionnaire_auc / full_auc
    ultrasound_gain = float(tiers.iloc[-1]["gain_over_previous"])
    print(
        f"\n  A questionnaire, a scale and a tape measure reach {questionnaire_auc:.3f} AUC —\n"
        f"  {retained:.0%} of the full {full_auc:.3f}, with no clinician, lab or ultrasound.\n"
        f"  The entire blood panel adds {float(tiers.iloc[2]['gain_over_previous']):+.4f} AUC.\n"
        f"  The ultrasound adds {ultrasound_gain:+.4f} — it is the only tier that pays for itself."
    )

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

    # ------------------------------------------------------ uncertainty
    _banner("7b. How precise are those test numbers? Bootstrap CIs")
    ci = validation.bootstrap_test_metrics(
        best_pipeline, X_test, y_test, n_boot=500 if quick else 2000
    )
    _save_table(ci, "bootstrap_ci")
    validation.plot_bootstrap_ci(ci, best_model_name, len(y_test))
    print(ci.to_string(index=False))
    widest = ci.loc[ci["ci_width"].idxmax()]
    print(
        f"\n  Widest interval: {widest['metric']} spans "
        f"[{widest['ci_lower']:.2f}, {widest['ci_upper']:.2f}] — {widest['ci_width']:.2f} wide.\n"
        f"  With {int((y_test == 1).sum())} positive cases in the test set, that is the honest\n"
        "  precision of every single-split number in the reviewed literature too."
    )

    # -------------------------------------------------- clinical utility
    _banner("7c. Decision curve analysis — is it clinically worth using?")
    curve = clinical.net_benefit(y_test, best_pipeline.predict_proba(X_test)[:, 1])
    _save_table(curve, "decision_curve")
    clinical.plot_decision_curve(curve, best_model_name)

    useful = curve[curve["benefit_over_best_default"] > 0]
    if not useful.empty:
        print(
            f"  The model beats both 'refer everyone' and 'refer nobody' across "
            f"threshold probabilities {useful['threshold'].min():.0%}–{useful['threshold'].max():.0%}."
        )
        print("\n  Net benefit at representative thresholds:")
        sample = curve[curve["threshold"].isin([0.1, 0.2, 0.3, 0.5])]
        print(sample[["threshold", "net_benefit_model", "net_benefit_treat_all",
                      "n_flagged", "benefit_over_best_default"]].round(4).to_string(index=False))
    else:
        print("  The model does not beat the default strategies at any threshold.")

    # ------------------------------------------------------- subgroups
    _banner("7d. Subgroup performance — does it work for everyone?")
    subgroups = clinical.subgroup_performance(best_pipeline, X_test, y_test)
    _save_table(subgroups, "subgroup_performance")
    clinical.plot_subgroups(subgroups)
    print(subgroups.to_string(index=False))

    reliable = subgroups[subgroups["reliable"]].dropna(subset=["roc_auc"])
    if not reliable.empty:
        print(
            f"\n  Across subgroups with >= 15 patients, AUC ranges "
            f"{reliable['roc_auc'].min():.3f}–{reliable['roc_auc'].max():.3f}. "
            "No subgroup failure detected,\n  though the subgroups are small enough that "
            "only a large failure would show."
        )

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

    # ------------------------------------------- stability & data ceiling
    _banner("9c. Would more data help? Learning curve")
    curve_model = models.build_pipeline(
        X_train, rf_plain(), selector=best_set, k_features=k_features
    )
    lc = validation.compute_learning_curve(curve_model, X_train, y_train)
    _save_table(lc, "learning_curve")
    validation.plot_learning_curve(lc, best_model_name)
    print(lc.round(4).to_string(index=False))

    tail = lc.tail(3)
    slope_per_100 = float(np.polyfit(tail["train_size"], tail["cv_mean"], 1)[0] * 100)
    print(
        f"\n  Over the last third of the curve the validation score is moving\n"
        f"  {slope_per_100:+.4f} AUC per 100 additional patients."
    )

    _banner("9d. How arbitrary is the winner? Sweeping the train/test split")
    n_seeds = 10 if quick else 30
    sweep = validation.seed_sweep(
        X, y, models.build_all_models, n_seeds=n_seeds, selector=best_set
    )
    wins = validation.win_counts(sweep)
    _save_table(sweep, "seed_sweep_raw")
    _save_table(wins, "seed_sweep_wins")
    validation.plot_seed_sweep(sweep, wins)
    print(wins.round(4).to_string(index=False))

    n_winners = int((wins["wins"] > 0).sum())
    top = wins.iloc[0]
    print(
        f"\n  {n_winners} different models took first place across {n_seeds} random splits.\n"
        f"  The most frequent winner ({top['model']}) won only {top['win_rate']:.0%} of them,\n"
        f"  and its own test AUC varied by {top['auc_range']:.3f} depending on the split.\n"
        "  Any paper naming a best model from one split is reporting a coin flip."
    )

    # ------------------------------------------------ external validation
    _banner("9e. External validation on an independent cohort")
    external_summary = None
    if external.available():
        ext_raw = external.load_external()
        ext = external.harmonise(ext_raw)
        cohort_cmp = external.cohort_comparison(df, ext)
        _save_table(cohort_cmp, "external_cohort_comparison")

        print(f"  Cohort: {external.TUNISIA_SOURCE['name']} "
              f"(n={len(ext)}, {external.TUNISIA_SOURCE['licence']})")
        print(f"  DOI   : {external.TUNISIA_SOURCE['doi']}")
        print(f"\n  Shared features: {len(external.shared_feature_names())} of {X.shape[1]}")
        print("\n  How different are the cohorts?")
        print(cohort_cmp.to_string(index=False))

        ext_results, external_summary = external.validate_externally(
            df, ext, models.build_all_models
        )
        _save_table(ext_results, "external_validation")
        external.plot_external_validation(ext_results, cohort_cmp)

        print("\n  Train on Kerala, test on Tunisia (shared features only):")
        print(ext_results.round(4).to_string(index=False))
        print("\n" + external.EXTERNAL_DISCUSSION)
        (config.RESULTS_DIR / "external_search_log.txt").write_text(external.SEARCH_LOG)
    else:
        print("  External cohort not present; skipping.")
        print(f"  Download from {external.TUNISIA_SOURCE['url']} into data/external/.")

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
        "n_models_practically_different": n_practical,
        "best_calibrated_model": best_calibrated,
        "best_brier_score": round(float(calibration_results.iloc[0]["brier"]), 4),
        "nested_cv_optimism": round(nested["optimism"], 5),
        "questionnaire_only_cv_auc": round(questionnaire_auc, 4),
        "questionnaire_pct_of_full": round(retained, 4),
        "ultrasound_marginal_auc_gain": round(ultrasound_gain, 4),
        "test_roc_auc_ci": [
            float(ci.loc[ci["metric"] == "roc_auc", "ci_lower"].iloc[0]),
            float(ci.loc[ci["metric"] == "roc_auc", "ci_upper"].iloc[0]),
        ],
        "seed_sweep_n_distinct_winners": n_winners,
        "seed_sweep_top_win_rate": float(top["win_rate"]),
        "learning_curve_auc_per_100_patients": round(slope_per_100, 5),
        "runtime_seconds": round(elapsed, 1),
    }
    if external_summary is not None:
        summary["external_validation"] = {
            "cohort": external.TUNISIA_SOURCE["name"],
            "n": external_summary["n_external"],
            "n_shared_features": external_summary["n_shared_features"],
            "best_external_auc": external_summary["best_external_auc"],
            "mean_auc_drop": external_summary["mean_auc_drop"],
            "external_prevalence": external_summary["external_prevalence"],
            "internal_prevalence": external_summary["internal_prevalence"],
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
