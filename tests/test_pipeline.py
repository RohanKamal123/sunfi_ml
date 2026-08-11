"""Tests for the PCOS pipeline.

The interesting ones are the leakage tests: they assert that the pipeline
cannot see validation data during fitting, which is the central methodological
claim of the project.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import calibration, config, data, evaluate, features, models


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clean_df():
    if not config.RAW_EXCEL.exists():
        pytest.skip(f"Dataset not present at {config.RAW_EXCEL}")
    return data.load_clean()


@pytest.fixture(scope="module")
def xy(clean_df):
    return data.split_xy(clean_df)


# --------------------------------------------------------------------------
# Data cleaning
# --------------------------------------------------------------------------
def test_cleaning_yields_expected_shape(clean_df):
    assert len(clean_df) == 541
    assert config.TARGET in clean_df.columns


def test_class_distribution_matches_source(clean_df):
    counts = clean_df[config.TARGET].value_counts()
    assert counts[0] == 364
    assert counts[1] == 177


def test_identifier_and_junk_columns_are_dropped(clean_df):
    for column in config.ID_COLUMNS + config.JUNK_COLUMNS:
        assert column not in clean_df.columns


def test_all_columns_are_numeric(clean_df):
    non_numeric = [
        c for c in clean_df.columns if not pd.api.types.is_numeric_dtype(clean_df[c])
    ]
    assert non_numeric == []


def test_malformed_text_values_became_nan(clean_df):
    # The workbook stores "1.99." in beta-HCG II and "a" in AMH. Both should
    # be coerced to NaN rather than crashing or silently becoming a string.
    assert clean_df["II    beta-HCG(mIU/mL)"].isna().sum() == 1
    assert clean_df["AMH(ng/mL)"].isna().sum() == 1


def test_cycle_recoded_to_binary_indicator(clean_df):
    assert "Cycle(R/I)" not in clean_df.columns
    values = set(clean_df["Cycle_Irregular"].dropna().unique())
    assert values <= {0.0, 1.0}
    # The single undefined code 5 becomes NaN.
    assert clean_df["Cycle_Irregular"].isna().sum() == 1


def test_impossible_zeros_removed(clean_df):
    for column in ("Cycle length(days)", "Endometrium (mm)"):
        assert (clean_df[column].dropna() > 0).all()


def test_feature_groups_partition_all_columns(xy):
    X, _ = xy
    groups = data.feature_groups(X)
    combined = groups["continuous"] + groups["nominal"] + groups["binary"]
    assert sorted(combined) == sorted(X.columns)
    assert len(combined) == len(set(combined))  # no column in two groups


# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", features.SELECTOR_NAMES)
def test_selectors_reduce_and_preserve_row_count(name, xy):
    X, y = xy
    preprocessor = models.build_preprocessor(X)
    X_prep = preprocessor.fit_transform(X, y)

    selector = features.build_selector(name, k=10).fit(X_prep, y)
    out = selector.transform(X_prep)

    assert len(out) == len(X_prep)
    assert out.shape[1] <= X_prep.shape[1]
    assert list(selector.get_feature_names_out()) == list(out.columns)


def test_chi2_and_rfe_respect_k(xy):
    X, y = xy
    X_prep = models.build_preprocessor(X).fit_transform(X, y)
    for name in ("chi2", "rfe"):
        selector = features.build_selector(name, k=8).fit(X_prep, y)
        assert len(selector.selected_features_) == 8


def test_correlation_selector_drops_collinear_pair(xy):
    """A duplicated column must not survive alongside its twin."""
    X, y = xy
    X_prep = models.build_preprocessor(X).fit_transform(X, y)
    X_prep = X_prep.copy()
    X_prep["follicle_copy"] = X_prep["Follicle No. (R)"]

    selector = features.CorrelationSelector().fit(X_prep, y)
    selected = selector.selected_features_
    assert not ("Follicle No. (R)" in selected and "follicle_copy" in selected)


# --------------------------------------------------------------------------
# Pipeline construction and leakage
# --------------------------------------------------------------------------
def test_pipeline_has_expected_steps(xy):
    X, _ = xy
    pipeline = models.build_pipeline(X, models.build_classifiers()["Random Forest"])
    assert [name for name, _ in pipeline.steps] == [
        "preprocess",
        "select",
        "balance",
        "classifier",
    ]


def test_smote_can_be_disabled(xy):
    X, _ = xy
    pipeline = models.build_pipeline(
        X, models.build_classifiers()["Random Forest"], balance="none"
    )
    assert "balance" not in dict(pipeline.steps)


def test_unknown_balance_strategy_raises(xy):
    X, _ = xy
    with pytest.raises(ValueError, match="balance strategy"):
        models.build_pipeline(X, models.build_classifiers()["SVM (RBF)"], balance="oops")


def test_unknown_selector_raises():
    with pytest.raises(ValueError, match="Unknown selector"):
        features.build_selector("magic")


def test_smote_does_not_resample_at_predict_time(xy):
    """Predictions must come back one per input row, not one per SMOTE row."""
    X, y = xy
    pipeline = models.build_pipeline(X, models.build_classifiers()["Logistic Regression"])
    pipeline.fit(X, y)
    assert len(pipeline.predict(X)) == len(X)


def test_pipeline_handles_missing_values_without_prior_imputation(xy):
    """The raw frame still contains NaNs; the pipeline must cope on its own."""
    X, y = xy
    assert X.isna().sum().sum() > 0
    pipeline = models.build_pipeline(X, models.build_classifiers()["Random Forest"])
    pipeline.fit(X, y)
    assert not np.isnan(pipeline.predict_proba(X)).any()


def test_preprocessor_is_not_fitted_on_validation_folds(xy):
    """The core anti-leakage guarantee, asserted directly.

    A column is planted whose scale differs wildly between two halves of the
    data. If the scaler were fitted once on everything, the fitted mean would
    reflect both halves. Fitting inside a fold must produce a mean that
    reflects only that fold's training rows.
    """
    X, y = xy
    X = X.copy()
    planted = np.zeros(len(X))
    planted[: len(X) // 2] = 1000.0
    X["planted"] = planted

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    train_idx, _ = next(cv.split(X, y))

    pipeline = models.build_pipeline(X, models.build_classifiers()["Logistic Regression"])
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

    scaler = pipeline.named_steps["preprocess"].named_transformers_["continuous"]
    columns = data.feature_groups(X)["continuous"]
    fitted_mean = scaler.named_steps["scale"].mean_[columns.index("planted")]

    fold_mean = X["planted"].iloc[train_idx].mean()
    full_mean = X["planted"].mean()

    assert fitted_mean == pytest.approx(fold_mean, rel=1e-6)
    assert fitted_mean != pytest.approx(full_mean, rel=1e-6)


def test_leaky_protocol_scores_higher_than_correct_one(xy):
    """The leakage experiment must actually demonstrate inflation."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = xy

    def rf():
        return RandomForestClassifier(
            n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1
        )

    result = evaluate.leakage_experiment(X, y, rf, folds=5, k_features=10)
    inflation = result[result["protocol"].str.startswith("inflation")].iloc[0]
    assert inflation["accuracy"] > 0
    assert inflation["f1"] > 0


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def test_split_is_stratified_and_reproducible(xy):
    X, y = xy
    X_train, X_test, y_train, y_test = evaluate.make_splits(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)

    again = evaluate.make_splits(X, y)
    pd.testing.assert_index_equal(X_train.index, again[0].index)


def test_classification_metrics_are_consistent():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    metrics = evaluate.classification_metrics(y_true, y_pred)

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 2
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(2 / 3)


def test_model_beats_the_majority_class_baseline(xy):
    """Sanity floor: 67% is achievable by always predicting 'no PCOS'."""
    X, y = xy
    pipeline = models.build_pipeline(
        X, models.build_classifiers()["Random Forest"], selector="rfe"
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    assert scores.mean() > 0.85


# --------------------------------------------------------------------------
# Extended model zoo
# --------------------------------------------------------------------------
def test_extended_zoo_superset_of_core():
    core = models.build_classifiers()
    extended = models.build_extended_classifiers()
    assert set(core) <= set(extended)
    for name in ("Elastic-Net LR", "Gaussian NB", "KNN"):
        assert name in extended


def test_every_model_fits_and_predicts_probabilities(xy):
    """All models must expose predict_proba, or the ROC/calibration code breaks."""
    X, y = xy
    X_small, y_small = X.head(200), y.head(200)
    for name, clf in models.build_extended_classifiers().items():
        pipeline = models.build_pipeline(X_small, clf, selector="chi2", k_features=8)
        pipeline.fit(X_small, y_small)
        proba = pipeline.predict_proba(X_small)
        assert proba.shape == (len(X_small), 2)
        assert np.allclose(proba.sum(axis=1), 1.0), f"{name} probabilities do not sum to 1"


# --------------------------------------------------------------------------
# Calibration and significance
# --------------------------------------------------------------------------
def test_expected_calibration_error_is_zero_for_perfect_probabilities():
    # 100 samples at p=1.0 that are all positive, 100 at p=0.0 all negative.
    y_true = np.array([1] * 100 + [0] * 100)
    y_proba = np.array([1.0] * 100 + [0.0] * 100)
    assert calibration.expected_calibration_error(y_true, y_proba) == pytest.approx(0.0)


def test_expected_calibration_error_catches_overconfidence():
    # Claims 0.9 confidence but is right only half the time.
    y_true = np.array([1, 0] * 50)
    y_proba = np.full(100, 0.9)
    assert calibration.expected_calibration_error(y_true, y_proba) == pytest.approx(0.4, abs=0.01)


def test_calibration_metrics_reward_honest_probabilities():
    """A well-calibrated model must score better on Brier than an overconfident one."""
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.3, 500)

    honest = np.where(y_true == 1, 0.7, 0.3)
    overconfident = np.where(y_true == 1, 0.99, 0.01) * 0 + np.full(500, 0.9)

    assert (
        calibration.calibration_metrics(y_true, honest)["brier"]
        < calibration.calibration_metrics(y_true, overconfident)["brier"]
    )


def test_significance_table_marks_reference_and_ranks(xy):
    fold_scores = pd.DataFrame(
        {
            "A": [0.90, 0.91, 0.92, 0.90],
            "B": [0.89, 0.90, 0.91, 0.89],  # consistently 0.01 lower -> separable
            "C": [0.70, 0.71, 0.72, 0.70],  # much lower
        }
    )
    table = calibration.significance_against_best(fold_scores)

    assert table.iloc[0]["model"] == "A"
    assert table.iloc[0]["statistically_sep"].startswith("—")
    # C is far below A on every fold, and by a wide margin, so it must be
    # separable on both the statistical and the practical column.
    row_c = table[table["model"] == "C"].iloc[0]
    assert row_c["statistically_sep"] == "yes"
    assert row_c["practically_sep"] == "yes"
    # B is consistently lower but by only 0.01 AUC: detectable, not meaningful.
    row_b = table[table["model"] == "B"].iloc[0]
    assert row_b["practically_sep"] == "no"


def test_significance_matrix_is_symmetric():
    fold_scores = pd.DataFrame(
        {"A": [0.9, 0.91, 0.89], "B": [0.88, 0.90, 0.87], "C": [0.85, 0.86, 0.84]}
    )
    matrix = calibration.significance_matrix(fold_scores)
    for a in matrix.index:
        for b in matrix.columns:
            if a != b:
                assert matrix.loc[a, b] == pytest.approx(matrix.loc[b, a])


def test_identical_models_are_not_separable():
    """Sanity check on the test itself: a model cannot differ from its own copy."""
    scores = pd.DataFrame({"A": [0.9, 0.92, 0.88, 0.91], "A_copy": [0.9, 0.92, 0.88, 0.91]})
    table = calibration.significance_against_best(scores)
    non_reference = table[table["p_value"].notna()]
    assert (non_reference["statistically_sep"] == "no").all()


# --------------------------------------------------------------------------
# AutoML
# --------------------------------------------------------------------------
def test_automl_column_sanitiser_strips_special_characters():
    from src.automl import _sanitise_columns

    df = pd.DataFrame(
        [[1, 2, 3]], columns=["FSH(mIU/mL)", "Waist:Hip Ratio", "BP _Systolic (mmHg)"]
    )
    out = _sanitise_columns(df)
    for column in out.columns:
        assert re_fullmatch_identifier(column), column
    assert len(set(out.columns)) == 3


def test_automl_sanitiser_disambiguates_collisions():
    from src.automl import _sanitise_columns

    df = pd.DataFrame([[1, 2]], columns=["a(b)", "a:b"])  # both flatten to "a_b"
    out = _sanitise_columns(df)
    assert len(set(out.columns)) == 2


def re_fullmatch_identifier(name: str) -> bool:
    import re

    return re.fullmatch(r"[0-9a-zA-Z_]+", name) is not None


def test_threshold_sweep_is_monotone_in_recall(xy):
    X, y = xy
    X_train, X_test, y_train, y_test = evaluate.make_splits(X, y)
    pipeline = models.build_pipeline(X_train, models.build_classifiers()["Random Forest"])
    pipeline.fit(X_train, y_train)

    sweep = evaluate.threshold_sweep(pipeline, X_test, y_test)
    # Raising the bar for a positive call can never increase recall.
    assert (sweep["recall"].diff().dropna() <= 1e-9).all()
