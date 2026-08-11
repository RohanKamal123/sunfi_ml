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

from src import config, data, evaluate, features, models


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


def test_threshold_sweep_is_monotone_in_recall(xy):
    X, y = xy
    X_train, X_test, y_train, y_test = evaluate.make_splits(X, y)
    pipeline = models.build_pipeline(X_train, models.build_classifiers()["Random Forest"])
    pipeline.fit(X_train, y_train)

    sweep = evaluate.threshold_sweep(pipeline, X_test, y_test)
    # Raising the bar for a positive call can never increase recall.
    assert (sweep["recall"].diff().dropna() <= 1e-9).all()
