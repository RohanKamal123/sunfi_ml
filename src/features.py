"""Feature selection strategies.

The project compares four feature sets:

``all``
    Every available feature. The baseline.
``correlation``
    Features whose absolute Pearson correlation with the target clears a
    threshold, after dropping one of every pair of near-duplicate features.
``chi2``
    Top-k features by the chi-square statistic (requires non-negative input,
    so the values are min-max scaled first).
``rfe``
    Recursive Feature Elimination driven by a logistic regression.

Every selector here is a scikit-learn transformer. That matters: it means
selection happens *inside* each cross-validation fold, fitted on training
data only. Selecting features once on the full dataset and then
cross-validating is a classic leak that inflates accuracy, and it is one of
the problems this project set out to avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from . import config

# Note the base-class order throughout this module: scikit-learn resolves
# estimator tags through the MRO, so the mixin must precede BaseEstimator.
# With the order reversed, BaseEstimator's tags win and downstream tooling
# can silently mis-classify the estimator.


class CorrelationSelector(TransformerMixin, BaseEstimator):
    """Keep features correlated with the target, drop redundant ones.

    Two passes:

    1. *Relevance* — drop features whose absolute correlation with the target
       is below ``target_threshold``.
    2. *Redundancy* — within the survivors, if two features correlate with
       each other above ``collinear_threshold``, drop whichever has the
       weaker relationship with the target.

    Parameters
    ----------
    target_threshold:
        Minimum |corr(feature, target)| for a feature to be kept.
    collinear_threshold:
        |corr(feature_i, feature_j)| above which the pair is considered
        redundant.
    """

    def __init__(self, target_threshold: float = 0.10, collinear_threshold: float = 0.85):
        self.target_threshold = target_threshold
        self.collinear_threshold = collinear_threshold

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y), index=X.index)

        target_corr = X.corrwith(y).abs().fillna(0.0)
        keep = target_corr[target_corr >= self.target_threshold].sort_values(
            ascending=False
        )

        if keep.empty:
            # Degenerate threshold: fall back to the single best feature so
            # the pipeline still has something to train on.
            keep = target_corr.sort_values(ascending=False).head(1)

        # Drop redundant pairs, keeping the more target-relevant member.
        # `keep` is already sorted best-first, so iterating in order means the
        # feature we retain is always the stronger of the pair.
        ordered = list(keep.index)
        feature_corr = X[ordered].corr().abs()
        selected: list[str] = []
        for col in ordered:
            if all(feature_corr.loc[col, s] < self.collinear_threshold for s in selected):
                selected.append(col)

        self.selected_features_ = selected
        self.target_correlations_ = target_corr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


class Chi2Selector(TransformerMixin, BaseEstimator):
    """Top-k features by chi-square statistic.

    Chi-square needs non-negative inputs, so a min-max scaler is fitted on
    the training fold and applied before scoring. The scaler is fitted here
    rather than reusing the pipeline's standard scaler, which produces
    negative values.
    """

    def __init__(self, k: int = 15):
        self.k = k

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        k = min(self.k, X.shape[1])

        self.scaler_ = MinMaxScaler().fit(X)
        X_scaled = np.clip(self.scaler_.transform(X), 0, None)

        self.selector_ = SelectKBest(score_func=chi2, k=k).fit(X_scaled, y)
        mask = self.selector_.get_support()
        self.selected_features_ = [c for c, m in zip(self.columns_, mask) if m]
        self.scores_ = pd.Series(self.selector_.scores_, index=self.columns_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


class RFESelector(TransformerMixin, BaseEstimator):
    """Recursive Feature Elimination with a logistic-regression estimator.

    A thin wrapper around :class:`sklearn.feature_selection.RFE` that keeps
    track of the surviving column *names*, which the bare RFE does not do
    once it is handed a NumPy array.
    """

    def __init__(self, n_features: int = 15, random_state: int = config.RANDOM_STATE):
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        n = min(self.n_features, X.shape[1])

        estimator = LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            random_state=self.random_state,
        )
        self.rfe_ = RFE(estimator=estimator, n_features_to_select=n, step=1).fit(X, y)

        mask = self.rfe_.get_support()
        self.selected_features_ = [c for c, m in zip(self.columns_, mask) if m]
        self.ranking_ = pd.Series(self.rfe_.ranking_, index=self.columns_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


class PassthroughSelector(TransformerMixin, BaseEstimator):
    """No-op selector, so that the "all features" baseline has the same shape."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.selected_features_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


def build_selector(name: str, k: int = 15):
    """Factory: return a fresh selector instance by name."""
    if name == "all":
        return PassthroughSelector()
    if name == "correlation":
        return CorrelationSelector()
    if name == "chi2":
        return Chi2Selector(k=k)
    if name == "rfe":
        return RFESelector(n_features=k)
    raise ValueError(
        f"Unknown selector {name!r}; expected one of "
        "'all', 'correlation', 'chi2', 'rfe'."
    )


SELECTOR_NAMES = ("all", "correlation", "chi2", "rfe")
