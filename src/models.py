"""Model definitions, assembled as leakage-safe pipelines.

Every model is a single :class:`imblearn.pipeline.Pipeline` containing the
whole chain:

``impute -> scale/encode -> select features -> SMOTE -> classifier``

Packaging it this way is the point of the module. When such a pipeline is
handed to ``cross_val_score``, scikit-learn refits *every* step on each
training fold, so the imputer, the scaler, the feature selector and SMOTE
never observe the held-out fold. The alternative — imputing, selecting and
oversampling the full dataset up front and cross-validating afterwards — is
what produces the 98-99% accuracies reported by several of the reviewed
papers, because SMOTE will happily synthesise a validation-set point from
its own training-set neighbours.

The imblearn ``Pipeline`` (rather than the scikit-learn one) is required
because SMOTE resamples rows: it must run on ``fit`` but be skipped on
``predict``, which only imblearn's pipeline knows how to do.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config
from .data import feature_groups
from .features import build_selector

# CatBoost is an optional dependency: it is the model Paper 3 reports as its
# best, so it is included when available, but the pipeline must still run
# without it.
try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_CATBOOST = False


def build_preprocessor(X) -> ColumnTransformer:
    """Impute, scale and encode, split by the kind of variable.

    * continuous — median imputation, then standardisation. The median is
      robust to the long right tails in the hormone measurements.
    * nominal (blood group) — most-frequent imputation, then one-hot.
    * binary indicators — most-frequent imputation, no scaling; they are
      already 0/1 and rescaling them only obscures interpretation.
    """
    groups = feature_groups(X)

    continuous = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    nominal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])

    preprocessor = ColumnTransformer(
        [
            ("continuous", continuous, groups["continuous"]),
            ("nominal", nominal, groups["nominal"]),
            ("binary", binary, groups["binary"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Pandas output keeps column names alive through the pipeline, which the
    # feature selectors and the SHAP plots both rely on.
    return preprocessor.set_output(transform="pandas")


def build_classifiers(random_state: int = config.RANDOM_STATE) -> dict:
    """The four core classifiers named in the project proposal.

    These are the models used as stacking base learners. The wider comparison
    set is :func:`build_extended_classifiers`.

    ``class_weight="balanced"`` is deliberately *not* set on top of SMOTE.
    Doing both corrects the same 2:1 imbalance twice and pushes the models
    into over-predicting PCOS. SMOTE is the chosen mechanism; the
    class-weighting alternative is measured separately in the ablation.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_extended_classifiers(random_state: int = config.RANDOM_STATE) -> dict:
    """Every algorithm in the comparison, including weak baselines.

    Beyond the four core models this adds:

    * **Elastic-net LR** — L1+L2 penalty, so the linear model performs its own
      feature selection. A methodological counterpoint to the separate
      RFE/chi-square stage.
    * **CatBoost** — the model Paper 3 reports as its best (95.7%). Included so
      that comparison rests on a run of our own rather than a quoted number.
    * **Gaussian Naive Bayes** and **KNN** — used by Papers 1 and 4. Both are
      expected to underperform, which is the point: they establish a weak-model
      floor, so "0.95 AUC" is measured against something other than the
      67% majority-class baseline.

    KNN is included with the caveat that it is the model most damaged by SMOTE:
    oversampling by interpolating between neighbours, then classifying by
    neighbours, is close to circular. Its result should be read with that in
    mind.
    """
    classifiers = dict(build_classifiers(random_state))

    classifiers["Elastic-Net LR"] = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=10000,
        random_state=random_state,
    )
    classifiers["Gaussian NB"] = GaussianNB()
    classifiers["KNN"] = KNeighborsClassifier(n_neighbors=11, weights="distance")

    if HAS_CATBOOST:
        classifiers["CatBoost"] = CatBoostClassifier(
            iterations=400,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            verbose=0,
            allow_writing_files=False,
        )

    return classifiers


def build_stacking(random_state: int = config.RANDOM_STATE) -> StackingClassifier:
    """Stacking ensemble over the four core base learners.

    The meta-learner is a plain logistic regression on the base models'
    predicted probabilities. ``cv=5`` makes the base predictions
    out-of-fold, so the meta-learner is not trained on the base models'
    own training-set predictions.
    """
    base = build_classifiers(random_state)
    return StackingClassifier(
        estimators=[(name, model) for name, model in base.items()],
        final_estimator=LogisticRegression(max_iter=5000, random_state=random_state),
        cv=5,
        stack_method="predict_proba",
        n_jobs=1,
    )


def build_pipeline(
    X,
    classifier,
    selector: str = "all",
    k_features: int = 15,
    balance: str = "smote",
    random_state: int = config.RANDOM_STATE,
) -> ImbPipeline:
    """Assemble the full preprocess -> select -> balance -> classify chain.

    Parameters
    ----------
    X:
        A sample of the feature frame, used only to work out which columns
        are continuous / nominal / binary.
    classifier:
        Any fitted-or-unfitted scikit-learn compatible estimator.
    selector:
        One of ``"all"``, ``"correlation"``, ``"chi2"``, ``"rfe"``.
    k_features:
        Number of features to keep, for the chi2 and RFE selectors.
    balance:
        ``"smote"`` to oversample the minority class, ``"none"`` to leave the
        2:1 imbalance in place. Class weighting is applied by passing an
        already-weighted classifier instead.
    """
    steps = [
        ("preprocess", build_preprocessor(X)),
        ("select", build_selector(selector, k=k_features)),
    ]

    if balance == "smote":
        # k_neighbors must stay below the minority-class count of the smallest
        # training fold. With ~140 PCOS cases per fold the default of 5 is
        # comfortably safe.
        steps.append(("balance", SMOTE(random_state=random_state, k_neighbors=5)))
    elif balance != "none":
        raise ValueError(f"Unknown balance strategy {balance!r}; use 'smote' or 'none'.")

    steps.append(("classifier", classifier))
    return ImbPipeline(steps)


def build_all_models(
    X, selector: str = "all", k_features: int = 15, extended: bool = True
) -> dict:
    """Every model in the comparison, wrapped in its pipeline.

    Parameters
    ----------
    extended:
        Include the wider algorithm set (elastic-net LR, CatBoost, Naive
        Bayes, KNN) alongside the four core models and the stacking ensemble.
        Set ``False`` for the smaller, faster core comparison.
    """
    source = build_extended_classifiers() if extended else build_classifiers()

    models = {}
    for name, clf in source.items():
        models[name] = build_pipeline(X, clf, selector=selector, k_features=k_features)
    models["Stacking Ensemble"] = build_pipeline(
        X, build_stacking(), selector=selector, k_features=k_features
    )
    return models


def build_calibrated(
    X,
    classifier,
    selector: str = "all",
    k_features: int = 15,
    method: str = "isotonic",
) -> ImbPipeline:
    """Wrap a classifier so its predicted probabilities are calibrated.

    Tree ensembles push probabilities toward the middle of the range: a
    Random Forest that outputs 0.8 is not right 80% of the time. That is
    harmless for ROC-AUC, which only depends on ranking, but it matters as
    soon as a probability is used to set a screening threshold.

    The calibrator sits *inside* the pipeline and uses its own internal CV, so
    it is fitted on training folds only, like every other fitted step.
    """
    from sklearn.calibration import CalibratedClassifierCV

    calibrated = CalibratedClassifierCV(classifier, method=method, cv=5)
    return build_pipeline(X, calibrated, selector=selector, k_features=k_features)


def selected_feature_names(fitted_pipeline: ImbPipeline) -> list[str]:
    """Feature names surviving the selector of an already-fitted pipeline."""
    return list(fitted_pipeline.named_steps["select"].get_feature_names_out())
