"""Loading and cleaning of the Kaggle PCOS dataset.

The workbook ships with a handful of data-quality problems that have to be
fixed before anything downstream will work:

* two measurement columns are stored as text because of stray characters
  ("1.99." in beta-HCG II and "a" in AMH);
* an almost entirely empty "Unnamed: 44" column;
* trailing/leading whitespace in most column names;
* one impossible ``Cycle(R/I)`` code (5, where only 2 and 4 are defined);
* two isolated missing values.

Note that missing-value *imputation* deliberately does not happen here. It is
part of the modelling pipeline (see :mod:`src.models`) so that the imputer is
fitted on training folds only and never sees the validation data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# Codes used by the source file for the menstrual cycle column.
CYCLE_REGULAR = 2
CYCLE_IRREGULAR = 4


def load_raw(path=None, sheet: str = config.RAW_SHEET) -> pd.DataFrame:
    """Read the raw workbook exactly as distributed, with no modifications."""
    path = path or config.RAW_EXCEL
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at {path}.\n"
            "Download 'PCOS_data_without_infertility.xlsx' from the Kaggle "
            "dataset 'prasoonkottarathil/polycystic-ovary-syndrome-pcos' and "
            f"place it in {config.RAW_DIR}."
        )
    return pd.read_excel(path, sheet_name=sheet)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning step and return a tidy, fully numeric frame.

    The returned frame still contains NaNs where values were genuinely
    missing or invalid; filling them is the pipeline's job.
    """
    df = df.copy()

    # 1. Normalise column names: the workbook has inconsistent whitespace
    #    ("Height(Cm) ", " Age (yrs)", "  I   beta-HCG(mIU/mL)").
    df.columns = [str(c).strip() for c in df.columns]

    # 2. Rename the target to something that survives being used as a
    #    Python identifier.
    df = df.rename(columns={"PCOS (Y/N)": config.TARGET})

    # 3. Drop identifiers and the near-empty column.
    drop = [c for c in config.ID_COLUMNS + config.JUNK_COLUMNS if c in df.columns]
    df = df.drop(columns=drop)

    # 4. Coerce the text-typed measurement columns. `errors="coerce"` turns
    #    the two malformed entries into NaN rather than crashing.
    for col in config.TEXT_TYPED_NUMERIC:
        col = col.strip()
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Recode the menstrual cycle column into a clean 0/1 indicator.
    #    2 = regular, 4 = irregular. A single row carries the value 5, which
    #    is not a defined code, so it becomes NaN and is imputed later.
    if "Cycle(R/I)" in df.columns:
        df["Cycle_Irregular"] = df["Cycle(R/I)"].map(
            {CYCLE_REGULAR: 0, CYCLE_IRREGULAR: 1}
        )
        df = df.drop(columns=["Cycle(R/I)"])

    # 6. Anything still non-numeric would break the estimators. Coerce it and
    #    let the imputer deal with the fallout.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 7. Physiologically impossible zeros. A cycle length or endometrial
    #    thickness of 0 mm is a recording failure, not a measurement.
    for col in ("Cycle length(days)", "Endometrium (mm)"):
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    df[config.TARGET] = df[config.TARGET].astype(int)
    return df.reset_index(drop=True)


def load_clean(path=None) -> pd.DataFrame:
    """Convenience wrapper: read the workbook and clean it in one call."""
    return clean(load_raw(path))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target vector."""
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET])
    return X, y


def feature_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    """Partition the feature columns into the three kinds of variable.

    Used to build the ColumnTransformer: continuous features get scaled,
    nominal features get one-hot encoded, and binary indicators are passed
    through untouched.
    """
    nominal = [c for c in config.NOMINAL_COLUMNS if c in X.columns]
    binary = [c for c in config.BINARY_COLUMNS if c in X.columns]
    continuous = [c for c in X.columns if c not in nominal and c not in binary]
    return {"continuous": continuous, "nominal": nominal, "binary": binary}


def data_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Summarise what cleaning changed, for the write-up."""
    rows = []
    raw_cols = [str(c).strip() for c in raw.columns]
    for col in cleaned.columns:
        missing = int(cleaned[col].isna().sum())
        rows.append(
            {
                "feature": config.pretty(col),
                "column": col,
                "dtype": str(cleaned[col].dtype),
                "missing": missing,
                "missing_pct": round(100 * missing / len(cleaned), 2),
                "n_unique": int(cleaned[col].nunique()),
                "was_text_in_source": col in [c.strip() for c in config.TEXT_TYPED_NUMERIC],
                "derived": col not in raw_cols,
            }
        )
    return pd.DataFrame(rows).sort_values("missing", ascending=False)
