"""Central configuration: paths, seed, and column groupings.

Everything that another module might want to tweak lives here so that no
magic strings or magic numbers are buried inside the pipeline code.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

MODELS_DIR = PROJECT_ROOT / "models"

RAW_EXCEL = RAW_DIR / "PCOS_data_without_infertility.xlsx"
RAW_SHEET = "Full_new"


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for path in (PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# Stratified k-fold settings used for model selection.
CV_FOLDS = 10
CV_REPEATS = 3  # repeated CV gives a more stable estimate on only 541 rows

# Fraction of the data held back and never touched until final reporting.
TEST_SIZE = 0.20


# --------------------------------------------------------------------------
# Column semantics
# --------------------------------------------------------------------------
TARGET = "PCOS"

# Row identifiers, and duplicates of each other in every row. They carry no
# clinical meaning, so they are dropped before modelling.
ID_COLUMNS = ["Sl. No", "Patient File No."]

# Present in the workbook but almost entirely empty (2 non-null values / 541).
JUNK_COLUMNS = ["Unnamed: 44"]

# Blood group is coded 11-18 (A+, A-, B+, B-, O+, O-, AB+, AB-). The codes are
# nominal, so the numeric ordering is meaningless and they get one-hot encoded.
NOMINAL_COLUMNS = ["Blood Group"]

# Already 0/1 in the source file; kept as-is (no scaling, no encoding).
BINARY_COLUMNS = [
    "Pregnant(Y/N)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
    "Cycle_Irregular",
]

# Columns stored as text in the workbook because of stray characters
# ("1.99." and "a"). They are numeric measurements and get coerced.
TEXT_TYPED_NUMERIC = ["II    beta-HCG(mIU/mL)", "AMH(ng/mL)"]

# Human-friendly names for the plots and result tables.
DISPLAY_NAMES = {
    "Age (yrs)": "Age (yrs)",
    "Weight (Kg)": "Weight (kg)",
    "Height(Cm)": "Height (cm)",
    "BMI": "BMI",
    "Pulse rate(bpm)": "Pulse rate (bpm)",
    "RR (breaths/min)": "Respiratory rate",
    "Hb(g/dl)": "Haemoglobin (g/dl)",
    "Cycle_Irregular": "Irregular cycle",
    "Cycle length(days)": "Cycle length (days)",
    "Marraige Status (Yrs)": "Years married",
    "No. of aborptions": "No. of abortions",
    "I   beta-HCG(mIU/mL)": "beta-HCG I",
    "II    beta-HCG(mIU/mL)": "beta-HCG II",
    "FSH(mIU/mL)": "FSH",
    "LH(mIU/mL)": "LH",
    "FSH/LH": "FSH/LH ratio",
    "Hip(inch)": "Hip (in)",
    "Waist(inch)": "Waist (in)",
    "Waist:Hip Ratio": "Waist:hip ratio",
    "TSH (mIU/L)": "TSH",
    "AMH(ng/mL)": "AMH",
    "PRL(ng/mL)": "Prolactin",
    "Vit D3 (ng/mL)": "Vitamin D3",
    "PRG(ng/mL)": "Progesterone",
    "RBS(mg/dl)": "Random blood sugar",
    "BP _Systolic (mmHg)": "BP systolic",
    "BP _Diastolic (mmHg)": "BP diastolic",
    "Follicle No. (L)": "Follicle count (L)",
    "Follicle No. (R)": "Follicle count (R)",
    "Avg. F size (L) (mm)": "Avg follicle size (L)",
    "Avg. F size (R) (mm)": "Avg follicle size (R)",
    "Endometrium (mm)": "Endometrium (mm)",
    "Weight gain(Y/N)": "Weight gain",
    "hair growth(Y/N)": "Hair growth",
    "Skin darkening (Y/N)": "Skin darkening",
    "Hair loss(Y/N)": "Hair loss",
    "Pimples(Y/N)": "Pimples",
    "Fast food (Y/N)": "Fast food",
    "Reg.Exercise(Y/N)": "Regular exercise",
    "Pregnant(Y/N)": "Pregnant",
}


def pretty(column: str) -> str:
    """Map a raw column name to its display name, falling back to itself."""
    return DISPLAY_NAMES.get(column, column)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
FIG_DPI = 150
PALETTE = {"no_pcos": "#4C78A8", "pcos": "#E45756"}
