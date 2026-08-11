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
# Physiologically plausible ranges
# --------------------------------------------------------------------------
# Values outside these bounds are recording errors, not measurements, and are
# set to NaN so the pipeline's imputer handles them.
#
# Several of these were only discovered by comparing feature distributions
# against an independent cohort (see src/external.py). Two blood-pressure
# readings of 12/80 and 120/8 are obvious digit-drops from 120/80, and a few
# hormone values are off by three orders of magnitude (FSH 5052, LH 2018,
# vitamin D 6014). They are blanked rather than "corrected": inferring the
# intended value would be guessing, and one imputed median is more honest
# than a plausible-looking invention.
PLAUSIBLE_RANGES = {
    "Age (yrs)": (10, 70),
    "Weight (Kg)": (25, 200),
    "Height(Cm)": (120, 200),
    "BMI": (10, 70),
    "Pulse rate(bpm)": (35, 200),
    "RR (breaths/min)": (8, 40),
    "Hb(g/dl)": (4, 20),
    "Cycle length(days)": (1, 60),
    "BP _Systolic (mmHg)": (70, 250),
    "BP _Diastolic (mmHg)": (40, 150),
    "FSH(mIU/mL)": (0.1, 200),
    "LH(mIU/mL)": (0.01, 200),
    "PRL(ng/mL)": (0.1, 400),
    "TSH (mIU/L)": (0.01, 100),
    "AMH(ng/mL)": (0.01, 50),
    "Vit D3 (ng/mL)": (1, 150),
    "RBS(mg/dl)": (40, 600),
    "Endometrium (mm)": (1, 30),
    "Waist(inch)": (15, 70),
    "Hip(inch)": (20, 80),
}


# --------------------------------------------------------------------------
# Feature acquisition cost tiers
# --------------------------------------------------------------------------
# The project's stated motivation is that PCOS testing "may not always be easy
# to access, affordable, or quick, especially in areas with limited healthcare
# resources". These tiers make that testable: each adds a class of measurement
# that costs more to obtain, so the model comparison shows what each tier of
# investment actually buys in predictive terms.
TIER_QUESTIONNAIRE = [
    # Free: a form, a scale and a tape measure. No clinician, no equipment.
    "Age (yrs)",
    "Weight (Kg)",
    "Height(Cm)",
    "BMI",
    "Waist(inch)",
    "Hip(inch)",
    "Waist:Hip Ratio",
    "Cycle_Irregular",
    "Cycle length(days)",
    "Marraige Status (Yrs)",
    "No. of aborptions",
    "Pregnant(Y/N)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
]

TIER_CLINIC = [
    # Adds a basic clinic visit: vitals a nurse can take in minutes.
    "Pulse rate(bpm)",
    "RR (breaths/min)",
    "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)",
    "Blood Group",
    "Hb(g/dl)",
]

TIER_LAB = [
    # Adds a venous blood draw and an endocrine assay panel.
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "FSH/LH",
    "TSH (mIU/L)",
    "AMH(ng/mL)",
    "PRL(ng/mL)",
    "Vit D3 (ng/mL)",
    "PRG(ng/mL)",
    "RBS(mg/dl)",
    "I   beta-HCG(mIU/mL)",
    "II    beta-HCG(mIU/mL)",
]

TIER_ULTRASOUND = [
    # Adds a transvaginal ultrasound and a trained sonographer.
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Avg. F size (L) (mm)",
    "Avg. F size (R) (mm)",
    "Endometrium (mm)",
]

# Cumulative tiers, cheapest first.
FEATURE_TIERS = {
    "questionnaire": TIER_QUESTIONNAIRE,
    "+ clinic vitals": TIER_QUESTIONNAIRE + TIER_CLINIC,
    "+ blood panel": TIER_QUESTIONNAIRE + TIER_CLINIC + TIER_LAB,
    "+ ultrasound (all)": (
        TIER_QUESTIONNAIRE + TIER_CLINIC + TIER_LAB + TIER_ULTRASOUND
    ),
}


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
FIG_DPI = 150
PALETTE = {"no_pcos": "#4C78A8", "pcos": "#E45756"}
