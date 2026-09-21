"""Canonical split definitions for concrete-logo.

Single source of truth for Random vs Age-LOGO vs SCM-LOGO splits so
reviewers can rerun every experiment. See README.md for the design.

Dataset notes (verified 2026-09-21, HF `foundry-ml/dataset_concrete_compressive_strength`):
- 1030 rows. Verbose column names are normalized to short names on load.
- This repackaging has 7 inputs, NOT 8: Fine Aggregate (component 7 of the
  UCI original) is absent. Analyses must note this vs Yeh 1998.
- Raw Age values: 1, 3, 7, 14, 28, 56, 90, 91, 100, 120, 180, 270, 360, 365.
  Small bins are merged for A3 (see AGE_BIN).
- 25 duplicated rows exist; kept as-is, noted in Methods.
- A1 leaves a gap: train Age <= 28, test Age >= 90. Ages 56..28<age<90 fall
  in neither set by design (extrapolation distance, not interpolation).

Anti-leakage rules (enforced by convention, verified in review):
- Fit scalers/imputers on train only, apply to test.
- No target-derived features. Fixed random_state everywhere.
"""

from datasets import load_dataset

SEEDS = (0, 1, 2)

FEATURES = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_agg",
    "age",
]
TARGET = "strength"

# Raw HF column -> short name. HF names have trailing spaces; stripped first.
COLUMN_MAP = {
    "Cement (component 1)(kg in a m^3 mixture)": "cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash",
    "Water  (component 4)(kg in a m^3 mixture)": "water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_agg",
    "Age (day)": "age",
    "Concrete compressive strength(MPa, megapascals)": "strength",
}

# Raw age -> A3 bin label. Tiny bins merged so every held-out group has n>=30:
#   1 (n=2) -> 3; 91/100/120 -> 90; 270/360/365 -> 180+.
# Resulting bins: 3 (136), 7 (126), 14 (62), 28 (425), 56 (91), 90 (131), 180+ (59).
def age_to_bin(a: int) -> str:
    if a <= 3:
        return "3"
    if a == 7:
        return "7"
    if a == 14:
        return "14"
    if a == 28:
        return "28"
    if a == 56:
        return "56"
    if 90 <= a <= 120:
        return "90"
    return "180+"


def load_concrete():
    """Load HF dataset, normalize columns to short names. Returns DataFrame."""
    ds = load_dataset(
        "foundry-ml/dataset_concrete_compressive_strength", split="train"
    )
    df = ds.to_pandas()
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns=COLUMN_MAP)


def get_xy(df):
    """Split DataFrame into X (features) and y (target)."""
    return df[FEATURES], df[TARGET]


def random_split(df, seed=0, test_size=0.15, val_size=0.15):
    """Baseline control: random train/val/test. Returns (train, val, test)."""
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(df, test_size=test_size + val_size,
                                   random_state=seed)
    val_frac = val_size / (test_size + val_size)
    val, test = train_test_split(temp, test_size=1 - val_frac,
                                 random_state=seed)
    return train, val, test


# --- Age-LOGO ---
def age_a1(df):
    """A1: train Age <= 28d, test Age >= 90d."""
    return df[df["age"] <= 28], df[df["age"] >= 90]


def age_a2(df):
    """A2: train on 3d+7d only, predict 28d."""
    return df[df["age"].isin([3, 7])], df[df["age"] == 28]


def age_a3_bins(df):
    """A3: dict bin_label -> (train, test) for leave-one-age-bin-out."""
    df = df.copy()
    df["age_bin"] = df["age"].map(age_to_bin)
    out = {}
    for b in ["3", "7", "14", "28", "56", "90", "180+"]:
        out[b] = (df[df["age_bin"] != b], df[df["age_bin"] == b])
    return out


# --- Composition-LOGO ---
def comp_c1_flyash(df):
    """C1: train Fly Ash == 0, test Fly Ash > 0."""
    return df[df["fly_ash"] == 0], df[df["fly_ash"] > 0]


def comp_c2_slag(df):
    """C2: train Slag == 0, test Slag > 0."""
    return df[df["slag"] == 0], df[df["slag"] > 0]


def comp_c3_superplasticizer(df):
    """C3 (optional): train w/o superplasticizer, test with."""
    return df[df["superplasticizer"] == 0], df[df["superplasticizer"] > 0]


ALL_SPLITS = {
    "random": random_split,
    "A1": age_a1,
    "A2": age_a2,
    "C1": comp_c1_flyash,
    "C2": comp_c2_slag,
    "C3": comp_c3_superplasticizer,
}
