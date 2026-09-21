# METHODS.md — methods executed so far (2026-09-21)

## 1. Environment
- Created `micromamba` env `concrete-logo` (python 3.11, conda-forge).
- Installed scikit-learn, pandas, numpy, matplotlib, seaborn, datasets,
  xgboost, jupyter; versions pinned in `requirements.txt`
  (sklearn 1.9.1, pandas 3.0.6, numpy 2.4.6, matplotlib 3.11.2,
  seaborn 0.13.2, datasets 5.0.1, xgboost 3.4.2).

## 2. Data loading
- Loaded `load_dataset("foundry-ml/dataset_concrete_compressive_strength",
  split="train")` → 1030 rows.
- Stripped whitespace from column names, renamed verbose HF names to
  `cement, slag, fly_ash, water, superplasticizer, coarse_agg, age, strength`
  (`splits.py:COLUMN_MAP`, `load_concrete()`).
- No row cleaning: 0 nulls, 25 duplicate rows retained (disclosed, see NOTES.md).

## 3. EDA (`eda.py`)
- Reported shape, null/dupe counts, raw age distribution, and train/test sizes
  for every planned split (random, A1, A2, A3 bins, C1, C2, C3).
- Saved `figures/age_histogram.png` (age distribution, 28d skew).
- Finding: main splits all have test n>100; A3 bins 14/56/180+ do not (see NOTES.md).

## 4. Split definitions (`splits.py`)
- Wrote canonical split functions (random 70/15/15, A1, A2, A3 bins, C1, C2, C3)
  plus `get_xy()`, `FEATURES`/`TARGET`, `SEEDS=(0,1,2)`.
- Implemented raw→bin map for A3 (`<=3→3, 90–120→90, ≥180→180+`).
- These definitions are written and EDA-verified for group sizes, but the full
  experiment matrix has NOT been run on them yet.

## 5. Baseline sanity check (single run, seed 0)
- Pipeline: `StandardScaler` (fit on train only) + `RandomForestRegressor
  (n_estimators=200, random_state=0)`, via `splits.get_xy`.
- Splits tested: random 70/15/15, A1, C1. Metrics: R² + RMSE
  (random 0.909/5.08, A1 0.340/10.36, C1 0.568/9.34).

## NOT yet done
- Full matrix: LR / RF / XGB / MLP × all splits × 3 seeds, mean ± std.
- 5-fold CV on random split; MAE reporting; summary table.
- Paper figures (pred-vs-true, error-vs-age, importance/SHAP).
