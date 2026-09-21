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

## 5. Full experiment matrix (2026-09-21, `experiments.py`)
- Models (all as `StandardScaler` + estimator Pipeline, scaler fit on train only):
  - LR: `LinearRegression()`
  - RF: `RandomForestRegressor(n_estimators=200, random_state=seed)`
  - XGB: `XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=seed)`
  - MLP: `MLPRegressor(hidden_layer_sizes=(64,32), max_iter=2000,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
    random_state=seed)`
- Splits: random 70/15/15 test per seed + A1, A2, C1, C2, C3, A3-{3,7,14,28,56,90,180+} × SEEDS=(0,1,2).
- Metrics per run: RMSE + MAE + R2; summary = mean ± std across 3 seeds
  (`results/runs.csv`, `results/summary.csv`).
- 5-fold CV control: `KFold(n_splits=5, shuffle=True, random_state=seed)`
  on full data per model × seed (`results/cv.csv`).
- Figure inputs saved: `results/preds_{LR,RF,XGB,MLP}_{random,A1,C1}_s0.csv`,
  `results/preds_RF_A1train_all.csv`, `results/error_vs_age_RF_A1train.csv`,
  `results/importance_RF_s0.csv`; versions in `results/versions.txt`.

### Results (mean ± std across 3 seeds; R2 shown, RMSE/MAE in `results/summary.csv`)
| model | random | A1 | A2 | C1 | C2 | C3 | 5-fold CV R2 |
|---|---|---|---|---|---|---|---|
| LR | 0.589±0.047 | -61.27±0.00 | -2.54±0.00 | 0.028±0.00 | -0.338±0.00 | 0.375±0.00 | 0.603±0.002 |
| RF | 0.891±0.027 | 0.338±0.001 | -0.177±0.013 | 0.568±0.003 | 0.318±0.005 | 0.087±0.011 | 0.909±0.002 |
| XGB | 0.919±0.016 | 0.386±0.002 | 0.012±0.009 | 0.667±0.015 | 0.435±0.029 | 0.335±0.043 | 0.935±0.003 |
| MLP | 0.727±0.173 | -148.4±37.6 | -28.3±2.2 | 0.314±0.147 | 0.161±0.058 | 0.082±0.272 | 0.855±0.005 |
Random RMSE: XGB 4.67, RF 5.40, MLP 8.36, LR 10.54 MPa.
A1 RMSE: XGB 9.99, RF 10.37 MPa (LR/MLP catastrophically off-scale — linear/MLP
extrapolation failure on unseen old ages, reported as-is, not tuned away).
A3-LOGO (RF seed-mean R2): hold-3 -0.10, 7: 0.55, 14: -0.37, 28: 0.36,
56: 0.73, 90: 0.84, 180+: 0.75. XGB same pattern, higher on 14 (0.51) / 180+ (0.88).
Early-age hold-outs fail; late-age hold-outs look easy (strength plateau) —
discuss, don't hide.

## 6. Figures (`figures.py`, regenerates from `results/`)
Main paper (3 — the README budget):
- `figures/fig1_pred_vs_true.png`: RF pred-vs-true, random vs A1 vs C1 (seed 0).
- `figures/fig2_error_vs_age.png`: RF trained on Age<=28, RMSE/MAE per age bin —
  in-train bins ~1.5–3.2 MPa, 56d 8.0, 90d 10.6, 180+d 9.9 MPa.
- `figures/fig4_shap.png`: XGB SHAP beeswarm + mean|SHAP| (replaces impurity bars).
Supplement:
- `figures/fig5_pdp.png`, `figures/fig6_a3_heatmap.png`,
  `figures/fig7_residuals.png`, `figures/age_histogram.png`.
Dropped 2026-09-21: `fig3_importance.png` (redundant with SHAP),
`fig8_bias_learning.png` (left duplicated Fig2, right learning curve tangential).
Bias-per-bin numbers live in `results/deep_bias_by_age.csv` + Fig7 annotations.
- Note: `figures/` is git-ignored; regenerate via `eda.py` + `figures.py`.
No titles on any figure — captions live in the paper, panels identified by axis labels.

## 7. Deep analysis (2026-09-21, `analysis_deep.py`, adds `shap==0.51.0`)
- Fig4 SHAP (XGB, random-train, seed 0): mean|SHAP| age 7.73, cement 6.98, water 3.60,
  slag 3.20, SP 2.21, coarse 0.97, fly_ash 0.54 MPa. Beeswarm shows high age/cement push
  up, high water pushes down. Permutation cross-check (RF, random-test): age 0.69,
  cement 0.50 R² drop when shuffled — same ranking.
- Fig5 PDP (XGB): age rises steeply 0→~60d then plateaus (explains why late hold-outs
  look easy); cement monotonic up; water monotonic down.
- Fig6 A3 heatmap (`results/deep_a3_r2.csv`): full 4×7 R² matrix, annotates the
  early-fail / late-easy asymmetry for every model.
- Fig7 residuals+calibration (RF seed 0): random bias −0.79 MPa, calibrated;
  A1 bias −9.33 MPa, calibration curve sags below diagonal (systematic under-prediction);
  C1 bias +2.68 MPa, wider scatter.
- Stats (`results/deep_drop_CI.csv`, 5000-rep bootstrap of seed means):
  RF random−A1 drop 0.553 [0.522,0.570]; XGB random−A1 0.532 [0.516,0.548];
  RF random−C1 0.323 [0.292,0.341]. All CIs exclude 0 by a wide margin.

## NOT yet done
- Paper draft (IMRaD, 1500–2500 words) + novelty Scholar check + mentor sign-off.
- SHAP (optional; impurity importance done as brief Fig 3).
