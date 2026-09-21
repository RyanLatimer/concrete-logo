# NOTES.md — things of note (verified 2026-09-21)

## Dataset vs README discrepancies (all verified against HF download)
1. **7 inputs, not 8.** `foundry-ml/dataset_concrete_compressive_strength` has
   cement, slag, fly ash, water, superplasticizer, coarse agg, age + target.
   **Fine Aggregate (UCI component 7) is absent** from this repackaging.
   Must be stated in the paper vs Yeh 1998.
2. **Extra raw Age values.** README lists 3,7,14,28,56,90,180,270,365; actual
   data also contains **1 (n=2), 91 (22), 100 (52), 120 (3), 360 (6)**.
   Full distribution: 1:2, 3:134, 7:126, 14:62, 28:425, 56:91, 90:54, 91:22,
   100:52, 120:3, 180:26, 270:13, 360:6, 365:14.
3. **25 duplicate rows**, 0 nulls. Kept as-is; note in Methods.
4. Column names are verbose with trailing spaces — `splits.py:COLUMN_MAP`
   normalizes to `cement, slag, fly_ash, water, superplasticizer, coarse_agg, age, strength`.

## A3 small-bin warning (README anticipated this)
Leave-one-age-bin-out held-out sizes: 3:136, 7:126, 14:**62**, 28:425,
56:**91**, 90:131, 180+:**59**. Bins 14 / 56 / 180+ are <100 rows.
Bin mapping (`splits.py:age_to_bin`): `<=3 -> "3"`, `90..120 -> "90"`,
`>=180 -> "180+"`. DECIDED 2026-09-21: keep bins as-is (option A),
report small-bin uncertainty via 3-seed std + one caveat sentence in paper.

## A1 has a gap by design
Train Age<=28 (n=749), test Age>=90 (n=190). Ages 56..89 fall in neither set
— extrapolation distance, not interpolation. State explicitly in paper.

## Baseline sanity check (RF-200, seed 0, scaler fit on train only)
random R2 **0.909** (RMSE 5.08) → A1 **0.340** (10.36) → C1 **0.568** (9.34).
Drop is real; age extrapolation harsher than the hypothesized 0.6–0.8.
CONFIRMED 2026-09-21 by full 3-seed matrix (`results/summary.csv`):
RF random 0.891±0.027 → A1 0.338±0.001 → C1 0.568±0.003;
XGB random 0.919±0.016 → A1 0.386±0.002 → C1 0.667±0.015.
LR/MLP collapse on A1 (-61 / -148): extrapolation failure, report as-is.

## Full-matrix findings (2026-09-21, `experiments.py` + `figures.py`)
- A2 (train 3d+7d → predict 28d): RF -0.177, XGB +0.012 — early→peak
  extrapolation fails even for trees. Strong second age result.
- A3 asymmetry: holding out 3 / 14 / 28 fails (RF -0.10 / -0.37 / +0.36)
  while holding out 56 / 90 / 180+ looks easy (RF 0.73 / 0.84 / 0.75,
  XGB up to 0.89). Late-age plateau effect — must discuss, not cherry-pick.
- C2/C3 also drop (RF 0.32 / 0.09, XGB 0.43 / 0.33): SCM/superplasticizer
  extrapolation is real but milder than A1.
- MLP seed variance on random (0.73±0.17) — needs the ±std reported, no
  best-seed picking. 5-fold CV matches hold-out: XGB 0.935, RF 0.909,
  MLP 0.855, LR 0.603.
- Fig2 (RF trained ≤28d, per-bin RMSE): 3d 1.6, 7d 3.1, 14d 1.5, 28d 2.0,
  then 56d 8.0, 90d 10.6, 180+d 9.9 MPa — textbook error-vs-distance curve.
- Fig3 importance: age (0.35) + cement (0.31) dominate; extrapolating the
  top feature explains the A1 collapse.

## Environment
`micromamba` env `concrete-logo`, python 3.11 (conda-forge).
Pins in `requirements.txt` (2026-09-21).
