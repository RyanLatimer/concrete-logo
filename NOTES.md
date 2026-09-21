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
`>=180 -> "180+"`. OPEN: keep with wide-CI caveat vs merge further.

## A1 has a gap by design
Train Age<=28 (n=749), test Age>=90 (n=190). Ages 56..89 fall in neither set
— extrapolation distance, not interpolation. State explicitly in paper.

## Baseline sanity check (RF-200, seed 0, scaler fit on train only)
random R2 **0.909** (RMSE 5.08) → A1 **0.340** (10.36) → C1 **0.568** (9.34).
Drop is real; age extrapolation harsher than the hypothesized 0.6–0.8.
Single seed only — full 3-seed matrix not yet run.

## Environment
`micromamba` env `concrete-logo`, python 3.11 (conda-forge).
Pins in `requirements.txt` (2026-09-21).
