# concrete-logo — Leave-One-Group-Out Generalization on Concrete Strength

## Goal
Quantify the validation gap flagged by Li et al. 2022: does a model trained with random splits still work when asked to extrapolate to new ages or new mix types?

This is project type C: generalization test. Publishable for student journals (JEI, NHSJS, YSJ) if done with proper controls.

## Dataset
- Hugging Face: `foundry-ml/dataset_concrete_compressive_strength`
- Source: UCI Concrete Compressive Strength (Yeh 1998), repackaged by Foundry-ML
- Size: 1,030 rows, ~29.5 kB, tabular parquet. Runs on laptop / free Colab.
- DOI: `10.18126/8k1f-mx77`
- License: check HF page (Other / CC-BY-4.0 for UCI original). Cite both.

Fields (8 inputs -> 1 target):
- Inputs: Cement (kg/m3), Blast Furnace Slag (kg/m3), Fly Ash (kg/m3), Water (kg/m3), Superplasticizer (kg/m3), Coarse Aggregate (kg/m3), Fine Aggregate (kg/m3), Age (days: 3,7,14,28,56,90,180,270,365 — heavily skewed to 28d)
- Target: Concrete compressive strength (MPa, 2.3–82.6)

Must-cite:
1. Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. Cem. Concr. Res. 28(12).
2. Li et al. (2022). Machine learning in concrete science: applications, challenges, and best practices. npj Comput. Mater. 8:127. https://doi.org/10.1038/s41524-022-00810-x — review that names temporal bias + deployment bias (lab vs field) as open problems, but does no LOGO benchmark itself.

## Research question + hypothesis
Q: How much does performance drop when we replace random CV with leave-one-group-out (LOGO) by age and by composition?
H: Tree ensembles will get R2 ~0.90+ on random 5-fold, but drop to ~0.6–0.8 on age extrapolation (train ≤28d, test 90d+) and SCM extrapolation (train without fly ash, test with fly ash).

A drop IS the result — it proves random CV overstates real-world performance.

## Experimental design (keep this exact for publishability)

### Splits — do all three, same models/features
1. Baseline (control): random 70/15/15 train/val/test + 5-fold CV.
2. Age-LOGO:
   - A1: train Age ≤28d, test Age >=90d
   - A2: train on 3d+7d only, predict 28d
   - A3: leave-one-age-bin-out (each bin 3,7,14,28,56,90,180+ as held-out)
3. Composition-LOGO:
   - C1: train Fly Ash == 0, test Fly Ash > 0
   - C2: train Slag == 0, test Slag > 0
   - C3 (optional): train w/o superplasticizer, test with

### Models (same splits for all)
- Linear Regression (baseline)
- Random Forest
- XGBoost or GradientBoostingRegressor
- Small MLP (optional 4th)
- 3 random seeds (e.g. 0,1,2), report mean ± std.

### Metrics
RMSE + MAE + R2 on every split. Never R2 alone. One table: rows=models, cols=Random / Age-LOGO / SCM-LOGO.

### Anti-leakage rules
- Fit scalers / imputers on train fold only, apply to test.
- No target-derived features.
- Fix `random_state`, log `sklearn`, `xgboost`, `pandas`, `datasets` versions.

### Figures (2–3 total)
1. Predicted-vs-true scatter: random test vs LOGO test (side-by-side).
2. Error-vs-age plot (does error grow with distance from train ages?).
3. Feature importance / SHAP (brief — supports discussion of why water/cement + age dominate).

## Why this is novel enough
- Original Li et al.: 80/20 random demo on 40 mixes only, no LOGO.
- 100s of follow-ups: random split, R2 0.90–0.94, + SHAP. Extrapolation noted as limitation, rarely tested.
- Claim honestly: "We systematically quantify age and SCM extrapolation on the public 1030-row set." Not "first ML on concrete."
- Novelty check before writing: Scholar search `"concrete compressive strength" "leave-one-group" OR "age extrapolation" UCI 1030`.

## Paper outline (1500–2500 words, IMRaD)
Abstract, Intro (why strength matters for cost/CO2, Li gap), Data (DOI, cleaning, age histogram), Methods (splits, models, metrics, seeds), Results (table + 2 figs, no interpretation), Discussion (why drop happens, limits: n=1030, lab-only, no field validation, no new experiments), Conclusion, References, Code/Data Availability (GitHub + HF link + Colab).

## What you need to get accepted
- [ ] Fixed splits + code on GitHub with `requirements.txt`, `README` with exact `load_dataset("foundry-ml/dataset_concrete_compressive_strength")` call
- [ ] Baselines + error bars, not best-seed cherry-pick
- [ ] Cite dataset DOI + Yeh 1998 + Li 2022 + 2–3 random-split papers you compare to
- [ ] Limitations paragraph (lab ≠ field, small n)
- [ ] Mentor sign-off (required by JEI/NHSJS)
- [ ] No overclaim ("we discovered a new concrete") — frame as methods/validation contribution

## Quickstart
```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("foundry-ml/dataset_concrete_compressive_strength", split="train")
df = ds.to_pandas()
print(df.shape)  # expect (1030, 9)
print(df["Age (day)"].value_counts().sort_index())
```

Suggested repo layout:
```
concrete-logo/
  README.md (this file)
  notebook.ipynb
  requirements.txt
  splits.py  # define Random vs Age-LOGO vs SCM-LOGO here so reviewers can rerun
```

## Timeline (6–10 wks)
Wks 1–2: EDA + age histogram + group counts (check C1/C2 have enough test rows)
Wks 3–5: models + LOGO runs
Wks 6–7: figures + table
Wks 8–10: write + mentor review + submit to JEI / NHSJS / science fair

## Pitfalls that get rejected
Tuning on test set, scaling before splitting, reporting only best seed, no baseline, missing DOI, claiming field applicability from lab data.

## Next step
Run EDA to confirm test-group sizes (e.g. how many rows have Age>=90d? Fly Ash>0?). If a group is <100 rows, merge bins (e.g. 90d+ together) — note it in Methods.
