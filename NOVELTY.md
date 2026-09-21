# NOVELTY.md — is the LOGO benchmark novel? (verified 2026-09-21, re-verified deep 2026-09-21)

## Search protocol (reproducible)
Date: 2026-09-21. Engine: web search (live + cached), 8 queries:
1. `"concrete compressive strength" "leave-one-group" OR "age extrapolation" UCI 1030`
2. `concrete compressive strength machine learning extrapolation new mixes fly ash slag generalization test`
3. `Li et al 2022 machine learning concrete science temporal bias deployment bias`
4. `"concrete compressive strength" UCI 1030 random split R2 0.9 XGBoost Random Forest review`
5. `"leave-one-out" OR "leave-one-group-out" concrete compressive strength age curing days`
6. `concrete strength prediction "train" "test" different ages 28 days 90 days machine learning`
7. `"concrete compressive strength" machine learning extrapolation out-of-distribution generalization age split`
8. `concrete compressive strength cross-dataset generalization Brazil UCI Yeh 1998 machine learning`
Plus full-text fetch of the closest hit (Fu et al. 2025, PMC12608167).

## What the literature does (dominant pattern, ~dozens of hits)
- Same UCI 1030 set (Yeh 1998), **random** 80/20 or 70/30 split, R² 0.90–0.94 (RF/XGB/ANN + SHAP), stop.
- Examples: Deepforest SciRep 2024 (8:2 random), TabPFN study (721/309 random 7:3),
  Infrastructure 2025 (80/20 stratified), Qi et al. RF-PSO (R 0.954 test), Feng/AdaBoost (1030, random),
  UHPC stacking (100× random splits, mean only). **None holds out an age or a mix family.**
- Li et al. 2022 (npj Comput. Mater. 8:127) is a *review*: names temporal bias + lab-vs-field
  deployment bias as open problems, runs only an 80/20 random demo on 40 mixes. **No LOGO benchmark.**

## Closest prior work (must cite as related, not as killers)
| # | Work | What they did | Why ours still differs |
|---|---|---|---|
| 1 | Refai et al. 2023, Materials 16:4977 — Brazilian dual-dataset (324 BR + Yeh 1030, MLP) | Cross-lab test: "Brazilian sample does not generalize well to the repository dataset and vice versa"; recommends limiting models to same lab | Cross-*lab* deployment bias on 7/28d data. No within-UCI age-LOGO (≤28→90+) or SCM-LOGO (FA=0→FA>0), no multi-model × multi-split matrix, no seeds/error bars |
| 2 | Cambridge Data-Centric Engineering — RF + just-in-time, 28/56d | Trains on own mixes, predicts 2 *unseen* mixes' 56d strength within 90% CI; uses 28d strength and slump as inputs | Prospective validation on n=2 mixes, single horizon, own data, leaks future strength as a feature. Ours: retrospective LOGO on public n=1030 *without* strength inputs, 13 splits × 4 models × 3 seeds |
| 3 | 2026 transfer-learning framework (early 3–7d → 28d) | Learns 28d from early-age curves + a few 28d labels | Our A2 is the *pure* version (train 3+7d only, zero 28d labels) and it **fails** (RF −0.18, XGB +0.01). Their success with transfer labels complements — cite as "extrapolation needs anchors" |
| 4 | Fu et al. 2025, Materials 18:5009 (223 own mixes, CatBoost R² 0.939, SHAP + BayesOpt) | 28-day-only mix optimization, 8:2 random, bootstrap CI | No age variation at all (all 28d), no extrapolation test. Best-practice template for reporting (bootstrap CI, SHAP) — we adopt the reporting, not the data |
| 5 | Yeh's own fly-ash 0–50% studies (3–56d) | Experimental + ANN fit of ash effect | Materials science of ash, not an ML generalization benchmark with fixed splits/seeds |

Searched explicitly for: `leave-one-group`, `leave-one-age-out`, `train <=28 test >=90`,
`train FA==0 test FA>0` on UCI-1030 — **zero hits**. The A3 asymmetry we found
(early hold-outs fail, late hold-outs look easy) is not reported anywhere found.

## Honest claim (use verbatim in paper)
> "We systematically quantify age extrapolation (train ≤28 d → test ≥90 d;
> train 3+7 d → 28 d; leave-one-age-bin-out) and SCM extrapolation
> (train without fly ash/slag → test with) for fixed models and features on the
> public 1030-row UCI/Foundry-ML set (7 inputs — fine aggregate absent vs Yeh 1998),
> reporting RMSE+MAE+R² mean ± std over 3 seeds plus 5-fold CV controls."
Do NOT claim: "first ML on concrete", "first SHAP on concrete", "field-ready model".
DO cite: Yeh 1998 (data) + Foundry-ML DOI 10.18126/8k1f-mx77 (repackaging) + Li et al. 2022 (gap)
+ Refai 2023 (cross-lab failure) + Cambridge JIT (prospective n=2) + Fu 2025 (reporting template).

## Threats / what could still kill novelty
- A thesis or non-indexed paper with the same splits (mitigation: re-run Scholar +
  Semantic Scholar before submission; splits are deterministic so a tie still leaves our
  SHAP+PDP+residual+Bias-CI package as incremental).
- HF repackaging missing fine aggregate (we disclose; a reviewer using full UCI-8 could
  ask for a sensitivity run — `splits.py` accepts extra columns with one-line change).
- Small-bin A3 cells (14: n=62, 180+: n=59): we report 3-seed std + bootstrap CI and
  state the caveat; do not over-interpret rank flips between RF and XGB on n=62.

## Deep-verification artifacts (this repo)
- `results/summary.csv` (13 splits × 4 models × 3 seeds), `results/cv.csv`,
  `results/deep_drop_CI.csv` (5000-rep bootstrap CIs: RF random−A1 drop 0.553 [0.522,0.570],
  XGB random−A1 0.532 [0.516,0.548] — CIs exclude 0 by a mile),
  `results/deep_shap_xgb.csv` (age 7.73, cement 6.98 MPa mean|SHAP|),
  `results/deep_permutation_RF.csv` (age 0.69, cement 0.50 drop in R² when shuffled),
  `results/deep_bias_by_age.csv` (XGB A1-bias ≈0 in-train, −6.6/−9.3/−8.5 MPa on 56/90/180+),
  `results/deep_a3_r2.csv` (heatmap source).
- Figures 1–8 regenerable via `experiments.py` → `figures.py` → `analysis_deep.py`.
