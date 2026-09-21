"""Full LOGO experiment matrix: LR / RF / XGB / MLP x splits x 3 seeds.

Splits (from splits.py, single source of truth):
  random (70/15/15 test), A1, A2, C1, C2, C3, A3-{bin} x7
  + 5-fold CV on full data (random control)

Anti-leakage: StandardScaler fit on train only via Pipeline.
Metrics: RMSE + MAE + R2 on test. Mean +/- std across SEEDS.

Run: micromamba run -n concrete-logo python experiments.py
Outputs: results/runs.csv, results/summary.csv, results/cv.csv,
         results/preds_*.csv (for figures), results/versions.txt
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from splits import (
    SEEDS,
    FEATURES,
    TARGET,
    age_a1,
    age_a2,
    age_a3_bins,
    comp_c1_flyash,
    comp_c2_slag,
    comp_c3_superplasticizer,
    get_xy,
    load_concrete,
    random_split,
)


def make_model(name, seed):
    if name == "LR":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "RF":
        return make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=200, random_state=seed, n_jobs=-1))
    if name == "XGB":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        return make_pipeline(
            StandardScaler(),
            XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=-1))
    if name == "MLP":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                         alpha=0.001, max_iter=2000, early_stopping=True,
                         validation_fraction=0.15, n_iter_no_change=20,
                         random_state=seed))
    raise ValueError(name)


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def eval_split(model, tr, te):
    Xtr, ytr = get_xy(tr)
    Xte, yte = get_xy(te)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    m = metrics(yte, pred)
    return m, np.asarray(yte), np.asarray(pred)


def main():
    os.makedirs("results", exist_ok=True)
    df = load_concrete()
    print(f"n={len(df)} features={FEATURES}")

    # Define all test splits (functions returning (train, test))
    a3 = age_a3_bins(df)
    split_fns = {
        "random": None,  # handled per-seed below
        "A1": lambda d: age_a1(d),
        "A2": lambda d: age_a2(d),
        "C1": lambda d: comp_c1_flyash(d),
        "C2": lambda d: comp_c2_slag(d),
        "C3": lambda d: comp_c3_superplasticizer(d),
    }
    for b, (tr, te) in a3.items():
        split_fns[f"A3-{b}"] = (lambda tr_, te_: lambda d: (tr_, te_))(tr, te)

    models = ["LR", "RF", "XGB", "MLP"] if HAS_XGB else ["LR", "RF", "MLP"]

    rows = []
    t0 = time.time()
    for model_name in models:
        for split_name, fn in split_fns.items():
            for seed in SEEDS:
                if split_name == "random":
                    tr, _, te = random_split(df, seed=seed)
                    ntrain, ntest = len(tr), len(te)
                else:
                    tr, te = fn(df)
                    ntrain, ntest = len(tr), len(te)
                model = make_model(model_name, seed)
                m, yte, pred = eval_split(model, tr, te)
                rows.append({"model": model_name, "split": split_name,
                             "seed": seed, "ntrain": ntrain, "ntest": ntest,
                             **m})
                print(f"{model_name:4} {split_name:6} seed={seed} "
                      f"R2={m['r2']:.3f} RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} "
                      f"(train={ntrain} test={ntest})", flush=True)
                # Save preds for figure splits at seed 0 (RF + all models for random/A1/C1)
                if seed == 0 and split_name in ("random", "A1", "C1"):
                    pd.DataFrame({"y_true": yte, "y_pred": pred}).to_csv(
                        f"results/preds_{model_name}_{split_name}_s0.csv",
                        index=False)
    runs = pd.DataFrame(rows)
    runs.to_csv("results/runs.csv", index=False)

    # Summary: mean +/- std across seeds
    summary = runs.groupby(["model", "split"]).agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        ntrain=("ntrain", "first"), ntest=("ntest", "first")).reset_index()
    summary.to_csv("results/summary.csv", index=False)

    # 5-fold CV on full data (random control), per model x seed
    X, y = get_xy(df)
    cv_rows = []
    for model_name in models:
        for seed in SEEDS:
            model = make_model(model_name, seed)
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
            sc = cross_validate(
                model, X, y, cv=cv,
                scoring={"rmse": "neg_root_mean_squared_error",
                         "mae": "neg_mean_absolute_error", "r2": "r2"},
                n_jobs=-1)
            cv_rows.append({"model": model_name, "seed": seed,
                            "rmse": float(-sc["test_rmse"].mean()),
                            "mae": float(-sc["test_mae"].mean()),
                            "r2": float(sc["test_r2"].mean())})
            print(f"CV {model_name} seed={seed}: "
                  f"R2={cv_rows[-1]['r2']:.3f} RMSE={cv_rows[-1]['rmse']:.2f}",
                  flush=True)
    pd.DataFrame(cv_rows).to_csv("results/cv.csv", index=False)

    # Error-vs-age data: RF seed 0 trained on A1-train, scored per age bin on full data
    from splits import age_to_bin
    model = make_model("RF", 0)
    tr, _ = age_a1(df)
    Xtr, ytr = get_xy(tr)
    model.fit(Xtr, ytr)
    dfall = df.copy()
    dfall["age_bin"] = dfall["age"].map(age_to_bin)
    dfall["pred"] = model.predict(dfall[FEATURES])
    dfall["abs_err"] = (dfall["strength"] - dfall["pred"]).abs()
    per_bin = dfall.groupby("age_bin").apply(
        lambda g: pd.Series({"rmse": float(np.sqrt(mean_squared_error(
            g["strength"], g["pred"]))), "mae": float(g["abs_err"].mean()),
            "r2": float(r2_score(g["strength"], g["pred"]))
            if len(g) > 2 else float("nan"), "n": len(g)},
            dtype=object), include_groups=False)
    order = ["3", "7", "14", "28", "56", "90", "180+"]
    per_bin = per_bin.reindex(order)
    per_bin.to_csv("results/error_vs_age_RF_A1train.csv")
    dfall[["age", "age_bin", "strength", "pred",
           "abs_err"]].to_csv("results/preds_RF_A1train_all.csv", index=False)

    # RF feature importance (seed 0, random-train) for Fig 3
    tr, _, _ = random_split(df, seed=0)
    m_rf = make_model("RF", 0)
    Xtr, ytr = get_xy(tr)
    m_rf.fit(Xtr, ytr)
    rf = m_rf.named_steps["randomforestregressor"]
    pd.DataFrame({"feature": FEATURES,
                  "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(
        "results/importance_RF_s0.csv", index=False)

    # Versions
    import sklearn, pandas, numpy, matplotlib, datasets
    with open("results/versions.txt", "w") as f:
        f.write(f"sklearn={sklearn.__version__}\n"
                f"pandas={pandas.__version__}\n"
                f"numpy={numpy.__version__}\n"
                f"matplotlib={matplotlib.__version__}\n"
                f"datasets={datasets.__version__}\n")
        if HAS_XGB:
            import xgboost
            f.write(f"xgboost={xgboost.__version__}\n")

    print(f"\ndone in {time.time()-t0:.1f}s. wrote results/*.csv")
    print("\n=== SUMMARY (mean+/-std across 3 seeds) ===")
    piv = summary.pivot_table(index="model",
                              columns="split", values="r2_mean")
    print(piv.round(3).to_string())
    print("\nFull table: results/summary.csv")


if __name__ == "__main__":
    main()
