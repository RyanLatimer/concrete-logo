"""Deep analysis: SHAP, PDP, A3 heatmap, residuals, stats.
Run: micromamba run -n concrete-logo python analysis_deep.py
Requires: shap (installed 2026-09-21), sklearn, seaborn.
Outputs results/deep_*.csv + figures/fig{4,5,6,7}_*.png
(No titles — captions live in the paper, not the images.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from splits import FEATURES, load_concrete, get_xy, random_split

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

os.makedirs("results", exist_ok=True)


def train_rf_xgb(seed=0):
    from experiments import make_model
    df = load_concrete()
    tr, _, te = random_split(df, seed=seed)
    Xtr, ytr = get_xy(tr)
    rf = make_model("RF", seed); rf.fit(Xtr, ytr)
    xgb = make_model("XGB", seed); xgb.fit(Xtr, ytr)
    return df, tr, te, rf, xgb


def fig_shap():
    import shap
    df, tr, te, rf, xgb = train_rf_xgb()
    Xtr, _ = get_xy(tr)
    # TreeExplainer on the underlying estimator (after scaler, use transformed X)
    # Pipeline: scaler -> model. Transform first for exact SHAP.
    scaler = xgb.named_steps["standardscaler"]
    est = xgb.named_steps["xgbregressor"]
    Xt = scaler.transform(Xtr)
    Xt = pd.DataFrame(Xt, columns=FEATURES)
    expl = shap.TreeExplainer(est)
    sv = expl.shap_values(Xt)
    # Save mean|SHAP|
    imp = pd.DataFrame({"feature": FEATURES,
                        "mean_abs_shap": np.abs(sv).mean(0)}).sort_values(
        "mean_abs_shap", ascending=False)
    imp.to_csv("results/deep_shap_xgb.csv", index=False)
    print(imp.to_string(index=False))

    # Beeswarm (fancy)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    plt.sca(axes[0])
    shap.summary_plot(sv, Xt, show=False, plot_size=None, max_display=7)
    axes[1].barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1])
    axes[1].set_xlabel("Mean |SHAP value| (MPa)")
    fig.tight_layout()
    fig.savefig("figures/fig4_shap.png", dpi=200)
    print("saved figures/fig4_shap.png")


def fig_pdp():
    df, tr, te, rf, xgb = train_rf_xgb()
    Xtr, ytr = get_xy(tr)
    Xtr = Xtr.astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for i, feat in enumerate(["age", "cement", "water"]):
        PartialDependenceDisplay.from_estimator(
            xgb, Xtr, [feat], ax=ax[i], grid_resolution=50)
    fig.tight_layout()
    fig.savefig("figures/fig5_pdp.png", dpi=200)
    print("saved figures/fig5_pdp.png")


def fig_a3_heatmap():
    runs = pd.read_csv("results/summary.csv")
    a3 = runs[runs["split"].str.startswith("A3-")].copy()
    a3["bin"] = a3["split"].str.replace("A3-", "")
    order = ["3", "7", "14", "28", "56", "90", "180+"]
    piv = a3.pivot_table(index="model", columns="bin",
                         values="r2_mean").reindex(columns=order)
    piv.to_csv("results/deep_a3_r2.csv")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn", center=0.4,
                vmin=-0.5, vmax=0.9, ax=ax, cbar_kws={"label": "R² (mean, 3 seeds)"})
    ax.set_xlabel("Held-out age bin")
    fig.tight_layout()
    fig.savefig("figures/fig6_a3_heatmap.png", dpi=200)
    print("saved figures/fig6_a3_heatmap.png")
    print(piv.round(3).to_string())


def fig_residuals():
    # Residual diagnostics for RF: random vs A1 vs C1 (seed 0 preds)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False)
    for j, split in enumerate(["random", "A1", "C1"]):
        d = pd.read_csv(f"results/preds_RF_{split}_s0.csv")
        resid = d["y_pred"] - d["y_true"]
        # top: residuals vs predicted
        ax = axes[0, j]
        ax.scatter(d["y_pred"], resid, s=12, alpha=0.5)
        ax.axhline(0, color="r", ls="--", lw=1)
        ax.set_xlabel(f"Predicted (MPa)\n{split}", fontsize=9)
        ax.text(0.05, 0.95, f"bias {resid.mean():+.2f} MPa",
                transform=ax.transAxes, va="top", ha="left", fontsize=9)
        if j == 0:
            ax.set_ylabel("Residual (pred - true)")
        # bottom: calibration by decile
        ax2 = axes[1, j]
        d["decile"] = pd.qcut(d["y_true"], 5, duplicates="drop")
        cal = d.groupby("decile", observed=True).agg(
            true=("y_true", "mean"), pred=("y_pred", "mean"))
        ax2.plot(cal["true"], cal["pred"], "o-")
        lo, hi = d["y_true"].min(), d["y_true"].max()
        ax2.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax2.set_xlabel(f"True (binned mean)\n{split}", fontsize=9)
        if j == 0:
            ax2.set_ylabel("Pred (binned mean)")
    fig.tight_layout()
    fig.savefig("figures/fig7_residuals.png", dpi=200)
    print("saved figures/fig7_residuals.png")


def stats_table():
    """Paired seed-wise drop tests + bootstrap CI for headline drops."""
    runs = pd.read_csv("results/runs.csv")
    rng = np.random.default_rng(0)
    lines = ["model,comparison,mean_drop_R2,bootstrap95CI"]
    for model in ["RF", "XGB", "LR", "MLP"]:
        for base, logo in [("random", "A1"), ("random", "C1"), ("random", "A2")]:
            b = runs[(runs.model == model) & (runs.split == base)].sort_values(
                "seed")["r2"].to_numpy()
            l = runs[(runs.model == model) & (runs.split == logo)].sort_values(
                "seed")["r2"].to_numpy()
            # seeds differ for random (different test sets) -> unpaired; bootstrap the means
            diffs = []
            for _ in range(5000):
                diffs.append(rng.choice(b, len(b), replace=True).mean()
                             - rng.choice(l, len(l), replace=True).mean())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            lines.append(f"{model},{base}-{logo},{np.mean(b)-np.mean(l):.3f},"
                         f"[{lo:.3f},{hi:.3f}]")
    open("results/deep_drop_CI.csv", "w").write("\n".join(lines) + "\n")
    print(open("results/deep_drop_CI.csv").read())
    # permutation importance (RF, random-train) as SHAP cross-check
    df, tr, te, rf, xgb = train_rf_xgb()
    Xte, yte = get_xy(te)
    pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0,
                                n_jobs=-1)
    piv = pd.DataFrame({"feature": FEATURES,
                        "perm_mean": pi.importances_mean,
                        "perm_std": pi.importances_std}).sort_values(
        "perm_mean", ascending=False)
    piv.to_csv("results/deep_permutation_RF.csv", index=False)
    print(piv.round(4).to_string(index=False))


if __name__ == "__main__":
    fig_shap()
    fig_pdp()
    fig_a3_heatmap()
    fig_residuals()
    stats_table()
