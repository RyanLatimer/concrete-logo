"""Deep analysis: SHAP, PDP, A3 heatmap, residuals, stats.
Run: micromamba run -n concrete-logo python analysis_deep.py
Requires: shap (installed 2026-09-21), sklearn, seaborn.
Outputs results/deep_*.csv, paper figures fig3_a3_heatmap + fig4_age_response,
and supplement figures figS1_shap, figS2_pdp, figS3_residuals.
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

from matplotlib.colors import LinearSegmentedColormap

import plot_style as ps

ps.apply()

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
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color=ps.BLUE,
            height=0.6)
    for y, v in enumerate(imp["mean_abs_shap"][::-1]):
        ax.text(v + 0.1, y, f"{v:.1f}", va="center", fontsize=7.5,
                color=ps.INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value| (MPa)")
    fig.tight_layout()
    fig.savefig("figures/figS1_shap.png")
    print("saved figures/figS1_shap.png")


def fig_pdp():
    df, tr, te, rf, xgb = train_rf_xgb()
    Xtr, ytr = get_xy(tr)
    Xtr = Xtr.astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for i, feat in enumerate(["age", "cement", "water"]):
        PartialDependenceDisplay.from_estimator(
            xgb, Xtr, [feat], ax=ax[i], grid_resolution=50,
            line_kw={"color": ps.BLUE, "lw": 2})
    fig.tight_layout()
    fig.savefig("figures/figS2_pdp.png")
    print("saved figures/figS2_pdp.png")


def fig_a3_heatmap():
    runs = pd.read_csv("results/summary.csv")
    a3 = runs[runs["split"].str.startswith("A3-")].copy()
    a3["bin"] = a3["split"].str.replace("A3-", "")
    order = ["3", "7", "14", "28", "56", "90", "180+"]
    piv = a3.pivot_table(index="model", columns="bin",
                         values="r2_mean").reindex(columns=order)
    piv.to_csv("results/deep_a3_r2.csv")
    piv = piv.reindex(["XGB", "RF", "MLP", "LR"])
    # Diverging around R² = 0 (no better than predicting the mean); clipped
    # at -1 so the catastrophic LR/MLP cells don't wash out the scale.
    cmap = LinearSegmentedColormap.from_list(
        "r2", [ps.DIV_NEG, ps.DIV_MID, ps.DIV_POS])
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    sns.heatmap(piv.clip(lower=-1), annot=piv, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, center=0, ax=ax, linewidths=2,
                linecolor="white", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Test R² (clipped at −1)", "ticks": [-1, 0, 1]})
    ax.set_xlabel("Held-out age bin (days)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig("figures/fig3_a3_heatmap.png")
    print("saved figures/fig3_a3_heatmap.png")
    print(piv.round(3).to_string())


def fig_residuals():
    # Residual diagnostics for RF: random vs A1 vs C1 (seed 0 preds)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=False)
    for j, split in enumerate(["random", "A1", "C1"]):
        d = pd.read_csv(f"results/preds_RF_{split}_s0.csv")
        resid = d["y_pred"] - d["y_true"]
        # top: residuals vs predicted
        ax = axes[0, j]
        ax.scatter(d["y_pred"], resid, s=8, alpha=0.55, color=ps.BLUE,
                   linewidths=0)
        ax.axhline(0, color=ps.MUTED, ls="--", lw=1)
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
        ax2.plot(cal["true"], cal["pred"], "o-", color=ps.BLUE)
        lo, hi = d["y_true"].min(), d["y_true"].max()
        ax2.plot([lo, hi], [lo, hi], color=ps.MUTED, ls="--", lw=1)
        ax2.set_xlabel(f"True (binned mean)\n{split}", fontsize=9)
        if j == 0:
            ax2.set_ylabel("Pred (binned mean)")
    fig.tight_layout()
    fig.savefig("figures/figS3_residuals.png")
    print("saved figures/figS3_residuals.png")


def fig_age_response(seed=0):
    """Fig 4: RF partial dependence on age when trained on all ages (random
    70% split) vs trained on Age<=28 only (A1). Shows the tree model's
    prediction freezing at the edge of the training ages."""
    from experiments import make_model
    from splits import age_a1
    df = load_concrete()
    tr_rand, _, _ = random_split(df, seed=seed)
    tr_a1, _ = age_a1(df)
    grid = np.arange(1, 366)
    curves = {}
    for name, tr in [("all", tr_rand), ("a1", tr_a1)]:
        m = make_model("RF", seed)
        m.fit(tr[FEATURES], tr["strength"])
        X = df[FEATURES].copy()  # average over every mix in the dataset
        pdp = []
        for a in grid:
            X["age"] = a
            pdp.append(m.predict(X).mean())
        curves[name] = np.array(pdp)
    pd.DataFrame({"age": grid, "rf_all_ages": curves["all"],
                  "rf_age_le28": curves["a1"]}).to_csv(
        "results/deep_age_response_RF.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.axvspan(0, 28, color=ps.SHADE, lw=0, zorder=0)
    ax.text(5.3, 52.5, "training ages (A1)", ha="center", va="top",
            fontsize=7.5, color=ps.INK_2)
    ax.plot(grid, curves["all"], color=ps.BLUE, label="Trained on all ages")
    ax.plot(grid, curves["a1"], color=ps.ORANGE, label="Trained on ≤28 d only")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 7, 14, 28, 56, 90, 180, 365],
                  ["1", "3", "7", "14", "28", "56", "90", "180", "365"])
    ax.minorticks_off()
    ax.set_xlim(1, 365)
    ax.set_ylim(15, 53)
    ax.set_xlabel("Age (days, log scale)")
    ax.set_ylabel("Mean predicted strength (MPa)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/fig4_age_response.png")
    print("saved figures/fig4_age_response.png")
    print(f"at 365 d: all-ages {curves['all'][-1]:.1f}, <=28 {curves['a1'][-1]:.1f}; "
          f"at 28 d: all {curves['all'][27]:.1f}, <=28 {curves['a1'][27]:.1f}")


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
    fig_age_response()
    stats_table()
