"""Paper figures. Run: micromamba run -n concrete-logo python figures.py
Uses results/preds_*.csv from experiments.py (seed 0, RF).
Fig1: pred-vs-true random vs A1 vs C1. Fig2: error-vs-age.
(No titles — captions live in the paper, not the images.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, split, label in zip(axes,
                                ["random", "A1", "C1"],
                                ["Random test", "Age-LOGO A1", "SCM-LOGO C1"]):
        import os
        p = f"results/preds_RF_{split}_s0.csv"
        if not os.path.exists(p):
            # fallback: XGB file
            p = p.replace("RF_", "XGB_")
        d = pd.read_csv(p)
        ax.scatter(d["y_true"], d["y_pred"], s=10, alpha=0.5)
        lo = min(d.min().min(), 0)
        hi = max(d.max().max(), 85)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"True strength (MPa)\n{label}", fontsize=9)
    axes[0].set_ylabel("Predicted strength (MPa)")
    fig.tight_layout()
    fig.savefig("figures/fig1_pred_vs_true.png", dpi=150)
    print("saved figures/fig1_pred_vs_true.png")


def fig2():
    d = pd.read_csv("results/error_vs_age_RF_A1train.csv", index_col=0)
    order = ["3", "7", "14", "28", "56", "90", "180+"]
    d = d.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(order, d["rmse"], "o-", label="RMSE")
    ax.plot(order, d["mae"], "s--", label="MAE")
    for x, n in zip(order, d["n"]):
        ax.annotate(f"n={int(n)}", (x, 0.5), fontsize=7, ha="center")
    ax.axvline("28", color="r", ls="--", lw=1, label="train edge (<=28d)")
    ax.set_xlabel("Age bin (days)")
    ax.set_ylabel("Error (MPa)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig2_error_vs_age.png", dpi=150)
    print("saved figures/fig2_error_vs_age.png")
    print(d.round(3).to_string())


if __name__ == "__main__":
    fig1()
    fig2()
