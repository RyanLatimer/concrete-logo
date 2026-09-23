"""Paper figures 1-2. Run: micromamba run -n concrete-logo python figures.py
Fig1: RF pred-vs-true, random vs A1 vs C1 (seed 0, from results/preds_RF_*).
Fig2: RF trained on Age<=28, error per age bin. In-range bins use 5-fold
      out-of-fold predictions so every bin is scored on unseen rows
      (writes results/error_vs_age_RF_A1train.csv).
(No titles — captions live in the paper, not the images.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import plot_style as ps
from splits import FEATURES, TARGET, age_a1, age_to_bin, load_concrete

ps.apply()
ORDER = ["3", "7", "14", "28", "56", "90", "180+"]


def fig1():
    panels = [("random", "Random test (70/15/15)"),
              ("A1", "A1: train ≤28 d → test ≥90 d"),
              ("C1", "C1: train no fly ash → test fly ash")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True,
                             sharey=True)
    for ax, (split, label) in zip(axes, panels):
        d = pd.read_csv(f"results/preds_RF_{split}_s0.csv")
        err = d["y_pred"] - d["y_true"]
        r2 = 1 - (err ** 2).sum() / ((d["y_true"] - d["y_true"].mean()) ** 2).sum()
        rmse = np.sqrt((err ** 2).mean())
        ax.plot([0, 90], [0, 90], color=ps.MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(d["y_true"], d["y_pred"], s=9, color=ps.BLUE, alpha=0.55,
                   linewidths=0, zorder=2)
        ax.text(0.04, 0.96,
                f"{label}\nR² {r2:.2f}   RMSE {rmse:.1f} MPa\n"
                f"bias {err.mean():+.1f} MPa",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                color=ps.INK, linespacing=1.4)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 90)
        ax.set_aspect("equal")
        ax.set_xticks([0, 30, 60, 90])
        ax.set_yticks([0, 30, 60, 90])
        ax.set_xlabel("Measured strength (MPa)")
    axes[0].set_ylabel("Predicted strength (MPa)")
    fig.tight_layout(w_pad=1.2)
    fig.savefig("figures/fig1_pred_vs_true.png")
    print("saved figures/fig1_pred_vs_true.png")


def error_vs_age_oof(seed=0):
    """RF trained on Age<=28. Bins <=28: 5-fold OOF within the training set.
    Bins >28 (56, 90, 180+): predictions from the model fit on all Age<=28."""
    from experiments import make_model
    df = load_concrete()
    df["age_bin"] = df["age"].map(age_to_bin)
    tr, _ = age_a1(df)
    pred = pd.Series(np.nan, index=df.index)
    for fit_idx, oof_idx in KFold(5, shuffle=True, random_state=seed).split(tr):
        m = make_model("RF", seed)
        m.fit(tr.iloc[fit_idx][FEATURES], tr.iloc[fit_idx][TARGET])
        pred.loc[tr.index[oof_idx]] = m.predict(tr.iloc[oof_idx][FEATURES])
    m = make_model("RF", seed)
    m.fit(tr[FEATURES], tr[TARGET])
    out = df.index.difference(tr.index)
    pred.loc[out] = m.predict(df.loc[out, FEATURES])
    df["pred"] = pred
    df["err"] = df["pred"] - df[TARGET]
    rows = []
    for b in ORDER:
        g = df[df["age_bin"] == b]
        rows.append({"age_bin": b,
                     "rmse": float(np.sqrt((g["err"] ** 2).mean())),
                     "mae": float(g["err"].abs().mean()),
                     "bias": float(g["err"].mean()),
                     "n": len(g),
                     "scored_as": "5-fold OOF" if b in ("3", "7", "14", "28")
                     else "extrapolation"})
    res = pd.DataFrame(rows)
    res.to_csv("results/error_vs_age_RF_A1train.csv", index=False)
    print(res.round(2).to_string(index=False))
    return res


def fig2():
    d = error_vs_age_oof()
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.axvspan(-0.5, 3.5, color=ps.SHADE, zorder=0, lw=0)
    ax.text(1.5, 11.9, "training ages (≤28 d)", ha="center", va="top",
            fontsize=7.5, color=ps.INK_2)
    ax.text(5, 11.9, "unseen ages", ha="center", va="top", fontsize=7.5,
            color=ps.INK_2)
    ax.axhline(0, color=ps.AXIS, lw=0.8, zorder=1)
    ax.plot(x, d["rmse"], "o-", color=ps.BLUE, label="RMSE", zorder=3)
    ax.plot(x, d["bias"], "s-", color=ps.ORANGE,
            label="Mean error (predicted − measured)", zorder=3)
    for xi, (r, n) in enumerate(zip(d["rmse"], d["n"])):
        ax.annotate(f"n={n}", (xi, r), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=6.5, color=ps.MUTED)
    ax.set_xticks(x, ORDER)
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.set_ylim(-12, 12.5)
    ax.set_xlabel("Age bin (days)")
    ax.set_ylabel("Error (MPa)")
    ax.legend(loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig("figures/fig2_error_vs_age.png")
    print("saved figures/fig2_error_vs_age.png")


if __name__ == "__main__":
    fig1()
    fig2()
