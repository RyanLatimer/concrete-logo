"""EDA: verify LOGO group sizes + age distribution. Run with:
    micromamba run -n concrete-logo python eda.py
Saves figures/figS4_age_counts.png. Uses splits.py as single source of truth.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style as ps

from splits import (
    age_a1,
    age_a2,
    age_a3_bins,
    comp_c1_flyash,
    comp_c2_slag,
    comp_c3_superplasticizer,
    load_concrete,
    random_split,
)


def main():
    df = load_concrete()
    print(f"shape: {df.shape} (expect (1030, 8) incl. target)")
    print(f"nulls: {int(df.isnull().sum().sum())}, dupes: {int(df.duplicated().sum())}")
    print("\nAge value_counts (raw):")
    print(df["age"].value_counts().sort_index().to_string())

    print("\n--- Split sizes (train -> test) ---")
    tr, val, te = random_split(df, seed=0)
    print(f"random 70/15/15 seed=0: train={len(tr)} val={len(val)} test={len(te)}")
    for name, fn in [
        ("A1 train<=28/test>=90", age_a1),
        ("A2 train{3,7}/test28", age_a2),
        ("C1 FA==0/FA>0", comp_c1_flyash),
        ("C2 slag==0/slag>0", comp_c2_slag),
        ("C3 SP==0/SP>0", comp_c3_superplasticizer),
    ]:
        tr, te = fn(df)
        flag = "  <-- WARNING: test <100 rows" if len(te) < 100 else ""
        print(f"{name}: train={len(tr)} test={len(te)}{flag}")
    print("A3 leave-one-age-bin-out:")
    for b, (tr, te) in age_a3_bins(df).items():
        flag = "  <-- WARNING: test <100 rows" if len(te) < 100 else ""
        print(f"  hold out {b:>4}: train={len(tr)} test={len(te)}{flag}")

    # Mix identity: rows sharing all six composition columns are the same
    # recipe tested at different ages. Random splits put siblings on both sides.
    comp = ["cement", "slag", "fly_ash", "water", "superplasticizer",
            "coarse_agg"]
    key = df[comp].astype(str).agg("|".join, axis=1)
    print(f"\nunique mixes: {key.nunique()} "
          f"(tested at >1 age: {(df.groupby(comp)['age'].nunique() > 1).sum()})")
    for seed in (0, 1, 2):
        tr, _, te = random_split(df, seed=seed)
        share = key[te.index].isin(set(key[tr.index])).mean()
        print(f"random seed={seed}: {share:.0%} of test rows share a mix with train")

    ps.apply()
    counts = df["age"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.bar(counts.index.astype(str), counts.values, color=ps.BLUE, width=0.7)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 6, str(v), ha="center", fontsize=7, color=ps.INK_2)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Age at test (days)")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig("figures/figS4_age_counts.png")
    print("\nsaved figures/figS4_age_counts.png")


if __name__ == "__main__":
    main()
