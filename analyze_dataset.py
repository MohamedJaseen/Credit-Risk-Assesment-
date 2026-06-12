"""
analyze_dataset.py
==================
Reads data/dataset.csv and prints a full analysis report.
Generates analysis_plots.png in the data/ folder.

Run:  python analyze_dataset.py
"""

import os, sys
import pandas as pd
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
PLOT_PATH = os.path.join(BASE_DIR, "data", "analysis_plots.png")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def analyze(path=DATA_PATH):
    if not os.path.exists(path):
        print(f"❌  dataset.csv not found at: {path}")
        print("   Run  python train_model.py  first to generate it.")
        sys.exit(1)

    df = pd.read_csv(path)

    div = "=" * 62

    print(f"\n{div}")
    print("   CREDIT RISK DATASET ANALYSIS REPORT")
    print(f"{div}")

    # ── 1. Basic info ──────────────────────────────────────────────
    print(f"\n📁  Dataset Size")
    print(f"    Rows    : {len(df):,}")
    print(f"    Columns : {df.shape[1]}")
    print(f"    Memory  : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # ── 2. Feature descriptions ────────────────────────────────────
    print(f"\n📊  Feature Summary")
    print(f"    {'Feature':<28}  {'Min':>10}  {'Max':>12}  {'Mean':>10}  {'Std':>10}")
    print("    " + "-" * 72)
    for col in df.columns:
        if col == "label":
            continue
        mn, mx = df[col].min(), df[col].max()
        me, sd = df[col].mean(), df[col].std()
        print(f"    {col:<28}  {mn:>10.2f}  {mx:>12.2f}  {me:>10.2f}  {sd:>10.2f}")

    # ── 3. Missing values ──────────────────────────────────────────
    print(f"\n🔍  Missing Values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("    ✅  No missing values found.")
    else:
        for col, cnt in missing[missing > 0].items():
            print(f"    ⚠️   {col}: {cnt} missing ({cnt/len(df)*100:.1f}%)")

    # ── 4. Class imbalance ─────────────────────────────────────────
    print(f"\n⚖️   Class Distribution  (label: 0=No Default, 1=Default)")
    vc  = df["label"].value_counts()
    for k, v in vc.items():
        bar   = "█" * int(v / len(df) * 40)
        lname = "Default (High Risk)" if k == 1 else "No Default (Low Risk)"
        print(f"    {lname:<22}  {v:>6,}  ({v/len(df)*100:.1f}%)  {bar}")

    ratio = vc.get(1, 0) / max(vc.get(0, 1), 1)
    if 0.3 <= ratio <= 2.0:
        print("    ✅  Dataset is reasonably balanced.")
    elif ratio < 0.2:
        print("    ⚠️   Imbalance detected (few defaults). Consider SMOTE or class_weight.")
    else:
        print("    ⚠️   Unusual balance — verify label generation logic.")

    # ── 5. Top correlations ────────────────────────────────────────
    print(f"\n🔗  Top Feature Correlations with 'label'")
    corr = df.corr()["label"].drop("label").abs().sort_values(ascending=False)
    for feat, val in corr.head(8).items():
        bar = "▓" * int(val * 30)
        print(f"    {feat:<28}  r = {val:.3f}  {bar}")

    # ── 6. Skewness ────────────────────────────────────────────────
    print(f"\n📐  Skewness (values > 2 may benefit from log transform)")
    for col in df.drop("label", axis=1).columns:
        sk = df[col].skew()
        if abs(sk) > 2:
            print(f"    ⚠️   {col:<28}  skew = {sk:.2f}")

    # ── 7. Preprocessing suggestions ──────────────────────────────
    print(f"\n💡  Preprocessing Recommendations")
    print("    • Apply StandardScaler to all numeric features (done in train_model.py)")
    print("    • employment_encoded is already integer-encoded (0/1/2)")
    if ratio < 0.25:
        print("    • Class imbalance: use class_weight='balanced' or SMOTE")
    print("    • No categorical encoding needed — all features are numeric")
    print("    • Train/test split: 80/20 with stratification (done in train_model.py)")

    # ── 8. Data quality score ─────────────────────────────────────
    issues = sum([
        missing.sum() > 0,
        ratio < 0.2 or ratio > 3,
        any(abs(df[c].skew()) > 3 for c in df.drop("label",axis=1).columns),
    ])
    score = max(0, 10 - issues * 2)
    print(f"\n✅  Data Quality Score: {score}/10")
    print(f"{div}\n")

    # ── 9. Plots ───────────────────────────────────────────────────
    print("📈  Generating plots…")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Credit Risk Dataset Analysis", fontsize=14, fontweight="bold")

    # Class distribution
    vc.plot(kind="bar", ax=axes[0,0], color=["#22c55e","#ef4444"], edgecolor="none")
    axes[0,0].set_title("Class Distribution"); axes[0,0].set_xticklabels(["No Default","Default"],rotation=0)
    axes[0,0].grid(axis="y", alpha=0.3)

    # Correlation bar
    corr_all = df.corr()["label"].drop("label").sort_values()
    corr_all.plot(kind="barh", ax=axes[0,1],
                  color=["#22c55e" if v<0 else "#ef4444" for v in corr_all], edgecolor="none")
    axes[0,1].axvline(0, color="black", lw=0.7); axes[0,1].set_title("Feature Correlation with Default")

    # Missing values heatmap
    sns.heatmap(df.isnull().T, ax=axes[0,2], cbar=False, yticklabels=True,
                cmap="Reds"); axes[0,2].set_title("Missing Values Heatmap")

    # Income distribution
    df["monthly_income"].plot(kind="hist", bins=40, ax=axes[1,0],
                              color="#3b82f6", edgecolor="none")
    axes[1,0].set_title("Monthly Income Distribution"); axes[1,0].set_xlabel("Income (₹)")

    # Default rate by missed payments
    df.groupby("missed_payments")["label"].mean().plot(
        kind="bar", ax=axes[1,1], color="#f59e0b", edgecolor="none")
    axes[1,1].set_title("Default Rate by Missed Payments")
    axes[1,1].set_ylabel("Default Rate"); axes[1,1].grid(axis="y",alpha=0.3)

    # Credit utilization vs default
    sample = df.sample(min(800, len(df)), random_state=42)
    axes[1,2].scatter(sample["credit_utilization"], sample["monthly_income"],
                      c=sample["label"], cmap="RdYlGn_r", alpha=0.4, edgecolors="none", s=18)
    axes[1,2].set_xlabel("Credit Utilization %"); axes[1,2].set_ylabel("Monthly Income")
    axes[1,2].set_title("Utilization vs Income (🔴=Default)")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {PLOT_PATH}")


if __name__ == "__main__":
    analyze()
