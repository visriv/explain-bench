import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

os.makedirs("results/plots", exist_ok=True)

df = pd.read_csv("results/leaderboard_clean.tsv", sep="\t")

# -------------------------------------------------------
# Helper: extract metric curves of the form name@k
# -------------------------------------------------------
def extract_metric_curve(df, metric_prefix):
    """
    Extracts columns like metric_faithfulness_drop@0.1
    Returns: dict explainer -> (ks, values)
    """
    pattern = re.compile(rf"^{metric_prefix}@([0-9.]+)$")

    curves = {}
    for explainer, subdf in df.groupby("explainer"):
        ks, vals = [], []

        for col in subdf.columns:
            m = pattern.match(col)
            if m:
                k = float(m.group(1))
                ks.append(k)
                vals.append(subdf[col].mean())

        if ks:
            ks, vals = zip(*sorted(zip(ks, vals)))
            curves[explainer] = (ks, vals)

    return curves


# -------------------------------------------------------
# 1) CLASSIFIER PERFORMANCE (heatmap)
# -------------------------------------------------------
if "val_auroc" in df.columns:
    pivot = df.pivot_table(
        index="model",
        columns="explainer",
        values="val_auroc",
        aggfunc="mean"
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, cmap="viridis")
    plt.title("Classifier AUROC (Model × Explainer)")
    plt.tight_layout()
    plt.savefig("results/plots/heatmap_val_auroc.png", dpi=200)
    plt.close()


# -------------------------------------------------------
# 2) FAITHFULNESS CURVE
# -------------------------------------------------------
faith_curves = extract_metric_curve(df, "metric_faithfulness_drop")

plt.figure(figsize=(7, 5))
for expl, (ks, vals) in faith_curves.items():
    plt.plot(ks, vals, marker="o", label=expl)

plt.xlabel("Fraction masked (k)")
plt.ylabel("Probability drop")
plt.title("Faithfulness vs k")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/faithfulness_curve.png", dpi=200)
plt.close()


# -------------------------------------------------------
# 3) COMPREHENSIVENESS CURVE
# -------------------------------------------------------
comp_curves = extract_metric_curve(df, "metric_comp")

plt.figure(figsize=(7, 5))
for expl, (ks, vals) in comp_curves.items():
    plt.plot(ks, vals, marker="o", label=expl)

plt.xlabel("Fraction masked (k)")
plt.ylabel("Probability increase")
plt.title("Comprehensiveness vs k")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/comprehensiveness_curve.png", dpi=200)
plt.close()


# -------------------------------------------------------
# 4) SUFFICIENCY CURVE
# -------------------------------------------------------
suff_curves = extract_metric_curve(df, "metric_suff")

plt.figure(figsize=(7, 5))
for expl, (ks, vals) in suff_curves.items():
    plt.plot(ks, vals, marker="o", label=expl)

plt.xlabel("Fraction retained (k)")
plt.ylabel("Probability drop")
plt.title("Sufficiency vs k")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/sufficiency_curve.png", dpi=200)
plt.close()


# -------------------------------------------------------
# 5) ACCURACY DROP CURVE (optional)
# -------------------------------------------------------
acc_curves = extract_metric_curve(df, "metric_accuracy")

if acc_curves:
    plt.figure(figsize=(7, 5))
    for expl, (ks, vals) in acc_curves.items():
        plt.plot(ks, vals, marker="o", label=expl)

    plt.xlabel("Fraction masked (k)")
    plt.ylabel("Accuracy change")
    plt.title("Accuracy change vs k")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/accuracy_curve.png", dpi=200)
    plt.close()

print("[OK] ICML-style benchmark plots generated in results/plots/")
