import pandas as pd

SRC = "results/leaderboard.tsv"
DST = "results/leaderboard_clean.tsv"

df = pd.read_csv(SRC, sep="\t")

# ---- sanity checks ----
required = {"data", "model", "explainer"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ----------------------------------------
# Step 1: remove fully identical rows
# ----------------------------------------
df = df.drop_duplicates()

# ----------------------------------------
# Step 2: deduplicate by (data, model, explainer)
# Policy: keep LAST occurrence (latest run)
# ----------------------------------------
before = len(df)

df = (
    df
    .sort_index()                      # preserve file order
    .drop_duplicates(
        subset=["data", "model", "explainer"],
        keep="last"
    )
)

after = len(df)

print(f"[INFO] Removed {before - after} duplicate rows")

# ----------------------------------------
# Optional: sort for readability
# ----------------------------------------
df = df.sort_values(by=["data", "model", "explainer"])

# ----------------------------------------
# Write clean leaderboard
# ----------------------------------------
df.to_csv(DST, sep="\t", index=False)
print(f"[OK] Wrote clean leaderboard to {DST}")
