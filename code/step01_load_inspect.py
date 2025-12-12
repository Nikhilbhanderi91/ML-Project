# step01_load_inspect.py
import os
import pandas as pd

# ---------- CONFIG ----------
CSV_PATH = "/Users/nikhilbhanderi/Documents/ML PROJECT/Dataset/student_placement_records.csv"
OUT_PREVIEW = "data_preview.csv"    # small file saved for quick view

# ---------- 1. Check file ----------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found. Update CSV_PATH in this script. Expected: {CSV_PATH}")

# ---------- 2. Load ----------
df = pd.read_csv(CSV_PATH)
print("Dataset loaded.")
print("Shape:", df.shape)

# ---------- 3. Quick head and dtypes ----------
print("\n--- First 5 rows ---")
print(df.head().to_string(index=False))

print("\n--- Column datatypes ---")
print(df.dtypes)

# ---------- 4. Missing values summary ----------
print("\n--- Missing values count per column ---")
print(df.isnull().sum().sort_values(ascending=False).head(30))

# ---------- 5. Ensure target exists and class balance ----------
if "placement_status" not in df.columns:
    raise ValueError("CSV must contain target column 'placement_status' (0/1).")

print("\n--- Placement class balance (counts & %) ---")
print(df["placement_status"].value_counts())
print(df["placement_status"].value_counts(normalize=True).mul(100).round(2))

# ---------- 6. List grade and numeric columns (helpful) ----------
grade_cols = [c for c in df.columns if c.lower().startswith("grade_")]
numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

print("\nIdentified columns summary:")
print(" - grade_cols:", len(grade_cols))
print(" - numeric_cols:", len(numeric_cols))
print(" - categorical_cols:", len(categorical_cols))

# ---------- 7. Save small preview CSV for manual checking ----------
preview = df.head(200).copy()
preview.to_csv(OUT_PREVIEW, index=False)
print(f"\nSaved preview (first 200 rows) to {OUT_PREVIEW}")

# ---------- 8. Short suggestions (auto) ----------
print("\n--- Quick suggestions ---")
if df['placement_status'].nunique() != 2:
    print(" * Note: 'placement_status' is not binary. Make sure values are 0 and 1.")
if df.isnull().values.any():
    print(" * Note: Dataset has missing values. Next step will show how to impute.")
else:
    print(" * No missing values detected (good). Next step will preprocess & encode categorical features.")

print("\nSTEP 1 complete. If everything looks correct, run Step 2 next (preprocessing).")