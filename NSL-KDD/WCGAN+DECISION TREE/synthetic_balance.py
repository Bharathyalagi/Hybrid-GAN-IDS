import pandas as pd

# Path to the synthetic normal dataset
path = "data/train_balanced.csv"

# Load dataset
df = pd.read_csv(path)

# === Display Info Systematically ===
print("\n📊 Synthetic Normal Dataset Summary")
print("======================================")
print(f"🧾 File Path           : {path}")
print(f"🔢 Number of Rows      : {df.shape[0]}")
print(f"🔢 Number of Columns   : {df.shape[1]}")

# Check if 'label' column exists and print value counts
print("======================================")
if 'label' in df.columns:
    print("🏷️  Label Distribution:")
    print(df['label'].value_counts())
else:
    print("⚠️  No 'label' column found in the dataset.")

# Print top 10 column names for quick overview
print("======================================")
print("🧩 Sample Column Names:")
print(df.columns[:10].tolist(), "... (showing first 10 only)")
print("======================================\n")
