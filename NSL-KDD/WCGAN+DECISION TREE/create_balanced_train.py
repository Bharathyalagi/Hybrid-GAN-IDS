import pandas as pd

# Load real training data
real_train_df = pd.read_csv("data/train_processed.csv")

# Load synthetic normal data
synthetic_normal = pd.read_csv("outputs/synthetic_normal.csv")
synthetic_normal['label'] = 0  # Mark it as normal

# Extract only real attack samples
real_attack = real_train_df[real_train_df['label'] != 0]

# Optional: also use some real normal data if you want
real_normal = real_train_df[real_train_df['label'] == 0]

# Combine all
balanced_df = pd.concat([synthetic_normal, real_attack, real_normal], ignore_index=True)

# Shuffle it
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Save to CSV
balanced_df.to_csv("data/train_balanced.csv", index=False)
print("✅ Balanced dataset saved to data/train_balanced.csv")
