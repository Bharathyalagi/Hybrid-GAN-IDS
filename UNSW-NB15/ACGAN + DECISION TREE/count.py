import pandas as pd

# === Load train and test data ===
train_df = pd.read_csv("data/train_processed.csv")
test_df = pd.read_csv("data/test_processed.csv")

# === Check class counts ===
train_counts = train_df['label'].value_counts().sort_index()
test_counts = test_df['label'].value_counts().sort_index()

print("📊 Train Set Class Distribution:")
print(f"Normal (0): {train_counts[0]}")
print(f"Attack  (1): {train_counts[1]}")

print("\n📊 Test Set Class Distribution:")
print(f"Normal (0): {test_counts[0]}")
print(f"Attack  (1): {test_counts[1]}")

# === Calculate how many synthetic samples needed to balance ===
def compute_synthetic_needed(normal_count, attack_count):
    if normal_count < attack_count:
        return attack_count - normal_count
    else:
        return 0

train_synthetic_needed = compute_synthetic_needed(train_counts[0], train_counts[1])
test_synthetic_needed = compute_synthetic_needed(test_counts[0], test_counts[1])

print("\n🧪 Synthetic Samples Needed to Balance:")
print(f"→ Train Set: {train_synthetic_needed} normal samples needed")
print(f"→ Test Set:  {test_synthetic_needed} normal samples needed (optional for evaluation)")
