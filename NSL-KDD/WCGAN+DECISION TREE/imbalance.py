import pandas as pd

# Load train and test datasets
train_df = pd.read_csv('data/NSL-KDD/train_processed.csv')
test_df = pd.read_csv('data/NSL-KDD/test_processed.csv')

# Count class distribution in both
print("✅ Train Set Class Distribution:")
print(train_df['label'].value_counts())

print("\n✅ Test Set Class Distribution:")
print(test_df['label'].value_counts())
