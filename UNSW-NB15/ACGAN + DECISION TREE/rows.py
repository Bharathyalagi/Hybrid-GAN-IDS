import pandas as pd

# Load your train data (make sure path is correct)
df = pd.read_csv('data/test_processed.csv')  # Adjust path if needed

# Check distribution
label_counts = df['label'].value_counts().sort_index()  # Ensure 0,1 order
print("Label Distribution:\n", label_counts)

# Determine target (max class size)
max_count = label_counts.max()

# Calculate synthetic samples required for balance
required_to_balance = max_count - label_counts
required_to_balance[required_to_balance < 0] = 0  # Ensure no negative values

# Build final DataFrame
balance_report = pd.DataFrame({
    'Class': ['Normal', 'Attack'],
    'Real Samples': label_counts.values,
    'Required to Balance': required_to_balance.values,
    'Synthetic Needed': required_to_balance.values
})

print("\nBalancing Requirement Table:\n", balance_report)
balance_report.to_csv("unsw_nb15_balance_summary.csv", index=False)
