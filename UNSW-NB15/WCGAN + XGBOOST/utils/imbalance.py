import pandas as pd

df = pd.read_csv("data/train_processed.csv")
print("Unique labels:", df['label'].unique())
print(df['label'].value_counts())
