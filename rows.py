import pandas as pd

df = pd.read_csv("data/NSL-KDD/train_processed.csv")
print(df["label"].value_counts())
print("🔢 Total Rows:", df.shape[0])
print("🧬 Total Columns:", df.shape[1])
