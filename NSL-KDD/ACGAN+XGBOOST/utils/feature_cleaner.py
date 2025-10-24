# utils/feature_cleaner.py

import pandas as pd
import numpy as np 

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop constant columns
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    df.drop(columns=constant_cols, inplace=True)

    # Drop duplicate columns
    df = df.loc[:, ~df.T.duplicated()]

    # Drop highly correlated features (Pearson > 0.95)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(columns=to_drop, inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
    return df

# Quick test
if __name__ == "__main__":
    clean_dataset("data/NSL-KDD/train_processed.csv", "data/NSL-KDD/train_cleaned.csv")
    clean_dataset("data/NSL-KDD/test_processed.csv", "data/NSL-KDD/test_cleaned.csv")
