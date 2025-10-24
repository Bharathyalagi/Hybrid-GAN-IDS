import pandas as pd
import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Set directories and scaler save path
RAW_DIR = 'data/raw/'
OUT_DIR = 'data/'
SCALER_PATH = os.path.join(OUT_DIR, 'minmax_scaler.pkl')
ENCODER_DIR = os.path.join(OUT_DIR, 'encoders')
os.makedirs(ENCODER_DIR, exist_ok=True)

# Custom LabelEncoder to handle unknown categories during transform
class LabelEncoderWithUnknown(LabelEncoder):
    def __init__(self, unknown_value=-1):
        super().__init__()
        self.unknown_value = unknown_value

    def transform(self, y):
        y = np.array(y)
        known_mask = np.isin(y, self.classes_)
        y_transformed = np.full(shape=len(y), fill_value=self.unknown_value, dtype=int)
        y_transformed[known_mask] = super().transform(y[known_mask])
        return y_transformed

def load_data():
    """Load UNSW-NB15 train and test CSV files."""
    train_df = pd.read_csv(os.path.join(RAW_DIR, 'UNSW_NB15_training-set.csv'))
    test_df = pd.read_csv(os.path.join(RAW_DIR, 'UNSW_NB15_testing-set.csv'))
    return train_df, test_df

def encode_categorical(df, is_train=True):
    """Encode categorical columns using LabelEncoderWithUnknown, save/load encoders for consistency."""
    cat_cols = ['proto', 'service', 'state']
    
    for col in cat_cols:
        if is_train:
            le = LabelEncoderWithUnknown()
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, os.path.join(ENCODER_DIR, f"{col}_encoder.pkl"))
        else:
            le = joblib.load(os.path.join(ENCODER_DIR, f"{col}_encoder.pkl"))
            df[col] = le.transform(df[col].astype(str))
    return df

def preprocess(df, is_train=True):
    """Preprocess the dataframe: drop unused columns, encode categorical, and validate labels."""
    df = df.copy()

    # Drop non-informative or unused columns
    df.drop(columns=[col for col in ['id', 'attack_cat'] if col in df.columns], inplace=True)

    # Encode categorical features
    df = encode_categorical(df, is_train)

    # Ensure 'label' is integer type (don't remap values)
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
        if df['label'].nunique() != 2 or not set(df['label'].unique()).issubset({0, 1}):
            raise ValueError(f"⚠️ Label column contains unexpected values: {df['label'].unique()}")

    return df

def scale_features(train_df, test_df):
    """Scale features with MinMaxScaler and return processed DataFrames."""
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_scaled_df['label'] = y_train.values

    test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_scaled_df['label'] = y_test.values

    return train_scaled_df, test_scaled_df

def main():
    print("📦 Loading data...")
    train_df, test_df = load_data()

    print("⚙️ Preprocessing training data...")
    train_df = preprocess(train_df, is_train=True)

    print("⚙️ Preprocessing testing data...")
    test_df = preprocess(test_df, is_train=False)

    print("📊 Scaling features...")
    train_processed, test_processed = scale_features(train_df, test_df)

    train_processed.to_csv(os.path.join(OUT_DIR, 'train_processed.csv'), index=False)
    test_processed.to_csv(os.path.join(OUT_DIR, 'test_processed.csv'), index=False)

    print("✅ Preprocessing complete.")
    print(f"📁 Saved: train_processed.csv, test_processed.csv")
    print(f"💾 Scaler saved at: {SCALER_PATH}")
    print(f"💾 Label encoders saved in: {ENCODER_DIR}")

if __name__ == "__main__":
    main()
