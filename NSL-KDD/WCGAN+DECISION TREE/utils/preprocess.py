import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import joblib

def load_data(base_path):
    train_path = os.path.join(base_path, "KDDTrain+.txt")
    test_path = os.path.join(base_path, "KDDTest+.txt")

    column_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
    ]

    train_df = pd.read_csv(train_path, names=column_names)
    test_df = pd.read_csv(test_path, names=column_names)

    return train_df, test_df

def preprocess_data(df, save_scaler=False):
    df = df.drop(columns=['difficulty'])
    df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    # Encode categorical features
    cat_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=cat_cols)

    # Encode labels: normal=0, attack=1
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    X = df.drop('label', axis=1)
    y = df['label']

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # ✅ Save the scaler only for training data
    if save_scaler:
        joblib.dump(scaler, 'utils/scaler.save')

    df_processed = pd.concat([X_scaled, y], axis=1)
    return df_processed

def save_processed_data(train_df, test_df, output_path):
    os.makedirs(output_path, exist_ok=True)
    train_df.to_csv(os.path.join(output_path, "train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test_processed.csv"), index=False)

def inverse_transform(dataframe, scaler_path='utils/scaler.save'):
    scaler = joblib.load(scaler_path)
    cols = dataframe.columns
    inversed = scaler.inverse_transform(dataframe)
    return pd.DataFrame(inversed, columns=cols)

def run_preprocessing():
    raw_data_path = "data/NSL-KDD"
    output_path = "data"  # where to save processed CSVs

    train_df, test_df = load_data(raw_data_path)

    train_processed = preprocess_data(train_df, save_scaler=True)
    test_processed = preprocess_data(test_df, save_scaler=False)

    save_processed_data(train_processed, test_processed, output_path)
    print("✅ Preprocessing complete. Processed files saved to:", output_path)

if __name__ == "__main__":
    run_preprocessing()
