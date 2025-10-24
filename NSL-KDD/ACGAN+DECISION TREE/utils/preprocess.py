import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def load_data(base_path):
    # Load raw data files
    train_path = os.path.join(base_path, "KDDTrain+.txt")
    test_path = os.path.join(base_path, "KDDTest+.txt")

    column_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
        "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
    ]

    train_df = pd.read_csv(train_path, names=column_names)
    test_df = pd.read_csv(test_path, names=column_names)

    return train_df, test_df


def preprocess_data(df):
    df = df.drop(columns=['difficulty'])  # Remove unnecessary column

    # Convert label to binary or multiclass
    df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    # Encode categorical features
    cat_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=cat_cols)

    # Label encode target
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])  # normal = 0, attack = 1

    # Scale features
    X = df.drop('label', axis=1)
    y = df['label']
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    df_processed = pd.concat([X_scaled, y], axis=1)
    return df_processed


def save_processed_data(train_df, test_df, output_path):
    train_df.to_csv(os.path.join(output_path, "train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test_processed.csv"), index=False)


def run_preprocessing():
    raw_data_path = "data/NSL-KDD"
    train_df, test_df = load_data(raw_data_path)

    train_processed = preprocess_data(train_df)
    test_processed = preprocess_data(test_df)

    save_processed_data(train_processed, test_processed, raw_data_path)
    print("✅ Preprocessing complete. Files saved to:", raw_data_path)


if __name__ == "__main__":
    run_preprocessing()
