import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(real_path, synthetic_path):
    real_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synthetic_path)

    combined = pd.concat([real_df, synth_df], axis=0).sample(frac=1).reset_index(drop=True)
    
    X = combined.drop(columns=["label"])
    y = combined["label"]
    return X, y

def train_xgboost(X, y, params=None):
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if params is None:
        params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6
        }

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',  # Required
        **params
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    y_pred = model.predict(X_val)

    evals_result = model.evals_result()  # ✅ Safe & clean way

    print("✅ Classification Report:\n")
    print(classification_report(y_val, y_pred))

    return model, y_val, y_pred, evals_result




def plot_confusion_matrix(y_true, y_pred, output_path="outputs/plots/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"📊 Confusion matrix saved to: {output_path}")
