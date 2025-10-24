import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def train_decision_tree_classifier(input_csv="data/train_balanced.csv", 
                                   model_path="outputs/dt_model.pkl",
                                   features_path="outputs/feature_columns.pkl",
                                   plot_dir="outputs/plots",
                                   test_size=0.2,
                                   random_state=42,
                                   dt_params=None):
    if dt_params is None:
        dt_params = {
            'criterion': 'entropy',
            'max_depth': 14,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None
        }

    print(f"Loading training data from {input_csv} ...")
    df = pd.read_csv(input_csv)
    X = df.drop(columns=["label"])
    y = df["label"]

    # Save feature columns for later use in testing
    feature_cols = X.columns.tolist()
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    joblib.dump(feature_cols, features_path)
    print(f"✅ Saved feature columns list to: {features_path}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Initialize and train model
    dt_model = DecisionTreeClassifier(**dt_params)
    dt_model.fit(X_train, y_train)

    # Training accuracy
    train_acc = accuracy_score(y_train, dt_model.predict(X_train)) * 100

    # Validation predictions & accuracy
    y_pred_val = dt_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val) * 100

    print(f"\n✅ Training Accuracy: {train_acc:.2f}%")
    print(f"✅ Validation Accuracy: {val_acc:.2f}%\n")

    # Classification report for validation
    print("Classification Report (Validation):")
    print(classification_report(y_val, y_pred_val, target_names=["Normal", "Attack"]))

    # Confusion matrix plot
    cm_val = confusion_matrix(y_val, y_pred_val)
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Validation Confusion Matrix")
    val_cm_path = os.path.join(plot_dir, "confusion_matrix_dt_val.png")
    plt.savefig(val_cm_path)
    plt.close()
    print(f"📊 Validation confusion matrix saved to: {val_cm_path}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(dt_model, model_path)
    print(f"✅ Decision Tree model saved to: {model_path}")

    return dt_model, feature_cols

if __name__ == "__main__":
    train_decision_tree_classifier()
