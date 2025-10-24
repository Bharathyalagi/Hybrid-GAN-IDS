import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def test_decision_tree_classifier(test_csv="data/test_processed.csv",
                                  model_path="outputs/dt_model.pkl",
                                  features_path="outputs/feature_columns.pkl",
                                  plot_dir="outputs/plots"):


    print(f"Loading test data from {test_csv} ...")
    test_df = pd.read_csv(test_csv)
    y_test = test_df["label"]
    X_test = test_df.drop(columns=["label"])

    # Load feature columns saved during training
    feature_cols = joblib.load(features_path)
    print(f"Loaded expected feature columns: {len(feature_cols)} features")

    # Add missing columns with zeros
    missing_cols = [c for c in feature_cols if c not in X_test.columns]
    if missing_cols:
        print(f"⚠️ Missing columns in test data, adding with zeros: {missing_cols}")
        for col in missing_cols:
            X_test[col] = 0

    # Reorder columns to match training
    X_test = X_test[feature_cols]

    # Load trained model
    print(f"Loading Decision Tree model from {model_path} ...")
    dt_model = joblib.load(model_path)

    # Predict
    y_pred = dt_model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred) * 100

    # Classification report with percentages
    report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], output_dict=True)
    print("\n✅ Classification Report (Test):\n")
    for label in ["Normal", "Attack"]:
        precision = report[label]['precision'] * 100
        recall = report[label]['recall'] * 100
        f1 = report[label]['f1-score'] * 100
        print(f"{label} → Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1-Score: {f1:.2f}%")

    print(f"\nOverall Accuracy: {acc:.2f}%")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix – Test")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    test_cm_path = os.path.join(plot_dir, "confusion_matrix_dt_test.png")
    plt.savefig(test_cm_path)
    plt.close()
    print(f"📊 Test confusion matrix saved to: {test_cm_path}")

if __name__ == "__main__":
    test_decision_tree_classifier()
