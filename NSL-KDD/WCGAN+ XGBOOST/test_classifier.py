import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# === Load test data ===
test_df = pd.read_csv("data/test_processed.csv")
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# === Match columns with train ===
expected_cols = joblib.load("outputs/feature_columns.pkl")

# Add missing columns with 0s
for col in expected_cols:
    if col not in X_test.columns:
        X_test[col] = 0

# Ensure order is same
X_test = X_test[expected_cols]

# === Load model ===
model = xgb.Booster()
model.load_model("outputs/xgb_model.json")

# === Predict ===
dtest = xgb.DMatrix(X_test, feature_names=expected_cols)
y_pred_probs = model.predict(dtest)
y_pred = (y_pred_probs > 0.5).astype(int)

# === Report
report = classification_report(y_test, y_pred, output_dict=True)
print("✅ Classification Report (in %):\n")
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"{'Normal' if label == '0' else 'Attack'} → "
              f"Precision: {metrics['precision']*100:.2f}% | "
              f"Recall: {metrics['recall']*100:.2f}% | "
              f"F1-Score: {metrics['f1-score']*100:.2f}%")

accuracy = report['accuracy'] * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%")

# === Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/confusion_matrix_test.png")
print("📊 Test confusion matrix saved to: outputs/plots/confusion_matrix_test.png")
