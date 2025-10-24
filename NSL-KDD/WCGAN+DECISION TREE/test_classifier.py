import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Load test data ===
test_df = pd.read_csv("data/NSL-KDD/test_processed.csv")
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# === Load expected features & model ===
expected_cols = joblib.load("outputs/feature_columns.pkl")
model = joblib.load("outputs/dt_model.joblib")

# === Align test features ===
for col in expected_cols:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[expected_cols]

# === Predict
y_pred = model.predict(X_test)

# === Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
print("✅ Classification Report (Test):\n")
for label in ["0", "1"]:
    name = "Normal" if label == "0" else "Attack"
    p = report[label]['precision'] * 100
    r = report[label]['recall'] * 100
    f1 = report[label]['f1-score'] * 100
    print(f"{name} → Precision: {p:.2f}% | Recall: {r:.2f}% | F1-Score: {f1:.2f}%")

acc = report['accuracy'] * 100
print(f"\nOverall Accuracy: {acc:.2f}%")

# === Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.title("Confusion Matrix – Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/confusion_matrix_dt_test.png")
print("📊 Test confusion matrix saved to: outputs/plots/confusion_matrix_dt_test.png")
