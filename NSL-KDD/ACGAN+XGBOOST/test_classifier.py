import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Load test data ===
test = pd.read_csv("data/NSL-KDD/test_cleaned.csv")
X_test = test.drop(columns=["label"])
y_test = test["label"]

# === Fix test feature alignment ===
feature_columns = joblib.load("outputs/feature_columns.pkl")
for col in feature_columns:
    if col not in X_test.columns:
        X_test[col] = 0  # Add missing columns with zeros
X_test = X_test[feature_columns]

# === Load model ===
model = joblib.load("outputs/decision_tree_model.pkl")
y_pred = model.predict(X_test)

# === Report ===
report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], output_dict=True)
print("\n✅ Test Classification Report (in %):\n")
for label in ["Normal", "Attack"]:
    print(f"{label} → Precision: {report[label]['precision']*100:.2f}% | Recall: {report[label]['recall']*100:.2f}% | F1-Score: {report[label]['f1-score']*100:.2f}%")
print(f"\nOverall Accuracy: {report['accuracy']*100:.2f}%")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set)")
plt.savefig("outputs/plots/confusion_matrix_dt_test.png")
print("📊 Test confusion matrix saved to: outputs/plots/confusion_matrix_dt_test.png")
