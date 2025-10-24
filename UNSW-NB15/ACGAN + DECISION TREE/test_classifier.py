import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Load test data ===
test_data = pd.read_csv("data/test_processed.csv")

X_test = test_data.drop(columns=["label"])
y_test = test_data["label"]

# === Load trained model and feature columns ===
model = joblib.load("outputs/decision_tree_model.pkl")
feature_columns = joblib.load("outputs/feature_columns.pkl")

# === Ensure test columns match training features ===
X_test = X_test[feature_columns]

# === Predict on test set ===
y_pred = model.predict(X_test)

# === Classification Report ===
report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], output_dict=True)
print("\n📋 Test Classification Report:\n")
for label in ["Normal", "Attack"]:
    print(f"{label} → Precision: {report[label]['precision']*100:.2f}% | Recall: {report[label]['recall']*100:.2f}% | F1: {report[label]['f1-score']*100:.2f}%")
print(f"\n🎯 Overall Test Accuracy: {report['accuracy']*100:.2f}%")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
os.makedirs("outputs/plots", exist_ok=True)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test)")
plt.savefig("outputs/plots/confusion_matrix_dt_test.png")
print("📊 Test confusion matrix saved to outputs/plots/confusion_matrix_dt_test.png")
