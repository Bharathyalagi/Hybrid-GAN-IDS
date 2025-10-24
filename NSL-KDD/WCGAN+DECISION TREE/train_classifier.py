import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Load and clean training data ===
df = pd.read_csv("data/train_balanced.csv")
print(f"🔍 Duplicate rows before cleaning: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"✅ Dataset shape after removing duplicates: {df.shape}")

# === Prepare features and labels ===
X = df.drop("label", axis=1)
y = df["label"]

# === Save feature columns for test alignment ===
os.makedirs("outputs", exist_ok=True)
feature_columns = list(X.columns)
joblib.dump(feature_columns, "outputs/feature_columns.pkl")
print("💾 Feature columns saved to: outputs/feature_columns.pkl")

# === Train/Validation split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Decision Tree hyperparameters ===
dt_params = {
    'criterion': 'entropy',
    'max_depth': 14,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None
}

# === Train model ===
dt_model = DecisionTreeClassifier(**dt_params)
dt_model.fit(X_train, y_train)

# === Evaluate on validation data ===
y_pred = dt_model.predict(X_val)

report = classification_report(y_val, y_pred, output_dict=True)
print("✅ Classification Report (in %):\n")
for label in ["0", "1"]:
    name = "Normal" if label == "0" else "Attack"
    precision = report[label]["precision"] * 100
    recall = report[label]["recall"] * 100
    f1 = report[label]["f1-score"] * 100
    print(f"{name} → Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1-Score: {f1:.2f}%")

accuracy = report["accuracy"] * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%")

# === Save model ===
joblib.dump(dt_model, "outputs/dt_model.joblib")
print("✅ Decision Tree model saved to: outputs/dt_model.joblib")

# === Confusion Matrix ===
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Validation")
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/confusion_matrix_dt_val.png")
print("📊 Validation confusion matrix saved to: outputs/plots/confusion_matrix_dt_val.png")
