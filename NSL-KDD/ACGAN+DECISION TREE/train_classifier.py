import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# === Load real + synthetic data ===
real = pd.read_csv("data/NSL-KDD/train_processed.csv")
synthetic_attack = pd.read_csv("outputs/synthetic_data_acgan.csv")
synthetic_normal = pd.read_csv("outputs/synthetic_data_normal.csv")

# === Balance per class ===
real_normal = real[real["label"] == 0]
real_attack = real[real["label"] == 1]

final_normal = pd.concat([real_normal, synthetic_normal], axis=0)
final_attack = pd.concat([real_attack, synthetic_attack], axis=0)

min_len = min(len(final_normal), len(final_attack))
final_normal = final_normal.sample(min_len, random_state=42)
final_attack = final_attack.sample(min_len, random_state=42)

full_data = pd.concat([final_normal, final_attack], axis=0).sample(frac=1).reset_index(drop=True)
print("🔥 Final class distribution in training:")
print(full_data["label"].value_counts())

X = full_data.drop(columns=["label"])
y = full_data["label"]

# === validation split ===
# 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# === Save
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/decision_tree_model.pkl")
joblib.dump(X.columns.tolist(), "outputs/feature_columns.pkl")

# === Eval
y_pred = model.predict(X_val)
report = classification_report(y_val, y_pred, target_names=["Normal", "Attack"], output_dict=True)
print("\n✅ Validation Classification Report (in %):\n")
for label in ["Normal", "Attack"]:
    print(f"{label} → Precision: {report[label]['precision']*100:.2f}% | Recall: {report[label]['recall']*100:.2f}% | F1-Score: {report[label]['f1-score']*100:.2f}%")
print(f"\nOverall Accuracy: {report['accuracy']*100:.2f}%")

# === Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Validation)")
plt.savefig("outputs/plots/confusion_matrix_dt_classifier.png")
print("📊 Confusion matrix saved to: outputs/plots/confusion_matrix_dt_classifier.png")
