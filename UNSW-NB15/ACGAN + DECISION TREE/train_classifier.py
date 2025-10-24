import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# === Load real and synthetic data ===
real = pd.read_csv("data/train_processed.csv")
synthetic_normal = pd.read_csv("outputs/acgan_normal.csv")

# === Separate real normal and real attack ===
real_normal = real[real["label"] == 0]
real_attack = real[real["label"] == 1]

# === Combine real normal + synthetic normal ===
final_normal = pd.concat([real_normal, synthetic_normal], axis=0)
final_attack = real_attack  # Use real attack only

# === Balance classes by sampling ===
min_len = min(len(final_normal), len(final_attack))
final_normal = final_normal.sample(min_len, random_state=42)
final_attack = final_attack.sample(min_len, random_state=42)

# === Merge and shuffle ===
full_data = pd.concat([final_normal, final_attack], axis=0).sample(frac=1).reset_index(drop=True)
print("🔥 Final training distribution:")
print(full_data["label"].value_counts())

X = full_data.drop(columns=["label"])
y = full_data["label"]

# === Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Decision Tree with Hyperparameter Tuning ===
model = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"\n✅ Best Hyperparameters: {best_params}")

best_model = grid_search.best_estimator_

# === Save the model and feature columns ===
os.makedirs("outputs/plots", exist_ok=True)
joblib.dump(best_model, "outputs/decision_tree_model.pkl")
joblib.dump(X.columns.tolist(), "outputs/feature_columns.pkl")

# === Validation Report ===
y_pred = best_model.predict(X_val)
report = classification_report(y_val, y_pred, target_names=["Normal", "Attack"], output_dict=True)
print("\n📋 Validation Classification Report:\n")
for label in ["Normal", "Attack"]:
    print(f"{label} → Precision: {report[label]['precision']*100:.2f}% | Recall: {report[label]['recall']*100:.2f}% | F1: {report[label]['f1-score']*100:.2f}%")
print(f"\n🎯 Overall Accuracy: {report['accuracy']*100:.2f}%")

# === Confusion Matrix ===
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Validation)")
plt.savefig("outputs/plots/confusion_matrix_dt_classifier.png")
print("📊 Confusion matrix saved to outputs/plots/confusion_matrix_dt_classifier.png")
