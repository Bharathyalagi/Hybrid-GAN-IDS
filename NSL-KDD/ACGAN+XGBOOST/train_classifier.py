import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# === Load data ===
real = pd.read_csv("D:/ACGAN2/data/NSL-KDD/train_cleaned.csv").dropna()
synthetic_normal = pd.read_csv("D:/ACGAN2/outputs/synthetic_data_normal2.csv").dropna()

# === Label enforcement ===
real["label"] = real["label"].astype(int)
synthetic_normal["label"] = 0  # force label = 0

# === Match columns (fill missing cols in synthetic) ===
for col in real.columns:
    if col not in synthetic_normal.columns:
        synthetic_normal[col] = 0

# === Align column order ===
synthetic_normal = synthetic_normal[real.columns]

# === Split classes ===
real_normal = real[real["label"] == 0]
real_attack = real[real["label"] == 1]

# === Combine ===
final_normal = pd.concat([real_normal, synthetic_normal], axis=0).dropna()
final_attack = real_attack.dropna()

# === Sample balanced ===
min_len = min(len(final_normal), len(final_attack))
final_normal = final_normal.sample(min_len, random_state=42)
final_attack = final_attack.sample(min_len, random_state=42)

# === Final dataset ===
full_data = pd.concat([final_normal, final_attack], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
print("🔥 Final class distribution in training:")
print(full_data["label"].value_counts())

X = full_data.drop(columns=["label"])
y = full_data["label"]

# === Check for NaNs ===
print(f"✅ NaNs in X: {X.isnull().sum().sum()} | NaNs in y: {y.isnull().sum()}")

# === Train-validation split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Decision Tree training ===
model = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

# === Best model and save ===
best_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")
os.makedirs("outputs/plots", exist_ok=True)
joblib.dump(best_model, "outputs/decision_tree_model.pkl")
joblib.dump(X.columns.tolist(), "outputs/feature_columns.pkl")

# === Evaluation ===
y_pred = best_model.predict(X_val)
report = classification_report(y_val, y_pred, target_names=["Normal", "Attack"], output_dict=True)
print("\n✅ Validation Classification Report (in %):\n")
for label in ["Normal", "Attack"]:
    print(f"{label} → Precision: {report[label]['precision']*100:.2f}% | Recall: {report[label]['recall']*100:.2f}% | F1-Score: {report[label]['f1-score']*100:.2f}%")
print(f"\nOverall Accuracy: {report['accuracy']*100:.2f}%")

# === Confusion matrix ===
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Validation)")
plt.savefig("outputs/plots/confusion_matrix_dt_classifier.png")
print("📊 Confusion matrix saved to: outputs/plots/confusion_matrix_dt1_classifier.png")
