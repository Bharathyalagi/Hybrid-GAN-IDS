import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# === Load imbalanced training data ===
data = pd.read_csv("data/train_processed.csv")

X = data.drop(columns=["label"])
y = data["label"]

print("🔥 Training distribution (imbalanced):")
print(y.value_counts())

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

# === Save model and features for reproducibility (optional) ===
os.makedirs("outputs/imbalanced", exist_ok=True)
joblib.dump(best_model, "outputs/imbalanced/decision_tree_model_imbalanced.pkl")
joblib.dump(X.columns.tolist(), "outputs/imbalanced/feature_columns_imbalanced.pkl")

# === Validation Report ===
y_pred = best_model.predict(X_val)
report = classification_report(y_val, y_pred, target_names=["Normal", "Attack"], output_dict=True)
print("\n📋 Validation Classification Report (Imbalanced Data):\n")
for label in ["Normal", "Attack"]:
    print(f"{label} → Precision: {report[label]['precision']*100:.2f}% | Recall: {report[label]['recall']*100:.2f}% | F1: {report[label]['f1-score']*100:.2f}%")
print(f"\n🎯 Overall Accuracy: {report['accuracy']*100:.2f}%")

# === Confusion Matrix ===
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Imbalanced Training Data)")
plt.savefig("outputs/imbalanced/confusion_matrix_imbalanced.png")
print("📊 Confusion matrix saved to outputs/imbalanced/confusion_matrix_imbalanced.png")
