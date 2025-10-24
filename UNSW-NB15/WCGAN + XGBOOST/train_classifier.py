import pandas as pd
from models.xgboost_classifier import train_xgboost, plot_confusion_matrix
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import os

# === Paths ===
DATA_PATH = "data/train_balanced.csv"
MODEL_PATH = "outputs/xgb_model.json"
FEATURES_PATH = "outputs/feature_columns.pkl"
PLOT_PATH = "outputs/plots/training_curve.png"
CONF_MATRIX_PATH = "outputs/plots/confusion_matrix.png"

# === Create output directories if not exist ===
os.makedirs("outputs/plots", exist_ok=True)

# === Load training data ===
df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1)
y = df["label"]

# === Tuned XGBoost Parameters ===
xgb_params = {
    "max_depth": 5,
    "learning_rate": 0.07,
    "n_estimators": 250,
    "scale_pos_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# === Train model
model, y_val, y_pred, evals_result = train_xgboost(X, y, params=xgb_params)

# === Save model
model.save_model(MODEL_PATH)
print(f"✅ XGBoost model saved to: {MODEL_PATH}")

# === Save feature column names
joblib.dump(X.columns.tolist(), FEATURES_PATH)
print(f"📄 Feature columns saved to: {FEATURES_PATH}")

# === Classification Report (Clean % Format)
report = classification_report(y_val, y_pred, target_names=["Normal", "Attack"], output_dict=True)

print("\n✅ Classification Report (in %):")
for label in ["Normal", "Attack"]:
    p = report[label]['precision'] * 100
    r = report[label]['recall'] * 100
    f1 = report[label]['f1-score'] * 100
    print(f"{label:<7} → Precision: {p:.2f}% | Recall: {r:.2f}% | F1-Score: {f1:.2f}%")

acc = report['accuracy'] * 100
print(f"\n🎯 Overall Accuracy: {acc:.2f}%")

# === Print Validation Accuracy
val_acc = (y_val == y_pred).mean() * 100
print(f"🧪 Validation Accuracy: {val_acc:.2f}%")

# === Save Confusion Matrix Plot
plot_confusion_matrix(y_val, y_pred, output_path=CONF_MATRIX_PATH)

# === Plot Log Loss Curve
train_loss = evals_result['validation_0']['logloss']
val_loss = evals_result['validation_1']['logloss']

plt.figure(figsize=(8, 5))
plt.plot(train_loss, label='Train Log Loss', color='blue', marker='o')
plt.plot(val_loss, label='Validation Log Loss', color='orange', marker='s')
plt.title('XGBoost Training vs Validation Log Loss (WCGAN+XGBoost)')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📉 Training curve saved to: {PLOT_PATH}")
