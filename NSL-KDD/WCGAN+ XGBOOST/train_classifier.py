import pandas as pd
from models.xgboost_classifier import load_and_prepare_data, train_xgboost, plot_confusion_matrix
from sklearn.metrics import classification_report
import joblib
import os
import matplotlib.pyplot as plt

# === Load the final GAN-balanced dataset ===
df = pd.read_csv("data/train_balanced.csv")
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

# === Train XGBoost ===
model, y_val, y_pred, evals_result = train_xgboost(X, y, params=xgb_params)

# === Save model
model.save_model("outputs/xgb_model.json")
print("✅ XGBoost model saved to: outputs/xgb_model.json")

# === Report (in %)
report = classification_report(y_val, y_pred, target_names=["Normal", "Attack"], output_dict=True)

print("\n✅ Classification Report (in %):\n")
for label in ["Normal", "Attack"]:
    p = report[label]['precision'] * 100
    r = report[label]['recall'] * 100
    f1 = report[label]['f1-score'] * 100
    print(f"{label} → Precision: {p:.2f}% | Recall: {r:.2f}% | F1-Score: {f1:.2f}%")

acc = report['accuracy'] * 100
print(f"\nOverall Accuracy: {acc:.2f}%")

# === Plot Confusion Matrix
plot_confusion_matrix(y_val, y_pred)

# === Save feature columns for test use
joblib.dump(X.columns.tolist(), "outputs/feature_columns.pkl")

# === Plot Training vs Validation Log Loss
train_loss = evals_result['validation_0']['logloss']
val_loss = evals_result['validation_1']['logloss']

plt.figure(figsize=(8, 5))
plt.plot(train_loss, label='Train Log Loss', color='blue', marker='o')
plt.plot(val_loss, label='Validation Log Loss', color='orange', marker='s')
plt.title('XGBoost Training vs Validation Log Loss (WCGAN2)')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the curve
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/training_curve_wcgan2.png")
print("📉 Training curve saved to: outputs/plots/training_curve_wcgan2.png")
