# NOTE: This script was used for combined testing.
# For clean modular execution, use train_gan.py and train_classifier.py

import pandas as pd
from models.wcgan import WCGANTrainer
from models.xgboost_classifier import load_and_prepare_data, train_xgboost, plot_confusion_matrix

# === Load preprocessed training data ===
df = pd.read_csv("data/NSL-KDD/train_processed.csv")
df_attack = df[df['label'] == 1]

# === Train WCGAN ===
#gan = WCGANTrainer(data=df_attack)
#gan.train(epochs=100)
#gan.save_synthetic(path="outputs/synthetic_data.csv", num_samples=5000)

# === Train XGBoost with real + synthetic data ===
X, y = load_and_prepare_data("data/NSL-KDD/train_processed.csv", "outputs/synthetic_data.csv")
model, y_val, y_pred = train_xgboost(X, y)

# === Plot Confusion Matrix ===
plot_confusion_matrix(y_val, y_pred)
