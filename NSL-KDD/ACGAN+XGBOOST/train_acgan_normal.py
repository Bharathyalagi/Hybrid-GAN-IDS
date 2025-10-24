import pandas as pd
from models.acgan import ACGANTrainer
import os

# === Load only normal class (label = 0) for synthetic data generation ===
df = pd.read_csv("data/NSL-KDD/train_cleaned.csv")
df_normal = df[df["label"] == 0]  # Filter normal class
X_normal = df_normal.drop(columns=["label"])  # Features for normal class
y_normal = df_normal["label"]  # Labels (only normal data)

# === Initialize and Train ACGAN ===
trainer = ACGANTrainer(data=X_normal, labels=y_normal)
trainer.train(epochs=100, batch_size=64)

# === Save the trained model ===
os.makedirs("outputs", exist_ok=True)
trainer.save("outputs/acgan_model_normal.pth")  # Save model as acgan_model_normal.pth
