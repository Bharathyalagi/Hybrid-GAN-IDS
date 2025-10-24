import pandas as pd
from models.acgan import ACGANTrainer
import os

# === Load only attack class (label = 1) for synthetic data generation ===
df = pd.read_csv("data/NSL-KDD/train_cleaned.csv")
df_attack = df[df["label"] == 1]  # Assuming you want only attack data for synthetic generation
X_attack = df_attack.drop(columns=["label"])
y_attack = df_attack["label"]

# === Train ACGAN ===
trainer = ACGANTrainer(data=X_attack, labels=y_attack)
trainer.train(epochs=100, batch_size=64)

# === Save model ===
os.makedirs("outputs", exist_ok=True)
trainer.save("outputs/acgan_model2.pth")
