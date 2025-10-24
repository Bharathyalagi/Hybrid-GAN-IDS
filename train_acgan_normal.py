import pandas as pd
from models.acgan import ACGANTrainer
import os

# Load normal class
df = pd.read_csv("data/NSL-KDD/train_processed.csv")
df_normal = df[df["label"] == 0]
X_normal = df_normal.drop(columns=["label"])
y_normal = df_normal["label"]

# Train ACGAN
trainer = ACGANTrainer(data=X_normal, labels=y_normal)
trainer.train(epochs=100, batch_size=64)

# Save model
os.makedirs("outputs", exist_ok=True)
trainer.save("outputs/acgan_model_normal.pth")
