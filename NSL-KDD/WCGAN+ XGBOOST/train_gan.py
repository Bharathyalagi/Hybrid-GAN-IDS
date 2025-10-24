import pandas as pd
from models.wcgan import WCGANTrainer

# === Load preprocessed training data (ATTACKS only) ===
df = pd.read_csv("data/NSL-KDD/train_processed.csv")
df_attack = df[df['label'] == 1]

# === Train WCGAN ===
gan = WCGANTrainer(data=df_attack)
gan.train(epochs=100)

gan.save_generator("outputs/generator_normal.pth")  # ✅ Save trained Generator

# === Save synthetic attack samples ===
gan.save_synthetic(path="outputs/synthetic_data.csv", num_samples=5000)
