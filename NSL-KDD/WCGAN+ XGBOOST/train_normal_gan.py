import pandas as pd
from models.wcgan import WCGANTrainer

# Load only NORMAL samples
df = pd.read_csv("data/NSL-KDD/train_processed.csv")
df_normal = df[df['label'] == 0]

# Train WCGAN on normal data
gan = WCGANTrainer(data=df_normal)
gan.train(epochs=100)

# Save generator separately
gan.save_generator("outputs/generator_normal.pth")
print("✅ Trained WCGAN on normal samples and saved generator.")
