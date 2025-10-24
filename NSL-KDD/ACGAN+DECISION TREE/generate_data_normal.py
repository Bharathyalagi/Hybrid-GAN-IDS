import torch
import pandas as pd
from models.acgan import Generator, latent_dim, device
import os

# Load model
checkpoint = torch.load("outputs/acgan_model_normal.pth", map_location=device)
generator = Generator().to(device)
generator.load_state_dict(checkpoint["generator"])
generator.eval()

# Generate 8713 normal samples (class 0)
z = torch.randn(8713, latent_dim).to(device)
labels = torch.zeros(8713, dtype=torch.long).to(device)

with torch.no_grad():
    gen_data = generator(z, labels).cpu().numpy()

gen_data = (gen_data + 1) / 2  # scale from tanh

columns = pd.read_csv("data/NSL-KDD/train_processed.csv").drop(columns=["label"]).columns
df_synthetic = pd.DataFrame(gen_data, columns=columns)
df_synthetic["label"] = 0

os.makedirs("outputs", exist_ok=True)
df_synthetic.to_csv("outputs/synthetic_data_normal.csv", index=False)
print("✅ Synthetic normal data saved to: outputs/synthetic_data_normal.csv")
