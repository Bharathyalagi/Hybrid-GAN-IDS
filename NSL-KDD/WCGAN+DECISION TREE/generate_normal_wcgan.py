import pandas as pd
import torch
from models.wcgan import Generator
from utils.preprocess import inverse_transform  # Make sure this works
import numpy as np
import os

# ========== Settings ==========
latent_dim = 32  # ✅ MUST match training value
num_samples = 8713
output_path = 'outputs/synthetic_normal.csv'
model_path = 'outputs/generator_normal.pth'
output_dim = 122  # Number of features

# ========== Load Generator ==========
generator = Generator(latent_dim=latent_dim, output_dim=output_dim)
generator.load_state_dict(torch.load(model_path))
generator.eval()

# ========== Generate Synthetic Data ==========
z = torch.randn(num_samples, latent_dim)
with torch.no_grad():
    synthetic_data = generator(z).cpu().numpy()

# ========== Inverse Scale ==========
synthetic_data = inverse_transform(pd.DataFrame(synthetic_data))

# ========== Save ==========
df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic['label'] = 0  # Label for normal samples

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_synthetic.to_csv(output_path, index=False)

print(f"✅ Saved {num_samples} synthetic normal samples to: {output_path}")
