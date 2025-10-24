import torch
import pandas as pd
from models.acgan import Generator, latent_dim, device, input_dim, n_classes
import os

# === Load generator ===
model_path = "outputs/acgan_model.pth"
checkpoint = torch.load(model_path, map_location=device)

generator = Generator().to(device)
generator.load_state_dict(checkpoint["generator"])
generator.eval()

# === Generate synthetic data ===
num_samples = 5000
z = torch.randn(num_samples, latent_dim).to(device)
labels = torch.full((num_samples,), 1, dtype=torch.long).to(device)  # Class 1 = attack

with torch.no_grad():
    gen_data = generator(z, labels).cpu().numpy()

# === Clip to [0,1] and create DataFrame
gen_data = (gen_data + 1) / 2  # Rescale from tanh to [0, 1]
columns = pd.read_csv("data/NSL-KDD/train_processed.csv").drop(columns=["label"]).columns
df_synthetic = pd.DataFrame(gen_data, columns=columns)
df_synthetic["label"] = 1  # Label as attack

# === Save synthetic data
os.makedirs("outputs", exist_ok=True)
df_synthetic.to_csv("outputs/synthetic_data_acgan.csv", index=False)
print("✅ Synthetic data saved to: outputs/synthetic_data_acgan.csv")
