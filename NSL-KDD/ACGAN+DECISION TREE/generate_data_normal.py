import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Import your GAN model and load the checkpoint
from models.wcgan import Generator

# Load model
generator = Generator()
checkpoint = torch.load("path_to_your_acgan_model.pth")  # Update with your model's path
generator.load_state_dict(checkpoint["generator"])
generator.eval()

# Generate synthetic normal data
def generate_synthetic_normal_data(generator, num_samples=8713):  # Change this to 8713
    z = torch.randn(num_samples, 100)  # Assuming your latent space size is 100
    synthetic_data = generator(z).detach().numpy()
    return synthetic_data

# Generate synthetic normal data
synthetic_normal_data = generate_synthetic_normal_data(generator)

# Convert to DataFrame and save to CSV
synthetic_normal_df = pd.DataFrame(synthetic_normal_data, columns=["feature_" + str(i) for i in range(synthetic_normal_data.shape[1])])
synthetic_normal_df["label"] = 0  # Label as "Normal"
synthetic_normal_df.to_csv("outputs/synthetic_data_acgan_normal2.csv", index=False)
print("Synthetic normal data saved to 'outputs/synthetic_data_acgan_normal2.csv'")
