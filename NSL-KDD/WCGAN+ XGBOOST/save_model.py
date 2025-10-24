# save_model_from_existing_gan.py
import pandas as pd
import torch
from models.wcgan import WCGANTrainer

# Load same attack dataset
df = pd.read_csv("data/NSL-KDD/train_processed.csv")
df_attack = df[df['label'] == 1]

# Initialize trainer
gan = WCGANTrainer(data=df_attack)

# Load generator weights from scratch-trained model (optional)
# gan.generator.load_state_dict(torch.load("outputs/wcgan_model.pth"))

# Save generator model to file
torch.save(gan.generator.state_dict(), "outputs/wcgan_model.pth")
print("✅ GAN model saved manually.")
