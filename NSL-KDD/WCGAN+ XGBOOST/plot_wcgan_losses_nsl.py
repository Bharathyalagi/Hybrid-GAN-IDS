import numpy as np
import matplotlib.pyplot as plt
import os

# === Paths ===
real_path = "outputs/losses/wcgan_d_loss_real.npy"
fake_path = "outputs/losses/wcgan_d_loss_fake.npy"

# === Check existence ===
if not os.path.exists(real_path) or not os.path.exists(fake_path):
    raise FileNotFoundError("❌ Loss files not found. Please train WCGAN and ensure loss arrays are saved.")

# === Load Losses ===
real_losses = np.load(real_path)
fake_losses = np.load(fake_path)

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(real_losses, label='Discriminator Loss on Real Data', color='blue')
plt.plot(fake_losses, label='Discriminator Loss on Generated Data', color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("WCGAN Discriminator Loss on NSL-KDD Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/wcgan_loss_plot_nsl.png", dpi=300)
plt.show()
