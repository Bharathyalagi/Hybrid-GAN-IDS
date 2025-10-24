import numpy as np
import matplotlib.pyplot as plt
import os

# === Paths to loss files (saved from ACGAN training) ===
real_loss_path = "outputs/acgan_real_losses_nsl.npy"
fake_loss_path = "outputs/acgan_fake_losses_nsl.npy"

# === Check if files exist ===
if not os.path.exists(real_loss_path) or not os.path.exists(fake_loss_path):
    raise FileNotFoundError("❌ Loss files not found. Please train ACGAN and ensure loss arrays are saved.")

# === Load saved losses ===
real_losses = np.load(real_loss_path)
fake_losses = np.load(fake_loss_path)

# === Plot real vs fake discriminator loss ===
plt.figure(figsize=(10, 5))
plt.plot(real_losses, label="Discriminator Loss on Real Data", color="blue")
plt.plot(fake_losses, label="Discriminator Loss on Generated Data", color="red")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("ACGAN Discriminator Loss on NSL-KDD Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Save and show ===
output_path = "outputs/acgan_loss_plot_nsl.png"
plt.savefig(output_path)
plt.show()

print(f"✅ Plot saved to: {output_path}")
