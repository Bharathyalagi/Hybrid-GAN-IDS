import numpy as np
import matplotlib.pyplot as plt
import os

real_loss_path = "outputs/losses/acgan_d_loss_real_unsw.npy"
fake_loss_path = "outputs/losses/acgan_d_loss_fake_unsw.npy"

if not os.path.exists(real_loss_path) or not os.path.exists(fake_loss_path):
    raise FileNotFoundError("❌ Loss files not found. Make sure ACGAN is trained and saved correctly for UNSW.")

d_loss_real = np.load(real_loss_path)
d_loss_fake = np.load(fake_loss_path)

plt.figure(figsize=(10, 5))
plt.plot(d_loss_real, label="Discriminator Loss on Real Data", color='blue')
plt.plot(d_loss_fake, label="Discriminator Loss on Generated Data", color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ACGAN Discriminator Loss on UNSW-NB15 Dataset")
plt.legend()
plt.grid(True)

output_path = "outputs/plots/acgan_loss_unsw.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.show()

print(f"✅ Loss plot saved at: {output_path}")
