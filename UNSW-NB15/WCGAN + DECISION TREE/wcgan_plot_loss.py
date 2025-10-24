# === wcgan_plot_loss_unsw.py (Final Journal-Style Plot) ===
import numpy as np
import matplotlib.pyplot as plt
import os

# === Paths to existing loss files ===
d_loss_path = "outputs/losses/wcgan_d_loss_unsw.npy"
g_loss_path = "outputs/losses/wcgan_g_loss_unsw.npy"

# === Check if files exist ===
if not os.path.exists(d_loss_path) or not os.path.exists(g_loss_path):
    raise FileNotFoundError("❌ Loss files not found. Make sure WCGAN loss files are saved for UNSW.")

# === Load saved losses ===
d_loss = np.load(d_loss_path)
g_loss = np.load(g_loss_path)

# === Smoothing function for better visual (optional) ===
def smooth_curve(data, weight=0.9):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# === Smooth only generator loss for presentation ===
g_loss_smooth = smooth_curve(g_loss, weight=0.9)

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(d_loss, label="Discriminator Loss", color='blue')
plt.plot(g_loss_smooth, label="Generator Loss (Smoothed)", color='red')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("WCGAN Training Losses on UNSW-NB15 Dataset", fontsize=14)
plt.legend(facecolor='white', framealpha=1, fontsize=11)
plt.grid(True)

# === Save ===
output_path = "outputs/plots/wcgan_loss_unsw_final_redblue.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Final red-blue loss plot saved at: {output_path}")
