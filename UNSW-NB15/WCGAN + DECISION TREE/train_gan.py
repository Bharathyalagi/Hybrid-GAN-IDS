import pandas as pd
import numpy as np
import os
from models.wcgan import WCGANTrainer

# Load preprocessed training data (ATTACKS only)
df = pd.read_csv("data/train_processed.csv")
df_attack = df[df['label'] == 1]
df_normal = df[df['label'] == 0]

num_real_attack = len(df_attack)
num_real_normal = len(df_normal)

num_synthetic_attack = max(0, num_real_normal - num_real_attack)

output_dir = "D:/UNSW_PROJECT/wcgan_dt/outputs"
os.makedirs(output_dir, exist_ok=True)

# Always train the WCGAN on attack data
print("Starting WCGAN training on attack data...")
gan = WCGANTrainer(data=df_attack)
gan.train(epochs=100)

# Save trained generator model
generator_path = os.path.join(output_dir, "generator_attack.pth")
gan.save_generator(generator_path)
print(f"✅ Generator model saved at: {generator_path}")

if num_synthetic_attack == 0:
    print("Synthetic attack samples generation not needed as attack samples are already more.")
else:
    # Generate synthetic attack samples
    synthetic_samples = gan.generate(num_synthetic_attack)

    # Clip negative values to zero
    synthetic_samples = np.clip(synthetic_samples, 0, None)

    feature_columns = [col for col in df_attack.columns if col != 'label']
    synthetic_df = pd.DataFrame(synthetic_samples, columns=feature_columns)
    synthetic_df['label'] = 1

    synthetic_path = os.path.join(output_dir, "synthetic_attack.csv")
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"✅ Generated {num_synthetic_attack} synthetic attack samples saved at: {synthetic_path}")

print("✅ WCGAN training and synthetic data generation complete.")
