import pandas as pd
import numpy as np
from models.wcgan import WCGANTrainer

def main():
    # Load dataset and filter normal samples only (label == 0)
    df = pd.read_csv("data/train_processed.csv")
    df_normal = df[df['label'] == 0]

    # Count real normal and attack samples
    num_real_attack = 45332
    num_real_normal = len(df_normal)
    num_synthetic_normal = max(0, num_real_attack - num_real_normal)

    print(f"Real normal samples: {num_real_normal}")
    print(f"Real attack samples: {num_real_attack}")
    print(f"Synthetic normal samples needed: {num_synthetic_normal}")

    # Initialize the GAN trainer with normal data
    gan = WCGANTrainer(data=df_normal)

    # Train GAN on normal data
    print("Starting GAN training on normal data...")
    gan.train(epochs=100)  # adjust epochs as needed

    # Save the trained generator model weights
    gan.save_generator("outputs/generator_normal.pth")

    # Generate synthetic normal samples if needed
    if num_synthetic_normal > 0:
        print(f"Generating {num_synthetic_normal} synthetic normal samples...")
        synthetic_df = gan.generate_synthetic(num_samples=num_synthetic_normal, label_value=0)

        # Clip negative values to zero (assuming features should be non-negative)
        feature_cols = df_normal.columns[:-1]
        synthetic_df[feature_cols] = synthetic_df[feature_cols].clip(lower=0)

        # Save synthetic data
        synthetic_df.to_csv("outputs/synthetic_normal.csv", index=False)
        print(f"✅ Synthetic normal samples saved at outputs/synthetic_normal.csv")
    else:
        print("No synthetic normal samples needed; dataset already balanced or normal samples exceed attack samples.")

if __name__ == "__main__":
    main()
