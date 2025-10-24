import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from models.acgan import ACGAN

# Create outputs directory if not exists
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/train_processed.csv")
print("Loaded data shape:", df.shape)

# Check label distribution
label_counts = df['label'].value_counts()
print("Label counts:\n", label_counts)

count_attack = label_counts.get(1, 0)
count_normal = label_counts.get(0, 0)
print(f"Attack samples (label=1): {count_attack}")
print(f"Normal samples (label=0): {count_normal}")

# Filter only normal samples for GAN training
normal_df = df[df['label'] == 0]
if normal_df.shape[0] == 0:
    raise ValueError("No normal samples found with label == 0!")

features = normal_df.drop(columns=['label'])
labels = normal_df['label']

# Encode labels (needed for ACGAN, even if all zeros)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Scale features to [0,1]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Initialize ACGAN
acgan = ACGAN(input_dim=features_scaled.shape[1], latent_dim=100, num_classes=1)

# Paths (renamed to avoid conflicts)
model_path = "outputs/acgan_model_unsw.pth"
synthetic_data_path = "outputs/acgan_normal_unsw.csv"

# Train fresh model
print("Training ACGAN model...")
acgan.train(features_scaled, labels_encoded, epochs=100, batch_size=64, sample_interval=10)
acgan.save_model(model_path)
print(f"✅ Model saved to {model_path}")

# Calculate how many synthetic samples to generate to balance dataset
n_samples_to_generate = max(0, count_attack - count_normal)
print(f"Generating {n_samples_to_generate} synthetic normal samples to balance dataset.")

if n_samples_to_generate > 0:
    generated_data = acgan.generate_synthetic(n_samples_to_generate, label=0)
    synthetic_df = pd.DataFrame(generated_data, columns=features.columns)
    synthetic_df['label'] = 0
    synthetic_df.to_csv(synthetic_data_path, index=False)
    print(f"✅ Synthetic normal samples saved to {synthetic_data_path}")
else:
    print("No synthetic samples needed; dataset is already balanced.")
