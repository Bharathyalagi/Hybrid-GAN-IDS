import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import pandas as pd
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Generator
# ========================
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# ========================
# Discriminator
# ========================
class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + label_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, labels):
        x = torch.cat([x, labels], dim=1)
        return self.model(x)

# ========================
# Gradient Penalty
# ========================
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    gradients = autograd.grad(outputs=d_interpolates,
                               inputs=interpolates,
                               grad_outputs=fake,
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ========================
# WCGAN Trainer
# ========================
class WCGANTrainer:
    def __init__(self, data, label_col='label', noise_dim=32, batch_size=64, lambda_gp=10):
        self.label_col = label_col
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.lambda_gp = lambda_gp

        self.data = data
        self.labels = data[label_col]
        self.features = data.drop(columns=[label_col])

        self.input_dim = self.features.shape[1]
        self.label_dim = 1  # binary (used only in discriminator)
        self.generator = Generator(noise_dim, self.input_dim).to(device)
        self.discriminator = Discriminator(self.input_dim, self.label_dim).to(device)

        self.optim_G = optim.Adam(self.generator.parameters(), lr=1e-4)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=1e-4)

    def sample_noise(self, num_samples):
        return torch.randn(num_samples, self.noise_dim).to(device)

    def sample_labels(self, num_samples, label_value):
        return torch.full((num_samples, 1), label_value).float().to(device)

    def train(self, epochs=100):
        X = torch.tensor(self.features.values, dtype=torch.float32).to(device)
        y = torch.tensor(self.labels.values, dtype=torch.float32).unsqueeze(1).to(device)

        self.d_losses_real = []
        self.d_losses_fake = []

        for epoch in range(epochs):
            real_epoch_loss = 0.0
            fake_epoch_loss = 0.0
            steps = 0

            for i in range(0, X.size(0), self.batch_size):
                real_data = X[i:i+self.batch_size]
                real_labels = y[i:i+self.batch_size]

                noise = self.sample_noise(real_data.size(0))
                fake_data = self.generator(noise)

                # === Train Discriminator ===
                self.optim_D.zero_grad()
                real_validity = self.discriminator(real_data, real_labels)
                fake_validity = self.discriminator(fake_data.detach(), real_labels)

                gp = compute_gradient_penalty(self.discriminator, real_data.data, fake_data.data, real_labels)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gp
                d_loss.backward()
                self.optim_D.step()

                # Track real/fake separately
                real_epoch_loss += real_validity.mean().item()
                fake_epoch_loss += fake_validity.mean().item()
                steps += 1

                # === Train Generator (every 5 steps) ===
                if i % 5 == 0:
                    self.optim_G.zero_grad()
                    gen_data = self.generator(noise)
                    g_loss = -torch.mean(self.discriminator(gen_data, real_labels))
                    g_loss.backward()
                    self.optim_G.step()

            # === Save average loss per epoch ===
            avg_real = real_epoch_loss / steps
            avg_fake = fake_epoch_loss / steps
            self.d_losses_real.append(avg_real)
            self.d_losses_fake.append(avg_fake)

            print(f"Epoch [{epoch+1}/{epochs}] - D Real Loss: {avg_real:.4f}, D Fake Loss: {avg_fake:.4f}")

        # === Save loss arrays ===
        os.makedirs("outputs/losses", exist_ok=True)
        np.save("outputs/losses/wcgan_d_loss_real.npy", self.d_losses_real)
        np.save("outputs/losses/wcgan_d_loss_fake.npy", self.d_losses_fake)
        print("✅ WCGAN discriminator loss arrays saved.")

    def generate_synthetic(self, num_samples, label_value=0):
        noise = self.sample_noise(num_samples)
        fake_data = self.generator(noise).detach().cpu().numpy()
        labels = np.full((num_samples, 1), label_value)
        df = pd.DataFrame(fake_data, columns=self.features.columns)
        df[self.label_col] = labels
        return df

    def save_synthetic(self, path="outputs/synthetic_normal.csv", num_samples=5000):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_synth = self.generate_synthetic(num_samples=num_samples, label_value=0)
        df_synth.to_csv(path, index=False)
        print(f"✅ Synthetic data saved at: {path}")

    def save_generator(self, path="outputs/generator_normal.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.generator.state_dict(), path)
        print(f"✅ Generator model saved at: {path}")
