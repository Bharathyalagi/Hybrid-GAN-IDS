# === wcgan_unsw.py ===
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Generator ===
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
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

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super().__init__()
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

# === Gradient Penalty ===
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

# === WCGAN Trainer ===
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
        self.label_dim = 1  # binary
        self.generator = Generator(noise_dim, self.input_dim).to(device)
        self.discriminator = Discriminator(self.input_dim, self.label_dim).to(device)

        self.optim_G = optim.Adam(self.generator.parameters(), lr=1e-4)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=1e-4)

        self.d_losses = []
        self.g_losses = []

    def sample_noise(self, num_samples):
        return torch.randn(num_samples, self.noise_dim).to(device)

    def sample_labels(self, num_samples, label_value):
        return torch.full((num_samples, 1), label_value).float().to(device)

    def train(self, epochs=100):
        X = torch.tensor(self.features.values, dtype=torch.float32).to(device)
        y = torch.tensor(self.labels.values, dtype=torch.float32).unsqueeze(1).to(device)

        for epoch in range(epochs):
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

                # === Train Generator ===
                if i % 5 == 0:
                    self.optim_G.zero_grad()
                    gen_data = self.generator(noise)
                    g_loss = -torch.mean(self.discriminator(gen_data, real_labels))
                    g_loss.backward()
                    self.optim_G.step()

            print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
            self.d_losses.append(d_loss.item())
            self.g_losses.append(g_loss.item())

        # === Save loss arrays ===
        os.makedirs("outputs/losses", exist_ok=True)
        np.save("outputs/losses/wcgan_d_loss_unsw.npy", self.d_losses)
        np.save("outputs/losses/wcgan_g_loss_unsw.npy", self.g_losses)
        print("✅ WCGAN loss arrays saved for UNSW.")

    def generate(self, num_samples):
        noise = self.sample_noise(num_samples)
        return self.generator(noise).detach().cpu().numpy()

    def save_generator(self, path="outputs/generator_attack_unsw.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.generator.state_dict(), path)
        print(f"✅ Generator saved at {path}")


