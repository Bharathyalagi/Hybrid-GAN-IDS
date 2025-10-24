import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, input_dim, class_dim, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(class_dim, class_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + class_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, class_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128)
        )
        self.adv_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.aux_layer = nn.Linear(128, class_dim)

    def forward(self, x):
        x = self.model(x)
        validity = self.adv_layer(x)
        label = self.aux_layer(x)
        return validity, label

class ACGAN:
    def __init__(self, input_dim, latent_dim, num_classes):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.generator = Generator(latent_dim, num_classes, input_dim).to(device)
        self.discriminator = Discriminator(input_dim, num_classes).to(device)

        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.d_losses_real = []
        self.d_losses_fake = []

    def train(self, X_train, y_train, epochs=100, batch_size=64, sample_interval=10):
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            d_real_loss_epoch = 0.0
            d_fake_loss_epoch = 0.0
            steps = 0

            for imgs, labels in dataloader:
                batch_size_i = imgs.size(0)
                valid = torch.ones((batch_size_i, 1), device=device)
                fake = torch.zeros((batch_size_i, 1), device=device)

                real_imgs = imgs.to(device)
                labels = labels.to(device)

                # === Generator ===
                self.optimizer_G.zero_grad()
                z = torch.randn(batch_size_i, self.latent_dim, device=device)
                gen_labels = torch.randint(0, self.num_classes, (batch_size_i,), device=device)
                gen_imgs = self.generator(z, gen_labels)
                validity, pred_label = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels)
                g_loss.backward()
                self.optimizer_G.step()

                # === Discriminator ===
                self.optimizer_D.zero_grad()
                validity_real, pred_real = self.discriminator(real_imgs)
                validity_fake, pred_fake = self.discriminator(gen_imgs.detach())
                d_real_loss = self.adversarial_loss(validity_real, valid) + self.auxiliary_loss(pred_real, labels)
                d_fake_loss = self.adversarial_loss(validity_fake, fake) + self.auxiliary_loss(pred_fake, gen_labels)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                d_real_loss_epoch += validity_real.mean().item()
                d_fake_loss_epoch += validity_fake.mean().item()
                steps += 1

            self.d_losses_real.append(d_real_loss_epoch / steps)
            self.d_losses_fake.append(d_fake_loss_epoch / steps)

            if (epoch + 1) % sample_interval == 0 or epoch == 0:
                print(f"[Epoch {epoch+1}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        os.makedirs("outputs/losses", exist_ok=True)
        np.save("outputs/losses/acgan_d_loss_real_unsw.npy", self.d_losses_real)
        np.save("outputs/losses/acgan_d_loss_fake_unsw.npy", self.d_losses_fake)
        print("✅ ACGAN loss arrays saved for UNSW.")

    def generate_synthetic(self, n_samples, label=0):
        self.generator.eval()
        z = torch.randn(n_samples, self.latent_dim, device=device)
        labels = torch.full((n_samples,), label, dtype=torch.long, device=device)
        with torch.no_grad():
            gen_samples = self.generator(z, labels)
        self.generator.train()
        return gen_samples.cpu().numpy()

    def save_model(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator.eval()
        self.discriminator.eval()
