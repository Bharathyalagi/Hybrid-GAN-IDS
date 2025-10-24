import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ========================
# CONFIGURATION
# ========================
latent_dim = 100     # Size of noise vector
n_classes = 2        # 0 = normal, 1 = attack
input_dim = 122      # NSL-KDD features after one-hot encoding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# GENERATOR
# ========================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim),
            nn.Tanh()  # Since we scale to 0–1, tanh is okay
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((noise, label_input), -1)
        out = self.model(x)
        return out


# ========================
# DISCRIMINATOR
# ========================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.aux_layer = nn.Sequential(
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.model(x)
        validity = self.adv_layer(features)
        label = self.aux_layer(features)
        return validity, label


# ========================
# TRAINING WRAPPER
# ========================
class ACGANTrainer:
    def __init__(self, data, labels):
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.real_data = torch.tensor(data.values, dtype=torch.float32).to(device)
        self.real_labels = torch.tensor(labels.values, dtype=torch.long).to(device)

    def train(self, epochs=100, batch_size=64):
        for epoch in range(epochs):
            idx = torch.randint(0, self.real_data.size(0), (batch_size,))
            real_imgs = self.real_data[idx]
            labels = self.real_labels[idx]

            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # === Train Generator ===
            self.optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, n_classes, (batch_size,)).to(device)

            gen_imgs = self.generator(z, gen_labels)
            validity, pred_label = self.discriminator(gen_imgs)

            g_loss = self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels)
            g_loss.backward()
            self.optimizer_G.step()

            # === Train Discriminator ===
            self.optimizer_D.zero_grad()
            real_pred, real_aux = self.discriminator(real_imgs)
            d_real_loss = self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)

            fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
            d_fake_loss = self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            self.optimizer_D.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    def save(self, path="outputs/acgan_model.pth"):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)
        print(f"✅ ACGAN model saved to: {path}")

    def generate(self, num_samples=5000, class_label=1):
        self.generator.eval()
        z = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
        gen_samples = self.generator(z, labels).detach().cpu()
        return gen_samples
