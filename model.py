import torch
from torch import nn

class VanillaVAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=18):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Changed variable naming representation to logvar for mathematical clarity
        self.mu_logvar = nn.Linear(128 * 28 * 28, latent_dim * 2)
        self.decoder_input = nn.Linear(latent_dim, 128 * 28 * 28)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # FIXED: Removed BatchNorm2d from here to prevent extreme squashing/NaNs
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )
        

    def encode(self, x):
        hidden = self.encoder(x)
        mu_logvar = self.mu_logvar(hidden)

        # mu and logvar split
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        return mu, logvar

    def reparameterization(self, mu, logvar):
        # Clamping logvar prevents exponential explosion to infinity
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

    def decode(self, z):
        hidden = self.decoder_input(z)
        hidden = hidden.view(-1, 128, 28, 28)
        output = self.decoder(hidden)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        x_hat = self.decode(z)
        
        return x_hat, mu, logvar