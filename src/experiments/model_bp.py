import torch
from torch import nn

class VAExp(nn.Module):
    def __init__(self, input_channels=1, latent_dim=18, multiplier=22, last_layer_channels=128):

        super().__init__()

        self.multiplier = multiplier
        self.last_layer_channels = last_layer_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, self.last_layer_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.last_layer_channels),
            nn.LeakyReLU(0.2),
            # nn.Flatten(),
        )
        
        # Changed variable naming representation to logvar for mathematical clarity
        self.mu_logvar = nn.Linear(self.last_layer_channels * self.multiplier * self.multiplier, latent_dim * 2)
        self.decoder_input = nn.Linear(latent_dim, self.last_layer_channels * self.multiplier * self.multiplier)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.last_layer_channels, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

           
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=1),
            nn.Sigmoid()
        )
        

    def encode(self, x):
        hidden = self.encoder(x)
        print(hidden.shape)
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
        hidden = hidden.view(-1, self.last_layer_channels, self.multiplier, self.multiplier)
        output = self.decoder(hidden)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        x_hat = self.decode(z)
        
        return x_hat, mu, logvar

