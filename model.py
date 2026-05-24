import torch
from torch import nn
import torch.functional as F

class VanillaVAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2) #chunk
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        

    def encode(self, x):

        hidden = self.encoder(x)
        mu, sigma = torch.chunk(hidden, 2, dim=-1)

        return mu, sigma

    def reparameterization(self, mu, sigma):

        std = torch.exp(0.5 * sigma)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        return z

    def decode(self, z):

        output = self.decoder(z)

        return output

    def forward(self, x):

        mu, sigma = self.encode(x)
        std = torch.exp(0.5 * sigma)
        z = self.reparameterization(mu, std)
        x_hat = self.decode(z)

        return x_hat, mu, std, sigma