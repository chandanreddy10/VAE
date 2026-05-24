import torch
from torch import nn

class VanillaVAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20):

        super().__init__()

        # Encoder
        self.enc_lay_1 = nn.Linear(input_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_lay_1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_lay_2 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encoder(self, x):

        hidden = self.relu(self.enc_lay_1(x))

        mu = self.mu(hidden)
        log_var = self.log_var(hidden)

        return mu, log_var

    def reparameterization(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        return z

    def decoder(self, z):

        hidden = self.relu(self.dec_lay_1(z))
        logits = self.dec_lay_2(hidden)

        return logits

    def forward(self, x):

        mu, log_var = self.encoder(x)
        z = self.reparameterization(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var