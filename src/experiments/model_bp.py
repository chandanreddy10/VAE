import torch
from torch import nn

class VAExp(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Unpack config
        encoder_layers = config["encoder_layers"]  
        decoder_layers = config["decoder_layers"]  
        kernel_size = config["kernel_size"]
        stride = config["stride"]
        
        self.multiplier = config["multiplier"]
        self.latent_dim = config["latent_dim"]
        self.last_layer_channels = encoder_layers[-1] #final channels for mu_var
        
        
        #Build Encoder
        enc_modules = []
        in_ch = 1  # FIxed input channel;s
        
        for out_ch in encoder_layers:
            enc_modules.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride))
            enc_modules.append(nn.BatchNorm2d(out_ch))
            enc_modules.append(nn.LeakyReLU(0.2))
            in_ch = out_ch
            
        enc_modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*enc_modules)
        
        
        flat_features = self.last_layer_channels * self.multiplier * self.multiplier
        self.mu_logvar = nn.Linear(flat_features, self.latent_dim * 2)
        self.decoder_input = nn.Linear(self.latent_dim , flat_features)

        #Same for decoder
        dec_modules = []
        in_ch = self.last_layer_channels
        
        for out_ch in decoder_layers:
            dec_modules.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride))
            dec_modules.append(nn.BatchNorm2d(out_ch))
            dec_modules.append(nn.LeakyReLU(0.2))
            in_ch = out_ch
            
        # reconstryct to channel 1
        dec_modules.append(nn.ConvTranspose2d(in_ch, 1, kernel_size=kernel_size, stride=stride))
        dec_modules.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*dec_modules)
        

    def encode(self, x):
        #print(x.shape)
        hidden = self.encoder(x)
        #print(hidden.shape)
        mu_logvar = self.mu_logvar(hidden)
        #print(mu_logvar.shape)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        return mu, logvar

    def reparameterization(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        hidden = self.decoder_input(z)
       # print(hidden.shape)
        hidden = hidden.view(-1, self.last_layer_channels, self.multiplier, self.multiplier)
        #print(hidden.shape)
        output = self.decoder(hidden)
        #print(output.shape)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        #print(z.shape)
        x_hat = self.decode(z)
        return x_hat, mu, logvar