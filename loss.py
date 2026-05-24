import torch 
import torch.nn.functional as F

# ELBO = reconstruction + KL
def vae_loss(x_hat, x, mu, logvar):
    """
    Computes total VAE loss.
    Expects logvar instead of raw sigma/variance.
    """
    rc = recon_loss(x_hat, x)
    kl = kl_loss(mu, logvar, size=x.size(0))

    return rc + kl, rc, kl

def recon_loss(x_hat, x):
    # Safety clamp: keeps pixel predictions away from hard 0.0 or 1.0
    # to protect Binary Cross Entropy from hitting log(0) -> NaN/Asserts
    x_hat = torch.clamp(x_hat, min=1e-7, max=1.0 - 1e-7)
    
    rc_loss = F.binary_cross_entropy(
        x_hat,
        x,
        reduction="sum"
    )
    return rc_loss / x.size(0)

def kl_loss(mu, logvar, size, kl_weight=0.01):
    # This formula calculates KL divergence given 'logvar'
    # Had to clamp logvar because of exploding KL. It exploded to 10 power 30ish. 
    logvar = torch.clamp(logvar, min=-10, max=10)
    
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / size
    
    return kl_per_sample * kl_weight