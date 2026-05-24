import torch 
import torch.nn.functional as F

#ELBO = reconstruction + KL
def vae_loss(x_hat, x, mu, sigma, std):
    
    rc = recon_loss(x_hat, x)
    kl = kl_loss(mu, sigma, std)

    return rc + kl , rc, kl

def recon_loss(x_hat, x):
    x = x.view(-1, 784)
    
    rc_loss = F.binary_cross_entropy_with_logits(
        x_hat,
        x,
        reduction="sum"
    )

    return rc_loss / x.size(0)

def kl_loss(mu, sigma, std, eps=1e-8):
    kl_per_sample = 0.5 * torch.sum(mu.pow(2) + sigma.exp() -1 - sigma, dim=1)
    
    return kl_per_sample.mean()