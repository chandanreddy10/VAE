import torch 
import torch.nn.functional as F

#ELBO

def vae_loss(x_hat, x, mu, log_var):
    x = x.view(-1, 784)
    batch_size = x_hat.shape[0]

    rc_loss = F.mse_loss(
        x_hat,
        x,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
                               )

    loss = (rc_loss + kl_loss) / batch_size

    return loss