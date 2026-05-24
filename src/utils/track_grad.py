import torch

def analyze_vae_gradients(model):
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # l2 Norm.
            grad_norm = param.grad.norm(2).item()
            
            # 2. Check for dead neurons
            zero_elements = (param.grad == 0).float()
            dead_percentage = (torch.sum(zero_elements) / param.grad.numel()).item() * 100
            
            grad_stats[name] = {
                "norm": grad_norm,
                "dead_pct": dead_percentage
            }
    return grad_stats