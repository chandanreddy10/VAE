import os
import csv
import json
import yaml 
import torch 
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys 

LIB_PATH = str(Path(__file__).resolve().parents[1])
if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)

from model_bp import VAExp
from loss import vae_loss

# CONFIG
ROOT_DIR = Path(__file__).parents[0]
CONFIG_FILE = ROOT_DIR / "experiment.yaml"

with open(CONFIG_FILE, "r") as file:
    MASTER_YAML = yaml.safe_load(file)

CHECKPOINT_BASE = ROOT_DIR / MASTER_YAML["output"]["checkpoint"]
LOGS_BASE = ROOT_DIR / MASTER_YAML["output"]["log"]
DATA_FOLDER = ROOT_DIR / MASTER_YAML["output"].get("data_folder", "data")
SAMPLES_BASE = ROOT_DIR / MASTER_YAML["output"].get("samples", "samples")

os.makedirs(CHECKPOINT_BASE, exist_ok=True)
os.makedirs(LOGS_BASE, exist_ok=True)
os.makedirs(SAMPLES_BASE, exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datasets
train_dataset = datasets.FashionMNIST(root=DATA_FOLDER, train=True, download=True, transform=ToTensor())
test_dataset = datasets.FashionMNIST(root=DATA_FOLDER, train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader =  DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

ARCH = MASTER_YAML["architectures"]
HYPG = MASTER_YAML["hyperparameter_grid"]

def analyze_vae_gradients(model):
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # 1. L2 Norm
            grad_norm = param.grad.norm(2).item()
            
            # 2. Check for dead neurons
            zero_elements = (param.grad == 0).float()
            dead_percentage = (torch.sum(zero_elements) / param.grad.numel()).item() * 100
            
            grad_stats[name] = {
                "norm": grad_norm,
                "dead_pct": dead_percentage
            }
    return grad_stats

#Experiment Config 
#Each latent, KL weights and architecture
#Build the Config that is used to build the model state
#Todo: Add other hyperparameters : lr etc.
#Each Config generates random samples.

for grid_config in HYPG:
    arch_name = grid_config.get("arch_name")
    latent_dims = grid_config.get("latent_dims")
    kl_weights = grid_config.get("kl_weights")

    for dim in latent_dims:
        for kl_weight in kl_weights:
            
            RUN_CONFIG = {
                "encoder_layers" : ARCH[arch_name]["encoder_channels"],
                "decoder_layers": ARCH[arch_name]["decoder_channels"],
                "kernel_size": ARCH[arch_name]["kernel_size"],
                "stride": ARCH[arch_name]["stride"],
                "multiplier": ARCH[arch_name]["multiplier"],
                "latent_dim": dim,
                "kl_weight": kl_weight
            }
            
            exp_id = f"arch_{arch_name}__dim_{dim}__kl_{kl_weight}"
            print(f"EXPERIMENT: {exp_id.upper()}")
            
            exp_checkpoint_dir = CHECKPOINT_BASE / exp_id
            exp_samples_dir = SAMPLES_BASE / exp_id
            os.makedirs(exp_checkpoint_dir, exist_ok=True)
            os.makedirs(exp_samples_dir, exist_ok=True)
            
            log_csv_path = LOGS_BASE / f"logs_{exp_id}.csv"
            log_grad_json_path = LOGS_BASE / f"gradients_{exp_id}.json"
            
            model = VAExp(config=RUN_CONFIG).to(device)
            optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.1)
            
            # Track gradients for further Analysis
            gradient_history = {}
            
            # CSV Init
            with open(log_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_recon", "train_kl", "test_loss", "test_recon", "test_kl", "lr"])
            
            # Train Loop
            for epoch in range(EPOCHS):
                print(f"\n--- [{exp_id}] Epoch {epoch+1}/{EPOCHS} ---")
                
                model.train()
                train_total, train_recon, train_kl = 0.0, 0.0, 0.0
                
                #for norms and dead percentages across the current epoch
                epoch_norm_accumulator = defaultdict(float)
                epoch_dead_accumulator = defaultdict(float)
                
                for index, (images, _) in enumerate(train_dataloader):
                    images = images.to(device)
                    optimizer.zero_grad()
                    
                    output, mu, logvar = model(images)
                    # print(output.shape)
                    loss, recon, kl = vae_loss(output, images, mu, logvar)
                    
                    total_loss = recon + (RUN_CONFIG["kl_weight"] * kl)
                    total_loss.backward()
                    
                    # Query and Accumulate Gradient Metrics
                    batch_grad_stats = analyze_vae_gradients(model)
                    for layer_name, stats in batch_grad_stats.items():
                        epoch_norm_accumulator[layer_name] += stats["norm"]
                        epoch_dead_accumulator[layer_name] += stats["dead_pct"]
                    
                    optimizer.step()

                    train_total += total_loss.item() * images.size(0)
                    train_recon += recon.item() * images.size(0)
                    train_kl += kl.item() * images.size(0)
                    
                    if index % 10 == 0 and index > 0:
                        denom = index * BATCH_SIZE
                        print(f"Batch {index} | Est. Loss: {train_total / denom:.4f} | Recon: {train_recon / denom:.4f} | KL: {train_kl / denom:.4f}")
                
                    #break
                avg_train_loss = train_total / len(train_dataset)
                avg_train_recon = train_recon / len(train_dataset)
                avg_train_kl = train_kl / len(train_dataset)
                
                num_batches = len(train_dataloader)
                gradient_history[f"epoch_{epoch+1}"] = {
                    layer_name: {
                        "norm": epoch_norm_accumulator[layer_name] / num_batches,
                        "dead_pct": epoch_dead_accumulator[layer_name] / num_batches
                    }
                    for layer_name in epoch_norm_accumulator.keys()
                }
                
                with open(log_grad_json_path, "w") as json_file:
                    json.dump(gradient_history, json_file, indent=4)
                    
                print(f"[Train Summary] Loss: {avg_train_loss:.4f} | Recon: {avg_train_recon:.4f} | KL: {avg_train_kl:.4f}")

                # Eval
                model.eval()
                test_total, test_recon, test_kl = 0.0, 0.0, 0.0
                
                with torch.no_grad():
                    for test_images, _ in test_dataloader:
                        test_images = test_images.to(device)
                        output, mu, logvar = model(test_images)
                        loss, recon, kl = vae_loss(output, test_images, mu, logvar)
                        
                        test_loss = recon + (RUN_CONFIG["kl_weight"] * kl)
                        
                        test_total += test_loss.item() * test_images.size(0)
                        test_recon += recon.item() * test_images.size(0)
                        test_kl += kl.item() * test_images.size(0)
                        
                avg_test_loss = test_total / len(test_dataset)
                avg_test_recon = test_recon / len(test_dataset)
                avg_test_kl = test_kl / len(test_dataset)
                print(f"[Test Summary]  Loss: {avg_test_loss:.4f} | Recon: {avg_test_recon:.4f} | KL: {avg_test_kl:.4f}")
                
                scheduler.step(avg_test_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Append to CSV
                with open(log_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, avg_train_loss, avg_train_recon, avg_train_kl, avg_test_loss, avg_test_recon, avg_test_kl, current_lr])

                # Save Checkpoints
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_test_loss,
                    'config': RUN_CONFIG
                }, f"{exp_checkpoint_dir}/checkpoint_latest.pt")

                # Generate Random Sample Visual Grids
                with torch.no_grad():
                    if epoch % 15 == 0:
                        random_latent = torch.randn(64, dim).to(device) 
                        generated_images = model.decode(random_latent).cpu()
                        utils.save_image(generated_images, f"{exp_samples_dir}/epoch_{epoch+1}.png", nrow=8)
                #break  
            print(f"Completed Experiment Pipeline for Run Variant: {exp_id}")

print("\n Logs Generated Successfully!")