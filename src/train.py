import os
import csv
import torch 
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


from model import VanillaVAE
from loss import vae_loss
from utils import track_grad

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("samples", exist_ok=True)
log_file_path = "checkpoints/training_logs_2.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datasets
train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader =  DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = VanillaVAE().to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)


with open(log_file_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_recon", "train_kl", "test_loss", "test_recon", "test_kl", "lr"])

#Noticed Sawtooth Behaviour adding scheduler.
scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.1)

# Train Loop
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    # 1. Train
    model.train()
    train_total, train_recon, train_kl = 0.0, 0.0, 0.0
    
    for index, (images, _) in enumerate(train_dataloader):
        images = images.to(device)

        optimizer.zero_grad()
        output, mu, logvar = model(images)
        # print(logvar.max(), mu.shape)
        loss, recon, kl = vae_loss(output, images, mu, logvar)
        
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        train_total += loss.item() * images.size(0)
        train_recon += recon.item() * images.size(0)
        train_kl += kl.item() * images.size(0)
        
        if index % 10:
            results = track_grad.analyze_vae_gradients(model)
            denom = index * BATCH_SIZE
            print(f"Epoch : {epoch} | Train Loss : {train_total /denom} | Recon Loss : {train_recon/denom} | Train KL : {train_kl/denom}")
            print(results)
        
        
        # break
    # break
    results = track_grad.analyze_vae_gradients(model)
    # Calculate averages
    avg_train_loss = train_total / len(train_dataset)
    avg_train_recon = train_recon / len(train_dataset)
    avg_train_kl = train_kl / len(train_dataset)
    
    print(f"[Train] Loss: {avg_train_loss:.4f} | Recon: {avg_train_recon:.4f} | KL: {avg_train_kl:.4f}")

    # 2. Test
    model.eval()
    test_total, test_recon, test_kl = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for test_images, _ in test_dataloader:
            test_images = test_images.to(device)
            output, mu, logvar = model(test_images)
            loss, recon, kl = vae_loss(output, test_images, mu, logvar)
            
            test_total += loss.item() * test_images.size(0)
            test_recon += recon.item() * test_images.size(0)
            test_kl += kl.item() * test_images.size(0)
            
    avg_test_loss = test_total / len(test_dataset)
    avg_test_recon = test_recon / len(test_dataset)
    avg_test_kl = test_kl / len(test_dataset)
    
    print(f"[Test]  Loss: {avg_test_loss:.4f} | Recon: {avg_test_recon:.4f} | KL: {avg_test_kl:.4f}")
    scheduler.step(avg_test_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"End of Epoch {epoch+1} | Current LR: {current_lr}")
    
    # 3. Log 
    with open(log_file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_train_loss, avg_train_recon, avg_train_kl, avg_test_loss, avg_test_recon, avg_test_kl, current_lr])

    # 4. Save Model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_test_loss,
    }, "checkpoints/vanilla_vae_latest_2.pt")

    # 5. Generate and Save Samples
    with torch.no_grad():

        random_latent = torch.randn(64, model.mu_logvar.out_features // 2).to(device) 
        generated_images = model.decode(random_latent).cpu()
        
        utils.save_image(generated_images, f"samples/epoch_{epoch+1}.png", nrow=8)
        
print("\nTraining Complete! Logs saved to 'checkpoints/training_logs_2.csv'")