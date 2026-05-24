import torch 
from torchvision import datasets 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import VanillaVAE
from loss import vae_loss

EPOCHS = 100
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader =  DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model =  VanillaVAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    print(f"Running Epoch ---- {epoch}")
    model.train()
    total_loss = 0
    for images, labels in train_dataloader:

        images = images.to(device)
        images = images.view(-1, 784)

        optimizer.zero_grad()

        output, mu, log_var = model(images)
        loss = vae_loss(output, images, mu, log_var)

        loss.backward()
        optimizer.step()

        total_loss +=loss
    
    print(f"Epoch Loss : {total_loss}")
