import torch 
from torch import nn

import os
import csv
import torch 
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import yaml 
from model_bp import VAExp


ROOT_DIR = Path(__file__).parents[0]
CONFIG_FILE = ROOT_DIR / "experiment.yaml"

with open(CONFIG_FILE, "r") as file:
    CONFIG = yaml.safe_load(file)

CHECKPOINT = ROOT_DIR / CONFIG["output"]["checkpoint"]
LOGS = ROOT_DIR / CONFIG["output"]["log"]


