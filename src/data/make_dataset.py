import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    """Processes smoker data."""
    file_path = "data/raw/"
    label_path ="data/processed/"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    for file in os.listdir(file_path):
        if file.startswith("Training"):
            trainingset = datasets.ImageFolder(file_path + file, transform=transform)
        elif file.startswith("Testing"):
            testset = datasets.ImageFolder(file_path + file, transform=transform)
        elif file.startswith("Validation"):
            valset = datasets.ImageFolder(file_path + file, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
    print(valloader)
