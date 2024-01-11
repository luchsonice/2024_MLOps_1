import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TEST_DATA_VAR = 'test'

if __name__ == "__main__":
    """Processes smoker data."""
    data_path = "data/raw/"
    data_processed_path = "data/processed/"
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # Create datasets
    for dir in os.listdir(data_path):
        if dir.startswith("Training"):
            trainingset = datasets.ImageFolder(data_path + dir, transform=transform)
        elif dir.startswith("Validation"):
            valset = datasets.ImageFolder(data_path + dir, transform=transform)
        elif dir.startswith("Testing"):
            testset = datasets.ImageFolder(data_path + dir, transform=transform)

    # Create initial dataloaders (only used for preprocessing)
    batch_size = 64
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Load all training data
    print('Loading training data')
    train_images_tensor = []
    train_labels_tensor = []
    for (ims, labs) in trainloader:
        train_images_tensor.append(ims)
        train_labels_tensor.append(labs)
    train_images_tensor = torch.concat(train_images_tensor, dim=0)
    train_labels_tensor = torch.concat(train_labels_tensor, dim=0)

    # Load all validation data
    print('Loading validation data')
    val_images_tensor = []
    val_labels_tensor = []
    for (ims, labs) in valloader:
        val_images_tensor.append(ims)
        val_labels_tensor.append(labs)
    val_images_tensor = torch.concat(val_images_tensor, dim=0)
    val_labels_tensor = torch.concat(val_labels_tensor, dim=0)

    # Load all test data
    print('Loading test data')
    test_images_tensor = []
    test_labels_tensor = []
    for (ims, labs) in testloader:
        test_images_tensor.append(ims)
        test_labels_tensor.append(labs)
    test_images_tensor = torch.concat(test_images_tensor, dim=0)
    test_labels_tensor = torch.concat(test_labels_tensor, dim=0)

    # Normalize images based on mean and std of training set
    print('Processing data')
    mean = torch.mean(train_images_tensor)
    std = torch.std(train_images_tensor)
    print(f'mean = {mean}, std = {std}')
    train_images_tensor = (train_images_tensor - mean) / std
    val_images_tensor = (val_images_tensor - mean) / std
    test_images_tensor = (test_images_tensor - mean) / std

    # Save processed images and labels
    print('Saving processed data')
    torch.save(train_images_tensor, data_processed_path + "train_images.pt")
    torch.save(train_labels_tensor, data_processed_path + "train_labels.pt")
    torch.save(val_images_tensor, data_processed_path + "val_images.pt")
    torch.save(val_labels_tensor, data_processed_path + "val_labels.pt")
    torch.save(test_images_tensor, data_processed_path + "test_images.pt")
    torch.save(test_labels_tensor, data_processed_path + "test_labels.pt")

    print('Finished')

# https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


# def get_dataloaders(batch_size: int = 64, **kwargs) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
#     data_processed_path = "data/processed/"
#     train_images_tensor = torch.load(data_processed_path + "train_images.pt")
#     train_labels_tensor = torch.load(data_processed_path + "train_labels.pt")
#     val_images_tensor = torch.load(data_processed_path + "val_images.pt")
#     val_labels_tensor = torch.load(data_processed_path + "val_labels.pt")
#     test_images_tensor = torch.load(data_processed_path + "test_images.pt")
#     test_labels_tensor = torch.load(data_processed_path + "test_labels.pt")

#     transform = v2.Compose([
#         # v2.RandomApply(torch.nn.ModuleList([
#         #     v2.RandomResizedCrop(size=(250, 250), scale=(0.8, 1.0), antialias=True)
#         # ]), p=0.5),
#         v2.RandomHorizontalFlip(p=0.5), 
#     ])

#     train_dataset = CustomTensorDataset((train_images_tensor, train_labels_tensor), transform)
#     val_dataset = torch.utils.data.TensorDataset(val_images_tensor, val_labels_tensor)
#     test_dataset = torch.utils.data.TensorDataset(test_images_tensor, test_labels_tensor)

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

#     return train_loader, val_loader, test_loader

