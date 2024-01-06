import torch
from torchvision import datasets, transforms
import os
from src.data import _DATA_PATH
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    """Processes smoker data."""
    file_path = _DATA_PATH + "raw/"
    label_path =_DATA_PATH + "processed/"

    train_images = torch.Tensor()
    train_label = torch.Tensor()
    test_images = torch.Tensor()
    test_label = torch.Tensor()
    val_images = torch.Tensor()
    val_label = torch.Tensor()
    transform = transforms.Compose([transforms.PILToTensor()])

    for file in os.listdir(file_path):
        if file.startswith("Training"):
            for imagepath in os.listdir(file_path + file + os.sep):
                image = Image.open(file_path + file + os.sep + imagepath)
                Tensorimage = transform(image)
                train_images = torch.cat((train_images, Tensorimage), dim=0)
                train_label = torch.cat((train_label, Tensorimage), dim=0)
        elif file.startswith("Testing"):
            for imagepath in os.listdir(file_path + file + os.sep):
                image = Image.open(file_path + file + os.sep + imagepath)
                Tensorimage = transform(image)
                test_images = torch.cat((test_images, Tensorimage), dim=0)
                test_label = torch.cat((test_label, Tensorimage), dim=0)
        elif file.startswith("Validation"):
            for imagepath in os.listdir(file_path + file + os.sep):
                image = Image.open(file_path + file + os.sep + imagepath)
                Tensorimage = transform(image)
                val_images = torch.cat((val_images, Tensorimage), dim=0)
                val_label = torch.cat((val_label, Tensorimage), dim=0)

    # Normalize tensors
    train_images = (train_images - torch.mean(train_images)) / torch.std(train_images)
    test_images = (test_images - torch.mean(train_images)) / torch.std(train_images)
    val_images = (val_images - torch.mean(train_images)) / torch.std(train_images)

    torch.save(train_images, label_path + "train_images.pt")
    torch.save(train_label, label_path + "train_label.pt")
    torch.save(test_images, label_path + "test_images.pt")
    torch.save(test_label, label_path + "test_label.pt")
    torch.save(val_images, label_path + "val_images.pt")
    torch.save(val_label, label_path + "val_label.pt")
