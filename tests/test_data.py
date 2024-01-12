import os
import pytest
import torch
from src.data.make_dataset import get_dataloaders

@pytest.mark.skipif(not (
    os.path.exists("data/processed/train_images.pt") and
    os.path.exists("data/processed/train_labels.pt") and
    os.path.exists("data/processed/val_images.pt") and
    os.path.exists("data/processed/val_labels.pt") and
    os.path.exists("data/processed/test_images.pt") and
    os.path.exists("data/processed/test_labels.pt")
), reason="Data files not found")
def test_data():
    # Test that there is the correct amount om images in test and train
    # Test that the dimensions of all images are correct
    # Test labels
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

    assert len(train_loader.dataset) == 716, "Training set has wrong number of samples"
    assert len(val_loader.dataset) == 180, "Validation set has wrong number of samples"
    assert len(test_loader.dataset) == 224, "Test set has wrong number of samples"

    for loader in [train_loader, val_loader, test_loader]:
        unique_labels = torch.Tensor([])
        for images, labels in loader:
            assert images.dim() == 4, "Image batch must have 4 dimensions"
            assert labels.dim() == 1, "Labels batch must have 1 dimension"
            assert images.shape[1:] == (3,250,250), "Images have wrong shape"
            unique_labels = torch.unique(torch.concat([unique_labels, labels]))
        assert torch.all(unique_labels == torch.arange(2)), "Labels are wrong"
        
        
