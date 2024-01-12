import os
import pytest
import torch

from src.models.model import ResNetModel

@pytest.mark.parametrize('bs', [1, 23, 64])
def test_model(bs):
    # Test if correct output given input
    model = ResNetModel('resnet18')
    input = torch.randn(size=(bs, 3, 250, 250))
    output = model(input)
    assert output.shape == (bs, 2)
