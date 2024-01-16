import argparse
from src.data.make_dataset import get_dataloaders
from src.models.model import ResNetModel
from pytorch_lightning import Trainer

"""
Test the model on the test set.
Run as

python src/test_model.py model_path

e.g.

python src/test_model.py "models/config/<timestamp>/epoch=x-step=x.ckpt"
"""

def test(model_path: str, batch_size=64):
    
    # Load model checkpoint
    model = ResNetModel.load_from_checkpoint(model_path)

    # Get dataloaders
    _, _, testloader = get_dataloaders(batch_size)

    # A trainer is needed for testing
    trainer = Trainer(enable_checkpointing=False, logger=False)

    # Test the model
    # Metrics are automatically printed to console
    metrics = trainer.test(model, testloader)

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="the path to the model checkpoint")
    args = parser.parse_args()
    test(args.model_path)