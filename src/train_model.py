import os
import logging
import datetime
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
import hydra
from omegaconf import DictConfig
from src.data.make_dataset import get_dataloaders
from src.models.model import ResNetModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def create_result_folders(experiment_name):
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)

def train(config = None, config_name = None):
    # Read hyperparameters
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['num_epochs']

    # Create model
    model = ResNetModel('resnet18', lr=lr)

    # split config_path
    if config_name is not None:
        experiment_name = config_name
    else:
        experiment_name = "sweep"

    # Create results folder
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    create_result_folders(os.path.join(experiment_name, time_stamp))

    # Create models folder
    model_path = os.path.join("models", experiment_name, time_stamp)
    os.makedirs(model_path, exist_ok=True)

    # Get dataloaders
    trainloader, valloader, _ = get_dataloaders(batch_size)

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path, monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    # Create Wandb logger
    wandb_logger = WandbLogger(config=config, project="MLOps_Project", entity="luxonice")

    # Create trainer
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="auto",
        max_epochs=num_epochs,
        logger=wandb_logger,
        log_every_n_steps=1
        )

    # Train the model
    trainer.fit(model, trainloader, valloader)

# Uses hydra to load the config file
@hydra.main(version_base=None, config_path=".." + os.sep + "configs", config_name="config")
def main(cfg : DictConfig) -> None:
    # Store name from config file
    config_name = cfg.name
    # Store hyperparameters from config file as a dict
    config = dict(cfg.hyperparameters)
    # Train the model with the given hyperparameters
    train(config, config_name)



if __name__ == '__main__':
    main()
