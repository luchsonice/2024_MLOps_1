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
import argparse
import wandb

def create_result_folders(config_name):
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", config_name), exist_ok=True)

def train(config = None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        lr = wandb.config.lr
        batch_size = wandb.config.batch_size
        num_epochs = wandb.config.num_epochs
        config_name = wandb.config.name

        # Create model
        model = ResNetModel('resnet18', lr=lr)

        # Create results folder
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        create_result_folders(os.path.join(config_name, time_stamp))

        # Create models folder
        model_path = os.path.join("models", config_name, time_stamp)
        os.makedirs(model_path, exist_ok=True)

        # Get dataloaders
        trainloader, valloader, _ = get_dataloaders(batch_size)

        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_path, monitor="val_loss", mode="min", save_top_k=1
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    config = args.config
    train(config)
