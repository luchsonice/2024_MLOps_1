import numpy as np
import os
import random
import torch
from tqdm import tqdm
from torch import optim
import logging
import datetime
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
from src.models.model import resnet18 as ResNet
import wandb
from src.data.make_dataset import get_dataloaders
import timm

def create_result_folders(experiment_name):
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)

def train(config = None):
    with wandb.init(config=config, 
                    project="MLOps_Project",
                    entity="luxonice",):
        print(wandb.config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print(f"Model will run on {device}")

        batch_size = wandb.config.batch_size
        lr = wandb.config.lr
        num_epochs = wandb.config.num_epochs
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # split config_path
        if config is not None:
            experiment_name = config.split('/')[-1].split('.')[0]
            logger = SummaryWriter(os.path.join("runs", experiment_name, time_stamp))
        else:
            experiment_name = "sweep"
            logger = SummaryWriter(os.path.join("sweeps", experiment_name, time_stamp))
        
        create_result_folders(os.path.join(experiment_name, time_stamp))
        trainloader, valloader, _ = get_dataloaders(batch_size)

        #model = ResNet()
        model = timm.create_model('resnet18', pretrained=True,num_classes=2)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        lossfunc = torch.nn.CrossEntropyLoss()

        l = len(trainloader)
        val_loss_current = np.inf

        for epoch in range(1, num_epochs + 1):
            logging.info(f"Starting epoch {epoch}:")
            num_batches = tqdm(trainloader)
            model.train()

            for i, (image, label) in enumerate(num_batches):
                image = image.to(device)
                label = label.to(device)

                with torch.cuda.amp.autocast():
                    prediction = model(image)
                    loss = lossfunc(prediction,label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({'train loss': loss})

                num_batches.set_postfix(MSE=loss.item())
                logger.add_scalar("train MSE", loss.item(), global_step=epoch * l + i)
            # time stamp
            
            num_batches_val = tqdm(valloader)
            model.eval()
            # pick a random integer between 0 and len(valloader)
            random_idx = random.randint(0, len(valloader)-1)
            val_loss = 0
            for i, (image, label) in enumerate(num_batches_val):
                image = image.to(device)
                label = label.to(device)
                
                with torch.cuda.amp.autocast():
                    prediction = model(image) # prediction
                    val_loss += lossfunc(prediction,label).item() # loss between noise and predicted noise

                if i == random_idx:
                    random_image = image
                    random_label = label
                    if random_image.shape[0] > 4:
                        random_image = random_image[:4]
                        random_label = random_label[:4]

            val_loss /= len(valloader)

            wandb.log({'val loss': val_loss})

            logger.add_scalar("val MSE", val_loss, global_step=epoch)
            
            if val_loss < val_loss_current:
                val_loss_current = val_loss
                torch.save(model.state_dict(), os.path.join("models", experiment_name, time_stamp, f"weights-{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    config = args.config
    train(config)