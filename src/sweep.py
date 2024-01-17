import wandb
import yaml
from src.train_model import train


if __name__ == "__main__":
    with open("configs/sweep.yaml", "r") as yamlfile:
        sweep_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="MLOps_Project",
        entity="luxonice",
    )
    wandb.agent(sweep_id, train, count=20)
