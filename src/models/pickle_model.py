import pickle
from src.models.model import ResNetModel

model_path = "models/config/2024-01-16_12-15-23/epoch=1-step=24.ckpt"

# Load model checkpoint
model = ResNetModel.load_from_checkpoint(model_path)

with open('model_to_deploy.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)