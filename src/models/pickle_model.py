import pickle
from src.models.model import ResNetModel

model_path = "models/config/2024-01-17_12-46-36/epoch=15-step=96.ckpt"

# Load model checkpoint
model = ResNetModel.load_from_checkpoint(model_path)

with open('model_to_deploy.pickle', 'wb') as handle:
    pickle.dump(model.net, handle, protocol=pickle.HIGHEST_PROTOCOL)
