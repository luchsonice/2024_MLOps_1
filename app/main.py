from fastapi import FastAPI
import torch
from torchvision import transforms
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from src.models.model import ResNetModel
from src.data.make_dataset import get_dataloaders

"""
Run: uvicorn --reload --port 8000 main:app
To launch API
Then go to: 
http://localhost:8000/
http://localhost:8000/docs
http://localhost:8000/metrics
"""
app = FastAPI()

@app.get("/")
def read_root():
    return {"Go to /metrics for metrics","Go to /interference/image_path for interference"}

@app.get("/interference/{image_path}")
def interference(image_path: str):
    img = Image.open("data/raw/Testing/smoking/smoking_0020.jpg")
    convert_tensor = transforms.ToTensor()
    x = convert_tensor(img)

    model = ResNetModel.load_from_checkpoint("models/config/2024-01-15_13-51-41/epoch=1-step=24.ckpt")
    model.eval()
    

    with torch.no_grad():
        y_hat = model(x)
    return {"predictions": y_hat}

Instrumentator().instrument(app).expose(app)