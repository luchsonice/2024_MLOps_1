from fastapi import FastAPI
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from src.models.model import ResNetModel

"""
First:
    Change checkpoint file path to a valid checkpoint
To launch API run: 
    uvicorn --reload --port 8000 app.main:app

Then go to: 
    http://localhost:8000/
    http://localhost:8000/docs
    http://localhost:8000/metrics
For interference go to:
    http://localhost:8000/interference/{test_image_path}
e.g.
    http://localhost:8000/interference/nonsmoking/notsmoking_0036.jpg
    http://localhost:8000/interference/smoking/smoking_0069.jpg
"""

app = FastAPI()

@app.get("/")
def read_root():
    return {"Go to /metrics for metrics","Go to /interference/image_path for interference"}

@app.get("/interference/smoking/{image_path}")
def interference_smoking(image_path: str):
    return interference("smoking/" + image_path)

@app.get("/interference/nonsmoking/{image_path}")
def interference_nonsmoking(image_path: str):
    return interference("nonsmoking/" + image_path)

def interference(image_path: str):
    img = Image.open("data/raw/Testing/" + image_path)
    
    convert_tensor = transforms.ToTensor()
    x = convert_tensor(img)
    x = x[None, :]
    model = ResNetModel.load_from_checkpoint("models/config/2024-01-15_13-51-41/epoch=1-step=24.ckpt")
    model.eval()
    
    with torch.no_grad():
        y_hat = model(x)

    prediction = np.argmax(y_hat.numpy())
    prediction = ["notsmoking", "smoking"][prediction]

    return {"prediction": prediction}

Instrumentator().instrument(app).expose(app)